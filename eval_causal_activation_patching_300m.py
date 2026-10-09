from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


PATCH_LAYERS = (2, 3, 4)


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, bos=True, eos=True)[:112]


def prompt(core: str, goal: str, focus: str) -> str:
    return f"הליבה היא {core}. היעד הנוכחי הוא {goal}. המוקד כעת הוא {focus}."


def clean_forward(model, ids: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SelfVariantPersonalNativeLM):
        base = model.base
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block, adapter in zip(base.blocks, model.self_adapters):
            x = block(x, None)
            x = adapter(x, model.self_anchor)
        x = base.final_norm(x)
    else:
        base = model
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block in base.blocks:
            x = block(x, None)
        x = base.final_norm(x)
    return x.mean(dim=1)


def _step_layer(model, x: torch.Tensor, layer_index: int, anchor_override=None) -> torch.Tensor:
    if isinstance(model, SelfVariantPersonalNativeLM):
        x = model.base.blocks[layer_index](x, None)
        adapter = model.self_adapters[layer_index]
        anchor = model.self_anchor if anchor_override is None else anchor_override
        x = adapter(x, anchor)
        return x
    return model.blocks[layer_index](x, None)


def forward_with_residual_patch(
    model,
    target_ids: torch.Tensor,
    source_ids: torch.Tensor,
    patch_layer: int,
    mode: str,
    seed: int,
) -> torch.Tensor:
    if target_ids.shape != source_ids.shape:
        raise ValueError("source/target token shapes must match for residual patching")
    base = model.base if isinstance(model, SelfVariantPersonalNativeLM) else model

    def init(ids):
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        return x + base.position_embedding(pos)[None, :, :]

    xt = init(target_ids)
    xs = init(source_ids)
    for i in range(len(base.blocks)):
        xt = _step_layer(model, xt, i)
        xs = _step_layer(model, xs, i)
        if i + 1 == patch_layer:
            delta = xs - xt
            if mode == "targeted":
                xt = xs.clone()
            elif mode == "random_matched":
                g = torch.Generator(device="cpu")
                g.manual_seed(seed)
                r = torch.randn(delta.shape, generator=g, dtype=torch.float32)
                r = r.to(delta.device, dtype=delta.dtype)
                flat_r = r.flatten(1)
                flat_d = delta.float().flatten(1)
                scale = (
                    flat_d.norm(dim=1, keepdim=True)
                    / flat_r.float().norm(dim=1, keepdim=True).clamp_min(1e-12)
                )
                r = (flat_r.float() * scale).view_as(delta).to(delta.dtype)
                xt = xt + r
            elif mode == "identity":
                pass
            else:
                raise ValueError(mode)

    xt = base.final_norm(xt)
    return xt.mean(dim=1)


def single_layer_self_forward(
    model: SelfVariantPersonalNativeLM,
    ids: torch.Tensor,
    layer: int,
    replacement: torch.Tensor,
) -> torch.Tensor:
    base = model.base
    x = base.token_embedding(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)
    x = x + base.position_embedding(pos)[None, :, :]
    for i, (block, adapter) in enumerate(zip(base.blocks, model.self_adapters), 1):
        x = block(x, None)
        anchor = replacement if i == layer else model.self_anchor
        x = adapter(x, anchor)
    x = base.final_norm(x)
    return x.mean(dim=1)


def build_centroids(reps: dict, triples: list[tuple[str, str, str, str]]):
    by = {"core": defaultdict(list), "goal": defaultdict(list), "focus": defaultdict(list)}
    for core, goal, focus, text in triples:
        r = unit(reps[text].unsqueeze(0))[0]
        by["core"][core].append(r)
        by["goal"][goal].append(r)
        by["focus"][focus].append(r)
    out = {}
    for comp in by:
        out[comp] = {
            label: unit(torch.stack(xs).mean(0, keepdim=True))[0]
            for label, xs in by[comp].items()
        }
    return out


def predict(rep: torch.Tensor, centroids: dict[str, torch.Tensor]) -> str:
    labels = sorted(centroids)
    c = torch.stack([centroids[x] for x in labels])
    scores = unit(rep.unsqueeze(0))[0] @ c.T
    return labels[int(scores.argmax())]


def score_margin(rep: torch.Tensor, centroids: dict[str, torch.Tensor], source: str, target: str) -> float:
    r = unit(rep.unsqueeze(0))[0]
    return float(r @ centroids[source] - r @ centroids[target])


def clean_accuracy(reps, triples, centroids):
    out = {}
    for comp, index in (("core", 0), ("goal", 1), ("focus", 2)):
        hits = 0
        for row in triples:
            label = row[index]
            text = row[3]
            hits += int(predict(reps[text], centroids[comp]) == label)
        out[comp] = hits / len(triples)
    return out


def pair_examples(tok, triples, component: str, limit: int = 72):
    index = {"core": 0, "goal": 1, "focus": 2}[component]
    lookup = {(c, g, f): t for c, g, f, t in triples}
    values = {
        "core": sorted({x[0] for x in triples}),
        "goal": sorted({x[1] for x in triples}),
        "focus": sorted({x[2] for x in triples}),
    }[component]
    pairs = []
    for c, g, f, target_text in triples:
        target = [c, g, f]
        target_ids = encode(tok, target_text)
        for src_value in values:
            if src_value == target[index]:
                continue
            src = list(target)
            src[index] = src_value
            src_tuple = tuple(src)
            source_text = lookup.get(src_tuple)
            if source_text is None:
                continue
            source_ids = encode(tok, source_text)
            if len(source_ids) != len(target_ids):
                continue
            key = hashlib.sha1((component + "|" + target_text + "|" + source_text).encode("utf-8")).hexdigest()
            pairs.append({
                "target": tuple(target),
                "source": src_tuple,
                "target_text": target_text,
                "source_text": source_text,
                "target_ids": target_ids,
                "source_ids": source_ids,
                "sort": key,
            })
            break
    pairs.sort(key=lambda x: x["sort"])
    return pairs[:limit]


@torch.no_grad()
def encode_clean(model, tok, triples, device):
    groups = defaultdict(list)
    for row in triples:
        text = row[3]
        ids = encode(tok, text)
        groups[len(ids)].append((text, ids))
    reps = {}
    for length in sorted(groups):
        items = groups[length]
        for start in range(0, len(items), 48):
            chunk = items[start:start + 48]
            ids = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
            out = clean_forward(model, ids).detach().cpu()
            for (text, _), r in zip(chunk, out):
                reps[text] = r
    return reps


@torch.no_grad()
def component_patching(model, tok, triples, clean_reps, centroids, device):
    result = {}
    comp_index = {"core": 0, "goal": 1, "focus": 2}
    for component in ("core", "goal", "focus"):
        pairs = pair_examples(tok, triples, component)
        if len(pairs) < 12:
            raise RuntimeError(f"too few equal-length {component} pairs: {len(pairs)}")
        result[component] = {"pair_count": len(pairs), "layers": {}}
        for layer in PATCH_LAYERS:
            modes = {}
            for mode in ("targeted", "random_matched"):
                switched = 0
                eligible = 0
                margin_shifts = []
                unchanged_hits = {"core": 0, "goal": 0, "focus": 0}
                unchanged_den = {"core": 0, "goal": 0, "focus": 0}
                effect_norms = []
                for p in pairs:
                    tid = torch.tensor([p["target_ids"]], dtype=torch.long, device=device)
                    sid = torch.tensor([p["source_ids"]], dtype=torch.long, device=device)
                    seed = int(p["sort"][:8], 16) + layer * 1009
                    patched = forward_with_residual_patch(
                        model, tid, sid, layer, mode, seed
                    )[0].detach().cpu()
                    clean_target = clean_reps[p["target_text"]]
                    clean_source = clean_reps[p["source_text"]]
                    effect_norms.append(float((unit(patched.unsqueeze(0))[0] - unit(clean_target.unsqueeze(0))[0]).norm()))

                    idx = comp_index[component]
                    target_label = p["target"][idx]
                    source_label = p["source"][idx]
                    target_clean_pred = predict(clean_target, centroids[component])
                    source_clean_pred = predict(clean_source, centroids[component])
                    if target_clean_pred == target_label and source_clean_pred == source_label:
                        eligible += 1
                        switched += int(predict(patched, centroids[component]) == source_label)

                    clean_margin = score_margin(
                        clean_target, centroids[component], source_label, target_label
                    )
                    patch_margin = score_margin(
                        patched, centroids[component], source_label, target_label
                    )
                    margin_shifts.append(patch_margin - clean_margin)

                    for unchanged, j in comp_index.items():
                        if unchanged == component:
                            continue
                        label = p["target"][j]
                        unchanged_den[unchanged] += 1
                        unchanged_hits[unchanged] += int(
                            predict(patched, centroids[unchanged]) == label
                        )

                modes[mode] = {
                    "eligible_clean_pairs": eligible,
                    "source_label_switch_rate": switched / max(1, eligible),
                    "mean_source_vs_target_margin_shift": sum(margin_shifts) / len(margin_shifts),
                    "mean_final_effect_norm": sum(effect_norms) / len(effect_norms),
                    "unchanged_component_accuracy": {
                        k: unchanged_hits[k] / max(1, unchanged_den[k])
                        for k in unchanged_hits
                        if unchanged_den[k]
                    },
                }
            t = modes["targeted"]
            r = modes["random_matched"]
            modes["targeted_minus_random"] = {
                "source_label_switch_rate": t["source_label_switch_rate"] - r["source_label_switch_rate"],
                "mean_source_vs_target_margin_shift": t["mean_source_vs_target_margin_shift"] - r["mean_source_vs_target_margin_shift"],
                "mean_final_effect_norm": t["mean_final_effect_norm"] - r["mean_final_effect_norm"],
            }
            result[component]["layers"][str(layer)] = modes
    return result


@torch.no_grad()
def self_layer_causality(model, tok, triples, clean_reps, centroids, device):
    if not isinstance(model, SelfVariantPersonalNativeLM):
        return None

    anchor = model.self_anchor.detach().cpu().float()
    n = anchor.norm().clamp_min(1e-12)
    radial2 = -anchor

    g = torch.Generator(device="cpu")
    g.manual_seed(991337)
    random_dir = torch.randn(anchor.shape, generator=g)
    random_dir = random_dir / random_dir.norm().clamp_min(1e-12)
    random2 = anchor + 2.0 * n * random_dir

    # A fixed deterministic subset controls runtime and keeps every variant matched.
    ordered = sorted(triples, key=lambda x: hashlib.sha1(x[3].encode("utf-8")).hexdigest())[:72]
    out = {}
    for layer in PATCH_LAYERS:
        layer_out = {}
        for name, replacement in (("radial_2x", radial2), ("random_2x", random2)):
            effects = []
            hits = {"core": 0, "goal": 0, "focus": 0}
            den = len(ordered)
            for c, g0, f0, text in ordered:
                ids = torch.tensor([encode(tok, text)], dtype=torch.long, device=device)
                rep = single_layer_self_forward(
                    model, ids, layer, replacement.to(device)
                )[0].detach().cpu()
                clean = clean_reps[text]
                effects.append(float((unit(rep.unsqueeze(0))[0] - unit(clean.unsqueeze(0))[0]).norm()))
                hits["core"] += int(predict(rep, centroids["core"]) == c)
                hits["goal"] += int(predict(rep, centroids["goal"]) == g0)
                hits["focus"] += int(predict(rep, centroids["focus"]) == f0)
            layer_out[name] = {
                "mean_final_effect_norm": sum(effects) / len(effects),
                "component_accuracy": {k: v / den for k, v in hits.items()},
            }
        rr = layer_out["radial_2x"]["mean_final_effect_norm"]
        rv = layer_out["random_2x"]["mean_final_effect_norm"]
        layer_out["radial_vs_random"] = {
            "effect_ratio": rr / max(rv, 1e-12),
            "effect_difference": rr - rv,
            "core_accuracy_delta": (
                layer_out["radial_2x"]["component_accuracy"]["core"]
                - layer_out["random_2x"]["component_accuracy"]["core"]
            ),
            "goal_accuracy_delta": (
                layer_out["radial_2x"]["component_accuracy"]["goal"]
                - layer_out["random_2x"]["component_accuracy"]["goal"]
            ),
            "focus_accuracy_delta": (
                layer_out["radial_2x"]["component_accuracy"]["focus"]
                - layer_out["random_2x"]["component_accuracy"]["focus"]
            ),
        }
        out[str(layer)] = layer_out
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    model.eval()

    cores = list(TEST_CORES)
    goals = list(TEST_GOALS[:6])
    foci = list(TEST_FOCI[:6])

    triples = [
        (c, g, f, prompt(c, g, f))
        for c in cores
        for g in goals
        for f in foci
    ]

    with torch.no_grad():
        clean_reps = encode_clean(model, tok, triples, args.device)
        centroids = build_centroids(clean_reps, triples)
        clean_acc = clean_accuracy(clean_reps, triples, centroids)
        patch = component_patching(
            model, tok, triples, clean_reps, centroids, args.device
        )
        self_causal = self_layer_causality(
            model, tok, triples, clean_reps, centroids, args.device
        )

    report = {
        "schema_version": 1,
        "experiment": "300M causal activation patching v1",
        "checkpoint": args.checkpoint,
        "variant": ckpt.get("variant", "baseline"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "patch_layers": list(PATCH_LAYERS),
        "clean_final_component_accuracy": clean_acc,
        "component_activation_patching": patch,
        "single_layer_self_causality": self_causal,
        "primary_questions": [
            "Does a matched residual-state transfer for CORE/GOAL/FOCUS causally move the corresponding downstream representation more than a random same-norm perturbation?",
            "Can CORE be causally switched while GOAL/FOCUS remain stable, indicating factorization rather than simple entanglement?",
            "At which of layers 2-4 does a single-layer SELF radial perturbation have a direction-specific downstream effect compared with a random same-displacement anchor perturbation?",
        ],
        "guardrails": [
            "Component patches only compare source/target prompts with identical token length and exactly one changed component.",
            "Random controls are matched to the full residual patch delta norm.",
            "A source-label switch is counted only when both clean source and clean target are correctly decoded.",
            "SELF radial and random replacements are matched at two learned-anchor norms of displacement.",
            "This establishes causal sensitivity of the tested representation path, not semantic selfhood.",
            "Results remain single-seed unless independently replicated.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "variant": report["variant"],
        "tokens_seen": report["tokens_seen"],
        "clean": clean_acc,
        "pair_counts": {k: v["pair_count"] for k, v in patch.items()},
        "out": args.out,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
