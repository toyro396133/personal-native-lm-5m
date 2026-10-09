from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_causal_activation_patching_300m import (
    build_centroids,
    clean_accuracy,
    encode,
    encode_clean,
    pair_examples,
    predict,
    prompt,
    score_margin,
    unit,
)
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


PATCH_LAYERS = (2, 3, 4)


def init_stream(model: SelfVariantPersonalNativeLM, ids: torch.Tensor) -> torch.Tensor:
    base = model.base
    x = base.token_embedding(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)
    return x + base.position_embedding(pos)[None, :, :]


def matched_random(delta: torch.Tensor, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    r = torch.randn(delta.shape, generator=g, dtype=torch.float32)
    r = r.to(delta.device)
    rf = r.flatten(1)
    df = delta.float().flatten(1)
    scale = df.norm(dim=1, keepdim=True) / rf.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return (rf * scale).view_as(delta).to(delta.dtype)


@torch.no_grad()
def forward_with_adapter_delta_patch(
    model: SelfVariantPersonalNativeLM,
    target_ids: torch.Tensor,
    source_ids: torch.Tensor,
    patch_layer: int,
    mode: str,
    seed: int,
):
    if target_ids.shape != source_ids.shape:
        raise ValueError("source and target must have identical token shape")

    xt = init_stream(model, target_ids)
    xs = init_stream(model, source_ids)
    patch_stats = None

    for i, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters), 1):
        pre_t = block(xt, None)
        pre_s = block(xs, None)

        out_t = adapter(pre_t, model.self_anchor)
        out_s = adapter(pre_s, model.self_anchor)

        if i == patch_layer:
            dt = out_t - pre_t
            ds = out_s - pre_s
            change = ds - dt

            if mode == "targeted":
                xt = out_t + change
            elif mode == "random_matched":
                xt = out_t + matched_random(change, seed)
            else:
                raise ValueError(mode)

            patch_stats = {
                "target_adapter_delta_norm": float(dt.float().norm()),
                "source_adapter_delta_norm": float(ds.float().norm()),
                "source_minus_target_delta_norm": float(change.float().norm()),
                "target_residual_norm": float(pre_t.float().norm()),
            }
        else:
            xt = out_t

        xs = out_s

    xt = model.base.final_norm(xt)
    return xt.mean(dim=1), patch_stats


@torch.no_grad()
def forward_with_adapter_ablation(
    model: SelfVariantPersonalNativeLM,
    ids: torch.Tensor,
    patch_layer: int,
    mode: str,
    seed: int,
):
    x = init_stream(model, ids)
    stats = None

    for i, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters), 1):
        pre = block(x, None)
        out = adapter(pre, model.self_anchor)

        if i == patch_layer:
            delta = out - pre
            if mode == "ablate":
                x = pre
            elif mode == "random_matched":
                x = out + matched_random(-delta, seed)
            else:
                raise ValueError(mode)
            stats = {
                "adapter_delta_norm": float(delta.float().norm()),
                "residual_norm": float(pre.float().norm()),
                "adapter_delta_over_residual": float(
                    delta.float().norm() / pre.float().norm().clamp_min(1e-12)
                ),
            }
        else:
            x = out

    x = model.base.final_norm(x)
    return x.mean(dim=1), stats


@torch.no_grad()
def adapter_source_delta_patching(model, tok, triples, clean_reps, centroids, device):
    comp_index = {"core": 0, "goal": 1, "focus": 2}
    result = {}

    for component in ("core", "goal", "focus"):
        pairs = pair_examples(tok, triples, component)
        if len(pairs) < 12:
            raise RuntimeError(f"too few equal-length {component} pairs: {len(pairs)}")

        result[component] = {"pair_count": len(pairs), "layers": {}}
        for layer in PATCH_LAYERS:
            modes = {}
            for mode in ("targeted", "random_matched"):
                switch_hits = 0
                switch_den = 0
                margin_shifts = []
                effect_norms = []
                unchanged_hits = {"core": 0, "goal": 0, "focus": 0}
                unchanged_den = {"core": 0, "goal": 0, "focus": 0}
                change_norms = []
                delta_ratios = []

                for p in pairs:
                    tid = torch.tensor([p["target_ids"]], dtype=torch.long, device=device)
                    sid = torch.tensor([p["source_ids"]], dtype=torch.long, device=device)
                    seed = int(p["sort"][:8], 16) + 701 * layer
                    patched, stats = forward_with_adapter_delta_patch(
                        model, tid, sid, layer, mode, seed
                    )
                    patched = patched[0].detach().cpu()
                    clean_t = clean_reps[p["target_text"]]
                    clean_s = clean_reps[p["source_text"]]

                    effect_norms.append(
                        float((unit(patched.unsqueeze(0))[0] - unit(clean_t.unsqueeze(0))[0]).norm())
                    )
                    change_norms.append(stats["source_minus_target_delta_norm"])
                    delta_ratios.append(
                        stats["source_minus_target_delta_norm"]
                        / max(stats["target_residual_norm"], 1e-12)
                    )

                    j = comp_index[component]
                    target_label = p["target"][j]
                    source_label = p["source"][j]
                    if (
                        predict(clean_t, centroids[component]) == target_label
                        and predict(clean_s, centroids[component]) == source_label
                    ):
                        switch_den += 1
                        switch_hits += int(
                            predict(patched, centroids[component]) == source_label
                        )

                    margin_shifts.append(
                        score_margin(
                            patched, centroids[component], source_label, target_label
                        )
                        - score_margin(
                            clean_t, centroids[component], source_label, target_label
                        )
                    )

                    for other, oi in comp_index.items():
                        if other == component:
                            continue
                        unchanged_den[other] += 1
                        unchanged_hits[other] += int(
                            predict(patched, centroids[other]) == p["target"][oi]
                        )

                modes[mode] = {
                    "eligible_clean_pairs": switch_den,
                    "source_label_switch_rate": switch_hits / max(1, switch_den),
                    "mean_source_vs_target_margin_shift": sum(margin_shifts) / len(margin_shifts),
                    "mean_final_effect_norm": sum(effect_norms) / len(effect_norms),
                    "mean_adapter_change_norm": sum(change_norms) / len(change_norms),
                    "mean_adapter_change_over_residual": sum(delta_ratios) / len(delta_ratios),
                    "unchanged_component_accuracy": {
                        k: unchanged_hits[k] / max(1, unchanged_den[k])
                        for k in unchanged_hits
                        if unchanged_den[k]
                    },
                }

            t = modes["targeted"]
            r = modes["random_matched"]
            modes["targeted_minus_random"] = {
                "source_label_switch_rate": (
                    t["source_label_switch_rate"] - r["source_label_switch_rate"]
                ),
                "mean_source_vs_target_margin_shift": (
                    t["mean_source_vs_target_margin_shift"]
                    - r["mean_source_vs_target_margin_shift"]
                ),
                "mean_final_effect_norm": (
                    t["mean_final_effect_norm"] - r["mean_final_effect_norm"]
                ),
            }
            result[component]["layers"][str(layer)] = modes

    return result


@torch.no_grad()
def adapter_ablation(model, tok, triples, clean_reps, centroids, device):
    ordered = sorted(
        triples, key=lambda x: hashlib.sha1(x[3].encode("utf-8")).hexdigest()
    )[:96]

    result = {}
    for layer in PATCH_LAYERS:
        modes = {}
        for mode in ("ablate", "random_matched"):
            effects = []
            hits = {"core": 0, "goal": 0, "focus": 0}
            ratios = []
            for idx, (c, g, f, text) in enumerate(ordered):
                ids = torch.tensor([encode(tok, text)], dtype=torch.long, device=device)
                rep, stats = forward_with_adapter_ablation(
                    model, ids, layer, mode, seed=330_001 + 1009 * layer + idx
                )
                rep = rep[0].detach().cpu()
                clean = clean_reps[text]
                effects.append(
                    float((unit(rep.unsqueeze(0))[0] - unit(clean.unsqueeze(0))[0]).norm())
                )
                ratios.append(stats["adapter_delta_over_residual"])
                hits["core"] += int(predict(rep, centroids["core"]) == c)
                hits["goal"] += int(predict(rep, centroids["goal"]) == g)
                hits["focus"] += int(predict(rep, centroids["focus"]) == f)

            modes[mode] = {
                "mean_final_effect_norm": sum(effects) / len(effects),
                "component_accuracy": {
                    k: v / len(ordered) for k, v in hits.items()
                },
                "mean_adapter_delta_over_residual": sum(ratios) / len(ratios),
            }

        a = modes["ablate"]
        r = modes["random_matched"]
        modes["ablate_minus_random"] = {
            "effect_difference": (
                a["mean_final_effect_norm"] - r["mean_final_effect_norm"]
            ),
            "effect_ratio": (
                a["mean_final_effect_norm"]
                / max(r["mean_final_effect_norm"], 1e-12)
            ),
            "core_accuracy_delta": (
                a["component_accuracy"]["core"]
                - r["component_accuracy"]["core"]
            ),
            "goal_accuracy_delta": (
                a["component_accuracy"]["goal"]
                - r["component_accuracy"]["goal"]
            ),
            "focus_accuracy_delta": (
                a["component_accuracy"]["focus"]
                - r["component_accuracy"]["focus"]
            ),
        }
        result[str(layer)] = modes
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("adapter causal assay requires a SELF variant")

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

    clean_reps = encode_clean(model, tok, triples, args.device)
    centroids = build_centroids(clean_reps, triples)

    result = {
        "schema_version": 1,
        "experiment": "300M SELF adapter contribution causal patching v2",
        "variant": ckpt.get("variant"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "patch_layers": list(PATCH_LAYERS),
        "clean_final_component_accuracy": clean_accuracy(
            clean_reps, triples, centroids
        ),
        "source_adapter_delta_patching": adapter_source_delta_patching(
            model, tok, triples, clean_reps, centroids, args.device
        ),
        "adapter_ablation": adapter_ablation(
            model, tok, triples, clean_reps, centroids, args.device
        ),
        "primary_question": (
            "Does the SELF adapter contribution itself carry component-specific "
            "causal information, beyond the base transformer residual stream?"
        ),
        "guardrails": [
            "Only the adapter contribution is patched or removed; the base transformer residual remains target-derived.",
            "Source/target prompts differ in exactly one component and have identical token shape.",
            "Random controls are matched to the exact adapter-level intervention norm.",
            "Ablation is compared with a random perturbation matched to the removed adapter-delta norm.",
            "This tests the explicit SELF adapter pathway, not semantic selfhood.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "variant": result["variant"],
        "tokens_seen": result["tokens_seen"],
        "clean": result["clean_final_component_accuracy"],
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
