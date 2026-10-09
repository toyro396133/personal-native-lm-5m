from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, bos=True, eos=True)[:112]


def prompt(core: str, goal: str, focus: str) -> str:
    return f"הליבה היא {core}. היעד הנוכחי הוא {goal}. המוקד כעת הוא {focus}."


def replacement_set(anchor: torch.Tensor) -> dict[str, torch.Tensor]:
    a = anchor.detach().cpu().float()
    n = a.norm().clamp_min(1e-12)

    g = torch.Generator(device="cpu")
    g.manual_seed(902_441)
    random_dir = torch.randn(a.shape, generator=g)
    random_dir = random_dir / random_dir.norm().clamp_min(1e-12)

    g.manual_seed(902_447)
    orth = torch.randn(a.shape, generator=g)
    orth = orth - (orth @ a) / a.square().sum().clamp_min(1e-12) * a
    orth = orth / orth.norm().clamp_min(1e-12)

    return {
        "normal": a.clone(),
        "radial_1x": torch.zeros_like(a),
        "radial_2x": -a,
        "random_1x": a + n * random_dir,
        "random_2x": a + 2.0 * n * random_dir,
        "orthogonal_1x": a + n * orth,
        "orthogonal_2x": a + 2.0 * n * orth,
    }


def init_stream(model: SelfVariantPersonalNativeLM, ids: torch.Tensor) -> torch.Tensor:
    base = model.base
    x = base.token_embedding(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)
    return x + base.position_embedding(pos)[None, :, :]


@torch.no_grad()
def final_rep_single_layer_anchor(
    model: SelfVariantPersonalNativeLM,
    ids: torch.Tensor,
    patch_layer: int | None,
    replacement: torch.Tensor | None,
) -> torch.Tensor:
    x = init_stream(model, ids)
    for i, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters), 1):
        x = block(x, None)
        anchor = model.self_anchor
        if patch_layer is not None and i == patch_layer:
            anchor = replacement.to(model.self_anchor)
        x = adapter(x, anchor)
    x = model.base.final_norm(x)
    return x.mean(dim=1)


@torch.no_grad()
def encode_texts(
    model: SelfVariantPersonalNativeLM,
    tok,
    texts: list[str],
    device: str,
    patch_layer: int | None = None,
    replacement: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    groups = defaultdict(list)
    for text in sorted(set(texts)):
        ids = encode(tok, text)
        groups[len(ids)].append((text, ids))

    out = {}
    for length in sorted(groups):
        items = groups[length]
        for start in range(0, len(items), 64):
            chunk = items[start : start + 64]
            ids = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
            reps = final_rep_single_layer_anchor(
                model, ids, patch_layer, replacement
            ).detach().cpu()
            for (text, _), rep in zip(chunk, reps):
                out[text] = rep
    return out


def build_centroids(
    cache: dict[str, torch.Tensor],
    triples: list[tuple[str, str, str, str]],
):
    by = {
        "core": defaultdict(list),
        "goal": defaultdict(list),
        "focus": defaultdict(list),
    }
    for core, goal, focus, text in triples:
        rep = unit(cache[text].unsqueeze(0))[0]
        by["core"][core].append(rep)
        by["goal"][goal].append(rep)
        by["focus"][focus].append(rep)

    centroids = {}
    for component, groups in by.items():
        centroids[component] = {
            label: unit(torch.stack(xs).mean(0, keepdim=True))[0]
            for label, xs in groups.items()
        }
    return centroids


def classify_and_margin(
    rep: torch.Tensor,
    centroids: dict[str, torch.Tensor],
    correct: str,
) -> tuple[bool, float]:
    labels = sorted(centroids)
    c = torch.stack([centroids[x] for x in labels])
    scores = unit(rep.unsqueeze(0))[0] @ c.T
    best = labels[int(scores.argmax())]
    ci = labels.index(correct)
    correct_score = float(scores[ci])
    wrong = [float(scores[i]) for i in range(len(labels)) if i != ci]
    margin = correct_score - max(wrong)
    return best == correct, margin


def evaluate_cache(
    cache: dict[str, torch.Tensor],
    triples: list[tuple[str, str, str, str]],
    centroids,
):
    index = {"core": 0, "goal": 1, "focus": 2}
    result = {}
    for component, idx in index.items():
        hits = 0
        margins = []
        for row in triples:
            correct = row[idx]
            ok, margin = classify_and_margin(
                cache[row[3]], centroids[component], correct
            )
            hits += int(ok)
            margins.append(margin)
        result[component] = {
            "accuracy": hits / len(triples),
            "mean_correct_vs_nearest_wrong_margin": sum(margins) / len(margins),
        }
    return result


def mean_effect(
    clean: dict[str, torch.Tensor],
    alt: dict[str, torch.Tensor],
    texts: list[str],
) -> float:
    vals = []
    for text in texts:
        a = unit(clean[text].unsqueeze(0))[0]
        b = unit(alt[text].unsqueeze(0))[0]
        vals.append(float((a - b).norm()))
    return sum(vals) / len(vals)


def summarize_layer(layer_payload: dict, clean_eval: dict):
    out = {}
    for scale in ("1x", "2x"):
        radial = layer_payload[f"radial_{scale}"]
        controls = [
            layer_payload[f"random_{scale}"],
            layer_payload[f"orthogonal_{scale}"],
        ]
        control_effect = sum(x["mean_final_effect_norm"] for x in controls) / len(controls)
        comp = {}
        for component in ("core", "goal", "focus"):
            clean_margin = clean_eval[component]["mean_correct_vs_nearest_wrong_margin"]
            radial_damage = (
                clean_margin
                - radial["components"][component]["mean_correct_vs_nearest_wrong_margin"]
            )
            control_damage = sum(
                clean_margin
                - x["components"][component]["mean_correct_vs_nearest_wrong_margin"]
                for x in controls
            ) / len(controls)
            comp[component] = {
                "radial_damage": radial_damage,
                "control_mean_damage": control_damage,
                "radial_minus_control_damage": radial_damage - control_damage,
                "radial_accuracy": radial["components"][component]["accuracy"],
                "control_mean_accuracy": sum(
                    x["components"][component]["accuracy"] for x in controls
                ) / len(controls),
            }

        downstream_damage = (
            comp["goal"]["radial_damage"] + comp["focus"]["radial_damage"]
        ) / 2.0
        control_downstream_damage = (
            comp["goal"]["control_mean_damage"]
            + comp["focus"]["control_mean_damage"]
        ) / 2.0

        out[scale] = {
            "effect_ratio_radial_vs_control_mean": (
                radial["mean_final_effect_norm"] / max(control_effect, 1e-12)
            ),
            "effect_difference_radial_minus_control_mean": (
                radial["mean_final_effect_norm"] - control_effect
            ),
            "components": comp,
            "radial_downstream_damage": downstream_damage,
            "control_mean_downstream_damage": control_downstream_damage,
            "radial_minus_control_downstream_damage": (
                downstream_damage - control_downstream_damage
            ),
            "radial_downstream_minus_core_damage": (
                downstream_damage - comp["core"]["radial_damage"]
            ),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("SELF-layer causal trajectory assay requires a SELF variant")

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
    texts = [x[3] for x in triples]

    clean = encode_texts(model, tok, texts, args.device)
    centroids = build_centroids(clean, triples)
    clean_eval = evaluate_cache(clean, triples, centroids)

    replacements = replacement_set(model.self_anchor.detach().cpu())
    layers = {}
    for layer in range(1, cfg.n_layers + 1):
        print(
            f"variant={ckpt.get('variant')} tokens={ckpt.get('tokens_seen')} layer={layer}",
            flush=True,
        )
        raw_modes = {}
        for name, replacement in replacements.items():
            if name == "normal":
                continue
            alt = encode_texts(
                model,
                tok,
                texts,
                args.device,
                patch_layer=layer,
                replacement=replacement,
            )
            raw_modes[name] = {
                "mean_final_effect_norm": mean_effect(clean, alt, texts),
                "components": evaluate_cache(alt, triples, centroids),
            }
        layers[str(layer)] = {
            "interventions": raw_modes,
            "matched_direction_summary": summarize_layer(raw_modes, clean_eval),
        }

    report = {
        "schema_version": 1,
        "experiment": "SELF layer causal trajectory v1",
        "checkpoint": args.checkpoint,
        "variant": ckpt.get("variant"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "layers": layers,
        "clean": clean_eval,
        "intervention_design": {
            "scope": "only the SELF anchor supplied to one adapter layer is changed",
            "directions": ["radial", "random", "orthogonal"],
            "magnitudes": ["1x learned anchor norm", "2x learned anchor norm"],
            "radial_1x": "zero anchor at one layer only",
            "radial_2x": "negated anchor at one layer only",
        },
        "primary_hypothesis": (
            "Projected SELF variants should show direction-specific downstream "
            "GOAL/FOCUS effects while preserving CORE more strongly than entangled controls."
        ),
        "guardrails": [
            "Only one adapter layer is intervened on per measurement.",
            "Random and orthogonal controls are exactly displacement-matched.",
            "Centroids are built from clean representations only.",
            "Both discrete accuracy and continuous correct-vs-nearest-wrong margin are reported.",
            "No semantic selfhood claim follows from this assay.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "variant": report["variant"],
                "tokens_seen": report["tokens_seen"],
                "clean": clean_eval,
                "out": args.out,
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
