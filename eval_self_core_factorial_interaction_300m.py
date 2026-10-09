from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_causal_activation_patching_300m import build_centroids, predict, prompt, unit
from eval_self_core_interaction_field_300m import (
    encode_condition,
    radial2,
    random2,
)
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


COMPONENTS = ("core", "goal", "focus")


def correct_margin(rep, centroids, true_label):
    r = unit(rep.unsqueeze(0))[0]
    true_score = float(r @ centroids[true_label])
    other = max(
        float(r @ v)
        for k, v in centroids.items()
        if k != true_label
    )
    return true_score - other


def build_margin_changes(normal_final, alt_final, triples, centroids):
    out = {}
    for c, g, f, text in triples:
        labels = {"core": c, "goal": g, "focus": f}
        out[text] = {}
        for comp in COMPONENTS:
            m0 = correct_margin(normal_final[text], centroids[comp], labels[comp])
            m1 = correct_margin(alt_final[text], centroids[comp], labels[comp])
            out[text][comp] = m1 - m0
    return out


def factorial_interaction_stats(changes, triples, factor: str, outcome: str):
    idx = {"core": 0, "goal": 1, "focus": 2}
    fi = idx[factor]
    other_idx = [i for i in range(3) if i != fi]

    groups = defaultdict(list)
    for row in triples:
        c, g, f, text = row
        key = tuple(row[i] for i in other_idx)
        factor_value = row[fi]
        groups[key].append((factor_value, changes[text][outcome]))

    pairwise = []
    context_stds = []
    ranges = []
    for items in groups.values():
        # one value per factor level in the full factorial grid
        items = sorted(items)
        vals = [v for _, v in items]
        if len(vals) < 2:
            continue
        context_stds.append(statistics.pstdev(vals))
        ranges.append(max(vals) - min(vals))
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairwise.append(vals[j] - vals[i])

    if not pairwise:
        raise RuntimeError(f"no factorial pairs for {factor}->{outcome}")

    abs_vals = [abs(x) for x in pairwise]
    sq = [x * x for x in pairwise]
    return {
        "pair_count": len(pairwise),
        "context_count": len(context_stds),
        "mean_abs_interaction": statistics.mean(abs_vals),
        "rms_interaction": math.sqrt(statistics.mean(sq)),
        "mean_context_std": statistics.mean(context_stds),
        "mean_context_range": statistics.mean(ranges),
        "signed_mean": statistics.mean(pairwise),
    }


def interaction_matrix(changes, triples):
    return {
        factor: {
            outcome: factorial_interaction_stats(changes, triples, factor, outcome)
            for outcome in COMPONENTS
        }
        for factor in COMPONENTS
    }


def component_accuracy(reps, triples, centroids):
    hits = {k: 0 for k in COMPONENTS}
    for c, g, f, text in triples:
        labels = {"core": c, "goal": g, "focus": f}
        for comp in COMPONENTS:
            hits[comp] += int(predict(reps[text], centroids[comp]) == labels[comp])
    n = len(triples)
    return {k: v / n for k, v in hits.items()}


def summarize_vs_random(radial_matrix, random_matrices):
    out = {}
    for factor in COMPONENTS:
        out[factor] = {}
        for outcome in COMPONENTS:
            out[factor][outcome] = {}
            for metric in (
                "mean_abs_interaction",
                "rms_interaction",
                "mean_context_std",
                "mean_context_range",
            ):
                radial = radial_matrix[factor][outcome][metric]
                rv = [x[factor][outcome][metric] for x in random_matrices]
                mean = statistics.mean(rv)
                sd = statistics.pstdev(rv) if len(rv) > 1 else 0.0
                out[factor][outcome][metric] = {
                    "radial": radial,
                    "random_mean": mean,
                    "random_sd": sd,
                    "radial_minus_random_mean": radial - mean,
                    "ratio_to_random_mean": radial / max(mean, 1e-12),
                    "z_vs_random": None if sd < 1e-12 else (radial - mean) / sd,
                    "radial_rank_among_9_desc": 1 + sum(v > radial for v in rv),
                    "random_values": rv,
                }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--random-shams", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("factorial interaction assay requires a SELF variant")
    model.eval()
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    cores = list(TEST_CORES)
    goals = list(TEST_GOALS[:6])
    foci = list(TEST_FOCI[:6])
    triples = [
        (c, g, f, prompt(c, g, f))
        for c in cores for g in goals for f in foci
    ]

    anchor = model.self_anchor.detach().cpu().float()
    radial = radial2(anchor)
    shams = [random2(anchor, 910_001 + i * 131) for i in range(args.random_shams)]

    by_layer = {}
    for layer in layers:
        print(f"[layer {layer}] normal", flush=True)
        normal = encode_condition(model, tok, triples, args.device, layer, None)
        centroids = build_centroids(normal["final"], triples)

        print(f"[layer {layer}] radial", flush=True)
        rad = encode_condition(model, tok, triples, args.device, layer, radial)
        radial_changes = build_margin_changes(
            normal["final"], rad["final"], triples, centroids
        )
        radial_matrix = interaction_matrix(radial_changes, triples)

        random_runs = []
        for i, replacement in enumerate(shams, 1):
            print(f"[layer {layer}] random {i}/{len(shams)}", flush=True)
            alt = encode_condition(model, tok, triples, args.device, layer, replacement)
            changes = build_margin_changes(
                normal["final"], alt["final"], triples, centroids
            )
            random_runs.append({
                "index": i,
                "matrix": interaction_matrix(changes, triples),
                "component_accuracy": component_accuracy(
                    alt["final"], triples, centroids
                ),
            })

        by_layer[str(layer)] = {
            "normal_component_accuracy": component_accuracy(
                normal["final"], triples, centroids
            ),
            "radial_component_accuracy": component_accuracy(
                rad["final"], triples, centroids
            ),
            "random_component_accuracy_mean": {
                comp: statistics.mean(
                    r["component_accuracy"][comp] for r in random_runs
                )
                for comp in COMPONENTS
            },
            "radial_interaction_matrix": radial_matrix,
            "random_runs": random_runs,
            "radial_vs_random": summarize_vs_random(
                radial_matrix, [r["matrix"] for r in random_runs]
            ),
        }

    result = {
        "schema_version": 1,
        "experiment": "300M factorial SELF by context functional interaction v4",
        "variant": ckpt.get("variant"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "layers": by_layer,
        "random_shams": args.random_shams,
        "primary_cells": [
            "CORE factor -> GOAL margin response",
            "CORE factor -> FOCUS margin response",
        ],
        "primary_question": (
            "Does changing SELF have a non-additive effect on downstream GOAL/FOCUS "
            "that depends on CORE value, beyond eight random same-displacement "
            "SELF directions?"
        ),
        "guardrails": [
            "CORE is varied independently in the full factorial prompt grid while GOAL/FOCUS are held fixed within each interaction comparison.",
            "The outcome is the change in correct-label decoder margin caused by the SELF intervention.",
            "Interaction magnitude is compared against eight random SELF directions with identical anchor displacement.",
            "A large generic SELF effect is not sufficient; the CORE-conditioned non-additivity must exceed random controls.",
            "This remains a single-seed mechanistic assay.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "variant": result["variant"],
        "layers": layers,
        "out": args.out,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
