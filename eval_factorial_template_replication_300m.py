from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_causal_activation_patching_300m import build_centroids
from eval_self_core_factorial_interaction_300m import (
    COMPONENTS,
    build_margin_changes,
    component_accuracy,
    interaction_matrix,
    summarize_vs_random,
)
from eval_self_core_interaction_field_300m import encode_condition, radial2, random2
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


TEMPLATES = {
    "canonical": "הליבה היא {core}. היעד הנוכחי הוא {goal}. המוקד כעת הוא {focus}.",
    "work_labels": "נושא העבודה: {core}. מטרת העבודה: {goal}. נקודת המיקוד: {focus}.",
    "declarative": "הנושא המרכזי הוא {core}. המטרה שנבחרה היא {goal}. כרגע מתמקדים בעניין {focus}.",
    "contextual": "בהקשר של {core}, היעד הוא {goal}, והעניין שבמוקד הוא {focus}.",
}


def make_triples(template_name: str):
    fmt = TEMPLATES[template_name]
    cores = list(TEST_CORES)
    goals = list(TEST_GOALS[:6])
    foci = list(TEST_FOCI[:6])
    return [
        (c, g, f, fmt.format(core=c, goal=g, focus=f))
        for c in cores for g in goals for f in foci
    ]


def aggregate_primary(heldouts: dict):
    out = {}
    for cell in ("core->goal", "core->focus"):
        factor, outcome = cell.split("->")
        for metric in ("rms_interaction", "mean_abs_interaction"):
            radial_by_template = []
            sham_means_by_index = None
            for payload in heldouts.values():
                rv = payload["radial_interaction_matrix"][factor][outcome][metric]
                radial_by_template.append(rv)
                sham_vals = [
                    r["matrix"][factor][outcome][metric]
                    for r in payload["random_runs"]
                ]
                if sham_means_by_index is None:
                    sham_means_by_index = [[] for _ in sham_vals]
                for i, v in enumerate(sham_vals):
                    sham_means_by_index[i].append(v)

            radial_mean = statistics.mean(radial_by_template)
            sham_aggregate = [
                statistics.mean(xs) for xs in sham_means_by_index
            ]
            random_mean = statistics.mean(sham_aggregate)
            random_sd = statistics.pstdev(sham_aggregate)
            out.setdefault(cell, {})[metric] = {
                "radial_template_mean": radial_mean,
                "random_template_mean": random_mean,
                "random_sd_across_shams": random_sd,
                "ratio_to_random_mean": radial_mean / max(random_mean, 1e-12),
                "z_vs_random": None if random_sd < 1e-12 else (
                    radial_mean - random_mean
                ) / random_sd,
                "radial_rank_among_9_desc": 1 + sum(
                    x > radial_mean for x in sham_aggregate
                ),
                "radial_by_heldout_template": radial_by_template,
                "random_sham_template_means": sham_aggregate,
            }
    return out


def aggregate_accuracy(heldouts: dict, key: str):
    return {
        comp: statistics.mean(
            payload[key][comp] for payload in heldouts.values()
        )
        for comp in COMPONENTS
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--random-shams", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("template replication requires a SELF variant")
    model.eval()
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    triples_by_template = {name: make_triples(name) for name in TEMPLATES}
    all_triples = [
        row for name in TEMPLATES for row in triples_by_template[name]
    ]

    anchor = model.self_anchor.detach().cpu().float()
    radial = radial2(anchor)
    shams = [random2(anchor, 1_210_001 + i * 149) for i in range(args.random_shams)]

    report_layers = {}
    for layer in layers:
        print(f"[layer {layer}] normal all templates", flush=True)
        normal = encode_condition(model, tok, all_triples, args.device, layer, None)
        print(f"[layer {layer}] radial all templates", flush=True)
        radial_cache = encode_condition(
            model, tok, all_triples, args.device, layer, radial
        )

        random_caches = []
        for i, replacement in enumerate(shams, 1):
            print(f"[layer {layer}] random {i}/{len(shams)}", flush=True)
            random_caches.append(
                encode_condition(
                    model, tok, all_triples, args.device, layer, replacement
                )
            )

        heldouts = {}
        for heldout_name, test_triples in triples_by_template.items():
            train_triples = [
                row
                for name, rows in triples_by_template.items()
                if name != heldout_name
                for row in rows
            ]
            clean_centroids = build_centroids(normal["final"], train_triples)

            radial_changes = build_margin_changes(
                normal["final"],
                radial_cache["final"],
                test_triples,
                clean_centroids,
            )
            radial_matrix = interaction_matrix(radial_changes, test_triples)

            random_runs = []
            for i, cache in enumerate(random_caches, 1):
                changes = build_margin_changes(
                    normal["final"],
                    cache["final"],
                    test_triples,
                    clean_centroids,
                )
                random_runs.append({
                    "index": i,
                    "matrix": interaction_matrix(changes, test_triples),
                    "component_accuracy": component_accuracy(
                        cache["final"], test_triples, clean_centroids
                    ),
                })

            heldouts[heldout_name] = {
                "train_templates": [
                    x for x in TEMPLATES if x != heldout_name
                ],
                "normal_component_accuracy": component_accuracy(
                    normal["final"], test_triples, clean_centroids
                ),
                "radial_component_accuracy": component_accuracy(
                    radial_cache["final"], test_triples, clean_centroids
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

        report_layers[str(layer)] = {
            "heldout_templates": heldouts,
            "aggregate": {
                "normal_component_accuracy": aggregate_accuracy(
                    heldouts, "normal_component_accuracy"
                ),
                "radial_component_accuracy": aggregate_accuracy(
                    heldouts, "radial_component_accuracy"
                ),
                "random_component_accuracy_mean": aggregate_accuracy(
                    heldouts, "random_component_accuracy_mean"
                ),
                "primary": aggregate_primary(heldouts),
            },
        }

    result = {
        "schema_version": 1,
        "experiment": "300M factorial SELF context held-out-template replication v1",
        "variant": ckpt.get("variant"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "layers": report_layers,
        "templates": TEMPLATES,
        "random_shams": args.random_shams,
        "design": {
            "leave_one_template_out": True,
            "centroids_fit_only_on_other_templates": True,
            "test_full_factorial_on_heldout_template": True,
            "primary_cells": ["core->goal", "core->focus"],
            "primary_metric": "rms_interaction",
        },
        "success_criterion": (
            "Projected variants retain radial SELF-by-CORE interaction above "
            "eight equal-displacement random sham directions across held-out "
            "wording families, while CORE is more stable than diff_anchor."
        ),
        "guardrails": [
            "The held-out wording family is never used to construct its decoder centroids.",
            "Radial SELF and all random sham directions use identical displacement magnitude.",
            "Interaction is measured as the context-dependence of intervention-induced correct-label margin changes.",
            "This is still a single training seed and does not establish semantic selfhood.",
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
        "layers": layers,
        "out": args.out,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
