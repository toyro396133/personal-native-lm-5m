from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    for p in sorted(Path(args.root).rglob("*.json")):
        try:
            x = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if x.get("experiment") != "300M causal activation patching v1":
            continue

        row = {
            "variant": x["variant"],
            "tokens_seen": x.get("tokens_seen"),
            "clean": x["clean_final_component_accuracy"],
            "component_patching": {},
            "self_layer_causality": {},
            "source": str(p),
        }

        for comp, payload in x["component_activation_patching"].items():
            row["component_patching"][comp] = {}
            for layer, modes in payload["layers"].items():
                row["component_patching"][comp][layer] = {
                    "pair_count": payload["pair_count"],
                    "targeted_switch": modes["targeted"]["source_label_switch_rate"],
                    "random_switch": modes["random_matched"]["source_label_switch_rate"],
                    "switch_advantage": modes["targeted_minus_random"]["source_label_switch_rate"],
                    "targeted_margin_shift": modes["targeted"]["mean_source_vs_target_margin_shift"],
                    "random_margin_shift": modes["random_matched"]["mean_source_vs_target_margin_shift"],
                    "margin_shift_advantage": modes["targeted_minus_random"]["mean_source_vs_target_margin_shift"],
                    "targeted_unchanged_accuracy": modes["targeted"]["unchanged_component_accuracy"],
                    "random_unchanged_accuracy": modes["random_matched"]["unchanged_component_accuracy"],
                }

        if x.get("single_layer_self_causality"):
            for layer, payload in x["single_layer_self_causality"].items():
                row["self_layer_causality"][layer] = {
                    "radial_effect": payload["radial_2x"]["mean_final_effect_norm"],
                    "random_effect": payload["random_2x"]["mean_final_effect_norm"],
                    "effect_ratio": payload["radial_vs_random"]["effect_ratio"],
                    "radial_accuracy": payload["radial_2x"]["component_accuracy"],
                    "random_accuracy": payload["random_2x"]["component_accuracy"],
                    "accuracy_delta_radial_minus_random": {
                        "core": payload["radial_vs_random"]["core_accuracy_delta"],
                        "goal": payload["radial_vs_random"]["goal_accuracy_delta"],
                        "focus": payload["radial_vs_random"]["focus_accuracy_delta"],
                    },
                }
        rows.append(row)

    if len(rows) < 4:
        raise SystemExit(f"expected at least 4 causal reports, found {len(rows)}")
    rows.sort(key=lambda r: r["variant"])

    payload = {
        "schema_version": 1,
        "experiment": "300M causal activation patching comparison v1",
        "rows": rows,
        "interpretation": {
            "positive_component_evidence": (
                "Targeted residual patch should exceed random matched-norm control "
                "on source-label switching or continuous source-vs-target margin shift, "
                "while preserving unchanged components."
            ),
            "positive_self_layer_evidence": (
                "Single-layer radial SELF perturbation should have a larger downstream "
                "effect than a random same-displacement anchor perturbation."
            ),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
