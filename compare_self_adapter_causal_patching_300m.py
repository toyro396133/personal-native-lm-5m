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
        if x.get("experiment") != "300M SELF adapter contribution causal patching v2":
            continue

        row = {
            "variant": x["variant"],
            "tokens_seen": x.get("tokens_seen"),
            "clean": x["clean_final_component_accuracy"],
            "source_adapter_delta_patching": {},
            "adapter_ablation": {},
            "source": str(p),
        }

        for comp, payload in x["source_adapter_delta_patching"].items():
            row["source_adapter_delta_patching"][comp] = {}
            for layer, modes in payload["layers"].items():
                row["source_adapter_delta_patching"][comp][layer] = {
                    "pair_count": payload["pair_count"],
                    "targeted_switch": modes["targeted"]["source_label_switch_rate"],
                    "random_switch": modes["random_matched"]["source_label_switch_rate"],
                    "switch_advantage": modes["targeted_minus_random"]["source_label_switch_rate"],
                    "targeted_margin_shift": modes["targeted"]["mean_source_vs_target_margin_shift"],
                    "random_margin_shift": modes["random_matched"]["mean_source_vs_target_margin_shift"],
                    "margin_shift_advantage": modes["targeted_minus_random"]["mean_source_vs_target_margin_shift"],
                    "targeted_effect": modes["targeted"]["mean_final_effect_norm"],
                    "random_effect": modes["random_matched"]["mean_final_effect_norm"],
                    "mean_adapter_change_over_residual": modes["targeted"]["mean_adapter_change_over_residual"],
                    "targeted_unchanged_accuracy": modes["targeted"]["unchanged_component_accuracy"],
                    "random_unchanged_accuracy": modes["random_matched"]["unchanged_component_accuracy"],
                }

        for layer, modes in x["adapter_ablation"].items():
            row["adapter_ablation"][layer] = {
                "ablate_effect": modes["ablate"]["mean_final_effect_norm"],
                "random_effect": modes["random_matched"]["mean_final_effect_norm"],
                "effect_ratio": modes["ablate_minus_random"]["effect_ratio"],
                "mean_adapter_delta_over_residual": modes["ablate"]["mean_adapter_delta_over_residual"],
                "ablate_accuracy": modes["ablate"]["component_accuracy"],
                "random_accuracy": modes["random_matched"]["component_accuracy"],
                "accuracy_delta_ablate_minus_random": {
                    "core": modes["ablate_minus_random"]["core_accuracy_delta"],
                    "goal": modes["ablate_minus_random"]["goal_accuracy_delta"],
                    "focus": modes["ablate_minus_random"]["focus_accuracy_delta"],
                },
            }

        rows.append(row)

    if len(rows) != 4:
        raise SystemExit(f"expected 4 v2 reports, found {len(rows)}")

    rows.sort(key=lambda r: r["variant"])
    payload = {
        "schema_version": 1,
        "experiment": "300M SELF adapter contribution causal comparison v2",
        "rows": rows,
        "interpretation": {
            "primary": (
                "Targeted source adapter-delta patches should exceed matched random "
                "controls in relevant source-vs-target margin shift if the explicit "
                "SELF adapter path carries component-specific causal information."
            ),
            "ablation": (
                "Ablation vs matched-random differences test whether the learned "
                "adapter contribution itself is necessary in an architecture- and "
                "layer-specific way."
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
