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
        if x.get("experiment") != "300M factorial SELF context held-out-template replication v1":
            continue
        for layer, payload in x["layers"].items():
            agg = payload["aggregate"]
            rows.append({
                "variant": x["variant"],
                "tokens_seen": x["tokens_seen"],
                "layer": int(layer),
                "normal_component_accuracy": agg["normal_component_accuracy"],
                "radial_component_accuracy": agg["radial_component_accuracy"],
                "random_component_accuracy_mean": agg["random_component_accuracy_mean"],
                "core_to_goal": agg["primary"]["core->goal"]["rms_interaction"],
                "core_to_focus": agg["primary"]["core->focus"]["rms_interaction"],
                "heldout_templates": {
                    name: {
                        "normal_component_accuracy": hp["normal_component_accuracy"],
                        "radial_component_accuracy": hp["radial_component_accuracy"],
                        "random_component_accuracy_mean": hp["random_component_accuracy_mean"],
                        "core_to_goal_ratio": hp["radial_vs_random"]["core"]["goal"]["rms_interaction"]["ratio_to_random_mean"],
                        "core_to_focus_ratio": hp["radial_vs_random"]["core"]["focus"]["rms_interaction"]["ratio_to_random_mean"],
                        "core_to_goal_rank": hp["radial_vs_random"]["core"]["goal"]["rms_interaction"]["radial_rank_among_9_desc"],
                        "core_to_focus_rank": hp["radial_vs_random"]["core"]["focus"]["rms_interaction"]["radial_rank_among_9_desc"],
                    }
                    for name, hp in payload["heldout_templates"].items()
                },
                "source": str(p),
            })

    if len(rows) != 4:
        raise SystemExit(f"expected 4 reports, found {len(rows)}")

    rows.sort(key=lambda r: r["variant"])
    result = {
        "schema_version": 1,
        "experiment": "300M held-out-template factorial causal comparison v1",
        "rows": rows,
        "decision_rule": (
            "Strong replication requires projected variants to rank radial SELF "
            "above all 8 shams for CORE->GOAL/FOCUS on the aggregate and across "
            "multiple held-out templates, without the CORE collapse characteristic "
            "of diff_anchor."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
