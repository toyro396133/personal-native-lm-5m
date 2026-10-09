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
        if x.get("experiment") != "300M factorial SELF by context functional interaction v4":
            continue

        milestone = int(round((x.get("tokens_seen") or 0) / 1_000_000))
        row = {
            "variant": x["variant"],
            "milestone_m": milestone,
            "tokens_seen": x.get("tokens_seen"),
            "layers": {},
            "source": str(p),
        }
        for layer, payload in x["layers"].items():
            def pick(factor, outcome):
                rms = payload["radial_vs_random"][factor][outcome]["rms_interaction"]
                ma = payload["radial_vs_random"][factor][outcome]["mean_abs_interaction"]
                return {
                    "rms": {
                        "radial": rms["radial"],
                        "random_mean": rms["random_mean"],
                        "ratio": rms["ratio_to_random_mean"],
                        "z": rms["z_vs_random"],
                        "rank": rms["radial_rank_among_9_desc"],
                    },
                    "mean_abs": {
                        "radial": ma["radial"],
                        "random_mean": ma["random_mean"],
                        "ratio": ma["ratio_to_random_mean"],
                        "z": ma["z_vs_random"],
                        "rank": ma["radial_rank_among_9_desc"],
                    },
                }

            row["layers"][layer] = {
                "normal_component_accuracy": payload["normal_component_accuracy"],
                "radial_component_accuracy": payload["radial_component_accuracy"],
                "core_to_goal": pick("core", "goal"),
                "core_to_focus": pick("core", "focus"),
                "core_to_core": pick("core", "core"),
                "goal_to_goal": pick("goal", "goal"),
                "focus_to_focus": pick("focus", "focus"),
            }
        rows.append(row)

    expected = 12
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} trajectory reports, found {len(rows)}")

    rows.sort(key=lambda r: (r["variant"], r["milestone_m"]))
    result = {
        "schema_version": 1,
        "experiment": "factorial SELF CORE causal trajectory 150-200M",
        "reuses_assay": "300M factorial SELF by context functional interaction v4",
        "primary_cells": ["core->goal", "core->focus"],
        "primary_metric": "rms_interaction",
        "rows": rows,
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
