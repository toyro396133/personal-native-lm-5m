from __future__ import annotations

import argparse
import json
from pathlib import Path


PRIMARY = (("core", "goal"), ("core", "focus"))


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

        row = {
            "variant": x["variant"],
            "tokens_seen": x.get("tokens_seen"),
            "layers": {},
            "source": str(p),
        }
        for layer, payload in x["layers"].items():
            cells = {}
            for factor in ("core", "goal", "focus"):
                cells[factor] = {}
                for outcome in ("core", "goal", "focus"):
                    cells[factor][outcome] = {}
                    for metric, block in payload["radial_vs_random"][factor][outcome].items():
                        cells[factor][outcome][metric] = block
            row["layers"][layer] = {
                "normal_component_accuracy": payload["normal_component_accuracy"],
                "radial_component_accuracy": payload["radial_component_accuracy"],
                "random_component_accuracy_mean": payload["random_component_accuracy_mean"],
                "cells": cells,
                "primary": {
                    f"{factor}->{outcome}": {
                        "rms": cells[factor][outcome]["rms_interaction"],
                        "mean_abs": cells[factor][outcome]["mean_abs_interaction"],
                    }
                    for factor, outcome in PRIMARY
                },
            }
        rows.append(row)

    if len(rows) != 4:
        raise SystemExit(f"expected 4 v4 reports, found {len(rows)}")

    rows.sort(key=lambda r: r["variant"])
    result = {
        "schema_version": 1,
        "experiment": "300M factorial SELF by context comparison v4",
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
