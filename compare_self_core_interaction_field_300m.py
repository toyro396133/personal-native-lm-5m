from __future__ import annotations

import argparse
import json
from pathlib import Path


def extract_metric(layer_payload, location, component, metric):
    block = layer_payload[f"radial_vs_random_{location}"][component][metric]
    return {
        "radial": block["radial"],
        "random_mean": block["random_mean"],
        "random_sd": block["random_sd"],
        "advantage": block["radial_minus_random_mean"],
        "z_vs_random": block["z_vs_random"],
        "rank": block["radial_rank_among_9_desc"],
    }


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
        if x.get("experiment") != "300M SELF by CORE interaction field v3":
            continue

        row = {
            "variant": x["variant"],
            "tokens_seen": x.get("tokens_seen"),
            "layers": {},
            "source": str(p),
        }
        for layer, payload in x["layers"].items():
            row["layers"][layer] = {
                "normal_component_accuracy": payload["normal_component_accuracy"],
                "radial_component_accuracy": payload["radial_component_accuracy"],
                "random_component_accuracy_mean": payload["random_component_accuracy_mean"],
                "radial_local_effect": payload["radial_local_effect"],
                "radial_final_effect": payload["radial_final_effect"],
                "local": {},
                "final": {},
            }
            for location in ("local", "final"):
                for component in ("core", "goal", "focus"):
                    row["layers"][layer][location][component] = {
                        "cross_context_accuracy": extract_metric(
                            payload, location, component, "cross_context_accuracy"
                        ),
                        "cluster_margin": extract_metric(
                            payload, location, component, "cluster_margin"
                        ),
                    }
        rows.append(row)

    if len(rows) != 4:
        raise SystemExit(f"expected 4 v3 reports, found {len(rows)}")

    rows.sort(key=lambda r: r["variant"])
    result = {
        "schema_version": 1,
        "experiment": "300M SELF by CORE interaction field comparison v3",
        "rows": rows,
        "primary_readout": (
            "Final downstream CORE cross-context accuracy and CORE cluster "
            "margin in the radial SELF effect field versus eight random "
            "same-displacement anchor directions."
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
