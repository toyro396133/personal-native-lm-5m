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
        if x.get("experiment") != "SELF layer causal trajectory v1":
            continue

        row = {
            "variant": x["variant"],
            "tokens_seen": x["tokens_seen"],
            "clean": x["clean"],
            "layers": {},
            "source": str(p),
        }
        for layer, payload in x["layers"].items():
            row["layers"][layer] = payload["matched_direction_summary"]
        rows.append(row)

    if len(rows) != 16:
        raise SystemExit(f"expected 16 reports, found {len(rows)}")

    rows.sort(key=lambda r: (r["variant"], r["tokens_seen"]))
    payload = {
        "schema_version": 1,
        "experiment": "SELF layer causal trajectory comparison 150M-300M",
        "rows": rows,
        "primary_reading": (
            "Track when a one-layer radial SELF intervention causes more "
            "downstream GOAL/FOCUS damage than displacement-matched random/"
            "orthogonal controls while preserving CORE."
        ),
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
