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
        if x.get("experiment") != "internal SELF x CORE subspace factorial intervention v1":
            continue
        rows.append({
            "variant": x["variant"],
            "tokens_seen": x["tokens_seen"],
            "layer": x["layer"],
            "core_subspace_rank": x["core_subspace"]["rank"],
            "mean_core_patch_fraction_of_target_pool": x["mean_core_patch_fraction_of_target_pool"],
            "core_manipulation_check": x["core_manipulation_check"],
            "interaction_outcomes": x["interaction_outcomes"],
            "source": str(p),
        })

    if len(rows) != 6:
        raise SystemExit(f"expected 6 reports, found {len(rows)}")

    rows.sort(key=lambda r: (r["variant"], r["tokens_seen"]))
    payload = {
        "schema_version": 1,
        "experiment": "internal SELF x CORE subspace causal comparison 200M-300M",
        "rows": rows,
        "primary_success_criteria": [
            "targeted CORE subspace changes source-CORE prediction more than random orthogonal control",
            "radial SELF x targeted CORE interaction exceeds eight random-SELF x targeted-CORE interactions",
            "radial SELF x targeted CORE interaction exceeds radial SELF x random-CORE interaction",
            "projected family shows cleaner GOAL/FOCUS interaction with less CORE collapse than diff_anchor",
        ],
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
