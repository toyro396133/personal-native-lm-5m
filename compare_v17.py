from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--phase", required=True)
    ap.add_argument("--milestones", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    variants = [
        "baseline",
        "self_v1",
        "self_v1_slow",
        "diff_only",
        "diff_anchor",
        "projected_diff",
        "projected_diff_slow",
    ]
    milestones = [int(x) for x in args.milestones.split(",") if x.strip()]
    root = Path(args.root)

    points = []
    hashes = set()
    for m in milestones:
        row = {"million_tokens": m, "variants": {}}
        for v in variants:
            candidates = sorted(root.glob(f"v17-p*-metrics-{v}/**/{m}m/train.json"))
            if len(candidates) != 1:
                raise SystemExit(
                    f"expected exactly one metrics file for variant={v} milestone={m}M; "
                    f"found={candidates}"
                )
            train_path = candidates[0]
            base = train_path.parent
            tr = json.loads(train_path.read_text())
            ev = json.loads((base / "eval.json").read_text())
            hashes.add(tr["base_sha256"])
            d = {
                "tokens_seen": tr["tokens_seen"],
                "last_loss": tr["last_loss"],
                "nats_per_char": ev["validation"]["nats_per_char"],
                "bits_per_char": ev["validation"]["bits_per_char"],
                "parameters": tr["parameters"],
                "extra_parameters": tr["extra_parameters"],
                "total_elapsed_seconds": tr["total_elapsed_seconds"],
            }
            if v != "baseline":
                d["anchor_lr_scale"] = tr["anchor_lr_scale"]
                d["anchor_norm"] = tr["self_diagnostics"]["anchor_norm"]
            row["variants"][v] = d
        points.append(row)

    if len(hashes) != 1:
        raise SystemExit(f"base initialization mismatch: {hashes}")

    final = points[-1]["variants"]
    report = {
        "experiment": "v0.17 SELF 100M four-stage seven-arm study",
        "phase": args.phase,
        "same_initial_base": True,
        "base_sha256": next(iter(hashes)),
        "milestones": milestones,
        "points": points,
        "ranking_at_phase_end": sorted(
            [
                {
                    "variant": v,
                    "nats_per_char": d["nats_per_char"],
                    "parameters": d["parameters"],
                    "extra_parameters": d["extra_parameters"],
                    "total_elapsed_seconds": d["total_elapsed_seconds"],
                    **({"anchor_norm": d["anchor_norm"]} if "anchor_norm" in d else {}),
                }
                for v, d in final.items()
            ],
            key=lambda x: x["nats_per_char"],
        ),
    }

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
