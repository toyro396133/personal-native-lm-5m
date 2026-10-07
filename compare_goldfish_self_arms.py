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
        if x.get("experiment") != "goldfish-124m-self-seven-arm-1m":
            continue
        final_ms = x.get("milestones", [])[-1] if x.get("milestones") else None
        if not final_ms:
            continue
        inter = final_ms.get("interventions") or {}
        modes = inter.get("modes", {}) if inter else {}
        rows.append({
            "variant_code": x["variant_code"],
            "parameters": x["parameter_count"],
            "extra_parameters": x["extra_parameter_count"],
            "initial_val_nats_per_token": x["initial_eval"]["nats_per_token"],
            "final_val_nats_per_token": final_ms["validation"]["nats_per_token"],
            "delta_val_nats_per_token": final_ms["validation"]["nats_per_token"] - x["initial_eval"]["nats_per_token"],
            "train_tokens_per_second": x["final"]["tokens_per_second"],
            "max_rss_mb": x["final"]["max_process_rss_mb"],
            "anchor_norm": (final_ms.get("self_diagnostics") or {}).get("anchor_norm"),
            "zero_delta_nll": (modes.get("zero") or {}).get("delta_nats_per_token"),
            "negate_delta_nll": (modes.get("negate") or {}).get("delta_nats_per_token"),
            "shuffle_delta_nll": (modes.get("shuffle") or {}).get("delta_nats_per_token"),
            "random_same_norm_delta_nll": (modes.get("random_same_norm") or {}).get("delta_nats_per_token"),
            "orthogonal_same_norm_delta_nll": (modes.get("orthogonal_same_norm") or {}).get("delta_nats_per_token"),
            "archive": x.get("archive"),
            "source": str(p),
        })

    rows.sort(key=lambda r: r["variant_code"])
    payload = {
        "schema_version": 1,
        "experiment": "goldfish-124m-self-seven-arm-1m",
        "interpretation_guardrails": [
            "This is a 1M-token transfer/feasibility canary, not a final SELF result.",
            "Language loss is a retention/health metric, not the primary SELF objective.",
            "Intervention mode and intervention magnitude are recorded separately; raw negate effects must not be interpreted as sign-only evidence.",
            "Learned-value specificity and intervention strength are distinct dimensions.",
            "No B/C/D causal claim is licensed without direct component interventions.",
        ],
        "rows": rows,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
