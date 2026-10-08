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
        if x.get("experiment") != "SELF CORE structural checkpoint audit v3":
            continue

        f = x["final"]
        learned = f.get("learned_reference_specificity")
        rows.append({
            "variant": x["variant"],
            "tokens_seen": x["tokens_seen"],
            "final_core_paraphrase_accuracy": f["core_paraphrase_accuracy"],
            "final_core_under_goal_accuracy": f["core_under_goal_accuracy"],
            "final_core_under_focus_accuracy": f["core_under_focus_accuracy"],
            "final_focus_minus_goal_core_accuracy": f["focus_minus_goal_core_accuracy"],
            "final_core_cluster_margin": f["core_cluster_margin"],
            "final_goal_relation_margin": f["goal_relation_margin"],
            "final_focus_relation_margin": f["focus_relation_margin"],
            "learned_reference_core_advantage": (
                None if learned is None else
                learned["advantages"]["core_margin"]["learned_minus_sham_mean"]
            ),
            "learned_reference_goal_advantage": (
                None if learned is None else
                learned["advantages"]["goal_relation_margin"]["learned_minus_sham_mean"]
            ),
            "learned_reference_focus_advantage": (
                None if learned is None else
                learned["advantages"]["focus_relation_margin"]["learned_minus_sham_mean"]
            ),
            "interventions_final": f["interventions"],
            "source_file": str(p),
        })

    expected = 7
    if len(rows) != expected:
        raise SystemExit(f"expected {expected} structural reports, found {len(rows)}")

    # No single scalar 'winner': preserve the factorial view from the blind follow-up.
    rankings = {
        "core_under_focus": sorted(rows, key=lambda r: r["final_core_under_focus_accuracy"], reverse=True),
        "goal_relation": sorted(rows, key=lambda r: r["final_goal_relation_margin"], reverse=True),
        "focus_relation": sorted(rows, key=lambda r: r["final_focus_relation_margin"], reverse=True),
        "learned_core_reference": sorted(
            [r for r in rows if r["learned_reference_core_advantage"] is not None],
            key=lambda r: r["learned_reference_core_advantage"],
            reverse=True,
        ),
        "learned_goal_reference": sorted(
            [r for r in rows if r["learned_reference_goal_advantage"] is not None],
            key=lambda r: r["learned_reference_goal_advantage"],
            reverse=True,
        ),
    }

    out = {
        "schema_version": 1,
        "experiment": "SELF CORE structural checkpoint audit v3 — seven-arm comparison",
        "primary_note": "No language-loss ranking is used here. The comparison is deliberately multi-axis.",
        "guardrails": [
            "Magnitude-matched intervention direction is interpreted separately from displacement size.",
            "SELF effect strength is not equated with learned-reference specificity.",
            "CORE/GOAL/FOCUS geometry is not called causal without direct component interventions.",
            "No single composite winner is constructed.",
        ],
        "rows": rows,
        "rankings": rankings,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "reports": len(rows),
        "variants": sorted(r["variant"] for r in rows),
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
