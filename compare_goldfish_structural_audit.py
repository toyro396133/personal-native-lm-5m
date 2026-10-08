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
        if x.get("experiment") != "Goldfish 124M SELF CORE structural audit v3 at 1M":
            continue

        f = x["final"]
        learned = f.get("learned_reference_specificity")
        radial2 = (f.get("interventions") or {}).get("radial_2x")
        random2 = (f.get("interventions") or {}).get("random_2x")
        orth2 = (f.get("interventions") or {}).get("orthogonal_2x")

        rows.append({
            "variant_code": x["variant_code"],
            "variant_name": x["variant_name"],
            "core_cluster_margin": f["core_cluster_margin"],
            "core_under_goal_accuracy": f["core_under_goal_accuracy"],
            "core_under_focus_accuracy": f["core_under_focus_accuracy"],
            "focus_minus_goal_core_accuracy": f["focus_minus_goal_core_accuracy"],
            "goal_relation_margin": f["goal_relation_margin"],
            "focus_relation_margin": f["focus_relation_margin"],
            "learned_core_reference_advantage": (
                None if learned is None else
                learned["advantages"]["core_margin"]["learned_minus_sham_mean"]
            ),
            "learned_goal_reference_advantage": (
                None if learned is None else
                learned["advantages"]["goal_relation_margin"]["learned_minus_sham_mean"]
            ),
            "learned_focus_reference_advantage": (
                None if learned is None else
                learned["advantages"]["focus_relation_margin"]["learned_minus_sham_mean"]
            ),
            "radial_2x_core_accuracy": None if radial2 is None else radial2["core_accuracy"],
            "radial_2x_effect_norm": None if radial2 is None else radial2["effect_norm"],
            "random_2x_core_accuracy": None if random2 is None else random2["core_accuracy"],
            "random_2x_effect_norm": None if random2 is None else random2["effect_norm"],
            "orthogonal_2x_core_accuracy": None if orth2 is None else orth2["core_accuracy"],
            "orthogonal_2x_effect_norm": None if orth2 is None else orth2["effect_norm"],
            "source_file": str(p),
        })

    if len(rows) != 7:
        raise SystemExit(f"expected 7 reports, found {len(rows)}")

    rows.sort(key=lambda r: r["variant_code"])
    payload = {
        "schema_version": 1,
        "experiment": "Goldfish 124M seven-arm structural transfer audit at 1M",
        "primary_note": (
            "No language-loss leaderboard. This comparison asks whether the "
            "small-model SELF/CORE structural signatures transfer to a pretrained "
            "Hebrew 124M backbone."
        ),
        "guardrails": [
            "Magnitude and perturbation direction are separated.",
            "SELF effect strength is distinct from learned-reference specificity.",
            "No semantic SELF claim follows from geometry alone.",
            "This is single-seed and only 1M continuation tokens.",
        ],
        "rows": rows,
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
