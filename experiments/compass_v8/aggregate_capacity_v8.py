"""Aggregate preregistered V8 rank-by-seed results without hiding action collapse."""
import json
from pathlib import Path
from statistics import mean

RANKS=(4,16,64)
SEEDS=(11,19,37)
MODES=("base_graph","base_policy","compass_graph","compass_policy")

def summarize():
    files=sorted(Path("results").glob("v8-rank*-seed*.json"))
    records=[json.loads(f.read_text(encoding="utf-8")) for f in files]
    pairs={(r["rank"],r["seed"]) for r in records}
    expected={(rank,seed) for rank in RANKS for seed in SEEDS}
    if len(records)!=9 or pairs!=expected:
        raise ValueError(f"Missing or duplicated trials: found {sorted(pairs)}")
    for r in records:
        assert r["training_steps"]==60 and sum(r["balanced_training_action_counts"].values())==60
        assert all(r["model_comparisons"][mode]["total"]==32 for mode in MODES)
        assert all(r["model_comparisons"][mode]["paired_ancillary_core_contrast_total"]==4 for mode in MODES)
        assert r["test_gold_histogram"]==records[0]["test_gold_histogram"]
    out={"n_trials":len(records),"n_distinct_test_projects":4,"total_test_items_per_trial":32,
         "same_held_out_items_reused":True,"by_rank":{}}
    lines=[
        "# COMPASS V8 — controlled capacity diagnostic results","",
        "**Nine training jobs**: rank 4/16/64 × seeds 11/19/37.",
        "All decisions come from **32 recurring synthetic authored holdout cases**, not 288 independent examples.",
        "This is forced-choice action-code scoring, NOT free-form verified hierarchy reasoning or independently adjudicated policy quality.","",
        "## Results by rank","",
        "| Rank | Weights | Mean adapter + graph action /32 | Balanced accuracy | Action diversity | Paired ancillary/core traps /4 | Critical scope/approval violations /8 | Frozen policy baseline /32 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for rank in RANKS:
        rs=[r for r in records if r["rank"]==rank]
        mode="compass_graph"
        mean_action=mean(r["model_comparisons"][mode]["correct"] for r in rs)
        mean_bal=mean(r["model_comparisons"][mode]["balanced_accuracy"] for r in rs)
        mean_div=mean(r["model_comparisons"][mode]["predicted_unique"] for r in rs)
        mean_traps=mean(r["model_comparisons"][mode]["paired_ancillary_core_contrast_correct"] for r in rs)
        mean_bad=mean(r["model_comparisons"][mode]["hard_boundary_violations"] for r in rs)
        mean_policy=mean(r["model_comparisons"]["base_policy"]["correct"] for r in rs)
        out["by_rank"][str(rank)]={
            "weights":2*960*rank,"mean_action_correct":mean_action,
            "mean_balanced_accuracy":mean_bal,"mean_predicted_actions":mean_div,
            "mean_counterfactual_pairs":mean_traps,
            "mean_hard_boundary_violations":mean_bad,
            "mean_frozen_policy_correct":mean_policy,
        }
        lines.append(f"| {rank} | {2*960*rank:,} | {mean_action:.1f}/32 | {mean_bal:.3f} | "+
                     f"{mean_div:.1f} | {mean_traps:.1f}/4 | {mean_bad:.1f}/8 | {mean_policy:.1f}/32 |")
    lines.extend(["","## Each separate seed","",
                  "| Rank | Seed | Adapter graph | Adapter + policy | Frozen graph | Frozen + policy | Majority control | Balanced adapter | Distinct adapter actions |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in sorted(records,key=lambda x:(x["rank"],x["seed"])):
        c=r["model_comparisons"]
        lines.append(f"| {r['rank']} | {r['seed']} | {c['compass_graph']['correct']}/32 | "+
                     f"{c['compass_policy']['correct']}/32 | {c['base_graph']['correct']}/32 | "+
                     f"{c['base_policy']['correct']}/32 | {r['majority_action_control']['correct']}/32 | "+
                     f"{c['compass_graph']['balanced_accuracy']:.3f} | "+
                     f"{c['compass_graph']['predicted_unique']} |")
    lines.extend(["","## Method limits and decision gate","",
                  "- Do **not** use aggregate action accuracy alone; a constant answer can appear successful.",
                  "- Evaluate action diversity, balanced accuracy, per-class recalls, critical forbidden-scope errors, and correctly flipped ancillary/core pairs.",
                  "- Candidate scoring uses exact shared model prompt per action code. Mean and total action-token log-probability results are recorded separately to diagnose token-length bias.",
                  "- The test set is a highly templated project-type holdout. Shared profile policies and answer template families may still bias results.",
                  "- Frozen graph+explicit-policy and deterministic authored-rule controls must be compared against trained adapters before arguing that capacity helps.",
                  "- More weights are not helpful if predictions collapse to one label or if hard-boundary violations increase.",
                  "- Results show capacity sensitivity on this fixture ONLY, not generalizable user-specific strategic judgment.",
                  "- Raw option log probabilities and all mistakes are preserved in each per-run artifact for audit.",
                 ])
    Path("v8-capacity-summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    Path("v8-capacity-summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines),flush=True)

if __name__=="__main__":
    summarize()
