"""V8.9 locked gold-only-after-generation scoring of English teacher and feedback."""
import argparse,json,collections
from pathlib import Path

METHODS={
 "small_head":"small_head",
 "direct_english":"direct_english",
 "reasoned_english":"reasoned_english",
 "bounded_feedback_english":"bounded_feedback_english",
}
def evaluate(records,method):
    rows=[]
    metrics={k:{"cases":0,"both_correct":0,"correct_choices":0,"choices":0,
                "correct_flip_behavior":0,"actual_flips":0,"abstained":0}
             for k in ("neutral","binding","all")}
    predictions={}
    for case in records:
        outputs=[case[k][method] for k in ("before","after")]
        status=case["constraint"]
        for group in (status,"all"):
            d=metrics[group];d["cases"]+=1
            if any(v is None for v in outputs):
                d["abstained"]+=1
                continue
            predicted=[v["choice"] for v in outputs]
            gold=case["gold"]
            d["correct_choices"]+=sum(a==b for a,b in zip(predicted,gold))
            d["choices"]+=2
            d["both_correct"]+=int(predicted==gold)
            d["actual_flips"]+=int(predicted[0]!=predicted[1])
            d["correct_flip_behavior"]+=int((predicted[0]!=predicted[1])==case["expected_flip"])
        if all(v is not None for v in outputs):
            predictions[(case["project"],case["family"],status,case["option_swap"])]=[
              v["choice"] for v in outputs]
    # Order swap should NOT change which semantic action is preferred.
    count=0;good=0
    for (project,family,status,swap),v in predictions.items():
        if swap!=0:continue
        alternative=predictions.get((project,family,status,1))
        if alternative is None:continue
        for x,y in zip(v,alternative):
            count+=1
            good+=int(x==1-y)
    return {"by_status":metrics,"order_invariance":{"correct":good,"total":count}}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",default="compass-v89-english-advisor.json")
    p.add_argument("--json",default="compass-v89-summary.json")
    p.add_argument("--markdown",default="compass-v89-summary.md")
    args=p.parse_args()
    data=json.loads(Path(args.input).read_text(encoding="utf8"))
    records=data["records"];assert len(records)==32
    assert sum(p["constraint"]=="neutral" for p in records)==16
    assert sum(p["constraint"]=="binding" for p in records)==16
    results={name:evaluate(records,name) for name in METHODS}
    reason_valid=sum(p[side]["reason_valid"] for p in records for side in ("before","after"))
    reasons=[p[side]["reason_text"] for p in records for side in ("before","after")]
    neutral=lambda model:results[model]["by_status"]["neutral"]["both_correct"]
    binding=lambda model:results[model]["by_status"]["binding"]["both_correct"]
    control="direct_english"; feedback="bounded_feedback_english"
    fb_order=results[feedback]["order_invariance"]
    gate={
      "direct_neutral_correct_min12":neutral(control)>=12,
      "direct_binding_correct_min12":binding(control)>=12,
      "direct_swap_invariance_min28of32":results[control]["order_invariance"]["correct"]>=28,
      "feedback_neutral_both_improves_at_least2":neutral(feedback)>=neutral(control)+2,
      "feedback_binding_must_not_degrade":binding(feedback)>=binding(control),
      "feedback_swap_invariance_min28of32":fb_order["correct"]>=28,
      "reason_valid_at_least48of64":reason_valid>=48,
    }
    # Registered goal: no size sweep until strong teacher *and* feedback, independently checked.
    state="GO_TO_SIZE_COMPARISON_ONLY_WITH_FRESH_HOLDOUT" if all(gate.values()) else "NO_GO_TO_SIZE_SWEEP"
    summary={
       "run_type":"English-only frozen language model active advisor and bounded personal-head feedback",
       "sample_pairs":32,
       "neutral_flip_required_pairs":16,
       "binding_no_flip_required_pairs":16,
       "evaluations":results,
       "reason_valid_count":reason_valid,
       "reasons_total":len(reasons),
       "unique_reasons":len(set(reasons)),
       "preregistered_gate":gate,
       "decision":state,
       "important_caveat":"All golds/situations synthesized within this research workflow; no independently adjudicated data",
    }
    Path(args.json).write_text(json.dumps(summary,indent=2),encoding="utf8")
    lines=[
       "# COMPASS V8.9 — English-only teacher, no size sweep unless gated",
       f"**Decision: {state}.**",
       "",
       "All examples are synthetic fictional cases and already use two previously exposed project profiles. They are NOT independently validated.",
       "",
       "| Arm | Neutral both/16 (must flip) | Binding both/16 (must not flip) | Order invariant/32 |",
       "|---|---:|---:|---:|",
    ]
    for name,metrics in results.items():
        neu=metrics["by_status"]["neutral"];bind=metrics["by_status"]["binding"]
        inv=metrics["order_invariance"]
        lines.append(f"| {name} | {neu['both_correct']}/16 | {bind['both_correct']}/16 | {inv['correct']}/{inv['total']} |")
    lines.append(f"\nValid generated reasons: {reason_valid}/64; unique strings: {len(set(reasons))}/64.\n")
    lines.append("## Prespecified gate")
    for k,v in gate.items():lines.append(f"- {k}: {'PASS' if v else 'FAIL'}")
    lines.append("\nThe first-pass small-head and active teacher decisions are untrusted. An externally authenticated source and explicit owner approval are REQUIRED before any action. No native SELF/5M or production deployment.")
    Path(args.markdown).write_text("\n".join(lines)+"\n",encoding="utf8")
    print(json.dumps({"results":{k:{"neutral":neutral(k),"binding":binding(k),
                       "swap":results[k]["order_invariance"]} for k in results},
                       "valid_reasons":reason_valid,"decision":state},indent=2),flush=True)
if __name__=="__main__":main()
