"""Aggregate independent SELF V5 fictional profiles and report strict reproducible metrics."""
import json
import statistics
from pathlib import Path

def rate(correct,total):
    return correct/total if total else None

def main():
    files=sorted(Path("results").glob("v5-profile-*.json"))
    if len(files)!=3:
        raise RuntimeError("Missing one or more profile results")
    runs=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    profiles=[r["profile"] for r in runs]
    if sorted(profiles)!=[0,1,2]:
        raise RuntimeError("Expected exactly profiles 0,1,2")
    modes=("frozen","adapter_only","retrieval","always_hybrid","gated_hybrid","direct_memory")
    summary={"profiles":profiles,"modes":{},"routing":{},"stage_results":[]}
    for mode in modes:
        choice=[]
        personal=[]
        revisions=[]
        general=[]
        for run in runs:
            stage=run["stages"][-1]
            end=run["final_open_answers"][mode]
            choice.append(rate(stage["metrics"]["scores"][mode]["correct"],12))
            personal.append(rate(end["personal_correct"],end["personal_total"]))
            revisions.append(rate(end["revision_correct"],end["revision_total"]))
            general.append(rate(end["generic_correct"],end["generic_total"]))
        summary["modes"][mode]={"binary_by_profile":choice,
                                "open_by_profile":personal,
                                "revision_by_profile":revisions,
                                "generic_by_profile":general,
                                "binary_mean":statistics.mean(choice),
                                "open_mean":statistics.mean(personal),
                                "revision_mean":statistics.mean(revisions),
                                "generic_mean":statistics.mean(general)}
    for run in runs:
        for stage in run["stages"]:
            summary["stage_results"].append({
              "profile":run["profile"],"stage":stage["stage"],"facts":stage["facts"],
              "retrieval_current_hit1":stage["metrics"]["retrieval_current_hit1"],
              "binary":{k:stage["metrics"]["scores"][k]["correct"] for k in modes}
            })
    summary["routing"]={str(r["profile"]):r["routing"] for r in runs}
    Path("v5-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# SELF V5 — three-profile summary","","Each profile has distinct facts and two late value corrections.",
           "Scores are a limited automatic proxy, not independent population estimates.","",
           "| Condition | Binary final | Open personal | Open corrected facts | General knowledge |",
           "|---|---:|---:|---:|---:|"]
    def fmt(values):
        return f"{100*statistics.mean(values):.1f}% ({100*min(values):.1f}–{100*max(values):.1f}%)"
    for mode in modes:
        d=summary["modes"][mode]
        lines.append(f"| {mode} | {fmt(d['binary_by_profile'])} | {fmt(d['open_by_profile'])} | {fmt(d['revision_by_profile'])} | {fmt(d['generic_by_profile'])} |")
    lines.extend(["","## Router controls","",
                  "| Profile | Personal queries routed | General queries bypassing adapter |",
                  "|---|---|---|"])
    for run in runs:
        s=run["routing"]
        lines.append(f"| {run['profile']} | {s['personal_enabled']}/{s['personal_total']} | {s['generic_disabled']}/{s['generic_total']} |")
    lines.extend(["","## All stages","",
                  "| Profile | Stage | Facts | Current hit@1 | Frozen | Adapter | Retrieval | Always hybrid | Gated hybrid | Direct memory |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for stage in summary["stage_results"]:
        b=stage["binary"]
        lines.append(f"| {stage['profile']} | {stage['stage']} | {stage['facts']} | {stage['retrieval_current_hit1']}/{stage['facts']} | "+
                     " | ".join(f"{b[mode]}/{stage['facts']}" for mode in modes)+" |")
    lines.extend(["","## Limitations and interpretation","",
     "- Extractor is a constrained-grammar parser, NOT a general extractor from unconstrained real chat.",
     "- Slots/values come from synthetic authored messages; retrieval uses question and slot labels, never answer keys.",
     "- Router is transparent hand-authored personal-cue + slot-overlap heuristic, NOT a trained classifier.",
     "- Three different profiles receive different seeds; profile and seed effects are confounded.",
     "- Direct memory can answer slot queries without language generation; use it as the honest factual baseline.",
     "- Automatic grading uses normalized numbers plus simple stale/negation/repetition penalties.",
     "- A routed generic query is forced through the frozen base, so its generic score is a routing property, not proof that the adapter preserves general ability.",
     "- Original histories are fictional; no user conversation was read or trained on.",
     "- Always compare raw outputs and stale-version failures before interpreting aggregate rates.",
    ])
    Path("v5-summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:17]),flush=True)

if __name__=="__main__":
    main()
