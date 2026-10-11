"""Create a consolidated report for three V4 benchmark runs; no external packages."""
import json
import statistics
from pathlib import Path

def main():
    files=sorted(Path("results").glob("v4-seed-*.json"))
    if len(files)!=3: raise RuntimeError(f"Expected 3 independent seed results, found {len(files)}")
    data=[json.loads(p.read_text(encoding="utf-8")) for p in files]
    seeds=[x["seed"] for x in data]
    if len(set(seeds))!=3: raise RuntimeError(f"Seed collision: {seeds}")
    conditions=("base","recent","personal","retrieval","hybrid")
    final={}
    for mode in conditions:
        choice=[r["stages"][-1]["scores"][mode]["all"]["correct"]/24 for r in data]
        free=[r["freeform"][mode]["personal_correct"]/24 for r in data]
        general=[r["freeform"][mode]["general_correct"]/len(r["freeform"][mode]["general_outputs"]) for r in data]
        final[mode]={"choice":choice,"freeform":free,"generic":general,
                     "choice_mean":statistics.mean(choice),
                     "freeform_mean":statistics.mean(free),
                     "generic_mean":statistics.mean(general)}
    retrieval_hit1=[r["stages"][-1]["retrieval"]["hit1"]/24 for r in data]
    retrieval_hitk=[r["stages"][-1]["retrieval"]["hitk"]/24 for r in data]
    summary={"seeds":seeds,"final":final,"retrieval_hit1":retrieval_hit1,"retrieval_hitk":retrieval_hitk}
    Path("v4-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=[
      "# SELF V4 three-seed experiment summary","",
      "Synthetic chat-history experiment; 24 facts, 6 stages, 2 preference changes and distractor utterances.",
      "Pairwise choice and deterministic free-form answer matching are limited proxy metrics.","",
      "## Final stage (24 personal questions and 8 generic questions per seed)","",
      "| Model condition | Pairwise accuracy avg (range) | Open-answer accuracy avg (range) | Generic QA avg (range) |",
      "|---|---:|---:|---:|"
    ]
    def fmt(values):
        return f"{statistics.mean(values)*100:.1f}% ({min(values)*100:.1f}–{max(values)*100:.1f}%)"
    for name in conditions:
        d=final[name]
        lines.append(f"| {name} | {fmt(d['choice'])} | {fmt(d['freeform'])} | {fmt(d['generic'])} |")
    lines.extend(["","## Retrieval correctness","",
                  f"Top-1: {fmt(retrieval_hit1)}; top-k: {fmt(retrieval_hitk)}.","",
                  "## All stage choices","",
                  "| Seed | Stage | Base | Recent-only adapter | Replay adapter | Retrieval | Hybrid |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    for report in data:
        for stage in report["stages"]:
            vals=[f"{stage['scores'][name]['all']['correct']}/{stage['facts_seen']}" for name in conditions]
            lines.append(f"| {report['seed']} | {stage['stage']} | "+" | ".join(vals)+" |")
    lines.extend(["","## Interpretation limits","",
       "- All profiles are fictional; no real personal data were accessed.",
       "- Retrieval searches synthetic user messages and does not receive correct labels; metadata IDs are used for scoring only.",
       "- This is three seeded runs on the same data; it is not three independent user populations.",
       "- Free-form grading uses a bounded literal phrase match, not human/semantic review.",
       "- Two adapters each receive 32 gradient steps at every stage; matched compute but each method sees different facts.",
       "- Final generic QA is limited to 8 questions; absence of regression is not established by a perfect score.",
       "- More natural interactions, longer horizons, real retrievers, and broader non-regression tests are needed."
    ])
    Path("v4-summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:14]))
    print("Seeds:",seeds,"saved v4-summary.md and v4-summary.json")

if __name__=="__main__":
    main()
