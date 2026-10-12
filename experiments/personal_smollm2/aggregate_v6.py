"""Aggregate SELF V6: keep profiles and seeds separate, report format-only and general."""
import json
from pathlib import Path
from statistics import mean

def main():
    files=sorted(Path("results").glob("v6-profile-*-seed-*.json"))
    if len(files)!=6:
        raise RuntimeError(f"Expected 6 artifacts, found {len(files)}: {files}")
    trials=[json.loads(path.read_text(encoding="utf-8")) for path in files]
    keys={(x["profile"],x["seed"]) for x in trials}
    if keys!={(p,s) for p in range(3) for s in (11,19)}:
        raise RuntimeError("Incomplete or duplicated profile-seed matrix")
    modes=("base","memory_prompt","adapter_only","adapter_plus_prompt","gated_adapter")
    summary={"trials":len(trials),"profiles":3,"seeds":[11,19],"modes":{}}
    lines=["# SELF V6 — style personalization study","",
       "Three distinct fictional user preferences, two seeds for each (6 training runs).",
       "Style scores check surface format only, not factual accuracy or answer usefulness.",
       "General scores use literal answer keywords, and are not a comprehensive capability benchmark.","",
       "| Condition | Format pass (mean; range) | General QA (mean; range) | Repetitive style output | General adapter usage |",
       "|---|---:|---:|---:|---:|"]
    for mode in modes:
        style=[]
        general=[]
        repetitive=0
        enabled=0
        for run in trials:
            a=run["evaluation"]["styles"][mode]
            b=run["evaluation"]["generic"][mode]
            style.append(a["correct"]/a["total"])
            general.append(b["correct"]/b["total"])
            repetitive+=a["repetitive"]
            enabled+=b["adapter_enabled"]
        summary["modes"][mode]={"format_rates":style,"general_rates":general,
                                "format_mean":mean(style),"general_mean":mean(general),
                                "style_repetitions":repetitive,"generic_adapter_uses":enabled}
        def fmt(v):
            return f"{100*mean(v):.1f}% ({100*min(v):.1f}–{100*max(v):.1f}%)"
        lines.append(f"| {mode} | {fmt(style)} | {fmt(general)} | {repetitive} | {enabled} |")
    lines.extend(["","## Per-profile and seed counts","",
       "| Profile | Seed | Style | Base | Explicit preference | Adapter only | Adapter + preference | Gated adapter | Frozen general | Adapted general | Gated general |",
       "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for r in sorted(trials,key=lambda t:(t["profile"],t["seed"])):
        s=r["evaluation"]["styles"]
        g=r["evaluation"]["generic"]
        lines.append(
            f"| {r['profile']} | {r['seed']} | {r['style']} | "+
            " | ".join(f"{s[m]['correct']}/{s[m]['total']}" for m in modes) +
            f" | {g['base']['correct']}/{g['base']['total']} | "+
            f"{g['adapter_only']['correct']}/{g['adapter_only']['total']} | "+
            f"{g['gated_adapter']['correct']}/{g['gated_adapter']['total']} |"
        )
    lines.extend(["","## Deterministic memory control","",
       "Each run exercises user-authored style declaration, updates, quoted/third-party text, hypotheticals, deletion, a two-reply temporary preference, expiry, and conservative gate bypass.",
       "The implementation uses limited explicit pattern matching; it is NOT a general natural-conversation understanding or privacy deletion service.",
       "All six synthetic memory-control suites must have passed for the six training results to exist.","",
       "## Interpretation and decision guards","",
       "- Frozen base and explicit-prompt baseline are necessary; an adapter must show improvement over **explicit preference prompting** on held-out queries, not merely improvement over an unpersonalized base.",
       "- Memory delete is only removal from active preference; historical audit metadata remains (research provenance). A real erasure implementation must handle stored logs, cache, adapters and backups.",
       "- The gate uses transparent heuristic recognition of practical/advice requests; general questions bypass the adapter by design.",
       "- The training examples and held-out prompts are synthetic, narrow and English-only. The format scorer is not a semantic or human judgement.",
       "- Even if the adapter scores well on format, that does not establish learned identity, independent reasoning, or full real-user personalization.",
       "- Every model output is preserved in per-run JSON for targeted manual review.",
    ])
    Path("v6-summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    Path("v6-summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines[:15]),flush=True)

if __name__=="__main__":
    main()
