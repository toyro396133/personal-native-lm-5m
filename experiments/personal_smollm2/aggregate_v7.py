"""Aggregate six V7 trials and prepare a fixed human-review sample (not yet reviewed)."""
import json
from pathlib import Path

MODES=("frozen","explicit_prompt","untrained_adapter_prompt","trained_adapter_only",
       "trained_adapter_prompt","gated_adapter_prompt")
KEYS=("format_strict","format_permissive","content_anchor_hit","format_and_content",
      "hit_token_limit","repetitions")

def main():
    files=sorted(Path("results").glob("v7-profile-*-seed-*.json"))
    trials=[json.loads(f.read_text(encoding="utf-8")) for f in files]
    expected={(p,s) for p in range(3) for s in (11,19)}
    if len(trials)!=6 or {(x["profile"],x["seed"]) for x in trials}!=expected:
        raise RuntimeError("Missing or duplicated V7 trial")
    for x in trials:
        assert x["memory_controls"]["passes"]
        assert set(x["evaluation"]["long"])==set(MODES)
        assert all(x["evaluation"]["long"][m]["total"]==18 for m in MODES)
    out={"profiles":3,"seeds":[11,19],"long":{},"short":{},"generic":{}}
    lines=["# SELF V7 — six-run study summary","",
           "Three synthetic style profiles and two training seeds each; 18 held-out advice tasks.",
           "These are format and lexical task-keyword proxies, NOT human-reviewed factual or content quality.",
           "Primary output budget 160 generated tokens, fixed before evaluation.","",
           "| Condition | Strict layout | Permissive layout | Task-keyword proxy | Both format + keyword | Hit generation limit |",
           "|---|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        entries=[x["evaluation"]["long"][mode] for x in trials]
        z={k:sum(v[k] for v in entries) for k in KEYS}
        out["long"][mode]={"scores":z,"total":108,
                            "per_trial":{k:[v[k] for v in entries] for k in KEYS}}
        lines.append(f"| {mode} | {z['format_strict']}/108 | {z['format_permissive']}/108 | "+
                     f"{z['content_anchor_hit']}/108 | {z['format_and_content']}/108 | "+
                     f"{z['hit_token_limit']}/108 |")
    lines.extend(["","## Matched short output budget (76 tokens)","",
                  "| Mode | Layout | Content | Both | Hit token budget |",
                  "|---|---:|---:|---:|---:|"])
    for mode in ("explicit_prompt","trained_adapter_prompt"):
        z={k:sum(x["evaluation"]["short"][mode][k] for x in trials) for k in KEYS}
        out["short"][mode]={"scores":z,"total":108}
        lines.append(f"| {mode} | {z['format_permissive']}/108 | {z['content_anchor_hit']}/108 | "+
                     f"{z['format_and_content']}/108 | {z['hit_token_limit']}/108 |")
    lines.extend(["","## Generic trivia probe","",
                  "| Condition | Keyword correct | Adapter invocations |",
                  "|---|---:|---:|"])
    for mode in ("frozen","trained_adapter_prompt","gated_adapter_prompt"):
        d=[x["evaluation"]["generic"][mode] for x in trials]
        a=sum(v["correct"] for v in d)
        b=sum(v["adapter_active"] for v in d)
        out["generic"][mode]={"correct":a,"total":48,"adapter_invocations":b}
        lines.append(f"| {mode} | {a}/48 | {b}/48 |")
    lines.extend(["","## Per-run format-and-content scores","",
                  "| Profile | Seed | Style | Explicit prompt | Adapter + prompt | Gated adapter + prompt |",
                  "|---:|---:|---|---:|---:|---:|"])
    for r in sorted(trials,key=lambda x:(x["profile"],x["seed"])):
        scores=r["evaluation"]["long"]
        cells=[f"{scores[m]['format_and_content']}/18" for m in
                ("explicit_prompt","trained_adapter_prompt","gated_adapter_prompt")]
        lines.append(f"| {r['profile']} | {r['seed']} | {r['style']} | "+" | ".join(cells)+" |")
    lines.extend(["","## Caveats","",
                  "- Untrained adapter is zero-residual initialized and should match the frozen model with an identical explicit prompt.",
                  "- The gated adapter uses the prompt for personal queries and bypasses the adapter on generic queries.",
                  "- Style compliance and topical keywords do not equal quality or factual accuracy.",
                  "- All data and training tasks are English, fictional, synthetic, and narrow.",
                  "- Six runs contain only three independently chosen style targets, not 108 independent users.",
                  "- Longer output tests isolate truncation effects but do not guarantee generalization.",
                  "- Memory deletion removes active state only, not stored audit records or trained adapter information.",
                  "- Human evaluation remains pending; see the locked review sample.",
                  ])
    Path("v7-summary.json").write_text(json.dumps(out,indent=2),encoding="utf-8")
    Path("v7-summary.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    review=["# V7 locked qualitative review sample","",
            "Human assessment pending. For each profile use seed 19 and held-out tasks 1 and 13.",
            "Review correctness, helpfulness, repetitions, and compliance; do not treat these as population estimates.",""]
    for r in sorted(trials,key=lambda x:(x["profile"],x["seed"])):
        if r["seed"]!=19:continue
        for index in (0,12):
            ex=r["evaluation"]["long"]["frozen"]["outputs"][index]
            review.extend([f"## Profile {r['profile']}: {r['style']}, task {index+1}",
                           "**Prompt:** "+ex["question"],
                           "**Lexical evaluation anchors:** "+", ".join(ex["reference_anchors"]),""])
            for mode in MODES:
                v=r["evaluation"]["long"][mode]["outputs"][index]
                review.extend([f"### {mode}","~~~text",v["output"],"~~~",
                    f"strict={v['format_strict']} permissive={v['format_permissive']} "+
                    f"content_proxy={v['content_anchor_hit']} limit_hit={v['limit_hit']}",""])
    Path("v7-review-sample.md").write_text("\n".join(review)+"\n",encoding="utf-8")
    print("\n".join(lines[:16]),flush=True)

if __name__=="__main__":
    main()
