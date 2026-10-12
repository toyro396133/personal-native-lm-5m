"""Exploratory post-hoc V6 surface-format re-score (NOT preregistered).
Read the six original per-run JSONs from the unmodified GitHub Actions artifacts.
"""
import json
import re
from pathlib import Path

def alternate_format(text, style):
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if style=="two_bullets":
        # Allow either two bullet marks or two consecutively numbered points.
        return len(lines)==2 and all(bool(re.match(r"^(?:[-*]|\d+[.)])\s+\S",s)) for s in lines)
    if style=="three_steps":
        return len(lines)==3 and all(bool(re.match(r"^"+str(i+1)+r"[.)]\s+\S",s))
                                      for i,s in enumerate(lines))
    if style=="one_sentence":
        return (len(lines)==1 and not bool(re.match(r"^[-*0-9]",lines[0])) and
                len(re.findall(r"[.!?](?:\s|$)",text))==1 and
                len(re.findall(r"\b\w+\b",text))<=50)
    raise ValueError(style)

def main():
    files=sorted(Path("results").glob("v6-profile-*-seed-*.json"))
    if len(files)!=6:raise ValueError("Require original JSON artifacts from six V6 runs")
    totals={}
    for mode in ("base","memory_prompt","adapter_only","adapter_plus_prompt","gated_adapter"):
        totals[mode]={"official":0,"alternate":0,"unfinished":0,"total":0,"by_profile":[]}
    for fn in files:
        run=json.loads(fn.read_text(encoding="utf-8"))
        style=run["style"]
        for mode,stat in totals.items():
            records=run["evaluation"]["styles"][mode]["outputs"]
            old=sum(bool(q["format_pass"]) for q in records)
            new=sum(alternate_format(q["output"],style) for q in records)
            unclosed=sum(not bool(re.search(r"[.!?]\s*$",q["output"])) for q in records)
            stat["official"]+=old
            stat["alternate"]+=new
            stat["unfinished"]+=unclosed
            stat["total"]+=len(records)
            stat["by_profile"].append({"profile":run["profile"],"seed":run["seed"],
                                       "style":style,"official":old,"alternate":new})
    assert all(row["total"]==72 for row in totals.values())
    Path("v6-posthoc-audit.json").write_text(json.dumps(totals,indent=2),encoding="utf-8")
    for k,v in totals.items():
        print(k, f"official={v['official']}/{v['total']}",
              f"alternate={v['alternate']}/{v['total']}",
              f"ending_without_punctuation={v['unfinished']}/{v['total']}",flush=True)

if __name__=="__main__":main()
