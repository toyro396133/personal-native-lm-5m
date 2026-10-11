"""Generate OFFLINE human-label review page from a PRIVATE local project Git clone.

This program writes private source excerpts ONLY to an explicitly named LOCAL
HTML file. It never accesses a network, approves COMPASS principles or sends
labels anywhere. Reviewer manually exports annotations as JSON.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
import random
from collections import defaultdict
from pathlib import Path
# Resolve the read-only V9.14 miner when invoked directly on Windows.
import sys as _sys
_sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"compass_v9_14"))
from decision_miner import inventory

CLASSES=["APPROVED_CURRENT","PROPOSED_UNAPPROVED","HISTORICAL_ONLY",
         "REVOKED_OR_SUPERSEDED","UNCERTAIN","NOT_RELEVANT"]
PAGE=r"""<!DOCTYPE html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>COMPASS — בדיקה אנושית למקורות פרויקט</title>
<style>
:root{color-scheme:dark;font:16px/1.6 system-ui,Arial,sans-serif}
*{box-sizing:border-box}body{background:#0e1929;color:#ebf4ff;margin:0}
header{padding:20px 5%;background:#152336;border-bottom:1px solid #344961}
main{max-width:950px;margin:auto;padding:24px}
h1{font-size:23px;margin:0}h2{font-size:20px;margin:0 0 8px}
p{margin:10px 0}.muted{color:#afc2d8}.box{background:#182a3e;padding:18px;margin:16px 0;border:1px solid #395169;border-radius:12px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:50vh;overflow:auto;background:#0d1724;border-radius:8px;padding:16px;font-size:14px}
button,input,select,textarea{font:inherit;border:1px solid #536d86;border-radius:8px;padding:9px}
button{background:#237fa6;color:#fff;cursor:pointer}button:disabled{opacity:.45}
select,input,textarea{background:#0d1724;color:#fff;width:100%}textarea{min-height:70px}
label{display:block;margin:12px 0;font-size:15px}
.actions{display:flex;flex-wrap:wrap;gap:10px}
.note{border-inline-start:3px solid #68c0cb;padding:10px;background:#1b394f}
</style></head><body>
<header><h1>COMPASS — סקירה אנושית של החלטות פרויקט</h1>
<p class="muted">קובץ מקומי, ללא רשת • אף תיוג כאן אינו מאשר שינוי בזיכרון • המידע הפרטי נשאר במחשב</p></header>
<main>
<p class="note">סווג לפי משמעות המקור: החלטה מאושרת ותקפה, הצעה בלבד, עדות על העבר, ביטול/תיקון, או חוסר ודאות. הימנע מלנחש אישור רק כי מופיעה המילה ״אושר״.</p>
<section class="box"><label>כינוי הבודק (ללא פרטים אישיים מזהים)<input id="reviewer" placeholder="reviewer-1"></label>
<div class="actions"><button id="previous">הקודם</button><button id="next">הבא</button><button id="export">ייצוא תיוגים לקובץ JSON</button></div>
<p id="progress" class="muted"></p><h2 id="heading"></h2><p id="source" class="muted"></p><pre id="evidence"></pre>
<label>סיווג עצמאי<select id="status"><option value="">טרם נבדק</option></select></label>
<label>על סמך איזו ראיה קבעת זאת? (חובה לכל סיווג)<textarea id="reason"></textarea></label>
<label>האם המקור דורש בירור סמכות או תחולה?<select id="confidence"><option value="UNKNOWN">טרם הוכרע</option><option value="SUPPORTED">הראיות מספיקות לקביעת מעמד המסמך</option><option value="NEEDS_REVIEW">דרוש בירור נוסף</option></select></label>
</section>
<p class="muted">הייצוא כולל קטעי מקור פרטיים כדי לאפשר ביקורת. שמור את הקובץ באופן מקומי בלבד ואל תעלה אותו למאגר ציבורי או לאתר חיצוני.</p>
</main>
<script>
"use strict";
const dataset=__DATA__;
const choices=__CHOICES__;
let current=0;
const answers={};
const $=id=>document.getElementById(id);
for(const [key,value] of choices){
 const option=document.createElement("option");
 option.value=key;option.textContent=value;$("status").appendChild(option);
}
function save(){
 const rec=dataset[current];if(!rec)return;
 const status=$("status").value,reason=$("reason").value.trim();
 if(status&&!reason)return false;
 answers[rec.id]={status:status||null,reason,reviewer:$("reviewer").value.trim(),
    confidence:$("confidence").value};
 return true;
}
function render(){
 const rec=dataset[current];
 $("progress").textContent=(current+1)+" מתוך "+dataset.length+
   " • תיוגים מלאים: "+Object.values(answers).filter(x=>x.status&&x.reason).length;
 $("heading").textContent=rec.heading;
 $("source").textContent=rec.source;
 $("evidence").textContent=rec.excerpt||"(אין תוכן מתחת לכותרת)";
 $("status").value=answers[rec.id]?.status||"";
 $("reason").value=answers[rec.id]?.reason||"";
 $("confidence").value=answers[rec.id]?.confidence||"UNKNOWN";
 $("previous").disabled=current===0;
 $("next").disabled=current===dataset.length-1;
}
function move(delta){
 if(!save()){alert("יש לכתוב נימוק לפני המשך לאחר בחירת סיווג.");return;}
 current=Math.max(0,Math.min(dataset.length-1,current+delta));render();
}
$("next").onclick=()=>move(1);$("previous").onclick=()=>move(-1);
$("export").onclick=()=>{
 if(!save()){alert("יש להזין נימוק לסיווג הנוכחי.");return;}
 const items=dataset.map(x=>({id:x.id,source:x.source,source_sha256:x.sha,
     label:answers[x.id]?.status||null,
     rationale:answers[x.id]?.reason||"",
     reviewer:answers[x.id]?.reviewer||"",
     confidence:answers[x.id]?.confidence||"UNKNOWN"}));
 const complete=items.filter(x=>x.label&&x.rationale).length;
 if(!complete){alert("לא מולאו עדיין תיוגים.");return;}
 const payload={kind:"COMPASS_INDEPENDENT_HUMAN_REVIEW_V915",
    created_at:new Date().toISOString(),
    warning:"private source references; keep local",
    complete,total:items.length,annotations:items};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:"application/json"});
 const href=URL.createObjectURL(blob);
 const a=document.createElement("a");a.href=href;
 a.download="compass-v915-human-review-private.json";
 a.click();URL.revokeObjectURL(href);
};
render();
</script></body></html>"""


def prepare(rows:list[dict],seed:int=15,limit:int=60)->list[dict]:
    by=defaultdict(list)
    for row in rows:
        if len(row["excerpt"])<70:
            continue
        # Sampling-only strata are intentionally hidden from reviewers.
        by[row["claim_status"]].append(row)
    rng=random.Random(seed)
    for xs in by.values():rng.shuffle(xs)
    quotas={
        "APPROVED_CLAIM":18,"PROPOSED_CLAIM":18,
        "HISTORICAL_EVIDENCE":12,"REVOKED_OR_SUPERSEDED_CLAIM":8,
        "UNRESOLVED":24
    }
    chosen=[]
    for category,quota in quotas.items():
        chosen.extend(by[category][:quota])
    if len(chosen)<limit:
        remainder=[row for group in by.values() for row in group if row not in chosen]
        rng.shuffle(remainder);chosen.extend(remainder[:limit-len(chosen)])
    chosen=chosen[:limit]
    rng.shuffle(chosen)
    return [{"id":x["candidate_id"],"source":x["source"],"heading":x["heading"],
             "excerpt":x["excerpt"],"sha":x["source_sha256"]} for x in chosen]


def render(rows:list[dict])->str:
    values=[
      ("APPROVED_CURRENT","החלטה מאושרת ותקפה"),
      ("PROPOSED_UNAPPROVED","הצעה / תכנון שטרם אושר"),
      ("HISTORICAL_ONLY","תיאור עבר / מצב קיים"),
      ("REVOKED_OR_SUPERSEDED","בוטל או הוחלף"),
      ("UNCERTAIN","לא מספיק ברור — בירור"),
      ("NOT_RELEVANT","לא רשומת החלטה"),
    ]
    # Encode < as unicode escapes to stop untrusted sources closing script.
    data=json.dumps(rows,ensure_ascii=False).replace("<",r"\u003c").replace("&",r"\u0026")
    choices=json.dumps(values,ensure_ascii=False)
    return PAGE.replace("__DATA__",data).replace("__CHOICES__",choices)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--repo",type=Path,required=True)
    p.add_argument("--ref",default="fe9863278a04dedf27a0c4fcb8cca56b91ec4544")
    p.add_argument("--project",default="chacham-lev")
    p.add_argument("--path",action="append",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limit",type=int,default=60)
    args=p.parse_args()
    if not 1<=args.limit<=200:raise SystemExit("limit must be 1..200")
    docs=inventory(args.repo,args.ref,args.path,args.project)
    selected=prepare(docs,limit=args.limit)
    if not selected:raise SystemExit("No suitable source sections")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(render(selected),encoding="utf8")
    print(f"Prepared {len(selected)} PRIVATE source-based human review cases at {args.output}.")
    print("No labels, no approval, no network upload. Keep the HTML and exported JSON private.")


if __name__=="__main__":
    main()
