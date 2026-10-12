"""Generate a PRIVATE, offline, Hebrew review UI for user feedback events.

Runs on the user's machine; never sends data to servers. Input is the exact
local JSONL fed to freeform_feedback.py. Human feedback is linked to an
SHA256 digest of the whole event and cannot mutate a real project policy.
"""
from __future__ import annotations
import argparse
import html
import json
from pathlib import Path

from freeform_feedback import event_fingerprint, literal_correction, validate

def make_review_page(events: list[dict]) -> str:
    seen=set()
    cards=[]
    for x in events:
        validate(x)
        key=(x["scope"], x["project_id"], x["id"])
        if key in seen:
            raise ValueError("Duplicate event")
        seen.add(key)
        literal=literal_correction(x)
        cards.append({
            "id":x["id"], "scope":x["scope"], "project":x["project_id"],
            "question":x["question"],"proposal":x["proposal"],
            "reply":x["user_reply"],"source":x["source"],
            "source_verified":x["source_verified"],
            "event_hash":event_fingerprint(x),
            "literal":bool(literal)
        })
    payload=json.dumps(cards,ensure_ascii=False,separators=(",",":"))
    payload=payload.replace("<",r"\u003c").replace(">",r"\u003e").replace("&",r"\u0026")
    return """<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><title>COMPASS V11.1 | סקירת משוב</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{font-family:Arial,sans-serif;color-scheme:light dark;line-height:1.55}
body{max-width:960px;margin:auto;padding:20px}
header{border-bottom:1px solid #888;padding-bottom:14px}
article{border:1px solid #aaa;border-radius:12px;padding:16px;margin:16px 0}
article h2{font-size:1.08rem;margin-top:0}
small{opacity:.77}p{white-space:pre-wrap}
textarea,select,input{display:block;box-sizing:border-box;width:100%;padding:9px;margin:6px 0 12px;font:inherit}
textarea{min-height:65px}button{cursor:pointer;padding:10px 16px;font:inherit}
.actionline{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.source{font:12px monospace;overflow-wrap:anywhere}
.hidden{display:none}label{display:block;font-weight:600}
.notice{border-inline-start:4px solid currentColor;padding-inline-start:12px}
</style></head><body>
<header><h1>COMPASS — בירור משמעות של משוב</h1>
<p>דברי המשתמש נשמרים במקורם. אפשר לאשר ידנית שהתשובה מתקנת הצעה קודמת, או להשאיר אותה לא מוכרעת. אין כאן שימוש במודל חיצוני או שליחה לרשת.</p>
<p class="notice"><strong>חשוב:</strong> אישור הסקירה מאפשר רק אימון ניסיוני של רכיב COMPASS המקומי. הוא אינו מאשר או משנה מדיניות בפרויקט. יש לוודא בנפרד שמקור הטקסט וההקשר אכן שייכים למשתמש.</p>
</header>
<section id="events"></section>
<div class="actionline"><button id="export" type="button">ייצא קובץ סקירות JSON</button>
<strong id="status" role="status"></strong></div>
<script>
"use strict";
const EVENTS=__CARDS__;
const root=document.getElementById("events");
const esc=(s)=>String(s??"");
function line(label,value){
  const p=document.createElement("p");
  const b=document.createElement("strong");b.textContent=label+" ";
  const t=document.createElement("span");t.textContent=esc(value);
  p.append(b,t);return p;
}
for(const [index,e] of EVENTS.entries()){
  const article=document.createElement("article"); article.dataset.idx=index;
  const title=document.createElement("h2");title.textContent=e.id+" — "+e.scope+" / "+(e.project||"אישי");
  article.append(title,line("שאלה:",e.question),line("הפירוש או ההצעה הקודמת:",e.proposal),
                 line("תשובת המשתמש המקורית:",e.reply));
  const source=document.createElement("small");source.className="source";
  source.textContent=e.source+" · source_verified="+e.source_verified;article.append(source);
  const lab=document.createElement("label");lab.textContent="מעמד לאחר סקירה";
  const select=document.createElement("select");select.className="review-action";
  for(const [val,label] of [
    ["KEEP_UNRESOLVED","עדיין לא ברור / לא לתרגם לדוגמת אימון"],
    ["CONFIRM_CORRECTION","אני מאשר/ת שזאת תיקון מפורש ומשמעותו ברורה"]
  ]){
    const opt=document.createElement("option");opt.value=val;opt.textContent=label;select.append(opt);
  }
  lab.append(select);article.append(lab);
  const details=document.createElement("div");details.className="hidden";
  function field(labelText,className,initial){
    const wrap=document.createElement("label");wrap.textContent=labelText;
    const input=document.createElement("textarea");input.className=className;
    input.value=initial;wrap.append(input);details.append(wrap);
  }
  field("מהו הפירוש הקודם שהמשתמש דחה?","rejected",e.proposal);
  field("מהו התיקון שהתכוון אליו המשתמש?","preferred","");
  field("מדוע ברור שזהו תיקון? נא לציין נימוק","reviewer-note","");
  article.append(details);
  select.addEventListener("change",()=>details.classList.toggle("hidden",select.value!=="CONFIRM_CORRECTION"));
  root.append(article);
}
document.getElementById("export").addEventListener("click",()=>{
  const exported=[];
  for(const article of root.querySelectorAll("article")){
    const e=EVENTS[Number(article.dataset.idx)];
    const action=article.querySelector(".review-action").value;
    const record={event_id:e.id,event_hash:e.event_hash,action,confirmed_by_user:false};
    if(action==="CONFIRM_CORRECTION"){
      if(!e.source_verified){alert("אין אישור בדיקת מקור עבור "+e.id);return;}
      record.rejected=article.querySelector(".rejected").value.trim();
      record.preferred=article.querySelector(".preferred").value.trim();
      record.reviewer_note=article.querySelector(".reviewer-note").value.trim();
      if(!record.rejected||!record.preferred||record.reviewer_note.length<9){
        alert("דרושים פירוש קודם, תיקון ברור ונימוק עבור "+e.id);return;
      }
      record.confirmed_by_user=true;
    }
    exported.push(record);
  }
  const blob=new Blob([JSON.stringify(exported,null,2)],{type:"application/json;charset=utf-8"});
  const url=URL.createObjectURL(blob);
  const a=document.createElement("a");a.href=url;a.download="compass-v11-1-private-human-reviews.json";
  a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  document.getElementById("status").textContent="קובץ הסקירות נוצר מקומית; הוא לא נשלח לשרת.";
});
</script></body></html>""".replace("__CARDS__",payload)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--events",type=Path,required=True,help="Private local input JSONL")
    p.add_argument("--html",type=Path,required=True,help="Private offline output HTML")
    args=p.parse_args()
    events=[json.loads(line) for line in args.events.read_text(encoding="utf8").splitlines()
            if line.strip()]
    args.html.write_text(make_review_page(events),encoding="utf8")
    print(f"Local private HTML built with {len(events)} evidence entries. No data uploaded.")

if __name__=="__main__":
    main()
