"""V8.8: new cold policy-only decision families; no gold labels in teacher prompts."""
import copy, hashlib
from fixtures_v87 import build as build_v87,twin_for
from fixtures_v85 import POLICY
from cache_v85 import selected

NEW_COLD=("deployment_communications","backlog_triage")
DATA={
"deployment_communications":{
 "en":{
 "options":("Send a concise immediate release advisory.","Issue a comprehensive release briefing."),
 "evidence":"Either communication route is authorized and sufficient for this release; there is no timing, safety or regulatory constraint selecting either route."},
 "he":{
 "options":("לשלוח הודעת שחרור מיידית ותמציתית.","לפרסם תדרוך שחרור מקיף ומפורט."),
 "evidence":"שתי דרכי התקשורת מאושרות ומספקות לשחרור; אין אילוץ זמן, בטיחות או דין שמחייב אחת מהן."},
},
"backlog_triage":{
 "en":{
 "options":("Handle one focused low-risk backlog item now.","Conduct a comprehensive review of outstanding backlog items."),
 "evidence":"Both backlog approaches safely satisfy the approved scope and are equally feasible; the current owner's preference is decisive."},
 "he":{
 "options":("לטפל כעת בפריט משימות ממוקד ובסיכון נמוך.","לבצע סקירה מקיפה של פריטי המשימות שנותרו."),
 "evidence":"שתי דרכי הטיפול במשימות בטוחות, מתאימות לתחום המאושר וישימות במידה שווה; העדפת הבעלים הנוכחית מכריעה."},
}}
def id_for(string):
    return hashlib.sha256(("v88|"+string).encode("utf-8")).hexdigest()[:18]
def build():
    inp,gold,tr,old_diag,cold_v87=build_v87()
    pool=inp["test"]
    selected_ids=selected(pool)
    projects={}
    for r in pool:
        if r["project"]["id"] in selected_ids:
            projects[(r["project"]["id"],r["lang"])]=r["project"]
    assert len(projects)==4
    cold=[]
    for (pid,lang),project in sorted(projects.items()):
        for kind in NEW_COLD:
            data=DATA[kind][lang]
            for swap in (0,1):
                choices=list(data["options"])
                if swap:choices.reverse()
                seed=f"{pid}|{lang}|{kind}|{swap}"
                row={
                  "project":copy.deepcopy(project),"lang":lang,
                  "task":{
                    "id":"ITEM-"+id_for(seed),"project_id":pid,"layer":"optional",
                    "proposal_id":"P-"+id_for(pid+kind),
                    "record":data["evidence"],
                    "record_source_id":"SRC-"+id_for(pid+lang+kind),
                    "choices":choices,
                    "authoritative_gate":{
                       "provenance":"fictional-owner-record","approval_granted":None,
                       "feature_in_scope":None,
                       "choice_intents":["hold","execute"] if swap else ["execute","hold"],
                       "governed_action":None,
                    },
                  }
                }
                semantic=0 if project["policy"]["mode"]=="rapid" else 1
                gold_ref={
                  "id":row["task"]["id"],"project_id":pid,"language":lang,
                  "kind":kind,"state":0,"swap":swap,
                  "correct_digit":semantic^swap,
                  "correct_semantic_option":semantic,
                  "label_source":"research-authored-v88-new-family",
                }
                cold.append(twin_for(row,gold_ref,"v88"))
    assert len(cold)==16
    return inp,gold,tr,old_diag,cold_v87,cold
