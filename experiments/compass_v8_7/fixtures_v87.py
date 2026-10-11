"""COMPASS V8.7 policy-pair training data and COLD policy-change families.

All labels/text synthetic and research-authored; no independent gold review.
Never train on the V8.6 exposed heldout policy pairs.
"""
import copy,hashlib
from fixtures_v86 import build as base_build
from fixtures_v85 import POLICY,NEW_KINDS
from cache_v85 import selected

COLD_POLICY=("maintenance_style","report_delivery")
OPTIONS={
"maintenance_style":{
 "en":("Ship a small focused maintenance patch.","Organize a comprehensive maintenance review."),
 "he":("למסור תיקון תחזוקה קטן וממוקד.","לארגן סקירת תחזוקה מקיפה.")},
"report_delivery":{
 "en":("Send a short preliminary status note.","Publish a detailed full status review."),
 "he":("לשלוח עדכון מצב ראשוני וקצר.","לפרסם סקירת מצב מלאה ומפורטת.")},
}
NEUTRAL={
"maintenance_style":{
 "en":"Both maintenance options are authorized, safe, achievable and equally adequate; the owner preference is the only tie breaker.",
 "he":"שתי אפשרויות התחזוקה מורשות, בטוחות, ישימות ומספקות במידה שווה; העדפת הבעלים היא שובר השוויון היחיד."},
"report_delivery":{
 "en":"Both communication formats satisfy the approved scope and current evidence; no external deadline or coverage rule forces either choice.",
 "he":"שני פורמטי הדיווח עומדים בתחום המאושר ובראיות הנוכחיות; אין מועד חיצוני או דרישת כיסוי שמחייבים אחת מהאפשרויות."},
}
def opaque(t):
    return hashlib.sha256(("V87::"+t).encode("utf8")).hexdigest()[:18]

def twin_for(row,gold,tag):
    """Change authorized policy only in model prompt; hidden IDs cannot leak labels."""
    assert gold["state"]==0
    other="thorough" if row["project"]["policy"]["mode"]=="rapid" else "rapid"
    alternative=copy.deepcopy(row)
    alternative["project"]["policy"]["mode"]=other
    alternative["project"]["policy"]["en"]=POLICY[other]["en"]
    alternative["project"]["policy"]["he"]=POLICY[other]["he"]
    alternative["project"]["policy"]["version"]=4
    alternative["task"]["id"]="ITEM-"+opaque("id"+tag+row["task"]["id"])
    alternative["task"]["record_source_id"]="SRC-"+opaque("src"+tag+row["task"]["record_source_id"])
    new_gold=dict(gold)
    new_gold["id"]=alternative["task"]["id"]
    new_gold["correct_digit"]=gold["correct_digit"]^1
    new_gold["correct_semantic_option"]=gold["correct_semantic_option"]^1
    new_gold["label_source"]="experimenter-authored-policy-twin"
    return {"original":row,"original_label":gold,
            "counterfactual":alternative,"counterfactual_label":new_gold}

def build():
    inputs,golds,old_test_twins=base_build()
    train_twins=[]
    for row,gold in zip(inputs["train"],golds["train"]):
        if gold["kind"] in NEW_KINDS and gold["state"]==0:
            train_twins.append(twin_for(row,gold,"train"))
    # The existing V8.6 test twins were already inspected. Report them as
    # EXPOSED DIAGNOSTIC only; no future blind-evaluation claim.
    cold=[]
    testids=selected(inputs["test"])
    projects={}
    for row in inputs["test"]:
        if row["project"]["id"] in testids:
            projects[(row["project"]["id"],row["lang"])]=row["project"]
    assert len(projects)==4
    for (pid,lang),p in sorted(projects.items()):
        for kind in COLD_POLICY:
            for swap in (0,1):
                actions=list(OPTIONS[kind][lang])
                if swap: actions.reverse()
                token=opaque(f"cold|{pid}|{lang}|{kind}|{swap}")
                t={
                    "id":"ITEM-"+token,"project_id":pid,"layer":"optional",
                    "proposal_id":"PROP-"+opaque(f"p|{pid}|{kind}"),
                    "record":NEUTRAL[kind][lang],
                    "record_source_id":"SRC-"+opaque(f"ref|{pid}|{lang}|{kind}"),
                    "record_source_description":("Current verified option-level summary" if lang=="en"
                                                  else "סיכום עדכני ומאומת של החלופות"),
                    "choices":actions,
                    "authoritative_gate":{"provenance":"fictional-owner-record",
                                          "approval_granted":None,"feature_in_scope":None,
                                          "choice_intents":["hold","execute"] if swap else ["execute","hold"],
                                          "governed_action":None},
                }
                a={"project":copy.deepcopy(p),"lang":lang,"task":t}
                semantic=0 if p["policy"]["mode"]=="rapid" else 1
                correct=semantic^swap
                g={"id":t["id"],"project_id":pid,"language":lang,
                   "kind":kind,"state":0,"swap":swap,
                   "correct_digit":correct,"correct_semantic_option":semantic,
                   "label_source":"research-authored-new-cold-policy-family"}
                cold.append(twin_for(a,g,"cold"))
    return inputs,golds,train_twins,old_test_twins,cold
