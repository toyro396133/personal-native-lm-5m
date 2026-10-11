"""V9.1 English-only new-project, new-family, policy-only causal interventions.

All texts/labels are EXPERIMENTER-AUTHORED synthetic research examples.
Personal policy DOES NOT appear in owner-blind neural prompt.
"""
import copy,hashlib
from fixtures_v90 import datasets as old_datasets
from fixtures_v85 import POLICY

FAMILIES={
 "technical_release":{
    "fast":"Ship a narrow, separately tested infrastructure patch today.",
    "deep":"Run a comprehensive infrastructure reliability review before shipping.",
    "neutral":"Both approved approaches meet current safety and quality requirements; there is no blocking release condition.",
    "binding":"A signed reliability gate explicitly requires the comprehensive infrastructure review before any release; a narrow patch alone is insufficient."},
 "catalog_consistency":{
    "fast":"Correct only the confirmed inaccurate catalogue entries.",
    "deep":"Perform an exhaustive catalogue-wide consistency audit.",
    "neutral":"Both work plans are authorized, safe, equally adequate and feasible; current owner priority selects their scope.",
    "binding":"A formally validated consistency failure requires an exhaustive catalogue-wide audit before any entries may be changed."},
 "support_handoff":{
    "fast":"Send a concise focused handoff to the support team.",
    "deep":"Prepare an extensive cross-team handoff with comprehensive verification.",
    "neutral":"Both handoff formats fully satisfy current obligations; no deadline or evidence rule forces either.",
    "binding":"A binding service-recovery instruction requires the extensive cross-team handoff and comprehensive verification."},
 "quality_inspection":{
    "fast":"Inspect the affected component with a small targeted quality check.",
    "deep":"Conduct a thorough end-to-end quality inspection across components.",
    "neutral":"Both quality checks are approved and equally sufficient for this optional case; user preference is the only tie-breaker.",
    "binding":"A documented integrity safeguard demands a thorough end-to-end quality inspection; targeted checks are inadequate."},
}
PROJECTS=(
 ("QX91-operations","Maintain stable logistics software and prioritize authorized engineering work.","rapid"),
 ("QX91-analysis","Build reliable research tools with transparent and verifiable results.","thorough"),
 ("QX91-services","Improve support-system reliability and operational delivery.","rapid"),
 ("QX91-workflows","Coordinate dependable internal workflows without unnecessary project scope.","thorough"),
)
def _id(s):
    return hashlib.sha256(("compass-v91|"+s).encode()).hexdigest()[:17]
def changed(row):
    v=copy.deepcopy(row);policy=v["project"]["policy"]
    new="thorough" if policy["mode"]=="rapid" else "rapid"
    policy["mode"]=new;policy["en"]=POLICY[new]["en"];policy["he"]=POLICY[new]["he"]
    policy["version"]=91
    v["task"]["id"]="ITEM-"+_id("alt|"+row["task"]["id"])
    return v
def pack(row,correct,kind,swap,status,split):
    second=changed(row)
    other=correct^1 if status=="neutral" else correct
    return {"original":row,"counterfactual":second,
       "original_label":{"correct_digit":correct},
       "counterfactual_label":{"correct_digit":other},
       "meta":{"family":kind,"swap":swap,"status":status,"expected_flip":status=="neutral","split":split}}
def build():
    training,dev,mandatory_train,mandatory_dev,_old_cold=old_datasets()
    train=[copy.deepcopy(x) for x in training if x["original"]["lang"]=="en"]
    validation=[copy.deepcopy(x) for x in dev if x["original"]["lang"]=="en"]
    def append_binding(to,dataset,tag):
        for row,correct in dataset:
            if row["lang"]!="en":continue
            new=copy.deepcopy(row)
            to.append(pack(new,correct,tag,0,"binding",tag))
    append_binding(train,mandatory_train,"binding_training")
    append_binding(validation,mandatory_dev,"binding_development")
    assert len(train)>=60 and len(validation)>=12,(len(train),len(validation))
    newcold=[]
    for pid,objective,mode in PROJECTS:
      for fam,text in FAMILIES.items():
       for status in ("neutral","binding"):
        for swap in (0,1):
         options=[text["fast"],text["deep"]]
         if swap:options.reverse()
         current={"project":{
           "id":pid,
           "objective":{"en":objective,"he":""},
           "policy":{"id":"POL-"+pid,"mode":mode,"version":1,
                     "en":POLICY[mode]["en"],"he":POLICY[mode]["he"]}},
           "lang":"en",
           "task":{"id":"ITEM-"+_id(pid+fam+status+str(swap)),
                   "record":text[status],"choices":options,"layer":"supporting",
                   "record_source_id":"SRC-"+_id("source"+pid+fam+status),
                   "authoritative_gate":{"provenance":"fictional-research-case",
                                         "approval_granted":None,
                                         "feature_in_scope":None}}}
         correct_sem=(0 if mode=="rapid" else 1) if status=="neutral" else 1
         newcold.append(pack(current,correct_sem^swap,fam,swap,status,"unseen_v91"))
    assert len(newcold)==64
    return train,validation,newcold

def audit():
    train,dev,cold=build()
    trainproj={p["original"]["project"]["id"] for p in train}
    devproj={p["original"]["project"]["id"] for p in dev}
    coldproj={p["original"]["project"]["id"] for p in cold}
    assert len(coldproj)==4 and not (coldproj & (trainproj|devproj))
    assert len({p["original"]["task"]["id"] for p in cold})==len(cold)
    assert {p["meta"]["family"] for p in cold}==set(FAMILIES)
    assert sum(p["meta"]["status"]=="neutral" for p in cold)==32
    assert sum(p["meta"]["status"]=="binding" for p in cold)==32
    for p in train+dev+cold:
        x=p["original"];y=p["counterfactual"]
        assert x["lang"]==y["lang"]=="en"
        assert x["task"]["record"]==y["task"]["record"]
        assert x["task"]["choices"]==y["task"]["choices"]
        assert x["project"]["objective"]==y["project"]["objective"]
        assert x["project"]["policy"]["mode"]!=y["project"]["policy"]["mode"]
        assert (p["original_label"]["correct_digit"]!=p["counterfactual_label"]["correct_digit"])==p["meta"]["expected_flip"]
        assert all(k not in str(x["task"]) for k in ("correct_digit","label_source","mode"))
    return {"status":"PASS","train_pairs":len(train),"dev_pairs":len(dev),
       "new_cold_pairs":len(cold),"cold_neutral":32,"cold_binding":32,
       "novel_project_ids":len(coldproj),"novel_families":list(FAMILIES),
       "caveat":"Synthetic labels and task texts authored by researcher, not independently reviewed"}
if __name__=="__main__":
 import json;print(json.dumps(audit(),ensure_ascii=False,indent=2))
