"""V9.0 train/dev & NEW test synthetic English personal-state interventions.

Neither gold labels, policy text, task IDs nor user IDs enter neural input text.
All scenarios/answers invented by researcher; no human blind ground truth.
"""
import copy,hashlib
from fixtures_v87 import build as v87_build
from fixtures_v89 import build as v89_build
from fixtures_v85 import POLICY,NEW_KINDS

NEW_FAMILIES={
 "change_review":(
   "Publish a focused low-risk change review.",
   "Publish a complete multi-team change assessment.",
   "Both forms of review are equally adequate, allowed and feasible; there is no binding review breadth requirement.",
   "A binding oversight directive requires a complete multi-team change assessment; a focused review is insufficient."
 ),
 "data_backfill":(
   "Fix only the affected recent data rows.",
   "Perform comprehensive historical data reprocessing.",
   "Both data repairs are verified, equally safe and authorized; owner preference decides scope.",
   "A documented integrity mandate requires comprehensive historical reprocessing; editing only recent rows is unauthorized."
 ),
 "performance_investigation":(
   "Investigate the single slow query and ship a focused correction.",
   "Audit the full performance path and validate all major components.",
   "Both performance approaches are safe, approved and sufficient for this optional task.",
   "A verified safety-critical failure demands the full performance-path audit; a single-query correction cannot substitute."
 ),
 "handover_process":(
   "Produce a short focused operational handover note.",
   "Complete a detailed end-to-end operational handover package.",
   "Both handover options satisfy all approved obligations and deadlines, so personal style preference decides.",
   "An approved compliance requirement demands a detailed end-to-end handover package; a short note does not satisfy it."
 )
}
def opaque(s):
    return hashlib.sha256(("v90-new-cold|"+s).encode("utf8")).hexdigest()[:18]

def datasets():
    inp,gold,pairs,_,_=v87_build()
    english=[p for p in pairs if p["original"]["lang"]=="en"]
    profiles=sorted({p["original"]["project"]["policy"]["id"] for p in english})
    assert len(profiles)==6 and len(english)==48,(len(profiles),len(english))
    dev_profile=profiles[-1]
    tr=[p for p in english if p["original"]["project"]["policy"]["id"]!=dev_profile]
    dev=[p for p in english if p["original"]["project"]["policy"]["id"]==dev_profile]
    mandatory_train=[];mandatory_dev=[]
    for r,g in zip(inp["train"],gold["train"]):
        if r["lang"]!="en" or g["kind"] not in NEW_KINDS or g["state"]!=1:continue
        item=(r,g["correct_digit"])
        if r["project"]["policy"]["id"]==dev_profile:mandatory_dev.append(item)
        else:mandatory_train.append(item)
    assert (len(tr),len(dev))==(40,8),(len(tr),len(dev))
    assert len(mandatory_train)>=20 and len(mandatory_dev)>=4
    test_profiles={}
    for p in v89_build():
        project=p["original"]["project"]
        test_profiles[project["id"]]=project
    assert len(test_profiles)==2
    cold=[]
    for pid,project in sorted(test_profiles.items()):
        for family,(quick,thorough,neutral,binding) in NEW_FAMILIES.items():
            for status,record in (("neutral",neutral),("binding",binding)):
                for swap in (0,1):
                    actions=[quick,thorough]
                    if swap:actions.reverse()
                    seed=f"{pid}|{family}|{status}|{swap}"
                    row={"project":copy.deepcopy(project),"lang":"en",
                         "task":{
                           "id":"ITEM-"+opaque(seed),"project_id":pid,"layer":"supporting",
                           "proposal_id":"PROP-"+opaque(pid+family),
                           "record":record,
                           "record_source_id":"SRC-"+opaque(pid+family+status),
                           "choices":actions,
                           "authoritative_gate":{"provenance":"fictional-owner-record",
                             "approval_granted":None,"feature_in_scope":None,
                             "governed_action":None},
                         }}
                    twin=copy.deepcopy(row)
                    old=row["project"]["policy"]["mode"]
                    new="rapid" if old=="thorough" else "thorough"
                    twin["project"]["policy"]["mode"]=new
                    twin["project"]["policy"]["en"]=POLICY[new]["en"]
                    twin["project"]["policy"]["he"]=POLICY[new]["he"]
                    twin["project"]["policy"]["version"]=90
                    twin["task"]["id"]="ITEM-"+opaque("alt|"+seed)
                    twin["task"]["record_source_id"]="SRC-"+opaque("alt|"+seed)
                    wanted0=(0 if old=="rapid" else 1) if status=="neutral" else 1
                    wanted1=(0 if new=="rapid" else 1) if status=="neutral" else 1
                    cold.append({
                      "original":row,"counterfactual":twin,
                      "original_label":{"correct_digit":wanted0^swap},
                      "counterfactual_label":{"correct_digit":wanted1^swap},
                      "meta":{"family":family,"status":status,"swap":swap,
                              "project":pid,"expected_flip":status=="neutral"},
                    })
    assert len(cold)==32
    return tr,dev,mandatory_train,mandatory_dev,cold

def audit():
    tr,dev,mand,manddev,cold=datasets()
    trainids={p["original"]["project"]["id"] for p in tr}
    coldids={p["original"]["project"]["id"] for p in cold}
    assert trainids.isdisjoint(coldids)
    assert set(NEW_FAMILIES).isdisjoint(
        {p["original_label"]["kind"] for p in tr})
    assert len({p["original"]["task"]["id"] for p in cold})==32
    for p in tr+dev+cold:
        a=p["original"];b=p["counterfactual"]
        assert a["project"]["objective"]==b["project"]["objective"]
        assert a["task"]["record"]==b["task"]["record"]
        assert a["task"]["choices"]==b["task"]["choices"]
        assert a["project"]["policy"]["mode"]!=b["project"]["policy"]["mode"]
        if p in cold:
            assert (p["original_label"]["correct_digit"]!=p["counterfactual_label"]["correct_digit"])==p["meta"]["expected_flip"]
        assert all(key not in str(a["task"]) for key in ("correct_digit","correct_semantic_option","label_source"))
    assert sum(p["meta"]["status"]=="neutral" for p in cold)==16
    assert sum(p["meta"]["status"]=="binding" for p in cold)==16
    return {"status":"PASS","lang":"en","train_policy_pairs":len(tr),
            "dev_policy_pairs":len(dev),"mandatory_training":len(mand),
            "mandatory_dev":len(manddev),"cold_pairs":len(cold),
            "cold_families":list(NEW_FAMILIES),
            "warning":"Synthetic researcher-authored labels; not human validated."}
if __name__=="__main__":
 import json;print(json.dumps(audit(),indent=2))
