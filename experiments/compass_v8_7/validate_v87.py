"""Check split isolation, policy-only invariance and cold-family disjointness."""
from collections import Counter
from fixtures_v87 import build,COLD_POLICY
from validate_v86 import audit as audit_v86
from trusted_sources_v87 import assert_contract

def audit():
    assert audit_v86()["status"]=="PASS"
    assert assert_contract()["status"]=="PASS"
    inputs,golds,training,exposed,cold=build()
    assert len(training)==96
    assert len(exposed)==16
    assert len(cold)==16
    source_train={r["project"]["id"] for r in inputs["train"]}
    source_test={r["project"]["id"] for r in inputs["test"]}
    assert source_train.isdisjoint(source_test)
    assert {r["project"]["id"] for r in inputs["test"] if r["lang"]=="he"}
    old_keys=set()
    for split in ("train","test"):
        for r in inputs[split]:
            assert r["task"]["id"] not in old_keys
            old_keys.add(r["task"]["id"])
    seen=set(old_keys)
    for cohort,pairs in (("train",training),("exposed",exposed),("cold",cold)):
        for p in pairs:
            a=p["original"];b=p["counterfactual"]
            g=p["original_label"];h=p["counterfactual_label"]
            assert g["state"]==h["state"]==0
            assert g["swap"]==h["swap"]
            assert g["correct_digit"]!=h["correct_digit"]
            assert a["project"]["id"]==b["project"]["id"]
            assert a["project"]["objective"]==b["project"]["objective"]
            assert a["task"]["record"]==b["task"]["record"]
            assert a["task"]["choices"]==b["task"]["choices"]
            assert a["project"]["policy"]["mode"]!=b["project"]["policy"]["mode"]
            assert a["project"]["policy"]["en"]!=b["project"]["policy"]["en"]
            assert a["project"]["policy"]["he"]!=b["project"]["policy"]["he"]
            assert b["task"]["id"] not in seen
            seen.add(b["task"]["id"])
            assert a["task"]["record_source_id"]!=b["task"]["record_source_id"]
            if cohort=="train":assert a["project"]["id"] in source_train
            else:assert a["project"]["id"] in source_test
            if cohort=="cold":assert g["kind"] in COLD_POLICY
            else:assert g["kind"] not in COLD_POLICY
    # All 16 counterfactual TEST cases are from two truly heldout projects.
    assert len({x["original"]["project"]["id"] for x in cold})==2
    assert Counter(x["original"]["lang"] for x in cold)=={"en":8,"he":8}
    assert {x["original"]["project"]["policy"]["mode"] for x in cold}=={"rapid","thorough"}
    profiles=sorted({r["project"]["policy"]["id"] for r in inputs["train"]})
    dev_profile=profiles[-1]
    tr=[x for x in training if x["original"]["project"]["policy"]["id"]!=dev_profile]
    dv=[x for x in training if x["original"]["project"]["policy"]["id"]==dev_profile]
    assert len(tr)==80 and len(dv)==16
    assert set(x["original"]["project"]["id"] for x in tr).isdisjoint(
               x["original"]["project"]["id"] for x in dv)
    # Each source/option context has a stable frozen representation;
    # all supervised labels live separately from representation prompt text.
    assert {x["original_label"]["kind"] for x in cold}.isdisjoint(
        {x["original_label"]["kind"] for x in tr})
    return {"status":"PASS","train_policy_pairs":len(tr),
            "development_policy_pairs":len(dv),
            "previously_exposed_test_pairs":len(exposed),
            "new_cold_policy_pairs":len(cold),
            "limitations":"Authored synthetic labels; no human or third-party independent adjudication"}
if __name__=="__main__":
    import json;print(json.dumps(audit(),indent=2))
