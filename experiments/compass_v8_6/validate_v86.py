"""V8.6 fixture invariants and locked-test policy/source counterfactual audit."""
from collections import Counter,defaultdict
from fixtures_v86 import build,NEW_TEST_ONLY,TRAIN_ONLY
from validate_v85 import audit as audit_v85

def audit():
    assert audit_v85()["status"]=="PASS"
    inputs,golds,twins=build()
    assert len(inputs["train"])==960 and len(inputs["test"])==528
    assert len(twins)==16
    seen=set();groups=defaultdict(dict)
    counts={}
    for split in ("train","test"):
        hist=Counter()
        for row,label in zip(inputs[split],golds[split]):
            t=row["task"];pid=row["project"]["id"]
            assert t["id"]==label["id"] and t["id"] not in seen
            seen.add(t["id"])
            assert t["project_id"]==label["project_id"]==pid
            assert len(t["choices"])==2 and len(set(t["choices"]))==2
            assert label["correct_digit"] in (0,1)
            assert label["correct_digit"]==(label["correct_semantic_option"]^label["swap"])
            assert "correct_digit" not in t and "kind" not in t and "swap" not in t
            key=(split,pid,row["lang"],label["kind"])
            state=(label["state"],label["swap"])
            assert state not in groups[key]
            groups[key][state]=(row,label)
            hist[(row["lang"],label["correct_digit"])]+=1
        assert len(set(hist.values()))==1,hist
        counts[split]={"rows":len(inputs[split]),"project_ids":len(set(r["project"]["id"] for r in inputs[split])),"by_lang_and_digit":{str(k):v for k,v in hist.items()}}
    assert set(r["project"]["id"] for r in inputs["train"]).isdisjoint(
        r["project"]["id"] for r in inputs["test"])
    for (split,pid,lang,kind),q in groups.items():
        assert set(q)=={(0,0),(0,1),(1,0),(1,1)}
        for state in (0,1):
            a=q[(state,0)][0]["task"];b=q[(state,1)][0]["task"]
            assert a["choices"]==list(reversed(b["choices"]))
            assert a["record"]==b["record"]
            assert a["record_source_id"]==b["record_source_id"]
        for swap in (0,1):
            a=q[(0,swap)];b=q[(1,swap)]
            assert a[0]["task"]["choices"]==b[0]["task"]["choices"]
            assert a[0]["task"]["record"]!=b[0]["task"]["record"]
            assert a[1]["correct_digit"]!=b[1]["correct_digit"]
        if split=="train":assert kind not in NEW_TEST_ONLY
        if split=="test":assert kind not in TRAIN_ONLY
    for item in twins:
        a=item["original"];b=item["counterfactual"]
        g=item["original_label"];g2=item["counterfactual_label"]
        assert a["project"]["id"]==b["project"]["id"]
        assert a["project"]["objective"]==b["project"]["objective"]
        assert a["task"]["record"]==b["task"]["record"]
        assert a["task"]["choices"]==b["task"]["choices"]
        assert a["project"]["policy"]["mode"]!=b["project"]["policy"]["mode"]
        assert a["project"]["policy"]["en"]!=b["project"]["policy"]["en"]
        assert a["project"]["policy"]["he"]!=b["project"]["policy"]["he"]
        assert g["kind"] in ("rollout_preference","coverage_preference")
        assert g["state"]==0 and g2["state"]==0 and g["swap"]==g2["swap"]
        assert g["correct_digit"]!=g2["correct_digit"]
        assert a["task"]["id"]!=b["task"]["id"]
        assert a["task"]["record_source_id"]!=b["task"]["record_source_id"]
        assert b["task"]["id"] not in seen
        seen.add(b["task"]["id"])
    selected=sorted({r["project"]["id"] for r in inputs["test"]})
    two={selected[0],selected[3]}
    assert {r["project"]["policy"]["mode"] for r in inputs["test"] if r["project"]["id"] in two}=={"rapid","thorough"}
    assert sum(r["project"]["id"] in two for r in inputs["test"])==176
    for kind in NEW_TEST_ONLY:
        assert sum(g["kind"]==kind and r["project"]["id"] in two
            for r,g in zip(inputs["test"],golds["test"]))==16
    return {"status":"PASS","counts":counts,"selected_test_projects":sorted(two),
            "policy_only_pairs":len(twins),"cold_families":list(NEW_TEST_ONLY),
            "unseen_test_cases":32,"source_retraction_old_diagnostic":16,
            "caveats":["Test gold and training scenarios still authored by one workflow; no external adjudication",
                       "V8.5 source_retracted was inspected in a previous experiment, not a pristine cold test",
                       "Policy flip is isolated within a fictional project and neutral evidence only",
                       "Order-equivalence is architectural, not learned decision competence"]}
if __name__=="__main__":
    import json
    print(json.dumps(audit(),ensure_ascii=False,indent=2))
