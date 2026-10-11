"""V8.8 preflight: separate train/test identities, policy-only updates, no oracle in teacher."""
from collections import Counter
from fixtures_v88 import build,NEW_COLD
from fixtures_v87 import COLD_POLICY
from validate_v87 import audit as older
from trusted_sources_v87 import assert_contract

def audit():
    assert older()["status"]=="PASS"
    assert assert_contract()["status"]=="PASS"
    inp,gold,training,old,exposed,new=build()
    assert len(training)==96 and len(old)==16 and len(exposed)==16 and len(new)==16
    train_ids={r["project"]["id"] for r in inp["train"]}
    test_ids={r["project"]["id"] for r in inp["test"]}
    assert train_ids.isdisjoint(test_ids)
    for tag,arr in (("train",training),("exposed_v86",old),
                    ("exposed_v87",exposed),("cold_v88",new)):
        ids=set()
        for item in arr:
            first=item["original"];second=item["counterfactual"]
            a=item["original_label"];b=item["counterfactual_label"]
            assert first["project"]["id"]==second["project"]["id"]
            assert first["project"]["objective"]==second["project"]["objective"]
            assert first["task"]["choices"]==second["task"]["choices"]
            assert first["task"]["record"]==second["task"]["record"]
            assert first["project"]["policy"]["mode"]!=second["project"]["policy"]["mode"]
            assert a["correct_digit"]!=b["correct_digit"]
            assert first["task"]["id"]!=second["task"]["id"]
            assert first["task"]["id"] not in ids
            ids.add(first["task"]["id"])
            for r in (first,second):
                # Neither gold nor synthetic approval authority enters model prompt.
                for forbidden in ("correct_digit","correct_semantic_option","label_source"):
                    assert forbidden not in r["task"]
            if tag=="train":assert first["project"]["id"] in train_ids
            else:assert first["project"]["id"] in test_ids
            if tag=="cold_v88":assert a["kind"] in NEW_COLD
            else:assert a["kind"] not in NEW_COLD
    assert Counter(x["original"]["lang"] for x in new)=={"en":8,"he":8}
    assert {x["original_label"]["kind"] for x in new}==set(NEW_COLD)
    assert {x["original_label"]["kind"] for x in new}.isdisjoint(
         {x["original_label"]["kind"] for x in training})
    return {"status":"PASS","policy_training_pairs":80,
            "development_pairs":16,"cold_policy_pairs":16,
            "previously_exposed_diagnostic_pairs":32,
            "source_authorization_contract":"synthetic fail-closed checks PASS",
            "limitations":"Only researcher-authored synthetic bilingual scenarios; no external adjudication"}
if __name__=="__main__":
    import json;print(json.dumps(audit(),ensure_ascii=False,indent=2))
