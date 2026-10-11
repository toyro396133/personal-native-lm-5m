"""COMPASS V8.2 preflight: no oracle labels in model-visible context,
balanced reversible choices, user/project holdout, fact-flip cases and
reference permission checks. This does NOT validate semantic correctness
of authored fictional labels independently.
"""
import json
from collections import Counter,defaultdict
from fixtures_v82 import generate,KINDS,LANGUAGES,OPTIONS,FACTS

def audit():
    ins,golds=generate()
    assert len(ins["train"])==576 and len(ins["test"])==288
    assert len(golds["train"])==576 and len(golds["test"])==288
    training_users={x["project"]["policy"]["id"] for x in ins["train"]}
    holdout_users={x["project"]["policy"]["id"] for x in ins["test"]}
    training_projects={x["project"]["id"] for x in ins["train"]}
    holdout_projects={x["project"]["id"] for x in ins["test"]}
    assert len(training_users)==6 and len(holdout_users)==3
    assert len(training_projects)==12 and len(holdout_projects)==6
    assert training_users.isdisjoint(holdout_users)
    assert training_projects.isdisjoint(holdout_projects)
    split_label_counts={}
    counts={s:Counter() for s in ("train","test")}
    all_ids=set()
    for split in ("train","test"):
        seen=defaultdict(dict)
        gold_ids={v["id"] for v in golds[split]}
        assert len(gold_ids)==len(golds[split])
        for row,oracle in zip(ins[split],golds[split]):
            p=row["project"];task=row["task"]
            assert task["project_id"]==p["id"]==oracle["project_id"]
            assert task["id"]==oracle["id"] and task["id"] not in all_ids
            all_ids.add(task["id"])
            assert oracle["kind"] in KINDS
            assert oracle["correct_digit"] in (0,1)
            assert len(task["choices"])==2
            assert len({*task["choices"]})==2
            assert p["objective"][row["lang"]]
            assert p["policy"]["source"]=="fictional-owner"
            assert task["record_source_description"]
            assert "label" not in json.dumps(row)
            assert not any(k in task for k in ("kind","state","swap","correct_digit"))
            k=(p["id"],row["lang"],oracle["kind"])
            seen[k][(oracle["state"],oracle["swap"])]=(row,oracle)
            counts[split][(row["lang"],oracle["correct_digit"])]+=1
        for (pid,lang,kind),cases in seen.items():
            assert len(cases)==4, (pid,lang,kind)
            for state in (0,1):
                nonswapped=cases[(state,0)]
                reversed_case=cases[(state,1)]
                assert list(reversed(nonswapped[0]["task"]["choices"]))==reversed_case[0]["task"]["choices"]
                assert nonswapped[0]["task"]["record"]==reversed_case[0]["task"]["record"]
                assert nonswapped[1]["correct_digit"]!=reversed_case[1]["correct_digit"]
            for swap in (0,1):
                first=cases[(0,swap)]
                second=cases[(1,swap)]
                assert first[0]["task"]["choices"]==second[0]["task"]["choices"]
                assert first[0]["task"]["record"]!=second[0]["task"]["record"]
                assert first[1]["correct_digit"]!=second[1]["correct_digit"]
            assert cases[(0,0)][0]["task"]["record"]==FACTS[kind][lang][0]
            assert cases[(1,0)][0]["task"]["record"]==FACTS[kind][lang][1]
            assert cases[(0,0)][0]["task"]["choices"]==list(OPTIONS[kind][lang])
        split_label_counts[split]={f"{l}-{code}":counts[split][(l,code)]
                                    for l in LANGUAGES for code in (0,1)}
        assert len(set(split_label_counts[split].values()))==1,split_label_counts
    assert len(all_ids)==576+288
    return {"status":"PASS","train_examples":576,"heldout_examples":288,
            "train_users":6,"heldout_users":3,"train_projects":12,"heldout_projects":6,
            "decision_families":len(KINDS),"languages":list(LANGUAGES),
            "label_balance":split_label_counts,
            "paired_fact_flips_per_split":{"train":12*2*len(KINDS),"test":6*2*len(KINDS)},
            "model_input_label_leakage_check":"PASS",
            "caveats":[
                "Synthetic scenario wording and labels are authored, not independently judged.",
                "User policy text is intentionally the same transferable principle in every profile, so cold-profile ID split does not establish differing-policy adaptation.",
                "Source references describe fictional evidence; no connected personal/project data accessed.",
                "Similar scenario grammar and identical decision families recur across train/test.",
                "This validates dataset structure and flips, not neural learning.",
            ]}
if __name__=="__main__":
    print(json.dumps(audit(),indent=2,ensure_ascii=False))
