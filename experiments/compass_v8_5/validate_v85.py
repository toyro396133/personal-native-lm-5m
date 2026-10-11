"""COMPASS V8.5 preflight for three exact-size personal rankers."""
import json
from collections import Counter,defaultdict
from fixtures_v85 import build, NEW_KINDS,NOVEL_TEST_KIND
from guard_v82 import enforce,InvalidContext

def audit():
    inputs,golds=build()
    assert len(inputs["train"])==768,(len(inputs["train"]))
    assert len(inputs["test"])==432,(len(inputs["test"]))
    assert len(golds["train"])==768 and len(golds["test"])==432
    projects={}
    counts={}
    unknowns=0
    for split in ("train","test"):
        user_policies=defaultdict(set)
        grouped=defaultdict(dict)
        seen_ids=set()
        for row,label in zip(inputs[split],golds[split]):
            pid=row["project"]["id"]
            task=row["task"]
            assert task["id"]==label["id"] and task["id"] not in seen_ids
            seen_ids.add(task["id"])
            assert task["project_id"]==pid==label["project_id"]
            assert task["record_source_id"].startswith("SRC-")
            assert task["authoritative_gate"]["provenance"]=="fictional-owner-record"
            assert "correct_digit" not in json.dumps(row,ensure_ascii=False)
            assert "kind" not in task and "state" not in task and "swap" not in task
            assert len(task["choices"])==2 and task["choices"][0]!=task["choices"][1]
            assert label["correct_digit"] in (0,1)
            assert label["correct_digit"]==(label["correct_semantic_option"]^label["swap"])
            assert row["project"]["policy"]["version"]==2
            user_policies[row["project"]["policy"]["mode"]].add(row["project"]["policy"]["id"])
            key=(pid,row["lang"],label["kind"])
            pos=(label["state"],label["swap"])
            assert pos not in grouped[key]
            grouped[key][pos]=(row,label)
            if label["kind"]==NOVEL_TEST_KIND:unknowns+=1
            decision=enforce(row,label["correct_digit"])
            assert decision["choice"] in (0,1)
        assert len(user_policies["rapid"])>=2 and len(user_policies["thorough"])>=1
        for key,quartet in grouped.items():
            assert set(quartet)=={(0,0),(0,1),(1,0),(1,1)},key
            for state in (0,1):
                first=quartet[(state,0)][0]
                swapped=quartet[(state,1)][0]
                assert first["task"]["choices"]==list(reversed(swapped["task"]["choices"]))
                assert first["task"]["record"]==swapped["task"]["record"]
                assert first["task"]["record_source_id"]==swapped["task"]["record_source_id"]
            for swap in (0,1):
                a=quartet[(0,swap)];b=quartet[(1,swap)]
                assert a[0]["task"]["choices"]==b[0]["task"]["choices"]
                assert a[0]["task"]["record"]!=b[0]["task"]["record"]
                assert a[1]["correct_digit"]!=b[1]["correct_digit"]
        projects[split]={r["project"]["id"] for r in inputs[split]}
        counts[split]={"examples":len(seen_ids),"quartets":len(grouped),
                       "modes":{mode:len(ids) for mode,ids in user_policies.items()},
                       "languages":dict(Counter(r["lang"] for r in inputs[split]))}
    assert projects["train"].isdisjoint(projects["test"])
    assert unknowns==48
    assert all(x["kind"]!=NOVEL_TEST_KIND for x in golds["train"])
    test_projects=sorted(projects["test"])
    chosen={test_projects[0],test_projects[3]}
    policy_on_chosen={r["project"]["policy"]["mode"] for r in inputs["test"]
                      if r["project"]["id"] in chosen}
    assert policy_on_chosen=={"rapid","thorough"},policy_on_chosen
    assert sum(r["project"]["id"] in chosen for r in inputs["test"])==144
    return {"status":"PASS","counts":counts,"novel_family_test_only":unknowns,
            "selected_test_projects":sorted(chosen),
            "selected_test_policies":sorted(policy_on_chosen),
            "design_limits":[
               "All cases, gold decisions and wording are synthesized by one workflow",
               "Not independent or human-adjudicated labels; semantically overlapping family templates",
               "Two opposite policies are fictional but truly differ in preference instructions",
               "All test metrics derive from related four-case groups, not independent decisions",
               "Authoritative gate is synthetic prototype; safe real execution needs trusted provenance"
            ]}
if __name__=="__main__":print(json.dumps(audit(),indent=2,ensure_ascii=False))
