"""Audit V8.1 project separation, counterfactuals, and authorization controls."""
import json
from collections import Counter
from fixtures_v81 import CASES,ACTIONS,generate
from guard_v81 import guard_action,InvalidProjectContext

PAIRS=(
    ("confirmed_failure","uncertain_failure"),
    ("targeted_regression","optional_checks"),
    ("critical_dependency","routine_dependency"),
    ("approved_change","pending_change"),
    ("approved_expansion","out_scope_expansion"),
    ("later_cosmetics","unclear_suggestion"),
)
def audit():
    inputs,golds=generate()
    assert {k:len(v) for k,v in inputs.items()}=={"train":192,"test":72}
    train_ids={x["project"]["id"] for x in inputs["train"]}
    test_ids={x["project"]["id"] for x in inputs["test"]}
    assert len(train_ids)==16 and len(test_ids)==6 and train_ids.isdisjoint(test_ids)
    assert {x["profile_key"] for x in inputs["train"]}.isdisjoint(
           {x["profile_key"] for x in inputs["test"]})
    assert {g["template_id"] for g in golds["train"]}.isdisjoint(
           {g["template_id"] for g in golds["test"]})
    all_pairs=[]
    guards=Counter()
    for split in ("train","test"):
        labels={g["task_id"]:g for g in golds[split]}
        assert len(labels)==len(inputs[split]) and len(set(labels))==len(labels)
        cases={}
        for row in inputs[split]:
            project=row["project"];task=row["task"]
            assert task["id"] in labels
            assert project["id"]==task["project_id"]==labels[task["id"]]["project_id"]
            assert all(d.get("source") and d.get("valid") for d in project["decisions"])
            assert sum(n["id"]==task["node_id"] for n in project["nodes"])==1
            assert labels[task["id"]]["action"] in ACTIONS
            assert "case_kind" not in json.dumps(row)
            assert "gold" not in json.dumps(row)
            assert "expected" not in json.dumps(row)
            assert task["id"].split("-")[1].isdigit() is False
            chosen=guard_action(row,"DO_NOW")
            expected=labels[task["id"]]["action"]
            if task["scope_status"]=="outside":
                assert chosen["action"]=="REJECT"==expected
                guards["out_scope"]+=1
            elif task["requires_owner_approval"] and task["approval_status"]=="pending":
                assert chosen["action"]=="ASK_USER"==expected
                guards["unapproved"]+=1
            else:
                assert chosen["action"]=="DO_NOW" and not chosen["overridden"]
            cases.setdefault(project["id"],{})[labels[task["id"]]["case_kind"]]=(row,labels[task["id"]])
        assert len(cases)==(16 if split=="train" else 6)
        for project_id,items in cases.items():
            assert set(items)==set(CASES)
            for left,right in PAIRS:
                a,b=items[left],items[right]
                assert a[1]["action"]!=b[1]["action"],(project_id,left,right)
                assert a[0]["task"]["node_id"]==b[0]["task"]["node_id"]
                assert a[0]["project"]["id"]==b[0]["project"]["id"]
                all_pairs.append({"project":project_id,"pair":[left,right]})
    foreign=json.loads(json.dumps(inputs["test"][0]))
    foreign["task"]["project_id"]="fiction-foreign"
    try:
        guard_action(foreign,"DO_NOW")
    except InvalidProjectContext:
        pass
    else:
        raise AssertionError("Cross-project protection failed")
    missing=json.loads(json.dumps(inputs["test"][0]))
    missing["task"].pop("scope_status")
    assert guard_action(missing,"DO_NOW")["action"]=="ASK_USER"
    missing=json.loads(json.dumps(inputs["test"][0]))
    missing["task"]["node_id"]="fiction-unknown-node"
    try:
        guard_action(missing,"DO_NOW")
    except InvalidProjectContext:pass
    else:raise AssertionError("Unknown node accepted")
    return {
        "status":"PASS","train":len(inputs["train"]),"test":len(inputs["test"]),
        "train_projects":len(train_ids),"test_projects":len(test_ids),
        "counterfactual_pairs":len(all_pairs),
        "test_counterfactual_pairs":6*len(PAIRS),
        "guarded_scope_cases":guards["out_scope"],
        "guarded_unapproved_cases":guards["unapproved"],
        "train_test_project_overlap":0,
        "train_test_user_profile_overlap":0,
        "train_test_wording_template_overlap":0,
        "caveats":[
          "Text templates and gold rules were authored by the same research team",
          "Changing two task descriptions can change more than one semantic cue",
          "No independent human adjudication or real user project history was used",
          "Explicit structured permission flags must come from trusted provenance in a live product",
          "This data audit is not a neural-model performance result",
        ]
    }
if __name__=="__main__":
    print(json.dumps(audit(),indent=2))
