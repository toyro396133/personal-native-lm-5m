"""No-label-leakage, train/test grouping, project integrity and trap checks.
This is an internal generator audit, NOT independent proof of learning validity.
"""
import argparse
import json
import re
from pathlib import Path

from fixtures_v8 import CASES, PROFILES, PROJECT_KINDS, generate, write_dataset
from baseline_v8 import evaluate, decide, ACTIONS, LAYERS

def assert_dataset():
    inputs,gold=generate()
    counts={"train":len(inputs["train"]),"test":len(inputs["test"])}
    assert counts=={"train":64,"test":32},counts
    input_projects={s:{r["project"]["id"] for r in inputs[s]} for s in ("train","test")}
    assert len(input_projects["train"])==8 and len(input_projects["test"])==4
    assert input_projects["train"].isdisjoint(input_projects["test"])
    templates={s:{r["template_id"] for r in gold[s]} for s in ("train","test")}
    assert len(templates["train"])==8 and len(templates["test"])==8
    assert templates["train"].isdisjoint(templates["test"])
    task_ids={s:set() for s in ("train","test")}
    profiles={s:set() for s in ("train","test")}
    for split in ("train","test"):
        assert len(inputs[split])==len(gold[split])
        for row,label in zip(inputs[split],gold[split]):
            project=row["project"];task=row["task"]
            tid=task["id"]
            assert tid not in task_ids[split]
            task_ids[split].add(tid)
            profiles[split].add(row["profile_key"])
            assert re.fullmatch(r"T-[TE]\d{4}",tid),f"Task ID leaks semantic label: {tid}"
            assert task["project_id"]==project["id"]
            assert label["project_id"]==project["id"]
            assert label["task_id"]==tid
            assert len([n for n in project["nodes"] if n["id"]==task["node_id"]])==1
            assert project["objective"] and project["phase"] and project["version"]==1
            assert len(project["decisions"])>=3
            assert all(d["valid"] and d["source"] for d in project["decisions"])
            assert label["action"] in ACTIONS and label["layer"] in LAYERS
            assert label["blocked_core"] in (True,False)
            assert len(label["evidence_ids"])>=1
            assert all(e in [d["id"] for d in project["decisions"]] for e in label["evidence_ids"])
            assert not any(k in task for k in ("label","case_kind","gold","priority_tier","expected_action"))
            serialized=json.dumps(row)
            assert not re.search(r'"(?:gold|expected_action|case_kind|template_id|correct_action|priority_tier)"\s*:',serialized)
            # Task identifiers and evidence IDs are opaque IDs with no case-kind words.
            assert not any(name in task["id"].lower() for name,_ in CASES)
            assert not any(name in str(task.get("evidence_source_id","")).lower() for name,_ in CASES)
        assert profiles[split]=={f"fiction-{p[0]}" for p in PROFILES}
    assert task_ids["train"].isdisjoint(task_ids["test"])

    baseline={s:evaluate(inputs[s],gold[s]) for s in ("train","test")}
    # Baseline is deliberately a strong hardcoded source-rule sanity control.
    assert all(res["correct"]==res["total"] for res in baseline.values()),baseline

    # Same ancillary node, opposite decisions depending on evidence of core impact.
    traps=[]
    for split in ("train","test"):
        byproject={}
        for inp,lab in zip(inputs[split],gold[split]):
            key=inp["project"]["id"]
            byproject.setdefault(key,{})[lab["case_kind"]]=(inp,lab)
        for pid,cases in byproject.items():
            broad=cases["broad_test"]
            targeted=cases["targeted_test"]
            assert broad[1]["layer"]=="ancillary"==targeted[1]["layer"]
            assert targeted[1]["action"]=="DO_NOW" and targeted[1]["blocked_core"]
            assert broad[1]["action"] in ("SCHEDULE","DEFER") and not broad[1]["blocked_core"]
            assert cases["confirmed_break"][1]["action"]=="DO_NOW"
            assert cases["suspected_issue"][1]["action"]=="INVESTIGATE"
            assert cases["needs_approval"][1]["action"]=="ASK_USER"
            assert cases["forbidden_scope"][1]["action"]=="REJECT"
            traps.append({"project_id":pid,"ancillary_opposite":True,
                          "confirmed_versus_suspected":True})
    assert len(traps)==12

    # A task must not be silently evaluated against a different project graph.
    foreign=json.loads(json.dumps(inputs["train"][0]))
    foreign["task"]["project_id"]="fiction-someone-else-workflow"
    try:
        decide(foreign)
    except AssertionError:
        pass
    else:
        raise AssertionError("Cross-project guard failed")
    return {
        "status":"PASS", "counts":counts,"projects_total":12,
        "profiles_total":4,"trap_project_pairs":len(traps),
        "train_test_project_overlap":0,"train_test_template_overlap":0,
        "rule_baseline":{s:{"correct":v["correct"],"total":v["total"]} for s,v in baseline.items()},
        "methodological_limitations":[
            "Authored miniature set with templated wording; not independent natural-language gold adjudication",
            "Deterministic baseline and generator share an authored research scope; 100% baseline is a fixture sanity check",
            "Pilot project split changes project type together with split; holdout effects are confounded",
            "All four fictional users appear in both splits; not a profile-level cold-transfer evaluation",
            "No user-specific confidential repository or personal conversation data are used"
        ]
    }

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",default="v8-fixtures")
    args=parser.parse_args()
    result=assert_dataset()
    write_dataset(args.root)
    target=Path(args.root)/"audit.json"
    target.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result,indent=2),flush=True)
