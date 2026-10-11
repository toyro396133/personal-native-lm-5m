"""COMPASS V8 stage-0: entirely fictional per-project hierarchy fixtures.

Train and test groups have disjoint project IDs *and* authoring template IDs.
The prompt-side context excludes oracle case IDs, labels, rankings and expected
rationales. Gold labels are stored only in a separate output tree.
"""
import argparse
import json
from pathlib import Path

PROFILES = (
    ("cedar", "Protect confirmed core blockers before optional expansions; defer broad QA unless required."),
    ("orchid", "Protect core blockers; schedule broad QA coverage as a later quality gate."),
    ("slate", "Fix reproducible blockers and preserve evidence; defer optional test expansion."),
    ("birch", "Protect core milestones; schedule systematic QA separately from incident response."),
)
PROJECT_KINDS = (
    ("workflow", "Complete a reliable record-save workflow", "Reliable record saving", "Core data integrity"),
    ("research", "Publish a reproducible research result", "Reproducible core pipeline", "Reliable experiment records"),
    ("planning", "Maintain an auditable priority planning process", "Decision quality", "Decision trace integrity"),
)
CASES = (
    ("confirmed_break", "core"),
    ("broad_test", "ancillary"),
    ("targeted_test", "ancillary"),
    ("suspected_issue", "supporting"),
    ("prerequisite", "supporting"),
    ("planned_work", "core"),
    ("forbidden_scope", "out_of_scope"),
    ("needs_approval", "core"),
)
# Distinct held-out wording families, not row-wise random splits.
WORDINGS = {
    "train": {
        "confirmed_break": "A reproducible failure in {core} prevents saving correct core output. Logged evidence: {evidence}.",
        "broad_test": "Proposal: add one hundred extra checks without any reported current core failure. No dependency to the current core milestone is established.",
        "targeted_test": "A focused regression test is required before releasing the fix for a confirmed {core} failure. Reproduction exists in {evidence}.",
        "suspected_issue": "A possible problem in {core} was mentioned, but no reliable reproduction or logs are available yet.",
        "prerequisite": "Complete the minimal supporting prerequisite required to unblock the confirmed {core} failure, backed by {evidence}.",
        "planned_work": "Implement the agreed next core capability in {core}; no current blocker or emergency has been reported.",
        "forbidden_scope": "Build an unrelated optional product, despite the current project's explicit scope restriction.",
        "needs_approval": "Make an irreversible change to the core objective, but the user's approval is missing.",
    },
    "test": {
        "confirmed_break": "Verified corruption of {core} was reproduced twice; without a fix the project cannot meet its main objective. Case {evidence}.",
        "broad_test": "The team suggests extending routine checks substantially. Nobody has linked this idea to a defect or an essential deliverable.",
        "targeted_test": "Before safely applying a remedy to the demonstrated {core} defect, write the smallest proof that the fault remains fixed ({evidence}).",
        "suspected_issue": "A vague complaint alleges an issue with {core}; it lacks a reproduction and its impact is not established.",
        "prerequisite": "The core-fix path is blocked until this supporting setup is completed, as documented by {evidence}.",
        "planned_work": "The next expected deliverable for {core} is ready to schedule, with no urgent error to resolve.",
        "forbidden_scope": "Allocate the team to a new side product that the agreed project boundaries expressly exclude.",
        "needs_approval": "Replace the accepted project mission with a different mission immediately, although no authorized decision exists.",
    },
}

ACTIONS=("DO_NOW","SCHEDULE","INVESTIGATE","DEFER","ASK_USER","REJECT")
LAYERS=("core","supporting","ancillary","out_of_scope")

def make_project(profile_index,project_index):
    name,purpose,core,integrity=PROJECT_KINDS[project_index]
    pid=f"fiction-{PROFILES[profile_index][0]}-{name}"
    return {
        "id":pid, "title":f"Fictional {name} initiative",
        "version":1, "objective":purpose, "phase":"foundation",
        "nodes":[
            {"id":f"{pid}:C","layer":"core","intent":core},
            {"id":f"{pid}:S","layer":"supporting","intent":integrity},
            {"id":f"{pid}:A","layer":"ancillary","intent":"Expanded optional test coverage"},
            {"id":f"{pid}:X","layer":"out_of_scope","intent":"Unrelated external product"},
        ],
        "decisions":[
            {"id":f"{pid}:D1","statement":"Confirmed failures of the main deliverable take precedence over optional expansion.",
             "source":"fictional-project-owner","valid":True},
            {"id":f"{pid}:D2","statement":"Changes to the main objective require explicit project-owner approval.",
             "source":"fictional-project-owner","valid":True},
            {"id":f"{pid}:D3","statement":"An unrelated side product is out of approved scope.",
             "source":"fictional-project-owner","valid":True},
        ],
    }

def oracle_case(case,profile_index,project,split):
    # Oracle never goes in test/model input; it is only stored in gold/*.jsonl.
    core_node=project["nodes"][0]["id"]
    extra={
        "confirmed_break":("DO_NOW",True,["D1"]),
        "broad_test":("SCHEDULE" if profile_index in (1,3) else "DEFER",False,["D1"]),
        "targeted_test":("DO_NOW",True,["D1"]),
        "suspected_issue":("INVESTIGATE",False,["D1"]),
        "prerequisite":("DO_NOW",True,["D1"]),
        "planned_work":("SCHEDULE",False,["D1"]),
        "forbidden_scope":("REJECT",False,["D3"]),
        "needs_approval":("ASK_USER",False,["D2"]),
    }
    action,blocked,decisions=extra[case]
    return {"action":action,"blocked_core":blocked,"evidence_ids":[f"{project['id']}:{d}" for d in decisions],
            "layer":dict(CASES)[case],"project_id":project["id"],"case_kind":case,
            "template_id":f"{split}-template-{case}"}

def make_case(profile_index,project_index,case,split):
    project=make_project(profile_index,project_index)
    node_type=dict(CASES)[case]
    node=next(n for n in project["nodes"] if n["layer"]==node_type)
    case_index=[name for name,_ in CASES].index(case)
    tid=f"T-{split[0].upper()}{profile_index}{project_index}{case_index:02d}"
    report_id=f"EV-{profile_index}{project_index}-{case_index:02d}"
    statement=WORDINGS[split][case].format(core=project["nodes"][0]["intent"],evidence=report_id)
    context={
        "profile_key":f"fiction-{PROFILES[profile_index][0]}",
        "project":project,
        "personal_policy":{
            "id":f"P{profile_index}","version":1,
            "approved_guidance":PROFILES[profile_index][1],
        },
        "task":{"id":tid,"project_id":project["id"],"node_id":node["id"],
                "proposal":statement,"evidence_source_id":report_id if case in (
                    "confirmed_break","targeted_test","prerequisite") else None},
    }
    gold=oracle_case(case,profile_index,project,split)
    gold.update({"task_id":tid,"node_id":node["id"]})
    return context,gold

def generate():
    sets={"train":[],"test":[]}
    truth={"train":[],"test":[]}
    for pi in range(len(PROFILES)):
        for ji in range(len(PROJECT_KINDS)):
            split="test" if ji==2 else "train"
            for case,_ in CASES:
                prompt,label=make_case(pi,ji,case,split)
                sets[split].append(prompt)
                truth[split].append(label)
    return sets,truth

def write_dataset(root):
    root=Path(root)
    inputs,golds=generate()
    (root/"inputs").mkdir(parents=True,exist_ok=True)
    (root/"gold").mkdir(parents=True,exist_ok=True)
    for split in ("train","test"):
        (root/"inputs"/f"{split}.jsonl").write_text(
            "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in inputs[split]),encoding="utf-8")
        (root/"gold"/f"{split}.jsonl").write_text(
            "".join(json.dumps(row,ensure_ascii=False)+"\n" for row in golds[split]),encoding="utf-8")
    return {"train":len(inputs["train"]),"test":len(inputs["test"]),
            "profiles":len(PROFILES),"projects":len(PROFILES)*len(PROJECT_KINDS),
            "core_blocker_cases":sum(x["blocked_core"] for x in golds["train"]+golds["test"])}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",default="v8-fixtures")
    args=p.parse_args()
    print(json.dumps(write_dataset(args.root),indent=2))
