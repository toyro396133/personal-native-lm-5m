"""Independent rule-based baseline for COMPASS V8 stage-0 cases.

This recognizer reads only prompt-side structured project memory and observable
task text, NEVER evaluation labels or case-kind metadata. It is an upper-control
on an authored miniature template set, not an independent policy oracle.
"""
import json
import re
from pathlib import Path

ACTIONS=("DO_NOW","SCHEDULE","INVESTIGATE","DEFER","ASK_USER","REJECT")
LAYERS=("core","supporting","ancillary","out_of_scope")

def decide(case):
    project=case["project"]
    task=case["task"]
    assert task["project_id"]==project["id"],"Cross-project context mismatch"
    hits=[n for n in project["nodes"] if n["id"]==task["node_id"]]
    if len(hits)!=1:raise ValueError("Task references unavailable project node")
    layer=hits[0]["layer"]
    proposal=task["proposal"].casefold()
    principles=case["personal_policy"]["approved_guidance"].casefold()
    all_decisions={d["id"]:d for d in project["decisions"] if d["valid"]}
    root=project["id"]
    rationale=f"{root}:D1"
    blocked=False

    if layer=="out_of_scope" and any(k in proposal for k in
                                    ("unrelated","side product","scope")):
        action="REJECT"; rationale=f"{root}:D3"
    elif any(k in proposal for k in ("approval is missing","no authorized decision")):
        action="ASK_USER"; rationale=f"{root}:D2"
    elif any(k in proposal for k in
             ("possible problem","vague complaint","no reliable reproduction","lacks a reproduction")):
        action="INVESTIGATE"
    elif any(k in proposal for k in
             ("one hundred extra checks","extending routine checks substantially")):
        action="SCHEDULE" if any(k in principles for k in
                                ("schedule broad qa","schedule systematic qa")) else "DEFER"
    elif any(k in proposal for k in
             ("regression test is required","smallest proof","supporting prerequisite",
              "supporting setup","reproducible failure","verified corruption")):
        action="DO_NOW";blocked=True
    elif any(k in proposal for k in ("agreed next core capability","next expected deliverable")):
        action="SCHEDULE"
    else:
        action="INVESTIGATE"
    assert rationale in all_decisions
    return {
        "project_id":project["id"],
        "structural_layer":layer,
        "action":action,
        "blocked_core":blocked,
        "evidence_ids":[rationale],
        "needs_user_decision":action=="ASK_USER",
    }

def evaluate(inputs,golds):
    labels={x["task_id"]:x for x in golds}
    if len(labels)!=len(golds):raise ValueError("Duplicate gold task id")
    details=[]
    for item in inputs:
        t=item["task"]["id"]
        expected=labels[t]
        pred=decide(item)
        hit=(pred["action"]==expected["action"] and
             pred["structural_layer"]==expected["layer"] and
             pred["blocked_core"]==expected["blocked_core"] and
             pred["project_id"]==expected["project_id"])
        details.append({"id":t,"hit":hit,"action":pred["action"],"expected_action":expected["action"],
                        "layer":pred["structural_layer"],"expected_layer":expected["layer"]})
    return {"correct":sum(x["hit"] for x in details),
            "total":len(details),"failures":[x for x in details if not x["hit"]]}

if __name__=="__main__":
    from fixtures_v8 import generate
    xs,ys=generate()
    print(json.dumps({s:evaluate(xs[s],ys[s]) for s in ("train","test")},indent=2))
