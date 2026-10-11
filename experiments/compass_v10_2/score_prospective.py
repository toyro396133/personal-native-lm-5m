"""COMPASS V10.2 — offline prospective blinded four-arm scoring (stdlib only).

No model/API calls. Raw prompts, responses and private project evidence stay
outside this public repo. Never interpret engineered test outcomes as real user
evaluation. This scorer consumes scores from at least two independent humans
who had no access to arm identities before submitting ratings.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime,timezone
from math import comb
import argparse
import json
from pathlib import Path

ARM_TYPES=("N","F","R","C")
FIELDS=("goal_alignment","evidence_fidelity","continuation_value",
        "uncertainty_calibration","unsupported_claims")
LOCKED_SHA="c7f972e385bf24489819402938dfbfa2baa988ef"
FREEZE=datetime.fromisoformat("2026-10-11T23:30:16+00:00")

def parsed_time(s: str)->datetime:
    if not isinstance(s,str):raise ValueError("Timestamp is missing")
    d=datetime.fromisoformat(s.replace("Z","+00:00"))
    if d.tzinfo is None:raise ValueError("Timezone must be explicit")
    return d.astimezone(timezone.utc)

def exact_sign_p(differences: list[float])->dict:
    npos=sum(x>0 for x in differences)
    nneg=sum(x<0 for x in differences)
    ties=len(differences)-npos-nneg
    n=npos+nneg
    p=min(1.,2*sum(comb(n,k) for k in range(min(npos,nneg)+1))/(2**n)) if n else 1.0
    return {"positive":npos,"negative":nneg,"ties":ties,
            "non_ties":n,"two_sided_p":round(p,8)}

def aggregate(records: dict)->dict:
    if records.get("schema")!="COMPASS_V10_2_PROSPECTIVE_V1":
        raise ValueError("Unknown review schema")
    if records.get("preregistered_protocol_sha")!=LOCKED_SHA:
        raise ValueError("Protocol hash mismatch")
    mapping=records.get("unblinding_arm_map",{})
    if not isinstance(mapping,dict) or len(mapping)!=4 or set(mapping.values())!=set(ARM_TYPES):
        raise ValueError("Arm map must be four distinct randomized blind codes")
    codes=set(mapping)
    if any(not isinstance(x,str) or len(x)>64 or not x for x in codes):
        raise ValueError("Invalid blinded arm IDs")
    tasks=records.get("tasks")
    if not isinstance(tasks,list):raise ValueError("tasks must be an array")
    seen=set();out=[];workstreams=set()
    for x in tasks:
        taskid=x.get("id")
        if not isinstance(taskid,str) or not taskid or taskid in seen:
            raise ValueError("Missing or duplicate task ID")
        seen.add(taskid)
        if parsed_time(x.get("created_at"))<=FREEZE:
            raise ValueError("Task predates frozen protocol")
        if (not x.get("fresh_independent_sessions") or
            not x.get("task_and_evidence_adjudicated") or
            not x.get("arm_id_hidden_until_ratings_finalized")):
            raise ValueError("Missing prospective independence or adjudication attestation")
        scope=x.get("workstream")
        if not isinstance(scope,str) or not scope.strip():
            raise ValueError("Missing workstream")
        workstreams.add(scope.strip())
        budgets=x.get("context_chars")
        if not isinstance(budgets,dict) or set(budgets)!=codes:
            raise ValueError("All four context character counts are required")
        if any(type(n) is not int or n<0 or n>5000 for n in budgets.values()):
            raise ValueError("Matched context budget exceeded")
        raters=x.get("ratings")
        if not isinstance(raters,list) or len(raters)<2:
            raise ValueError("Two independent reviewers required")
        reviewer_ids=set()
        byarm=defaultdict(list)
        for reviewer in raters:
            person=reviewer.get("reviewer_id")
            if not isinstance(person,str) or not person or person in reviewer_ids:
                raise ValueError("Duplicate or missing reviewer")
            reviewer_ids.add(person)
            scores=reviewer.get("scores")
            if not isinstance(scores,dict) or set(scores)!=codes:
                raise ValueError("Reviewer must score all blind arms")
            for blind,vals in scores.items():
                if not isinstance(vals,dict) or set(vals)!=set(FIELDS):
                    raise ValueError("Missing frozen scoring fields")
                if any(type(value) is not int or not 0<=value<=4 for value in vals.values()):
                    raise ValueError("Scores must be integers from 0 to 4")
                byarm[mapping[blind]].append(vals)
        score={}
        for arm in ARM_TYPES:
            sample=byarm[arm]
            means={f:sum(a[f] for a in sample)/len(sample) for f in FIELDS}
            utility=sum(means[f] for f in FIELDS[:-1])-2*means["unsupported_claims"]
            score[arm]={"utility":utility,
                "unsupported":means["unsupported_claims"],
                "severe_unsupported":means["unsupported_claims"]>=3}
        out.append({"id":taskid,"workstream":scope,"by_arm":score,
                    "C_minus_R":score["C"]["utility"]-score["R"]["utility"]})
    differences=[x["C_minus_R"] for x in out]
    sign=exact_sign_p(differences)
    means={arm:round(sum(t["by_arm"][arm]["utility"] for t in out)/len(out),4)
           if out else None for arm in ARM_TYPES}
    severe={arm:sum(t["by_arm"][arm]["severe_unsupported"] for t in out)
            for arm in ARM_TYPES}
    advantage=(sum(differences)/len(differences)) if differences else None
    eligible=len(out)>=12 and len(workstreams)>=3
    evidence_gate=(eligible and advantage>=2.0 and
                   sign["two_sided_p"]<0.05 and severe["C"]<=severe["R"])
    return {
        "study":"COMPASS_V10_2_INDEPENDENT_REAL_PROJECT_CONTINUATION",
        "preregistered_protocol_sha":LOCKED_SHA,
        "task_count":len(out),"workstreams":sorted(workstreams),
        "frozen_primary_contrast":"C_minus_R",
        "by_arm_mean_utility":means,
        "paired_mean_C_minus_R":round(advantage,4) if advantage is not None else None,
        "paired_sign_test":sign,"severe_unsupported_count":severe,
        "minimum_evaluable_tasks_reached":eligible,
        "locked_gate_passed":bool(evidence_gate),
        "verdict":("PROVISIONAL_ONE_PROJECT_COMPASS_ADVANTAGE" if evidence_gate else
                   "INSUFFICIENT_PROSPECTIVE_EVIDENCE" if not eligible else
                   "NO_PROVEN_ADDED_VALUE_OVER_RAG"),
        "per_task_aggregates":out,
        "limitations":"Scores are submitted human ratings. Validate reviewer blindness; not proof of cross-project transfer."
    }

def main()->None:
    a=argparse.ArgumentParser()
    a.add_argument("--ratings",required=True,type=Path)
    a.add_argument("--report",required=True,type=Path)
    args=a.parse_args()
    source=json.loads(args.ratings.read_text(encoding="utf8"))
    result=aggregate(source)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({k:result[k] for k in
                      ("task_count","paired_mean_C_minus_R","paired_sign_test","verdict")},
                     ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
