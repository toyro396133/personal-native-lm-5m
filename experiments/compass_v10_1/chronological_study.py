"""V10.1 retrospective project-state replay — SANDBOX ONLY.

A private, user-local teacher-proposed project history can be replayed through
the genuine V9.12 approval journal in a DISPOSABLE temporary SQLite database.
The "approver" is named SYNTHETIC_SIMULATION_NOT_REAL_USER and every output is
marked as simulated. This NEVER grants a production memory approval.
"""
from __future__ import annotations
import argparse
import json
import tempfile
from pathlib import Path

from compass_core import CompassStore
from learning_loop import propose,model_snapshot,compile_brief

STAGES=("A_EARLY","B_INTERMEDIATE","C_CORRECTIONS")
REVIEWER="SYNTHETIC_SIMULATION_NOT_REAL_USER"
def replay(input_doc: dict, *,project_id:str="chacham-lev") -> dict:
    if input_doc.get("simulation_only") is not True or input_doc.get("autoapproval") is not False:
        raise ValueError("Refusing any dataset not explicitly marked for simulation-only")
    items=input_doc.get("candidates")
    if not isinstance(items,list) or not items or len(items)>200:
        raise ValueError("Study requires 1..200 validated proposals")
    if input_doc.get("count")!=len(items):
        raise ValueError("Missing/stale study case count")
    if sorted(set(x.get("phase") for x in items))!=sorted(STAGES):
        raise ValueError("Missing stages or unrecognized phase")
    expected = {"phone.route_identity", "phone.assignment_authority", "phone.daily_workflow"}
    snapshots={}; summaries={}; simulated_approvals=0
    with tempfile.TemporaryDirectory(prefix="compass-v101-SIMULATION-") as folder:
        db=Path(folder)/"disposable-not-user-memory.sqlite3"
        for phase in STAGES:
            phase_items=[x for x in items if x["phase"]==phase]
            for record in phase_items:
                if record.get("review_state")!="GPT6_TEACHER_PROPOSAL_NOT_USER_APPROVED":
                    raise ValueError("Real user approvals cannot be simulated by this utility")
                claim=record["record"]
                result=propose(db,claim,project_id=project_id)
                if result["approved"] is not False:
                    raise RuntimeError("The V10.1 staging layer unexpectedly autoapproved")
                with CompassStore(db) as store:
                    store.approve(result["proposal_id"],actor=REVIEWER,
                                  reason="DISPOSABLE REPLAY ONLY; GPT-6 teacher label is NOT human approval")
                simulated_approvals+=1
            snap=model_snapshot(db,project_id)
            version_by_facet={x["facet"]:x["revision"]
                              for k,rows in snap["kinds"].items() for x in rows}
            state_by_facet={x["facet"]:x["claim"]
                              for rows in snap["kinds"].values() for x in rows}
            snapshots[phase]={"active_count":snap["approved_active_count"],
                              "pending_count":snap["pending_count"],
                              "facet_revisions":version_by_facet,
                              "active_text_by_facet":state_by_facet,
                              "historical_version_count":len(snap["history"]),
                              "missing_dependencies":snap["missing_dependencies"],
                              "integrity_ok":snap["audit"]["ok"]}
            summaries[phase]=compile_brief(db,project_id,task="What is the evolving project purpose and what changed?")
        first=snapshots["A_EARLY"];mid=snapshots["B_INTERMEDIATE"];last=snapshots["C_CORRECTIONS"]
        errors=[]
        # This is an ENGINEERING replay check. It does not score model cognition.
        if any(f in first["facet_revisions"] for f in expected):
            errors.append("Future telephone decisions leaked into earliest snapshot")
        for f in expected:
            if mid["facet_revisions"].get(f)!=1 or last["facet_revisions"].get(f)!=2:
                errors.append("Missing simulated revision continuity for "+f)
            if mid["active_text_by_facet"].get(f)==last["active_text_by_facet"].get(f):
                errors.append("Version update did not alter the active understanding for "+f)
        if first["active_text_by_facet"].get("goal.reuse")!=last["active_text_by_facet"].get("goal.reuse"):
            errors.append("Project WHY was erased by later subsystem edits")
        if last["historical_version_count"]!=len(items):
            errors.append("Lost a historical version of a simulated project decision")
        if any(not s["integrity_ok"] for s in snapshots.values()):
            errors.append("An event-journal integrity check failed")
        if last["pending_count"]!=0:
            errors.append("Residual proposed event was not simulated correctly")
        return {
           "study":"COMPASS_V10_1_RETROSPECTIVE_SIMULATION",
           "source_labels":"GPT6_TEACHER_UNVERIFIED",
           "independent_human_review":False,
           "source_project":project_id,
           "disposable_synthetic_approvals":simulated_approvals,
           "approved_in_real_user_memory":0,
           "persisted_private_project_database":False,
           "stages":snapshots,
           "replay_invariants_passed":not errors,
           "invariant_errors":errors,
           "context_delta":{"early_facets":len(first["facet_revisions"]),
                            "intermediate_facets":len(mid["facet_revisions"]),
                            "later_facets":len(last["facet_revisions"]),
                            "updated_concept_count":len(expected)},
           "surrogate_learning_claim":"NONE: Retrospective teacher-known correction sequence, not a blind model inference.",
           "verdict":"ENGINEERING_TEMPORAL_PROJECT_MODEL_REPLAY_ONLY"
        }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--private-input",required=True,type=Path)
    parser.add_argument("--private-report",required=True,type=Path)
    args=parser.parse_args()
    d=json.loads(args.private_input.read_text(encoding="utf8"))
    result=replay(d)
    args.private_report.parent.mkdir(parents=True,exist_ok=True)
    args.private_report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print("Retrospective sandbox replay:", result["replay_invariants_passed"])
    print("Synthetic approvals in disposable database:", result["disposable_synthetic_approvals"])
    print("Real memory approvals:",result["approved_in_real_user_memory"])
    print("Report only on local machine:",args.private_report)

if __name__=="__main__":
    main()
