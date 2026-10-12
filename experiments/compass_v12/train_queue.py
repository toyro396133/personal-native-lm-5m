"""COMPASS V12 — run PRIVATE reviewed V11.1 feedback on a frozen model.

The input queue must come from from_v11_feedback.py or an equivalent trusted
manual review. Weights update for exactly one user/project scope per event.
This is training-on-known-correction pairs, NOT evaluation on future decisions.
"""
from __future__ import annotations

import copy
import re
from typing import Callable

import torch

from detachable_layer import (
    PersonalProjectState,FrozenResidualBridge,check_id,check_source,
    personalized_update,pair_margins
)

QUEUE_SCHEMA="COMPASS_V12_PRIVATE_TRAINING_QUEUE_V1"
PAIR_SCHEMA="COMPASS_V12_TRAINING_PAIR_NOT_EVALUATION"

def validate_queue(q:dict,bank:PersonalProjectState,bridge:FrozenResidualBridge)->None:
    if not isinstance(q,dict) or q.get("schema") != QUEUE_SCHEMA or q.get("evaluation_pairs") != 0:
        raise ValueError("Only user reviewed training queue is allowed; no evaluation-label leakage")
    uid=check_id(q.get("user_id"))
    entries=q.get("pairs")
    if not isinstance(entries,list) or len(entries)>300:
        raise ValueError("At most 300 reviewed feedback entries per call")
    if bank.rank!=bridge.rank:
        raise ValueError("COMPASS memory rank mismatches bridge")
    found=set()
    for row in entries:
        if not isinstance(row,dict) or row.get("schema") != PAIR_SCHEMA:
            raise ValueError("Malformed historical feedback pair")
        if row.get("user_id")!=uid or row.get("chosen_index")!=1:
            raise ValueError("Training must use original user-reviewed candidate order")
        if row.get("prospective_holdout_eligible") is not False or row.get("project_policy_approved") is not False:
            raise ValueError("Historical corrections are not prospective gold/approved policy")
        if row.get("review_status") not in ("USER_CONFIRMED","LITERAL_EXPLICIT_USER_CORRECTION"):
            raise ValueError("Training pair not grounded in a direct user correction")
        if row.get("scope") not in ("personal","project"):
            raise ValueError("Unknown scope")
        project=row.get("project_id")
        if row["scope"]=="project":
            check_id(project)
        elif project not in (None,""):
            raise ValueError("Project data cannot automatically transfer to personal memory")
        source=check_source(row.get("source"))
        key=check_id(row.get("event_id"))
        inputs=row.get("candidates")
        task=row.get("question")
        if (not isinstance(task,str) or not task.strip() or len(task)>2500 or
            not isinstance(inputs,list) or len(inputs)!=2 or
            any(not isinstance(v,str) or not v.strip() or len(v)>2500 for v in inputs) or
            inputs[0]==inputs[1]):
            raise ValueError("Feedback requires original task and two different candidate texts")
        feedback=row.get("feedback")
        if (not isinstance(feedback,dict) or feedback.get("event_id")!=key or
            feedback.get("source")!=source or feedback.get("speaker")!="user" or
            feedback.get("confirmed_by_user") is not True or
            feedback.get("source_verified") is not True):
            raise ValueError("Feedback must be source-linked and explicitly user-reviewed")
        index=f"{row['scope']}:{uid}:{project}:{key}"
        if index in found or index in bank.event_ids:
            raise ValueError("Duplicate/replayed user review event")
        found.add(index)


def train_queue(model,bridge:FrozenResidualBridge,bank:PersonalProjectState,
                queue:dict, *, encode:Callable[[str,str],torch.Tensor],
                yes:int,no:int,steps:int=6,lr:float=0.15)->dict:
    """Prevalidate entire batch and commit state only if every update succeeds."""
    validate_queue(queue,bank,bridge)
    # No actual project policy approval takes place. Work in detached clone to
    # ensure a later malformed/failed example cannot partially corrupt memory.
    working=PersonalProjectState.from_data(copy.deepcopy(bank.to_data()))
    logs=[]
    for row in queue["pairs"]:
        candidates=[encode(row["question"],candidate) for candidate in row["candidates"]]
        uid,pid=row["user_id"],row["project_id"]
        before_u,before_p=working.latent(uid,pid)
        with torch.no_grad():
            m0=pair_margins(model,bridge,before_u,before_p,candidates,yes,no)
        result=personalized_update(
            model,bridge,working,user_id=uid,
            project_id=pid if row["scope"]=="project" else None,
            scope=row["scope"],candidate_ids=candidates,chosen_index=1,
            yes_token_id=yes,no_token_id=no,feedback=row["feedback"],
            steps=steps,lr=lr)
        after_u,after_p=working.latent(uid,pid)
        with torch.no_grad():
            m1=pair_margins(model,bridge,after_u,after_p,candidates,yes,no)
        logs.append({
            "event_id":row["event_id"],"scope":row["scope"],
            "project_id":row["project_id"],"source":row["source"],
            "state_delta_l2":result["state_delta_l2"],
            "pair_margin_before":float((m0[1]-m0[0]).detach()),
            "pair_margin_after":float((m1[1]-m1[0]).detach()),
            "project_policy_approved":False
        })
    bank.users,bank.projects=working.users,working.projects
    bank.events,bank.event_ids=working.events,working.event_ids
    return {"status":"TRAINED_ONLY_ON_PRIOR_USER_FEEDBACK",
            "user_id":queue["user_id"],
            "updates":len(logs),"events":logs,
            "authoritative_project_decisions_changed":False,
            "model_backbone_trained":False,"bridge_trained":False,
            "new_unseen_decisions_evaluated":0,
            "conclusion":"Online personal-state update achieved; no semantic inference proven."}
