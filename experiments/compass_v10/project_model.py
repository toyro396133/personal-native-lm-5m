#!/usr/bin/env python3
"""COMPASS V10 — read-only source-scoped project model interpreter.

This generic core reads a LOCAL project model bundle (not included in this
public repository). It does not extract documents, train weights, use an API,
or approve/modify any user or project principle.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

ALLOWED={"SUPPORT","CONFLICT","ABSTAIN"}
EARLY="55b0ac175513"

def verified(bundle:dict)->tuple[dict, list]:
    if bundle.get("model")!="COMPASS_V10_PROJECT_UNDERSTANDING_PROTOTYPE":
        raise ValueError("Unrecognized model version")
    if bundle.get("scope")!="chacham-lev" or bundle.get("early_source_commit")!=EARLY:
        raise ValueError("Project mismatch or historical source cutoff changed")
    if bundle.get("user_approved_memory_updates")!=0:
        raise ValueError("Local snapshot must not claim automatically approved updates")
    claims=bundle.get("claims",[])
    by_id={}
    for claim in claims:
        ident=claim.get("id")
        if not isinstance(ident,str) or ident in by_id or not re.fullmatch(r"G6-[0-9]{3}",ident):
            raise ValueError("Invalid/duplicate claim identifier")
        if claim.get("status")!="TEACHER_CANDIDATE_NOT_VERIFIED":
            raise ValueError("Inferred teacher data presented as verified")
        if not claim.get("source","").startswith("git:"+EARLY+":docs/"):
            raise ValueError("Later/private source leaked into earlier model")
        lines=claim.get("source_lines",[])
        if len(lines)!=2 or lines[0]<=0 or lines[1]<lines[0]:
            raise ValueError("Missing line range")
        by_id[ident]=claim
    if len(by_id)==0:
        raise ValueError("Empty model")
    rules=bundle.get("goal_dependencies",[])
    if not isinstance(rules,list):
        raise ValueError("Invalid rules")
    for rule in rules:
        if rule.get("outcome") not in ALLOWED or not rule.get("requires"):
            raise ValueError("Invalid rule")
        if not rule.get("evidence") or any(x not in by_id for x in rule["evidence"]):
            raise ValueError("Source trace missing")
        if rule["outcome"] in ("SUPPORT","CONFLICT") and not any(
                by_id[k]["label"]=="APPROVED" for k in rule["evidence"]):
            raise ValueError("Normative claim has no documented product principle")
        if rule["outcome"]=="ABSTAIN" and not any(
                by_id[k]["label"]=="OPEN" for k in rule["evidence"]):
            raise ValueError("Uncertain case has no documented unresolved policy")
    return by_id,rules

def assess(bundle:dict,signals:list[str],*,project_id:str="chacham-lev")->dict:
    if project_id!="chacham-lev":
        raise ValueError("No automatic cross-project or personal scope transfer")
    facts,rules=verified(bundle)
    matches=[r for r in rules if set(r["requires"]).issubset(signals)]
    if not matches:
        return {"outcome":"ABSTAIN","because":["The historical model has no grounded policy for this proposed change."],
                "source_refs":[],"teacher_interpretation_only":True,
                "requires_human_review":True,"autoapproved":False}
    u=[r for r in matches if r["outcome"]=="ABSTAIN"]
    c=[r for r in matches if r["outcome"]=="CONFLICT"]
    s=[r for r in matches if r["outcome"]=="SUPPORT"]
    outcome="ABSTAIN" if u or (c and s) else "CONFLICT" if c else "SUPPORT"
    selected=u or (matches if c and s else c or s)
    keys=list(dict.fromkeys(k for r in selected for k in r["evidence"]))
    return {"outcome":outcome,
            "because":[r["rationale"] for r in selected],
            "matched_rules":[r["id"] for r in selected],
            "source_refs":[{"id":k,"text":facts[k]["claim"],
                            "source":facts[k]["source"],"lines":facts[k]["source_lines"]} for k in keys],
            "teacher_interpretation_only":True,
            "requires_human_review":True,"autoapproved":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",type=Path,required=True,help="LOCAL project model JSON, never commit private user data")
    p.add_argument("--signal",action="append",required=True)
    p.add_argument("--project",default="chacham-lev")
    args=p.parse_args()
    bundle=json.loads(args.model.read_text(encoding="utf-8"))
    print(json.dumps(assess(bundle,args.signal,project_id=args.project),ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
