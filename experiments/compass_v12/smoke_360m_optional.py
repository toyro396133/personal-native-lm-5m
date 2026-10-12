"""OPTIONAL real frozen 360M smoke: one user's detached neural COMPASS state.

This script is never called by CI, so expensive model weights are downloaded
only when the user explicitly runs it. Labels are SYNTHETIC TEST USER, not real
personal preference data. Report model logits and actual hook causality ONLY.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from detachable_layer import (
    FrozenResidualBridge, PersonalProjectState, freeze_backbone,
    personalized_update, pair_margins
)

MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
SRC = "chat:research12:feedback01"

def prompt(quest: str, candidate: str) -> str:
    return (
        "Choose the most appropriate project action considering existing goals. "
        "Evaluate whether the CANDIDATE is appropriate: answer Yes or No.\n"
        "Task: "+quest+"\nCandidate: "+candidate+"\nAnswer:"
    )

def tokenize(tok,text):
    ids=tok.apply_chat_template([{"role":"user","content":text}],
                                tokenize=True,add_generation_prompt=True)
    if not ids or len(ids)>512:
        raise ValueError("Prompt too long/empty for valid smoke")
    return torch.tensor([ids],dtype=torch.long)

def get_ids(tok):
    for a,b in (("Yes","No"),(" Yes"," No")):
        aa=tok.encode(a,add_special_tokens=False)
        bb=tok.encode(b,add_special_tokens=False)
        if len(aa)==len(bb)==1 and aa[0]!=bb[0]:
            return aa[0],bb[0]
    raise ValueError("Cannot compare distinct single-token Yes/No logits")

def sampled_backbone_checks(model):
    sd=model.state_dict()
    keys=list(sd)
    take=sorted(set(keys[:2]+keys[-2:]))
    return {name:hashlib.sha256(sd[name].detach().cpu().numpy().tobytes()).hexdigest()
            for name in take}

def run(args):
    print("COMPASS V12 OPTIONAL real-360M frozen-smoke",flush=True)
    print("Synthetic test case, NOT an actual user-approved policy.",flush=True)
    torch.set_num_threads(2)
    torch.manual_seed(12)
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32)
    model.config.use_cache=False
    freeze_backbone(model)
    yes,no=get_ids(tok)
    h=model.config.hidden_size
    bridge=FrozenResidualBridge(h,rank=args.rank,gain=args.gain)
    if args.v90_checkpoint:
        bridge.load_v90(Path(args.v90_checkpoint))
        bridge_source="V9.0 historical pretrained down/up (user-provided)"
    else:
        bridge_source="RANDOM BRIDGE ONLY; engineering demonstration, not genuine semantic transfer"
    bank=PersonalProjectState(args.rank)
    quest="A project needs continuity across years. Which action is preferable?"
    actions=[
        "Discard all historical project notes at the end of each year",
        "Retain past project findings with provenance and update them"
    ]
    prompts=[tokenize(tok,prompt(quest,a)) for a in actions]
    original_check=sampled_backbone_checks(model)
    personal,project=bank.latent("synthetic-user","fictional-project")
    with torch.no_grad():
        prior=pair_margins(model,bridge,personal,project,prompts,yes,no).tolist()
    result=personalized_update(
        model,bridge,bank,user_id="synthetic-user",project_id="fictional-project",
        scope="project",candidate_ids=prompts,chosen_index=1,
        yes_token_id=yes,no_token_id=no,
        feedback={"event_id":"feedback01","source":SRC,"speaker":"user",
                  "confirmed_by_user":True,"source_verified":True},
        steps=args.steps,lr=args.lr,retention=0.001)
    changed_check=sampled_backbone_checks(model)
    personal,project=bank.latent("synthetic-user","fictional-project")
    with torch.no_grad():
        after=pair_margins(model,bridge,personal,project,prompts,yes,no).tolist()
    if original_check!=changed_check:
        raise AssertionError("Frozen backbone CHECKED tensors unexpectedly changed!")
    report={
        "status":"V12_FROZEN_360M_LIVE_HOOK_SMOKE",
        "model":MODEL,"source":"synthetic user feedback, no real personal data",
        "adapter_initialization":bridge_source,
        "backbone_checked_tensors_unchanged":True,
        "backbone_whole_tensor_hash_verified":False,
        "bridge_trainable_weights_during_update":0,
        "personal_state_rank":args.rank,"before_yes_no_margins":prior,
        "after_yes_no_margins":after,
        "initial_choice":int(prior[1]>prior[0]),
        "updated_choice":int(after[1]>after[0]),
        "state_update":result,
        "real_project_intent_learning_proven":False,
        "cross_model_transfer_proven":False
    }
    args.report.parent.mkdir(exist_ok=True,parents=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    if args.state:
        bank.save(args.state)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--state",type=Path)
    p.add_argument("--v90-checkpoint",type=Path)
    p.add_argument("--rank",type=int,default=16)
    p.add_argument("--gain",type=float,default=1.)
    p.add_argument("--steps",type=int,default=6)
    p.add_argument("--lr",type=float,default=0.2)
    args=p.parse_args()
    run(args)

if __name__=="__main__":
    main()
