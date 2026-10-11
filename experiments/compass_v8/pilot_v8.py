"""COMPASS V8 Stage-1 rank-4 single-seed CPU pilot.

The stage-0 schema/trap audit MUST succeed before this job runs. 32 selected
supervised updates, 16 held-out decisions from unseen projects/wording, 3
candidate-score arms. Frozen language model + optional rank-4 personal policy
adapter, trained only on separate gold/train.jsonl actions.
Caution: forced-choice token losses are not free-form planning quality.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import BASE, PersonalAdapter, attach, encode_pair
from fixtures_v8 import ACTIONS, generate
from validate_v8 import assert_dataset
from baseline_v8 import evaluate as score_rules

def user_task_prompt(context,show_policy):
    project=context["project"]
    task=context["task"]
    node=next(n for n in project["nodes"] if n["id"]==task["node_id"])
    decisions="\n".join(
        "- "+d["id"]+": "+d["statement"]
        for d in project["decisions"] if d["valid"]
    )
    policy=(context["personal_policy"]["approved_guidance"]
            if show_policy else "(no user-specific policy provided)")
    return (
        "You are making a project-priority decision. Pick the next action using "
        "the project core, scope, evidence and task dependencies. "
        "Do not treat ancillary location as automatic low urgency. "
        "Choose exactly ONE code: DO_NOW, SCHEDULE, INVESTIGATE, DEFER, ASK_USER, REJECT.\n"
        "Project: "+project["id"]+"\n"
        "Goal: "+project["objective"]+"\n"
        "Phase: "+project["phase"]+"\n"
        "Task's structural layer: "+node["layer"]+"\n"
        "Task component: "+node["intent"]+"\n"
        "Current project rules:\n"+decisions+"\n"
        "Personal decision policy: "+policy+"\n"
        "Task evidence: "+task["proposal"]+"\n"
        "Decision code:"
    )

def make_selected_train(inputs):
    # Every user and every authored case is represented once, but from alternating
    # train projects. This is a deterministic selected 32-of-64 pilot subset.
    assert len(inputs)==64
    plan=[]
    for pi in range(4):
        for case_index in range(8):
            train_project=case_index%2
            plan.append(inputs[pi*16 + train_project*8 + case_index])
    assert len(plan)==32
    return plan

def maybe_attach(model,adapter):
    return attach(model,adapter,1) if adapter is not None else None

def action_logloss(tok,model,prompt,action):
    ids,labels=encode_pair(tok,prompt,action,512)
    return model(input_ids=ids,labels=labels).loss

@torch.no_grad()
def decide_options(tok,model,prompt):
    # Six exact candidate responses, all using the same prompt and EOS.
    losses={code:float(action_logloss(tok,model,prompt,code)) for code in ACTIONS}
    winner=min(ACTIONS,key=lambda c:losses[c])
    return winner,losses

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,default=11)
    parser.add_argument("--steps",type=int,default=32)
    parser.add_argument("--output",default="compass-v8-pilot.json")
    args=parser.parse_args()
    assert_dataset()
    if args.steps<8:raise ValueError("Need at least 8 updates for pilot")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    start=time.monotonic()
    ins,golds=generate()
    labels={g["task_id"]:g for g in golds["train"]}
    plan=make_selected_train(ins["train"])
    # Fixed pilot evaluation: four adversarial cases per held-out project.
    keep={"confirmed_break","broad_test","targeted_test","suspected_issue"}
    held=[(x,y) for x,y in zip(ins["test"],golds["test"]) if y["case_kind"] in keep]
    assert len(held)==16 and all(x["project"]["id"].endswith("-planning") for x,_ in held)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,4)
    n=sum(p.numel() for p in adapter.parameters())
    assert n==7680
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.0005)
    training_losses=[]
    handle=attach(model,adapter,1)
    try:
        for step in range(args.steps):
            record=plan[step%len(plan)]
            target=labels[record["task"]["id"]]["action"]
            prompt=user_task_prompt(record,show_policy=False)
            optimizer.zero_grad(set_to_none=True)
            loss=action_logloss(tok,model,prompt,target)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite pilot loss")
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            optimizer.step()
            training_losses.append(round(float(loss),5))
            if (step+1)%8==0:
                print(f"COMPASS pilot rank4 step={step+1}/{args.steps} loss={float(loss):.4f}",flush=True)
    finally:
        handle.remove()

    predictions=[]
    modes=("frozen_graph_only","frozen_graph_policy",
           "adapter_graph_only","adapter_graph_policy")
    for row,gold in held:
        decisions={}
        losses={}
        for mode in modes:
            learned=mode.startswith("adapter_")
            with_policy=mode.endswith("_policy")
            prompt=user_task_prompt(row,show_policy=with_policy)
            h=attach(model,adapter,1) if learned else None
            try:
                action,values=decide_options(tok,model,prompt)
            finally:
                if h:h.remove()
            decisions[mode]=action
            losses[mode]=values
        predictions.append({
            "id":row["task"]["id"],"project_id":row["project"]["id"],
            "actual_case_kind":gold["case_kind"],"gold":gold["action"],
            "structural_layer":gold["layer"],"blocked_core":gold["blocked_core"],
            "predictions":decisions,"candidate_losses":losses,
        })
        print(f"evaluate {len(predictions)}/{len(held)} expected={gold['action']} "+ " ".join(
            m+"="+decisions[m] for m in modes),flush=True)
    correct={m:sum(x["predictions"][m]==x["gold"] for x in predictions) for m in modes}
    hardcases=[x for x in predictions if x["actual_case_kind"] in
               {"targeted_test","broad_test"}]
    traps={m:sum(x["predictions"][m]==x["gold"] for x in hardcases) for m in modes}
    rules=score_rules([a for a,_ in held],[b for _,b in held])
    report={
        "kind":"V8 rank4 stage1 pilot, not the full rank-by-seed evaluation",
        "seed":args.seed,"train_examples_selected":len(plan),
        "training_updates":args.steps,"test_examples":len(held),
        "trainable_parameters":n,"base":BASE,"time_seconds":round(time.monotonic()-start,1),
        "train_losses":training_losses,
        "correct":correct,"trap_correct":traps,"trap_total":len(hardcases),
        "rule_baseline":{"correct":rules["correct"],"total":rules["total"]},
        "predictions":predictions,
        "limitations":[
            "Only one rank, one seed, 32 supervised updates, 16 held-out test situations",
            "Same four fictional profiles are present in training and test",
            "Project kind confounded with test split and authored sentence templates",
            "Model is scored using 6 forced-choice candidate log-losses; no verified JSON/rationale generation",
            "No independent human label adjudication or project-level interval estimates yet",
            "Gold labels are only read from separate gold train/test arrays; none in model context"
        ],
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
              str(out.with_suffix(".safetensors")))
    print("PILOT RESULTS",json.dumps({k:report[k] for k in
        ("correct","trap_correct","test_examples","time_seconds")}),flush=True)

if __name__=="__main__":main()
