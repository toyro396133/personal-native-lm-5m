"""COMPASS V8.1: matched single-token categorical preference classification.

Smoke-only study. Trains pairs of fictional situations with different actions,
then tests wholly unseen fictional project and policy profile identifiers.
Primary scores exclude deterministic guards; guarded scores are separate and
must NOT be represented as learned project-reasoning ability.
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM,AutoTokenizer

from run import BASE,PersonalAdapter,attach
from fixtures_v81 import generate,ACTIONS
from validate_v81 import audit,PAIRS
from guard_v81 import guard_action

CODES={a:str(i) for i,a in enumerate(ACTIONS)}
NEUTRAL_PROMPT=(
  "Choose one decision ID for a generic unspecified project task. "
  "Use only one digit. 0 DO_NOW, 1 SCHEDULE, 2 INVESTIGATE, "
  "3 DEFER, 4 ASK_USER, 5 REJECT. Decision ID:"
)

def prompt_for(row):
    p=row["project"];t=row["task"];policy=row["policy"]
    node=next(n for n in p["nodes"] if n["id"]==t["node_id"])
    policy_text=policy["text"]
    d="\n".join(f"{r['id']} [{r['source']}]: {r['statement']}"
                for r in p["decisions"] if r["valid"])
    return (
      "Select the next approved action according to the authorized project "
      "objective, the actual evidence and this fictional user's decision policy. "
      "An ancillary task can be urgent when required to repair the core; "
      "unverified reports call for investigation, not confident execution. "
      "Answer with one DIGIT only:\n"
      "0 = DO_NOW\n1 = SCHEDULE\n2 = INVESTIGATE\n3 = DEFER\n"
      "4 = ASK_USER\n5 = REJECT\n"
      f"Project: {p['id']} (version {p['version']})\n"
      f"Objective: {p['objective']}\nPhase: {p['phase']}\n"
      f"Current authorized rules:\n{d}\n"
      f"Personal preference: {policy_text} [source {policy['id']}, version {policy['version']}]\n"
      f"Task: {t['proposal']}\n"
      f"Task's structural layer: {node['layer']}\n"
      f"Task node intent: {node['intent']}\n"
      f"Evidence status: {t['evidence_status']}\n"
      f"Core currently blocked by this task: {str(t['blocked_core']).lower()}\n"
      f"Scope status: {t['scope_status']}\n"
      f"Requires owner approval: {str(t['requires_owner_approval']).lower()}\n"
      f"Owner approval: {t['approval_status']}\n"
      "Decision ID:"
    )

def prefix_ids(tok,s):
    tokens=tok.apply_chat_template(
        [{"role":"user","content":s}],tokenize=True,add_generation_prompt=True)
    if len(tokens)>750:raise ValueError(f"Context exceeds 750 token ceiling: {len(tokens)}")
    return torch.tensor([tokens],dtype=torch.long)

def action_token_ids(tok):
    ids=[tok.encode(CODES[a],add_special_tokens=False) for a in ACTIONS]
    if not all(len(t)==1 for t in ids) or len({t[0] for t in ids})!=6:
        raise ValueError(f"Six chosen categorical codes are not distinct single tokens: {ids}")
    return [t[0] for t in ids]

def score_next(model,ids,code_token_ids):
    logits=model(input_ids=ids).logits[0,-1,:].float()
    return logits[code_token_ids]

def choose(scores):
    return ACTIONS[int(torch.argmax(scores).item())]

def contrastive_training_plan(inputs,gold,seed,updates):
    rng=random.Random(seed)
    grouped={}
    for row in inputs:
        grouped.setdefault(row["project"]["id"],{})[gold[row["task"]["id"]]["case_kind"]]=row
    groups=list(grouped.values())
    assert len(groups)==16 and all(set(g).issuperset({c for pair in PAIRS for c in pair}) for g in groups)
    pair_types=list(PAIRS)
    rng.shuffle(pair_types)
    samples=[]
    for i in range(updates):
        pair=pair_types[i%len(pair_types)]
        relevant=[g for g in groups]
        rng.shuffle(relevant)
        selected=relevant[(i//len(pair_types))%len(relevant)]
        first,second=selected[pair[0]],selected[pair[1]]
        assert gold[first["task"]["id"]]["action"]!=gold[second["task"]["id"]]["action"]
        if rng.randrange(2):first,second=second,first
        samples.append((first,second))
    return samples

def per_class(gold,chosen):
    out={}
    for a in ACTIONS:
        inds=[i for i,g in enumerate(gold) if g==a]
        out[a]={"support":len(inds),"correct":sum(chosen[i]==a for i in inds),
                "predicted":sum(v==a for v in chosen)}
        out[a]["recall"]=out[a]["correct"]/len(inds) if inds else None
    recalls=[x["recall"] for x in out.values() if x["recall"] is not None]
    return {"correct":sum(a==b for a,b in zip(gold,chosen)),
            "total":len(gold),
            "balanced_accuracy":sum(recalls)/len(recalls),
            "unique_actions_predicted":len(set(chosen)),
            "histogram":dict(Counter(chosen)),"by_class":out}

def evaluate_pairs(records,mode):
    # All six pair kinds are tested from the same held-out project's task graph.
    kinds={}
    for row in records:
        kinds.setdefault(row["project_id"],{})[row["case_kind"]]=row
    result={}
    for left,right in PAIRS:
        total=0;correct=0
        for project in kinds.values():
            a=project[left];b=project[right]
            total+=1
            if a["predicted"][mode]==a["gold"] and b["predicted"][mode]==b["gold"]:
                correct+=1
        result[left+"__"+right]={"correct":correct,"total":total}
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rank",type=int,choices=(4,16,64),default=4)
    p.add_argument("--seed",type=int,choices=(11,19,37),default=11)
    p.add_argument("--updates",type=int,default=48)
    p.add_argument("--eval-projects",type=int,default=2)
    p.add_argument("--output",default="compass-v81-r4-s11.json")
    a=p.parse_args()
    if a.updates<12 or a.eval_projects not in (2,4,6):p.error("At least 12 updates; eval-projects 2/4/6")
    pre=audit()
    random.seed(a.seed);torch.manual_seed(a.seed);torch.set_num_threads(2)
    start=time.monotonic()
    inputs,golds=generate()
    train_gold={x["task_id"]:x for x in golds["train"]}
    plan=contrastive_training_plan(inputs["train"],train_gold,a.seed,a.updates)
    tok=AutoTokenizer.from_pretrained(BASE)
    code_ids=action_token_ids(tok)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,a.rank)
    n=sum(x.numel() for x in adapter.parameters())
    assert n==2*model.config.hidden_size*a.rank
    # Publicly specified fixed neutral calibration from the UNADAPTED backbone.
    with torch.no_grad():
        neutral=score_next(model,prefix_ids(tok,NEUTRAL_PROMPT),code_ids).detach()
    # Sanity control: a freshly zero-residual adapter must reproduce the frozen model.
    check=prefix_ids(tok,prompt_for(inputs["test"][0]))
    with torch.no_grad():
        unadapted=score_next(model,check,code_ids)
        h=attach(model,adapter,1)
        try:no_op=score_next(model,check,code_ids)
        finally:h.remove()
    if not torch.allclose(unadapted,no_op,atol=1e-5):
        raise AssertionError("Zero-residual adapter changed logits before training")

    optim=torch.optim.AdamW(adapter.parameters(),lr=0.0005)
    losses=[]
    h=attach(model,adapter,1)
    try:
        for step,(first,second) in enumerate(plan,1):
            optim.zero_grad(set_to_none=True)
            per_example=[]
            for row in (first,second):
                target=ACTIONS.index(train_gold[row["task"]["id"]]["action"])
                logits=score_next(model,prefix_ids(tok,prompt_for(row)),code_ids)
                # Cross-entropy over the SIX fixed labels: identical train and eval task.
                loss=nn.functional.cross_entropy(logits.unsqueeze(0),
                    torch.tensor([target],dtype=torch.long))
                per_example.append(loss)
            avg=sum(per_example)/2
            if not torch.isfinite(avg):raise RuntimeError(f"nonfinite loss step {step}")
            avg.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            optim.step()
            losses.append(round(avg.detach().item(),5))
            if step%12==0:print(f"training rank={a.rank} seed={a.seed} {step}/{a.updates} loss={losses[-1]:.4f}",flush=True)
    finally:h.remove()

    project_ids=sorted({r["project"]["id"] for r in inputs["test"]})[:a.eval_projects]
    held=[(r,g) for r,g in zip(inputs["test"],golds["test"])
          if r["project"]["id"] in project_ids]
    assert len(held)==12*a.eval_projects
    records=[]
    for row,gold in held:
        ids=prefix_ids(tok,prompt_for(row))
        with torch.no_grad():
            frozen=score_next(model,ids,code_ids)
            h=attach(model,adapter,1)
            try:learned=score_next(model,ids,code_ids)
            finally:h.remove()
        choices={
            "frozen_raw":choose(frozen),"compass_raw":choose(learned),
            "frozen_calibrated":choose(frozen-neutral),
            "compass_calibrated":choose(learned-neutral),
        }
        # The authorization safety gate is evaluated SEPARATELY from the neural predictions.
        protected={k+"_guarded":guard_action(row,v)["action"] for k,v in choices.items()}
        choices.update(protected)
        records.append({"id":row["task"]["id"],"project_id":row["project"]["id"],
                        "case_kind":gold["case_kind"],"gold":gold["action"],
                        "layer":gold["layer"],"predicted":choices,
                        "raw_logits":{"frozen":frozen.tolist(),"compass":learned.tolist()},
                        "authorization_guard":guard_action(row,choices["compass_raw"])})
    mode_names=list(records[0]["predicted"])
    result={}
    for mode in mode_names:
        g=[r["gold"] for r in records]
        p=[r["predicted"][mode] for r in records]
        measures=per_class(g,p)
        measures["contrast_pairs"]=evaluate_pairs(records,mode)
        measures["forbidden_or_unapproved_as_action"]=sum(
            r["predicted"][mode] in ("DO_NOW","SCHEDULE")
            for r in records if r["case_kind"] in ("pending_change","out_scope_expansion"))
        result[mode]=measures
    freq=Counter(r["gold"] for r in records)
    majority=max(ACTIONS,key=lambda x:freq[x])
    result["constant_majority"]={**per_class(
        [r["gold"] for r in records],[majority]*len(records)),"label":majority}
    report={
        "experiment":"COMPASS V8.1 preliminary balanced paired single-token classifier",
        "seed":a.seed,"rank":a.rank,"parameter_count":n,"updates":a.updates,
        "selected_test_projects":project_ids,"test_items":len(records),
        "training_examples_per_step":2,"token_codes":CODES,
        "token_ids":dict(zip(ACTIONS,code_ids)),
        "shared_neutral_calibration_logits":neutral.tolist(),
        "preflight":pre,"time_sec":round(time.monotonic()-start,1),
        "training_loss":losses,"results":result,"predictions":records,
        "limitations":[
            "Fictional templated data, not independently human-adjudicated",
            "Classifier chooses one of six hardcoded categories, not free-form plan with grounded rationale",
            "Deterministic authorization enforcement is separate and not neural intelligence",
            "Single small seed/rank run cannot justify deployment or claims about rank capacity",
            "Only controlled paired scenario differences, not changing live project maps or revising memories"
        ]
    }
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_file({k:v.detach().contiguous().cpu() for k,v in adapter.state_dict().items()},
              str(path.with_suffix(".safetensors")))
    for k in ("frozen_raw","compass_raw","frozen_calibrated","compass_calibrated",
              "compass_raw_guarded"):
        s=result[k]
        print(f"EVAL {k} correct={s['correct']}/{s['total']} "+
              f"balanced={s['balanced_accuracy']:.3f} classes={s['unique_actions_predicted']} "+
              f"guard_violations={s['forbidden_or_unapproved_as_action']}",flush=True)
    print(f"ARTIFACT {path}",flush=True)

if __name__=="__main__":main()
