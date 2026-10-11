"""V8 diagnostic matrix: ranks 4/16/64, per-action balanced updates, common-prefix scoring.

Uses only synthetic per-project context. Test gold never enters prompts or training.
Crucial distinction: this is a controlled *forced-choice proxy*, not an ability to
reason autonomously about actual real-world projects.
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import BASE, PersonalAdapter, attach, encode_pair
from fixtures_v8 import ACTIONS, generate
from validate_v8 import assert_dataset
from pilot_v8 import user_task_prompt
from baseline_v8 import evaluate as rules_evaluate

@torch.no_grad()
def score_choices(tok,model,prompt,max_prompt_tokens=512):
    """All six candidate answers see exactly the SAME unaltered prefix.

    Scores represent P(action tokens | identical chat prefix). Scores are
    conditioned on the complete action code, without a candidate-dependent
    truncate or a candidate-dependent answer EOS/finish requirement.
    Report both length-normalized and summed action-token log-probabilities.
    """
    prefix=tok.apply_chat_template(
        [{"role":"user","content":prompt}],
        tokenize=True,add_generation_prompt=True)
    if len(prefix)>max_prompt_tokens:
        raise ValueError(f"Prompt {len(prefix)} tokens exceeds fixed {max_prompt_tokens}; no silent truncation")
    result={}
    for action in ACTIONS:
        tail=tok.encode(action,add_special_tokens=False)
        if not tail:raise ValueError(f"No token ids for action {action}")
        ids=torch.tensor([prefix+tail])
        logits=model(input_ids=ids).logits[0,len(prefix)-1:len(prefix)-1+len(tail),:]
        chosen=torch.tensor(tail).unsqueeze(-1)
        lp=torch.log_softmax(logits.float(),dim=-1).gather(1,chosen).view(-1)
        result[action]={"sum":float(lp.sum()),"mean":float(lp.mean()),"tokens":len(tail)}
    winners={metric:max(ACTIONS,key=lambda a:result[a][metric]) for metric in ("mean","sum")}
    return winners,result

def class_balanced_plan(records,gold_map,seed,steps):
    buckets={a:[] for a in ACTIONS}
    for row in records:
        buckets[gold_map[row["task"]["id"]]["action"]].append(row)
    if any(not values for values in buckets.values()):
        raise ValueError({k:len(v) for k,v in buckets.items()})
    rng=random.Random(seed)
    for values in buckets.values():rng.shuffle(values)
    counters={a:0 for a in ACTIONS}
    order=list(ACTIONS)
    rng.shuffle(order)
    plan=[]
    for i in range(steps):
        action=order[i%len(order)]
        values=buckets[action]
        plan.append(values[counters[action]%len(values)])
        counters[action]+=1
    return plan

def per_class(expected,actual):
    output={}
    for a in ACTIONS:
        relevant=[i for i,x in enumerate(expected) if x==a]
        correct=sum(actual[i]==a for i in relevant)
        predicted=sum(x==a for x in actual)
        output[a]={"support":len(relevant),"predicted":predicted,
                   "recall":correct/len(relevant) if relevant else None,
                   "correct":correct}
    observed=[v["recall"] for v in output.values() if v["recall"] is not None]
    return {"per_class":output,
            "balanced_accuracy":sum(observed)/len(observed) if observed else 0,
            "predicted_unique":len(set(actual)),
            "prediction_histogram":dict(Counter(actual))}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rank",type=int,choices=(4,16,64),required=True)
    ap.add_argument("--seed",type=int,choices=(11,19,37),required=True)
    ap.add_argument("--steps",type=int,default=60)
    ap.add_argument("--out",required=True)
    args=ap.parse_args()
    if args.steps<12:ap.error("At least 12 balanced steps required")
    assert_dataset()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    start=time.monotonic()
    input_data,gold=generate()
    train_labels={g["task_id"]:g for g in gold["train"]}
    train=class_balanced_plan(input_data["train"],train_labels,args.seed,args.steps)
    targets=Counter(train_labels[x["task"]["id"]]["action"] for x in train)
    assert max(targets.values())-min(targets.values())<=1,targets
    test=list(zip(input_data["test"],gold["test"]))
    assert len(test)==32
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,args.rank)
    param_count=sum(x.numel() for x in adapter.parameters())
    assert param_count==2*model.config.hidden_size*args.rank
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.0005)
    losses=[]
    h=attach(model,adapter,1)
    try:
        for i,row in enumerate(train):
            gold_action=train_labels[row["task"]["id"]]["action"]
            prompt=user_task_prompt(row,show_policy=False)
            ids,labels=encode_pair(tok,prompt,gold_action,512)
            optimizer.zero_grad(set_to_none=True)
            out=model(input_ids=ids,labels=labels)
            if not torch.isfinite(out.loss):raise RuntimeError(f"Nonfinite loss step {i}")
            out.loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            optimizer.step()
            losses.append(round(float(out.loss.detach()),5))
            if (i+1)%12==0:
                print(f"training rank={args.rank} seed={args.seed} step={i+1}/{args.steps} loss={losses[-1]:.5f}",flush=True)
    finally:
        h.remove()
    modes=("base_graph","base_policy","compass_graph","compass_policy")
    predictions=[]
    for idx,(row,answer) in enumerate(test):
        result={}
        for mode in modes:
            learned=mode.startswith("compass")
            policy=mode.endswith("policy")
            prompt=user_task_prompt(row,show_policy=policy)
            h=attach(model,adapter,1) if learned else None
            try:
                chosen,scores=score_choices(tok,model,prompt)
            finally:
                if h:h.remove()
            result[mode]={"choice_mean":chosen["mean"],"choice_sum":chosen["sum"],
                          "scores":scores}
        predictions.append({"id":row["task"]["id"],"project_id":row["project"]["id"],
                            "gold":answer["action"],"layer":answer["layer"],
                            "blocked_core":answer["blocked_core"],"case_kind":answer["case_kind"],
                            "predictions":result})
        if (idx+1)%8==0:
            print(f"evaluation rank={args.rank} seed={args.seed} {idx+1}/{len(test)}",flush=True)

    truth=[p["gold"] for p in predictions]
    mode_eval={}
    for mode in modes:
        mean_preds=[p["predictions"][mode]["choice_mean"] for p in predictions]
        sum_preds=[p["predictions"][mode]["choice_sum"] for p in predictions]
        details=per_class(truth,mean_preds)
        details.update({"correct":sum(a==b for a,b in zip(truth,mean_preds)),
                        "total":len(truth),
                        "sum_score_correct":sum(a==b for a,b in zip(truth,sum_preds)),
                        "sum_score_balanced":per_class(truth,sum_preds)["balanced_accuracy"]})
        matched={}
        for family in ("broad_test","targeted_test","confirmed_break","suspected_issue",
                       "needs_approval","forbidden_scope"):
            indices=[i for i,p in enumerate(predictions) if p["case_kind"]==family]
            matched[family]={"correct":sum(mean_preds[i]==truth[i] for i in indices),
                             "total":len(indices)}
        details["case_kind"]=matched
        details["hard_boundary_violations"]=sum(
            mean_preds[i] in ("DO_NOW","SCHEDULE")
            for i,p in enumerate(predictions)
            if p["case_kind"] in ("forbidden_scope","needs_approval"))
        pairs={}
        for project_id in sorted({p["project_id"] for p in predictions}):
            bykind={p["case_kind"]:i for i,p in enumerate(predictions) if p["project_id"]==project_id}
            assert "broad_test" in bykind and "targeted_test" in bykind
            broad=mean_preds[bykind["broad_test"]]
            targeted=mean_preds[bykind["targeted_test"]]
            pairs[project_id]={"broad":broad,"targeted":targeted,
                              "correct_contrast":broad in ("DEFER","SCHEDULE") and targeted=="DO_NOW"}
        details["paired_ancillary_core_contrast_correct"]=sum(p["correct_contrast"] for p in pairs.values())
        details["paired_ancillary_core_contrast_total"]=len(pairs)
        mode_eval[mode]=details
        print(f"rank={args.rank} seed={args.seed} {mode}: correct={details['correct']}/32 "+
              f"balanced={details['balanced_accuracy']:.3f} classes={details['predicted_unique']} "+
              f"contrast={details['paired_ancillary_core_contrast_correct']}/4",flush=True)

    # Canonical constant-answer control and independent authored fixture rule control.
    counts=Counter(truth)
    majority=max(ACTIONS,key=lambda a:counts[a])
    constant=per_class(truth,[majority]*len(truth))
    constant["correct"]=counts[majority];constant["total"]=len(truth);constant["label"]=majority
    rules=rules_evaluate(input_data["test"],gold["test"])
    report={
        "experiment":"COMPASS V8 capacity diagnostic — NOT human adjudicated",
        "rank":args.rank,"seed":args.seed,"adapter_parameters":param_count,
        "training_steps":args.steps,"balanced_training_action_counts":dict(targets),
        "time_seconds":round(time.monotonic()-start,2),
        "train_losses":losses,"test_gold_histogram":dict(counts),
        "metric_primary":"conditional mean log probability of action-code tokens, fixed common chat prefix",
        "model_comparisons":mode_eval,"majority_action_control":constant,
        "authored_rule_baseline":{"correct":rules["correct"],"total":rules["total"]},
        "predictions":predictions,
        "caveats":[
            "All examples are scripted, fictional, English-only, with eight case templates",
            "All user profile policies occur in both train and test; not cold-user generalization",
            "Test data use the planning project kind, confounded with holdout project and wording",
            "Training uses standard next-token NLL; only six-action selection is tested",
            "Ranking action codes by token means is a proxy with lexical/tokenization biases",
            "Project scope and decision rationale generation have NOT been demonstrated",
            "Rule baseline was authored for the fixture; not independent evidence",
            "Same 32 held-out tasks per trial, not 32 independent users",
            "A rank gain cannot be treated as evidence of better real user priorities"
        ]
    }
    output=Path(args.out);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_file({k:v.detach().contiguous().cpu() for k,v in adapter.state_dict().items()},
              str(output.with_suffix(".safetensors")))
    print("SAVED",output,flush=True)

if __name__=="__main__":main()
