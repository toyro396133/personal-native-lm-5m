"""COMPASS V8.4 frozen semantic embeddings + shared pairwise ranking head.

Core diagnostic: compare a small *shared option scorer* with historical rank-4
late-layer residual (V8.3) and a frozen backbone. Two options are encoded
independently with the SAME contextual prompt. Shared scoring guarantees
swap-equivariance: score(A)-score(B) negates when the candidate order is swapped.

All examples fictional, bilingual, project-structured; this is NOT production.
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import AutoTokenizer, AutoModelForCausalLM

from run import BASE
from fixtures_v82 import generate, LANGUAGES, KINDS
from validate_v82 import audit
from guard_v82 import enforce
from pilot_v82 import breakdown

class OptionRanker(nn.Module):
    """Shared tiny nonlinear scorer. No action-position-dependent parameters."""
    def __init__(self, hidden_size, width=4):
        super().__init__()
        self.proj=nn.Linear(hidden_size,width)
        self.out=nn.Linear(width,1,bias=False)
    def forward(self, features):
        # Normalize frozen hidden-state magnitude per example.
        normalized=F.layer_norm(features,(features.shape[-1],))
        return self.out(torch.tanh(self.proj(normalized))).squeeze(-1)

def factual_candidate_prompt(row,candidate):
    p=row["project"];t=row["task"];language=row["lang"]
    if language=="he":
        return (
            "הערך התאמה של פעולה אחת ליעד הפרויקט ולראיות, "
            "בלי להתייחס לסדר של חלופות אחרות. "
            "עדיפות היא פעולה שמקדמת ליבה מאושרת, עם ראיות תקפות ותלות אמיתית. "
            "זהה מתי נדרש אישור ומתי יש לדחות מהלך.\n"
            f"יעד הפרויקט: {p['objective']['he']}\n"
            f"מדיניות: {p['policy']['he']}\n"
            f"שכבה: {t['layer']}\n"
            f"נתון מאושר נוכחי: {t['record']}\n"
            f"פעולה מוצעת: {candidate}\n"
            "האם הפעולה המוצעת מתאימה לראיות? השב Yes או No:"
        )
    return (
        "Assess ONE candidate action against current project goal and "
        "documented evidence, independently of any other option's position. "
        "Prefer authorized core progress, important prerequisites and "
        "verified facts over irrelevant or unapproved changes.\n"
        f"Project objective: {p['objective']['en']}\n"
        f"Approved policy: {p['policy']['en']}\n"
        f"Task layer: {t['layer']}\n"
        f"Current evidence: {t['record']}\n"
        f"Candidate action: {candidate}\n"
        "Does this candidate fit the current evidence? Answer Yes or No:"
    )

def representation(model,tok,row,candidate,max_tokens=1100):
    phrase=factual_candidate_prompt(row,candidate)
    ids=tok.apply_chat_template(
        [{"role":"user","content":phrase}],tokenize=True,add_generation_prompt=True)
    if len(ids)>max_tokens:
        raise ValueError(f"{row['lang']} candidate exceeds {max_tokens}: {len(ids)}")
    tokens=torch.tensor([ids],dtype=torch.long)
    with torch.inference_mode():
        hidden=model.model(input_ids=tokens).last_hidden_state[0,-1].float().detach()
    return hidden.cpu().clone(),len(ids)

def key_for(row,candidate):
    # Source id is opaque and stable between order-swapped variants.
    t=row["task"]
    return (row["project"]["id"],row["lang"],t["record_source_id"],candidate)

def quad_groups(inputs,labels):
    groups=defaultdict(dict)
    for row,gold in zip(inputs,labels):
        groups[(row["project"]["id"],row["lang"],gold["kind"])][(gold["state"],gold["swap"])]=(row,gold)
    if any(set(rows)!={(0,0),(0,1),(1,0),(1,1)} for rows in groups.values()):
        raise AssertionError("Missing fact/order reversal in training or evaluation")
    return groups

def construct_features(rows,tok,model,cache,logs,frozen_scores,yes_id,no_id):
    # Encoding each distinct contextual (evidence,candidate) exactly once
    # means swapped-option cases share the exact same representations.
    for row in rows:
        for candidate in row["task"]["choices"]:
            k=key_for(row,candidate)
            if k in cache: continue
            emb,length=representation(model,tok,row,candidate)
            cache[k]=emb
            with torch.inference_mode():
                next_logits=model.lm_head(emb.unsqueeze(0).to(model.device)).float()[0]
                frozen_scores[k]=float(next_logits[yes_id]-next_logits[no_id])
            logs.append({"lang":row["lang"],"tokens":length})
            if len(cache)%96==0:
                print(f"EXTRACT features={len(cache)} max_tokens={max(x['tokens'] for x in logs)}",flush=True)

def get_two(cache,row):
    return torch.stack([cache[key_for(row,c)] for c in row["task"]["choices"]],dim=0)

def create_training_quads(grouped,cache):
    out=[]
    for key,group in sorted(grouped.items()):
        left0,_=group[(0,0)]
        left1,_=group[(1,0)]
        v0=get_two(cache,left0);v1=get_two(cache,left1)
        # Labels deliberately stored separately; opposite fact requires
        # opposite chosen semantic action in canonical option order.
        out.append((key,v0,v1))
    return out

def signed_pair_margin(head,vec):
    s=head(vec)
    return s[0]-s[1]

def paired_loss(head,batch,target=0.6,consistency=0.15):
    vals=[]
    for _,state_zero,state_one in batch:
        d0=signed_pair_margin(head,state_zero)
        d1=signed_pair_margin(head,state_one)
        # State zero => canonical action 0 wins; state one => 1 wins.
        # Margin softplus is smooth, penalizes ambiguous decisions.
        base=(F.softplus(target-d0)+F.softplus(target+d1))/2
        symmetry=(torch.tanh(d0)+torch.tanh(d1)).square()
        vals.append(base+consistency*symmetry)
    return torch.stack(vals).mean()

def evaluate(model_head,groups,cache,frozen_scores):
    rows=[]
    with torch.no_grad():
        for key,quad in sorted(groups.items()):
            for point,(row,gold) in sorted(quad.items()):
                scores=model_head(get_two(cache,row))
                value=float(scores[0]-scores[1])
                guess=int(value<0)
                candidates=row["task"]["choices"]
                frozen_margin=(frozen_scores[key_for(row,candidates[0])]
                               -frozen_scores[key_for(row,candidates[1])])
                frozen_guess=int(frozen_margin<0)
                frozen_guarded=enforce(row,frozen_guess)["choice"]
                guarded=enforce(row,guess)
                rows.append({
                    "id":row["task"]["id"],"project_id":row["project"]["id"],
                    "lang":row["lang"],"kind":gold["kind"],"state":gold["state"],
                    "swap":gold["swap"],"gold":gold["correct_digit"],
                    "prediction":{"frozen_option_yes_no":frozen_guess,
                                  "head_raw":guess,
                                  "frozen_guarded":frozen_guarded,
                                  "head_guarded":guarded["choice"]},
                    "signed_margin":value,
                    "frozen_yes_no_score_margin":frozen_margin,"authorization_override":guarded["overridden"]
                })
    stats={mode:breakdown(rows,mode) for mode in (
        "frozen_option_yes_no","head_raw","frozen_guarded","head_guarded")}
    prior={}
    for mode in ("head_raw","frozen_option_yes_no"):
        prior={}
        for lang in LANGUAGES:
            subset=[x for x in rows if x["lang"]==lang]
            prior[lang]={"correct":sum(x["prediction"][mode]==x["gold"] for x in subset),
                         "total":len(subset),
                         "predict_zero":sum(x["prediction"][mode]==0 for x in subset),
                         "predict_one":sum(x["prediction"][mode]==1 for x in subset)}
        stats[mode]["language_diagnostics"]=prior
    stats["head_guarded"]["external_overrides"]=sum(x["authorization_override"] for x in rows)
    return stats,rows

def select_projects(inputs,amount):
    unique=sorted({row["project"]["id"] for row in inputs})
    if amount==2:return [unique[0],unique[3]]
    if amount==6:return unique
    raise ValueError("Use either 2 or 6 held-out projects")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--epochs",type=int,default=14)
    ap.add_argument("--width",type=int,default=4)
    ap.add_argument("--test-projects",type=int,choices=(2,6),default=2)
    ap.add_argument("--output",default="compass-v84-ranker-w4-s11.json")
    args=ap.parse_args()
    if args.width not in (4,8) or args.epochs<4:
        ap.error("This pilot supports width 4/8 and at least 4 epochs")
    structural=audit()
    torch.manual_seed(args.seed);random.seed(args.seed);torch.set_num_threads(2)
    start=time.monotonic()
    inputs,labels=generate()

    # Reserve one entire TRAIN profile for development/epoch selection.
    # No heldout test projects enter optimizer, scheduler or early stopping.
    train_profile=sorted({x["project"]["policy"]["id"] for x in inputs["train"]})[-1]
    training=[(row,gold) for row,gold in zip(inputs["train"],labels["train"])
              if row["project"]["policy"]["id"]!=train_profile]
    develop=[(row,gold) for row,gold in zip(inputs["train"],labels["train"])
             if row["project"]["policy"]["id"]==train_profile]
    chosen_projects=select_projects(inputs["test"],args.test_projects)
    held=[(row,gold) for row,gold in zip(inputs["test"],labels["test"])
          if row["project"]["id"] in chosen_projects]
    assert len(training)==480 and len(develop)==96
    assert len(held)==args.test_projects*48
    assert set(r["project"]["id"] for r,_ in training).isdisjoint(
           set(r["project"]["id"] for r,_ in develop))
    assert set(r["project"]["id"] for r,_ in develop).isdisjoint(chosen_projects)

    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    hidden=model.config.hidden_size
    yes_no=[tok.encode(v,add_special_tokens=False) for v in ("Yes","No")]
    if len(yes_no[0])!=1 or len(yes_no[1])!=1 or yes_no[0]==yes_no[1]:
        raise ValueError(f"Yes/No diagnostic requires distinct single tokens: {yes_no}")
    yes_id,no_id=yes_no[0][0],yes_no[1][0]
    cache={};logs=[];frozen_scores={}
    for name,records in (("train",training),("development",develop),("locked_test",held)):
        construct_features([r for r,_ in records],tok,model,cache,logs,frozen_scores,yes_id,no_id)
        print(f"CACHED {name}: {len(records)} input rows; unique option contexts so far={len(cache)}",flush=True)

    ng=lambda pairs:quad_groups([r for r,_ in pairs],[g for _,g in pairs])
    train_groups=ng(training);dev_groups=ng(develop);test_groups=ng(held)
    assert len(train_groups)==120 and len(dev_groups)==24 and len(test_groups)==args.test_projects*12
    train_pairs=create_training_quads(train_groups,cache)

    head=OptionRanker(hidden,args.width)
    num=sum(p.numel() for p in head.parameters())
    optimizer=torch.optim.AdamW(head.parameters(),lr=0.003,weight_decay=0.01)
    best_dev=-1
    best_state=None
    best_epoch=None
    history=[]
    for epoch in range(args.epochs):
        head.train()
        rng=random.Random(args.seed*1009+epoch)
        order=list(range(len(train_pairs)));rng.shuffle(order)
        updates=[]
        for pos in range(0,len(order),20):
            group=[train_pairs[i] for i in order[pos:pos+20]]
            optimizer.zero_grad(set_to_none=True)
            loss=paired_loss(head,group)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite pairwise head loss")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.0)
            optimizer.step()
            updates.append(float(loss.detach()))
        head.eval()
        development,_=evaluate(head,dev_groups,cache,frozen_scores)
        dev=development["head_raw"]
        # Only development-set quartet count selects an epoch.
        score=(dev["contrast"]["all_four_correct"],dev["correct"])
        if best_epoch is None or score>best_dev:
            best_dev=score;best_epoch=epoch+1
            best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()}
        history.append({"epoch":epoch+1,"train_loss":round(sum(updates)/len(updates),5),
                        "dev_correct":dev["correct"],
                        "dev_all_four":dev["contrast"]["all_four_correct"]})
        print(f"HEAD epoch {epoch+1}/{args.epochs} loss={history[-1]['train_loss']:.4f} "+
              f"dev={dev['correct']}/96 fullquad={dev['contrast']['all_four_correct']}/24",flush=True)

    head.load_state_dict(best_state);head.eval()
    # Locked test consulted once, after epoch selection on development data.
    scores,decisions=evaluate(head,test_groups,cache,frozen_scores)
    for name,stat in scores.items():
        contrast=stat["contrast"]
        print(f"RESULT {name} {stat['correct']}/{stat['total']} "+
              f"facts={contrast['correct_fact_flip_pairs']}/{contrast['fact_flip_pairs']} "+
              f"orders={contrast['correct_order_flip_pairs']}/{contrast['order_flip_pairs']} "+
              f"complete={contrast['all_four_correct']}/{contrast['quadruples']}",flush=True)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    report={
        "experiment":"COMPASS V8.4 shared candidate ranker over frozen bilingual project embeddings",
        "base":BASE,"seed":args.seed,"ranker_width":args.width,"trainable_parameters":num,
        "trainable_architecture":"shared 960->4->1 option scorer; no backbone updates",
        "train_examples":len(training),"development_examples":len(develop),
        "heldout_evaluated":len(held),"heldout_pool":len(inputs["test"]),
        "train_profiles":5,"development_profiles":1,"test_projects":chosen_projects,
        "selected_epoch":best_epoch,"development_selection":list(best_dev),
        "epochs_requested":args.epochs,"history":history,
        "token_lengths_max":{lang:max(x["tokens"] for x in logs if x["lang"]==lang) for lang in LANGUAGES},
        "unique_embedding_contexts":len(cache),
        "matched_frozen_yes_no_token_ids":{"Yes":yes_id,"No":no_id},
        "matched_frozen_prompt_control":"Per-candidate frozen LM Yes-minus-No logits on identical encoding used by trained head",
        "baseline_historical_v83_frozen_correct":48,
        "baseline_historical_v83_residual_correct":50,
        "baseline_historical_v83_test_total":96,
        "historical_baseline_caveat":"The candidate prompt differs from V8.3 and results are not prompt-matched.",
        "constant_position_baseline_correct":len(held)//2,
        "scores":scores,"decisions":decisions,
        "time_seconds":round(time.monotonic()-start,2),
        "dataset_integrity":structural,
        "limitations":[
            "Synthetic bilingual task labels were not independently human-adjudicated.",
            "Holdout users differ by name, not by genuinely different personal policies.",
            "The source text and action templates remain shared across project splits.",
            "Single model seed and one hyperparameter configuration; small test sample.",
            "A shared scorer enforces order consistency architecturally, not learned understanding.",
            "External permission guard must never be credited to neural learning.",
            "Frozen base vs trained V8.3 historic comparison changes the prompt structure.",
            "No autonomous rationale, document retrieval or real project execution tested."
        ]
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    save_file({k:v.contiguous().cpu() for k,v in best_state.items()},
              str(output.with_suffix(".safetensors")))
    print("SAVED",output,flush=True)

if __name__=="__main__":
    main()
