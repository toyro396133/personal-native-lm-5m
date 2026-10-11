"""COMPASS V8.5 fixed-compute exact-parameter-count personal model comparison.

Ranks 4/16/64 each use EXACTLY 7,680 / 30,720 / 122,880
trainable parameters: two bias-free matrices of shape [rank, 960].
Same frozen SmolLM2-360M embeddings, identical train/dev/test split,
optimizer step count and synthetic policies across all ranks and seeds.
"""
import argparse
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from fixtures_v85 import build,NEW_KINDS,NOVEL_TEST_KIND
from validate_v85 import audit
from cache_v85 import dataset_fingerprint,selected
from ranker_v84 import (
    quad_groups,get_two,key_for,
)
from guard_v82 import enforce
from pilot_v82 import breakdown

class PersonalRanker(nn.Module):
    """Exactly 2 * 960 * rank learned weights; shared option-wise scorer."""
    def __init__(self,hidden_size,rank):
        super().__init__()
        self.left=nn.Linear(hidden_size,rank,bias=False)
        self.right=nn.Linear(hidden_size,rank,bias=False)
        nn.init.normal_(self.left.weight,mean=0.0,std=0.025)
        nn.init.normal_(self.right.weight,mean=0.0,std=0.025)
        self.scale=1.0/math.sqrt(rank)
    def forward(self,embeddings):
        emb=F.layer_norm(embeddings,(embeddings.shape[-1],))
        return (torch.tanh(self.left(emb))*self.right(emb)).sum(dim=-1)*self.scale

def evaluate(head,groups,cache,frozen_scores):
    output=[]
    with torch.no_grad():
        for _,quartet in sorted(groups.items()):
            for (state,swap),(row,gold) in sorted(quartet.items()):
                vec=get_two(cache,row)
                raw=head(vec)
                margin=float(raw[0]-raw[1])
                chosen=int(margin<0)
                actions=row["task"]["choices"]
                base_margin=(frozen_scores[key_for(row,actions[0])]
                             -frozen_scores[key_for(row,actions[1])])
                frozen=int(base_margin<0)
                guarded=enforce(row,chosen)
                frozen_safe=enforce(row,frozen)
                output.append({
                    "id":row["task"]["id"],
                    "project":row["project"]["id"],
                    "project_id":row["project"]["id"],
                    "language":row["lang"],
                    "lang":row["lang"],
                    "kind":gold["kind"],"state":state,"swap":swap,
                    "policy_mode":row["project"]["policy"]["mode"],
                    "gold":gold["correct_digit"],
                    "prediction":{
                        "head_raw":chosen,"frozen":frozen,
                        "head_guarded":guarded["choice"],
                        "frozen_guarded":frozen_safe["choice"],
                    },
                    "margin":margin,"frozen_margin":base_margin,
                    "guard_overridden":guarded["overridden"],
                })
    result={mode:breakdown(output,mode)
            for mode in ("head_raw","frozen","head_guarded","frozen_guarded")}
    for mode in ("head_raw","frozen"):
        result[mode]["by_language"]={}
        result[mode]["by_policy"]={}
        for field,dest in (("lang","by_language"),("policy_mode","by_policy")):
            for label in sorted({x[field] for x in output}):
                items=[r for r in output if r[field]==label]
                result[mode][dest][label]={
                    "correct":sum(r["gold"]==r["prediction"][mode] for r in items),
                    "total":len(items),
                    "chosen_zero":sum(r["prediction"][mode]==0 for r in items),
                    "chosen_one":sum(r["prediction"][mode]==1 for r in items),
                }
    return result,output

def personal_train_quads(groups,cache):
    """Preserve explicitly authored owner preference labels for each state.

    V8.4 assumed that state 0 always selects semantic option A. In V8.5
    careful/thorough preference states intentionally REVERSE this, so using
    the old training routine would silently train on the wrong answer.
    """
    out=[]
    for key,quartet in sorted(groups.items()):
        original0,label0=quartet[(0,0)]
        original1,label1=quartet[(1,0)]
        assert label0["correct_digit"] != label1["correct_digit"]
        for state in (0,1):
            assert quartet[(state,1)][1]["correct_digit"] == (
                quartet[(state,0)][1]["correct_digit"]^1)
        out.append((key,get_two(cache,original0),get_two(cache,original1),
                    label0["correct_digit"],label1["correct_digit"]))
    return out

def personal_pair_loss(head,batch,target=0.6,consistency=0.15):
    """Train on actual gold preferences, including thorough-policy flips."""
    values=[]
    for key,state0,state1,desired0,desired1 in batch:
        margin0=head(state0)[0]-head(state0)[1]
        margin1=head(state1)[0]-head(state1)[1]
        sign0=1.0 if desired0==0 else -1.0
        sign1=1.0 if desired1==0 else -1.0
        assert sign0==-sign1,(key,desired0,desired1)
        classification=(F.softplus(target-sign0*margin0)
                        +F.softplus(target-sign1*margin1))/2
        # Evidence reversal should yield opposite signed score margins.
        reversal=(torch.tanh(margin0)+torch.tanh(margin1)).square()
        values.append(classification+consistency*reversal)
    return torch.stack(values).mean()

def run(args):
    audit()
    torch.manual_seed(args.seed);random.seed(args.seed);torch.set_num_threads(2)
    inputs,golds=build()
    project_ids=selected(inputs["test"])
    both=inputs["train"]+[r for r in inputs["test"] if r["project"]["id"] in project_ids]
    archived=torch.load(args.features,map_location="cpu",weights_only=True)
    assert archived["source_fixture"]=="v85"
    assert archived["fingerprint"]==dataset_fingerprint(both),"Fixture changed after cache creation"
    assert archived["selected_test_projects"]==sorted(project_ids)
    assert archived["backbone"]=="HuggingFaceTB/SmolLM2-360M-Instruct"
    cache=archived["features"]
    freeze=archived["frozen_yesno"]
    ids=sorted({x["project"]["policy"]["id"] for x in inputs["train"]})
    dev_id=ids[-1]
    training=[(r,g) for r,g in zip(inputs["train"],golds["train"])
              if r["project"]["policy"]["id"]!=dev_id]
    dev=[(r,g) for r,g in zip(inputs["train"],golds["train"])
         if r["project"]["policy"]["id"]==dev_id]
    test=[(r,g) for r,g in zip(inputs["test"],golds["test"])
          if r["project"]["id"] in project_ids]
    assert len(training)==640 and len(dev)==128 and len(test)==144,(
        len(training),len(dev),len(test))
    make_group=lambda pairs:quad_groups([r for r,_ in pairs],[g for _,g in pairs])
    train_groups=make_group(training)
    development=make_group(dev)
    locked_test=make_group(test)
    assert len(train_groups)==160 and len(development)==32 and len(locked_test)==36
    pairs=personal_train_quads(train_groups,cache)
    head=PersonalRanker(960,args.rank)
    param_count=sum(x.numel() for x in head.parameters())
    assert param_count=={4:7680,16:30720,64:122880}[args.rank]
    optimizer=torch.optim.AdamW(head.parameters(),lr=0.001,weight_decay=0.01)
    history=[];best=None;best_params=None;best_epoch=None
    started=time.monotonic()
    for epoch in range(args.epochs):
        order=list(range(len(pairs)))
        random.Random(args.seed*1009+epoch).shuffle(order)
        head.train();losses=[]
        for pos in range(0,len(order),20):
            subset=[pairs[i] for i in order[pos:pos+20]]
            optimizer.zero_grad(set_to_none=True)
            loss=personal_pair_loss(head,subset)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss rank {args.rank}, epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(),1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        scores,_=evaluate(head,development,cache,freeze)
        raw=scores["head_raw"]
        metric=(raw["contrast"]["all_four_correct"],raw["correct"])
        if best is None or metric>best:
            best=metric;best_epoch=epoch+1
            best_params={name:v.detach().clone() for name,v in head.state_dict().items()}
        history.append({
            "epoch":epoch+1,"loss":round(sum(losses)/len(losses),5),
            "dev_correct":raw["correct"],"dev_total":raw["total"],
            "dev_quadruples":raw["contrast"]["all_four_correct"],
        })
        if (epoch+1)%3==0 or epoch==0:
            print(f"TRAIN rank={args.rank} seed={args.seed} epoch={epoch+1}/{args.epochs} "+
                  f"loss={history[-1]['loss']:.4f} dev={raw['correct']}/{raw['total']} "+
                  f"groups={raw['contrast']['all_four_correct']}/32",flush=True)
    head.load_state_dict(best_params);head.eval()
    scores,rows=evaluate(head,locked_test,cache,freeze)
    for mode in ("frozen","head_raw","head_guarded"):
        r=scores[mode];c=r["contrast"]
        print(f"RESULT rank={args.rank} seed={args.seed} {mode}: "+
              f"{r['correct']}/{r['total']} fact={c['correct_fact_flip_pairs']}/{c['fact_flip_pairs']} "+
              f"full={c['all_four_correct']}/{c['quadruples']} "+
              f"EN={r['by_language']['en']['correct'] if 'by_language' in r else 'N/A'} "+
              f"HE={r['by_language']['he']['correct'] if 'by_language' in r else 'N/A'}",flush=True)
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    report={
        "experiment":"COMPASS V8.5 exact personal ranker parameter-capacity comparison",
        "rank":args.rank,"seed":args.seed,"trainable_parameters":param_count,
        "backbone":archived["backbone"],"frozen_embeddings":archived["feature_count"],
        "matched_feature_digest":archived["fingerprint"],
        "training":len(training),"development":len(dev),"locked_test":len(test),
        "test_pool":len(inputs["test"]),"test_projects":sorted(project_ids),
        "training_epochs":args.epochs,"batch_size_quartets":20,
        "optimizer_updates_per_epoch":len(range(0,len(pairs),20)),
        "selected_epoch":best_epoch,"selected_dev_metric":list(best),
        "history":history,"scores":scores,"decisions":rows,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "limits":[
            "Only synthetic authored bilingual decisions; no independent human gold labels.",
            "The unseen source-retraction dilemma is test-only, not trained or selected.",
            "Different genuine fictional rapid/thorough preferences are included; not actual individual policy learning.",
            "Fixed frozen 360M feature extractor; exact counts refer ONLY to personal ranker parameters.",
            "The same template families recur across most train and test scenarios.",
            "Only supplied candidate actions scored; no free-form multistep planning.",
            "Separate explicit permission guard is not learned model competence."
        ]
    }
    out.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    save_file({k:v.detach().cpu().contiguous() for k,v in best_params.items()},
              str(out.with_suffix(".safetensors")))
    print("SAVED",out,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--rank",type=int,choices=(4,16,64),required=True)
    p.add_argument("--seed",type=int,choices=(11,19,37),required=True)
    p.add_argument("--epochs",type=int,default=14)
    p.add_argument("--features",default="v85-embeddings.pt")
    p.add_argument("--output",required=True)
    args=p.parse_args()
    run(args)
