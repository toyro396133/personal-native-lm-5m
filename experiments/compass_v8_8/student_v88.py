"""V8.8 test whether 360M actively reasoned advice benefits a 7,680-weight scorer.

Compare unchanged warm start, text-only paired training, active-advisor
augmented paired training and frozen advisor's own action choice. Same seed,
training labels, optimizer and development selection. Never train on cold.
"""
import argparse,json,random,time
from collections import defaultdict
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file
from fixtures_v88 import build
from advisor_v88 import teacher_key,augmented
from validate_v88 import audit
from matrix_v85 import PersonalRanker,personal_train_quads,personal_pair_loss
from ranker_v84 import quad_groups,get_two
from pilot_v86 import eval_full,evaluate_policy_twins,score_head
from cache_v85 import selected

def evaluate_pairs(head,pairs,features,teacher=None):
    score=defaultdict(lambda:{"total":0,"both_correct":0,"flipped":0})
    items=[]
    with torch.no_grad():
        for p in pairs:
            outs=[];teacher_outs=[]
            for which in ("original","counterfactual"):
                row=p[which]
                if teacher:
                    t=teacher[teacher_key(row)]
                    row=augmented(row,t)
                pred=score_head(head,features,row)
                outs.append(pred)
            both=(outs[0]==p["original_label"]["correct_digit"]
                  and outs[1]==p["counterfactual_label"]["correct_digit"])
            flipped=(outs[0]!=outs[1])
            for k in ("all",p["original"]["lang"]):
                score[k]["total"]+=1
                score[k]["both_correct"]+=int(both)
                score[k]["flipped"]+=int(flipped)
            items.append({"family":p["original_label"]["kind"],"lang":p["original"]["lang"],
                          "gold":[p["original_label"]["correct_digit"],p["counterfactual_label"]["correct_digit"]],
                          "chosen":outs,"both_correct":both,"flipped":flipped})
    return {k:dict(v) for k,v in score.items()},items

def teacher_scores(pairs,teacher):
    scores=defaultdict(lambda:{"total":0,"both_correct":0,"flipped":0,"matched_gold":0})
    outputs=[]
    for p in pairs:
        ans=[]
        for k in ("original","counterfactual"):
            row=p[k];t=teacher[teacher_key(row)]
            ans.append(row["task"]["choices"].index(t["choice_text"]))
        both=ans==[p["original_label"]["correct_digit"],p["counterfactual_label"]["correct_digit"]]
        for k in ("all",p["original"]["lang"]):
            scores[k]["total"]+=1
            scores[k]["both_correct"]+=int(both)
            scores[k]["flipped"]+=int(ans[0]!=ans[1])
            scores[k]["matched_gold"]+=int(ans[0]==p["original_label"]["correct_digit"])+int(ans[1]==p["counterfactual_label"]["correct_digit"])
        outputs.append({"family":p["original_label"]["kind"],"lang":p["original"]["lang"],
                        "gold":[p["original_label"]["correct_digit"],p["counterfactual_label"]["correct_digit"]],
                        "teacher_choice":ans,"both_correct":both})
    return {k:dict(v) for k,v in scores.items()},outputs

def pairing_loss(head,batch,features,teacher=None):
    losses=[]
    for p in batch:
        a=p["original"];b=p["counterfactual"]
        aa=augmented(a,teacher[teacher_key(a)]) if teacher else a
        bb=augmented(b,teacher[teacher_key(b)]) if teacher else b
        ma=head(get_two(features,aa));mb=head(get_two(features,bb))
        m0=ma[0]-ma[1];m1=mb[0]-mb[1]
        ga=p["original_label"]["correct_digit"];gb=p["counterfactual_label"]["correct_digit"]
        assert ga!=gb
        sign=(1 if ga==0 else -1)
        gold=(F.softplus(0.65-sign*m0)+F.softplus(0.65+sign*m1))/2
        reversal=0.2*(torch.tanh(m0)+torch.tanh(m1)).square()
        losses.append(gold+reversal)
    return torch.stack(losses).mean()

def run(args):
    audit()
    torch.manual_seed(args.seed);random.seed(args.seed);torch.set_num_threads(2)
    blob=torch.load(args.features,map_location="cpu",weights_only=True)
    assert blob["source_fixture"]=="v88"
    features=blob["features"];teacher=blob["teacher"]
    inputs,gold,pairs,old_diag,old_cold,newcold=build()
    chosen=selected(inputs["test"])
    project_profiles=sorted({r["project"]["policy"]["id"] for r in inputs["train"]})
    dev_profile=project_profiles[-1]
    training=[(r,g) for r,g in zip(inputs["train"],gold["train"])
              if r["project"]["policy"]["id"]!=dev_profile]
    development=[(r,g) for r,g in zip(inputs["train"],gold["train"])
                 if r["project"]["policy"]["id"]==dev_profile]
    policy_train=[p for p in pairs if p["original"]["project"]["policy"]["id"]!=dev_profile]
    policy_dev=[p for p in pairs if p["original"]["project"]["policy"]["id"]==dev_profile]
    assert (len(training),len(development),len(policy_train),len(policy_dev),
            len(old_cold),len(newcold))==(800,160,80,16,16,16)
    train_quads=quad_groups([r for r,_ in training],[g for _,g in training])
    train_reversal=personal_train_quads(train_quads,features)
    baseline_head=PersonalRanker(960,4)
    baseline_head.load_state_dict(load_file(args.checkpoint,device="cpu"))
    baseline_head.eval()
    assert sum(p.numel() for p in baseline_head.parameters())==7680
    # No new labels were shown to teacher; source status/owner authorization separate.
    teacher_train,_=teacher_scores(policy_train,teacher)
    teacher_dev,_=teacher_scores(policy_dev,teacher)
    teacher_cold,teacher_cases=teacher_scores(newcold,teacher)
    base_raw,base_raw_cases=evaluate_pairs(baseline_head,newcold,features,None)
    base_advice,base_advice_cases=evaluate_pairs(baseline_head,newcold,features,teacher)
    results={}
    outputs={}
    for use_teacher in (False,True):
        arm="teacher_assisted" if use_teacher else "text_only_paired_control"
        model=PersonalRanker(960,4)
        model.load_state_dict(baseline_head.state_dict())
        optimizer=torch.optim.AdamW(model.parameters(),lr=0.0003,weight_decay=0.01)
        best=None;best_epoch=0
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        best_dev=None;history=[]
        # Keep EXACT same optimization schedule between the two arms.
        for epoch in range(1,args.epochs+1):
            ids=list(range(len(train_reversal)))
            random.Random(args.seed*1511+epoch).shuffle(ids)
            pair_ids=list(range(len(policy_train)))
            random.Random(args.seed*1303+epoch).shuffle(pair_ids)
            model.train();losses=[]
            for batch,offset in enumerate(range(0,len(ids),20)):
                old=[train_reversal[i] for i in ids[offset:offset+20]]
                new=[policy_train[pair_ids[(batch*20+j)%len(pair_ids)]]
                     for j in range(20)]
                optimizer.zero_grad(set_to_none=True)
                loss=personal_pair_loss(model,old)+0.8*pairing_loss(
                    model,new,features,teacher if use_teacher else None)
                if not torch.isfinite(loss):raise RuntimeError("nonfinite training")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
                losses.append(float(loss.detach()))
            model.eval()
            dev_pairs,_=evaluate_pairs(model,policy_dev,features,teacher if use_teacher else None)
            # Score on 160 DEVELOPMENT tasks, not the 176 previously exposed test items.
            dev_group=quad_groups([r for r,_ in development],[g for _,g in development])
            scores,_,_=eval_full(model,development,features,blob["frozen_yesno"])
            metric=(dev_pairs["all"]["both_correct"],
                    scores["head_raw"]["contrast"]["all_four_correct"],
                    scores["head_raw"]["correct"])
            if best is None or metric>best:
                best=metric;best_epoch=epoch
                state={k:v.detach().clone() for k,v in model.state_dict().items()}
            history.append({"epoch":epoch,"loss":round(sum(losses)/len(losses),6),
                            "dev_policy_both_correct":metric[0],
                            "dev_fact_groups":metric[1],
                            "dev_correct":metric[2]})
            if epoch in (1,6,12,args.epochs):
                print(f"TRAIN {arm} epoch={epoch} policy-dev={metric[0]}/16 "+
                      f"ordinary-dev={metric[2]}/160",flush=True)
        model.load_state_dict(state);model.eval()
        cold,cold_cases=evaluate_pairs(model,newcold,features,teacher if use_teacher else None)
        exposed,exposed_cases=evaluate_pairs(model,old_cold,features,teacher if use_teacher else None)
        original,original_cases=evaluate_pairs(model,old_diag,features,teacher if use_teacher else None)
        past=[(r,g) for r,g in zip(inputs["test"],gold["test"]) if r["project"]["id"] in chosen]
        prev,_,_=eval_full(model,past,features,blob["frozen_yesno"])
        results[arm]={
            "epoch":best_epoch,"selected_development":list(best),
            "cold":cold,"exposed_v87":exposed,"exposed_v86":original,
            "previously_inspected_176_correct":prev["head_raw"]["correct"],
            "training_history":history,
        }
        outputs[arm]={"new_cold":cold_cases,"exposed_v87":exposed_cases}
        out=Path(args.output)
        path=out.with_name(out.stem+"-"+arm+".safetensors")
        save_file({k:v.detach().cpu().contiguous() for k,v in state.items()},str(path))
        print(f"RESULT {arm} cold={cold['all']['both_correct']}/{cold['all']['total']} "+
              f"swaps={cold['all']['flipped']}/{cold['all']['total']} "+
              f"old176={prev['head_raw']['correct']}/176",flush=True)
    payload={
        "experiment":"COMPASS V8.8 active frozen 360M advisor teaches personal 7680-weight model",
        "description":"Teacher generates short advice and forced A/B suggestion with no gold; two same-budget student heads compared",
        "personal_trainable_parameters":7680,"seed":args.seed,
        "epochs":args.epochs,"teacher_contexts":blob["teacher_contexts"],
        "new_cold_families":["deployment_communications","backlog_triage"],
        "test_research_status":"New scripted synthetic family authored by researcher, not independently reviewed",
        "teacher_generated_on_unseen_test":"Generated without gold, never optimized on test. Teacher test success measured separately",
        "teacher_train":teacher_train,"teacher_development":teacher_dev,
        "teacher_cold":teacher_cold,"teacher_cold_predictions":teacher_cases,
        "unchanged_v86_base_on_cold":base_raw,
        "unchanged_v86_base_with_teacher_analysis":base_advice,
        "arms":results,"predictions":outputs,
        "strict_caveats":[
          "Frozen SmolLM2-360M generated analysis; its accuracy and policy understanding cannot be assumed",
          "The same project/evidence/options policy contrast is authored synthetic fiction; no human adjudication",
          "Old V8.6 and V8.7 diagnostic data were already exposed, NOT blind test data",
          "Teacher advice was included in student inference input, so extra frozen-LM inference is required",
          "Teacher vote is positional canonical-sorted action A/B, NOT a proof of general semantic reasoning",
          "The outside trusted-source enforcement registry has no real provider or cryptographic verification"
        ]
    }
    out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf8")
    print("SAVED",out,flush=True)
if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--epochs",type=int,default=18)
    ap.add_argument("--features",default="v88-active-teacher.pt")
    ap.add_argument("--checkpoint",default="compass-v86-r4-s11.safetensors")
    ap.add_argument("--output",default="compass-v88-active-r4-s11.json")
    run(ap.parse_args())
