"""COMPASS V8.6 - small personal ranker against source invalidation and policy flips.

Exactly 7,680 personalized learned weights; shared SmolLM2-360M frozen.
Source/permission hard gates are reported SEPARATELY from raw learning.
"""
import argparse,random,json,time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from fixtures_v86 import build,NEW_TEST_ONLY
from validate_v86 import audit
from cache_v85 import dataset_fingerprint,selected
from matrix_v85 import PersonalRanker,personal_train_quads,personal_pair_loss,evaluate as old_evaluate
from ranker_v84 import quad_groups,get_two
from pilot_v82 import breakdown

def source_gate(row,choice):
    """Use ONLY executor-trusted fictional metadata, never model text/logits."""
    gate=row["task"].get("authoritative_gate")
    if not isinstance(gate,dict) or gate.get("provenance")!="fictional-owner-record":
        raise ValueError("No authenticated source metadata")
    validity=gate.get("source_valid")
    if validity is None: return choice,False
    if validity not in (True,False):raise ValueError("Unexpected source validity")
    intents=gate.get("choice_intents")
    if not isinstance(intents,list) or sorted(intents)!=["execute","hold"]:
        raise ValueError("Invalid executor action intent mapping")
    if validity is False and intents[choice]=="execute":
        return intents.index("hold"),True
    return choice,False

def score_head(head,cache,row):
    with torch.no_grad():
        values=head(get_two(cache,row))
        return int(float(values[0]-values[1])<0)

def eval_full(head,records,cache,freeze):
    grouped=quad_groups([r for r,_ in records],[g for _,g in records])
    scores,rows=old_evaluate(head,grouped,cache,freeze)
    by_id={r["task"]["id"]:r for r,_ in records}
    by_family=defaultdict(lambda:{"total":0,"correct":0,
                                   "valid_correct":0,"invalid_correct":0,
                                   "valid_total":0,"invalid_total":0})
    status_gate_overrides=0
    for row in rows:
        task=by_id[row["id"]]
        corrected,override=source_gate(task,row["prediction"]["head_guarded"])
        row["prediction"]["head_source_guarded"]=corrected
        row["source_override"]=override
        status_gate_overrides+=int(override)
        detail=by_family[row["kind"]]
        detail["total"]+=1
        right=int(row["gold"]==row["prediction"]["head_raw"])
        detail["correct"]+=right
        if row["state"]==0:
            detail["valid_total"]+=1;detail["valid_correct"]+=right
        else:
            detail["invalid_total"]+=1;detail["invalid_correct"]+=right
    scores["head_source_guarded"]=breakdown(rows,"head_source_guarded")
    scores["head_source_guarded"]["source_gate_overrides"]=status_gate_overrides
    return scores,rows,{k:dict(v) for k,v in sorted(by_family.items())}

def evaluate_policy_twins(head,pairs,cache):
    breakdowns=defaultdict(lambda:{"total":0,"both_correct":0,"both_wrong":0,
                                   "preferred_action_flipped":0})
    output=[]
    for p in pairs:
        a=p["original"];b=p["counterfactual"]
        label=p["original_label"];target=p["counterfactual_label"]
        pred=score_head(head,cache,a);other=score_head(head,cache,b)
        both=(pred==label["correct_digit"] and other==target["correct_digit"])
        changed=(pred!=other)
        key=a["lang"]
        for name in ("overall",key):
            summary=breakdowns[name];summary["total"]+=1
            summary["both_correct"]+=int(both)
            summary["preferred_action_flipped"]+=int(changed)
            summary["both_wrong"]+=int(pred!=label["correct_digit"] and other!=target["correct_digit"])
        output.append({
            "project":a["project"]["id"],"language":key,
            "kind":label["kind"],"swap":label["swap"],
            "policy_before":a["project"]["policy"]["mode"],
            "policy_after":b["project"]["policy"]["mode"],
            "evidence_unchanged":a["task"]["record"]==b["task"]["record"],
            "choices_unchanged":a["task"]["choices"]==b["task"]["choices"],
            "gold":[label["correct_digit"],target["correct_digit"]],
            "predictions":[pred,other],"both_correct":both,"preference_flip":changed
        })
    return {k:dict(v) for k,v in breakdowns.items()},output

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rank",type=int,choices=(4,16,64),default=4)
    p.add_argument("--seed",type=int,default=11)
    p.add_argument("--epochs",type=int,default=14)
    p.add_argument("--features",default="v86-embeddings.pt")
    p.add_argument("--output",default="compass-v86-r4-s11.json")
    args=p.parse_args()
    audit()
    torch.manual_seed(args.seed);random.seed(args.seed);torch.set_num_threads(2)
    inputs,golds,policy_twins=build()
    selection=selected(inputs["test"])
    trainrows=inputs["train"]
    test=[(r,g) for r,g in zip(inputs["test"],golds["test"])
          if r["project"]["id"] in selection]
    allrows=trainrows+[r for r,_ in test]+[p["counterfactual"] for p in policy_twins]
    artifact=torch.load(args.features,map_location="cpu",weights_only=True)
    assert artifact["source_fixture"]=="v86"
    assert artifact["fingerprint"]==dataset_fingerprint(allrows)
    assert artifact["selected_test_projects"]==sorted(selection)
    cache=artifact["features"];freeze=artifact["frozen_yesno"]
    ids=sorted({r["project"]["policy"]["id"] for r in trainrows})
    dev_id=ids[-1]
    training=[(r,g) for r,g in zip(inputs["train"],golds["train"])
              if r["project"]["policy"]["id"]!=dev_id]
    dev=[(r,g) for r,g in zip(inputs["train"],golds["train"])
         if r["project"]["policy"]["id"]==dev_id]
    assert len(training)==800 and len(dev)==160 and len(test)==176
    make_group=lambda pairs:quad_groups([r for r,_ in pairs],[g for _,g in pairs])
    train_group=make_group(training);dev_group=make_group(dev)
    assert len(train_group)==200 and len(dev_group)==40
    pairs=personal_train_quads(train_group,cache)
    head=PersonalRanker(960,args.rank)
    size=sum(x.numel() for x in head.parameters())
    assert size=={4:7680,16:30720,64:122880}[args.rank]
    optimizer=torch.optim.AdamW(head.parameters(),lr=0.001,weight_decay=0.01)
    history=[];best=None;weights=None;epoch_best=None
    started=time.monotonic()
    for e in range(args.epochs):
        order=list(range(len(pairs)))
        random.Random(args.seed*1009+e).shuffle(order)
        head.train();losses=[]
        for offset in range(0,len(order),20):
            group=[pairs[i] for i in order[offset:offset+20]]
            optimizer.zero_grad(set_to_none=True)
            loss=personal_pair_loss(head,group)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite ranking loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        head.eval()
        dev_scores,_=old_evaluate(head,dev_group,cache,freeze)
        metric=(dev_scores["head_raw"]["contrast"]["all_four_correct"],
                dev_scores["head_raw"]["correct"])
        if best is None or metric>best:
            best=metric;epoch_best=e+1
            weights={k:v.detach().clone() for k,v in head.state_dict().items()}
        history.append({"epoch":e+1,"loss":round(sum(losses)/len(losses),5),
                        "dev_correct":metric[1],"dev_quads":metric[0]})
        if (e+1)%3==0 or e==0:
            print(f"TRAIN rank={args.rank} seed={args.seed} epoch {e+1}/{args.epochs} "+
                  f"dev={metric[1]}/160 full={metric[0]}/40",flush=True)
    head.load_state_dict(weights);head.eval()
    scores,outputs,families=eval_full(head,test,cache,freeze)
    policy_scores,policy_records=evaluate_policy_twins(head,policy_twins,cache)
    cold={k:families[k] for k in NEW_TEST_ONLY}
    raw=scores["head_raw"];fact=raw["contrast"]
    print(f"RESULT raw={raw['correct']}/{raw['total']} "+
          f"evidence_flips={fact['correct_fact_flip_pairs']}/{fact['fact_flip_pairs']} "+
          f"full={fact['all_four_correct']}/{fact['quadruples']}",flush=True)
    for name,v in cold.items():
        print(f"COLD {name}: {v['correct']}/{v['total']} valid={v['valid_correct']}/{v['valid_total']} "+
              f"revoked={v['invalid_correct']}/{v['invalid_total']}",flush=True)
    print(f"POLICY only-pairs both-correct={policy_scores['overall']['both_correct']}/"+
          f"{policy_scores['overall']['total']} flip={policy_scores['overall']['preferred_action_flipped']}",flush=True)
    output=Path(args.output)
    report={
        "experiment":"COMPASS V8.6 source-validity and project-held policy counterfactual pilot",
        "rank":args.rank,"seed":args.seed,"trainable_personal_weights":size,
        "frozen_backbone":artifact["backbone"],"cache_fingerprint":artifact["fingerprint"],
        "training_rows":len(training),"development_rows":len(dev),
        "locked_test_rows":len(test),"test_source_families":list(NEW_TEST_ONLY),
        "policy_only_pairs":len(policy_twins),
        "best_epoch":epoch_best,"development_best":list(best),
        "training_history":history,"scores":scores,
        "family_summary":families,"cold_families":cold,
        "policy_contrasts":policy_scores,
        "policy_contrast_predictions":policy_records,
        "all_raw_predictions":outputs,"elapsed_seconds":round(time.monotonic()-started,2),
        "limitations":[
          "All examples and gold labels constructed by research workflow, no independent expert review",
          "V8.5 source_retracted family is a previously inspected diagnostic, NOT a novel cold test",
          "Two new test-only family names/lexical patterns differ from training but share validity semantics",
          "Only one 7,680-weight seed in this launch; extra model sizes not yet evaluated",
          "Policy-only test varies authorized policy and leaves project/evidence/candidates unchanged",
          "External source revocation and owner/authorization guards are NOT neural learning",
          "Frozen Yes/No comparator uses English next-token codes on Hebrew prompts",
          "No free-form planning, real source authentication or real-world user-history personalization"
        ]
    }
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    save_file({k:v.detach().cpu().contiguous() for k,v in weights.items()},
              str(output.with_suffix(".safetensors")))
    print("SAVED",output,flush=True)
if __name__=="__main__":
    main()
