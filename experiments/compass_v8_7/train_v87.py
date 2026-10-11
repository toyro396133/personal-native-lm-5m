"""V8.7 frozen-360M personalized scorer: matched same-project policy-pair training.

Comparison: V8.6 pretrained personal rank4 7,680 weights, frozen baseline
vs identical head fine-tuned with explicit rapid↔thorough policy-only pairs.
Raw neural scores always separate from deterministic source execution gate.
"""
import argparse,json,random,time
from pathlib import Path
import torch
from torch.nn import functional as F
from safetensors.torch import load_file,save_file

from fixtures_v87 import build
from validate_v87 import audit
from cache_v87 import all_rows
from cache_v85 import dataset_fingerprint,selected
from matrix_v85 import PersonalRanker,personal_train_quads,personal_pair_loss
from ranker_v84 import quad_groups,get_two
from pilot_v86 import eval_full,evaluate_policy_twins
from trusted_sources_v87 import assert_contract

def pair_loss(head,pairs,target=0.65):
    """The project's facts/options remain constant; only owner policy flips."""
    losses=[]
    for pair in pairs:
        a=pair["original"];b=pair["counterfactual"]
        ga=pair["original_label"];gb=pair["counterfactual_label"]
        assert ga["correct_digit"]!=gb["correct_digit"]
        av=head(get_two(FEATURES,a));bv=head(get_two(FEATURES,b))
        a_margin=av[0]-av[1];b_margin=bv[0]-bv[1]
        sign=1. if ga["correct_digit"]==0 else -1.
        other=1. if gb["correct_digit"]==0 else -1.
        losses.append((F.softplus(target-sign*a_margin)+
                       F.softplus(target-other*b_margin))/2+
                      0.2*(torch.tanh(a_margin)+torch.tanh(b_margin)).square())
    return torch.stack(losses).mean()

def run(args):
    global FEATURES
    audit();assert_contract()
    torch.manual_seed(args.seed);random.seed(args.seed);torch.set_num_threads(2)
    inputs,golds,train_pairs,exposed,cold=build()
    cached=torch.load(args.features,map_location="cpu",weights_only=True)
    assert cached["source_fixture"]=="v87"
    assert cached["fingerprint"]==dataset_fingerprint(all_rows())
    assert cached["backbone"]=="HuggingFaceTB/SmolLM2-360M-Instruct"
    FEATURES=cached["features"];frozen=cached["frozen_yesno"]
    train_profiles=sorted({r["project"]["policy"]["id"] for r in inputs["train"]})
    dev_profile=train_profiles[-1]
    training=[(r,g) for r,g in zip(inputs["train"],golds["train"])
              if r["project"]["policy"]["id"]!=dev_profile]
    development=[(r,g) for r,g in zip(inputs["train"],golds["train"])
                 if r["project"]["policy"]["id"]==dev_profile]
    policy_train=[p for p in train_pairs if p["original"]["project"]["policy"]["id"]!=dev_profile]
    policy_dev=[p for p in train_pairs if p["original"]["project"]["policy"]["id"]==dev_profile]
    selection=selected(inputs["test"])
    held=[(r,g) for r,g in zip(inputs["test"],golds["test"])
          if r["project"]["id"] in selection]
    assert (len(training),len(development),len(policy_train),len(policy_dev),
            len(held),len(cold),len(exposed))==(800,160,80,16,176,16,16)
    tr_group=quad_groups([r for r,_ in training],[g for _,g in training])
    dev_group=quad_groups([r for r,_ in development],[g for _,g in development])
    quads=personal_train_quads(tr_group,FEATURES)
    head=PersonalRanker(960,4)
    weights=load_file(args.checkpoint,device="cpu")
    head.load_state_dict(weights)
    assert sum(p.numel() for p in head.parameters())==7680
    head.eval()
    base_test,base_rows,base_families=eval_full(head,held,FEATURES,frozen)
    base_exposed,_=evaluate_policy_twins(head,exposed,FEATURES)
    base_cold,base_cold_rows=evaluate_policy_twins(head,cold,FEATURES)
    base_dev,_=evaluate_policy_twins(head,policy_dev,FEATURES)
    old_dev,_=eval_full(head,development,FEATURES,frozen)
    best=(base_dev["overall"]["both_correct"],
          old_dev["head_raw"]["contrast"]["all_four_correct"],
          old_dev["head_raw"]["correct"])
    best_epoch=0
    selected_weights={k:v.detach().clone() for k,v in head.state_dict().items()}
    optimizer=torch.optim.AdamW(head.parameters(),lr=0.0003,weight_decay=0.01)
    started=time.monotonic();history=[]
    for epoch in range(1,args.epochs+1):
        rng=random.Random(args.seed*1511+epoch)
        seq=list(range(len(quads)));rng.shuffle(seq)
        pseq=list(range(len(policy_train)));rng.shuffle(pseq)
        head.train();losses=[]
        for batch,start in enumerate(range(0,len(seq),20)):
            q=[quads[i] for i in seq[start:start+20]]
            p=[policy_train[pseq[(batch*20+j)%len(pseq)]] for j in range(20)]
            optimizer.zero_grad(set_to_none=True)
            loss=personal_pair_loss(head,q)+0.8*pair_loss(head,p)
            if not torch.isfinite(loss):raise RuntimeError("nonfinite policy-aware loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        head.eval()
        dev,_=evaluate_policy_twins(head,policy_dev,FEATURES)
        dev_base,_=eval_full(head,development,FEATURES,frozen)
        key=(dev["overall"]["both_correct"],
             dev_base["head_raw"]["contrast"]["all_four_correct"],
             dev_base["head_raw"]["correct"])
        if key>best:
            best=key;best_epoch=epoch
            selected_weights={k:v.detach().clone() for k,v in head.state_dict().items()}
        history.append({
            "epoch":epoch,"loss":round(sum(losses)/len(losses),5),
            "development_policy_both_correct":dev["overall"]["both_correct"],
            "development_groups_correct":key[1],
            "development_raw_correct":key[2],
        })
        if epoch%3==0 or epoch==1:
            print(f"TRAIN v87 epoch {epoch}/{args.epochs} pair={key[0]}/16 "+
                  f"dev={key[2]}/160 full={key[1]}/40",flush=True)
    head.load_state_dict(selected_weights);head.eval()
    curr_test,curr_rows,curr_families=eval_full(head,held,FEATURES,frozen)
    curr_exposed,curr_exposed_rows=evaluate_policy_twins(head,exposed,FEATURES)
    curr_cold,curr_cold_rows=evaluate_policy_twins(head,cold,FEATURES)
    for category,vals in [
        ("BASE_RAW",base_test["head_raw"]),("NEW_RAW",curr_test["head_raw"])]:
        c=vals["contrast"]
        print(f"RESULT {category} {vals['correct']}/{vals['total']} "+
              f"fact={c['correct_fact_flip_pairs']}/{c['fact_flip_pairs']} "+
              f"full={c['all_four_correct']}/{c['quadruples']}",flush=True)
    for category,scores in [
        ("BASE_COLD",base_cold),("NEW_COLD",curr_cold),
        ("BASE_EXPOSED",base_exposed),("NEW_EXPOSED",curr_exposed)]:
        x=scores["overall"]
        print(f"RESULT {category} pairs={x['both_correct']}/{x['total']} "+
              f"flips={x['preferred_action_flipped']}/{x['total']}",flush=True)
    print(f"SELECTED epoch {best_epoch} dev={best}",flush=True)
    out=Path(args.output)
    report={
        "experiment":"COMPASS V8.7 owner-policy-only same-project pair preference training",
        "backbone":cached["backbone"],"trainable_personal_parameters":7680,
        "seed":args.seed,"training_epochs":args.epochs,"chosen_epoch":best_epoch,
        "train_cases":len(training),"development_cases":len(development),
        "train_policy_pairs":len(policy_train),"development_policy_pairs":len(policy_dev),
        "heldout_test_cases":len(held),"exposed_policy_pairs":len(exposed),
        "cold_policy_pairs":len(cold),"frozen_cache_sha256":cached["fingerprint"],
        "source_contract":assert_contract(),
        "baseline_before_train":{
            "raw":base_test,"family":base_families,
            "policy_exposed":base_exposed,"policy_cold":base_cold,
        },
        "after_policy_train":{
            "raw":curr_test,"family":curr_families,
            "policy_exposed":curr_exposed,"policy_cold":curr_cold,
        },
        "cold_policy_predictions_before":base_cold_rows,
        "cold_policy_predictions_after":curr_cold_rows,
        "exposed_policy_predictions_after":curr_exposed_rows,
        "development_trace":history,
        "elapsed_seconds":round(time.monotonic()-started,2),
        "limitations":[
            "All fixtures and gold authored by this research workflow, no human independent validation.",
            "V8.6 heldout ordinary 176 cases and exposed policy probes were inspected already; diagnostic only.",
            "Only the 16 newly authored policy pairs have fresh family names; related concepts appear in training.",
            "Core source-authentication contract is tested only using synthetic verified lookup records.",
            "No real source signature authentication or production execution; no actual user preference history.",
            "Model is frozen 360M backbone plus exactly 7,680 trained personal scorer parameters."
        ]
    }
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    save_file({k:v.detach().cpu().contiguous() for k,v in selected_weights.items()},
              str(out.with_suffix(".safetensors")))
    print("SAVED",out,flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--epochs",type=int,default=18)
    ap.add_argument("--features",default="v87-embeddings.pt")
    ap.add_argument("--checkpoint",default="compass-v86-r4-s11.safetensors")
    ap.add_argument("--output",default="compass-v87-policy-r4-s11.json")
    run(ap.parse_args())
