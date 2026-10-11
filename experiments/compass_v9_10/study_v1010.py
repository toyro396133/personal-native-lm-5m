"""V9.10 locked multi-principle and bilingual transfer evaluation.

Train ONLY on existing English V9.4 TRAIN. Cold fixture multilingual and
researcher-authored. Signed UI source identifies owner but not applicability
of an active/revoked/expired obligation.
"""
import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from fixtures_v94 import build
from fixtures_v1010 import create,uncertain,audit,PRIMARY
from fixtures_v99 import MOCK_SIGNERS
from study_v93 import FrozenSemanticEncoder,train_model
from study_v94 import FrozenFeatureCache,evaluate
from study_v98 import train_scope_head,train_router,IsolatedChooser,DEV_FAMILIES
from study_v99 import SourceChooser,SourceResolver,SourcePrototype

ASK=2

def metrics_by(cases,predict):
    groups=defaultdict(lambda:{"n":0,"both_correct":0,"decision_correct":0,
        "expected_flip_agreement":0,"actual_flips":0})
    details=[]
    for case in cases:
        got=[int(predict(case,s)) for s in ("original","changed")]
        gold=[case["original"]["gold"],case["changed"]["gold"]]
        success=got==gold
        line={"id":case["id"],"language":case["language"],"scope":case["scope"],
              "regime":case["regime"],"expected":gold,"predicted":got,
              "both_correct":success}
        details.append(line)
        for key in ("all","language_"+case["language"],
                    "scope_"+case["scope"],"regime_"+case["regime"],
                    case["language"]+"_"+case["scope"],
                    case["language"]+"_"+case["regime"]):
            g=groups[key]
            g["n"]+=1;g["both_correct"]+=int(success)
            g["decision_correct"]+=sum(int(a==b) for a,b in zip(got,gold))
            g["actual_flips"]+=int(got[0]!=got[1])
            g["expected_flip_agreement"]+=int(
                (got[0]!=got[1])==case["expected_flip"])
    return {"summary":dict(groups),"cases":details}

def mock_structured_override(case,side,source):
    """Privileged: reads fixture-built type and active status. NOT neural reasoning."""
    if case["policy_manifest_for_privileged_control_only"]["valid"]:
        mode=case["policy_manifest_for_privileged_control_only"]["typed_semantic_operation"]
        return mode^case["swapped"]
    return source.predict(case,side)

def privileged_complete_oracle(case,side):
    # Only sanity check for the evaluator: forbidden as a model baseline.
    return case[side]["gold"]

def mock_origin_ask_stress(test,source):
    failures=[]
    for c in test:
        for side in ("original","changed"):
            ans=source.predict(c,side)
            if ans!=ASK:
                failures.append({"id":c["id"],"side":side,"actual":ans})
    return {"n_pairs":len(test),"pair_ask":len(test)-len(set(x["id"] for x in failures)),
            "failure_rows":failures}

def isolate(cases,source):
    changes=0;total=0
    for case in cases:
        variant=copy.deepcopy(case)
        lang=case["language"]
        if case["scope"]=="user":
            for side in ("original","changed"):
                for key in ("a","b"):
                    variant[side]["histories"][key]=PRIMARY[lang]["project"][1][:4]+[
                        "unrelated project history is irrelevant" if lang=="en" else "היסטוריה של פרויקט אחר אינה רלוונטית"]
        else:
            for side in ("original","changed"):
                variant[side]["histories"]["user"]=PRIMARY[lang]["user"][1][:4]+[
                    "unrelated personal information is irrelevant" if lang=="en" else "מידע אישי שאינו קשור"]
        for side in ("original","changed"):
            a=source.predict(case,side)
            b=source.predict(variant,side)
            changes+=int(a!=b);total+=1
    return {"total":total,"changed":changes}

def swap_audit(cases,source):
    count=ok=0
    with torch.no_grad():
        for c in cases[:64]:
            owner,why=source.route(c)
            if owner not in ("personal","project"):continue
            for side in ("original","changed"):
                u,p,q,options=source.cache.pair(c,side)
                before=source.from_features((u,p,q,options),owner)
                after=source.from_features((u,p,q,options.flip(0)),owner)
                ok+=int(before!=after);count+=1
    return {"n":count,"correct":ok}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--steps",type=int,default=600)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--head-epochs",type=int,default=100)
    ap.add_argument("--head-lr",type=float,default=.0015)
    ap.add_argument("--router-epochs",type=int,default=100)
    ap.add_argument("--router-lr",type=float,default=.001)
    ap.add_argument("--output",default="compass-v1010-results.json")
    args=ap.parse_args()
    assert args.seed in (11,19,23)
    now=time.monotonic()
    random.seed(args.seed);torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    training,_=build(args.seed,384,128)
    encoder=FrozenSemanticEncoder()
    encoder.cache_training_vocab()
    class TrainArgs:pass
    cfg=TrainArgs();cfg.width=48;cfg.seed=args.seed;cfg.steps=args.steps
    cfg.batch=32;cfg.lr=.003;cfg.check_every=50
    memory,mem_info=train_model(cfg,encoder)
    memory.requires_grad_(False);memory.eval()
    cache=FrozenFeatureCache(encoder,memory)
    eligible=[c for c in training if not c["binding"]]
    tr=[c for c in eligible if c["family"] not in DEV_FAMILIES]
    dv=[c for c in eligible if c["family"] in DEV_FAMILIES]
    assert {x["family"] for x in tr}.isdisjoint({x["family"] for x in dv})
    personal,pinfo=train_scope_head(args,cache,tr,dv,"user")
    project,projinfo=train_scope_head(args,cache,tr,dv,"project")
    router,rinfo=train_router(args,cache,tr,dv)
    resolver=SourceResolver(MOCK_SIGNERS)
    prototype=SourcePrototype(cache,tr,resolver)
    source=SourceChooser(personal,project,cache,resolver)
    question_route=IsolatedChooser(personal,project,router,cache)
    for model in (memory,personal,project,router):
        assert not any(p.requires_grad for p in model.parameters())
    # No cold features seen before weights and control rules frozen.
    cold=create(args.seed)
    audit_info=audit(cold)
    unknown=uncertain(cold)
    methods={
       "v99_source_routed_readouts":source.predict,
       "v99_question_routed_readouts":question_route.predict,
       "source_routed_train_numeric_prototype":prototype.predict,
       "structured_task_contract_oracle":lambda c,s:mock_structured_override(c,s,source),
       "privileged_gold_sanity_check":privileged_complete_oracle,
       "constant_action_zero":lambda c,s:0,
    }
    observed={}
    for name,fn in methods.items():
        observed[name]=metrics_by(cold,fn)
        print("V1010 METHOD",args.seed,name,
              json.dumps(observed[name]["summary"],ensure_ascii=False),flush=True)
    ask=mock_origin_ask_stress(unknown,source)
    separate=isolate(cold,source)
    swap=swap_audit(cold,source)
    # The mock source is checked independently of labels on every cold pair.
    route_errors=[]
    for c in cold:
        owner,_=source.route(c)
        declared="personal" if c["scope"]=="user" else "project"
        if owner!=declared:route_errors.append(c["id"])
    actual=observed["v99_source_routed_readouts"]["summary"]
    proto=observed["source_routed_train_numeric_prototype"]["summary"]
    gates={
       "overall_88_of_128":actual["all"]["both_correct"]>=88,
       "english_48_of_64":actual["language_en"]["both_correct"]>=48,
       "hebrew_40_of_64":actual["language_he"]["both_correct"]>=40,
       "active_override_24_of_32":actual["regime_active"]["both_correct"]>=24,
       "user_40_of_64":actual["scope_user"]["both_correct"]>=40,
       "project_40_of_64":actual["scope_project"]["both_correct"]>=40,
       "valid_mock_owner_128_of_128":not route_errors,
       "invalid_mock_owner_ASK_32_of_32":ask["pair_ask"]==32,
       "irrelevant_scope_history_invariance_at_least_95pct":
           separate["changed"]<=.05*separate["total"],
       "candidate_order_symmetry_at_least_95pct":
           swap["correct"]>=.95*swap["n"],
    }
    verdict=("PROVISIONAL_SYNTHETIC_BILINGUAL_CONDITIONAL_PASS"
             if all(gates.values()) else "NO_GO_TO_LM_ADAPTER_INTEGRATION")
    weights={}
    for prefix,mod in (("personal",personal),("project",project),("router",router)):
        weights.update({prefix+"."+k:v.detach().cpu().contiguous()
                        for k,v in mod.state_dict().items()})
    save_file(weights,str(Path(args.output).with_suffix(".safetensors")))
    result={
       "version":"9.10","seed":args.seed,"verdict":verdict,
       "fixtures":audit_info,"model_evaluations":observed,
       "unknown_source_ASK":ask,"isolation":separate,
       "option_symmetry":swap,"signed_source_route_errors":route_errors,
       "registered_per_seed_gates":gates,
       "memory_training":mem_info,"personal_training":pinfo,
       "project_training":projinfo,"router_training":rinfo,
       "time_seconds":round(time.monotonic()-now,2),
       "limitations":[
        "Researcher-authored synthetic English/Hebrew tasks and artificial one-bit preference states.",
        "MiniLM is English-centric, and only original English TRAIN episodes were used for fitting.",
        "Verified UI origin is a fake signed selector correlated by author with intended scope.",
        "Golden oracle receives hidden status and correct operation, unlike neural baseline.",
        "No real humans independently labeled examples or real policy source attestations.",
        "No in-Transformer 360M SELF adapter or actual long-term personal adaptation."
       ]
    }
    Path(args.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("V1010 FINAL",json.dumps({
      "seed":args.seed,"verdict":verdict,
      "candidate":actual,
      "prototype":proto,
      "oracle":observed["structured_task_contract_oracle"]["summary"],
      "unknown_source_ASK":ask["pair_ask"],
      "invariance":separate,"option_order":swap,"gates":gates,
      "time_seconds":result["time_seconds"]},ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
