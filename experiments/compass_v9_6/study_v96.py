"""V9.6 robustness study. All training = V9.4 TRAIN only.

Independent user/project numeric memories and frozen text semantics; no gold
metadata, hidden scope or binding flags routed into inference. This is NOT
a full SELF neural adapter or a source-authenticated requirement solver.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from fixtures_v94 import FAMILY, build
from fixtures_v96 import fresh_v96_cold, stress_neutral, audit_v96
from study_v93 import FrozenSemanticEncoder, train_model
from study_v94 import FrozenFeatureCache, train_bridge, evaluate, symbolic_baseline
from study_v95 import ConstraintHead, Policies, fresh_cold as v95_fresh_cold

TRAIN_WITHDRAWN=[
 "This is not a binding instruction: an earlier safety requirement was formally withdrawn and neither option is presently compulsory.",
 "For historical context alone, the previous mandatory interface preservation rule has expired. No current rule selects the design.",
 "An outdated checklist used to oblige extra review but it has been cancelled and imposes no current constraint.",
 "Past contractual compatibility language has been expressly superseded and does not restrict today's choice.",
 "An obsolete requirement that once demanded prior verification no longer applies to this decision.",
 "The previously required public-protocol compatibility clause was retired and the current task has no binding mandate.",
]
TRAIN_BENIGN=[
 "Separate administrative information: a project tracker changed its label; both strategies remain feasible.",
 "A past meeting included this subject in its agenda, but it did not amend either action or impose an obligation.",
 "A new calendar row references this issue without altering any of the available alternatives.",
 "A housekeeping note explains an archive name change; it has no bearing on which choice matches the principle.",
]
BIND_TRAIN={
 "user":[
  "A valid mandatory risk-control procedure currently prohibits trials before detailed investigation.",
  "An approved binding operating rule demands a risk assessment before any implementation attempt.",
  "A safety decision with current authority obligates examining possible failure modes first.",
  "Presently applicable risk governance requires uncertainty analysis ahead of experimental action.",
  "The currently controlling safety checklist specifies verification before a prototype.",
 ],
 "project":[
  "An in-force integration contract obligates retention of externally published interface behavior.",
  "A binding downstream customer agreement demands compatibility with the existing client protocol.",
  "An approved and currently applicable interface obligation prevents changing public API behavior.",
  "An active formal commitment mandates keeping external message formats unchanged.",
  "A valid compatibility requirement directs preserving current outward-facing integration semantics.",
 ],
}
BIND_DEV={
 "user":"The operative process approval explicitly conditions all attempts on a thorough evaluation of potential failures first.",
 "project":"An effective interoperability undertaking specifically forbids disrupting the established external API contract.",
}
NEG_DEV=[
 "A previously binding rule is documented here solely because its authority ended; it is not in force and neither action is restricted.",
 "An earlier compulsory requirement has now been rescinded, with no replacement obligation controlling either choice.",
]
DEV_FAMILIES={"documentation","reporting"}

def modified_case(case,text,position="suffix"):
 c=copy.deepcopy(case)
 if position=="prefix":
  c["question"]=text+" "+c["question"]
 elif position=="embedded":
  j=c["question"].find(". ")
  c["question"]=c["question"][:j+2]+text+" "+c["question"][j+2:] if j>=0 else c["question"]+" "+text
 else:
  c["question"]=c["question"]+" "+text
 return c

def pref_features(case,cache):
 return (cache.pair(case,"original"),cache.pair(case,"changed"),case)

def signed_loss(bridge,features,case):
 fa,fb=features
 va=bridge(*fa);vb=bridge(*fb)
 s0=1. if case["original"]["gold"]==0 else -1.
 s1=1. if case["changed"]["gold"]==0 else -1.
 ma=va[0]-va[1];mb=vb[0]-vb[1]
 base=(F.softplus(.6-s0*ma)+F.softplus(.6-s1*mb))*.5
 if case["binding"]:
  return base+.08*(ma-mb).square(),(ma,mb)
 return base+.4*F.softplus(.8-s0*(ma-mb)),(ma,mb)

def finetune_robust_bridge(args,initial,cache,tr,dev):
 """Neutral variants only train; baseline model weights copied then fine-tuned."""
 import itertools
 torch.manual_seed(args.seed+966)
 random.seed(args.seed+966)
 bridge=copy.deepcopy(initial)
 bridge.train()
 optimizer=torch.optim.AdamW(bridge.parameters(),lr=args.robust_lr,weight_decay=.002)
 tdata=[]
 for i,case in enumerate(tr):
  base=pref_features(case,cache)
  additions=[]
  if not case["binding"]:
   for j in range(2):
    clause=TRAIN_WITHDRAWN[(i+j)%len(TRAIN_WITHDRAWN)] if j==0 else TRAIN_BENIGN[(i+j)%len(TRAIN_BENIGN)]
    variant=modified_case(case,clause,"prefix" if (i+j)%2 else "suffix")
    additions.append(pref_features(variant,cache)[:2])
  tdata.append((base,additions))
 ddata=[]
 for i,case in enumerate(dev):
  base=pref_features(case,cache)
  extras=[]
  if not case["binding"]:
   # TRAIN-template only, dev histories unseen by optimizer.
   for j,clause in enumerate((TRAIN_WITHDRAWN[(i+3)%len(TRAIN_WITHDRAWN)],
                              TRAIN_BENIGN[(i+2)%len(TRAIN_BENIGN)])):
    variant=modified_case(case,clause,"embedded" if j==0 else "suffix")
    extras.append(pref_features(variant,cache)[:2])
  ddata.append((base,extras))
 best=None;maxscore=-1;history=[]
 for ep in range(1,args.robust_epochs+1):
  order=list(range(len(tdata)))
  random.Random(args.seed*700+ep).shuffle(order)
  losses=[]
  for lo in range(0,len(order),args.batch):
   work=[]
   for index in order[lo:lo+args.batch]:
    (fa,fb,c),aug=tdata[index]
    base,(m0,m1)=signed_loss(bridge,(fa,fb),c)
    if aug:
     alt=aug[(ep+index)%len(aug)]
     augmented,(am0,am1)=signed_loss(bridge,alt,c)
     # Preserve the same action differences for semantically irrelevant text.
     consistency=.12*((torch.tanh(m0/3)-torch.tanh(am0/3)).square()+
                       (torch.tanh(m1/3)-torch.tanh(am1/3)).square())
     base=.5*(base+augmented)+consistency
    work.append(base)
   loss=torch.stack(work).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward()
   nn.utils.clip_grad_norm_(bridge.parameters(),1.5)
   optimizer.step();losses.append(float(loss.detach()))
  bridge.eval();both=0;possible=0
  with torch.no_grad():
   for (fa,fb,c),extras in ddata:
    for A,B in [(fa,fb),*extras]:
     o=int(bridge(*A)[1]>bridge(*A)[0])
     n=int(bridge(*B)[1]>bridge(*B)[0])
     both+=int(o==c["original"]["gold"] and n==c["changed"]["gold"])
     possible+=1
  if both>maxscore:
   maxscore=both
   best={k:v.detach().clone() for k,v in bridge.state_dict().items()}
   chosen=ep
  if ep==1 or ep%5==0 or ep==args.robust_epochs:
   print("V96 ROBUST",json.dumps({"epoch":ep,"dev_pairs_both":both,"dev_slots":possible,
                                  "loss":round(sum(losses)/len(losses),6)}),flush=True)
  history.append({"epoch":ep,"dev_both":both,"dev_count":possible})
  bridge.train()
 bridge.load_state_dict(best)
 bridge.eval();bridge.requires_grad_(False)
 return bridge,{"training_pairs":len(tr),"dev_pairs":len(dev),
                "chosen_epoch":chosen,"chosen_dev_both":maxscore,"history":history,
                "robust_training_uses_cold":False}

def replace_binding(case,clause):
 c=copy.deepcopy(case)
 assert c["binding"]
 original=FAMILY[c["scope"]]["binding_train"]
 assert c["question"].endswith(original)
 c["question"]=c["question"][:-len(original)]+clause
 return c

def encoded_row(cache,case,detect,as_compliant):
 _,_,q,options=cache.pair(case,"original")
 return q,options,int(detect),int(as_compliant),case["id"]

def build_policy_rows(cache,cases,dev=False):
 rows=[]
 for i,c in enumerate(cases):
  rows.append(encoded_row(cache,c,c["binding"],c["original"]["gold"]))
  if c["binding"]:
   options=[BIND_DEV[c["scope"]]] if dev else BIND_TRAIN[c["scope"]]
   for suffix in options:
    variant=replace_binding(c,suffix)
    rows.append(encoded_row(cache,variant,1,c["original"]["gold"]))
  else:
   clauses=NEG_DEV if dev else [TRAIN_WITHDRAWN[i%len(TRAIN_WITHDRAWN)],
                                TRAIN_WITHDRAWN[(i+2)%len(TRAIN_WITHDRAWN)],
                                TRAIN_BENIGN[(i+3)%len(TRAIN_BENIGN)]]
   for j,clause in enumerate(clauses):
    variant=modified_case(c,clause,"embedded" if (i+j)%2 else "prefix")
    rows.append(encoded_row(cache,variant,0,0))
 return rows

@torch.no_grad()
def policy_detection_metrics(head,rows,threshold):
 TP=TN=FP=FN=0
 option_correct=option_total=0
 for q,options,flag,gold,_ in rows:
  raw,action=head(q,options)
  pred=int(torch.sigmoid(raw).item()>=threshold)
  if flag: TP+=pred;FN+=1-pred
  else: FP+=pred;TN+=1-pred
  if flag:
   option_total+=1;option_correct+=int(int(action[1]>action[0])==gold)
 return {"tp":TP,"fn":FN,"tn":TN,"fp":FP,
         "sensitivity":round(TP/max(1,TP+FN),4),
         "specificity":round(TN/max(1,TN+FP),4),
         "option_accuracy":round(option_correct/max(1,option_total),4)}

def train_group_head(args,cache,training):
 train=[c for c in training if c["family"] not in DEV_FAMILIES]
 dev=[c for c in training if c["family"] in DEV_FAMILIES]
 assert len(train)>=200 and len(dev)>=64
 assert not ({c["family"] for c in train}&{c["family"] for c in dev})
 torch.manual_seed(args.seed+906)
 head=ConstraintHead();optimizer=torch.optim.AdamW(head.parameters(),lr=args.policy_lr,weight_decay=.002)
 t=build_policy_rows(cache,train)
 d=build_policy_rows(cache,dev,dev=True)
 chosen=None;best=-1;hist=[]
 for ep in range(1,args.policy_epochs+1):
  order=list(range(len(t)))
  random.Random(args.seed*910+ep).shuffle(order)
  head.train();batchloss=[]
  for b in range(0,len(order),args.batch):
   losses=[]
   for idx in order[b:b+args.batch]:
    q,opts,flag,gold,_=t[idx]
    raw,pred=head(q,opts)
    loss=F.binary_cross_entropy_with_logits(raw,raw.new_tensor(float(flag)))
    if flag:
     loss=loss+F.cross_entropy(pred.unsqueeze(0),torch.tensor([gold],dtype=torch.long))
    losses.append(loss)
   loss=torch.stack(losses).mean()
   optimizer.zero_grad(set_to_none=True);loss.backward()
   nn.utils.clip_grad_norm_(head.parameters(),1.5)
   optimizer.step()
   batchloss.append(float(loss.detach()))
  head.eval()
  m=policy_detection_metrics(head,d,.5)
  # Family-held-out development, not a random split of repeated templates.
  utility=m["sensitivity"]+m["specificity"]+.1*m["option_accuracy"]
  if utility>best:
   best=utility;chosen=ep
   bestweights={k:v.detach().clone() for k,v in head.state_dict().items()}
  if ep==1 or ep%5==0 or ep==args.policy_epochs:
   print("V96 DETECTOR",json.dumps({"epoch":ep,"dev":m,
     "loss":round(sum(batchloss)/len(batchloss),5)}),flush=True)
  hist.append({"epoch":ep,"dev":m})
 head.load_state_dict(bestweights);head.eval();head.requires_grad_(False)
 options=[]
 for threshold in (.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95):
  m=policy_detection_metrics(head,d,threshold)
  score=(min(m["sensitivity"],m["specificity"]),
         (m["sensitivity"]+m["specificity"])*.5,-abs(threshold-.5))
  options.append((score,threshold,m))
 _,threshold,dev_metrics=max(options,key=lambda a:a[0])
 return head,threshold,{"trained_families":sorted({c["family"] for c in train}),
  "heldout_development_families":sorted(DEV_FAMILIES),
  "trained_cases":len(train),"development_cases":len(dev),
  "trained_rows":len(t),"development_rows":len(d),
  "threshold_from_family_held_dev":threshold,
  "selected_epoch":chosen,"dev":dev_metrics,
  "history":hist}

def audit_routing(cases,policy):
 data={"binding":{"total":0,"override":0},
       "neutral":{"total":0,"override":0}}
 for c in cases:
  key="binding" if c["binding"] else "neutral"
  _,flag,_=policy.score(policy.inputs(c,"original"))
  data[key]["total"]+=1;data[key]["override"]+=int(flag)
 pos=data["binding"];neg=data["neutral"]
 return {"sensitivity":round(pos["override"]/max(1,pos["total"]),4),
         "specificity":round(1-neg["override"]/max(1,neg["total"]),4),
         "binding":pos,"neutral":neg}

def check_symmetry(cases,policy):
 correct=total=0
 for c in cases[:64]:
  for side in ("original","changed"):
   u,p,q,actions=policy.inputs(c,side)
   a=policy.score((u,p,q,actions))[0]
   b=policy.score((u,p,q,actions.flip(0)))[0]
   correct+=int(a!=b);total+=1
 return {"correct":correct,"total":total}

def check_project_isolation(cases,policy):
 total=changed=0
 for c in cases:
  if c["scope"]!="project":continue
  for side in ("original",):
   other="b" if c["project"]=="a" else "a"
   cc=copy.deepcopy(c)
   cc[side]["histories"][other]=[
     "Conflicting unrelated project statement; preserve this project's old interface.",
     "Unrelated workspace holds an entirely different principle.",
     "The other project explicitly rejects the target project's goals."
   ]
   a=policy.hybrid(c,side);b=policy.hybrid(cc,side)
   changed+=int(a!=b);total+=1
   if total==32:break
  if total==32:break
 assert total==32
 return {"tested":total,"unexpected_changes":changed}

def pair_regressions(old,new,cases,kind):
 x=[r for r in old if r["binding"]==kind]
 y=[r for r in new if r["binding"]==kind]
 assert len(x)==len(y)
 return sum(a["both_correct"] and not b["both_correct"] for a,b in zip(x,y))

def scored(cases,models):
 return {name:{"score":v,"cases":d} for name,f in models.items()
         for v,d in [evaluate(cases,f)]}

def main():
 a=argparse.ArgumentParser()
 a.add_argument("--seed",type=int,default=11)
 a.add_argument("--steps",type=int,default=600)
 a.add_argument("--epochs",type=int,default=45)
 a.add_argument("--batch",type=int,default=16)
 a.add_argument("--bridge-width",type=int,default=112)
 a.add_argument("--lr",type=float,default=.002)
 a.add_argument("--robust-epochs",type=int,default=20)
 a.add_argument("--robust-lr",type=float,default=.00065)
 a.add_argument("--policy-epochs",type=int,default=25)
 a.add_argument("--policy-lr",type=float,default=.001)
 a.add_argument("--output",default="compass-v96-results.json")
 args=a.parse_args()
 assert args.seed==11
 start=time.monotonic()
 torch.set_num_threads(2);torch.manual_seed(args.seed);random.seed(args.seed)
 full_train,v94_reused=build(args.seed,384,128)
 train=[c for i,c in enumerate(full_train) if i%5]
 dev=[c for i,c in enumerate(full_train) if i%5==0]
 enc=FrozenSemanticEncoder();enc.cache_training_vocab()
 class A: pass
 margs=A();margs.width=48;margs.seed=args.seed;margs.steps=args.steps
 margs.batch=32;margs.lr=.003;margs.check_every=50
 memory,mem_info=train_model(margs,enc);memory.eval();memory.requires_grad_(False)
 cache=FrozenFeatureCache(enc,memory)
 baseline,baseline_info=train_bridge(args,cache,train,dev)
 baseline.requires_grad_(False);baseline.eval()
 robust,robust_info=finetune_robust_bridge(args,baseline,cache,train,dev)
 head,threshold,head_info=train_group_head(args,cache,full_train)
 # Freeze before constructing NEW V9.6 cold cases.
 assert not any(p.requires_grad for p in robust.parameters())
 assert not any(p.requires_grad for p in head.parameters())
 new=fresh_v96_cold(args.seed)
 audit=audit_v96(new)
 rev=stress_neutral(new,"revocation")
 benign=stress_neutral(new,"benign")
 retrospective=v95_fresh_cold(args.seed,384,128)
 purebase=Policies(head,threshold,baseline,cache)
 hybrid=Policies(head,threshold,robust,cache)
 model={
  "original_v94_neural_only":purebase.neural,
  "robust_v96_neural_only":hybrid.neural,
  "base_bridge_plus_same_gate":purebase.hybrid,
  "robust_bridge_plus_same_gate":hybrid.hybrid,
  "privileged_fixture_scope_rule":symbolic_baseline,
 }
 cold=scored(new,model)
 coldrev=scored(rev,model)
 coldb=scored(benign,model)
 retro=scored(retrospective,model)
 routing=audit_routing(new,hybrid)
 revoked_route=audit_routing(rev,hybrid)
 symmetry=check_symmetry(new,hybrid)
 isolation=check_project_isolation(new,hybrid)
 b=cold["base_bridge_plus_same_gate"]
 h=cold["robust_bridge_plus_same_gate"]
 rb=coldrev["base_bridge_plus_same_gate"]
 rh=coldrev["robust_bridge_plus_same_gate"]
 db=coldb["base_bridge_plus_same_gate"]
 dh=coldb["robust_bridge_plus_same_gate"]
 neutral_reg=pair_regressions(b["cases"],h["cases"],new,False)
 gates={
  "binding_pairs_at_least_48_64":h["score"]["binding"]["both_correct"]>=48,
  "binding_no_regression_vs_base_hybrid":
   h["score"]["binding"]["both_correct"]>=b["score"]["binding"]["both_correct"],
  "neutral_pairs_at_least_45_64":h["score"]["neutral"]["both_correct"]>=45,
  "neutral_regressions_at_most_3":neutral_reg<=3,
  "binding_detection_sensitivity_at_least_90pct":routing["sensitivity"]>=.90,
  "binding_detection_specificity_at_least_90pct":routing["specificity"]>=.90,
  "revocation_pairs_at_least_40_64":rh["score"]["all"]["both_correct"]>=40,
  "revocation_improves_at_least_12_pairs":
    rh["score"]["all"]["both_correct"]-rb["score"]["all"]["both_correct"]>=12,
  "benign_pairs_at_least_40_64":dh["score"]["all"]["both_correct"]>=40,
  "benign_no_regression_vs_base_hybrid":
    dh["score"]["all"]["both_correct"]>=db["score"]["all"]["both_correct"],
  "revocation_override_at_most_3_64":revoked_route["neutral"]["override"]<=3,
  "symmetry_at_least_95pct":symmetry["correct"]>=.95*symmetry["total"],
  "unrelated_project_isolation_no_changes":isolation["unexpected_changes"]==0,
 }
 verdict="PROVISIONAL_SYNTHETIC_PASS_NO_LM_INTEGRATION" if all(gates.values()) else "NO_GO_TO_LM_ADAPTER_INTEGRATION"
 tensors={**{"baseline."+k:v.detach().cpu().contiguous() for k,v in baseline.state_dict().items()},
          **{"robust."+k:v.detach().cpu().contiguous() for k,v in robust.state_dict().items()},
          **{"head."+k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}}
 save_file(tensors,str(Path(args.output).with_suffix(".safetensors")))
 out={
  "version":"9.6","verdict":verdict,"synthetic_pre_registered_audit":audit,
  "new_cold":cold,"new_revocation":coldrev,"new_benign_distraction":coldb,
  "v95_retrospective_only":retro,"binding_detection_new":routing,
  "revocation_gate":revoked_route,"candidate_order_symmetry":symmetry,
  "unrelated_project_isolation":isolation,"neutral_regressions":neutral_reg,
  "registered_gates":gates,"memory_training":mem_info,
  "baseline_bridge_training":baseline_info,"robust_bridge_training":robust_info,
  "group_held_policy_training":head_info,
  "time_seconds":round(time.monotonic()-start,2),
  "limitations":[
   "All tasks, options and ground-truth states author-generated binary synthetic examples.",
   "A new family split does not imply independent language or truly unknown human intentions.",
   "Only text encodings of obligation syntax: no trusted authorization or revocation source checks.",
   "No 360M LLM control was rerun and no in-model adapter or long-term continuous update exists.",
   "The symbolic fixture-rule comparison gets hidden metadata and is not a fair equal-information baseline."
  ]
 }
 Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
 summary={"verdict":verdict,
  "new_cold":{k:v["score"] for k,v in cold.items()},
  "new_revocation":{k:v["score"]["all"] for k,v in coldrev.items()},
  "new_benign":{k:v["score"]["all"] for k,v in coldb.items()},
  "routing":routing,"revoked_routing":revoked_route,
  "order":symmetry,"project_isolation":isolation,
  "gates":gates,"seconds":out["time_seconds"]}
 print("V96 FINAL",json.dumps(summary),flush=True)

if __name__=="__main__":
 main()
