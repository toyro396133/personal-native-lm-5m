"""COMPASS V9.5: separated text-derived binding gate, isolated from latent style.

No V9.4/V9.5 cold case gold, oracle binding, scope, family or swapped
metadata enter policy inference. V9.4 cold reused for diagnostics only.
All scenarios researcher-generated; output NEVER certifies real permissions.
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

import fixtures_v94
from fixtures_v94 import build, FAMILY
from study_v93 import FrozenSemanticEncoder, train_model, prototypes
from study_v94 import (FrozenFeatureCache, train_bridge, evaluate,
                       symbolic_baseline, semantic_baseline)

# Freshly authored V9.5 task families. None overlaps with V9.4 train/cold.
FRESH_USER = [
 ("rollback","An unexpected deployment regression needs a first response. Both approaches are allowed.",
  "Carefully establish causes and possible side effects before making changes.",
  "Ship a limited reversible correction promptly and revise after feedback."),
 ("profiling","You must personally approach a novel slowdown; neither plan is mandatory.",
  "Investigate measurements and uncertainty before choosing the intervention.",
  "Try a narrow, safe performance probe and refine it from evidence."),
 ("triage","A low-risk operational problem is ambiguous; either next step is permitted.",
  "Analyze competing explanations and consequences before taking action.",
  "Begin a small recoverable experiment and iterate on its results."),
 ("quality","An unfamiliar quality regression has two acceptable initial responses.",
  "Inspect risks and assumptions in detail before committing to a fix.",
  "Create a small controlled patch, measure, and adjust."),
]
FRESH_PROJECT = [
 ("inventory","An inventory platform is being rebuilt. Both directions are feasible unless an active external obligation applies.",
  "Reorganize inventories to better deliver the fundamental visibility objective.",
  "Preserve the existing externally published inventory event protocol."),
 ("messaging","A notification subsystem is modernized. Both approaches are available when unconstrained.",
  "Remake notifications around the actual communication outcome rather than legacy steps.",
  "Maintain the established interface contract consumed by downstream applications."),
 ("enrollment","A registration workflow needs replacing. No approach is mandated by default.",
  "Redesign registration in service of its core purpose, regardless of former screens.",
  "Keep the public registration API's compatibility guarantees for existing integrators."),
 ("indexing","A search-index layer needs a fresh design. Both candidate strategies could work.",
  "Improve indexing around the real discovery goal, even when old mechanics disappear.",
  "Honor the published indexing protocol for current external users."),
]
FRESH_BINDING = {
 "user":"A currently approved safety directive explicitly requires evaluating failure risks before any trial, even if personal habits differ.",
 "project":"A currently enforceable signed interface requirement expressly demands preserving the documented downstream compatibility contract.",
}
REVOKED_TRAIN = (
 "A former safety obligation was explicitly withdrawn and is no longer binding; no approach is mandatory. ",
 "An earlier mandatory integration contract has expired and has no force; neither action is required. ",
)
REVOKED_TEST = (
 "A past compulsory operational rule has been formally rescinded. It imposes no current obligation; both choices remain permitted. ",
 "An old mandatory integration safeguard has lapsed, so neither candidate is currently legally or operationally required. ",
)

def fresh_cold(seed, train_n, cold_n):
 """Temporary fixture mutation only during construction; restore before baseline."""
 before = {key:(copy.deepcopy(FAMILY[key]["cold"]),FAMILY[key]["binding_cold"])
           for key in ("user","project")}
 try:
  FAMILY["user"]["cold"] = FRESH_USER
  FAMILY["project"]["cold"] = FRESH_PROJECT
  for key in ("user","project"):
   FAMILY[key]["binding_cold"] = FRESH_BINDING[key]
  _,cases=build(seed+2026,train_n,cold_n)
 finally:
  for key in ("user","project"):
   FAMILY[key]["cold"],FAMILY[key]["binding_cold"]=before[key]
 assert len(cases)==cold_n
 return cases

def revoked_stress(cold):
 """Only neutral data; withdrawal is text, not a model-readable metadata flag."""
 out=[]
 i=0
 for c in cold:
  if c["binding"]:
   continue
  row=copy.deepcopy(c)
  row["id"]="REVOKED-"+row["id"]
  row["question"]=REVOKED_TEST[i%len(REVOKED_TEST)]+row["question"]
  i+=1
  out.append(row)
 return out

class ConstraintHead(nn.Module):
 """Task-derived gate and independently scored compliance, with no histories."""
 def __init__(self,width=64):
  super().__init__()
  self.detect=nn.Sequential(nn.Linear(384,width),nn.LayerNorm(width),
                            nn.GELU(),nn.Linear(width,1))
  self.question=nn.Sequential(nn.Linear(384,width),nn.Tanh())
  self.candidate=nn.Sequential(nn.Linear(384,width),nn.Tanh())
  self.compliance=nn.Sequential(nn.Linear(width*4,width),nn.GELU(),
                                nn.Linear(width,32),nn.GELU(),nn.Linear(32,1))
 def forward(self,question,options):
  binding_logit=self.detect(question).reshape(())
  q=self.question(question)
  a=self.candidate(options)
  q2=q.expand(options.shape[0],-1)
  features=torch.cat((q2,a,q2*a,torch.abs(q2-a)),dim=-1)
  return binding_logit,self.compliance(features).squeeze(-1)

def make_policy_features(cases,fc,with_negative=False):
 rows=[]
 new_neg=[]
 for i,c in enumerate(cases):
  _,_,q,opts=fc.pair(c,"original")
  rows.append((q,opts,int(c["binding"]),int(c["original"]["gold"]),c["id"]))
  if with_negative and not c["binding"]:
   txt=REVOKED_TRAIN[i%len(REVOKED_TRAIN)]+c["question"]
   new_neg.append((txt,opts,c["id"]))
 if new_neg:
  fc.encoder.encode_missing([t for t,_,_ in new_neg])
  for txt,opts,ident in new_neg:
   # Encoder caches tensors under inference_mode; clone outside that context.
   q=fc.encoder.vectors[txt].detach().clone()
   rows.append((q,opts,0,0,"REVOKED-TRAIN-"+ident))
 return rows

def detect_metrics(head,rows,threshold):
 TP=TN=FP=FN=0
 bind_right=bind_total=0
 with torch.no_grad():
  for q,opts,flag,gold,_ in rows:
   raw,policy=head(q,opts)
   predicted=int(torch.sigmoid(raw).item()>=threshold)
   if flag and predicted:TP+=1
   elif flag:FN+=1
   elif predicted:FP+=1
   else:TN+=1
   if flag:
    bind_total+=1
    bind_right+=int(int(policy[1]>policy[0])==gold)
 return {"tp":TP,"tn":TN,"fp":FP,"fn":FN,
         "sensitivity":round(TP/max(1,TP+FN),4),
         "specificity":round(TN/max(1,TN+FP),4),
         "binding_option_accuracy":round(bind_right/max(1,bind_total),4)}

def train_constraint_head(args,fc,train,dev):
 random.seed(args.seed+191)
 torch.manual_seed(args.seed+191)
 head=ConstraintHead()
 optimizer=torch.optim.AdamW(head.parameters(),lr=args.policy_lr,
                             weight_decay=.002)
 t=make_policy_features(train,fc,True)
 d=make_policy_features(dev,fc,True)
 best=None
 bestscore=-1
 history=[]
 for ep in range(1,args.policy_epochs+1):
  order=list(range(len(t)))
  random.Random(args.seed*990+ep).shuffle(order)
  head.train();all_losses=[]
  for pos in range(0,len(order),args.batch):
   total=[]
   for idx in order[pos:pos+args.batch]:
    q,opts,flag,gold,_=t[idx]
    raw,actions=head(q,opts)
    loss=F.binary_cross_entropy_with_logits(raw,raw.new_tensor(float(flag)))
    if flag:
     loss=loss+1.5*F.cross_entropy(actions.unsqueeze(0),
                          torch.tensor([gold],dtype=torch.long))
    total.append(loss)
   loss=torch.stack(total).mean()
   optimizer.zero_grad(set_to_none=True)
   loss.backward();nn.utils.clip_grad_norm_(head.parameters(),1.5)
   optimizer.step()
   all_losses.append(float(loss.detach()))
  head.eval()
  stats=detect_metrics(head,d,.5)
  # Training-template development only; favors detector and compliant choice.
  utility=stats["tp"]+stats["tn"]+stats["binding_option_accuracy"]*stats["tp"]
  if utility>bestscore:
   bestscore=utility
   best={k:v.detach().clone() for k,v in head.state_dict().items()}
  if ep==1 or ep%10==0 or ep==args.policy_epochs:
   print("V95 HEAD",json.dumps({"epoch":ep,"train_loss":round(sum(all_losses)/len(all_losses),5),
     "dev":stats}),flush=True)
  history.append({"epoch":ep,"train_loss":round(sum(all_losses)/len(all_losses),5),
                  "dev":stats})
 head.load_state_dict(best)
 head.requires_grad_(False);head.eval()
 # Lock threshold using only in-template dev, before cold evaluation.
 threshold_candidates=(.25,.35,.45,.5,.6,.7,.8)
 scored=[]
 for value in threshold_candidates:
  m=detect_metrics(head,d,value)
  balanced=(m["sensitivity"]+m["specificity"])/2
  scored.append((balanced,-abs(value-.5),value,m))
 _,_,threshold,devscore=max(scored)
 return head,threshold,{"train_records":len(t),"dev_records":len(d),
   "trained_weights":sum(p.numel() for p in head.parameters()),
   "threshold_from_dev":threshold,"dev_at_threshold":devscore,
   "history":history}

class Policies:
 def __init__(self,head,threshold,bridge,fc):
  self.head=head;self.threshold=threshold;self.bridge=bridge;self.fc=fc
 def inputs(self,case,side):
  return self.fc.pair(case,side)
 @torch.no_grad()
 def score(self,features):
  u,p,q,opts=features
  logit,policy=self.head(q,opts)
  is_binding=torch.sigmoid(logit).item()>=self.threshold
  unconstrained=self.bridge(u,p,q,opts)
  picked=policy if is_binding else unconstrained
  return int(picked[1]>picked[0]),bool(is_binding),float(torch.sigmoid(logit))
 def hybrid(self,case,side):
  return self.score(self.inputs(case,side))[0]
 def neural(self,case,side):
  with torch.no_grad():
   s=self.bridge(*self.inputs(case,side))
   return int(s[1]>s[0])
 def option_only(self,case,side):
  with torch.no_grad():
   _,s=self.head(*self.inputs(case,side)[2:])
   return int(s[1]>s[0])

def routing_audit(cases,policy):
 p={"binding":{"n":0,"overrides":0},"neutral":{"n":0,"overrides":0}}
 for c in cases:
  key="binding" if c["binding"] else "neutral"
  _,override,_=policy.score(policy.inputs(c,"original"))
  p[key]["n"]+=1;p[key]["overrides"]+=int(override)
 bind=p["binding"];neutral=p["neutral"]
 return {"binding_detection_sensitivity":bind["overrides"]/max(bind["n"],1),
         "binding_detection_specificity":1-neutral["overrides"]/max(neutral["n"],1),
         "binding_overrides":bind,"neutral_false_overrides":neutral}

def order_symmetry(cases,policy):
 total=good=0
 for c in cases[:64]:
  for side in ("original","changed"):
   features=policy.inputs(c,side)
   base=policy.score(features)[0]
   u,p,q,a=features
   flipped=policy.score((u,p,q,a.flip(0)))[0]
   good+=int(base!=flipped);total+=1
 return {"correct":good,"total":total}

def evaluate_block(cases,policy,encoder,proto):
 funcs={
  "v94_neural_only":policy.neural,
  "v95_separate_constraint_gate":policy.hybrid,
  "policy_only_ignore_preference":policy.option_only,
  "privileged_fixture_rule":symbolic_baseline,
  "privileged_semantic_prototype":lambda c,s:semantic_baseline(c,s,encoder,proto),
 }
 return {name:{"scores":scores,"cases":rows}
         for name,fn in funcs.items()
         for scores,rows in [evaluate(cases,fn)]}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--seed",type=int,default=11)
 p.add_argument("--steps",type=int,default=600)
 p.add_argument("--epochs",type=int,default=45)
 p.add_argument("--policy-epochs",type=int,default=30)
 p.add_argument("--policy-lr",type=float,default=.0015)
 p.add_argument("--batch",type=int,default=16)
 p.add_argument("--bridge-width",type=int,default=112)
 p.add_argument("--lr",type=float,default=.002)
 p.add_argument("--train-n",type=int,default=384)
 p.add_argument("--cold-n",type=int,default=128)
 p.add_argument("--output",default="compass-v95-results.json")
 args=p.parse_args()
 assert args.train_n==384 and args.cold_n==128 and args.seed==11, "fixed registered design"
 start=time.monotonic()
 torch.set_num_threads(2);random.seed(args.seed);torch.manual_seed(args.seed)
 training,reused_cold=build(args.seed,args.train_n,args.cold_n)
 # Fresh blind cases built from PRECOMMITTED NEW families; do not inspect until frozen.
 train=[c for i,c in enumerate(training) if i%5]
 dev=[c for i,c in enumerate(training) if i%5==0]
 enc=FrozenSemanticEncoder();enc.cache_training_vocab()
 class MemoryArgs:pass
 conf=MemoryArgs()
 conf.width=48;conf.seed=args.seed;conf.steps=args.steps;conf.batch=32
 conf.lr=.003;conf.check_every=50
 memory,mem_info=train_model(conf,enc)
 memory.eval();memory.requires_grad_(False)
 fc=FrozenFeatureCache(enc,memory)
 bridge,bridge_info=train_bridge(args,fc,train,dev)
 bridge.eval();bridge.requires_grad_(False)
 head,threshold,head_info=train_constraint_head(args,fc,train,dev)
 assert not any(p.requires_grad for p in bridge.parameters())
 assert not any(p.requires_grad for p in head.parameters())
 # Model and threshold are now LOCKED. No adaptation to any cold results.
 selected=fresh_cold(args.seed,args.train_n,args.cold_n)
 revoked=revoked_stress(selected)
 policy=Policies(head,threshold,bridge,fc)
 proto=prototypes(enc)
 result_old=evaluate_block(reused_cold,policy,enc,proto)
 result_new=evaluate_block(selected,policy,enc,proto)
 result_rev=evaluate_block(revoked,policy,enc,proto)
 routing=routing_audit(selected,policy)
 revoke_route=routing_audit(revoked,policy)
 symmetry=order_symmetry(selected,policy)
 rnew=result_new
 n=rnew["v94_neural_only"]["scores"]
 h=rnew["v95_separate_constraint_gate"]["scores"]
 neutral_regressions=sum(
   a["both_correct"] and not b["both_correct"]
   for a,b in zip(rnew["v94_neural_only"]["cases"],
                  rnew["v95_separate_constraint_gate"]["cases"])
   if not a["binding"])
 gates={
  "new_binding_pairs_at_least_48_of_64":h["binding"]["both_correct"]>=48,
  "new_binding_improves_over_v94_neural":h["binding"]["both_correct"]>n["binding"]["both_correct"],
  "new_neutral_pairs_at_least_45_of_64":h["neutral"]["both_correct"]>=45,
  "new_neutral_regressions_at_most_3":neutral_regressions<=3,
  "new_binding_detection_sensitivity_at_least_90pct":routing["binding_detection_sensitivity"]>=.9,
  "new_binding_detection_specificity_at_least_90pct":routing["binding_detection_specificity"]>=.9,
  "revoked_requirement_override_rate_at_most_5pct":
   revoke_route["neutral_false_overrides"]["overrides"]<=.05*len(revoked),
  "candidate_swap_symmetry_at_least_95pct":symmetry["correct"]>=.95*symmetry["total"],
 }
 verdict="PROVISIONAL_SYNTHETIC_PROGRESS_NO_LM_INTEGRATION" if all(gates.values()) else "NO_GO_TO_LM_ADAPTER_INTEGRATION"
 print("V95 HEAD LOCKED",json.dumps({"threshold":threshold,"dev":head_info["dev_at_threshold"]}),flush=True)
 for title,data in [("REUSED_V94",result_old),("NEW_FAMILIES",result_new),("REVOKED",result_rev)]:
  print("V95",title,json.dumps({k:v["scores"] for k,v in data.items()}),flush=True)
 weights={**{"bridge."+k:v.detach().cpu().contiguous() for k,v in bridge.state_dict().items()},
          **{"constraint."+k:v.detach().cpu().contiguous() for k,v in head.state_dict().items()}}
 save_file(weights,str(Path(args.output).with_suffix(".safetensors")))
 report={
  "version":"9.5","verdict":verdict,"source":"researcher-authored synthetic tasks",
  "pre_registered_protocol":"experiments/compass_v9_5/EXPERIMENT_V9_5.md",
  "reused_v94_cold_not_blind":result_old,
  "fresh_v95_family_cold_not_independently_human_created":result_new,
  "revoked_obligation_stress":result_rev,
  "detection_audit_fresh":routing,"detection_audit_revoked":revoke_route,
  "option_order_symmetry":symmetry,"fresh_neutral_regressions":neutral_regressions,
  "registered_gates":gates,"memory_training":mem_info,"bridge_training":bridge_info,
  "constraint_head_training":head_info,"time_seconds":round(time.monotonic()-start,2),
  "limitations":[
   "Synthetic researcher-authored gold and tasks, not independent human dilemmas.",
   "Shared history paraphrases and simple two-way preferences remain.",
   "Threshold chosen only using repeated TRAIN-template dev, not cold generalization.",
   "Constraint validity, revocation and authority are only linguistic descriptions, never authenticated.",
   "This is not a neural adapter inside the frozen 360M LLM, nor real continual learning.",
   "Privileged fixture baselines are upper-bound sanity checks, not equal-access controls.",
   "No re-run of frozen 360M baseline on the new V9.5 scenarios."
  ]
 }
 Path(args.output).write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
 print("V95 FINAL",json.dumps({"verdict":verdict,"fresh_hybrid":h,
  "fresh_v94_neural":n,"fresh_detection":routing,
  "revoked_detection":revoke_route,"neutral_regressions":neutral_regressions,
  "symmetry":symmetry,"gates":gates,
  "seconds":report["time_seconds"]}),flush=True)

if __name__=="__main__":
 main()
