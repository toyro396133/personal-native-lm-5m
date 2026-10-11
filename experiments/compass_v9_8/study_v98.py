"""V9.8 — separate trainable personal/project concept-to-option heads.

Synthetic-only. Personal head never sees a project tensor; project head never
sees the user's numeric state. Learned question-only router chooses arm without
receiving cold scope / binding / gold / action-order metadata.
"""
import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

from study_v93 import FrozenSemanticEncoder,train_model
from study_v94 import FrozenFeatureCache,train_bridge,evaluate
from fixtures_v94 import build
from fixtures_v98 import build_cold,perturb,audit as fixture_audit
from fixtures_v97 import new_cases as v97_cases

DEV_FAMILIES={"documentation","reporting"}

class IntentReadout(nn.Module):
 """Permutation-equivariant action scorer given ONE authoritative 48D state."""
 def __init__(self,width=96):
  super().__init__()
  self.state=nn.Sequential(nn.Linear(48,width),nn.LayerNorm(width),nn.Tanh())
  self.question=nn.Sequential(nn.Linear(384,width),nn.Tanh())
  self.action=nn.Sequential(nn.Linear(384,width),nn.Tanh())
  self.scorer=nn.Sequential(nn.Linear(width*5,128),nn.GELU(),
                             nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
 def forward(self,state,question,actions):
  a=self.action(actions)
  s=self.state(state).expand(actions.shape[0],-1)
  q=self.question(question).expand(actions.shape[0],-1)
  values=torch.cat((a,s,q,a*s,a*q),dim=-1)
  return self.scorer(values).flatten()

class ScopeRouter(nn.Module):
 """Only receives text embedding of new task. No fixture scope at inference."""
 def __init__(self):
  super().__init__()
  self.net=nn.Sequential(nn.Linear(384,48),nn.LayerNorm(48),nn.GELU(),
                         nn.Linear(48,2))
 def forward(self,q):return self.net(q)

def only_state(features,scope):
 u,p,q,options=features
 return (u if scope=="user" else p),q,options

def signed_pair_loss(head,fa,fb,case,scope):
 s0,q0,a0=only_state(fa,scope)
 s1,q1,a1=only_state(fb,scope)
 v0=head(s0,q0,a0);v1=head(s1,q1,a1)
 margin0=v0[0]-v0[1];margin1=v1[0]-v1[1]
 sign0=1 if case["original"]["gold"]==0 else -1
 sign1=1 if case["changed"]["gold"]==0 else -1
 assert sign0==-sign1 and not case["binding"]
 return (.5*(F.softplus(.7-sign0*margin0)+
              F.softplus(.7-sign1*margin1))
         +.35*F.softplus(.8-sign0*(margin0-margin1)))

def train_scope_head(args,cache,train,dev,scope):
 torch.manual_seed(args.seed+(880 if scope=="user" else 990))
 head=IntentReadout()
 optim=torch.optim.AdamW(head.parameters(),lr=args.head_lr,weight_decay=.002)
 t=[(cache.pair(c,"original"),cache.pair(c,"changed"),c) for c in train if c["scope"]==scope]
 d=[(cache.pair(c,"original"),cache.pair(c,"changed"),c) for c in dev if c["scope"]==scope]
 assert len(t)>=48 and len(d)>=16,(scope,len(t),len(d))
 best=-1;weights=None;selected_epoch=0;logs=[]
 for epoch in range(1,args.head_epochs+1):
  head.train()
  indices=list(range(len(t)))
  random.Random(args.seed*1180+epoch).shuffle(indices)
  losses=[]
  for start in range(0,len(indices),args.batch):
   batch=indices[start:start+args.batch]
   loss=torch.stack([signed_pair_loss(head,t[idx][0],t[idx][1],t[idx][2],scope)
                     for idx in batch]).mean()
   optim.zero_grad(set_to_none=True)
   loss.backward()
   nn.utils.clip_grad_norm_(head.parameters(),1.5)
   optim.step();losses.append(float(loss.detach()))
  head.eval()
  right=0
  with torch.no_grad():
   for a,b,c in d:
    s0,q0,o0=only_state(a,scope)
    s1,q1,o1=only_state(b,scope)
    right+=int(int(head(s0,q0,o0)[1]>head(s0,q0,o0)[0])==c["original"]["gold"] and
               int(head(s1,q1,o1)[1]>head(s1,q1,o1)[0])==c["changed"]["gold"])
  if right>best:
   best=right;selected_epoch=epoch
   weights={k:v.detach().clone() for k,v in head.state_dict().items()}
  if epoch==1 or epoch%20==0 or epoch==args.head_epochs:
   print("V98 HEAD",json.dumps({"seed":args.seed,"scope":scope,"epoch":epoch,
    "dev_both":right,"dev_n":len(d),"loss":round(sum(losses)/len(losses),5)}),flush=True)
  logs.append({"epoch":epoch,"dev_both":right})
 head.load_state_dict(weights);head.eval();head.requires_grad_(False)
 return head,{"scope":scope,"train_pairs":len(t),"group_heldout_dev_pairs":len(d),
              "chosen_epoch":selected_epoch,"dev_both":best,
              "trainable_weights":sum(p.numel() for p in head.parameters()),
              "curve":logs}

def train_router(args,cache,train,dev):
 torch.manual_seed(args.seed+701)
 router=ScopeRouter()
 optim=torch.optim.AdamW(router.parameters(),lr=args.router_lr,weight_decay=.002)
 t=[(cache.pair(c,"original")[2],int(c["scope"]=="project")) for c in train]
 d=[(cache.pair(c,"original")[2],int(c["scope"]=="project")) for c in dev]
 assert len(t)>=128 and len(d)>=32
 weights=None;best=-1;chosen_epoch=0;logs=[]
 for epoch in range(1,args.router_epochs+1):
  router.train()
  indices=list(range(len(t)));random.Random(args.seed*570+epoch).shuffle(indices)
  losses=[]
  for st in range(0,len(indices),args.batch):
   ix=indices[st:st+args.batch]
   qs=torch.stack([t[i][0] for i in ix])
   targets=torch.tensor([t[i][1] for i in ix],dtype=torch.long)
   loss=F.cross_entropy(router(qs),targets)
   optim.zero_grad(set_to_none=True);loss.backward()
   nn.utils.clip_grad_norm_(router.parameters(),1.5)
   optim.step();losses.append(float(loss.detach()))
  router.eval()
  with torch.no_grad():
   good=sum(int(torch.argmax(router(q)).item()==label) for q,label in d)
  if good>best:
   weights={k:v.detach().clone() for k,v in router.state_dict().items()}
   best=good;chosen_epoch=epoch
  if epoch==1 or epoch%20==0 or epoch==args.router_epochs:
   print("V98 ROUTER",json.dumps({"seed":args.seed,"epoch":epoch,
                                  "dev_correct":good,"dev_n":len(d)}),flush=True)
  logs.append({"epoch":epoch,"dev_correct":good})
 router.load_state_dict(weights);router.eval();router.requires_grad_(False)
 return router,{"train_n":len(t),"dev_n":len(d),"dev_correct":best,
                "chosen_epoch":chosen_epoch,"curve":logs}

class IsolatedChooser:
 def __init__(self,user_head,project_head,router,cache):
  self.user_head=user_head
  self.project_head=project_head
  self.router=router
  self.cache=cache
 @torch.no_grad()
 def from_features(self,features,forced_scope=None):
  u,p,q,opts=features
  # forced_scope is available ONLY in the labeled-oracle diagnostic ablation.
  scope=forced_scope if forced_scope is not None else ("project" if torch.argmax(self.router(q)).item()==1 else "user")
  state=u if scope=="user" else p
  head=self.user_head if scope=="user" else self.project_head
  score=head(state,q,opts)
  return int(score[1]>score[0]),scope
 def predict(self,case,side):
  return self.from_features(self.cache.pair(case,side))[0]
 def oracle(self,case,side):
  return self.from_features(self.cache.pair(case,side),forced_scope=case["scope"])[0]

class NumericPrototype:
 """Same numeric states, pretrained question and option vectors, no oracle scope.

 Training-only state/option prototypes; router from question similarity to
 TRAIN examples, no fresh test metadata.
 """
 def __init__(self,cache,train):
  self.cache=cache
  state=defaultdict(list);question=defaultdict(list);actions=defaultdict(list)
  for c in train:
   typ=c["scope"]
   for side in ("original","changed"):
    u,p,q,opt=cache.pair(c,side)
    semantic=c[side]["gold"]^c["swapped"]
    state[(typ,semantic)].append(u if typ=="user" else p)
   _,_,q,opt=cache.pair(c,"original")
   question[typ].append(q)
   for mode in (0,1):
    actions[(typ,mode)].append(opt[mode^c["swapped"]])
  self.states={k:F.normalize(torch.stack(v).mean(0),dim=0) for k,v in state.items()}
  self.questions={k:F.normalize(torch.stack(v).mean(0),dim=0) for k,v in question.items()}
  self.actions={k:F.normalize(torch.stack(v).mean(0),dim=0) for k,v in actions.items()}
 @torch.no_grad()
 def predict(self,c,side):
  u,p,q,opts=self.cache.pair(c,side)
  inferred=max(("user","project"),key=lambda typ:float(F.cosine_similarity(
    q,self.questions[typ],dim=0)))
  st=u if inferred=="user" else p
  sem=max((0,1),key=lambda m:float(F.cosine_similarity(st,self.states[(inferred,m)],dim=0)))
  av=self.actions[(inferred,sem)]
  return int(torch.argmax(torch.mv(F.normalize(opts,dim=-1),av)).item())

def as_result(cases,fn):
 score,detail=evaluate(cases,fn)
 return {"score":score,"cases":detail}

def compare_noise(cases,variants,model):
 changed=0;total=0
 by_scope={}
 for scope,variant in variants.items():
  base=[c for c in cases if c["scope"]==scope]
  diff=0
  for c in base:
   x=perturb(c,variant)
   for side in ("original","changed"):
    diff+=int(model.predict(c,side)!=model.predict(x,side))
    total+=1
  by_scope[variant]={"cases":len(base),"tested_decisions":len(base)*2,
                     "unexpected_changes":diff}
  changed+=diff
 return {"overall_decisions_checked":total,"unexpected_changes":changed,
         "categories":by_scope}

def choice_order_equivariance(cases,model):
 total=good=0
 for c in cases[:64]:
  for side in ("original","changed"):
   u,p,q,opts=model.cache.pair(c,side)
   a=model.from_features((u,p,q,opts))[0]
   b=model.from_features((u,p,q,opts.flip(0)))[0]
   good+=int(a!=b);total+=1
 return {"correct":good,"total":total}

def route_accuracy(cases,model):
 rows=[];correct=0
 for c in cases:
  _,scope=model.from_features(model.cache.pair(c,"original"))
  good=scope==c["scope"]
  correct+=int(good)
  rows.append({"id":c["id"],"family":c["family"],"truth":c["scope"],"predicted":scope})
 return {"correct":correct,"total":len(cases),"cases":rows}

def scope_regressions(cases,base,candidate,scope):
 a={x["id"]:x for x in base["cases"] if x["scope"]==scope}
 b={x["id"]:x for x in candidate["cases"] if x["scope"]==scope}
 assert a.keys()==b.keys()
 return sum(int(row["both_correct"] and not b[k]["both_correct"]) for k,row in a.items())

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--seed",type=int,default=11)
 ap.add_argument("--steps",type=int,default=600)
 ap.add_argument("--epochs",type=int,default=45)
 ap.add_argument("--batch",type=int,default=16)
 ap.add_argument("--lr",type=float,default=.002)
 ap.add_argument("--bridge-width",type=int,default=112)
 ap.add_argument("--head-epochs",type=int,default=100)
 ap.add_argument("--head-lr",type=float,default=.0015)
 ap.add_argument("--router-epochs",type=int,default=100)
 ap.add_argument("--router-lr",type=float,default=.001)
 ap.add_argument("--output",default="compass-v98-results.json")
 args=ap.parse_args();assert args.seed in (11,19,23)
 t0=time.monotonic()
 torch.set_num_threads(2)
 torch.manual_seed(args.seed);random.seed(args.seed)
 train_cases,_=build(args.seed,384,128)
 encoder=FrozenSemanticEncoder();encoder.cache_training_vocab()
 class Args:pass
 ma=Args();ma.width=48;ma.seed=args.seed;ma.steps=args.steps
 ma.batch=32;ma.lr=.003;ma.check_every=50
 memory,mem_info=train_model(ma,encoder)
 memory.requires_grad_(False);memory.eval()
 cache=FrozenFeatureCache(encoder,memory)
 # Published baseline uses within-training per-case dev split.
 basetrain=[c for i,c in enumerate(train_cases) if i%5]
 basedev=[c for i,c in enumerate(train_cases) if i%5==0]
 baseline,base_info=train_bridge(args,cache,basetrain,basedev)
 baseline.eval();baseline.requires_grad_(False)
 # New heads use ONLY neutral examples and family-heldout TRAIN development.
 eligible=[c for c in train_cases if not c["binding"]]
 tr=[c for c in eligible if c["family"] not in DEV_FAMILIES]
 dv=[c for c in eligible if c["family"] in DEV_FAMILIES]
 assert {c["family"] for c in tr}.isdisjoint({c["family"] for c in dv})
 user_head,uinfo=train_scope_head(args,cache,tr,dv,"user")
 project_head,pinfo=train_scope_head(args,cache,tr,dv,"project")
 router,rinfo=train_router(args,cache,tr,dv)
 proto=NumericPrototype(cache,tr)
 # All learning and hyperparameters fixed BEFORE fresh test feature creation.
 for m in (memory,baseline,user_head,project_head,router):
  assert not any(p.requires_grad for p in m.parameters())
 new=build_cold(args.seed)
 chk=fixture_audit(new)
 model=IsolatedChooser(user_head,project_head,router,cache)
 @torch.no_grad()
 def base(c,s):
  logits=baseline(*cache.pair(c,s))
  return int(logits[1]>logits[0])
 methods={
  "published_v94_joint_bridge":base,
  "v98_text_routed_isolated_heads":model.predict,
  "v98_oracle_scope_ablation":model.oracle,
  "numeric_prototype_text_routed_reference":proto.predict,
  "no_history_choose_option_a":lambda c,s:0,
 }
 outcome={name:as_result(new,fn) for name,fn in methods.items()}
 for name,res in outcome.items():
  print("V98 NEW",args.seed,name,json.dumps(res["score"]),flush=True)
 routing=route_accuracy(new,model)
 isolation=compare_noise(new,{
   "user":"personal_project_noise",
   "project":"project_user_noise"
 },model)
 switch=compare_noise(new,{"user":"personal_workspace_switch"},model)
 separate_user=[c for c in new if c["scope"]=="user"]
 opposed=[perturb(c,"personal_project_noise") for c in separate_user]
 opposed_score=as_result(opposed,model.predict)
 order=choice_order_equivariance(new,model)
 # Known V9.7 test cases are explicitly retrospective, not model selection data.
 prev,_=v97_cases(args.seed)
 retros=[c for c in prev if c["scope"]=="user" and c["scenario"] in
         ("no_claim","revoked","expired","other_project")]
 assert len(retros)==64
 retro={"baseline":as_result(retros,base),
        "candidate":as_result(retros,model.predict)}
 base_score=outcome["published_v94_joint_bridge"]["score"]
 candidate=outcome["v98_text_routed_isolated_heads"]["score"]
 user_score=candidate["user"]["both_correct"]
 base_user=base_score["user"]["both_correct"]
 project_score=candidate["project"]["both_correct"]
 base_project=base_score["project"]["both_correct"]
 gates={
  "user_pairs_at_least_48_of_64":user_score>=48,
  "user_gain_at_least_8_pairs_vs_matched_v94":user_score-base_user>=8,
  "project_pairs_at_least_48_of_64":project_score>=48,
  "project_loss_at_most_4_vs_matched_v94":base_project-project_score<=4,
  "task_scope_router_correct_at_least_120_of_128":routing["correct"]>=120,
  "personal_other_project_history_invariance_128_of_128":
   isolation["categories"]["personal_project_noise"]["unexpected_changes"]==0,
  "project_irrelevant_user_history_invariance_128_of_128":
   isolation["categories"]["project_user_noise"]["unexpected_changes"]==0,
  "personal_workspace_switch_invariance_128_of_128":
   switch["unexpected_changes"]==0,
  "personal_opposed_project_distraction_at_least_45_64":
   opposed_score["score"]["all"]["both_correct"]>=45,
  "candidate_option_swap_equivariance_at_least_95pct":
   order["correct"]>=.95*order["total"],
 }
 verdict=("PROVISIONAL_SYNTHETIC_PERSONAL_TRANSFER_PASS_SEED"
          if all(gates.values()) else "NO_GO_TO_LM_ADAPTER_INTEGRATION")
 weights={}
 for prefix,mod in (("user",user_head),("project",project_head),
                   ("router",router),("baseline",baseline)):
  weights.update({prefix+"."+k:v.detach().cpu().contiguous()
                  for k,v in mod.state_dict().items()})
 save_file(weights,str(Path(args.output).with_suffix(".safetensors")))
 report={
  "version":"9.8","seed":args.seed,"verdict":verdict,
  "new_cold_fixture_audit":chk,"new_cold_matched":outcome,
  "new_cold_question_scope_router":routing,
  "unrelated_history_counterfactual_invariance":isolation,
  "personal_workspace_switch_invariance":switch,
  "new_personal_opposed_project_distraction":opposed_score,
  "candidate_option_order_symmetry":order,
  "v97_personal_unconstrained_retrospective_only":retro,
  "project_regressions_vs_v94":scope_regressions(new,
    outcome["published_v94_joint_bridge"],
    outcome["v98_text_routed_isolated_heads"],"project"),
  "registered_gates":gates,
  "memory_train":mem_info,"baseline_bridge_train":base_info,
  "new_user_readout_train":uinfo,"new_project_readout_train":pinfo,
  "new_router_train":rinfo,"runtime_seconds":round(time.monotonic()-t0,2),
  "limitations":[
   "All test tasks and binary choice labels written by the researcher; no independent human task prompts.",
   "Global and project modes are narrow synthetic axes (inspect vs experiment, project purpose vs API contract).",
   "Train and test history statements share underlying one-bit intent generator, despite fresh paraphrases.",
   "Router learns personal/project role from question wording and is not an authentic general intent interpreter.",
   "Neural readouts are external tiny models, not inside frozen 360M Transformer.",
   "Baseline uses within-family train/dev and new heads family-heldout development; equal test inputs but not equal training objective.",
   "The numeric prototype is a simple fitted embedding reference, not an optimized pretrained LLM or RAG."
  ]
 }
 Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
 print("V98 FINAL",json.dumps({"seed":args.seed,"verdict":verdict,
  "baseline":base_score,"candidate":candidate,
  "prototype":outcome["numeric_prototype_text_routed_reference"]["score"],
  "route":{k:v for k,v in routing.items() if k!="cases"},
  "isolation":isolation,"workspace_switch":switch,
  "opposed":opposed_score["score"],"symmetry":order,"gates":gates,
  "runtime_seconds":report["runtime_seconds"]}),flush=True)
if __name__=="__main__":main()
