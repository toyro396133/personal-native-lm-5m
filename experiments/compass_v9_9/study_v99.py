"""V9.9 explicit mock-source routing + measured personal concept transfer.

NO cold gold/scope/family enters prediction. The mock signed origin is an
extra trusted (synthetic) control-plane input: model does NOT infer ownership.
"""
import argparse
import hmac
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from safetensors.torch import save_file

from fixtures_v94 import build
from fixtures_v99 import MOCK_SIGNERS,canonical,fresh,uncertainty,perturb,audit
from study_v93 import FrozenSemanticEncoder,train_model
from study_v94 import FrozenFeatureCache,train_bridge,evaluate
from study_v98 import (train_scope_head,train_router,IsolatedChooser,
                        DEV_FAMILIES)

ASK=2
class SourceResolver:
 """Only accepts an explicit UI request-origin contract signed with mock key."""
 def __init__(self,keys):self.keys=keys
 def route(self,records,workspace):
  if not isinstance(records,list) or not records:return None,"MISSING_ORIGIN"
  authorized=[]
  for record in records:
   if not isinstance(record,dict):return None,"INVALID_RECORD"
   issuer=record.get("issuer");kind=record.get("intent_type")
   event_type=record.get("event_type")
   claimed_ws=record.get("workspace")
   if issuer=="profile-ui" and kind=="personal" and event_type=="PERSONAL_PROFILE_DECISION":
    if claimed_ws!="*":return None,"INVALID_PERSONAL_SCOPE"
   elif issuer=="workspace-ui" and kind=="project" and event_type=="WORKSPACE_GOAL_DECISION":
    if claimed_ws!=workspace:return None,"FOREIGN_WORKSPACE"
   else:
    return None,"UNAUTHORIZED_ORIGIN"
   key=self.keys.get(issuer)
   mac=record.get("mac")
   if not key or not isinstance(mac,str):return None,"UNVERIFIED_ORIGIN"
   expected=hmac.new(key,canonical(record),"sha256").hexdigest()
   if not hmac.compare_digest(expected,mac):return None,"BAD_MAC"
   authorized.append(kind)
  if len(authorized)!=1:return None,"AMBIGUOUS_ORIGIN"
  return authorized[0],"VERIFIED_MOCK_ORIGIN"

class SourceChooser:
 def __init__(self,head_user,head_project,cache,resolver):
  self.user=head_user;self.project=head_project
  self.cache=cache;self.resolver=resolver
 def route(self,case):
  return self.resolver.route(case.get("source_events"),case["active_workspace"])
 @torch.no_grad()
 def from_features(self,features,owner):
  u,p,q,options=features
  state=u if owner=="personal" else p
  head=self.user if owner=="personal" else self.project
  logits=head(state,q,options)
  return int(logits[1]>logits[0])
 def predict(self,case,side):
  owner,_=self.route(case)
  if owner is None:return ASK
  return self.from_features(self.cache.pair(case,side),owner)

class SourcePrototype:
 """Same mock-origin access; prototypes trained on original TRAIN only.

 Prototype state and semantic-option direction are determined by training
 examples. No hidden cold original gold or option swap is used in predict().
 """
 def __init__(self,cache,train,resolver):
  self.cache=cache;self.resolver=resolver
  states=defaultdict(list);actions=defaultdict(list)
  for c in train:
   owner="personal" if c["scope"]=="user" else "project"
   for side in ("original","changed"):
    u,p,q,option_vecs=cache.pair(c,side)
    sem=int(c[side]["gold"]^c["swapped"])
    states[(owner,sem)].append(u if owner=="personal" else p)
   _,_,_,option_vecs=cache.pair(c,"original")
   for mode in (0,1):
    actions[(owner,mode)].append(option_vecs[mode^c["swapped"]])
  self.state={k:F.normalize(torch.stack(v).mean(0),dim=0)
              for k,v in states.items()}
  self.action={k:F.normalize(torch.stack(v).mean(0),dim=0)
               for k,v in actions.items()}
 @torch.no_grad()
 def predict(self,case,side):
  owner,_=self.resolver.route(case["source_events"],case["active_workspace"])
  if owner is None:return ASK
  u,p,_,opts=self.cache.pair(case,side)
  state=u if owner=="personal" else p
  semantic=int(torch.argmax(torch.tensor([
    F.cosine_similarity(state,self.state[(owner,k)],dim=0).item()
    for k in (0,1)])).item())
  query=self.action[(owner,semantic)]
  return int(torch.argmax(torch.mv(F.normalize(opts,dim=-1),query)).item())

def evaluate_method(cases,predict):
 scores,rows=evaluate(cases,predict)
 return {"score":scores,"cases":rows}

def route_audit(cases,source):
 right=0;traces=[]
 for c in cases:
  resolved,reason=source.route(c)
  declared="personal" if c["scope"]=="user" else "project"
  right+=int(resolved==declared)
  traces.append({"id":c["id"],"expected_declaration":declared,
                 "observed_route":resolved,"reason":reason})
 return {"correct":right,"total":len(cases),"cases":traces}

def stress_audit(cases,model):
 ans=[]
 for c in cases:
  original=model.predict(c,"original")
  changed=model.predict(c,"changed")
  ans.append({"id":c["id"],"scenario":c["uncertainty_type"],
              "predicted":[original,changed],
              "both_asked":original==ASK and changed==ASK})
 return {"total":len(cases),"all_asked":sum(r["both_asked"] for r in ans),
         "rows":ans}

def compare_perturb(cases,model,scope,variant):
 chosen=[c for c in cases if c["scope"]==scope]
 changes=0;rows=[]
 for c in chosen:
  item=perturb(c,variant)
  for side in ("original","changed"):
   a=model.predict(c,side);b=model.predict(item,side)
   changes+=int(a!=b)
   rows.append({"id":c["id"],"side":side,"before":a,"after":b})
 return {"paired_cases":len(chosen),"decision_comparisons":len(rows),
         "changed":changes,"rows":rows}

def option_symmetry(cases,model):
 good=total=0
 for c in cases[:64]:
  owner,_=model.route(c)
  assert owner in ("personal","project")
  for side in ("original","changed"):
   u,p,q,opts=model.cache.pair(c,side)
   x=model.from_features((u,p,q,opts),owner)
   y=model.from_features((u,p,q,opts.flip(0)),owner)
   total+=1;good+=int(x!=y)
 return {"correct":good,"total":total}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--seed",type=int,default=11)
 p.add_argument("--steps",type=int,default=600)
 p.add_argument("--epochs",type=int,default=45)
 p.add_argument("--head-epochs",type=int,default=100)
 p.add_argument("--head-lr",type=float,default=.0015)
 p.add_argument("--router-epochs",type=int,default=100)
 p.add_argument("--router-lr",type=float,default=.001)
 p.add_argument("--batch",type=int,default=16)
 p.add_argument("--bridge-width",type=int,default=112)
 p.add_argument("--lr",type=float,default=.002)
 p.add_argument("--output",default="compass-v99-results.json")
 a=p.parse_args();assert a.seed in (11,19,23)
 t0=time.monotonic()
 torch.set_num_threads(2);random.seed(a.seed);torch.manual_seed(a.seed)
 training,_=build(a.seed,384,128)
 enc=FrozenSemanticEncoder();enc.cache_training_vocab()
 class X:pass
 margs=X();margs.width=48;margs.seed=a.seed;margs.steps=a.steps
 margs.batch=32;margs.lr=.003;margs.check_every=50
 memory,meminfo=train_model(margs,enc)
 memory.eval();memory.requires_grad_(False)
 cache=FrozenFeatureCache(enc,memory)
 eligible=[c for c in training if not c["binding"]]
 train=[c for c in eligible if c["family"] not in DEV_FAMILIES]
 dev=[c for c in eligible if c["family"] in DEV_FAMILIES]
 assert {c["family"] for c in train}.isdisjoint({c["family"] for c in dev})
 personal,pinfo=train_scope_head(a,cache,train,dev,"user")
 project,projinfo=train_scope_head(a,cache,train,dev,"project")
 router,routeinfo=train_router(a,cache,train,dev)
 reference=SourcePrototype(cache,train,SourceResolver(MOCK_SIGNERS))
 # Frozen before NEW V9.9 cold is created, no test label checkpoint selection.
 assert all(not any(x.requires_grad for x in m.parameters())
            for m in (memory,personal,project,router))
 cases=fresh(a.seed);check=audit(cases)
 uncertain=uncertainty(cases)
 source=SourceChooser(personal,project,cache,SourceResolver(MOCK_SIGNERS))
 neural=IsolatedChooser(personal,project,router,cache)
 model_scores={
  "source_routed_heads":evaluate_method(cases,source.predict),
  "question_only_routed_same_heads":evaluate_method(cases,neural.predict),
  "source_routed_numeric_prototypes":evaluate_method(cases,reference.predict),
  "constant_option_zero":evaluate_method(cases,lambda c,s:0),
 }
 for name,result in model_scores.items():
  print("V99 NEW",a.seed,name,json.dumps(result["score"]),flush=True)
 routing=route_audit(cases,source)
 uncertainty_results=stress_audit(uncertain,source)
 personal_noise=compare_perturb(cases,source,"user","personal_other_project_noise")
 personal_switch=compare_perturb(cases,source,"user","personal_workspace_switch")
 project_noise=compare_perturb(cases,source,"project","project_irrelevant_user_noise")
 symmetry=option_symmetry(cases,source)
 scored=model_scores["source_routed_heads"]["score"]
 question=model_scores["question_only_routed_same_heads"]["score"]
 proto=model_scores["source_routed_numeric_prototypes"]["score"]
 user=scored["user"]["both_correct"]
 q_user=question["user"]["both_correct"]
 target_project=scored["project"]["both_correct"]
 gates={
  "verified_origin_routes_128_128_no_ask":routing["correct"]==128,
  "unknown_or_contradictory_origin_ask_96_96":uncertainty_results["all_asked"]==96,
  "personal_project_history_invariance_128":personal_noise["changed"]==0,
  "personal_workspace_switch_invariance_128":personal_switch["changed"]==0,
  "project_irrelevant_user_invariance_128":project_noise["changed"]==0,
  "personal_head_at_least_48_64":user>=48,
  "personal_gain_or_router_already_good":user-q_user>=8 or (q_user>=user-7 and user>=48),
  "project_head_at_least_48_64":target_project>=48,
  "head_or_prototype_personal_at_least_48_64":max(user,proto["user"]["both_correct"])>=48,
  "option_swap_symmetry_at_least_95pct":symmetry["correct"]>=.95*symmetry["total"],
 }
 verdict="PROVISIONAL_SYNTHETIC_SOURCE_ROUTING_PASS_NOT_SELF_MODEL" if all(gates.values()) else "NO_GO_TO_LM_ADAPTER_INTEGRATION"
 weights={}
 for prefix,model in (("personal",personal),("project",project),("router",router)):
  weights.update({prefix+"."+k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()})
 save_file(weights,str(Path(a.output).with_suffix(".safetensors")))
 out={
  "version":"9.9","seed":a.seed,"verdict":verdict,
  "fixture_audit":check,"new_cold_matched":model_scores,
  "mock_provenance_origin_audit":routing,"uncertain_origin_stress":uncertainty_results,
  "personal_ignores_other_project_histories":personal_noise,
  "personal_ignores_workspace_switch":personal_switch,
  "project_ignores_irrelevant_user_histories":project_noise,
  "option_swap_symmetry":symmetry,
  "registered_gates":gates,"memory_train":meminfo,
  "personal_train":pinfo,"project_train":projinfo,"question_router_train":routeinfo,
  "duration_seconds":round(time.monotonic()-t0,2),
  "limitations":[
   "Source event is synthetic explicitly declared input, constructed to match researcher gold scope.",
   "Demo HMAC keys in public code are not real authentication, and missing provenance abstention may be trivial.",
   "Same synthetic binary intent axes and three-message histories as previous runs.",
   "No independent human-written dilemmas, uncertain mixed ownership, source consent or real historical projects.",
   "Results cannot establish improved semantic personal understanding from provenance metadata.",
   "External isolated numeric readouts, not a 360M internal neural adapter."
  ]
 }
 Path(a.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
 print("V99 FINAL",json.dumps({
  "seed":a.seed,"verdict":verdict,
  "head":scored,"question_route":question,"prototype":proto,
  "verified_origin":routing["correct"],"unknown_asked":uncertainty_results["all_asked"],
  "personal_project_noise":personal_noise["changed"],"personal_switch":personal_switch["changed"],
  "project_user_noise":project_noise["changed"],"symmetry":symmetry,
  "gates":gates,"seconds":out["duration_seconds"]}),flush=True)

if __name__=="__main__":main()
