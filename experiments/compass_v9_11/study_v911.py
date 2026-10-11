"""COMPASS V9.11 persistent multi-slot personal/project principles.

This is NOT neural SELF training or an in-transformer adapter. Reuses English-
trained V9.3 GRU + V9.8 scope-specific heads WITHOUT optimization on V9.11.
Metadata-backed concept selection and policy checks are explicit software.
"""
import argparse
import copy
import datetime
import hmac
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from fixtures_v94 import build
from fixtures_v911 import (fresh,audit,tamper,sign_slot,sign_policy,
                           SLOT_SIGNERS,POLICY_KEY,canonical)
from fixtures_v1010 import uncertain
from fixtures_v99 import MOCK_SIGNERS
from study_v93 import FrozenSemanticEncoder,train_model
from study_v94 import FrozenFeatureCache
from study_v98 import train_scope_head,DEV_FAMILIES
from study_v99 import SourceChooser,SourceResolver
from study_v1010 import metrics_by

ASK=2
class VersionedCompassReader:
 def __init__(self,cache,user_head,project_head):
  self.cache=cache
  self.base=SourceChooser(user_head,project_head,cache,SourceResolver(MOCK_SIGNERS))
 def selection(self,case,side):
  owner,why=self.base.route(case)
  if owner is None:return None,{"mode":"ASK","reason":why}
  key="user" if owner=="personal" else case["active_workspace"]
  slots=case["compass_slots"][side][key]
  if not isinstance(slots,list) or not slots:
   return None,{"mode":"ASK","reason":"MISSING_SLOTS"}
  active=[];inactive=[]
  for s in slots:
   if not isinstance(s,dict):return None,{"mode":"ASK","reason":"BAD_SLOT_RECORD"}
   if s.get("owner_type")!=owner or s.get("scope_id")!=("*" if owner=="personal" else key):
    return None,{"mode":"ASK","reason":"SCOPE_MISMATCH"}
   if s.get("kind") not in ("preference","context"):
    return None,{"mode":"ASK","reason":"UNKNOWN_SLOT_KIND"}
   try:
    expect=hmac.new(SLOT_SIGNERS[owner].encode(),canonical(s),"sha256").hexdigest()
    if not hmac.compare_digest(expect,str(s.get("signature",""))):
     return None,{"mode":"ASK","reason":"UNVERIFIED_SLOT_SIGNATURE"}
   except (ValueError,TypeError):
    return None,{"mode":"ASK","reason":"MALFORMED_SLOT_SIGNATURE"}
   if not isinstance(s.get("evidence"),list) or not all(isinstance(t,str) and t for t in s["evidence"]):
    return None,{"mode":"ASK","reason":"INVALID_EVIDENCE"}
   if s.get("active") and s["kind"]=="preference":active.append(s)
   if not s.get("active") and s["kind"]=="preference":inactive.append(s)
  if len(active)!=1:return None,{"mode":"ASK","reason":"AMBIGUOUS_ACTIVE_PREFERENCE"}
  p=active[0]
  # Explicit version/supersession: a newer active preference can supersede older,
  # but an unrelated context slot must NOT rewrite a distinct preference slot.
  if any(int(s["version"])>=int(p["version"]) for s in inactive):
   return None,{"mode":"ASK","reason":"INCONSISTENT_VERSION"}
  old_ids={s["slot_id"] for s in inactive}
  if old_ids and p.get("supersedes") not in old_ids:
   return None,{"mode":"ASK","reason":"MISSING_SUPERSESSION_LINK"}
  if len(p["evidence"])!=3:return None,{"mode":"ASK","reason":"UNSUPPORTED_EVIDENCE_LENGTH"}
  fact={"mode":"PREFERENCE","owner":owner,"scope_id":p["scope_id"],
        "slot_id":p["slot_id"],"version":p["version"],
        "reason":p["reason"],"source":p["source"],
        "superseded":sorted(old_ids)}
  return p["evidence"],fact
 def reduced(self,case,side):
  ev,record=self.selection(case,side)
  if ev is None:return None,record
  c=copy.deepcopy(case)
  # Old FrozenFeatureCache stacks all user/project streams in one tensor and
  # requires equal sequence lengths. Non-authoritative streams are shape-only
  # inputs to the isolated readout; never used as the selected principle.
  # Truncate them to the TRAIN length while preserving active slot evidence.
  for stream in ("user","a","b"):
   c[side]["histories"][stream]=c[side]["histories"][stream][:3]
  key="user" if record["owner"]=="personal" else case["active_workspace"]
  c[side]["histories"][key]=list(ev)
  return c,record
 def predict(self,case,side):
  c,diag=self.reduced(case,side)
  if c is None:return ASK
  return self.base.predict(c,side)
 def logits(self,case,side):
  c,diag=self.reduced(case,side)
  if c is None:return None
  owner=diag["owner"]
  u,p,q,options=self.cache.pair(c,side)
  head=self.base.user if owner=="personal" else self.base.project
  state=u if owner=="personal" else p
  return head(state,q,options)

class MockPolicyController:
 """Illustrative typed contract with toy HMAC, same input for both methods."""
 def apply(self,c,action):
  x=c["policy_contract"]
  try:
   expect=sign_policy(x)
   if not isinstance(x.get("signature"),str) or not hmac.compare_digest(expect,x["signature"]):
    return ASK
  except Exception:return ASK
  scope=x.get("workspace")
  if scope not in ("*",c["active_workspace"]):return ASK
  if not x.get("verified"):return ASK
  try:
   current=datetime.date(2026,10,11)
   begin=datetime.date.fromisoformat(x["valid_from"])
   until=datetime.date.fromisoformat(x["valid_until"])
  except (ValueError,KeyError,TypeError):return ASK
  if begin>current or current>=until:return action
  if x.get("state")=="revoked" and x.get("revoked_on"):
   try:
    if datetime.date.fromisoformat(x["revoked_on"])<=current:return action
   except ValueError:return ASK
  if x.get("state")=="active":
   op=x["required_operation"]
   if op not in c["action_ids"]:return ASK
   return c["action_ids"].index(op)
  return action

def scores(cases,pred):
 result=metrics_by(cases,pred)
 d=result["summary"]
 return result,{"all":d["all"]["both_correct"],
                "en":d["language_en"]["both_correct"],
                "he":d["language_he"]["both_correct"],
                "active":d["regime_active"]["both_correct"],
                "user":d["scope_user"]["both_correct"],
                "project":d["scope_project"]["both_correct"],
                "neutral":sum(d["regime_"+s]["both_correct"]
                              for s in ("open","expired","revoked"))}

def source_ask(cases,reader):
 test=uncertain(cases)
 assert len(test)==32
 return sum(all(reader.predict(c,s)==ASK for s in ("original","changed")) for c in test)

def slot_ask(cases,reader):
 test=[tamper(c,"bad_signature" if i%2==0 else "unknown_scope")
       for i,c in enumerate(cases[:32])]
 return sum(all(reader.predict(c,s)==ASK for s in ("original","changed")) for c in test)

def slot_permutation(cases,reader):
 total=changed=0
 for c in cases:
  for typ in ("shuffle","superseded_last"):
   altered=tamper(c,typ)
   for side in ("original","changed"):
    before=reader.predict(c,side)
    after=reader.predict(altered,side)
    total+=1;changed+=int(before!=after)
 return {"decisions":total,"unexpected_changes":changed}

def noninterference(cases,reader,policy):
 total=changed=0
 # Do not alter the owned slot, but heavily alter ALL other inputs.
 for c in cases[:64]:
  altered=copy.deepcopy(c)
  owned="user" if c["scope"]=="user" else c["project"]
  for side in ("original","changed"):
   for key in ("user","a","b"):
    if key!=owned:
     altered[side]["histories"][key]=[
      "A foreign memory should never control this decision.",
      "Contradictory unrelated project concepts.",
      "Unrelated preference changes have no authority.",
      "This does not revoke the target slot.",
      "The foreign source changed again."
     ]
  for side in ("original","changed"):
   left=policy.apply(c,reader.predict(c,side))
   right=policy.apply(altered,reader.predict(altered,side))
   total+=1;changed+=int(left!=right)
 return {"decisions":total,"unexpected_changes":changed}

def symmetry(cases,reader):
 total=ok=0
 with torch.no_grad():
  for c in cases[:64]:
   for side in ("original","changed"):
    a=reader.logits(c,side)
    if a is None:continue
    v,diagnostic=reader.reduced(c,side)
    owner=diagnostic["owner"]
    u,p,q,opts=reader.cache.pair(v,side)
    head=reader.base.user if owner=="personal" else reader.base.project
    state=u if owner=="personal" else p
    b=head(state,q,opts.flip(0))
    ok+=int(torch.allclose(a,b.flip(0),atol=1e-5,rtol=1e-5))
    total+=1
 return {"correct":ok,"total":total}

def main():
 a=argparse.ArgumentParser()
 a.add_argument("--seed",type=int,default=11)
 a.add_argument("--steps",type=int,default=600)
 a.add_argument("--head-epochs",type=int,default=100)
 a.add_argument("--head-lr",type=float,default=.0015)
 a.add_argument("--batch",type=int,default=16)
 a.add_argument("--output",default="compass-v911-results.json")
 args=a.parse_args()
 assert args.seed in (11,19,23)
 start=time.monotonic()
 random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
 training,_=build(args.seed,384,128)
 encoder=FrozenSemanticEncoder();encoder.cache_training_vocab()
 class Args:pass
 conf=Args();conf.width=48;conf.seed=args.seed;conf.steps=args.steps
 conf.batch=32;conf.lr=.003;conf.check_every=50
 memory,mi=train_model(conf,encoder);memory.eval();memory.requires_grad_(False)
 cache=FrozenFeatureCache(encoder,memory)
 eligible=[c for c in training if not c["binding"]]
 tr=[c for c in eligible if c["family"] not in DEV_FAMILIES]
 dev=[c for c in eligible if c["family"] in DEV_FAMILIES]
 user,ui=train_scope_head(args,cache,tr,dev,"user")
 project,pi=train_scope_head(args,cache,tr,dev,"project")
 assert all(not p.requires_grad for m in (memory,user,project) for p in m.parameters())
 old=SourceChooser(user,project,cache,SourceResolver(MOCK_SIGNERS))
 reader=VersionedCompassReader(cache,user,project)
 policy=MockPolicyController()
 # At this point all learned weights and hyperparameters are frozen.
 test=fresh(args.seed)
 locked_audit=audit(test)
 methods={
  "old_full_history_no_policy":lambda c,s:old.predict(c,s),
  "old_full_history_plus_same_policy":lambda c,s:policy.apply(c,old.predict(c,s)),
  "new_compass_slot_no_policy":lambda c,s:reader.predict(c,s),
  "new_compass_slot_plus_same_policy":lambda c,s:policy.apply(c,reader.predict(c,s)),
  "first_three_position_shortcut_plus_same_policy":
    lambda c,s:policy.apply(c,old.predict(first_three(c,s),s)),
 }
 outputs={}
 keyscore={}
 for name,method in methods.items():
  evaluation,counts=scores(test,method)
  outputs[name]=evaluation
  keyscore[name]=counts
  print("V911 METHOD",json.dumps({"seed":args.seed,"method":name,
    "scores":counts}),flush=True)
 req=keyscore["new_compass_slot_plus_same_policy"]
 base=keyscore["old_full_history_plus_same_policy"]
 asks=source_ask(test,reader)
 invalid=slot_ask(test,reader)
 order=slot_permutation(test,reader)
 isolation=noninterference(test,reader,policy)
 perms=symmetry(test,reader)
 audit_trace=[]
 for c in test[:12]:
  details={}
  for side in ("original","changed"):
   _,meta=reader.selection(c,side)
   details[side]=meta
  audit_trace.append({"id":c["id"],"selected_records":details})
 gates={
  "valid_signed_mock_origin_all_128":all(reader.base.route(c)[0] in
    ("personal","project") for c in test),
  "bad_source_ASK_32_32":asks==32,
  "slot_order_and_supersession_invariance":order["unexpected_changes"]==0,
  "invalid_slot_provenance_ASK_32_32":invalid==32,
  "total_both_correct_at_least_40":req["all"]>=40,
  "total_gain_vs_equal_policy_full_history_at_least_20":req["all"]-base["all"]>=20,
  "english_both_correct_at_least_36":req["en"]>=36,
  "hebrew_both_correct_at_least_16":req["he"]>=16,
  "active_policy_compliance_32_32_both_methods":req["active"]==32 and base["active"]==32,
  "unrelated_scope_invariance_128":isolation["decisions"]==128 and isolation["unexpected_changes"]==0,
  "candidate_order_equivariance_at_least_95pct":perms["correct"]>=.95*perms["total"],
 }
 verdict="PROVISIONAL_SYNTHETIC_COMPASS_SLOT_MEMORY_PASS" if all(gates.values()) else "NO_GO_COMPASS_REAL_WORLD"
 weights={}
 for prefix,model in (("personal",user),("project",project)):
  weights.update({prefix+"."+k:v.detach().cpu().contiguous()
                  for k,v in model.state_dict().items()})
 save_file(weights,str(Path(args.output).with_suffix(".safetensors")))
 report={
  "version":"9.11","component":"COMPASS_NOT_SELF","seed":args.seed,
  "verdict":verdict,"fixture_audit":locked_audit,
  "comparisons":outputs,"scorecard":keyscore,"registered_gates":gates,
  "bad_source_asked":asks,"bad_slot_asked":invalid,"slot_reordering":order,
  "unrelated_memory_isolation":isolation,"option_order_equivariance":perms,
  "selected_slot_provenance_sample":audit_trace,
  "memory_training":mi,"personal_head_training":ui,
  "project_head_training":pi,
  "seconds":round(time.monotonic()-start,2),
  "limitations":[
   "All tasks, preferences and policy labels researcher-authored synthetic; no independent human tasks.",
   "The source and version metadata is provided in reliable structured form by the fixture.",
   "Mock public HMAC signing does not authenticate identity or consent.",
   "A valid typed policy contract supplies a correct action operation ID outside the neural model.",
   "English-only sentence encoder and training; Hebrew generalization likely weak.",
   "Neither adapter nor 360M SELF weights were touched. This is COMPASS schema+readout research."
  ]
 }
 Path(args.output).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
 print("V911 FINAL",json.dumps({"seed":args.seed,"verdict":verdict,
   "scores":keyscore,"bad_source":asks,"bad_slot":invalid,
   "slot_reordering":order,"scope_isolation":isolation,
   "option_equivariance":perms,"gates":gates,
   "seconds":report["seconds"]}),flush=True)

def first_three(case,side):
 c=copy.deepcopy(case)
 for key in ("user","a","b"):
  c[side]["histories"][key]=c[side]["histories"][key][:3]
 return c
if __name__=="__main__":main()
