"""Predeclared V9.7 unseen synthetic tasks with mock policy evidence.

The claim generator has artificial authority keys and a revocation ledger;
runtime receives them as explicit external evidence, NOT as hidden gold labels.
"""
import copy
import hmac
import json
import hashlib
import random
from fixtures_v94 import FAMILY, pack

TASK_FAMILIES={
 "user":[
  ("recovery","An intermittent cache anomaly calls for your own next move. Both options are approved.",
   "Review evidence and uncertainty before making a consequential change.",
   "Try a tiny reversible cache adjustment, measure and iterate."),
  ("diagnostics","You are personally approaching an unpredictable environment fault. Either response is allowed.",
   "Inspect risk boundaries and assumptions in depth before intervening.",
   "Launch a limited and reversible probe to learn from live observations."),
  ("reliability","A low-impact reliability issue needs your personal first step. Neither option is required.",
   "Investigate the possible failure paths and evaluate tradeoffs before acting.",
   "Deploy a safe minimal experiment and refine after checking results."),
  ("tooling","An unfamiliar toolchain issue requires a personal strategy choice. Both are permitted.",
   "Study possible consequences and competing explanations before a decision.",
   "Implement a small recoverable trial and adjust based on feedback."),
 ],
 "project":[
  ("warehouse","A warehouse information layer is being redesigned. Either action is allowed absent a valid duty.",
   "Rebuild warehouse flows around the real goal of accurate stock visibility.",
   "Keep the existing warehouse integration endpoint contract compatible."),
  ("telemetry","A telemetry system is being replaced. Neither approach is inherently required.",
   "Redesign telemetry to satisfy its original observability purpose.",
   "Preserve the documented telemetry message contract for current clients."),
  ("publishing","A publishing platform has to be modernized. Both choices are operationally feasible.",
   "Reshape publishing around the intended reader outcome rather than old pages.",
   "Maintain the outward-facing publishing API behavior for dependent consumers."),
  ("queueing","A task queue platform needs refactoring. Both options would function.",
   "Reorganize queue logic around its actual reliable-delivery objective.",
   "Retain backward-compatible externally visible queue protocol semantics."),
 ]}
OPS={"user":("risk-review","reversible-probe"),
     "project":("mission-redesign","public-contract-compatibility")}
SCENARIOS=("no_claim","active","revoked","expired","other_project",
           "unverified","forged","conflict")
DATE="2026-10-11"
TRUSTED_ISSUERS={
 "safety-office":{"kind":"user","secret":b"test-only-safety-key-not-production"},
 "integration-office":{"kind":"project","secret":b"test-only-integrations-key-not-production"}
}
def canon(claim):
 return json.dumps({k:v for k,v in claim.items() if k!="signature"},
                   sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def signature(claim,secret):
 return hmac.new(secret,canon(claim),hashlib.sha256).hexdigest()
def claim_for(case,tag,required_op,kind=None):
 kind=kind or case["scope"]
 issuer="safety-office" if kind=="user" else "integration-office"
 obj={"id":"POLICY-"+case["id"]+"-"+tag,
      "issuer":issuer,"kind":kind,
      "scope_id":"*" if kind=="user" else case["project"],
      "required_operation":required_op,
      "effective_from":"2026-01-01",
      "expires_at":"2027-01-01",
      "description":"Formal operational requirement on a typed action, not a textual preference."}
 obj["signature"]=signature(obj,TRUSTED_ISSUERS[issuer]["secret"])
 return obj
def new_cases(seed=11):
 prior={k:FAMILY[k]["cold"] for k in ("user","project")}
 try:
  for k in prior:FAMILY[k]["cold"]=TASK_FAMILIES[k]
  rng=random.Random(9700+seed)
  items=[]
  for index in range(512):
   if (index//8)%2==1:continue   # V9.4 "binding" fixture stays FALSE by construction
   p=pack(rng,"cold",index)
   assert not p["binding"],"do not secretly inject old fixture binding gold"
   items.append(p)
 finally:
  for k in prior:FAMILY[k]["cold"]=prior[k]
 assert len(items)==256
 revoked=set()
 for j,case in enumerate(items):
  scenario=SCENARIOS[j//32]
  kind=case["scope"]
  semantic_to_index=lambda k:k^case["swapped"]
  case["operation_ids"]=[OPS[kind][0],OPS[kind][1]]
  if case["swapped"]:case["operation_ids"].reverse()
  case["active_workspace"]=case["project"]
  case["claims"]=[]
  case["scenario"]=scenario
  case["expected_resolution"]="PREFERENCE"
  if scenario=="active":
   semantic=(j//2)%2
   op=OPS[kind][semantic]
   case["claims"]=[claim_for(case,"ACTIVE",op)]
   case["expected_resolution"]="ENFORCE"
   for side in ("original","changed"):case[side]["gold"]=semantic_to_index(semantic)
   case["expected_flip"]=False
  elif scenario=="revoked":
   c=claim_for(case,"WITHDRAWN",OPS[kind][(j//2)%2])
   case["claims"]=[c]
   revoked.add(c["id"])
  elif scenario=="expired":
   c=claim_for(case,"EXPIRED",OPS[kind][(j//2)%2])
   c["expires_at"]="2026-08-01"
   c["signature"]=signature(c,TRUSTED_ISSUERS[c["issuer"]]["secret"])
   case["claims"]=[c]
  elif scenario=="other_project":
   if kind=="project":
    c=claim_for(case,"OTHER",OPS[kind][(j//2)%2])
    c["scope_id"]="OTHER-"+case["project"]
   else:
    # Global source type is not authority for a project-only constraint.
    c=claim_for(case,"OTHER",OPS["project"][(j//2)%2],"project")
    c["scope_id"]="OTHER-"+case["project"]
   c["signature"]=signature(c,TRUSTED_ISSUERS[c["issuer"]]["secret"])
   case["claims"]=[c]
  elif scenario=="unverified":
   c=claim_for(case,"UNVERIFIED",OPS[kind][(j//2)%2])
   c["issuer"]="unregistered-committee"
   c["signature"]="unverifiable-claim"
   case["claims"]=[c]
   case["expected_resolution"]="ABSTAIN"
   for side in ("original","changed"):case[side]["gold"]=2
   case["expected_flip"]=False
  elif scenario=="forged":
   c=claim_for(case,"FORGED",OPS[kind][(j//2)%2])
   c["signature"]="0"*64
   case["claims"]=[c]
   case["expected_resolution"]="ABSTAIN"
   for side in ("original","changed"):case[side]["gold"]=2
   case["expected_flip"]=False
  elif scenario=="conflict":
   case["claims"]=[claim_for(case,"C1",OPS[kind][0]),
                    claim_for(case,"C2",OPS[kind][1])]
   case["expected_resolution"]="ABSTAIN"
   for side in ("original","changed"):case[side]["gold"]=2
   case["expected_flip"]=False
  assert len(set(case["operation_ids"]))==2
  assert set(case["operation_ids"])==set(OPS[kind])
 assert len({x["id"] for x in items})==len(items)
 return items,revoked

def audit(items,revoked):
 assert len(items)==256
 for scenario in SCENARIOS:
  subset=[c for c in items if c["scenario"]==scenario]
  assert len(subset)==32,(scenario,len(subset))
  assert sum(c["scope"]=="user" for c in subset)==16
  assert sum(c["swapped"] for c in subset)==16
  for c in subset:
   assert c["original"]["histories"]["user"]==c["changed"]["histories"]["user"] or c["scope"]=="user"
   assert (c["original"]["gold"]!=c["changed"]["gold"])==c["expected_flip"]
   assert all(c["original"]["histories"][k]==c["changed"]["histories"][k]
         for k in ("user","a","b") if k!=("user" if c["scope"]=="user" else c["project"]))
  # All task texts and action descriptions fixed across original and changed histories.
 assert sum(x["scenario"]=="revoked" for x in items)==len(revoked)
 assert len({c["family"] for c in items})==8
 for k in TASK_FAMILIES:
  assert {x[0] for x in TASK_FAMILIES[k]}.isdisjoint(
    {x[0] for x in FAMILY[k]["train"]+FAMILY[k]["cold"]})
 return {"cases":len(items),"paired_decisions":len(items)*2,
         "scenarios":{k:sum(c["scenario"]==k for c in items) for k in SCENARIOS},
         "revocation_entries":len(revoked),"independent_human_labels":False,
         "source_authentication_real_world":False}
if __name__=="__main__":
 print(json.dumps(audit(*new_cases()),indent=2))
