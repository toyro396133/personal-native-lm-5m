"""V9.9 fresh synthetic intent-transfer cases and mocked request-origin evidence.

Origin events explicitly indicate the request owner's own chosen UI context.
This is SIMULATED user-supplied metadata strongly correlated to research gold;
its usefulness does not measure autonomous understanding of the user's intent.
"""
import copy
import hashlib
import hmac
import json
import random

from fixtures_v94 import FAMILY,NEW_HISTORY,pack
from fixtures_v98 import TASKS as PREVIOUS_TASKS,ALL_OLD_TASKS

TASKS={
 "user":[
  ("sandbox-fault","You personally need a first approach to a harmless sandbox fault; both paths are allowed.",
   "Enumerate possible failure chains and verify risky assumptions before change.",
   "Run a minimal recoverable sandbox probe and modify it from what you observe."),
  ("quota-pressure","Your own response to temporary quota pressure is under consideration; neither method is mandatory.",
   "Evaluate the uncertainty, tradeoffs and consequences first.",
   "Launch a small safe trial under the quota and iterate according to results."),
  ("restore-point","You must select your personal approach to a recoverable restore-point problem. Either is permitted.",
   "Inspect potential causes and backup risks before executing a plan.",
   "Try a tiny rollback rehearsal immediately, then refine based on feedback."),
  ("memory-cache","An unfamiliar memory-cache hiccup requires your personal technique. Both techniques are allowed.",
   "Analyze the suspected causes and possible failure modes before changing a cache.",
   "Implement a limited reversible cache experiment and learn from data."),
  ("terminal-error","You are choosing your own process for an unexpected terminal error; neither option is required.",
   "Carefully investigate unknown assumptions and risks before making a fix.",
   "Attempt a small disposable reproduction and adapt from its behavior."),
  ("dependency-drift","Your personal response to a fresh dependency mismatch is needed. Both responses are approved.",
   "Examine the version dependencies and possible consequences thoroughly before acting.",
   "Try a constrained reversible dependency experiment and iterate."),
  ("deploy-check","For a modest deployment uncertainty you may choose your personal next step; both strategies are viable.",
   "Study alternatives, hidden risks and assumptions ahead of intervention.",
   "Run a minor reversible deployment experiment and improve after measurements."),
  ("metrics-gap","A benign gap in telemetry needs your own first approach; neither approach is obligatory.",
   "Review potential blind spots and risk exposure before choosing a change.",
   "Introduce a tiny safe diagnostic probe and refine based on the resulting readings."),
 ],
 "project":[
  ("contact-hub","A contact hub is being rewritten, with no binding design constraint on either choice.",
   "Redesign the hub around its enduring information-sharing objective.",
   "Preserve the existing externally documented contact synchronization API."),
  ("delivery-planner","A delivery planner is being rebuilt. Both proposed strategies are feasible.",
   "Focus the new planner on the real objective of dependable delivery.",
   "Keep the public delivery message protocol backwards compatible."),
  ("case-notes","A case notes service needs modernization. Both directions are allowable.",
   "Rebuild notes around the underlying knowledge continuity goal.",
   "Maintain the existing externally published case export contract."),
  ("ticket-router","A support ticket router needs a new implementation. Neither candidate is mandatory.",
   "Change the workflow around the essential routing outcome rather than old interfaces.",
   "Continue honoring the established outside ticket-routing API."),
  ("schedule-export","A schedule export service requires redesign. Either choice is approved.",
   "Reorganize the export experience to achieve the true coordination benefit.",
   "Preserve the documented public calendar export format for clients."),
  ("document-feed","A document feed must evolve. Both alternatives are authorized by default.",
   "Improve the feed around its fundamental retrieval goal rather than old pages.",
   "Keep the documented external document feed endpoint behavior unchanged."),
  ("event-collector","An event collection service needs refactoring; either route would work.",
   "Rebuild collection according to the intended information quality objective.",
   "Retain the published event ingestion contract used by dependent tools."),
  ("client-sync","A client synchronization layer is being modernized. Both plans are practical.",
   "Redesign around the real benefit of accurate shared client state.",
   "Preserve externally visible client sync protocol compatibility."),
 ]}
COLD_HISTORY={
 "user":{
 0:["I first need a complete picture of uncertainty and possible failure costs.",
    "When choosing, I prefer careful risk examination and evidence review.",
    "Before implementation, question the assumptions and evaluate alternatives.",
    "I avoid a hasty fix when the risks and unknowns have not been assessed."],
 1:["Run a tiny reversible prototype rather than prolonging speculative planning.",
    "I favor a small workable experiment and improvement from actual observations.",
    "A cheap safe first trial is my preferred method of learning.",
    "Start with a recoverable attempt and adjust once measurements are available."]},
 "project":{
 0:["The reason our platform exists should determine the new approach, not inherited layouts.",
    "Our real outcome matters more than rebuilding incidental legacy workflows.",
    "Focus the rewrite on its original goal even if earlier machinery is replaced.",
    "Legacy screens are only past choices; the enduring project benefit comes first."],
 1:["Existing external consumers rely on our published connection contract.",
    "Keeping stable public interfaces is a central principle of this particular project.",
    "We must preserve the outward-facing protocol our partners already depend on.",
    "Modernization should honor our project's established integration guarantees."]}}
MOCK_SIGNERS={"profile-ui":b"v99-public-demo-personal-only",
              "workspace-ui":b"v99-public-demo-project-only"}

def canonical(obj):
 return json.dumps({k:v for k,v in obj.items() if k!="mac"},
                   sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sign(record):
 issuer=record["issuer"]
 return hmac.new(MOCK_SIGNERS[issuer],canonical(record),hashlib.sha256).hexdigest()
def origin(case,kind=None):
 typ=kind or ("personal" if case["scope"]=="user" else "project")
 issuer="profile-ui" if typ=="personal" else "workspace-ui"
 record={"issuer":issuer,"event_id":"UI-"+case["id"]+"-"+typ,
         "event_type":"PERSONAL_PROFILE_DECISION" if typ=="personal" else "WORKSPACE_GOAL_DECISION",
         "workspace":"*" if typ=="personal" else case["project"],
         "intent_type":typ}
 record["mac"]=sign(record)
 return record

def fresh(seed=11):
 rng=random.Random(99000+seed)
 old_family={k:FAMILY[k]["cold"] for k in ("user","project")}
 old_hist={k:{m:NEW_HISTORY[k][m] for m in (0,1)} for k in ("user","project")}
 try:
  for kind in ("user","project"):
   FAMILY[kind]["cold"]=TASKS[kind]
   for mode in (0,1):NEW_HISTORY[kind][mode]=COLD_HISTORY[kind][mode]
  cases=[]
  for i in range(256):
   if (i//8)%2:continue
   c=pack(rng,"cold",i)
   assert not c["binding"]
   kind=c["scope"]
   fi=((i//16)+(i//2)%4)%8
   family,q,option0,option1=TASKS[kind][fi]
   c["family"]=family
   c["question"]=q
   c["options"]=[option0,option1]
   if c["swapped"]:c["options"].reverse()
   c["source_events"]=[origin(c)]
   c["active_workspace"]=c["project"]
   cases.append(c)
 finally:
  for k in ("user","project"):
   FAMILY[k]["cold"]=old_family[k]
   for m in (0,1):NEW_HISTORY[k][m]=old_hist[k][m]
 audit(cases)
 return cases

STRESS_TYPES=("missing_origin","bad_signature","conflicting_sources","wrong_workspace")
def uncertainty(cases):
 """96 reshaped cases, 24 each scenario, exact same new cases as main."""
 stress=[]
 for i,typ in enumerate(STRESS_TYPES):
  chunk=cases[i*24:(i+1)*24]
  assert len(chunk)==24
  for c in chunk:
   row=copy.deepcopy(c)
   row["id"]="UNCERTAIN-"+typ+"-"+c["id"]
   row["uncertainty_type"]=typ
   if typ=="missing_origin":
    row["source_events"]=[]
   elif typ=="bad_signature":
    row["source_events"][0]["mac"]="0"*64
   elif typ=="conflicting_sources":
    row["source_events"].append(origin(c,"project" if c["scope"]=="user" else "personal"))
   elif typ=="wrong_workspace":
    # Both event type and signer still valid, but workspace context disagrees.
    row["source_events"][0]["workspace"]="OTHER-UNRELATED-WORKSPACE"
    row["source_events"][0]["mac"]=sign(row["source_events"][0])
   for side in ("original","changed"):row[side]["gold"]=2
   row["expected_flip"]=False
   stress.append(row)
 return stress

def perturb(case,typ):
 c=copy.deepcopy(case)
 if typ=="personal_other_project_noise":
  assert c["scope"]=="user"
  for side in ("original","changed"):
   c[side]["histories"]["a"]=COLD_HISTORY["project"][0][:3]
   c[side]["histories"]["b"]=COLD_HISTORY["project"][1][:3]
 elif typ=="personal_workspace_switch":
  assert c["scope"]=="user"
  c["project"]="b" if c["project"]=="a" else "a"
  c["active_workspace"]=c["project"]
 elif typ=="project_irrelevant_user_noise":
  assert c["scope"]=="project"
  for side in ("original","changed"):
   c[side]["histories"]["user"]=COLD_HISTORY["user"][1][:3]
 else:raise ValueError(typ)
 c["id"]=typ.upper()+"-"+c["id"]
 return c

def audit(cases):
 assert len(cases)==128
 assert len({c["id"] for c in cases})==128
 assert sum(c["scope"]=="user" for c in cases)==64
 assert sum(c["scope"]=="project" for c in cases)==64
 assert sum(c["swapped"] for c in cases)==64
 for kind in ("user","project"):
  subset=[c for c in cases if c["scope"]==kind]
  names={c["family"] for c in subset}
  assert len(names)==8,(kind,names)
  assert names.isdisjoint({x[0] for x in PREVIOUS_TASKS[kind]}|
                         {x[0] for x in FAMILY[kind]["train"]+FAMILY[kind]["cold"]}|
                         ALL_OLD_TASKS[kind])
  for c in subset:
   assert c["original"]["gold"]!=c["changed"]["gold"]
   target="user" if kind=="user" else c["project"]
   assert all(c["original"]["histories"][key]==c["changed"]["histories"][key]
             for key in ("user","a","b") if key!=target)
   assert c["source_events"][0]["mac"]==sign(c["source_events"][0])
 assert len(uncertainty(cases))==96
 return {"fresh_pairs":128,"personal":64,"project":64,
         "families":16,"uncertain_pairs":96,
         "stress_types":{x:24 for x in STRESS_TYPES},
         "real_world_verified_identity":False,
         "task_origin_matches_author_scope_by_construction":True,
         "human_independent_gold":False}
if __name__=="__main__":
 print(json.dumps(audit(fresh()),indent=2))
