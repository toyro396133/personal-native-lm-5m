"""V9.8 synthetic personal+project intent family transfer, new statements and tasks.

This cold generator is researcher-authored; no independent human labels.
Same narrow binary intents as V9.4, different vocabulary and 16 task families.
"""
import copy,random
from fixtures_v94 import FAMILY, NEW_HISTORY, pack

TASKS={
 "user":[
 ("rollback-plan","You are deciding your own response to a surprising service rollback. Either first step is acceptable.",
  "Map hidden dependencies and failure risks before proceeding.",
  "Attempt a minimal reversible rollback pilot and improve from observation."),
 ("outage-review","An unusual outage is recoverable and neither personal approach is required.",
  "Review evidence and alternative explanations prior to making changes.",
  "Test an inexpensive reversible recovery action and iterate."),
 ("latency-hypothesis","Your own approach to unexplained latency is open; both methods are permitted.",
  "Check assumptions, compare hypotheses and assess possible consequences first.",
  "Launch a small latency experiment now and refine it from measurements."),
 ("capacity-alert","You personally choose the next step for a temporary resource saturation. No rule decides it.",
  "Investigate likely bottlenecks and their risks before modifying resources.",
  "Trial a constrained reversible capacity adjustment and learn from it."),
 ("anomaly-detection","You must decide your own method for a benign anomaly. Both choices are reasonable.",
  "Analyze uncertain explanations and their side effects in detail.",
  "Try a tiny live anomaly probe and adapt according to the outcome."),
 ("handover-review","A confusing team handover needs your preferred approach. Both options are authorized.",
  "Examine unknowns, possible handover risks and alternative plans.",
  "Run a minimal recoverable handover experiment and revise afterward."),
 ("build-failure","You are selecting how personally to respond to a novel build failure. No mandate applies.",
  "Study failure modes and supporting evidence before attempting a fix.",
  "Try a small reversible build change and adjust using feedback."),
 ("cache-repair","Your own strategy is needed for harmless cache inconsistencies. Either action is allowed.",
  "Evaluate hidden causes and risks before making the first intervention.",
  "Perform a contained cache experiment and iterate once results are visible."),
 ],
 "project":[
 ("archive-search","An archive search engine requires a redesign. Either route is possible when unconstrained.",
  "Change architecture to achieve the real long-term discovery outcome.",
  "Retain the published archive-search API interface for dependent clients."),
 ("log-routing","A log delivery platform needs replacing. Both proposals are viable by default.",
  "Reimagine log routing for its actual reliability objective.",
  "Keep the externally published log event contract backwards compatible."),
 ("inventory-sync","An inventory synchronization service is rebuilt. No explicit rule mandates either choice.",
  "Redesign syncing around the true objective of consistent inventories.",
  "Preserve the documented inventory integration handshake protocol."),
 ("message-index","A messaging index is modernized. The two choices are available.",
  "Reconstruct indexing around the intended user search benefit.",
  "Maintain the outward-facing messaging integration API contract."),
 ("alert-service","A system of alerts must be rethought. Either approach can work in this task.",
  "Focus redesign around accurate actionable alerts instead of inherited screens.",
  "Continue supporting the established external alert webhook behavior."),
 ("queue-rebuild","An internal queue is redesigned. Both alternatives have been authorized.",
  "Build a queue based on the original delivery assurance goal, replacing incidental steps.",
  "Honor published queue compatibility guarantees for existing clients."),
 ("reporting-engine","An analytics reporting engine needs a fresh implementation. Both choices are feasible.",
  "Optimize the reporting process for the underlying information purpose.",
  "Maintain the existing externally visible reporting query protocol."),
 ("catalog-rewrite","The catalog service is being rewritten. Either design would be technically practical.",
  "Reorganize catalog features around the intended information value.",
  "Keep the public catalog API consistent for dependent consumers."),
 ],
}
NEW_STATEMENTS={
 "user":{
  0:[
   "Before deciding, I want the uncertainty map and its dangerous edge cases laid out.",
   "Investigate plausible failure chains carefully rather than charging ahead.",
   "I favor a rigorous comparison of risks and explanations prior to action.",
   "First identify what is poorly understood and test the assumptions on paper."],
  1:[
   "A tiny executable trial teaches me more than another planning document.",
   "Prefer an affordable reversible attempt and course corrections from data.",
   "Get a small experiment running first, then improve based on its findings.",
   "I would rather probe a limited solution quickly and revise it afterward."]},
 "project":{
  0:[
   "Our project exists for a concrete benefit; legacy mechanics are negotiable.",
   "The original purpose should guide changes rather than accidental screen designs.",
   "Modernize around the goal even when the previous workflow disappears.",
   "This project's outcome matters more than faithfully copying old plumbing."],
  1:[
   "We promised dependable outside clients an enduring interface contract.",
   "The documented external integration must stay stable throughout a rewrite.",
   "Dependent software should keep receiving the same published protocol.",
   "Avoid breaking the public-facing API that established consumers depend on."]}}
ALL_OLD_TASKS={
 "user":{"capacity","workflow","research","incident","recovery","diagnostics","reliability","tooling","backpressure","observability","handoff","validation",
         "rollback","profiling","triage","quality"},
 "project":{"ledger","dispatch","knowledge","scheduler","warehouse","telemetry","publishing","queueing","catalog","streaming","archives","routing",
            "inventory","messaging","enrollment","indexing"},
}

def build_cold(seed=11):
 rng=random.Random(98800+seed)
 old_f={k:FAMILY[k]["cold"] for k in ("user","project")}
 old_stat={k:{m:NEW_HISTORY[k][m] for m in (0,1)} for k in ("user","project")}
 try:
  for k in ("user","project"):
   FAMILY[k]["cold"]=TASKS[k]
   for m in (0,1):NEW_HISTORY[k][m]=NEW_STATEMENTS[k][m]
  cases=[]
  for i in range(256):
   if (i//8)%2:continue  # neutral-only; never train policy words as gold
   c=pack(rng,"cold",i)
   assert not c["binding"]
   scope=c["scope"]
   # Distribute all 8 new heldout families despite pack()'s 8-on/8-off indices.
   family_index=((i//8)+i%4)%8
   family,q,a0,a1=TASKS[scope][family_index]
   c["family"]=family
   c["question"]=q
   c["options"]=[a0,a1]
   if c["swapped"]:c["options"].reverse()
   c["new_workspace_id"]="V98-UNSEEN-"+str((i//32)%4)
   cases.append(c)
 finally:
  for k in ("user","project"):
   FAMILY[k]["cold"]=old_f[k]
   for m in (0,1):NEW_HISTORY[k][m]=old_stat[k][m]
 audit(cases)
 return cases

def perturb(case,variant):
 c=copy.deepcopy(case)
 if variant=="personal_project_noise":
  assert case["scope"]=="user"
  # Each project history deliberately pulls toward an opposite project intent.
  c["original"]["histories"]["a"]=NEW_STATEMENTS["project"][0][:3]
  c["original"]["histories"]["b"]=NEW_STATEMENTS["project"][1][:3]
  c["changed"]["histories"]["a"]=NEW_STATEMENTS["project"][1][:3]
  c["changed"]["histories"]["b"]=NEW_STATEMENTS["project"][0][:3]
 elif variant=="personal_workspace_switch":
  assert case["scope"]=="user"
  c["project"]="b" if c["project"]=="a" else "a"
 elif variant=="project_user_noise":
  assert case["scope"]=="project"
  c["original"]["histories"]["user"]=NEW_STATEMENTS["user"][0][:3]
  c["changed"]["histories"]["user"]=NEW_STATEMENTS["user"][1][:3]
 else:raise ValueError(variant)
 c["id"]=variant.upper()+"-"+case["id"]
 return c

def audit(cases):
 assert len(cases)==128
 assert sum(c["scope"]=="user" for c in cases)==64
 assert sum(c["scope"]=="project" for c in cases)==64
 assert sum(c["swapped"] for c in cases)==64
 for scope in ("user","project"):
  names={c["family"] for c in cases if c["scope"]==scope}
  assert len(names)==8,(scope,names)
  assert names.isdisjoint({x[0] for x in FAMILY[scope]["train"]+FAMILY[scope]["cold"]}|ALL_OLD_TASKS[scope])
  assert all(c["binding"] is False for c in cases)
 for c in cases:
  assert c["original"]["gold"]!=c["changed"]["gold"]
  assert c["options"] and len(set(c["options"]))==2
  target="user" if c["scope"]=="user" else c["project"]
  assert c["original"]["histories"][target]!=c["changed"]["histories"][target]
  assert all(c["original"]["histories"][k]==c["changed"]["histories"][k]
         for k in ("user","a","b") if k!=target)
 return {"new_pairs":len(cases),"new_user":64,"new_project":64,
         "new_task_families":16,"new_history_phrases":16,
         "independently_human_authored":False}
if __name__=="__main__":
 import json
 print(json.dumps(audit(build_cold()),indent=2))
