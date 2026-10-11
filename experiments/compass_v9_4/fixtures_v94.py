"""V9.4 preregistered synthetic intent-to-new-decision task, English only.

GOLD and metadata are never included in the neural or frozen-LM task text.
A paired intervention replaces just ONE history (personal or one project's)
while preserving the same new problem, choices, and unrelated histories.
"""
import copy,hashlib,random
from study_v92 import H

FAMILY={
"user":{
 "train":[
  ("debugging","You need to choose an approach to an unfamiliar engineering failure. Both options are safe and available.",
   "Examine causes, uncertainties and risks before committing.","Try a limited reversible fix and adjust after observing results."),
  ("investigation","You are choosing how to approach a fresh technical question. Neither approach is required by a rule.",
   "Compare competing explanations and inspect limitations first.","Start a small trial, observe what happens, and iterate."),
  ("documentation","You are planning your own next research task. Both approaches are permitted.",
   "Evaluate the evidence and tradeoffs before settling on a direction.","Create a tiny workable example now and refine it after feedback."),
  ("coordination","You personally need to tackle an unfamiliar work challenge. No binding requirement decides the method.",
   "Investigate assumptions carefully before choosing a route.","Take an inexpensive experimental first step and learn as you go.")],
 "cold":[
  ("capacity","Your own response to an unexpected capacity problem must be chosen. Both paths are authorized and reversible.",
   "Study possible bottlenecks and test each assumption before proceeding.","Run a constrained capacity trial promptly and improve according to measurements."),
  ("workflow","You are deciding on your personal approach to an ambiguous workflow issue. Either approach can succeed.",
   "Review potential consequences and unresolved details prior to a decision.","Build a small demonstration, collect observations and adjust course."),
  ("research","You are selecting a personal method for exploring a new research question. No outside directive decides it.",
   "Scrutinize the uncertainties and weigh alternatives before acting.","Try a low-cost preliminary experiment and adapt as new data arrives."),
  ("incident","You personally need to choose your first step in a recoverable incident. Both choices are sufficient.",
   "Investigate the underlying conditions and review risks before executing.","Deploy a tiny reversible probe to discover what is happening.")],
 "binding_train":"A documented safety obligation requires validating assumptions and risks first, regardless of working style.",
 "binding_cold":"An independently approved operational safeguard mandates careful verification before a trial, regardless of preference.",
 "bind_semantic":0,
},
"project":{
 "train":[
  ("migration","A replacement interface is being designed. Both strategies are permissible unless a documented obligation says otherwise.",
   "Rebuild the workflow around its intended benefits, even if past screens change.",
   "Keep the established public interface behavior compatible for current clients."),
  ("reporting","The reporting service must be redesigned. Neither approach is intrinsically incorrect or unauthorized.",
   "Optimize reporting around the original information goal, not the inherited implementation.",
   "Retain the published integration protocol for existing reporting consumers."),
  ("search","A legacy lookup system is being modernized. Either design is approved in this situation.",
   "Achieve the real user outcome with a better lookup path despite older mechanics.",
   "Preserve the external lookup contract relied upon by downstream software."),
  ("automation","A workflow automation is being rebuilt. Both designs satisfy current safety and feasibility criteria.",
   "Replace accidental old steps when doing so better realizes the original purpose.",
   "Maintain backward-compatible externally visible automation behavior.")],
 "cold":[
  ("ledger","A financial ledger component will be reworked. Neither route is barred unless an explicit requirement applies.",
   "Choose a clearer ledger design that fulfills the underlying recordkeeping objective.",
   "Protect the documented ledger API contract used by established integrations."),
  ("dispatch","A dispatch system needs a new implementation. Both strategies are presently authorized.",
   "Reorganize dispatch around the fundamental operational result, not old forms.",
   "Keep the published dispatch message protocol stable for existing consumers."),
  ("knowledge","A knowledge platform is being improved. Both implementations could satisfy present requirements.",
   "Restructure the knowledge experience around the original learning benefit.",
   "Retain externally observable integration behavior for connected knowledge clients."),
  ("scheduler","The scheduling platform is being replaced. Either approach is viable unless specifically constrained.",
   "Redesign scheduling around the actual coordination goal despite historical workflow.",
   "Preserve the existing public scheduling contract relied upon by external users.")],
 "binding_train":"A formally approved external integration requirement demands the published compatibility contract be maintained regardless of project ambition.",
 "binding_cold":"A verified contractual obligation mandates preserving the existing public protocol for downstream clients, overriding design preference.",
 "bind_semantic":1,
}}
# All-new far test history sentences; NO prior V9.2 or V9.3 FAR phrases.
NEW_HISTORY={
"user":{
 0:[
 "Before proceeding, map the uncertainties and confirm the possible failure modes.",
 "The earlier snap decision was risky; examine alternatives and evidence in depth.",
 "I prefer to scrutinize assumptions and risks ahead of taking action.",
 "Pause the implementation choice until its limitations have been investigated."],
 1:[
 "Build a minimal working slice first; learn by measuring its results.",
 "Stop extending the planning phase; launch a reversible pilot to discover the path.",
 "A modest hands-on attempt should come before broad speculation.",
 "Try the smallest feasible experiment, then revise what you build."]},
"project":{
 0:[
 "The project's reason to exist matters more than preserving accidental implementation details.",
 "Our destination is the intended benefit, not recreating legacy mechanics verbatim.",
 "Redesign obsolete workflows when a new mechanism better delivers the actual objective.",
 "Past screens are contingent choices; prioritize the outcome they were built to achieve."],
 1:[
 "The published outward-facing behavior is a promise that replacements must honor.",
 "Keep the existing contract that dependent services already rely on.",
 "Breaking the established consumer protocol is incompatible with this project's goal.",
 "Modernization must preserve stable integration behavior for external clients."]}}
PROJECTS_TRAIN=("TRAIN-P1","TRAIN-P2","TRAIN-P3","TRAIN-P4","TRAIN-P5")
PROJECTS_COLD=("COLD-PORT","COLD-RELAY","COLD-LATTICE","COLD-CANOPY")
def hid(s):return hashlib.sha256(("v94|"+s).encode()).hexdigest()[:18]
def histories(rng,split,states):
 result={}
 for k,mode in states.items():
  cat="user" if k=="user" else "project"
  train=H[cat]["train"]
  if split=="train":
   current=rng.sample(train[mode],3)
   if rng.random()<.5:current=rng.sample(train[1-mode],2)+[rng.choice(train[mode])]
  else:
   source=NEW_HISTORY[cat]
   current=[rng.choice(source[mode]),rng.choice(source[mode]),rng.choice(source[mode])]
   if rng.random()<.5:current=rng.sample(source[1-mode],2)+[rng.choice(source[mode])]
  result[k]=current
 return result
def choose_state(states,scope,project):
 return states["user"] if scope=="user" else states[project]
def pack(rng,split,index):
 family_keys=("user","project")
 cat=family_keys[index%2]
 fi=(index//2)%len(FAMILY[cat][split])
 name,question,action0,action1=FAMILY[cat][split][fi]
 projects=PROJECTS_TRAIN if split=="train" else PROJECTS_COLD
 project=projects[(index//32)%len(projects)]
 scope=(index//(2*len(FAMILY[cat][split])*len(projects)))%2
 # If "user", user history authoritative; if "project", target project.
 # Use words in QUESTION only, not a hidden scope selector in the trained bridge.
 if cat=="user":
  scope="user";which="a" if rng.random()<.5 else "b"
 else:
  scope="project";which="a" if rng.random()<.5 else "b"
 target="user" if scope=="user" else which
 states={k:rng.randrange(2) for k in ("user","a","b")}
 if scope=="project":states["b" if which=="a" else "a"]=1-states[which]
 start=histories(rng,split,states)
 # Force counterfactual TARGET history to show the opposite principle.
 alternative=copy.deepcopy(start)
 new_mode=1-states[target]
 alternative[target]=histories(rng,split,{target:new_mode})[target]
 binding=(index//8)%2==1
 # Ensure balanced binding cases across project and personal contexts.
 full_question=question
 if binding:full_question+=" "+FAMILY[cat]["binding_train" if split=="train" else "binding_cold"]
 swapped=(index//16)%2
 actions=[action0,action1]
 if swapped:actions.reverse()
 original_choice=FAMILY[cat]["bind_semantic"] if binding else states[target]
 other_choice=FAMILY[cat]["bind_semantic"] if binding else new_mode
 return {
  "id":"TASK-"+hid(f"{split}-{index}-{rng.randrange(1000000)}"),
  "scope":scope,"project":which,"family":name,"binding":binding,"swapped":swapped,
  "question":full_question,"options":actions,
  "original":{"histories":start,"gold":original_choice^swapped},
  "changed":{"histories":alternative,"gold":other_choice^swapped},
  "expected_flip":not binding
 }
def build(seed=11,train_n=384,test_n=128):
 rng=random.Random(7700+seed)
 training=[pack(rng,"train",i) for i in range(train_n)]
 rng=random.Random(8800+seed)
 cold=[pack(rng,"cold",i) for i in range(test_n)]
 audit(training,cold)
 return training,cold
def audit(training,cold):
 assert set(PROJECTS_TRAIN).isdisjoint(set(PROJECTS_COLD))
 assert set(NEW_HISTORY["user"][0]).isdisjoint(set(H["user"]["train"][0]))
 assert len(training)>=128 and len(cold)>=32
 assert not (set(FAMILY["user"]["train"][i][0] for i in range(4))&
             set(FAMILY["user"]["cold"][i][0] for i in range(4)))
 for examples in (training,cold):
  assert len({x["id"] for x in examples})==len(examples)
  assert set(x["scope"] for x in examples)=={"user","project"}
  assert set(x["binding"] for x in examples)=={False,True}
  assert set(x["swapped"] for x in examples)=={0,1}
  for p in examples:
   a=p["original"];b=p["changed"];target="user" if p["scope"]=="user" else p["project"]
   assert a["gold"] in (0,1) and b["gold"] in (0,1)
   assert a["histories"][target]!=b["histories"][target]
   assert (a["gold"]!=b["gold"])==p["expected_flip"]
   assert all(a["histories"][k]==b["histories"][k] for k in ("user","a","b") if k!=target)
   assert "gold" not in p["question"] and "correct_choice" not in str(p["question"])
 return {"status":"PASS","train_pairs":len(training),"cold_pairs":len(cold),
  "cold_families":sorted({p["family"] for p in cold}),"train_families":sorted({p["family"] for p in training}),
  "cold_binding":sum(x["binding"] for x in cold),"cold_neutral":sum(not x["binding"] for x in cold),
  "synthetic_author_gold":True}
if __name__=="__main__":
 import json;print(json.dumps(audit(*build()),indent=2))
