"""COMPASS V9.2: synthetic sequential concept learning; no LM integration.

One frozen *future* LM will receive these user/project latents only AFTER
concept generalization passes independent tests.  The baseline here is an
explicit full-history lexical reader, NOT an LLM with excellent retrieval.
All histories/labels researcher-authored English. No oracle label in model input.
"""
import argparse,hashlib,json,random,re,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file

# Index 0 / 1 are contrasting USER and PROJECT principles.
H={
 "user":{
  "train":[
   ["I corrected the plan: inspect assumptions and verify risks before committing.",
    "A hurried decision missed a constraint, so evaluate the evidence first.",
    "Review alternatives carefully before selecting our approach.",
    "Investigate unknowns and explain tradeoffs before work begins."],
   ["I corrected the workflow: launch a small prototype and learn from feedback.",
    "Rather than prolonged planning, try a minimal experiment and iterate quickly.",
    "Move to a narrow test early; improve it after observing the outcome.",
    "Make a lightweight pilot before writing a detailed plan."]],
  "cold":[
   ["My priority is to examine uncertainty and validate reasoning prior to commitment.",
    "The earlier shortcut was premature; we ought to inspect potential consequences.",
    "Do not settle on an option until its limitations have been explored."],
   ["Instead of debating every possibility, start with an inexpensive trial and revise it.",
    "The previous long review stalled us; I now want a small-scale attempt first.",
    "Test something modest, gather observations, then adjust direction."]]},
 "project":{
  "train":[
   ["The project aim is to preserve the intended outcome, not the old implementation.",
    "Historic screens are examples, not the purpose; achieve the real objective.",
    "A replacement may change workflow if it better serves the original goal.",
    "Do not copy accidental legacy constraints; protect why the feature exists."],
   ["The project aim is to preserve its published interface and exact compatibility.",
    "Existing integration contracts must remain stable even when internals change.",
    "A replacement must continue established externally visible behavior.",
    "Do not break legacy interface obligations during redesign."]],
  "cold":[
   ["What matters is the intended benefit; yesterday's implementation details are replaceable.",
    "The older workflow was only a means. Preserve the end result through better design.",
    "The project's deeper objective has priority over recreating old steps."],
   ["External users rely on our previous behavioral contract, which must not be broken.",
    "Even with improved implementation, keep the integration protocol backward-compatible.",
    "Success requires public behavior to remain indistinguishable to existing clients."]]}
}
# Novel application dilemma prose, not included in model input for this
# FIRST concept-representation probe. Downstream semantic decision is still
# routed by explicit trusted task type, NOT solved by free-form LM reasoning.
DILEMMAS={
 "user":[
 ("An uncertain design decision needs a working method.","verify uncertainties","prototype quickly"),
 ("The team needs an approach to a new task.","review consequences","test and adjust"),
 ("A difficult engineering task lacks a clear path.","inspect assumptions","try an initial pilot")],
 "project":[
 ("A better new tool might differ from historical screens.","serve intended outcome","retain public contract"),
 ("An upgrade modifies some earlier interface assumptions.","protect core purpose","keep compatibility"),
 ("A replacement changes the old workflow and internals.","follow the deeper goal","preserve external behavior")]
}
def make_episode(rng,split):
 u=rng.randrange(2);a=rng.randrange(2);b=1-a if rng.random()<.8 else rng.randrange(2)
 modes={"user":u,"a":a,"b":b}
 corrected={k:rng.random()<.45 for k in modes}
 histories={}
 for key,mode in modes.items():
  cat="user" if key=="user" else "project"
  train=H[cat]["train"];cold=H[cat]["cold"]
  if corrected[key]:
   histories[key]=rng.sample(train[1-mode],2)+[rng.choice(train[mode] if split in ("train","dev") else cold[mode])]
  elif split in ("train","dev"):
   histories[key]=rng.sample(train[mode],3)
  else:
   histories[key]=[rng.choice(train[mode]),rng.choice(cold[mode]),rng.choice(cold[mode])]
 return {"events":histories,"truth":modes,"corrected":corrected}
def audit():
 assert set(H)=={"user","project"}
 assert len(DILEMMAS["user"])==len(DILEMMAS["project"])==3
 for c in H:
  for a,b in zip(H[c]["train"],H[c]["cold"]):
   assert len(a)>=4 and len(b)>=3 and not (set(a)&set(b))
 for split in ("train","dev","cold"):
  rng=random.Random(517)
  for _ in range(120):
   e=make_episode(rng,split)
   for k in ("user","a","b"):
    assert len(e["events"][k])==3 and e["truth"][k] in (0,1)
 return {"pass":True,"cold_paraphrases_not_used_for_training_or_model_selection":True,
         "history_templates_training":sum(len(a) for k in H.values() for a in k["train"]),
         "history_templates_cold":sum(len(a) for k in H.values() for a in k["cold"]),
         "limitation":"All data and synthetic conceptual axes defined by researcher."}

WORDS=4096; MAXWORDS=28
def word_id(w):
 return 1+int.from_bytes(hashlib.blake2b(w.encode(),digest_size=4).digest(),"little")%(WORDS-1)
def encode(events):
 return [[([word_id(w) for w in re.findall("[a-z]+",s.lower())][:MAXWORDS]+[0]*MAXWORDS)[:MAXWORDS]
           for s in events] for events in (events["user"],events["a"],events["b"])]
def pack(rows):
 inputs=[[],[],[]];labels=[[],[],[]]
 for e in rows:
  a=encode(e["events"])
  for i,k in enumerate(("user","a","b")):
   inputs[i].append(a[i]);labels[i].append(e["truth"][k])
 return tuple(torch.tensor(a,dtype=torch.long) for a in inputs),tuple(torch.tensor(a,dtype=torch.long) for a in labels)
class Learner(nn.Module):
 def __init__(self):
  super().__init__()
  self.embedding=nn.Embedding(WORDS,24,padding_idx=0)
  self.user_gru=nn.GRU(24,28,batch_first=True)
  self.project_gru=nn.GRU(24,28,batch_first=True)
  self.user_head=nn.Linear(28,2)
  self.project_head=nn.Linear(28,2)
 def head(self,x,cat):
  mask=(x!=0).unsqueeze(-1)
  evt=(self.embedding(x)*mask).sum(2)/mask.sum(2).clamp_min(1)
  gru,cls=(self.user_gru,self.user_head) if cat=="user" else (self.project_gru,self.project_head)
  seq,_=gru(evt)
  return cls(seq[:,-1])
 def forward(self,u,a,b):
  return self.head(u,"user"),self.head(a,"project"),self.head(b,"project")

def train(args):
 rng=random.Random(args.seed+19)
 net=Learner();opt=torch.optim.AdamW(net.parameters(),lr=.003)
 # No cold paraphrase template is used here, even for dev selection.
 dr=random.Random(args.seed+990)
 dev=[make_episode(dr,"dev") for _ in range(128)]
 best=-1;best_step=0;state=None;history=[]
 for step in range(1,args.steps+1):
  examples=[make_episode(rng,"train") for _ in range(args.batch)]
  x,y=pack(examples);pred=net(*x)
  loss=sum(F.cross_entropy(q,g) for q,g in zip(pred,y))/3
  opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1);opt.step()
  if step==1 or step%args.eval_every==0 or step==args.steps:
   net.eval()
   with torch.no_grad():
    dx,dy=pack(dev)
    dp=net(*dx)
    correct=sum(int((p.argmax(-1)==g).sum()) for p,g in zip(dp,dy))
   score=correct/(3*len(dev))
   history.append({"step":step,"loss":round(float(loss.detach()),6),"dev_acc":round(score,6)})
   print(f"V92 TRAIN step={step} dev={correct}/{3*len(dev)} loss={float(loss.detach()):.5f}",flush=True)
   if score>best:
    best=score;best_step=step;state={k:v.detach().clone() for k,v in net.state_dict().items()}
   net.train()
 net.load_state_dict(state);net.eval()
 return net,{"dev_best":best,"selected_step":best_step,"history":history,"steps":args.steps,
              "train_episodes_processed":args.steps*args.batch,
              "learned_weights":sum(p.numel() for p in net.parameters())}
@torch.no_grad()
def predict(net,episodes):
 preds=[]
 for n in range(0,len(episodes),128):
  x,_=pack(episodes[n:n+128]);z=net(*x)
  preds.extend([{k:int(p[i].argmax()) for k,p in zip(("user","a","b"),z)}
                for i in range(len(x[0]))])
 return preds

def text_rule(events,kind):
 last=events[-1].lower()
 words={
  "user":(("uncertainty","validate","examine","shortcut","limitations","explored","risk","assumptions","tradeoffs","consequences","review"),
          ("trial","revise","attempt","observation","prototype","test","feedback","quick","pilot","iteration")),
  "project":(("benefit","outcome","purpose","objective","old steps","replaceable","goal"),
             ("contract","backward","public behavior","clients","protocol","compatib","interface"))}
 a,b=(sum(w in last for w in grp) for grp in words[kind])
 return int(b>a)

def score(episodes,preds):
 cats={k:{"correct":0,"total":0,"correction_correct":0,"correction_total":0} for k in ("user","a","b")}
 all_dec={"neutral":{"correct":0,"total":0},"binding":{"correct":0,"total":0}}
 distinct=0;distinct_good=0;examples=[]
 for e,p in zip(episodes,preds):
  gold=e["truth"]
  for k in gold:
   d=cats[k];d["total"]+=1;d["correct"]+=int(p[k]==gold[k])
   if e["corrected"][k]:
    d["correction_total"]+=1;d["correction_correct"]+=int(p[k]==gold[k])
   # Each of 3 cold task templates, two option orders, two evidence regimes.
   for _ in DILEMMAS["user" if k=="user" else "project"]:
    for swapped in (0,1):
     for constraint in (0,1):
      target=0 if constraint else gold[k]
      chosen=0 if constraint else p[k]
      bucket=all_dec["binding" if constraint else "neutral"]
      bucket["total"]+=1;bucket["correct"]+=int((target^swapped)==(chosen^swapped))
  if gold["a"]!=gold["b"]:
   distinct+=1;distinct_good+=int(p["a"]!=p["b"])
  if len(examples)<12:examples.append({"gold":gold,"pred":p,"corrections":e["corrected"]})
 total=sum(c["correct"] for c in cats.values())
 corrected_total=sum(c["correction_total"] for c in cats.values())
 return {"concept_accuracy":round(total/(len(episodes)*3),6),
         "corrected_accuracy":round(sum(c["correction_correct"] for c in cats.values())/max(1,corrected_total),6),
         "distinct_project_recovery":round(distinct_good/max(1,distinct),6),
         "by_state":cats,"new_task_controls":all_dec,"samples":examples}

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--audit-only",action="store_true")
 ap.add_argument("--steps",type=int,default=600)
 ap.add_argument("--batch",type=int,default=32)
 ap.add_argument("--eval-every",type=int,default=50)
 ap.add_argument("--cold",type=int,default=600)
 ap.add_argument("--seed",type=int,default=11)
 ap.add_argument("--output",default="compass-v92-concept-transfer.json")
 args=ap.parse_args()
 print("V92 AUDIT",json.dumps(audit()),flush=True)
 if args.audit_only:return
 assert args.steps>=1 and args.batch>0 and args.cold>=100
 torch.set_num_threads(2);torch.manual_seed(args.seed)
 start=time.monotonic()
 net,training=train(args)
 rng=random.Random(args.seed+7919)
 cold=[make_episode(rng,"cold") for _ in range(args.cold)]
 neural=predict(net,cold)
 baseline={
  "no_memory":[{"user":0,"a":0,"b":0} for e in cold],
  "personal_only":[{"user":x["user"],"a":0,"b":0} for x in neural],
  "project_only":[{"user":0,"a":x["a"],"b":x["b"]} for x in neural],
  "hierarchical_neural":neural,
  "textual_full_history_keyword_reader":[{k:text_rule(e["events"][k],"user" if k=="user" else "project") for k in ("user","a","b")} for e in cold],
  "known_principle_oracle":[e["truth"].copy() for e in cold]
 }
 stats={k:score(cold,v) for k,v in baseline.items()}
 # Controlled cross-project B swap: does NOT alter global or project A latent.
 modified=[{**e,"events":{**e["events"],"b":e["events"]["a"][:]} } for e in cold[:128]]
 changed=predict(net,modified)
 separation={"cases":len(modified),
             "global_unchanged":sum(x["user"]==y["user"] for x,y in zip(neural,changed)),
             "project_a_unchanged":sum(x["a"]==y["a"] for x,y in zip(neural,changed)),
             "project_b_uses_swapped_a_history":sum(y["b"]==y["a"] for y in changed)}
 assert separation["global_unchanged"]==separation["project_a_unchanged"]==separation["project_b_uses_swapped_a_history"]==separation["cases"]
 n=stats["hierarchical_neural"];t=stats["textual_full_history_keyword_reader"]
 gates={
  "cold_concept_at_least_85pct":n["concept_accuracy"]>=.85,
  "corrected_concept_at_least_80pct":n["corrected_accuracy"]>=.8,
  "opposing_project_separation_at_least_80pct":n["distinct_project_recovery"]>=.8,
  "hierarchical_better_than_each_single_stream":n["concept_accuracy"]>max(stats["personal_only"]["concept_accuracy"],stats["project_only"]["concept_accuracy"]),
  "neural_beats_textual_keyword_reader":n["concept_accuracy"]>t["concept_accuracy"],
  "strong_project_b_isolation":separation["cases"]==separation["global_unchanged"]==separation["project_a_unchanged"]
 }
 verdict="CONCEPT_PROBE_PROVISIONAL_PASS" if all(gates.values()) else "NO_GO_TO_LM_COUPLING"
 p=Path(args.output)
 save_file({k:v.detach().contiguous().cpu() for k,v in net.state_dict().items()},str(p.with_suffix(".safetensors")))
 output={"experiment":"COMPASS V9.2 hierarchical sequential concept learning",
  "verdict":verdict,"preflight":audit(),"train":training,"cold_size":args.cold,
  "arms":stats,"isolation":separation,"gates":gates,
  "elapsed_sec":round(time.monotonic()-start,2),
  "limitations":[
   "Synthetic invented binary concept axes and correction histories, not deep natural project intent.",
   "Cold paraphrase templates are disjoint, but vocabulary/meaning and task types overlap with TRAIN.",
   "Cold task descriptions are NOT interpreted by the neural model; decisions use a fixed scope-aware symbolic decoder.",
   "Text baseline is a keyword parser of full conversation history, not a frozen 360M LLM + strong RAG.",
   "No actual large-language-model neural coupling; this is deliberately a prerequisite representation test.",
   "No continuously persisted per-user neural state, independently trained individual adapter, or real-user evaluation."]}
 p.write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding="utf8")
 print("V92 FINAL",json.dumps({"verdict":verdict,"weights":training["learned_weights"],
      "arms":{k:{"concept":v["concept_accuracy"],"corrected":v["corrected_accuracy"],
                  "project_separation":v["distinct_project_recovery"]} for k,v in stats.items()},
      "gates":gates,"isolation":separation,"elapsed_sec":output["elapsed_sec"]}),flush=True)
if __name__=="__main__":main()
