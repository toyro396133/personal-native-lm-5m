"""SELF/COMPASS V9.3 — frozen semantic encoder + trainable sequential project/user memory.

This is a controlled CONCEPT RECOVERY study, not connected to a large LM.
All cases and gold labels are researcher-authored English. The purported
'new dilemmas' are NOT answered in this experiment. No user-specific gradient
training; separate transient per-user/per-project RNN states.
"""
import argparse,copy,json,random,time
from pathlib import Path
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from study_v92 import H,make_episode,text_rule

BASE="sentence-transformers/all-MiniLM-L6-v2"
DIM=384
# Unseen-in-training, even for model selection. Test-only author paraphrases.
FAR={
 "user":{
 0:[
  "Before signing off, trace the ways this could go wrong and scrutinize the trade-offs.",
  "A confident guess is not enough here; establish what might fail before choosing.",
  "I would rather investigate uncertain consequences than commit prematurely.",
  "Consider the drawbacks carefully, then move once the unknowns are addressed."],
 1:[
  "Build a tiny working slice, see what it reveals, and redirect based on evidence.",
  "Waiting for a complete plan is costly; take a reversible first step to discover more.",
  "An inexpensive hands-on trial is the best way to figure out our next move.",
  "Start with something small that runs, learn from it, and refine the approach."]},
 "project":{
 0:[
  "The present interface is merely a vehicle; optimize for the benefit people need.",
  "Discard inherited mechanics when they obstruct the result we originally sought.",
  "The north star is the outcome, even if reaching it means replacing familiar steps.",
  "Revisit implementation choices whenever they weaken the underlying mission."],
 1:[
  "Consumers depend on the old public behavior: backward compatibility takes precedence.",
  "Modernize beneath the surface, but third-party callers must observe the same contract.",
  "The integration promise to external systems cannot change during this rebuild.",
  "Preserving the established boundary for clients outweighs internal refactoring ease."]},
}

def far_episode(rng):
 e=make_episode(rng,"cold")
 for key,mode in e["truth"].items():
  cat="user" if key=="user" else "project"
  first=1-mode if e["corrected"][key] else mode
  e["events"][key]=[
      rng.choice(FAR[cat][first]),
      rng.choice(FAR[cat][first]),
      rng.choice(FAR[cat][mode]),
  ]
 return e

def audit():
 assert set(FAR)=={"user","project"}
 for cat in FAR:
  for mode in (0,1):
   train=set(H[cat]["train"][mode]);near=set(H[cat]["cold"][mode]);far=set(FAR[cat][mode])
   assert len(far)>=4 and not (far & train) and not (far & near)
 assert len(set(x for v in FAR.values() for arr in v.values() for x in arr))==16
 for split in ("train","dev","cold","far"):
  rng=random.Random(731)
  cases=[far_episode(rng) if split=="far" else make_episode(rng,split) for _ in range(50)]
  assert all(all(len(a)==3 for a in e["events"].values()) for e in cases)
 return {"status":"PASS","encoder":BASE,"frozen_semantic_dim":DIM,
         "cold_near_templates":sum(len(a) for v in H.values() for a in v["cold"]),
         "cold_far_templates":sum(len(a) for v in FAR.values() for a in v.values()),
         "known_limit":"All class definitions, corrections, and paraphrases written by the researcher"}

class FrozenSemanticEncoder:
 def __init__(self,device="cpu"):
  from transformers import AutoModel,AutoTokenizer
  self.tokenizer=AutoTokenizer.from_pretrained(BASE)
  self.model=AutoModel.from_pretrained(BASE).to(device).eval()
  self.model.requires_grad_(False)
  self.device=device
  self.vectors={}
  self.calls=0
 @torch.inference_mode()
 def encode_missing(self,texts):
  todo=sorted(set(t for t in texts if t not in self.vectors))
  for start in range(0,len(todo),24):
   items=todo[start:start+24]
   b=self.tokenizer(items,return_tensors="pt",truncation=True,padding=True,max_length=128).to(self.device)
   z=self.model(**b).last_hidden_state
   mask=b["attention_mask"].unsqueeze(-1)
   emb=F.normalize((z*mask).sum(1)/mask.sum(1).clamp_min(1),dim=-1).cpu()
   for t,v in zip(items,emb):self.vectors[t]=v.contiguous()
   self.calls+=1
  if todo:print(f"V93 ENCODER cached={len(self.vectors)} sentences; additional={len(todo)}",flush=True)
 def cache_training_vocab(self):
  self.encode_missing([text for cat in H.values() for classes in cat["train"] for text in classes])
 def encode_cases(self,episodes):
  self.encode_missing([s for e in episodes for stream in e["events"].values() for s in stream])
 def tensors(self,episodes):
  return tuple(torch.stack([torch.stack([self.vectors[s] for s in e["events"][k]])
       for e in episodes]) for k in ("user","a","b"))

class ConceptState(nn.Module):
 """Shared encoder interface; persistent 3-turn state PER SCOPE, no context leakage."""
 def __init__(self,width=48):
  super().__init__()
  self.compress=nn.Sequential(nn.Linear(DIM,64),nn.LayerNorm(64),nn.Tanh())
  self.user_gru=nn.GRUCell(64,width)
  self.project_gru=nn.GRUCell(64,width)
  self.user_head=nn.Linear(width,2)
  self.project_head=nn.Linear(width,2)
  self.width=width
 def scan(self,events,cat):
  gru,head=(self.user_gru,self.user_head) if cat=="user" else (self.project_gru,self.project_head)
  h=torch.zeros(events.shape[0],self.width,dtype=events.dtype,device=events.device)
  for i in range(events.shape[1]):
   h=gru(self.compress(events[:,i,:]),h)
  return head(h),h
 def forward(self,u,a,b):
  U,_=self.scan(u,"user")
  A,_=self.scan(a,"project")
  B,_=self.scan(b,"project")
  return U,A,B
 def step(self,event_vector,prior,scope):
  """One online memory update from a new statement, WITHOUT backprop/optimizer."""
  z=self.compress(event_vector)
  g=self.user_gru if scope=="user" else self.project_gru
  return g(z,prior)
 def run_stream(self,events,scope):
  state=torch.zeros(events.shape[0],self.width,device=events.device)
  for t in range(events.shape[1]):state=self.step(events[:,t,:],state,scope)
  return state

def targets(episodes):
 return tuple(torch.tensor([e["truth"][k] for e in episodes],dtype=torch.long)
       for k in ("user","a","b"))

@torch.inference_mode()
def predict(net,embedding,episodes,batch=96):
 outputs=[]
 for i in range(0,len(episodes),batch):
  x=embedding.tensors(episodes[i:i+batch]);p=net(*x)
  outputs.extend([{key:int(p[j][t].argmax().item()) for j,key in enumerate(("user","a","b"))}
     for t in range(len(x[0]))])
 return outputs

def score(episodes,predictions):
 by={k:{"right":0,"total":0,"corrected_right":0,"corrected_total":0,
        "class0_right":0,"class0_total":0,"class1_right":0,"class1_total":0}
       for k in ("user","a","b")}
 conflicts=0;distinct=0
 for e,p in zip(episodes,predictions):
  for scope in by:
   a=by[scope];truth=e["truth"][scope]
   good=int(p[scope]==truth)
   a["right"]+=good;a["total"]+=1
   a[f"class{truth}_right"]+=good;a[f"class{truth}_total"]+=1
   if e["corrected"][scope]:
    a["corrected_right"]+=good;a["corrected_total"]+=1
  if e["truth"]["a"]!=e["truth"]["b"]:
   conflicts+=1;distinct+=int(p["a"]!=p["b"])
 return {"overall_accuracy":round(sum(a["right"] for a in by.values())/(len(episodes)*3),6),
  "user_accuracy":round(by["user"]["right"]/len(episodes),6),
  "project_accuracy":round((by["a"]["right"]+by["b"]["right"])/(len(episodes)*2),6),
  "corrected_accuracy":round(sum(a["corrected_right"] for a in by.values())/
        max(1,sum(a["corrected_total"] for a in by.values())),6),
  "different_project_recovery":round(distinct/max(1,conflicts),6),
  "by_state":by}

def keyword_predictions(episodes):
 return [{k:text_rule(e["events"][k],"user" if k=="user" else "project")
          for k in ("user","a","b")} for e in episodes]
def prototypes(encoder):
 return {scope:torch.stack([
  F.normalize(torch.stack([encoder.vectors[t] for t in H[scope]["train"][mode]]).mean(0),dim=0)
  for mode in (0,1)]) for scope in ("user","project")}
def prototype_predictions(episodes,encoder,weights=(0.,0.,1.)):
 pr=prototypes(encoder);result=[]
 for e in episodes:
  modes={}
  for k in ("user","a","b"):
   w=torch.tensor(weights,dtype=torch.float32)
   # EXACT SAME full-history texts available to non-neural nearest-neighbor comparator.
   v=F.normalize(sum(encoder.vectors[s]*float(w[i]) for i,s in enumerate(e["events"][k])),dim=0)
   cat="user" if k=="user" else "project"
   modes[k]=int((pr[cat]@v).argmax())
  result.append(modes)
 return result

def train_model(args,encoder):
 torch.manual_seed(args.seed);random.seed(args.seed)
 net=ConceptState(args.width)
 opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=0.001)
 rng=random.Random(511+args.seed)
 d_rng=random.Random(9731+args.seed)
 # Dev uses TRAIN statement templates only; no unseen paraphrases used in selection.
 dev=[make_episode(d_rng,"dev") for _ in range(192)]
 encoder.encode_cases(dev)
 Dx=encoder.tensors(dev);Dy=targets(dev)
 best=-1;best_step=0;best_sd=None;history=[]
 for step in range(1,args.steps+1):
  batch=[make_episode(rng,"train") for _ in range(args.batch)]
  X=encoder.tensors(batch);Y=targets(batch)
  q=net(*X);loss=sum(F.cross_entropy(z,t) for z,t in zip(q,Y))/3
  opt.zero_grad(set_to_none=True);loss.backward()
  nn.utils.clip_grad_norm_(net.parameters(),1.0);opt.step()
  if step==1 or step%args.check_every==0 or step==args.steps:
   net.eval()
   with torch.no_grad():
    out=net(*Dx)
    accurate=sum(int((p.argmax(-1)==y).sum()) for p,y in zip(out,Dy))
   dev_accuracy=accurate/(3*len(dev))
   if dev_accuracy>best:
    best=dev_accuracy;best_sd={k:v.detach().clone() for k,v in net.state_dict().items()};best_step=step
   history.append({"step":step,"dev_accuracy":round(dev_accuracy,6),"train_loss":round(float(loss.detach()),6)})
   print(f"V93 TRAIN step={step} dev_correct={accurate}/{3*len(dev)} loss={float(loss.detach()):.5f}",flush=True)
   net.train()
 net.load_state_dict(best_sd);net.eval()
 return net,{"best_step":best_step,"best_dev":best,"steps":args.steps,"batch":args.batch,
             "trainable_memory_weights":sum(x.numel() for x in net.parameters()),"history":history}

@torch.inference_mode()
def causal_audit(net,encoder,cold):
 left=cold[:128]
 moved=copy.deepcopy(left)
 for e in moved:e["events"]["b"]=e["events"]["a"][:]
 encoder.encode_cases(moved)
 a=predict(net,encoder,left);b=predict(net,encoder,moved)
 isolation={
   "cases":len(left),
   "personal_unchanged":sum(x["user"]==y["user"] for x,y in zip(a,b)),
   "project_a_unchanged":sum(x["a"]==y["a"] for x,y in zip(a,b)),
   "project_b_follows_replaced_a_history":sum(y["b"]==y["a"] for y in b)}
 assert all(isolation[x]==len(left) for x in isolation if x!="cases")
 # Verify truly sequential update computes the exact full-scan memory.
 tensors=encoder.tensors(left[:16])
 for i,scope in enumerate(("user","project","project")):
  logits,h=net.scan(tensors[i],scope)
  stepped=net.run_stream(tensors[i],scope)
  torch.testing.assert_close(h,stepped,rtol=1e-5,atol=1e-6)
 return isolation

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--audit-only",action="store_true")
 ap.add_argument("--steps",type=int,default=600)
 ap.add_argument("--batch",type=int,default=32)
 ap.add_argument("--check-every",type=int,default=50)
 ap.add_argument("--near",type=int,default=600)
 ap.add_argument("--far",type=int,default=600)
 ap.add_argument("--seed",type=int,default=11)
 ap.add_argument("--width",type=int,default=48)
 ap.add_argument("--lr",type=float,default=.003)
 ap.add_argument("--output",default="compass-v93-semantic-memory.json")
 args=ap.parse_args()
 check=audit();print("V93 AUDIT",json.dumps(check),flush=True)
 if args.audit_only:return
 assert args.steps>0 and args.batch>0 and args.near>=128 and args.far>=128
 torch.set_num_threads(2);start=time.monotonic()
 encoder=FrozenSemanticEncoder()
 encoder.cache_training_vocab()
 net,train_info=train_model(args,encoder)
 near_rng=random.Random(8221+args.seed)
 far_rng=random.Random(9331+args.seed)
 near=[make_episode(near_rng,"cold") for _ in range(args.near)]
 far=[far_episode(far_rng) for _ in range(args.far)]
 encoder.encode_cases(near+far)
 outputs={};diagnostics={}
 for title,data in (("near",near),("far",far)):
  model_pred=predict(net,encoder,data)
  # Ablations: same trained encoder+weights with one stream blanked.
  user_only=[{"user":p["user"],"a":0,"b":0} for p in model_pred]
  project_only=[{"user":0,"a":p["a"],"b":p["b"]} for p in model_pred]
  arms={
   "no_memory":[{"user":0,"a":0,"b":0} for _ in data],
   "hierarchical_neural":model_pred,
   "user_only":user_only,
   "project_only":project_only,
   "full_history_keyword_reader":keyword_predictions(data),
   "frozen_semantic_last_statement_prototypes":prototype_predictions(data,encoder),
   "frozen_semantic_full_history_prototypes":prototype_predictions(data,encoder,(.15,.15,.70)),
   "oracle":[e["truth"] for e in data],
  }
  outputs[title]={k:score(data,v) for k,v in arms.items()}
  diagnostics[title]=[
    {"history":e["events"],"gold":e["truth"],"correction":e["corrected"],"neural":p}
    for e,p in zip(data[:24],model_pred[:24])]
  print("V93 COLD",title,json.dumps({k:{"overall":v["overall_accuracy"],
       "user":v["user_accuracy"],"project":v["project_accuracy"],
       "corrected":v["corrected_accuracy"]} for k,v in outputs[title].items()}),flush=True)
 isolate=causal_audit(net,encoder,far)
 neural_far=outputs["far"]["hierarchical_neural"]
 competitors=outputs["far"]
 max_text=max(competitors[k]["overall_accuracy"] for k in (
   "full_history_keyword_reader","frozen_semantic_last_statement_prototypes",
   "frozen_semantic_full_history_prototypes"))
 gates={
  "near_accuracy_at_least_85pct":outputs["near"]["hierarchical_neural"]["overall_accuracy"]>=.85,
  "far_accuracy_at_least_75pct":neural_far["overall_accuracy"]>=.75,
  "far_user_at_least_70pct":neural_far["user_accuracy"]>=.70,
  "far_correction_at_least_75pct":neural_far["corrected_accuracy"]>=.75,
  "far_conflicting_projects_at_least_80pct":neural_far["different_project_recovery"]>=.80,
  "neural_outperforms_best_text_or_semantic_baseline_by_2_points":
     neural_far["overall_accuracy"]>=max_text+.02,
  "project_isolation_and_incremental_state":isolate["personal_unchanged"]==
     isolate["project_a_unchanged"]==isolate["project_b_follows_replaced_a_history"]==isolate["cases"],
 }
 decision="PROVISIONAL_CONCEPT_PILOT_PASS_NOT_LM_PROOF" if all(gates.values()) else "NO_GO_TO_BACKBONE_COUPLING"
 p=Path(args.output)
 save_file({k:v.detach().cpu().contiguous() for k,v in net.state_dict().items()},
           str(p.with_suffix(".safetensors")))
 summary={
  "experiment":"COMPASS V9.3 frozen semantic encoder and incremental hierarchical concept memory",
  "scientific_verdict":decision,"frozen_encoder":BASE,"frozen_encoder_dim":DIM,
  "preflight":check,"train":train_info,"heldout":{"near":args.near,"far":args.far},
  "results":outputs,"causal_isolation":isolate,"registered_gates":gates,
  "sample_prediction_cases":diagnostics,"encoder_unique_texts":len(encoder.vectors),
  "elapsed_sec":round(time.monotonic()-start,3),
  "caveats":[
   "All concept axes and paraphrase cases synthetic; independently authored blind human test NOT performed.",
   "No LM inference; learned concepts NOT used in a frozen general LM to reason about novel dilemmas.",
   "Strongest explicit textual baseline here is full-history rule or semantic exemplar matching, NOT LLM+RAG.",
   "One shared 48-width recurrent updater, not separately trained individual and project weights.",
   "Two concepts per scope, rather than an open-world continuous learned project vision.",
   "Pretrained semantic features are computed for each new statement; fixed encoder may be costly at scale.",
   "No deployed persistent per-user storage, deletion, permissions or factual source authority checks."]}
 p.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf8")
 print("V93 FINAL",json.dumps({
   "scientific_verdict":decision,"trainable":train_info["trainable_memory_weights"],
   "near_hierarchical":outputs["near"]["hierarchical_neural"]["overall_accuracy"],
   "far_hierarchical":neural_far["overall_accuracy"],
   "far_user":neural_far["user_accuracy"],
   "far_text_keyword":outputs["far"]["full_history_keyword_reader"]["overall_accuracy"],
   "far_semantic_exemplars":outputs["far"]["frozen_semantic_full_history_prototypes"]["overall_accuracy"],
   "gates":gates,"isolation":isolate,"elapsed_sec":summary["elapsed_sec"]}),flush=True)
if __name__=="__main__":main()
