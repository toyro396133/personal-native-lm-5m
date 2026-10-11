"""V9.4 proof-of-concept: learned concept state -> decisions about NEW tasks.

V9.3 small semantic memory weights are trained afresh using TRAIN-only
synthetic histories, then frozen; a separate tiny decision bridge learns
to evaluate the natural language of genuinely new task families.
FROZEN 360M FULL HISTORY prompting is measured on a matched test subset.

This is not a full frozen-LLM in-block neural adapter. No oracle task
scope/principle/gold label is passed into the neural bridge. Selected project
is supplied as workspace identity; no project gold or policy string is supplied.
"""
import argparse,copy,json,random,time,hashlib
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from study_v93 import FrozenSemanticEncoder,train_model,score as concept_score,prototypes
from fixtures_v94 import build,audit,FAMILY

def forward_memory(memory,vecs,project):
    with torch.no_grad():
        # input vector shape [3,3,384], streams: user, project A, project B
        states=[]
        for i,scope in enumerate(("user","project","project")):
            _,h=memory.scan(vecs[i].unsqueeze(0),scope)
            states.append(h[0].detach())
        return states[0],states[1 if project=="a" else 2]

class DecisionBridge(nn.Module):
    """Symmetric option-wise scoring; no labels, task family, gold, or scope bits."""
    def __init__(self,width=112):
        super().__init__()
        self.state=nn.Sequential(nn.Linear(96,64),nn.LayerNorm(64),nn.Tanh())
        self.question=nn.Sequential(nn.Linear(384,64),nn.Tanh())
        self.action=nn.Sequential(nn.Linear(384,64),nn.Tanh())
        self.scorer=nn.Sequential(nn.Linear(64*3+64,112),nn.GELU(),
                                   nn.Linear(112,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,u,p,q,actions):
        """Broadcast 48D user & project and 384D task & option embeddings."""
        c=self.state(torch.cat((u,p),dim=-1))
        qv=self.question(q)
        av=self.action(actions)
        # option-aware conditioned interaction; gate can *actually* change choice
        f=torch.cat((av,c.expand(2,-1),qv.expand(2,-1),av*qv),dim=-1)
        return self.scorer(f).squeeze(-1)

class FrozenFeatureCache:
 def __init__(self,encoder,memory):
  self.encoder=encoder;self.memory=memory;self.cache={}
 def key(self,history,project,question,options):
  return (tuple((k,tuple(history[k])) for k in ("user","a","b")),
          project,question,tuple(options))
 @torch.no_grad()
 def feature(self,history,project,question,options):
  key=self.key(history,project,question,options)
  if key in self.cache:return self.cache[key]
  alltext=history["user"]+history["a"]+history["b"]+[question]+list(options)
  self.encoder.encode_missing(alltext)
  vecs=torch.stack([torch.stack([self.encoder.vectors[s] for s in history[k]])
           for k in ("user","a","b")])
  u,p=forward_memory(self.memory,vecs,project)
  q=self.encoder.vectors[question]
  act=torch.stack([self.encoder.vectors[t] for t in options])
  data=(u,p,q,act)
  self.cache[key]=data
  return data
 def pair(self,case,side):
  row=case[side]
  return self.feature(row["histories"],case["project"],case["question"],case["options"])

def evaluate(cases,predict_fn,with_details=True):
 results=defaultdict(lambda:{"n":0,"both_correct":0,"correct_decisions":0,
                   "actual_flips":0,"expected_flip_agreement":0})
 detail=[]
 for case in cases:
  answers=[int(predict_fn(case,"original")),int(predict_fn(case,"changed"))]
  actual=[case["original"]["gold"],case["changed"]["gold"]]
  result=answers==actual
  for k in ("all","binding" if case["binding"] else "neutral",
            "user" if case["scope"]=="user" else "project"):
    g=results[k];g["n"]+=1;g["both_correct"]+=int(result)
    g["correct_decisions"]+=int(answers[0]==actual[0])+int(answers[1]==actual[1])
    g["actual_flips"]+=int(answers[0]!=answers[1])
    g["expected_flip_agreement"]+=int((answers[0]!=answers[1])==case["expected_flip"])
  if with_details:
    detail.append({"id":case["id"],"scope":case["scope"],"family":case["family"],
      "project":case["project"],"binding":case["binding"],"swapped":case["swapped"],
      "truth":actual,"predicted":answers,"both_correct":result})
 return dict(results),detail

def train_bridge(args,fc,train,dev):
 random.seed(args.seed+42);torch.manual_seed(args.seed+42)
 bridge=DecisionBridge(args.bridge_width)
 optim=torch.optim.AdamW(bridge.parameters(),lr=args.lr,weight_decay=.002)
 # Cache frozen encoded inputs before fitting; test embeddings not materialized yet.
 tdata=[(fc.pair(c,"original"),fc.pair(c,"changed"),c) for c in train]
 ddata=[(fc.pair(c,"original"),fc.pair(c,"changed"),c) for c in dev]
 selected=-1;selection_epoch=0;best=None;history=[]
 for epoch in range(1,args.epochs+1):
  random.Random(args.seed*1100+epoch).shuffle(tdata)
  batchloss=[]
  bridge.train()
  for pos in range(0,len(tdata),args.batch):
   batch=tdata[pos:pos+args.batch]
   losses=[]
   for a,b,case in batch:
    va=bridge(*a);vb=bridge(*b)
    signs=[1. if case[x]["gold"]==0 else -1. for x in ("original","changed")]
    margins=[va[0]-va[1],vb[0]-vb[1]]
    supervised=sum(F.softplus(.6-sign*margin) for sign,margin in zip(signs,margins))/2
    if case["binding"]:
     contrast=.08*(margins[0]-margins[1]).square()
    else:
     contrast=.4*F.softplus(.8-signs[0]*(margins[0]-margins[1]))
    losses.append(supervised+contrast)
   loss=torch.stack(losses).mean()
   optim.zero_grad(set_to_none=True);loss.backward()
   nn.utils.clip_grad_norm_(bridge.parameters(),1.5);optim.step()
   batchloss.append(float(loss.detach()))
  bridge.eval()
  with torch.no_grad():
   both=sum(
    int(int(bridge(*a)[1]>bridge(*a)[0])==case["original"]["gold"] and
        int(bridge(*b)[1]>bridge(*b)[0])==case["changed"]["gold"])
    for a,b,case in ddata)
  if both>selected:
   selected=both;selection_epoch=epoch
   best={k:v.detach().clone() for k,v in bridge.state_dict().items()}
  if epoch==1 or epoch%10==0 or epoch==args.epochs:
   print(f"V94 BRIDGE epoch={epoch} dev_both={both}/{len(ddata)} loss={sum(batchloss)/len(batchloss):.5f}",flush=True)
  history.append({"epoch":epoch,"dev_both":both,
                  "train_loss":round(sum(batchloss)/len(batchloss),5)})
 bridge.load_state_dict(best);bridge.eval()
 return bridge,{"selected_epoch":selection_epoch,"best_dev_both":selected,"dev_pairs":len(ddata),
                "training_pairs":len(tdata),"epochs":args.epochs,
                "bridge_trainable_weights":sum(p.numel() for p in bridge.parameters()),
                "history":history}

def text_rule(history,scope):
 """Cheap textual decision baseline: explicit latest-message textual keyword reader.
 Uses original V9.2 vocabulary, not ground-truth latent state.
 """
 from study_v92 import text_rule as tr
 kind="user" if scope=="user" else "project"
 return tr(history,kind)

def symbolic_baseline(case,side):
 # Researcher-authored synthetic task contract oracle. NOT a fair free-form LM.
 # A rule with access to fixture scope/binding indicates benchmark easiness.
 hist=case[side]["histories"]
 target="user" if case["scope"]=="user" else case["project"]
 if case["binding"]:
  semantic=FAMILY[case["scope"]]["bind_semantic"]
 else:
  semantic=text_rule(hist[target],case["scope"])
 return semantic ^case["swapped"]

def semantic_baseline(case,side,encoder,protos):
 hist=case[side]["histories"]
 key="user" if case["scope"]=="user" else case["project"]
 target="user" if key=="user" else case["project"]
 v=encoder.vectors[hist[target][-1]]
 mode=int(torch.argmax(protos[key]@v).item())
 mode=FAMILY[key]["bind_semantic"] if case["binding"] else mode
 return mode^case["swapped"]

def lm_yes_no_tokens(tok):
 for a,b in (("Yes","No"),(" Yes"," No")):
  x=tok.encode(a,add_special_tokens=False);y=tok.encode(b,add_special_tokens=False)
  if len(x)==len(y)==1 and x[0]!=y[0]:return x[0],y[0]
 raise RuntimeError("No unique yes/no tokens; no valid frozen LM baseline")

def frozen_lm_baseline(args,cases):
 """Genuinely runs SmolLM2 360M on full personal+selected-project text.

 Contrast: neural bridge does NOT receive these historical strings at
 decision-time, only their learned numeric representations; this baseline
 DOES receive full relevant history as part of LM prompt.
 """
 from transformers import AutoTokenizer,AutoModelForCausalLM
 model_id="HuggingFaceTB/SmolLM2-360M-Instruct"
 tok=AutoTokenizer.from_pretrained(model_id)
 yes,no=lm_yes_no_tokens(tok)
 model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch.float32)
 model.eval();model.requires_grad_(False)
 count=0
 def predict(case,side):
  nonlocal count
  hist=case[side]["histories"]
  project=case["project"]
  preface=(
   "You are judging a NEW technical decision using previously agreed principles. "
   "Interpret the user's GENERAL principles and the active PROJECT's principles, "
   "pay attention to any newer corrections, and honor any binding requirement "
   "stated in the problem. Pick the action better aligned with this context.\n"
   "Prior user statements (oldest to newest):\n"+
   "\n".join("- "+s for s in hist["user"])+
   "\nPrior statements about active project (oldest to newest):\n"+
   "\n".join("- "+s for s in hist[project])+
   "\nNew problem: "+case["question"]+
   "\nOption A: "+case["options"][0]+
   "\nOption B: "+case["options"][1]+"\n"
  )
  margins=[]
  for candidate in case["options"]:
   prompt=preface+"Candidate action: "+candidate+"\nIs this candidate the BETTER permitted action? Answer Yes or No."
   ids=tok.apply_chat_template([{"role":"user","content":prompt}],
       tokenize=True,add_generation_prompt=True,return_tensors="pt")
   with torch.no_grad():
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits[0,-1]
   margins.append(float(logits[yes]-logits[no]))
   count+=1
  if count%32==0:print(f"V94 FROZEN LM candidate forwards={count}",flush=True)
  return int(margins[1]>margins[0])
 scored,rows=evaluate(cases,predict)
 del model
 return {"model":model_id,"inputs_include_full_relevant_history":True,
         "candidate_forward_calls":count,"scores":scored,"case_outputs":rows}

def verify_swapped_order(bridge,cache,cases):
 good=0;total=0
 with torch.no_grad():
  for case in cases[:64]:
   for side in ("original","changed"):
    a=cache.pair(case,side)
    original=int(bridge(*a)[1]>bridge(*a)[0])
    u,p,q,opts=a
    flipped=int(bridge(u,p,q,opts.flip(0))[1]>bridge(u,p,q,opts.flip(0))[0])
    good+=int(original!=flipped);total+=1
 return {"correct":good,"total":total}

def choose_lm_subset(cold,count=32):
 # Every scope/binding/swap cell represented equally; no outcome peeking.
 slots=defaultdict(list)
 for c in cold:
  slots[(c["scope"],c["binding"],c["swapped"])].append(c)
 assert len(slots)==8
 result=[]
 for key in sorted(slots,key=str):
  result+=slots[key][:count//8]
 assert len(result)==count
 return result

def main():
 ap=argparse.ArgumentParser()
 ap.add_argument("--steps",type=int,default=600)
 ap.add_argument("--epochs",type=int,default=45)
 ap.add_argument("--batch",type=int,default=16)
 ap.add_argument("--bridge-width",type=int,default=112)
 ap.add_argument("--lr",type=float,default=.002)
 ap.add_argument("--seed",type=int,default=11)
 ap.add_argument("--train-n",type=int,default=384)
 ap.add_argument("--cold-n",type=int,default=128)
 ap.add_argument("--lm-subset",type=int,default=32)
 ap.add_argument("--output",default="compass-v94-concept-to-decision.json")
 args=ap.parse_args()
 t0=time.monotonic();torch.set_num_threads(2)
 # Reserve cold before any optimization, including selection.
 training,cold=build(args.seed,args.train_n,args.cold_n)
 chk=audit(training,cold)
 assert args.train_n>=128 and args.cold_n==128 and args.lm_subset==32
 print("V94 AUDIT",json.dumps(chk),flush=True)
 encoder=FrozenSemanticEncoder()
 encoder.cache_training_vocab()
 class A:pass
 conf=A();conf.width=48;conf.seed=args.seed;conf.steps=args.steps
 conf.batch=32;conf.lr=.003;conf.check_every=50
 memory,pretrained_info=train_model(conf,encoder)
 memory.requires_grad_(False);memory.eval()
 # Split training vs development by INDEX, not gold or cold outcome.
 tr=[c for i,c in enumerate(training) if i%5]
 dv=[c for i,c in enumerate(training) if i%5==0]
 assert len(tr)>200 and len(dv)>60
 cache=FrozenFeatureCache(encoder,memory)
 bridge,training_info=train_bridge(args,cache,tr,dv)
 # Freeze weights before ANY blind-test feature is materialized or inspected.
 save_file({k:v.detach().cpu().contiguous() for k,v in bridge.state_dict().items()},
           str(Path(args.output).with_suffix(".safetensors")))
 bridge.requires_grad_(False)
 # Existing histories in cold were never fed into optimizer.
 for c in cold:
  for side in ("original","changed"):
   cache.pair(c,side)
 prototype=prototypes(encoder)
 def neural(c,s):
  with torch.no_grad():
   a=bridge(*cache.pair(c,s))
   return int(a[1]>a[0])
 comps={}
 for name,fn in (
   ("neural_numeric_histories_and_new_semantic_task",neural),
   ("text_history_keyword_plus_trusted_task_routing",symbolic_baseline),
   ("frozen_semantic_prototype_plus_trusted_task_routing",lambda c,s:semantic_baseline(c,s,encoder,prototype)),
   ("no_personal_history_constant_action_zero",lambda c,s:c["swapped"]),
 ):
  score,detail=evaluate(cold,fn)
  comps[name]={"score":score,"cases":detail}
  print("V94 COLD",name,json.dumps(score),flush=True)
 order=verify_swapped_order(bridge,cache,cold)
 selected=choose_lm_subset(cold,args.lm_subset)
 neural_subset,_=evaluate(selected,neural)
 text_subset,_=evaluate(selected,symbolic_baseline)
 lm=frozen_lm_baseline(args,selected)
 result_score=comps["neural_numeric_histories_and_new_semantic_task"]["score"]
 neutral=result_score["neutral"];binding=result_score["binding"]
 n_subset=neural_subset["all"]["both_correct"]
 lm_subset=lm["scores"]["all"]["both_correct"]
 gates={
  "cold_neutral_pairs_both_correct_at_least_60pct":neutral["both_correct"]>=.60*neutral["n"],
  "cold_binding_pairs_both_correct_at_least_75pct":binding["both_correct"]>=.75*binding["n"],
  "choice_order_invariance_at_least_95pct":order["correct"]>=.95*order["total"],
  "neural_beats_frozen_lm_with_full_history_on_matched_subset":
      n_subset>lm_subset,
  "neural_beats_explicit_synthetic_rule_on_full_cold":
      result_score["all"]["both_correct"]>comps["text_history_keyword_plus_trusted_task_routing"]["score"]["all"]["both_correct"],
 }
 verdict="PROVISIONAL_SYNTHETIC_PASS_NOT_GENERAL_INTENT" if all(gates.values()) else "NO_GO_TO_LM_ADAPTER_INTEGRATION"
 out={
  "version":"9.4","verdict":verdict,"synthetic_preregistered_audit":chk,
  "learned_memory":pretrained_info,"decision_bridge":training_info,
  "pretrained_semantic_encoder":"sentence-transformers/all-MiniLM-L6-v2",
  "downstream_frozen_lm_baseline":lm,
  "matched_subset":{
   "cases":len(selected),"neural":neural_subset,"lexical":text_subset,
   "frozen_lm":lm["scores"]
  },
  "cold_result":comps,"choice_order_invariance":order,"registered_gates":gates,
  "runtime_seconds":round(time.monotonic()-t0,2),
  "limitations":[
    "Synthetic author-invented one-bit global/project principles and choices; not independently blind human-created natural dilemmas.",
    "Decision bridge is a small neural scorer operating on pretrained 384D semantic task/action embeddings and learned numeric histories; NOT inside frozen 360M Transformer.",
    "Full-history LLM control genuinely runs frozen 360M for 32 balanced paired situations; do not compare 32-pair scores directly to full 128-pair scores.",
    "Metadata selects active project for workspace context, but no task scope/label/expected answer is passed into neural model. Deterministic baselines use trusted task scope and binding oracle, giving them an advantage.",
    "No online real-user model checkpoint updates, no permissions/source-auth verification or real-world deployment.",
    "Two project states and global personal state are shared recurrent memory weights, not an independently trained language model for each user."] }
 Path(args.output).write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
 print("V94 FINAL",json.dumps({
  "verdict":verdict,
  "neural_cold":result_score,
  "frozen_lm_subset":lm["scores"],
  "neural_subset":neural_subset,
  "order":order,"gates":gates,
  "duration":out["runtime_seconds"]
 }),flush=True)
if __name__=="__main__":main()
