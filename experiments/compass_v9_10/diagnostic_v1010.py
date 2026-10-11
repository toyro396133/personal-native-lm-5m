"""Post-hoc V9.10 memory ablation. NOT blind; fixed frozen weights.

No cold cases, gold or observed V9.10 mistakes update any weights. Each
history transformation is applied to every item, state and seed identically.
"""
import argparse
import copy
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch
from fixtures_v94 import build
from fixtures_v1010 import create
from study_v93 import FrozenSemanticEncoder,train_model
from study_v94 import FrozenFeatureCache
from study_v98 import train_scope_head,DEV_FAMILIES
from study_v99 import SourceChooser,SourceResolver
from fixtures_v99 import MOCK_SIGNERS
from study_v1010 import metrics_by

VARIANTS=(
 "full5_original",
 "first3_preference_only",
 "first4_one_secondary",
 "last3_secondary_dominant",
 "secondary_first_five",
 "repeat_primary_tail_five",
)
def transform(case,kind):
 c=copy.deepcopy(case)
 for side in ("original","changed"):
  for stream in ("user","a","b"):
   h=c[side]["histories"][stream]
   assert len(h)==5
   if kind=="full5_original":out=h
   elif kind=="first3_preference_only":out=h[:3]
   elif kind=="first4_one_secondary":out=h[:4]
   elif kind=="last3_secondary_dominant":out=h[2:]
   elif kind=="secondary_first_five":out=h[3:]+h[:3]
   elif kind=="repeat_primary_tail_five":out=h[:3]+[h[2],h[2]]
   else:raise ValueError(kind)
   c[side]["histories"][stream]=out
 return c

@torch.no_grad()
def state_effect(cases,cache):
 scores=defaultdict(list)
 for c in cases:
  a=cache.pair(c,"original");b=cache.pair(c,"changed")
  ix=0 if c["scope"]=="user" else 1
  x=a[ix];y=b[ix]
  scores[c["language"]+"_"+c["scope"]].append({
   "l2":float(torch.norm(x-y).item()),
   "cosine":float(torch.nn.functional.cosine_similarity(x,y,dim=0).item())
  })
 return {k:{"n":len(v),"l2_mean":round(sum(e["l2"] for e in v)/len(v),6),
           "cosine_mean":round(sum(e["cosine"] for e in v)/len(v),6)}
         for k,v in scores.items()}

def main():
 p=argparse.ArgumentParser()
 p.add_argument("--seed",type=int,default=11)
 p.add_argument("--steps",type=int,default=600)
 p.add_argument("--batch",type=int,default=16)
 p.add_argument("--head-epochs",type=int,default=100)
 p.add_argument("--head-lr",type=float,default=.0015)
 p.add_argument("--output",default="v1010-posthoc.json")
 args=p.parse_args()
 assert args.seed in (11,19,23)
 start=time.monotonic()
 random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
 training,_=build(args.seed,384,128)
 encoder=FrozenSemanticEncoder();encoder.cache_training_vocab()
 class C:pass
 cfg=C();cfg.width=48;cfg.seed=args.seed;cfg.steps=args.steps
 cfg.batch=32;cfg.lr=.003;cfg.check_every=50
 memory,mi=train_model(cfg,encoder);memory.eval();memory.requires_grad_(False)
 cache=FrozenFeatureCache(encoder,memory)
 eligible=[c for c in training if not c["binding"]]
 tr=[c for c in eligible if c["family"] not in DEV_FAMILIES]
 dv=[c for c in eligible if c["family"] in DEV_FAMILIES]
 user,ui=train_scope_head(args,cache,tr,dv,"user")
 project,pi=train_scope_head(args,cache,tr,dv,"project")
 chooser=SourceChooser(user,project,cache,SourceResolver(MOCK_SIGNERS))
 cold=create(args.seed)
 experiments={}
 for variant in VARIANTS:
  examples=[transform(c,variant) for c in cold]
  outcome=metrics_by(examples,chooser.predict)
  numeric=state_effect(examples,cache)
  experiments[variant]={"summary":outcome["summary"],
                        "numeric_state_intervention":numeric,
                        "per_case":outcome["cases"]}
  summ=outcome["summary"]
  print("V1010 DIAGNOSTIC",json.dumps({"seed":args.seed,"variant":variant,
    "all":summ["all"],"en":summ["language_en"],
    "he":summ["language_he"],"neutral":summ["regime_open"],
    "active":summ["regime_active"],"memory":numeric}),flush=True)
 doc={
  "version":"9.10-D","seed":args.seed,
  "analysis_kind":"POST_HOC_REUSE_OF_ALREADY_SEEN_V9_10_COLD",
  "scientific_acceptance_unchanged":"NO_GO_TO_LM_ADAPTER_INTEGRATION",
  "variants":experiments,"train_memory":mi,
  "train_user":ui,"train_project":pi,
  "time_seconds":round(time.monotonic()-start,2),
  "limitations":[
   "Observational post-hoc ablation after cold answers were inspected; not a blind test.",
   "Variants alter sequence length/order/content at inference only; no model fitted to five-event histories.",
   "Hebrew and English share an English-centric sentence encoder, trained only on synthetic English.",
   "Improved recency ordering cannot prove real long-term memory stability."
  ]
 }
 Path(args.output).write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding="utf8")
 print("V1010 DIAGNOSTIC FINAL",json.dumps({"seed":args.seed,
   "pairs":{k:v["summary"]["all"]["both_correct"] for k,v in experiments.items()},
   "expected_flips":{k:v["summary"]["all"]["expected_flip_agreement"] for k,v in experiments.items()},
   "seconds":doc["time_seconds"]}),flush=True)
if __name__=="__main__":main()
