"""Compute expensive frozen candidate embeddings ONCE for all V8.5 sizes/seeds.

No labels/gold are used here. Only synthetic prompts and frozen 360M base.
The resulting cache is shared unchanged by all 3x3 training jobs.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from run import BASE
from fixtures_v85 import build
from validate_v85 import audit
from ranker_v84 import factual_candidate_prompt,representation,key_for

def selected(rows):
    ids=sorted({r["project"]["id"] for r in rows})
    return {ids[0],ids[3]}

def dataset_fingerprint(rows):
    h=hashlib.sha256()
    for row in rows:
        h.update(row["task"]["id"].encode())
        for a in row["task"]["choices"]:
            h.update(factual_candidate_prompt(row,a).encode("utf8"))
    return h.hexdigest()

def prepare(output):
    audit()
    inputs,golds=build()
    sel=selected(inputs["test"])
    groups={
        "train_and_development":inputs["train"],
        "locked_test":[r for r in inputs["test"] if r["project"]["id"] in sel],
    }
    allrows=groups["train_and_development"]+groups["locked_test"]
    tok=AutoTokenizer.from_pretrained(BASE)
    ids=[tok.encode(t,add_special_tokens=False) for t in ("Yes","No")]
    if not all(len(x)==1 for x in ids) or ids[0]==ids[1]:
        raise RuntimeError(f"Expected distinct single token Yes/No: {ids}")
    yes,no=ids[0][0],ids[1][0]
    torch.set_num_threads(2)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    frozen_features={}
    frozen_yesno={}
    maxlens={"en":0,"he":0}
    started=time.monotonic()
    for group,rows in groups.items():
        for row in rows:
            for candidate in row["task"]["choices"]:
                key=key_for(row,candidate)
                if key in frozen_features:continue
                hidden,tokens=representation(model,tok,row,candidate,max_tokens=1200)
                frozen_features[key]=hidden.detach().cpu().contiguous()
                maxlens[row["lang"]]=max(maxlens[row["lang"]],tokens)
                with torch.inference_mode():
                    logits=model.lm_head(hidden.unsqueeze(0)).float()[0]
                frozen_yesno[key]=float(logits[yes]-logits[no])
                if len(frozen_features)%96==0:
                    print(f"CACHE {len(frozen_features)} option embeddings from {group}, elapsed={time.monotonic()-started:.0f}s",flush=True)
    assert len(frozen_features)>500
    data={
        "fingerprint":dataset_fingerprint(allrows),
        "source_fixture":"v85",
        "features":frozen_features,"frozen_yesno":frozen_yesno,
        "selected_test_projects":sorted(sel),
        "split_sizes":{name:len(rows) for name,rows in groups.items()},
        "token_maxima":maxlens,
        "token_ids":{"Yes":yes,"No":no},
        "feature_count":len(frozen_features),
        "backbone":BASE,
    }
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(data,path)
    metadata={k:v for k,v in data.items() if k not in ("features","frozen_yesno")}
    metadata["elapsed_seconds"]=round(time.monotonic()-started,2)
    path.with_suffix(".json").write_text(json.dumps(metadata,indent=2),encoding="utf8")
    print("SAVED",path,metadata,flush=True)
    return metadata

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",default="v85-embeddings.pt")
    args=parser.parse_args();prepare(args.output)
