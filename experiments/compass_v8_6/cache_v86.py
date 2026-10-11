"""Extend the previously computed V8.5 frozen embeddings with only V8.6 cases.

The shared 360M backbone is loaded ONCE. All three personalization sizes
reuse the same frozen cache, with a versioned dataset fingerprint.
"""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from run import BASE
from fixtures_v85 import build as build_v85
from fixtures_v86 import build
from validate_v86 import audit
from cache_v85 import dataset_fingerprint,selected
from ranker_v84 import key_for,representation

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reuse",default="v85-embeddings.pt")
    p.add_argument("--output",default="v86-embeddings.pt")
    args=p.parse_args()
    audit()
    old=torch.load(args.reuse,map_location="cpu",weights_only=True)
    old_inputs,_=build_v85()
    old_selection=selected(old_inputs["test"])
    old_rows=old_inputs["train"]+[r for r in old_inputs["test"] if r["project"]["id"] in old_selection]
    assert old["fingerprint"]==dataset_fingerprint(old_rows),"Historical V8.5 frozen embeddings mismatch"
    assert old["selected_test_projects"]==sorted(old_selection)
    inputs,golds,twins=build()
    test_selection=selected(inputs["test"])
    assert test_selection==old_selection
    train=inputs["train"]
    test=[r for r in inputs["test"] if r["project"]["id"] in test_selection]
    counterfactuals=[p["counterfactual"] for p in twins]
    rows=train+test+counterfactuals
    cache=dict(old["features"])
    baseline=dict(old["frozen_yesno"])
    prior_count=len(cache)
    pairs=[]
    for row in rows:
        for candidate in row["task"]["choices"]:
            k=key_for(row,candidate)
            if k not in cache and k not in {x[0] for x in pairs}:
                pairs.append((k,row,candidate))
    assert len(pairs)>100
    torch.set_num_threads(2)
    started=time.monotonic()
    tok=AutoTokenizer.from_pretrained(BASE)
    yes,no=(tok.encode(v,add_special_tokens=False) for v in ("Yes","No"))
    assert len(yes)==len(no)==1 and yes[0]!=no[0]
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    maxima={"en":0,"he":0}
    for index,(k,row,candidate) in enumerate(pairs,1):
        emb,tokens=representation(model,tok,row,candidate,max_tokens=1200)
        cache[k]=emb.detach().cpu().contiguous()
        with torch.inference_mode():
            logits=model.lm_head(emb.unsqueeze(0)).float()[0]
        baseline[k]=float(logits[yes[0]]-logits[no[0]])
        maxima[row["lang"]]=max(maxima[row["lang"]],tokens)
        if index%64==0:
            print(f"EXTRACT V8.6 additions {index}/{len(pairs)} elapsed={time.monotonic()-started:.1f}s",flush=True)
    for row in rows:
        for action in row["task"]["choices"]:
            assert key_for(row,action) in cache
    result={
        "source_fixture":"v86",
        "fingerprint":dataset_fingerprint(rows),
        "backbone":BASE,
        "selected_test_projects":sorted(test_selection),
        "features":cache,
        "frozen_yesno":baseline,
        "feature_count":len(cache),
        "reused_feature_count":prior_count,
        "new_feature_count":len(pairs),
        "extra_policy_only":len(counterfactuals),
        "split_counts":{"train_pool":len(train),"test":len(test),"policy_twins":len(counterfactuals)},
        "token_maxima_new":maxima,
    }
    out=Path(args.output)
    torch.save(result,out)
    meta={k:v for k,v in result.items() if k not in ("features","frozen_yesno")}
    meta["elapsed_seconds"]=round(time.monotonic()-started,2)
    out.with_suffix(".json").write_text(json.dumps(meta,indent=2),encoding="utf8")
    print("SAVED V8.6 feature cache",meta,flush=True)
if __name__=="__main__":
    main()
