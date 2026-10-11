"""Extend V8.6 frozen LM feature cache once for V8.7 policy interventions."""
import argparse,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from run import BASE
from fixtures_v86 import build as build_v86
from fixtures_v87 import build
from validate_v87 import audit
from cache_v85 import dataset_fingerprint,selected
from ranker_v84 import key_for,representation

def original_rows():
    base,_,twins=build_v86()
    chosen=selected(base["test"])
    return base["train"]+[r for r in base["test"] if r["project"]["id"] in chosen]+[
        p["counterfactual"] for p in twins]

def all_rows():
    inp,_,train_pairs,_,cold=build()
    chosen=selected(inp["test"])
    return inp["train"]+[r for r in inp["test"] if r["project"]["id"] in chosen]+[
        p["counterfactual"] for p in train_pairs]+[
        item[k] for item in cold for k in ("original","counterfactual")]+[
        item["counterfactual"] for item in build()[3]]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reuse",default="v86-embeddings.pt")
    p.add_argument("--output",default="v87-embeddings.pt")
    args=p.parse_args()
    audit()
    old=torch.load(args.reuse,map_location="cpu",weights_only=True)
    assert old["fingerprint"]==dataset_fingerprint(original_rows())
    rows=all_rows()
    features=dict(old["features"]);frozen=dict(old["frozen_yesno"])
    to_add={}
    for r in rows:
        for choice in r["task"]["choices"]:
            k=key_for(r,choice)
            if k not in features:to_add[k]=(r,choice)
    print(f"CACHE reuse={len(features)} new={len(to_add)}",flush=True)
    assert len(to_add)>100
    tok=AutoTokenizer.from_pretrained(BASE)
    yt,nt=[tok.encode(v,add_special_tokens=False) for v in ("Yes","No")]
    assert len(yt)==len(nt)==1 and yt[0]!=nt[0]
    torch.set_num_threads(2)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    started=time.monotonic();max_tokens={"en":0,"he":0}
    for i,(k,(r,choice)) in enumerate(to_add.items(),1):
        feat,n=representation(model,tok,r,choice,max_tokens=1200)
        features[k]=feat.detach().cpu().contiguous()
        with torch.inference_mode():
            logits=model.lm_head(feat.unsqueeze(0)).float()[0]
        frozen[k]=float(logits[yt[0]]-logits[nt[0]])
        max_tokens[r["lang"]]=max(max_tokens[r["lang"]],n)
        if i%64==0:print(f"CACHE encoded={i}/{len(to_add)} elapsed={time.monotonic()-started:.0f}s",flush=True)
    assert all(key_for(r,c) in features for r in rows for c in r["task"]["choices"])
    out=Path(args.output)
    result={
        "source_fixture":"v87",
        "backbone":BASE,
        "fingerprint":dataset_fingerprint(rows),
        "selected_test_projects":sorted(selected(build()[0]["test"])),
        "features":features,"frozen_yesno":frozen,
        "reused_count":len(old["features"]),"new_count":len(to_add),
        "elapsed_seconds":round(time.monotonic()-started,2),
        "max_new_token_lengths":max_tokens,
    }
    torch.save(result,out)
    out.with_suffix(".json").write_text(
        json.dumps({k:v for k,v in result.items() if k not in ("features","frozen_yesno")},indent=2),
        encoding="utf8")
    print("SAVED",out,flush=True)
if __name__=="__main__":main()
