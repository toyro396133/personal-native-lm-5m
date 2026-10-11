"""COMPASS V9.0: real owner-conditioned decoder-block residual, frozen 360M.

Trains only a shared low-rank gate; the CURRENT per-user owner state can switch
without a gradient, without changing language model prompt, and without
updating backbone weights. Uses cached final-block hidden vectors for fast
training but verifies the same gate in the actual live Transformer forward hook.
Not a full independently-trained user-specific neural network yet.
"""
import argparse,copy,hashlib,json,random,time
from collections import defaultdict
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM,AutoTokenizer
from fixtures_v90 import datasets,audit
from personal_gate_v90 import PersonalNeuralGate,hook_last_block,personal_state,params
from run import BASE

def prompt(row,candidate,inject_owner=False):
    p=row["project"];t=row["task"];choices=t["choices"]
    o=("Approved OWNER preference: "+p["policy"]["en"]+"\n") if inject_owner else ""
    return (
      "Choose the better project action given available evidence and binding requirements. "
      "Candidate suitability can depend on the owner's current priority, but binding "
      "evidence always overrides personal preference. "
      "Answer Yes or No for this candidate.\n"
      "Project objective: "+p["objective"]["en"]+"\n"+
      o+"Evidence: "+t["record"]+"\n"+
      "Option A: "+choices[0]+"\nOption B: "+choices[1]+"\n"+
      "Candidate: "+candidate+"\nIs the candidate the better authorized action? Answer:"
    )

def tokenize(tok,text):
    ids=tok.apply_chat_template(
        [{"role":"user","content":text}],tokenize=True,add_generation_prompt=True)
    if len(ids)>1100:raise ValueError("Prompt too long")
    return torch.tensor([ids],dtype=torch.long)

def get_yes_no(tok):
    for yes,no in (("Yes","No"),(" Yes"," No")):
        a=tok.encode(yes,add_special_tokens=False)
        b=tok.encode(no,add_special_tokens=False)
        if len(a)==len(b)==1 and a[0]!=b[0]:return a[0],b[0],(yes,no)
    raise ValueError("Requires different one-token Yes and No; score not valid.")

class FrozenBlockFeatures:
    """Capture decoder last-block pre-final-norm hidden values, owner-blind."""
    def __init__(self,model,tokenizer):
        self.model=model;self.tok=tokenizer;self.cached={};self.tok_count=0
    def get(self,text):
        hashkey=hashlib.sha256(text.encode("utf8")).hexdigest()
        if hashkey in self.cached:return self.cached[hashkey]
        captures=[]
        def hook(_module,_inputs,output):
            raw=output[0] if isinstance(output,tuple) else output
            captures.append(raw[0,-1,:].float().detach().clone())
        h=self.model.model.layers[-1].register_forward_hook(hook)
        try:
            ids=tokenize(self.tok,text)
            with torch.no_grad():
                self.model(input_ids=ids)
        finally:h.remove()
        if len(captures)!=1 or captures[0].shape[-1]!=self.model.config.hidden_size:
            raise RuntimeError("Incorrect last-layer hidden capture")
        self.tok_count+=ids.shape[1]
        v=captures[0].contiguous()
        self.cached[hashkey]=v
        if len(self.cached)%40==0:
            print(f"CACHE unique prompts={len(self.cached)} total_input_tokens={self.tok_count}",flush=True)
        return v
    def pair(self,row,inject_owner=False):
        return torch.stack([self.get(prompt(row,c,inject_owner)) for c in row["task"]["choices"]])

def action_margin(model,feats,yes,no,gate=None,owner=None):
    if gate is not None:
        feats=gate(feats,personal_state(owner))
    end=model.model.norm(feats)
    w=model.lm_head.weight
    # Equivalence to projected model logits is verified against full forward.
    return F.linear(end,w[[yes]])[:,0]-F.linear(end,w[[no]])[:,0]

def predictions(model,feats,yes,no,gate=None,mode=None):
    with torch.no_grad():
        scores=action_margin(model,feats,yes,no,gate,mode)
        return {"choice":int(scores[1]>scores[0]),
                "scores":[round(float(v),5) for v in scores.detach()],
                "margin":round(float(scores[1]-scores[0]),5)}

def lexical_policy(row):
    """Transparent rule baseline, written before test metrics are inspected."""
    evid=row["task"]["record"].lower()
    choices=[c.lower() for c in row["task"]["choices"]]
    deep=["comprehensive","full ","detailed","complete ","end-to-end",
          "historical","all major","multi-team","entire ","audit the full"]
    votes=[sum(word in text for word in deep) for text in choices]
    if votes[0]==votes[1]:return None
    full=int(votes[1]>votes[0])
    forced=any(term in evid for term in ("mandate","binding","requires","demands","required"))
    if forced:return full
    return full if row["project"]["policy"]["mode"]=="thorough" else 1-full

def summarize(pairs,selector):
    result=defaultdict(lambda:{"total":0,"both_correct":0,"flipped":0,
                               "flip_correct":0,"correct_choices":0,"abstained":0})
    examples=[]
    for p in pairs:
        states=[]
        for label in ("original","counterfactual"):
            val=selector(p[label])
            states.append(val)
        expect=[p["original_label"]["correct_digit"],p["counterfactual_label"]["correct_digit"]]
        record={"family":p["meta"]["family"],"status":p["meta"]["status"],
                "swap":p["meta"]["swap"],"truth":expect,"predicted":states}
        examples.append(record)
        for group in ("all",p["meta"]["status"]):
            stat=result[group];stat["total"]+=1
            if None in states:
                stat["abstained"]+=1
                continue
            stat["both_correct"]+=int(states==expect)
            stat["flipped"]+=int(states[0]!=states[1])
            stat["flip_correct"]+=int((states[0]!=states[1])==p["meta"]["expected_flip"])
            stat["correct_choices"]+=sum(a==b for a,b in zip(states,expect))
    return {k:dict(v) for k,v in result.items()},examples

def prepare(features,pairs,force_text=False):
    material={}
    for p in pairs:
        for side in ("original","counterfactual"):
            row=p[side]
            key=(row["task"]["id"],force_text)
            if key not in material:material[key]=features.pair(row,force_text)
    return material

def build_cases(pairs,mandatory,cached):
    data=[]
    for p in pairs:
        for side in ("original","counterfactual"):
            row=p[side]
            data.append((cached[(row["task"]["id"],False)],
                         row["project"]["policy"]["mode"],
                         p[side+"_label"]["correct_digit"]))
    for row,gold in mandatory:
        data.append((cached[(row["task"]["id"],False)],
                     row["project"]["policy"]["mode"],gold))
    return data

def train_gate(gate,model,yes,no,train,dev,epochs,seed):
    optim=torch.optim.AdamW(gate.parameters(),lr=0.003,weight_decay=0.002)
    best_metric=-1;best_state=None;best_epoch=0;logs=[]
    for epoch in range(1,epochs+1):
        random.Random(seed+epoch).shuffle(train)
        gate.train();loss_vals=[]
        for offset in range(0,len(train),16):
            batch=train[offset:offset+16]
            losses=[]
            for feats,mode,label in batch:
                pred=action_margin(model,feats,yes,no,gate,mode)
                gap=pred[0]-pred[1]
                sign=1.0 if label==0 else -1.0
                losses.append(F.softplus(0.6-sign*gap))
            loss=torch.stack(losses).mean()
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),2.0)
            optim.step()
            loss_vals.append(float(loss.detach()))
        gate.eval()
        with torch.no_grad():
            dev_correct=sum(
                predictions(model,v,yes,no,gate,m)["choice"]==label
                for v,m,label in dev)
        if dev_correct>best_metric:
            best_metric=dev_correct;best_epoch=epoch
            best_state={k:v.detach().clone() for k,v in gate.state_dict().items()}
        if epoch in (1,10,25,epochs):
            print(f"TRAIN epoch={epoch}/{epochs} dev_correct={dev_correct}/{len(dev)} "+
                  f"loss={sum(loss_vals)/len(loss_vals):.5f}",flush=True)
        logs.append({"epoch":epoch,"dev":dev_correct,"loss":round(sum(loss_vals)/len(loss_vals),5)})
    gate.load_state_dict(best_state);gate.eval()
    return {"best_epoch":best_epoch,"best_dev_correct":best_metric,
            "total_dev":len(dev),"history":logs}

def live_hook_test(model,tok,gate,row,yes,no):
    text=prompt(row,row["task"]["choices"][0])
    ids=tokenize(tok,text)
    with torch.no_grad():
        bare=model(input_ids=ids).logits[0,-1,[yes,no]].float()
    # Hooks are removable handles, not context managers; use explicit finally.
    handle=hook_last_block(model,gate,personal_state("rapid"))
    try:
        with torch.no_grad():
            rapid=model(input_ids=ids).logits[0,-1,[yes,no]].float()
    finally:handle.remove()
    handle=hook_last_block(model,gate,personal_state("thorough"))
    try:
        with torch.no_grad():
            thorough=model(input_ids=ids).logits[0,-1,[yes,no]].float()
    finally:handle.remove()
    captured=FrozenBlockFeatures(model,tok).get(text)
    with torch.no_grad():
        base=action_margin(model,captured.unsqueeze(0),yes,no)[0]
        left=action_margin(model,captured.unsqueeze(0),yes,no,gate,"rapid")[0]
        right=action_margin(model,captured.unsqueeze(0),yes,no,gate,"thorough")[0]
    # Exact same final layer RMSNorm + LM head when using captured states and live hook.
    assert abs(float((bare[0]-bare[1])-base))<1e-3,"cached/bare forward mismatch"
    assert abs(float((rapid[0]-rapid[1])-left))<1e-3,"live-hook/rapid mismatch"
    assert abs(float((thorough[0]-thorough[1])-right))<1e-3,"live-hook/thorough mismatch"
    return {"base_margin":round(float(base),5),
            "rapid_margin":round(float(left),5),
            "thorough_margin":round(float(right),5),
            "rapid_vs_thorough":round(float(abs(left-right)),7),
            "hook_matches_cached":True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="compass-v90-personal-neural.json")
    ap.add_argument("--rank",type=int,default=16)
    ap.add_argument("--epochs",type=int,default=60)
    ap.add_argument("--seed",type=int,default=11)
    args=ap.parse_args()
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    start=time.monotonic()
    fixture=audit()
    trainpairs,devpairs,mand,manddev,cold=datasets()
    tokenizer=AutoTokenizer.from_pretrained(BASE)
    yes,no,tokens=get_yes_no(tokenizer)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    frozen_checksum=[model.model.layers[-1].self_attn.q_proj.weight.detach().clone()]
    gate=PersonalNeuralGate(model.config.hidden_size,args.rank)
    print(f"PERSONAL RESIDUAL learned_shared={params(gate)} frozen_base={sum(p.numel() for p in model.parameters())}",flush=True)
    features=FrozenBlockFeatures(model,tokenizer)
    train_cache=prepare(features,trainpairs+devpairs)
    for row,_ in mand+manddev:
        train_cache[(row["task"]["id"],False)]=features.pair(row)
    train=build_cases(trainpairs,mand,train_cache)
    dev=build_cases(devpairs,manddev,train_cache)
    info=train_gate(gate,model,yes,no,train,dev,args.epochs,args.seed)
    # Gold remains held out from optimizer; only generate NEW cold features after training.
    test_cache=prepare(features,cold)
    lookup=lambda row:text_selector_key(row,False)
    def choice_for(row,variant):
        if variant=="symbolic":return lexical_policy(row)
        feat=test_cache[lookup(row)]
        if variant=="adapter":return predictions(model,feat,yes,no,gate,row["project"]["policy"]["mode"])["choice"]
        return predictions(model,feat,yes,no)["choice"]
    baseline,baseline_details=summarize(cold,lambda r:choice_for(r,"frozen"))
    tuned,tuned_details=summarize(cold,lambda r:choice_for(r,"adapter"))
    lex,lex_details=summarize(cold,lambda r:choice_for(r,"symbolic"))
    # Strong prompt-injection comparison, same frozen LM without a learned personal module.
    text_cache=prepare(features,cold,force_text=True)
    text,txt_details=summarize(cold,lambda r:
        predictions(model,text_cache[text_selector_key(r,True)],yes,no)["choice"])
    # Module active in real forward pass and owner state gets swapped without retraining.
    probe=live_hook_test(model,tokenizer,gate,cold[0]["original"],yes,no)
    assert torch.equal(frozen_checksum[0],model.model.layers[-1].self_attn.q_proj.weight),"Backbone weights changed"
    # Dynamic real preference loading requires no per-user training in this pilot:
    sample=cold[0]["original"];key=text_selector_key(sample,False)
    rapid=predictions(model,test_cache[key],yes,no,gate,"rapid")
    thorough=predictions(model,test_cache[key],yes,no,gate,"thorough")
    extra={"same_prompt":prompt(sample,sample["task"]["choices"][0]),
           "rapid":rapid,"thorough":thorough,
           "personal_text_absent":sample["project"]["policy"]["en"] not in prompt(sample,sample["task"]["choices"][0])}
    assert extra["personal_text_absent"]
    state={k:v.detach().cpu().contiguous() for k,v in gate.state_dict().items()}
    from safetensors.torch import save_file
    chk=Path(args.out).with_suffix(".safetensors")
    save_file(state,str(chk))
    out={
      "name":"COMPASS V9.0 direct neural personal state-conditioned frozen Transformer",
      "frozen_backbone":BASE,"backbone_parameters":sum(p.numel() for p in model.parameters()),
      "shared_adaptor_parameters":params(gate),
      "per_user_personal_state_dim":1,
      "per_user_gradient_steps":0,"rank":args.rank,
      "tokenizer_yes_no":tokens,
      "fixture":fixture,"development":info,
      "cold_test_results":{"frozen_no_personal_state":baseline,
                           "neural_in_block":tuned,
                           "frozen_text_preference_baseline":text,
                           "transparent_lexical_rule":lex},
      "cold_test_predictions":{"frozen":baseline_details,"neural":tuned_details,
                               "text":txt_details,"rule":lex_details},
      "live_hook":probe,"state_swap_probe":extra,
      "backbone_unchanged":True,
      "new_cold_neutral_gate":tuned["neutral"]["both_correct"]>=12,
      "new_cold_binding_gate":tuned["binding"]["both_correct"]>=12,
      "better_than_text_on_cold":tuned["all"]["both_correct"]>text["all"]["both_correct"],
      "runtime_seconds":round(time.monotonic()-start,2),
      "caveats":[
        "Only one layer's final-token block output is modified: this is limited in-block conditioning, not full multi-layer recurrent personal thinking.",
        "The shared adapter learns once on research-authored profiles; only a 1D owner setting is switched per user. No separately trained person-specific neural weights yet.",
        "Synthetic labels and lexical baseline are written by researcher, not independently reviewed. Holdout project identity used in previously inspected V8.9.",
        "Text owner preference baseline can have more explicit semantics than scalar owner state; improve data and fairness before concluding.",
        "Not production safe, not real source authentication, no native 5M model changes."
      ]
    }
    p=Path(args.out);p.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding="utf8")
    print("V90 RESULTS",json.dumps({"cold":out["cold_test_results"],"live":probe,
                                  "backbone_unchanged":True,"elapsed_seconds":out["runtime_seconds"]}),flush=True)
    print("SAVED",p,chk,flush=True)
def text_selector_key(row,flag):
    return (row["task"]["id"],flag)
if __name__=="__main__":main()
