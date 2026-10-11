"""COMPASS V9.1: early-block PERSONAL neural state, paired contrastive learning.

Frozen SmolLM2-360M. Train only a shared 30,784-weight state-conditioned
residual in decoder block -3, before 2 subsequent attention blocks.
The same tokens are fed for rapid/thorough; the state is a tensor, NOT text.
Complete cold metrics generated only AFTER dev-selected checkpoint is frozen.

Research-only, invented labels, not independent human adjudication.
"""
import argparse,copy,hashlib,json,random,time
from pathlib import Path
from collections import defaultdict
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from safetensors.torch import save_file
from fixtures_v91 import build,audit
from gate_v91 import PersonalConditioner,state_for,attach,count
from train_v90 import prompt,get_yes_no,lexical_policy
from run import BASE

def ids_for(tok,txt):
    ids=tok.apply_chat_template(
      [{"role":"user","content":txt}],tokenize=True,add_generation_prompt=True)
    if len(ids)>1150:raise ValueError("Input exceeds 1150 tokens")
    return torch.tensor([ids],dtype=torch.long)

class Inference:
    def __init__(self,model,tok,yes,no):
        self.model=model;self.tok=tok;self.yes=yes;self.no=no
        self.cache={};self.model_calls=0
    def tokens(self,row,candidate,include_policy=False):
        text=prompt(row,candidate,inject_owner=include_policy)
        key=hashlib.sha256(text.encode("utf-8")).hexdigest()
        if key not in self.cache:self.cache[key]=ids_for(self.tok,text)
        return self.cache[key]
    def two(self,row,gate=None,override_state=None,include_policy=False):
        mode=override_state or row["project"]["policy"]["mode"]
        handle=None
        if gate is not None:
            handle=attach(self.model,gate,state_for(mode),layer_from_end=3)
        try:
            vals=[]
            for action in row["task"]["choices"]:
                t=self.tokens(row,action,include_policy=include_policy)
                raw=self.model(input_ids=t).logits[0,-1,:]
                vals.append(raw[self.yes]-raw[self.no])
                self.model_calls+=1
            return torch.stack(vals)
        finally:
            if handle is not None:handle.remove()
    def choice(self,row,gate=None,override_state=None,include_policy=False):
        with torch.no_grad():
            vals=self.two(row,gate,override_state,include_policy)
            scores=vals.float().cpu().tolist()
        return {"choice":int(scores[1]>scores[0]),
                "scores":[round(float(x),6) for x in scores],
                "margin":round(float(scores[0]-scores[1]),6)}

def pair_loss(model,engine,gate,p):
    # IDENTICAL input text in original/counterfactual; only personal tensor swaps.
    a=p["original"];b=p["counterfactual"]
    va=engine.two(a,gate);vb=engine.two(b,gate)
    m0=va[0]-va[1];m1=vb[0]-vb[1]
    ta=p["original_label"]["correct_digit"]
    tb=p["counterfactual_label"]["correct_digit"]
    sa=1 if ta==0 else -1
    sb=1 if tb==0 else -1
    supervised=(F.softplus(.7-sa*m0)+F.softplus(.7-sb*m1))/2
    if p["meta"]["expected_flip"]:
        assert sa==-sb
        # Explicitly teach the same evidence/options, opposite owner state
        # should cause directionally opposite relative-action margins.
        structural=.35*F.softplus(1.0-sa*(m0-m1))
    else:
        assert sa==sb
        # Prefer identical action choice if binding evidence overrides policy.
        structural=.05*(m0-m1).square()
    return supervised+structural

def run_training(model,engine,gate,train,dev,epochs,seed,lr=0.002):
    optimizer=torch.optim.AdamW(gate.parameters(),lr=lr,weight_decay=0.001)
    best=-1;best_params=None;best_epoch=0;history=[]
    normal=[x for x in train if x["meta"]["expected_flip"]]
    binding=[x for x in train if not x["meta"]["expected_flip"]]
    assert len(normal)>=24 and len(binding)>=12
    for epoch in range(1,epochs+1):
        rng=random.Random(seed*1001+epoch)
        chosen=rng.sample(normal,24)+rng.sample(binding,12)
        rng.shuffle(chosen)
        losses=[]
        gate.train()
        for idx,p in enumerate(chosen,1):
            optimizer.zero_grad(set_to_none=True)
            loss=pair_loss(model,engine,gate,p)
            if not torch.isfinite(loss):raise RuntimeError("Nonfinite paired loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gate.parameters(),1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            if idx%12==0:
                print(f"V91 TRAIN epoch={epoch} pairs={idx}/36 mean_loss={sum(losses)/len(losses):.4f}",flush=True)
        gate.eval()
        with torch.no_grad():
            valid=[]
            for p in dev:
                x=engine.choice(p["original"],gate)["choice"]
                y=engine.choice(p["counterfactual"],gate)["choice"]
                valid.append(int(x==p["original_label"]["correct_digit"] and
                                 y==p["counterfactual_label"]["correct_digit"]))
        score=sum(valid)
        if score>best:
            best=score;best_epoch=epoch
            best_params={k:v.detach().clone() for k,v in gate.state_dict().items()}
        history.append({"epoch":epoch,"mean_loss":round(sum(losses)/len(losses),5),
                        "dev_pair_both":score,"dev_pair_total":len(dev)})
        print(f"V91 DEV epoch={epoch} pair_both={score}/{len(dev)} best={best}",flush=True)
    gate.load_state_dict(best_params)
    gate.eval()
    return {"selected_epoch":best_epoch,"dev_both_correct":best,
            "dev_total":len(dev),"history":history,
            "train_pairs_per_epoch":36,"epochs":epochs}

def evaluate(cold,fn):
    totals=defaultdict(lambda:{"total":0,"both_correct":0,"actual_flips":0,
                              "expected_flip_correct":0,"correct_choices":0})
    preds=[]
    for p in cold:
        a=fn(p["original"]);b=fn(p["counterfactual"])
        expect=[p["original_label"]["correct_digit"],p["counterfactual_label"]["correct_digit"]]
        decisions=[a,b]
        detail={"family":p["meta"]["family"],"project":p["meta"]["split"]+
                ":"+p["meta"]["family"] if False else p["original"]["project"]["id"],
                "swap":p["meta"]["swap"],"status":p["meta"]["status"],
                "gold":expect,"chosen":decisions,"both_correct":decisions==expect}
        preds.append(detail)
        for key in (p["meta"]["status"],"all"):
            s=totals[key];s["total"]+=1
            s["both_correct"]+=int(decisions==expect)
            s["actual_flips"]+=int(a!=b)
            s["expected_flip_correct"]+=int((a!=b)==p["meta"]["expected_flip"])
            s["correct_choices"]+=sum(x==y for x,y in zip(decisions,expect))
    # Order-swap control: identical family/project/evidence except option order.
    cache={(d["project"],d["family"],d["status"],d["swap"]):d["chosen"] for d in preds}
    good=0;total=0
    for (pid,fam,status,swap),v in cache.items():
        if swap!=0:continue
        alt=cache.get((pid,fam,status,1))
        if alt is None:continue
        for a,b in zip(v,alt):
            good+=int(a!=b);total+=1
    return {"breakdown":dict(totals),"order_stability":{"correct":good,"total":total}},preds

def active_probe(engine,gate,row):
    """Checks intermediate tensor changes + final output, same owner-blind tokens."""
    action=row["task"]["choices"][0]
    t=engine.tokens(row,action)
    captures=[]
    def observer(_mod,_inputs,out):
        h=out[0] if isinstance(out,tuple) else out
        captures.append(h[0,-1,:].detach().clone())
    watch=engine.model.model.layers[-2].register_forward_hook(observer)
    try:
        with torch.no_grad():
            basic=engine.model(input_ids=t).logits[0,-1,:]
            h1=attach(engine.model,gate,state_for("rapid"),layer_from_end=3)
            try:rapid=engine.model(input_ids=t).logits[0,-1,:]
            finally:h1.remove()
            h2=attach(engine.model,gate,state_for("thorough"),layer_from_end=3)
            try:thorough=engine.model(input_ids=t).logits[0,-1,:]
            finally:h2.remove()
    finally:watch.remove()
    assert len(captures)==3
    def margin(logits):
        return float((logits[engine.yes]-logits[engine.no]).detach())
    delta_hidden=float(torch.max(torch.abs(captures[1]-captures[2])))
    delta_logit=abs(margin(rapid)-margin(thorough))
    return {"frozen_margin":round(margin(basic),6),
            "rapid_margin":round(margin(rapid),6),
            "thorough_margin":round(margin(thorough),6),
            "max_downstream_layer_hidden_difference":round(delta_hidden,9),
            "output_yesno_margin_difference":round(delta_logit,9),
            "same_neural_prompt_under_both_users":True,
            "live_hook_in_transformer":True,
            "downstream_attention_executed":True}

def run(args):
    fixture=audit();t0=time.monotonic()
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    train,dev,cold=build()
    tok=AutoTokenizer.from_pretrained(BASE)
    yes,no,tokpair=get_yes_no(tok)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    # Check representative frozen backbone parameters before and after.
    snapshot={
      "q_proj":model.model.layers[-2].self_attn.q_proj.weight.detach().clone(),
      "rms_norm":model.model.norm.weight.detach().clone(),
    }
    gate=PersonalConditioner(model.config.hidden_size,args.rank,4)
    params=count(gate)
    print(f"V91 initialized shared_gate={params} frozen_base={sum(p.numel() for p in model.parameters())}; rank={args.rank}",flush=True)
    e=Inference(model,tok,yes,no)
    progress=run_training(model,e,gate,train,dev,args.epochs,args.seed)
    print("V91 trained. Beginning untouched cold test.",flush=True)
    measurements={}
    examples={}
    methods=[
     ("neural_internal",lambda r:e.choice(r,gate)["choice"]),
     ("frozen_owner_blind",lambda r:e.choice(r)["choice"]),
     ("frozen_text_preference",lambda r:e.choice(r,include_policy=True)["choice"]),
     ("explicit_lexical_rule",lambda r:lexical_policy(r)),
    ]
    for name,fn in methods:
        stats,records=evaluate(cold,fn)
        measurements[name]=stats;examples[name]=records
        print("V91 COLD",name,stats,flush=True)
    probe=active_probe(e,gate,cold[0]["original"])
    unchanged=(torch.equal(snapshot["q_proj"],model.model.layers[-2].self_attn.q_proj.weight)
               and torch.equal(snapshot["rms_norm"],model.model.norm.weight))
    assert unchanged,"Frozen base weights changed!"
    assert probe["max_downstream_layer_hidden_difference"]>0, "Owner state did not alter downstream activations"
    assert probe["output_yesno_margin_difference"]>0, "Owner state did not alter logits"
    # Extra test that the explicit policy words are genuinely missing.
    coldrow=cold[0]["original"]
    no_text=prompt(coldrow,coldrow["task"]["choices"][0],False)
    assert coldrow["project"]["policy"]["en"] not in no_text
    selection=measurements["neural_internal"]["breakdown"]
    txt=measurements["frozen_text_preference"]["breakdown"]
    rule=measurements["explicit_lexical_rule"]["breakdown"]
    target=24
    gates={
      "real_in_model_activation_delta":probe["max_downstream_layer_hidden_difference"]>0,
      "real_output_logit_delta":probe["output_yesno_margin_difference"]>0,
      "frozen_backbone_unchanged":unchanged,
      "new_neutral_24_of_32":selection["neutral"]["both_correct"]>=target,
      "new_binding_24_of_32":selection["binding"]["both_correct"]>=target,
      "beats_text_baseline":selection["all"]["both_correct"]>txt["all"]["both_correct"],
      "beats_task_specific_lexical_rule":selection["all"]["both_correct"]>rule["all"]["both_correct"],
      "neural_choice_order_stability_56_of_64":
         measurements["neural_internal"]["order_stability"]["correct"]>=56,
    }
    outcome=("PROVISIONAL_PASS_PENDING_INDEPENDENT_HUMAN_HOLDOUT" if all(gates.values())
             else "NO_GO_TO_USER_MODULE_SCALE_OR_DEPLOYMENT")
    p=Path(args.output)
    save_file({k:v.detach().cpu().contiguous() for k,v in gate.state_dict().items()},
              str(p.with_suffix(".safetensors")))
    result={
      "version":"9.1","description":"Early-block frozen Transformer state-conditioned gate; paired opposite-preference contrastive training",
      "run_status":"COMPLETED","scientific_verdict":outcome,
      "model":BASE,"personal_state_dim":4,
      "state_information_degree":"1 binary preference; 2 of 4 channels active, 2 reserved",
      "learned_shared_adapter_parameters":params,
      "per_user_optim_steps":0,"frozen_base_parameters":sum(p.numel() for p in model.parameters()),
      "gate_layer_from_end":3,"downstream_decoder_blocks":2,
      "fixture":fixture,"training":progress,"cold":measurements,
      "cold_predictions":examples,"live_probe":probe,
      "frozen_backbone_spot_check_unchanged":unchanged,
      "neural_prompt_has_no_owner_policy_text":True,"tokenizer_yesno":tokpair,
      "registered_gates":gates,"model_calls":e.model_calls,
      "elapsed_sec":round(time.monotonic()-t0,2),
      "caveats":[
        "Synthetic, research-authored gold; unseen project IDs/family texts are still not external human validation.",
        "This learns one shared adapter; each user's four-dimensional state currently encodes only ONE binary distinction, not an independently trained individual model.",
        "A live hook at -3 changes last-token representation so 2 subsequent attention blocks can process it; no recurrent durable user-model service yet.",
        "Frozen backbone equality tests check representative tensors, not a cryptographic fingerprint of every parameter.",
        "No real user auth, source verification, personalization memory, production deployment, or native SELF 5M integration."
      ],
    }
    p.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print("V91 FINAL",json.dumps({"verdict":outcome,
      "cold":{k:{"neutral":v["breakdown"]["neutral"]["both_correct"],
                   "binding":v["breakdown"]["binding"]["both_correct"],
                   "order":v["order_stability"]} for k,v in measurements.items()},
      "probe":probe,"gates":gates,"seconds":result["elapsed_sec"]}),flush=True)
if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--rank",type=int,default=16)
    cli.add_argument("--epochs",type=int,default=8)
    cli.add_argument("--seed",type=int,default=11)
    cli.add_argument("--output",default="compass-v91-early-personal.json")
    args=cli.parse_args()
    assert args.rank>0 and args.epochs>0
    run(args)
