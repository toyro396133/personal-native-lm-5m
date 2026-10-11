"""COMPASS V8.3: paired-margin preference pilot with joint fact/option-order training.

The frozen SmolLM2-360M backbone and V8.2 bilingual synthetic fixtures are
unchanged. For each training update a quartet contains both truth states and
both action orderings of the SAME project decision. A rank-4 residual COMPASS
adapter learns to rank option 0 vs 1 using a differentiable, signed pairwise
margin, with penalties for broken reversal consistency.

This is a synthetic feasibility test, NOT project management intelligence.
"""
import argparse
import json
import math
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import BASE, PersonalAdapter, attach
from fixtures_v82 import generate, LANGUAGES, KINDS
from validate_v82 import audit
from pilot_v82 import prompt, chat_ids, code_ids, logits_for, breakdown
from guard_v82 import enforce

def make_groups(rows, labels):
    """Group by project, language and dilemma kind (no test gold in training)."""
    groups = defaultdict(dict)
    for row in rows:
        gold=labels[row["task"]["id"]]
        key=(row["project"]["id"],row["lang"],gold["kind"])
        point=(gold["state"],gold["swap"])
        if point in groups[key]:
            raise AssertionError(f"Repeated state/order case: {key}/{point}")
        groups[key][point]=(row,gold)
    if not groups or any(set(g)!={(0,0),(0,1),(1,0),(1,1)} for g in groups.values()):
        raise AssertionError("One or more contrasting cases are missing")
    return groups

def group_plan(groups,seed,n_steps):
    """Balance six dilemma families and both languages across each 12-step cycle."""
    rng=random.Random(seed)
    keys=list(groups)
    families=sorted({(lang,kind) for _,lang,kind in keys})
    assert len(families)==len(LANGUAGES)*len(KINDS)==12
    for step in range(n_steps):
        # Cycle through 12 language×task groups; vary the project across cycles.
        family=families[step%len(families)]
        eligible=[k for k in keys if k[1:]==family]
        eligible.sort()
        rng.shuffle(eligible)
        chosen=eligible[(step//len(families))%len(eligible)]
        yield chosen,groups[chosen]

def margin_loss(quartet,margin_target=0.75,order_weight=0.2,fact_weight=0.2):
    """Calculate differentiable objective from FOUR 0-minus-1 logit margins.

    Desired label on fixture is state XOR option-swap. The observed positive
    margin must favor digit 0. Both evidence and option-order flips should
    invert the sign, conditioned on constant project/language/kind.
    """
    if set(quartet)!={(0,0),(0,1),(1,0),(1,1)}:
        raise ValueError("Expected all four contrast states")
    correct=[]
    for state in (0,1):
        for swap in (0,1):
            margin=quartet[(state,swap)]
            sign=1.0 if (state^swap)==0 else -1.0
            correct.append(F.softplus(margin_target-sign*margin))
    # tanh makes the invariant penalty bounded enough not to swamp all choices.
    m={k:torch.tanh(v/2) for k,v in quartet.items()}
    order=(
       (m[(0,0)]+m[(0,1)]).pow(2)
      +(m[(1,0)]+m[(1,1)]).pow(2)
    )/2
    fact=(
       (m[(0,0)]+m[(1,0)]).pow(2)
      +(m[(0,1)]+m[(1,1)]).pow(2)
    )/2
    return torch.stack(correct).mean()+order_weight*order+fact_weight*fact

def signed_margin(model,ids,token_ids):
    output=logits_for(model,ids,token_ids)
    return output[0]-output[1]

def summarize(rows):
    results={mode:breakdown(rows,mode) for mode in
             ("frozen","compass","frozen_guarded","compass_guarded")}
    for mode in ("frozen","compass"):
        biases={}
        for lang in LANGUAGES:
            selected=[row for row in rows if row["lang"]==lang]
            pred=[r["prediction"][mode] for r in selected]
            marg=[r["margin"][mode] for r in selected]
            biases[lang]={"total":len(pred),"predict_zero":pred.count(0),
                          "predict_one":pred.count(1),
                          "mean_signed_margin_0_minus_1":round(sum(marg)/len(marg),5)}
        results[mode]["language_prior_diagnostics"]=biases
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rank",type=int,choices=(4,16,64),default=4)
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--steps",type=int,default=48)
    ap.add_argument("--test-projects",type=int,choices=(2,6),default=2)
    ap.add_argument("--output",default="compass-v83-r4-s11.json")
    args=ap.parse_args()
    if args.steps<12 or args.steps%12:
        ap.error("Training steps must be a positive multiple of 12")
    structural=audit()
    random.seed(args.seed);torch.manual_seed(args.seed);torch.set_num_threads(2)
    beginning=time.monotonic()
    inputs,golds=generate()
    train_gold={g["id"]:g for g in golds["train"]}
    groups=make_groups(inputs["train"],train_gold)
    plan=list(group_plan(groups,args.seed,args.steps))
    assert len(groups)==144
    counts=Counter((key[1],key[2]) for key,_ in plan)
    assert max(counts.values())-min(counts.values())<=1
    tok=AutoTokenizer.from_pretrained(BASE)
    pair_tokens=code_ids(tok)
    all_examples=inputs["train"]+inputs["test"]
    lengths={}
    for lang in LANGUAGES:
        lengths[lang]=max(
            len(tok.apply_chat_template([{"role":"user","content":prompt(row)}],
                                        tokenize=True,add_generation_prompt=True))
            for row in all_examples if row["lang"]==lang)
    if max(lengths.values())>1200:
        raise ValueError(f"Input exceeds 1200 tokens; no truncation permitted: {lengths}")
    print("PREFLIGHT FULL-PROMPT TOKEN LENGTHS",lengths,flush=True)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,args.rank)
    num_params=sum(x.numel() for x in adapter.parameters())
    assert num_params==2*model.config.hidden_size*args.rank
    # A zero-residual adapter is an exact no-op, as a necessary control.
    first_ids=chat_ids(tok,prompt(inputs["train"][0]))
    with torch.no_grad():
        before=logits_for(model,first_ids,pair_tokens)
        h=attach(model,adapter,1)
        try:after=logits_for(model,first_ids,pair_tokens)
        finally:h.remove()
    assert torch.allclose(before,after,atol=1e-5)

    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.0005)
    history=[]
    h=attach(model,adapter,1)
    try:
        for step,(key,quad) in enumerate(plan,1):
            optimizer.zero_grad(set_to_none=True)
            margins={}
            # Each update sees both evidence states and both option orders.
            for point in ((0,0),(0,1),(1,0),(1,1)):
                row,gold=quad[point]
                assert gold["correct_digit"]==(point[0]^point[1])
                ids=chat_ids(tok,prompt(row))
                margins[point]=signed_margin(model,ids,pair_tokens)
            loss=margin_loss(margins)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite contrast loss at step {step}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            optimizer.step()
            history.append({"step":step,"loss":round(float(loss.detach()),5),
                            "language":key[1],"family":key[2]})
            if step%12==0:
                print(f"TRAIN rank={args.rank} step={step}/{args.steps} loss={history[-1]['loss']:.4f}",flush=True)
    finally:h.remove()

    projects=sorted({r["project"]["id"] for r in inputs["test"]})
    selected=[projects[0],projects[3]] if args.test_projects==2 else projects
    held=[(row,gold) for row,gold in zip(inputs["test"],golds["test"])
          if row["project"]["id"] in selected]
    assert len(held)==48*args.test_projects
    records=[]
    for i,(row,gold) in enumerate(held,1):
        ids=chat_ids(tok,prompt(row))
        with torch.no_grad():
            frozen=signed_margin(model,ids,pair_tokens)
            h=attach(model,adapter,1)
            try:trained=signed_margin(model,ids,pair_tokens)
            finally:h.remove()
        predictions={"frozen":int(float(frozen)<0),"compass":int(float(trained)<0)}
        predictions["frozen_guarded"]=enforce(row,predictions["frozen"])["choice"]
        predictions["compass_guarded"]=enforce(row,predictions["compass"])["choice"]
        records.append({
            "id":row["task"]["id"],"project_id":row["project"]["id"],
            "kind":gold["kind"],"state":gold["state"],"swap":gold["swap"],
            "lang":row["lang"],"gold":gold["correct_digit"],
            "prediction":predictions,
            "margin":{"frozen":float(frozen),"compass":float(trained)},
            "guard_overrides":{
                "frozen":enforce(row,predictions["frozen"])["overridden"],
                "compass":enforce(row,predictions["compass"])["overridden"]},
        })
        if i%24==0:print(f"EVAL {i}/{len(held)}",flush=True)
    scores=summarize(records)
    majority={"accuracy":0.5,"explanation":"Both digits balanced; always 0 or always 1 yields 50% and zero complete quadruples."}
    report={
        "experiment":"COMPASS V8.3 paired-margin and flip-invariant bilingual pilot",
        "backbone":BASE,"rank":args.rank,"seed":args.seed,
        "adapter_parameters":num_params,"updates":args.steps,
        "examples_per_update":4,"training_groups":len(groups),
        "training_schedule_by_language_and_kind":{f"{k[0]}::{k[1]}":v for k,v in sorted(counts.items())},
        "train_inputs":len(inputs["train"]),"heldout_pool":len(inputs["test"]),
        "evaluated_items":len(records),"evaluated_projects":selected,
        "language_prompt_max_tokens":lengths,
        "time_seconds":round(time.monotonic()-beginning,2),
        "optimization_history":history,
        "majority_baseline":majority,"results":scores,"decisions":records,
        "data_audit":structural,
        "limitations":[
            "Fictional bilingual templates and labels authored by one research workflow; not human validated.",
            "Different fictional user IDs, but no distinct individual preference policies in the held-out fixtures.",
            "Training explicitly couples four semantic permutations; test families share templated structure.",
            "No unprompted generation or grounded free-form justification is tested.",
            "External permission gates are not learned neural policy.",
            "One seed/rank cannot demonstrate reliable generalization or a capacity advantage."
        ]
    }
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
              str(output.with_suffix(".safetensors")))
    for mode,s in scores.items():
        c=s["contrast"]
        print(f"RESULT {mode}: {s['correct']}/{s['total']} "+
              f"facts={c['correct_fact_flip_pairs']}/{c['fact_flip_pairs']} "+
              f"orders={c['correct_order_flip_pairs']}/{c['order_flip_pairs']} "+
              f"complete={c['all_four_correct']}/{c['quadruples']} "+
              f"EN={s['english_correct']} HE={s['hebrew_correct']}",flush=True)
    print("SAVED",output,flush=True)

if __name__=="__main__":
    main()
