"""COMPASS V8.2: bilingual paired-choice priority pilot on frozen SmolLM2-360M.

Experimental question: does a 7,680-weight residual adapter learn to rank
two actions under a controlled changed core fact, *rather than* guess one of
six fixed action names? Two matched single-token output codes (0/1).
No private user data. No changes to the separate SELF self-model.
"""
import argparse
import json
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoTokenizer,AutoModelForCausalLM

from run import BASE,PersonalAdapter,attach
from fixtures_v82 import generate,LANGUAGES,KINDS
from validate_v82 import audit
from guard_v82 import enforce

def prompt(row):
    p=row["project"];t=row["task"];lang=row["lang"]
    # IDs and authorization enforcement metadata are deliberately not model input.
    if lang=="he":
        return (
            "אתה מסייע לקבוע סדרי עדיפויות בפרויקט. בחר את הצעד שמשרת טוב יותר "
            "את מטרת הליבה על פי הראיות העדכניות. מיקום מבני משני אינו אומר "
            "שהמשימה אינה דחופה; בדיקת רגרסיה שחוסמת תיקון יכולה להיות חיונית. "
            "כבד החלטות ואישורים. השב בספרה 0 או 1 בלבד.\n"
            f"מטרת הפרויקט: {p['objective']['he']}\n"
            f"שלב: {p['phase']}\n"
            f"עיקרון מאושר: {p['policy']['he']}\n"
            f"שכבת המשימה: {t['layer']}\n"
            f"ראיה נוכחית ({t['record_source_description']}): {t['record']}\n"
            f"0: {t['choices'][0]}\n1: {t['choices'][1]}\n"
            "הצעד המועדף (0 או 1):"
        )
    return (
        "Decide which of two actions best advances this project's core "
        "given current verified evidence. A structurally ancillary task can "
        "be urgent when it blocks a core repair. Respect user approvals. "
        "Reply with the single digit 0 or 1 only.\n"
        f"Project goal: {p['objective']['en']}\n"
        f"Phase: {p['phase']}\n"
        f"Approved principle: {p['policy']['en']}\n"
        f"Structural layer: {t['layer']}\n"
        f"Current evidence ({t['record_source_description']}): {t['record']}\n"
        f"0: {t['choices'][0]}\n1: {t['choices'][1]}\n"
        "Preferred next action (0 or 1):"
    )

def chat_ids(tok,description):
    tokens=tok.apply_chat_template(
        [{"role":"user","content":description}],
        tokenize=True,add_generation_prompt=True)
    if len(tokens)>1200:
        raise ValueError(f"Prompt too long for 1200-token cap: {len(tokens)} tokens")
    return torch.tensor([tokens],dtype=torch.long)

def code_ids(tok):
    toks=[tok.encode(d,add_special_tokens=False) for d in ("0","1")]
    if any(len(t)!=1 for t in toks) or toks[0]==toks[1]:
        raise ValueError(f"0 and 1 must each be a different single token: {toks}")
    return [t[0] for t in toks]

def logits_for(model,token_ids,labels):
    return model(input_ids=token_ids).logits[0,-1,labels].float()

def chosen(logits):
    return int(torch.argmax(logits).item())

def schedule(inputs,labels,seed,steps):
    """Stratified over 6 dilemmas × 2 states × 2 orderings × 2 languages.

    Alternates records from varied train projects and profiles, so any single
    fixed digit guess is at exactly 50% on the training schedule.
    """
    rng=random.Random(seed)
    indexed=defaultdict(list)
    for row in inputs:
        lab=labels[row["task"]["id"]]
        indexed[(row["lang"],lab["kind"],lab["state"],lab["swap"])].append(row)
    keys=sorted(indexed)
    assert len(keys)==48
    for records in indexed.values():rng.shuffle(records)
    rng.shuffle(keys)
    offsets={k:0 for k in keys}
    result=[]
    for i in range(steps):
        k=keys[i%len(keys)]
        rows=indexed[k]
        item=rows[offsets[k]%len(rows)]
        offsets[k]+=1
        result.append(item)
    cnt=Counter(labels[x["task"]["id"]]["correct_digit"] for x in result)
    assert abs(cnt[0]-cnt[1])<=1
    return result

def breakdown(rows,field):
    truth=[r["gold"] for r in rows]
    values=[r["prediction"][field] for r in rows]
    n=len(truth)
    stats={"correct":sum(a==b for a,b in zip(truth,values)),
           "total":n,"predicted_zero":sum(x==0 for x in values),
           "predicted_one":sum(x==1 for x in values)}
    stats["english_correct"]=sum(r["gold"]==r["prediction"][field] for r in rows if r["lang"]=="en")
    stats["hebrew_correct"]=sum(r["gold"]==r["prediction"][field] for r in rows if r["lang"]=="he")
    grouped=defaultdict(dict)
    for r in rows:
        grouped[(r["project_id"],r["lang"],r["kind"])][(r["state"],r["swap"])]=r
    consistency={"all_four_correct":0,"correct_fact_flip_pairs":0,
                 "correct_order_flip_pairs":0,"quadruples":len(grouped),
                 "fact_flip_pairs":2*len(grouped),"order_flip_pairs":2*len(grouped)}
    family_stats=defaultdict(lambda:{"correct":0,"total":0})
    for r in rows:
        k=family_stats[r["kind"]]
        k["total"]+=1;k["correct"]+=int(r["prediction"][field]==r["gold"])
    for group in grouped.values():
        if len(group)!=4:raise AssertionError("Incomplete held-out contrast unit")
        ok={(state,swap):group[(state,swap)]["prediction"][field]==group[(state,swap)]["gold"]
            for state in (0,1) for swap in (0,1)}
        consistency["all_four_correct"]+=int(all(ok.values()))
        for state in (0,1):
            consistency["correct_order_flip_pairs"]+=int(ok[(state,0)] and ok[(state,1)])
        for swap in (0,1):
            consistency["correct_fact_flip_pairs"]+=int(ok[(0,swap)] and ok[(1,swap)])
    stats["contrast"]=consistency
    stats["families"]={k:dict(v) for k,v in sorted(family_stats.items())}
    return stats

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--rank",type=int,choices=(4,16,64),default=4)
    ap.add_argument("--seed",type=int,default=11)
    ap.add_argument("--steps",type=int,default=96)
    ap.add_argument("--test-projects",type=int,default=2)
    ap.add_argument("--output",default="compass-v82-r4-s11.json")
    args=ap.parse_args()
    if args.steps%48 or args.test_projects not in (1,2,3,6):
        ap.error("--steps must be a multiple of 48; --test-projects in 1,2,3,6")
    qa=audit()
    torch.manual_seed(args.seed);random.seed(args.seed);torch.set_num_threads(2)
    begin=time.monotonic()
    inputs,golds=generate()
    labels={g["id"]:g for g in golds["train"]}
    plan=schedule(inputs["train"],labels,args.seed,args.steps)
    tok=AutoTokenizer.from_pretrained(BASE)
    choices=code_ids(tok)
    # Scan every fictional input before allocating the 360M backbone.
    # Bilingual tokenization may be much longer than English; do not trim
    # away decision evidence or silently use candidate-dependent truncation.
    by_lang={}
    for lang in ("en","he"):
        length=max(len(tok.apply_chat_template(
            [{"role":"user","content":prompt(row)}],
            tokenize=True,add_generation_prompt=True))
            for split in ("train","test") for row in inputs[split] if row["lang"]==lang)
        by_lang[lang]=length
        if length>1200:
            raise ValueError(f"{lang} sample reaches {length} tokens, over agreed cap")
    print("PREFLIGHT bilingual token lengths",by_lang,flush=True)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,args.rank)
    count=sum(p.numel() for p in adapter.parameters())
    assert count==2*model.config.hidden_size*args.rank

    # Sanity: zero residual must have no effect before optimization.
    sample=chat_ids(tok,prompt(inputs["test"][0]))
    with torch.no_grad():
        base=logits_for(model,sample,choices)
        hook=attach(model,adapter,1)
        try:initial=logits_for(model,sample,choices)
        finally:hook.remove()
    assert torch.allclose(base,initial,rtol=1e-5,atol=1e-5)

    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.0005)
    train_losses=[]
    hook=attach(model,adapter,1)
    try:
        for step,row in enumerate(plan,1):
            desired=labels[row["task"]["id"]]["correct_digit"]
            ids=chat_ids(tok,prompt(row))
            optimizer.zero_grad(set_to_none=True)
            logits=logits_for(model,ids,choices)
            loss=nn.functional.cross_entropy(logits.unsqueeze(0),torch.tensor([desired]))
            if not torch.isfinite(loss):raise RuntimeError("NaN or Inf train loss")
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            optimizer.step()
            train_losses.append(round(loss.item(),5))
            if step%24==0:
                print(f"TRAIN rank={args.rank} seed={args.seed} step={step}/{args.steps} loss={loss.item():.4f}",flush=True)
    finally:hook.remove()

    available=sorted(set(row["project"]["id"] for row in inputs["test"]))
    # For test-projects=2, use different held-out user profiles and project families.
    chosen_projects=[available[0],available[3]] if args.test_projects==2 else available[:args.test_projects]
    examples=[(row,gold) for row,gold in zip(inputs["test"],golds["test"])
              if row["project"]["id"] in chosen_projects]
    assert len(examples)==args.test_projects*48
    outputs=[]
    for index,(row,gold) in enumerate(examples,1):
        ids=chat_ids(tok,prompt(row))
        with torch.no_grad():
            frozen=logits_for(model,ids,choices)
            hook=attach(model,adapter,1)
            try:trained=logits_for(model,ids,choices)
            finally:hook.remove()
        raw={"frozen":chosen(frozen),"compass":chosen(trained)}
        guarded={"frozen_guarded":enforce(row,raw["frozen"])["choice"],
                 "compass_guarded":enforce(row,raw["compass"])["choice"]}
        raw.update(guarded)
        outputs.append({
            "id":row["task"]["id"],"project_id":row["project"]["id"],
            "kind":gold["kind"],"state":gold["state"],"swap":gold["swap"],
            "lang":row["lang"],"gold":gold["correct_digit"],
            "prediction":raw,
            "raw_logits":{"frozen":frozen.tolist(),"compass":trained.tolist()},
            "guard_override":{
              "frozen":enforce(row,raw["frozen"])["overridden"],
              "compass":enforce(row,raw["compass"])["overridden"]},
            "model_input_prompt":prompt(row),
        })
        if index%24==0:print(f"EVAL {index}/{len(examples)}",flush=True)

    scores={name:breakdown(outputs,name) for name in
            ("frozen","compass","frozen_guarded","compass_guarded")}
    for name in scores:
        s=scores[name]
        print(f"RESULT {name} {s['correct']}/{s['total']} "+
              f"fact_flips={s['contrast']['correct_fact_flip_pairs']}/{s['contrast']['fact_flip_pairs']} "+
              f"all4={s['contrast']['all_four_correct']}/{s['contrast']['quadruples']} "+
              f"english={s['english_correct']} hebrew={s['hebrew_correct']}",flush=True)
    report={
        "experiment":"COMPASS V8.2 bilingual pairwise decision preference pilot",
        "rank":args.rank,"seed":args.seed,"parameters":count,"steps":args.steps,
        "training_label_histogram":dict(Counter(labels[x["task"]["id"]]["correct_digit"] for x in plan)),
        "base":BASE,"train_size":len(inputs["train"]),"heldout_pool":len(inputs["test"]),
        "evaluated_projects":chosen_projects,"evaluated_items":len(outputs),
        "seconds":round(time.monotonic()-begin,2),"audit":qa,
        "scores":scores,"losses":train_losses,"predictions":outputs,
        "limitations":[
          "Entirely fictional task labels and project conditions; no independent human grading.",
          "Shared authored dilemma families recur across train and holdout; wording and user/project names differ.",
          "Same generic policy text across fictional users; no demonstrated learned per-user policy.",
          "The user sees only a choice of two supplied candidate actions, not a freely reasoned plan.",
          "Authoritative action guard is NOT evidence of neural learning.",
          "This pilot uses one rank and seed, no compute-matched capacity inference."
        ]
    }
    target=Path(args.output);target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
              str(target.with_suffix(".safetensors")))
    print(f"SAVED {target}",flush=True)

if __name__=="__main__":main()
