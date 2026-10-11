"""V5 — explicit revisioned personal memory, selective adapter gate, open answer audits.
Research prototype only: extraction accepts a documented constrained syntax.
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import BASE, PersonalAdapter, attach
from benchmark_v4 import target_loss, generate
from personal_v5_data import SLOTS, GENERIC_QA, utterances, test_rows

DECLARE = re.compile(r"^For ([a-z ]+), my choice is (.+)\.$",re.I)
REVISE = re.compile(r"^Correction for ([a-z ]+): use (.+) instead of (.+)\.$",re.I)
STOP = set("what which is are do did does can could tell the my me i a an for in on of please choice chosen use be from this has and to should how long did settle".split())
NUMBER = dict(zip(("zero one two three four five six seven eight nine ten eleven twelve").split(),
                  [str(i) for i in range(13)]))

def terms(text):
    return [t for t in re.findall(r"[a-z0-9]+",text.lower()) if t not in STOP]

def canonical_history(messages):
    """Parse only constrained first-person assertions; preserve supersession audit trail."""
    memory={}
    for seq,msg in enumerate(messages):
        if msg.get("role")!="user":
            continue
        value=msg["text"].strip()
        init=DECLARE.fullmatch(value)
        change=REVISE.fullmatch(value)
        if init:
            slot,answer=init.groups()
            key=slot.strip().lower()
            if key in memory:
                raise ValueError(f"Duplicate declaration: {key}")
            memory[key]={"slot":key,"current":answer.strip(),
                         "revisions":[{"seq":seq,"value":answer.strip(),"source":value,"supersedes":None}]}
        elif change:
            slot,new,old=change.groups()
            key=slot.strip().lower()
            if key not in memory:
                raise ValueError(f"Correction to missing topic: {key}")
            if memory[key]["current"].casefold()!=old.strip().casefold():
                raise ValueError(f"Correction does not match current value: {key}")
            revisions=memory[key]["revisions"]
            revisions.append({"seq":seq,"value":new.strip(),"source":value,
                              "supersedes":revisions[-1]["seq"]})
            memory[key]["current"]=new.strip()
    return memory

def retrieve(question,memory,k=2):
    """Use question tokens and active slot names only: never inspect truth or question ID."""
    query=set(terms(question))
    scored=[]
    for slot,entry in memory.items():
        keyword=set(terms(slot))
        overlap=len(query.intersection(keyword))
        # An exact phrase match is stronger than partial token match.
        exact=3 if slot in question.casefold() else 0
        score=exact + overlap/max(1,len(keyword))
        scored.append((score,slot,entry))
    scored.sort(key=lambda t:(-t[0],t[1]))
    return [entry for score,_,entry in scored if score>0][:k]

def get_context(question,memory,k=2):
    entries=retrieve(question,memory,k)
    if not entries:
        return ""
    return "Current user preferences (newer statements replace older ones):\n"+\
        "\n".join(f"- For {e['slot']}, current choice is {e['current']}." for e in entries)

def should_use_adapter(question,memory):
    # Transparent heuristic gate. Not an ML router: user-specific phrasing + matching slot.
    personal=bool(re.search(r"\b(my|mine|me|i)\b",question.lower()))
    return personal and bool(retrieve(question,memory,k=1))

def normalized(value):
    parts=re.findall(r"[a-z0-9]+",value.lower())
    return " ".join(NUMBER.get(token,token) for token in parts)

def grade(generated,expected,stale=None):
    text=" "+normalized(generated)+" "
    target=" "+normalized(expected)+" "
    contained=target in text
    stale_seen=bool(stale and (" "+normalized(stale)+" ") in text)
    # Avoid accepting a correct phrase only in a sentence negating that phrase.
    negated=bool(re.search(r"\b(?:not|never|instead of|used to be)\s+"+re.escape(normalized(expected)),normalized(generated)))
    # Penalize pathological duplicated answer fragments.
    words=normalized(generated).split()
    repeated=any(words[i:i+3]==words[i+3:i+6] for i in range(max(0,len(words)-5)))
    return {"pass":contained and not stale_seen and not negated and not repeated,
            "stale_mentioned":stale_seen,"repetition":repeated}

@torch.no_grad()
def choose(tokenizer,model,rows,contexts=None):
    results={}
    for item in rows:
        ctx=contexts[item["id"]] if contexts else ""
        wrong=item["old"] if item["old"] else item["wrong"]
        pos=target_loss(tokenizer,model,item["test"],item["right"],ctx).item()
        neg=target_loss(tokenizer,model,item["test"],wrong,ctx).item()
        results[item["id"]]=bool(pos<neg)
    return results

def make_train_plan(rows,seed,stage,steps):
    # Never replay obsolete values. Each current fact is seen; newly added and changed facts are repeated.
    mandatory=list(rows)
    mandatory.extend(rows[-4:])
    mandatory.extend(row for row in rows if row["old"] is not None)
    if len(mandatory)>steps:raise ValueError("steps smaller than mandatory replay")
    gen=torch.Generator().manual_seed(seed*101+stage)
    while len(mandatory)<steps:
        mandatory.append(rows[torch.randint(len(rows),(1,),generator=gen).item()])
    order=torch.randperm(len(mandatory),generator=gen).tolist()
    return [mandatory[i] for i in order]

def score_binary(rows,answers):
    return {"correct":sum(bool(answers[r["id"]]) for r in rows),
            "total":len(rows),"by_id":answers}

def eval_stage(tokenizer,model,adapter,rows,memory):
    contexts={r["id"]:get_context(r["test"],memory) for r in rows}
    frozen=choose(tokenizer,model,rows)
    retrieval=choose(tokenizer,model,rows,contexts)
    handle=attach(model,adapter,1)
    try:
        personal=choose(tokenizer,model,rows)
        hybrid=choose(tokenizer,model,rows,contexts)
    finally:
        handle.remove()
    hits={}
    for r in rows:
        found=retrieve(r["test"],memory,1)
        hits[r["id"]]=bool(found and found[0]["slot"]==r["slot"])
    gated={x["id"]:(hybrid if should_use_adapter(x["test"],memory) else retrieval)[x["id"]] for x in rows}
    direct={}
    for x in rows:
        found=retrieve(x["test"],memory,1)
        direct[x["id"]]=bool(found and normalized(found[0]["current"])==normalized(x["right"]))
    return {"scores":{name:score_binary(rows,v) for name,v in
                       (("frozen",frozen),("adapter_only",personal),("retrieval",retrieval),
                        ("always_hybrid",hybrid),("gated_hybrid",gated),
                        ("direct_memory",direct))},
            "retrieval_current_hit1":sum(hits.values()),
            "retrieval_total":len(rows),"retrieval_by_id":hits}

def open_answers(tokenizer,model,adapter,rows,memory):
    modes=("frozen","adapter_only","retrieval","always_hybrid","gated_hybrid")
    modes=modes+("direct_memory",)
    result={}
    for mode in modes:
        learned=mode in ("adapter_only","always_hybrid","gated_hybrid")
        contexted=mode in ("retrieval","always_hybrid","gated_hybrid")
        outputs=[]
        generic=[]
        for item in rows:
            question=item["test"]
            with_adapter=learned and (mode!="gated_hybrid" or should_use_adapter(question,memory))
            if mode=="direct_memory":
                match=retrieve(question,memory,1)
                predicted=match[0]["current"] if match else "unknown"
            else:
                handle=attach(model,adapter,1) if with_adapter else None
                try:
                    predicted=generate(tokenizer,model,question,get_context(question,memory) if contexted else "",tokens=28)
                finally:
                    if handle:handle.remove()
            quality=grade(predicted,item["right"],item["old"])
            outputs.append({"id":item["id"],"slot":item["slot"],"expected":item["right"],
                            "stale":item["old"],"generated":predicted,**quality})
        for question,expected in GENERIC_QA:
            with_adapter=learned and (mode!="gated_hybrid" or should_use_adapter(question,memory))
            if mode=="direct_memory":
                match=retrieve(question,memory,1) if should_use_adapter(question,memory) else []
                predicted=match[0]["current"] if match else generate(tokenizer,model,question,tokens=28)
            else:
                handle=attach(model,adapter,1) if with_adapter else None
                try:
                    predicted=generate(tokenizer,model,question,
                        get_context(question,memory) if contexted and should_use_adapter(question,memory) else "",tokens=28)
                finally:
                    if handle:handle.remove()
            generic.append({"question":question,"expected":expected,"generated":predicted,
                            "adapter_enabled":with_adapter,**grade(predicted,expected)})
        revised=[r for r in outputs if r["stale"] is not None]
        result[mode]={"personal_correct":sum(a["pass"] for a in outputs),"personal_total":len(outputs),
                      "revision_correct":sum(a["pass"] for a in revised),"revision_total":len(revised),
                      "generic_correct":sum(a["pass"] for a in generic),"generic_total":len(generic),
                      "generic_adapter_on":sum(a["adapter_enabled"] for a in generic),
                      "personal_outputs":outputs,"generic_outputs":generic}
        print("open",mode,result[mode]["personal_correct"],"/",len(outputs),"generic",result[mode]["generic_correct"],"/",len(generic),"revisions",result[mode]["revision_correct"],"/",len(revised),flush=True)
    return result

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",type=int,choices=range(3),required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--steps",type=int,default=24)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,4)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.0005)
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    report={"protocol":"V5 explicit-constrained-utterance extraction","profile":args.profile,"seed":args.seed,
            "base":BASE,"steps_each_stage":args.steps,"stages":[],
            "adapter_parameters":sum(x.numel() for x in adapter.parameters()),
            "limitations":"Synthetic constrained phrasing, heuristic router, 3 profiles, 1 seed each; free-answer heuristic grader."}
    for stage in range(1,4):
        rows=test_rows(args.profile,stage)
        messages=utterances(args.profile,stage)
        memory=canonical_history(messages)
        assert len(memory)==len(rows)
        for row in rows:
            assert memory[row["slot"]]["current"]==row["right"],"Canonical memory did not apply latest revision"
        plan=make_train_plan(rows,args.seed,stage,args.steps)
        exposure=Counter(x["id"] for x in plan)
        assert set(exposure)==set(x["id"] for x in rows)
        losses=[]
        handle=attach(model,adapter,1)
        try:
            for sample in plan:
                optimizer.zero_grad(set_to_none=True)
                current=target_loss(tokenizer,model,sample["train"],sample["right"])
                if not torch.isfinite(current):raise RuntimeError("nonfinite loss")
                current.backward()
                nn.utils.clip_grad_norm_(adapter.parameters(),1)
                optimizer.step()
                losses.append(round(current.item(),4))
        finally:
            handle.remove()
        metrics=eval_stage(tokenizer,model,adapter,rows,memory)
        report["stages"].append({"stage":stage,"facts":len(rows),"revisions":sum(len(x["revisions"])-1 for x in memory.values()),
                                 "exposures":dict(exposure),"last_train_loss":losses[-1],
                                 "metrics":metrics,
                                 "active_memory":{key:{"current":v["current"],"revisions":v["revisions"]} for key,v in memory.items()}})
        save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
                  str(out.parent/f"v5-profile-{args.profile}-stage-{stage}.safetensors"))
        out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(f"profile={args.profile} seed={args.seed} stage={stage} "
              f"revised={report['stages'][-1]['revisions']} "
              f"retrieval-hit1={metrics['retrieval_current_hit1']}/{len(rows)} "
              f"frozen={metrics['scores']['frozen']['correct']}/{len(rows)} "
              f"adapter={metrics['scores']['adapter_only']['correct']}/{len(rows)} "
              f"retrieval={metrics['scores']['retrieval']['correct']}/{len(rows)} "
              f"hybrid={metrics['scores']['always_hybrid']['correct']}/{len(rows)}",flush=True)
    report["final_open_answers"]=open_answers(tokenizer,model,adapter,rows,memory)
    report["routing"]={"personal_enabled":sum(should_use_adapter(x["test"],memory) for x in rows),
                       "personal_total":len(rows),
                       "generic_disabled":sum(not should_use_adapter(q,memory) for q,_ in GENERIC_QA),
                       "generic_total":len(GENERIC_QA)}
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("V5 complete",out,flush=True)

if __name__=="__main__":
    main()
