"""V4: synthetic chat-history personalization, real top-k lookup, matched-update replay,
three independent seeded jobs, final open-ended answer checks, and generic regression.
Never processes real user conversations. No oracle labels are given to retrieval.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import BASE, PersonalAdapter, attach
from personal_v4_data import FACTS, UPDATES, DISTRACTORS, GENERIC_QA

STOP = set("a an and are as at be by can could did do does for from have how i in is it me my of on or our please should tell the there this to use what when where which with would you your that into".split())

def words(s):
    return [x for x in re.findall(r"[a-z0-9]+",s.lower()) if x not in STOP]

def normalize(s):
    return " ".join(re.findall(r"[a-z0-9]+",s.casefold()))

def freeform_matches(produced, expected):
    answer=normalize(expected)
    actual=normalize(produced)
    if not answer or not actual:
        return False
    if answer.startswith("a "):
        answer=answer[2:]
    # Literal phrase bounded by words; a conservative heuristic, not semantic grading.
    if f" {answer} " not in f" {actual} ":
        return False
    if re.search(r"\b(?:not|never|isn't|isnt|instead of)\s+"+re.escape(answer),actual):
        return False
    return True

def state(stage):
    items=[]
    for index,record in enumerate(FACTS[:stage*4]):
        source,train_q,test_q,right,wrong=record
        latest=(right,wrong)
        for update in UPDATES:
            if update["stage"]<=stage and update["index"]==index:
                latest=(update["answer"],update["wrong"])
        items.append({"id":f"f{index+1}","index":index,"source":source,
                      "train":train_q,"test":test_q,"right":latest[0],"wrong":latest[1]})
    return items

def history(stage):
    logs=[]
    for s in range(1,stage+1):
        for index in range(4*(s-1),4*s):
            logs.append({"id":f"f{index+1}","stage":s,"rev":False,
                         "text":FACTS[index][0],"role":"user"})
            logs.append({"id":None,"stage":s,"rev":False,
                         "text":"Understood. I'll take that into account.","role":"assistant"})
        for i in range(2):
            logs.append({"id":None,"stage":s,"rev":False,
                         "text":DISTRACTORS[((s-1)*2+i)%len(DISTRACTORS)],"role":"user"})
            logs.append({"id":None,"stage":s,"rev":False,
                         "text":"Here is a general explanation of the topic.","role":"assistant"})
        for upd in UPDATES:
            if upd["stage"]==s:
                logs.append({"id":f"f{upd['index']+1}","stage":s,"rev":True,
                             "text":upd["message"],"role":"user"})
                logs.append({"id":None,"stage":s,"rev":False,
                             "text":"Okay, I understand the updated preference.","role":"assistant"})
    return logs

def rank_memory(question, documents, k):
    """Search only user utterance text. IDs and correct answers are never features."""
    q=words(question)
    qset=set(q)
    pair=set(zip(q,q[1:]))
    docs=[d for d in documents if d["role"]=="user"]
    tokens=[words(d["text"]) for d in docs]
    df=Counter(t for ts in tokens for t in set(ts))
    scored=[]
    for doc,ts in zip(docs,tokens):
        common=qset.intersection(ts)
        lexical=sum(math.log(1+(len(docs)+1)/(1+df[t]))**2 for t in common)
        bigram=1.2*len(pair.intersection(zip(ts,ts[1:])))
        # Weak preference for updates among lexically relevant notes.
        score=lexical+bigram+(0.025*doc["stage"] if common else 0)
        scored.append((score,doc))
    scored.sort(key=lambda x:(-x[0],-x[1]["stage"]))
    return [d for _,d in scored[:k]]

def retrieval_context(docs):
    return "Relevant excerpts from earlier user messages (a later correction overrides an earlier one):\n"+\
           "\n".join(f"- Earlier message: {d['text']}" for d in docs)

def chat_prefix(tokenizer,question,context=""):
    prompt=(context+"\n\n" if context else "") + \
           "Question: "+question+"\nAnswer using only the short requested value. If you do not know, say unknown."
    return tokenizer.apply_chat_template([{"role":"user","content":prompt}], tokenize=True,add_generation_prompt=True)

def target_loss(tokenizer,model,question,answer,context="",max_tokens=384):
    head=chat_prefix(tokenizer,question,context)
    tail=tokenizer.encode(" "+answer,add_special_tokens=False)
    if len(tail)>=max_tokens: raise ValueError("Target exceeds token budget")
    if len(head)+len(tail)>max_tokens:
        raise RuntimeError("Prompt exceeded token budget; refusing silent history truncation")
    ids=head+tail
    labels=[-100]*len(head)+tail
    return model(input_ids=torch.tensor([ids]),labels=torch.tensor([labels])).loss*len(tail)

@torch.no_grad()
def choose(tokenizer,model,items,contexts=None):
    scores={}
    for item in items:
        ctx=contexts[item["id"]] if contexts else ""
        a=target_loss(tokenizer,model,item["test"],item["right"],ctx).item()
        b=target_loss(tokenizer,model,item["test"],item["wrong"],ctx).item()
        scores[item["id"]]={"pass":bool(a<b),"margin":round(b-a,4)}
    return scores

@torch.no_grad()
def generate(tokenizer,model,question,context="",tokens=24):
    prefix=chat_prefix(tokenizer,question,context)
    input_ids=torch.tensor([prefix])
    result=model.generate(input_ids,max_new_tokens=tokens,do_sample=False,
                          pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(result[0,len(prefix):],skip_special_tokens=True).strip()

def train_one(tokenizer,model,adapter,optimizer,plan):
    handle=attach(model,adapter,1)
    losses=[]
    try:
        for item in plan:
            optimizer.zero_grad(set_to_none=True)
            loss=target_loss(tokenizer,model,item["train"],item["right"])
            if not torch.isfinite(loss):raise RuntimeError("Non-finite personal training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
    finally:
        handle.remove()
    return losses

def plan_for(items,stage,seed,steps,mode):
    rng=torch.Generator().manual_seed(seed*10007+stage*173+(mode=="recent"))
    new=items[-4:]
    changed={f"f{u['index']+1}" for u in UPDATES if u["stage"]==stage}
    changes=[x for x in items if x["id"] in changed]
    if mode=="replay":
        plan=list(items)+list(new)+changes
        assert len(plan)<=steps, "Enough steps are required to replay all facts and new facts"
        while len(plan)<steps:
            plan.append(items[torch.randint(len(items),(1,),generator=rng).item()])
    else:
        # 32 steps concentrated on the 8 newest facts + any explicit corrections.
        active=items[-8:]
        by_id={x["id"]:x for x in active+changes}
        recent=list(by_id.values())
        plan=[recent[i%len(recent)] for i in range(steps)]
    indices=torch.randperm(len(plan),generator=rng).tolist()
    result=[plan[i] for i in indices]
    if mode=="replay":
        counts=Counter(x["id"] for x in result)
        assert all(counts[x["id"]]>=1 for x in items)
        assert all(counts[x["id"]]>=2 for x in new)
    return result

def counts(scores, ids):
    return {"correct":sum(bool(scores[k]["pass"]) for k in ids),"total":len(ids)}

def evaluate_stage(tokenizer,model,rows,logs,replay,recent,k):
    old_ids=[x["id"] for x in rows[:-4]]
    new_ids=[x["id"] for x in rows[-4:]]
    ranks={x["id"]:rank_memory(x["test"],logs,k) for x in rows}
    contexts={fid:retrieval_context(ds) for fid,ds in ranks.items()}
    hits1={x["id"]:bool(ranks[x["id"]] and ranks[x["id"]][0]["id"]==x["id"]) for x in rows}
    hitsk={x["id"]:any(d["id"]==x["id"] for d in ranks[x["id"]]) for x in rows}
    outcomes={"base":choose(tokenizer,model,rows),
              "retrieval":choose(tokenizer,model,rows,contexts)}
    for key,adapter,context in (
        ("personal",replay,None),("recent",recent,None),("hybrid",replay,contexts)
    ):
        hook=attach(model,adapter,1)
        try:
            outcomes[key]=choose(tokenizer,model,rows,context)
        finally:
            hook.remove()
    metrics={key:{"all":counts(v,old_ids+new_ids),"old":counts(v,old_ids),
                  "new":counts(v,new_ids),"details":v} for key,v in outcomes.items()}
    return {"scores":metrics,
            "retrieval":{"hit1":sum(hits1.values()),"hitk":sum(hitsk.values()),
                         "total":len(rows),"topk":k,
                         "incorrect_top1":[x for x in hits1 if not hits1[x]],
                         "selected":{fid:[d["id"] for d in ds] for fid,ds in ranks.items()}},
            "contexts":contexts}

def open_answer_tests(tokenizer,model,rows,logs,replay,recent,k):
    rank={x["id"]:rank_memory(x["test"],logs,k) for x in rows}
    contexts={fid:retrieval_context(ds) for fid,ds in rank.items()}
    out={}
    for key,adapter,with_context in (
        ("base",None,False),("retrieval",None,True),
        ("personal",replay,False),("recent",recent,False),("hybrid",replay,True)
    ):
        hook=attach(model,adapter,1) if adapter is not None else None
        try:
            results=[]
            for row in rows:
                predicted=generate(tokenizer,model,row["test"],
                                   contexts[row["id"]] if with_context else "")
                results.append({"id":row["id"],"expected":row["right"],"generated":predicted,
                                "pass":freeform_matches(predicted,row["right"])})
            general=[{"question":q,"expected":answer,
                      "generated":(pred:=generate(tokenizer,model,q)),
                      "pass":freeform_matches(pred,answer)} for q,answer in GENERIC_QA]
        finally:
            if hook is not None:hook.remove()
        out[key]={"personal_correct":sum(x["pass"] for x in results),
                  "personal_total":len(results),"personal_outputs":results,
                  "general_correct":sum(x["pass"] for x in general),
                  "general_total":len(general),"general_outputs":general}
        print(f"freeform {key}: {out[key]['personal_correct']}/{len(rows)}; generic {out[key]['general_correct']}/{len(general)}",flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--steps-per-stage",type=int,default=32)
    ap.add_argument("--topk",type=int,default=3)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    if args.steps_per_stage<32 or not 1<=args.topk<=5:
        ap.error("steps-per-stage >=32 and 1<=topk<=5 required")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    replay=PersonalAdapter(model.config.hidden_size,4)
    recent=PersonalAdapter(model.config.hidden_size,4)
    opt_replay=torch.optim.AdamW(replay.parameters(),lr=0.001)
    opt_recent=torch.optim.AdamW(recent.parameters(),lr=0.001)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    report={"version":"V4","base":BASE,"seed":args.seed,"topk":args.topk,
            "train_steps_per_stage_per_adapter":args.steps_per_stage,
            "frozen_base":True,"facts":len(FACTS),"stages":[],
            "note":"Synthetic chat, forced-choice stage tests, open-ended final tests; token-normalized answer matching is heuristic."}
    for stage in range(1,7):
        rows=state(stage)
        logs=history(stage)
        replay_plan=plan_for(rows,stage,args.seed,args.steps_per_stage,"replay")
        recent_plan=plan_for(rows,stage,args.seed,args.steps_per_stage,"recent")
        replay_losses=train_one(tokenizer,model,replay,opt_replay,replay_plan)
        recent_losses=train_one(tokenizer,model,recent,opt_recent,recent_plan)
        result=evaluate_stage(tokenizer,model,rows,logs,replay,recent,args.topk)
        stage_info={"stage":stage,"facts_seen":len(rows),
                    "history_messages":len(logs),
                    "train_exposures":{"replay":dict(Counter(x["id"] for x in replay_plan)),
                                       "recent":dict(Counter(x["id"] for x in recent_plan))},
                    "train_loss_last":{"replay":round(replay_losses[-1],4),
                                       "recent":round(recent_losses[-1],4)},
                    "scores":result["scores"],"retrieval":result["retrieval"]}
        report["stages"].append(stage_info)
        save_file({k:v.detach().cpu().contiguous() for k,v in replay.state_dict().items()},
                  str(output.parent/f"replay-stage-{stage}.safetensors"))
        save_file({k:v.detach().cpu().contiguous() for k,v in recent.state_dict().items()},
                  str(output.parent/f"recent-stage-{stage}.safetensors"))
        output.write_text(json.dumps(report,indent=2),encoding="utf-8")
        score=result["scores"]
        print(f"seed={args.seed} stage={stage} N={len(rows)} hit1={result['retrieval']['hit1']}/{len(rows)} "+
              " ".join(f"{name}={score[name]['all']['correct']}/{len(rows)}" for name in
                       ("base","recent","personal","retrieval","hybrid")),flush=True)
    print("Starting final open-ended personal and generic evaluation",flush=True)
    report["freeform"]=open_answer_tests(tokenizer,model,rows,logs,replay,recent,args.topk)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    print("Completed seed",args.seed,"results",str(output),flush=True)

if __name__=="__main__":
    main()
