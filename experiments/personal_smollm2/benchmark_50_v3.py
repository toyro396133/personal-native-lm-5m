"""V3: exhaustive rehearsal plus query-only lexical retrieval and a hybrid adapter."""
import argparse
import json
from pathlib import Path
from collections import Counter
from math import log
import re
import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import PersonalAdapter, attach, BASE

FIELDS = [
    ('project','My project {i} is named {v}.','What is project {i} named?','Remind me of my project {i} name.', ['Copper Finch','Silver Otter','Amber Comet','Quiet Harbor','Maple Circuit','Blue Compass']),
    ('dashboard','My dashboard {i} uses {v}.','What color does dashboard {i} use?','Which color did I choose for dashboard {i}?',['amber','teal','violet','navy','coral','olive']),
    ('duration','My work session {i} lasts {v}.','How long is work session {i}?','What duration is my work session {i}?',['15 minutes','20 minutes','25 minutes','35 minutes','40 minutes','50 minutes']),
    ('notebook','My notebook {i} is called {v}.','What is notebook {i} called?','What name did I choose for notebook {i}?',['Lantern Vault','Atlas Shelf','Copper Archive','Blue Drawer','Stone Cabinet','Maple Library']),
    ('drink','I drink {v} at break {i}.','What do I drink at break {i}?','Which drink do I like at break {i}?',['mint tea','cold water','apple juice','black coffee','lemon water','ginger tea']),
    ('tool','My tool for workflow {i} is {v}.','What is my tool for workflow {i}?','What tool do I use for workflow {i}?',['a notebook','a kanban board','a text editor','a timer','a spreadsheet','a whiteboard']),
    ('day','I review project {i} on {v}.','When do I review project {i}?','On which day do I review project {i}?',['Monday','Tuesday','Wednesday','Thursday','Friday','Sunday']),
    ('format','My report {i} uses {v}.','How do I format report {i}?','Which report format did I choose for report {i}?',['a checklist','one paragraph','a table','a flowchart','bullet points','a timeline']),
    ('place','Workspace {i} is named {v}.','What is workspace {i} named?','Remind me of the name of workspace {i}.',['Cedar Studio','Northern Loft','Copper Room','Glass Workshop','Pine Corner','Marble Hall']),
    ('activity','For break {i} I prefer {v}.','What do I prefer for break {i}?','Which activity did I choose for break {i}?',['a short walk','stretching','a quiet pause','drawing','reading','a puzzle']),
]

def facts():
    rows=[]
    for j in range(5):
        for key,statement,train,test,values in FIELDS:
            i=str(j+1)
            rows.append({'id':f'{key}-{i}','statement':statement.format(i=i,v=values[j]),'train':train.format(i=i),'test':test.format(i=i),'right':values[j],'wrong':values[j+1]})
    return rows

def loss(tok,model,question,answer,context=''):
    user=(context+'\n' if context else '')+'Answer briefly: '+question
    head=tok.apply_chat_template([{'role':'user','content':user}],tokenize=True,add_generation_prompt=True)
    tail=tok.encode(' '+answer,add_special_tokens=False)
    # Bounded prompt, with question retained at the end. Retrieval truncation is a confound.
    head=head[-(512-len(tail)):]
    labels=[-100]*len(head)+tail
    return model(input_ids=torch.tensor([head+tail]),labels=torch.tensor([labels])).loss*len(tail)

@torch.no_grad()
def check(tok,model,records,context=''):
    return {x['id']:(loss(tok,model,x['test'],x['right'],context).item()<loss(tok,model,x['test'],x['wrong'],context).item()) for x in records}

def metric(scores,keys):
    n=sum(scores[k] for k in keys)
    return {'correct':n,'total':len(keys),'accuracy':round(n/len(keys),4) if keys else None}


STOP = set("what which did do does is are i me my of the a an for on to in as have use did tell remind name called that it choose chose".split())

def tokenize(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]

def rank_notes(question, seen):
    """Entirely query-based; never consults answer, ground-truth ID, or evaluation labels."""
    query = tokenize(question)
    documents = [tokenize(item["statement"]) for item in seen]
    counts = Counter(token for doc in documents for token in set(doc))
    q_set = set(query)
    pairs = set(zip(query, query[1:]))
    ranked = []
    for index, doc in enumerate(documents):
        overlap = q_set.intersection(doc)
        tfidf = sum(log(1 + (len(seen) + 1) / (1 + counts[token]))**2 for token in overlap)
        proximity = len(pairs.intersection(set(zip(doc, doc[1:]))))
        ranked.append((tfidf + 1.5*proximity, index))
    ranked.sort(key=lambda p: (-p[0], p[1]))
    return [seen[index] for _, index in ranked]

@torch.no_grad()
def check_with_notes(tok, model, rows, contexts):
    return {
        item["id"]: (
            loss(tok, model, item["test"], item["right"], contexts[item["id"]]).item() <
            loss(tok, model, item["test"], item["wrong"], contexts[item["id"]]).item()
        )
        for item in rows
    }

def summarize(scores, old, new):
    ids=old+new
    return {"all":metric(scores,ids),"old":metric(scores,old),"new":metric(scores,new),"by_fact":scores}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--seed',type=int,default=19)
    parser.add_argument('--new-exposures',type=int,default=2)
    parser.add_argument('--output',default='benchmark-50-v3-results.json')
    args=parser.parse_args()
    if args.new_exposures<1: parser.error('new-exposures must be positive')
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,4)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.001)
    records=facts()
    report={'base':BASE,'seed':args.seed,'stages':[],'facts_total':len(records),'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'sampling':'each previously seen fact once plus each new fact repeated','new_exposures':args.new_exposures,'retrieval_control':'query-only TF-IDF unigram + adjacency bigram, top-1 indexed statement; NOT oracle','evaluation':'binary candidate log-likelihood ranking, NOT free-form recall'}
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    base_all=check(tok,model,records)
    for stage in range(1,11):
        seen=records[:5*stage]
        old=[x['id'] for x in seen[:-5]]
        new=[x['id'] for x in seen[-5:]]
        base={item["id"]:base_all[item["id"]] for item in seen}
        ranked={item["id"]:rank_notes(item["test"],seen) for item in seen}
        contexts={key:"Personal note: "+ranked[key][0]["statement"] for key in ranked}
        hits={item["id"]:ranked[item["id"]][0]["id"]==item["id"] for item in seen}
        top3={item["id"]:item["id"] in [r["id"] for r in ranked[item["id"]][:3]] for item in seen}
        retrieved={item["id"]:ranked[item["id"]][0]["id"] for item in seen}
        retrieval=check_with_notes(tok,model,seen,contexts)
        # Every old fact is rehearsed; every new fact gets configured exposures.
        old_records=seen[:-5]
        new_records=seen[-5:]
        plan=list(old_records)+list(new_records)*args.new_exposures
        generator=torch.Generator().manual_seed(args.seed+1000*stage)
        plan=[plan[i] for i in torch.randperm(len(plan),generator=generator).tolist()]
        exposures=Counter(item["id"] for item in plan)
        assert len(exposures)==len(seen)
        assert all(exposures[item["id"]]>=1 for item in old_records)
        assert all(exposures[item["id"]]==args.new_exposures for item in new_records)
        hook=attach(model,adapter,1)
        losses=[]
        try:
            for ex in plan:
                optimizer.zero_grad(set_to_none=True)
                current=loss(tok,model,ex['train'],ex['right'])
                if not torch.isfinite(current): raise RuntimeError('Nonfinite loss')
                current.backward()
                nn.utils.clip_grad_norm_(adapter.parameters(),1.0)
                optimizer.step()
                losses.append(round(current.item(),4))
            personal=check(tok,model,seen)
            hybrid=check_with_notes(tok,model,seen,contexts)
        finally:
            hook.remove()
        scores={name:summarize(s,old,new) for name,s in [
            ("base",base),("retrieval",retrieval),("personal",personal),("hybrid",hybrid)
        ]}
        retrieval_quality={
            "hit_at_1":metric(hits,old+new),
            "hit_at_3":metric(top3,old+new),
            "selected_id":retrieved,
        }
        report["stages"].append({
            "stage":stage,"facts_seen":len(seen),"optimizer_steps":len(plan),
            "train_loss_first":losses[0],"train_loss_last":losses[-1],
            "exposures":dict(exposures),"retrieval_quality":retrieval_quality,
            "scores":scores,
        })
        save_file(
            {k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
            str(out.parent/f"personal-v3-stage-{stage}.safetensors")
        )
        out.write_text(json.dumps(report,indent=2),encoding="utf-8")
        print(
            f"stage={stage} steps={len(plan)} base={scores['base']['all']['correct']}/{len(seen)} "
            f"personal={scores['personal']['all']['correct']}/{len(seen)} "
            f"retrieval={scores['retrieval']['all']['correct']}/{len(seen)} "
            f"hybrid={scores['hybrid']['all']['correct']}/{len(seen)} "
            f"hit1={retrieval_quality['hit_at_1']['correct']}/{len(seen)} "
            f"old={scores['personal']['old']['correct']}/{len(old)} "
            f"new={scores['personal']['new']['correct']}/5",
            flush=True
        )
    print("Done: ",out,flush=True)

if __name__=='__main__':main()
