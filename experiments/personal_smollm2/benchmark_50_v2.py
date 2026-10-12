"""V2: 50 fictional facts, balanced rehearsal, oracle single-fact text retrieval."""
import argparse
import json
from pathlib import Path
from collections import Counter
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

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--steps-per-stage',type=int,default=25)
    parser.add_argument('--output',default='benchmark-50-v2-results.json')
    args=parser.parse_args()
    if args.steps_per_stage<10: parser.error('steps must be >= 10, to train each new fact twice')
    torch.manual_seed(19)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    adapter=PersonalAdapter(model.config.hidden_size,4)
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.001)
    records=facts()
    report={'base':BASE,'stages':[],'facts_total':len(records),'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'sampling':'all new examples twice plus old rehearsal','retrieval_control':'oracle relevant single fact (upper-bound; not a practical retriever)'}
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    for stage in range(1,11):
        seen=records[:5*stage]
        old=[x['id'] for x in seen[:-5]]
        new=[x['id'] for x in seen[-5:]]
        base=check(tok,model,seen)
        # Oracle passes only the correct retrieved fact; no long-history truncation.
        retrieval={x['id']:check(tok,model,[x],'Personal note: '+x['statement'])[x['id']] for x in seen}
        # The schedule guarantees >=2 exposures for each newly introduced fact.
        recent=seen[-5:]
        previous=seen[:-5]
        rng=torch.Generator().manual_seed(202610+stage)
        def permute(items):
            return [items[i] for i in torch.randperm(len(items),generator=rng).tolist()]
        plan=permute(recent)+permute(recent)
        if previous:
            replay=permute(previous)
            offset=(stage*7)%len(replay)
            replay=replay[offset:]+replay[:offset]
            plan.extend(replay[i%len(replay)] for i in range(args.steps_per_stage-len(plan)))
        else:
            plan.extend(recent[i%len(recent)] for i in range(args.steps_per_stage-len(plan)))
        plan=permute(plan)
        exposures=Counter(x['id'] for x in plan)
        assert all(exposures[x['id']]>=2 for x in recent)

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
        finally:
            hook.remove()
        all_ids=old+new
        scores={name:{'all':metric(results,all_ids),'old':metric(results,old),'new':metric(results,new),'by_fact':results} for name,results in [('base',base),('retrieval',retrieval),('personal',personal)]}
        report['stages'].append({'stage':stage,'facts_seen':len(seen),'train_loss_first':losses[0],'train_loss_last':losses[-1],'exposures':dict(exposures),'scores':scores})
        save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},str(out.parent/f'personal-v2-stage-{stage}.safetensors'))
        out.write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(f"stage={stage} base={scores['base']['all']['correct']}/{len(seen)} personal={scores['personal']['all']['correct']}/{len(seen)} retrieval={scores['retrieval']['all']['correct']}/{len(seen)} old={scores['personal']['old']['correct']}/{len(old)} new={scores['personal']['new']['correct']}/5",flush=True)

if __name__=='__main__':main()
