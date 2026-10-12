"""V7: frozen SmolLM2 vs trained/untrained style adapter; fully gated prompt arm.

Pre-registered structure and content-anchor proxy checks; 18 held-out tasks, six
conditions, 160-token primary budget and matched 76-token truncation ablation.
Never uses real user data. This is not a production memory/privacy service.
"""
import argparse
import json
import re
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from run import BASE, PersonalAdapter, attach
from memory_v6 import Memory, STYLES, STYLE_TEXT, advice_task, prompt_for_style
from benchmark_v6 import (
    TRAINING, HELDOUT, GENERAL, target, train_loss, style_grade, generic_grade, repeated
)

# Locked test prompts and concept-anchor phrases, none appearing verbatim in the
# training task question set. Anchors are eval metadata, not part of the prompt.
TASKS = [
    ("How can I keep a shared folder organized?", ("folder","files","permissions")),
    ("How should I plan a study session?", ("topic","break","timer","review")),
    ("Help me make an easy budget.", ("expenses","income","spending","cost")),
    ("What steps should I take to test a new program?", ("tests","test","bugs","cases")),
    ("How can I improve a project handoff?", ("document","notes","owner","checklist")),
    ("How do I set up a simple weekly review?", ("week","progress","tasks","review")),
    ("Help me write a clear bug report.", ("steps","reproduce","expected","actual")),
    ("How should I clean up messy data?", ("duplicates","missing","validate","errors")),
    ("How can I prepare to give a short workshop?", ("agenda","practice","outline","rehearse")),
    ("What steps should I follow to review a pull request?", ("code","tests","changes","diff")),
    ("How can I avoid distractions during work?", ("notifications","focus","time","silence")),
    ("How do I break a difficult assignment into tasks?", ("smaller","steps","priorities","milestones")),
    ("How can I archive finished project files?", ("archive","backup","folder","label")),
    ("Help me prepare a short status update.", ("progress","done","next","blockers")),
    ("What steps should I take to onboard a new colleague?", ("access","guide","welcome","questions")),
    ("How should I prepare for a design review?", ("feedback","design","goals","examples")),
    ("How can I track a recurring maintenance task?", ("schedule","checklist","record","calendar")),
    ("How can I make a troubleshooting checklist?", ("symptoms","steps","verify","test")),
]
assert len(TASKS)==18 and all(q in [z[0] for z in TASKS] for q in HELDOUT)
assert not ({z[0] for z in TASKS} & {z[0] for z in TRAINING})

MODES = (
    "frozen", "explicit_prompt", "untrained_adapter_prompt",
    "trained_adapter_only", "trained_adapter_prompt", "gated_adapter_prompt"
)
MAIN_BUDGET=160
SHORT_BUDGET=76
SHORT_MODES=("explicit_prompt","trained_adapter_prompt")

def permissive_format(text,style):
    lines=[v.strip() for v in text.splitlines() if v.strip()]
    if repeated(text):return False
    if style=="two_bullets":
        return len(lines)==2 and all(bool(re.match(r"^(?:[-*]|\d+[.)])\s+\S",v)) for v in lines)
    if style=="three_steps":
        return len(lines)==3 and all(bool(re.match(r"^"+str(i+1)+r"[.)]\s+\S",v))
                                      for i,v in enumerate(lines))
    if style=="one_sentence":
        return (len(lines)==1 and not bool(re.match(r"^[-*0-9]",lines[0])) and
                len(re.findall(r"[.!?](?:\s|$)",text))==1 and
                len(re.findall(r"\b\w+\b",text))<=50)
    raise ValueError(style)

def content_proxy(text,anchors):
    words=set(re.findall(r"[a-z0-9]+",text.casefold()))
    return sum(anchor in words for anchor in anchors)>=1 and not repeated(text)

@torch.no_grad()
def generate(tok,model,question,budget):
    prefix=tok.apply_chat_template([{"role":"user","content":question}],
                                  tokenize=True,add_generation_prompt=True)
    ids=torch.tensor([prefix])
    seq=model.generate(ids,max_new_tokens=budget,do_sample=False,
                       pad_token_id=tok.eos_token_id)
    extra=seq[0,len(prefix):]
    text=tok.decode(extra,skip_special_tokens=True).strip()
    last=int(extra[-1]) if len(extra) else -1
    hit_limit=len(extra)>=budget and last!=tok.eos_token_id
    return {"output":text,"limit_hit":hit_limit,"generated_tokens":len(extra)}

def score_outputs(records,style):
    output=[]
    for question,anchors,generated in records:
        response=generated["output"]
        strict=style_grade(response,style)
        tolerant=permissive_format(response,style)
        content=content_proxy(response,anchors)
        output.append({
            "question":question,"reference_anchors":anchors,
            **generated,"format_strict":strict,"format_permissive":tolerant,
            "content_anchor_hit":content,"format_and_content":tolerant and content,
            "repeated":repeated(response),
        })
    return {
        "total":len(output),
        "format_strict":sum(x["format_strict"] for x in output),
        "format_permissive":sum(x["format_permissive"] for x in output),
        "content_anchor_hit":sum(x["content_anchor_hit"] for x in output),
        "format_and_content":sum(x["format_and_content"] for x in output),
        "hit_token_limit":sum(x["limit_hit"] for x in output),
        "repetitions":sum(x["repeated"] for x in output),
        "outputs":output,
    }

def memory_controls(style):
    m=Memory()
    statement=f"Please answer my practical questions in {STYLE_TEXT[style]}."
    assert m.observe(statement)=="set"
    assert m.peek()==style
    assert m.observe('My teammate prefers a single sentence.')=="ignored_quoted_or_third_party"
    assert m.observe("I might prefer three numbered steps someday.")=="ignored_hypothetical"
    assert m.peek()==style
    assert m.observe("Forget my response format preference.")=="deleted"
    assert m.peek() is None
    assert not (advice_task(TASKS[0][0]) and m.peek() is not None)
    assert m.observe("For my next 2 replies, use one concise sentence.")=="temporary"
    assert m.consume()=="one_sentence"
    assert m.consume()=="one_sentence"
    assert m.peek() is None
    assert m.observe(statement)=="set"
    assert m.peek()==style
    return {"passes":True,"audit_events":len(m.audit),"final_style":m.peek(),
            "delete_applies_to_active_memory_only":True}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--profile",type=int,choices=range(3),required=True)
    p.add_argument("--seed",type=int,required=True)
    p.add_argument("--steps",type=int,default=48)
    p.add_argument("--output",required=True)
    args=p.parse_args()
    if args.steps<1:p.error("--steps must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    style=STYLES[args.profile]
    controls=memory_controls(style)
    memory=Memory()
    assert memory.observe(f"Please answer my practical questions in {STYLE_TEXT[style]}.")=="set"
    adapter=PersonalAdapter(model.config.hidden_size,4)
    unused=PersonalAdapter(model.config.hidden_size,4)
    assert torch.count_nonzero(unused.up.weight)==0
    optimizer=torch.optim.AdamW(adapter.parameters(),lr=0.0003)
    losses=[]
    handle=attach(model,adapter,1)
    try:
        for i in range(args.steps):
            question,*ideas=TRAINING[i%len(TRAINING)]
            optimizer.zero_grad(set_to_none=True)
            current=train_loss(tok,model,question,target(style,ideas))
            if not torch.isfinite(current):
                raise RuntimeError("Nonfinite V7 training loss")
            current.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),0.5)
            optimizer.step()
            losses.append(round(current.item(),4))
    finally:
        handle.remove()
    report={
        "version":"V7","profile":args.profile,"seed":args.seed,"style":style,
        "frozen_backbone":BASE,"trainable_parameters":sum(p.numel() for p in adapter.parameters()),
        "steps":args.steps,"train_losses":losses,
        "memory_controls":controls,"main_budget":MAIN_BUDGET,"short_budget":SHORT_BUDGET,
        "evaluation":{"long":{},"short":{},"generic":{}},
        "note":"Synthetic held-out tasks; format-only and literal topic-keyword scores; no human content rating."
    }
    # Each condition is run against the identical prompts and 160-token budget.
    for mode in MODES:
        records=[]
        for question,anchors in TASKS:
            qualified=advice_task(question) and memory.peek() is not None
            instruction=prompt_for_style(memory.peek()) if mode in (
                "explicit_prompt","untrained_adapter_prompt","trained_adapter_prompt",
                "gated_adapter_prompt") and qualified else ""
            enabled=mode in ("trained_adapter_only","trained_adapter_prompt") or (
                mode=="gated_adapter_prompt" and qualified)
            which=unused if mode=="untrained_adapter_prompt" else adapter
            hook=attach(model,which,1) if enabled or mode=="untrained_adapter_prompt" else None
            try:
                generated=generate(tok,model,instruction+question,MAIN_BUDGET)
            finally:
                if hook:hook.remove()
            records.append((question,anchors,generated))
        result=score_outputs(records,style)
        report["evaluation"]["long"][mode]=result
        print(f"profile={args.profile} seed={args.seed} mode={mode} "+
              f"strict={result['format_strict']}/{result['total']} "+
              f"permissive={result['format_permissive']}/{result['total']} "+
              f"content={result['content_anchor_hit']}/{result['total']} "+
              f"joint={result['format_and_content']}/{result['total']} "+
              f"limit={result['hit_token_limit']}/{result['total']}",flush=True)
    # Matched 76-token generation ablation, using exact same prompts and task set.
    for mode in SHORT_MODES:
        records=[]
        for question,anchors in TASKS:
            instruction=prompt_for_style(memory.peek()) if advice_task(question) else ""
            hook=attach(model,adapter,1) if mode=="trained_adapter_prompt" else None
            try:
                generated=generate(tok,model,instruction+question,SHORT_BUDGET)
            finally:
                if hook:hook.remove()
            records.append((question,anchors,generated))
        report["evaluation"]["short"][mode]=score_outputs(records,style)
    # Generic queries must bypass the gated adapter; keep task prompts the same.
    for mode in ("frozen","trained_adapter_prompt","gated_adapter_prompt"):
        answers=[]
        for question,accepted in GENERAL:
            gate=advice_task(question) and memory.peek() is not None
            enabled=mode=="trained_adapter_prompt" or (mode=="gated_adapter_prompt" and gate)
            handle=attach(model,adapter,1) if enabled else None
            try:
                generated=generate(tok,model,question,48)
            finally:
                if handle:handle.remove()
            passed=generic_grade(generated["output"],accepted)
            answers.append({"question":question,"expected":accepted,**generated,
                            "pass":passed,"adapter_enabled":enabled})
        report["evaluation"]["generic"][mode]={
            "correct":sum(x["pass"] for x in answers),"total":len(answers),
            "adapter_active":sum(x["adapter_enabled"] for x in answers),
            "outputs":answers,
        }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_file({k:v.detach().cpu().contiguous() for k,v in adapter.state_dict().items()},
              str(out.parent/f"v7-profile-{args.profile}-seed-{args.seed}.safetensors"))
    print(f"Saved {out}; generic="+str({
        k:f"{v['correct']}/{v['total']} gated={v['adapter_active']}"
        for k,v in report["evaluation"]["generic"].items()}),flush=True)

if __name__=="__main__":
    main()
