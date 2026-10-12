"""V6: style-adapter study, explicit-memory prompting baselines, gated generic control.

All data are fictional. Gold style labels are evaluation metadata only; the
memory baseline gets the style from parsed user utterances. No real user data.
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
from memory_v6 import Memory, STYLES, STYLE_TEXT, advice_task, prompt_for_style, memory_smoke

TRAINING = [
  ("How can I organize my workday?", "List tasks by urgency", "Reserve time for deep work", "Review progress before finishing"),
  ("How should I prepare for a presentation?", "Identify the main message", "Rehearse your examples", "Check the slides and timing"),
  ("Help me clean up a cluttered desk.", "Clear items you no longer need", "Group related supplies", "Keep daily tools close"),
  ("How do I manage a small project?", "Define the outcome", "Assign the next actions", "Review the work regularly"),
  ("How can I improve my note taking?", "Capture key ideas", "Organize notes by topic", "Review the summary later"),
  ("How do I start learning to code?", "Pick a small task", "Practice writing code", "Debug and revise your solution"),
  ("What steps should I take for a team meeting?", "Write an agenda", "Ask for updates", "Capture decisions and owners"),
  ("How can I plan a short trip?", "Choose your destination", "Check travel arrangements", "Pack essentials in advance"),
  ("Help me reduce unnecessary notifications.", "Turn off low-value alerts", "Choose a review window", "Check urgent channels separately"),
  ("How can I finish a draft?", "Outline the message", "Write the first version", "Edit for clarity"),
  ("How should I track progress?", "Choose a meaningful metric", "Write regular updates", "Adjust the plan from feedback"),
  ("Help me prepare for an interview.", "Research the role", "Practice relevant examples", "Prepare questions to ask"),
]
HELDOUT = [
  "How can I keep a shared folder organized?",
  "How should I plan a study session?",
  "Help me make an easy budget.",
  "What steps should I take to test a new program?",
  "How can I improve a project handoff?",
  "How do I set up a simple weekly review?",
  "Help me write a clear bug report.",
  "How should I clean up messy data?",
  "How can I prepare to give a short workshop?",
  "What steps should I follow to review a pull request?",
  "How can I avoid distractions during work?",
  "How do I break a difficult assignment into tasks?",
]
GENERAL = [
  ("What is the capital of France?",("paris",)),
  ("What is seven plus two?",("9","nine")),
  ("What is the chemical formula for water?",("h2o",)),
  ("How many days are in a week?",("7","seven")),
  ("Which planet is called the red planet?",("mars",)),
  ("What is the capital of Italy?",("rome",)),
  ("What is six plus three?",("9","nine")),
  ("Which gas do plants absorb from the air?",("carbon dioxide","co2")),
]

def target(style,ideas):
    a,b,c=ideas
    if style=="two_bullets":
        return f"- {a}.\n- {b}."
    if style=="three_steps":
        return f"1. {a}.\n2. {b}.\n3. {c}."
    if style=="one_sentence":
        return f"{a}, then {b.lower()}."
    raise ValueError(style)

def head_ids(tokenizer,question):
    return tokenizer.apply_chat_template(
        [{"role":"user","content":question}],
        tokenize=True,add_generation_prompt=True)

def train_loss(tok,model,question,answer):
    head=head_ids(tok,question)
    tail=tok.encode(answer,add_special_tokens=False)+[tok.eos_token_id]
    labels=[-100]*len(head)+tail
    return model(input_ids=torch.tensor([head+tail]),labels=torch.tensor([labels])).loss

@torch.no_grad()
def generate(tok,model,question):
    head=head_ids(tok,question)
    ids=torch.tensor([head])
    out=model.generate(ids,max_new_tokens=76,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(out[0,len(head):],skip_special_tokens=True).strip()

def repeated(text):
    tokens=re.findall(r"[a-z0-9]+",text.lower())
    return any(tokens[i:i+4]==tokens[i+4:i+8] for i in range(max(0,len(tokens)-7)))

def style_grade(text,style):
    lines=[s.strip() for s in text.splitlines() if s.strip()]
    if repeated(text):return False
    if style=="two_bullets":
        return len(lines)==2 and all(re.match(r"^[-*] +\S",line) for line in lines)
    if style=="three_steps":
        return len(lines)==3 and all(re.match(r"^"+str(i+1)+r"[.)] +\S",line) for i,line in enumerate(lines))
    if style=="one_sentence":
        return (len(lines)==1 and not re.match(r"^[-*0-9]",lines[0]) and
                len(re.findall(r"[.!?](?:\s|$)",text))==1 and
                len(re.findall(r"\b\w+\b",text))<=30)
    raise ValueError(style)

def generic_grade(text,accepted):
    cleaned=re.sub(r"[^a-z0-9]+"," ",text.lower()).strip()
    if repeated(text):return False
    return any(re.search(r"(?<!\w)"+re.escape(a)+r"(?!\w)",cleaned) for a in accepted)

def memory_test(style):
    mem=Memory()
    mem.observe(f"Please answer my practical questions in {STYLE_TEXT[style]}.")
    assert mem.peek()==style
    # Injection-resistant bounded extraction; quoted/third-party/hypothetical text ignored.
    assert mem.observe('The blog told me "use one sentence".')=="ignored_quoted_or_third_party"
    assert mem.observe("I might prefer three steps someday.")=="ignored_hypothetical"
    assert mem.peek()==style
    mem.observe("Forget my response format preference.")
    assert mem.peek() is None
    assert not (advice_task("How can I organize my desk?") and mem.peek())
    mem.observe("For my next 2 replies, use one concise sentence.")
    assert mem.consume()=="one_sentence"
    assert mem.consume()=="one_sentence"
    assert mem.peek() is None
    mem.observe(f"Please answer my practical questions in {STYLE_TEXT[style]}.")
    assert mem.peek()==style
    return {"tests_passed":True,"active":mem.peek(),"audit":mem.audit,
            "outdated_or_deleted_visible":False}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",type=int,choices=range(3),required=True)
    ap.add_argument("--seed",type=int,required=True)
    ap.add_argument("--steps",type=int,default=48)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    if args.steps<1:ap.error("steps must be positive")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    style=STYLES[args.profile]
    adapter=PersonalAdapter(model.config.hidden_size,4)
    opt=torch.optim.AdamW(adapter.parameters(),lr=0.0003)
    memory=Memory()
    assert memory.observe(f"Please answer my practical questions in {STYLE_TEXT[style]}.")=="set"
    assert memory.peek()==style
    memory_controls=memory_test(style)
    print("Profile",args.profile,"style",style,"seed",args.seed,"memory tests",memory_controls["tests_passed"],flush=True)
    steps=[]
    hook=attach(model,adapter,1)
    try:
        for step in range(args.steps):
            question,*ideas=TRAINING[step%len(TRAINING)]
            opt.zero_grad(set_to_none=True)
            loss=train_loss(tok,model,question,target(style,ideas))
            if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(adapter.parameters(),0.5)
            opt.step()
            steps.append(round(loss.item(),4))
    finally:
        hook.remove()
    print(f"Finished adapter training: {len(steps)} steps, first={steps[0]} last={steps[-1]}",flush=True)
    report={"version":"V6_style","profile":args.profile,"style":style,"seed":args.seed,
            "backbone":BASE,"trainable_parameters":sum(p.numel() for p in adapter.parameters()),
            "train_steps":args.steps,"training_losses":steps,"memory_controls":memory_controls,
            "evaluation":{"styles":{},"generic":{}},"limitations":
            "Synthetic tasks, literal format-only scorer, small generic set, personal extractor is bounded syntax; no real chat."}
    modes=("base","memory_prompt","adapter_only","adapter_plus_prompt","gated_adapter")
    for mode in modes:
        style_out=[]
        general_out=[]
        on=mode in ("adapter_only","adapter_plus_prompt","gated_adapter")
        prompted=mode in ("memory_prompt","adapter_plus_prompt")
        for question in HELDOUT:
            instruction=prompt_for_style(memory.peek()) if prompted else ""
            enabled=on and (mode!="gated_adapter" or (advice_task(question) and memory.peek() is not None))
            hook=attach(model,adapter,1) if enabled else None
            try:
                reply=generate(tok,model,instruction+question)
            finally:
                if hook:hook.remove()
            style_out.append({"question":question,"output":reply,"format_pass":style_grade(reply,style),
                              "repetitive":repeated(reply),"adapter_enabled":enabled})
        for question,answers in GENERAL:
            enabled=on and (mode!="gated_adapter" or (advice_task(question) and memory.peek() is not None))
            hook=attach(model,adapter,1) if enabled else None
            try:
                reply=generate(tok,model,question)
            finally:
                if hook:hook.remove()
            general_out.append({"question":question,"output":reply,"semantic_proxy_pass":generic_grade(reply,answers),
                                "repetitive":repeated(reply),"adapter_enabled":enabled})
        report["evaluation"]["styles"][mode]={
            "correct":sum(x["format_pass"] for x in style_out),
            "total":len(style_out),"repetitive":sum(x["repetitive"] for x in style_out),
            "outputs":style_out}
        report["evaluation"]["generic"][mode]={
            "correct":sum(x["semantic_proxy_pass"] for x in general_out),
            "total":len(general_out),"repetitive":sum(x["repetitive"] for x in general_out),
            "adapter_enabled":sum(x["adapter_enabled"] for x in general_out),
            "outputs":general_out}
        print(f"profile={args.profile} seed={args.seed} mode={mode} style={report['evaluation']['styles'][mode]['correct']}/{len(HELDOUT)} general={report['evaluation']['generic'][mode]['correct']}/{len(GENERAL)}",flush=True)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2),encoding="utf-8")
    save_file({k:v.detach().contiguous().cpu() for k,v in adapter.state_dict().items()},
              str(output.parent/f"v6-style-profile-{args.profile}-seed-{args.seed}.safetensors"))
    print("Saved",output,flush=True)

if __name__=="__main__":
    main()
