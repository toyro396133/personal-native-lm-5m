"""V8.9 English-only teacher test: explicit candidate likelihood and bounded student feedback.

DO NOT use gold, task family, meta, hidden state, or task IDs in teacher prompts.
Frozen 360M is an actual active generator, NOT trained. The student provides
an untrusted preference-weighted first-pass signal in ONE additional round.
Outputs deliberately capture errors including no usable explanation / bias.
"""
import argparse, json, time, hashlib, math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from safetensors.torch import load_file
from fixtures_v89 import build,audit
from matrix_v85 import PersonalRanker
from ranker_v84 import representation,key_for,get_two
from run import BASE

def project_context(row, choices):
    p=row["project"];t=row["task"]
    return (
      "PROJECT GOAL: "+p["objective"]["en"]+"\n"
      "CURRENT APPROVED OWNER PRIORITY: "+p["policy"]["en"]+"\n"
      "CURRENT EVIDENCE: "+t["record"]+"\n"
      "OPTION A: "+choices[0]+"\nOPTION B: "+choices[1]+"\n"
    )

def reason_prompt(row):
    c=row["task"]["choices"]
    return (
       "You are a project decision analyst. The owner's CURRENT preference can change; "
       "binding evidence and permission requirements always override preferences. "
       "Compare the tradeoff between BOTH options and name the relevant owner preference. "
       "Do not assume A is better. Give ONE short, concrete reason, at most 35 words, "
       "without echoing these instructions or inventing facts.\n"
       +project_context(row,c)+"ANALYSIS:"
    )

def candidate_prompt(row, candidate, advice="",feedback=""):
    choices=row["task"]["choices"]
    return (
       "Decide whether the CANDIDATE below is the best authorized choice for this "
       "hypothetical project, given BOTH options, the CURRENT owner policy, and evidence. "
       "The options' order is arbitrary. If evidence requires an option, obey it, "
       "even when owner preference favors the other. "
       "Output a single Yes or No, no explanation.\n"+
       project_context(row,choices)+
       "CANDIDATE: "+candidate+"\n"+
       ("ADVISORY ANALYSIS (may contain errors): "+advice+"\n" if advice else "")+
       (feedback+"\n" if feedback else "")+
       "Is CANDIDATE the better authorized option? Yes or No?\nAnswer:"
    )

def prefix_ids(tok,prompt):
    return tok.apply_chat_template(
        [{"role":"user","content":prompt}],tokenize=True,add_generation_prompt=True)

def yesno_ids(tok):
    # Single-token tokens; fail loudly rather than silently implementing old A/B bug.
    for a,b in (("Yes","No"),(" Yes"," No")):
        ai=tok.encode(a,add_special_tokens=False)
        bi=tok.encode(b,add_special_tokens=False)
        if len(ai)==len(bi)==1 and ai[0]!=bi[0]:
            return ai[0],bi[0],(a,b)
    raise RuntimeError("No single-token Yes/No pair; revise scorer not silently")

def score_yn(model,tok,prompt,yes_id,no_id):
    ids=prefix_ids(tok,prompt)
    if len(ids)>1250:raise ValueError("Prompt too long")
    with torch.inference_mode():
        logits=model(torch.tensor([ids],dtype=torch.long)).logits[0,-1].float()
    return round(float(logits[yes_id]-logits[no_id]),5)

def score_pair(model,tok,row,yes_id,no_id,advice="",feedback=""):
    c=row["task"]["choices"]
    value=[score_yn(model,tok,candidate_prompt(row,x,advice,feedback),yes_id,no_id) for x in c]
    return {"choice":int(value[1]>value[0]),"candidate_yesno_margins":value,
            "candidate_separation":round(abs(value[1]-value[0]),5)}

def student_eval(model,tok,head,cache,row):
    for c in row["task"]["choices"]:
        key=key_for(row,c)
        if key not in cache:
            h,_=representation(model,tok,row,c,max_tokens=1200)
            cache[key]=h.contiguous()
    with torch.no_grad():
        vals=head(get_two(cache,row)).cpu().float().tolist()
    chosen=int(vals[1]>vals[0])
    return {"choice":chosen,"scores":[round(float(v),5) for v in vals],
            "margin":round(abs(float(vals[1]-vals[0])),5)}

def render_feedback(row,student):
    c=row["task"]["choices"]
    own=row["project"]["policy"]["en"]
    return (
       "FEEDBACK FROM A SEPARATE SMALL PERSONAL PREFERENCE RANKER "
       "(UNVERIFIED; not an authority): "
       "It scored option A "+str(student["scores"][0])+", option B "+
       str(student["scores"][1])+", and prefers \""+c[student["choice"]]+"\". "
       "It was trained to reflect owner preferences but may misread constraints. "
       "Reconsider BOTH the stated CURRENT OWNER PRIORITY and BINDING EVIDENCE. "
       "You may explicitly disagree with the ranker; never adopt its conclusion blindly."
    )

def validate_reason(s):
    lower=s.lower().strip()
    if len(s.strip())<16:return False
    bad=("you are a project decision analyst","project goal:","current approved owner priority")
    if any(lower.startswith(x) for x in bad):return False
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--checkpoint",default="compass-v86-r4-s11.safetensors")
    ap.add_argument("--output",default="compass-v89-english-advisor.json")
    ap.add_argument("--reason-tokens",type=int,default=48)
    args=ap.parse_args()
    metadata=audit();pairs=build()
    torch.set_num_threads(2)
    tokenizer=AutoTokenizer.from_pretrained(BASE)
    yes,no,yn_tokens=yesno_ids(tokenizer)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    head=PersonalRanker(960,4)
    head.load_state_dict(load_file(args.checkpoint,device="cpu"))
    head.eval();head.requires_grad_(False)
    assert sum(p.numel() for p in head.parameters())==7680
    cache={};records=[];start=time.monotonic()
    for i,p in enumerate(pairs,1):
        states=[]
        for side in ("original","counterfactual"):
            row=p[side]
            own=student_eval(model,tokenizer,head,cache,row)
            direct=score_pair(model,tokenizer,row,yes,no)
            prompt=reason_prompt(row)
            ids=prefix_ids(tokenizer,prompt)
            if len(ids)>1150:raise ValueError("Analysis prompt too long")
            with torch.inference_mode():
                generated=model.generate(
                    torch.tensor([ids],dtype=torch.long),
                    max_new_tokens=args.reason_tokens,do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            text=tokenizer.decode(generated[0,len(ids):],skip_special_tokens=True).strip()
            reason=" ".join(text.split())[:500]
            valid=validate_reason(reason)
            with_reason=score_pair(model,tokenizer,row,yes,no,advice=reason) if valid else None
            feedback=render_feedback(row,own)
            revise=score_pair(model,tokenizer,row,yes,no,advice=reason if valid else "",
                             feedback=feedback)
            states.append({
                "small_head":own,"direct_english":direct,"reasoned_english":with_reason,
                "bounded_feedback_english":revise,
                "reason_text":reason,"reason_valid":valid,
                "token_count_generated":int(generated.shape[1]-len(ids)),
                "owner_policy_text":row["project"]["policy"]["en"],
                "choices":row["task"]["choices"],
                "evidence":row["task"]["record"],
            })
        records.append({
           "case":i,"family":p["meta"]["family"],"constraint":p["meta"]["status"],
           "option_swap":p["meta"]["swap"],"project":p["meta"]["project"],
           "expected_flip":p["meta"]["expected_flip"],
           "gold":[p["original_label"]["correct_digit"],
                   p["counterfactual_label"]["correct_digit"]],
           "before":states[0],"after":states[1],
        })
        if i%4==0:
            print(f"TEACHER completed pairs {i}/{len(pairs)}; elapsed={time.monotonic()-start:.1f}s",flush=True)
    result={
      "experiment":"COMPASS V8.9 English-only teacher test with one bounded small-to-large feedback round",
      "frozen_backbone":BASE,"personal_parameters":7680,
      "yesno_tokens":yn_tokens,"fixture":metadata,
      "records":records,"seconds":round(time.monotonic()-start,2),
      "limitations":[
          "All situations and labels are synthetic and researcher authored",
          "Previously seen test projects but novel family text, not independent human validation",
          "The model's natural-language text is not proof of reasoning",
          "No learning or gradient updates for small/large model in this phase",
          "Small-ranker feedback is not authoritative user preference or source truth",
          "Only one feedback round; no production authorization or actual conversation memory"
      ]
    }
    path=Path(args.output)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print("SAVED",path,flush=True)

if __name__=="__main__":main()
