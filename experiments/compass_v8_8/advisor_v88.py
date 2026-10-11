"""V8.8 frozen 360M ACTIVE advisor: generate short analyses, score A/B choices.

No gold labels, oracle, state labels, policy_mode, source-gate metadata or
task IDs enter model prompts. Teacher output is UNTRUSTED and logged; student
and teacher correctness are assessed separately after test isolation.
"""
import copy,hashlib,json,time,argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from fixtures_v88 import build
from cache_v87 import all_rows as old_rows
from cache_v85 import dataset_fingerprint
from ranker_v84 import key_for,representation
from run import BASE

def teacher_key(row):
    """Unordered choices; stable across swap and re-identified test labels."""
    p=row["project"];t=row["task"];lang=row["lang"]
    values=[p["id"],lang,p["objective"][lang],p["policy"][lang],t["record"]]+sorted(t["choices"])
    return hashlib.sha256(json.dumps(values,ensure_ascii=False).encode("utf-8")).hexdigest()

def canonical(row):
    return sorted(row["task"]["choices"])

def advisor_prompt(row):
    lang=row["lang"];p=row["project"];t=row["task"]
    a,b=canonical(row)
    if lang=="he":
        return ("אתה יועץ תיעדוף בפרויקט. הסבר בקצרה כיצד ההעדפה העדכנית של הבעלים והראיות "+
                "משפיעות על ההתאמה של שתי הפעולות. אל תניח שאפשרות א עדיפה רק בגלל מיקומה. "+
                "אל תמציא הרשאות או עובדות. כתוב משפט הסבר קצר אחד בלבד.\n"
                f"יעד: {p['objective'][lang]}\nמדיניות מאושרת: {p['policy'][lang]}\n"
                f"ראיות: {t['record']}\nפעולה A: {a}\nפעולה B: {b}\n"
                "הסבר:")
    return ("You advise project priorities. Analyze how the CURRENT approved owner's "+
            "preference and given facts affect the two options. Do not assume the "+
            "first item is better due to its position. Never invent permissions or "+
            "facts. State one concise reason (not the label).\n"
            f"Objective: {p['objective'][lang]}\nApproved policy: {p['policy'][lang]}\n"
            f"Evidence: {t['record']}\nAction A: {a}\nAction B: {b}\nReason:")

def augmented(row,trace):
    """Inject the active teacher's reasoning + selected action as separate advisory."""
    v=copy.deepcopy(row);task=v["task"]
    text=("Advisor's generated nonauthoritative analysis: "+trace["reason"]+
          "\nAdvisor's suggested action, to verify independently: "+trace["choice_text"])
    task["record"]=task["record"]+"\n"+text
    # Must represent the same user preference in exactly the same way after swapping options.
    task["record_source_id"]="TCHR-"+teacher_key(row)[:24]
    return v

def all_pairs():
    inp,gold,training,exposed,exposed_cold,newcold=build()
    return {"train":training,"exposed_v86":exposed,
            "exposed_v87":exposed_cold,"cold_v88":newcold}

def all_teacher_rows(pairs):
    return [p[k] for group in pairs.values() for p in group
            for k in ("original","counterfactual")]

def rows_for_cache():
    pairs=all_pairs()
    return old_rows()+[p[k] for p in pairs["cold_v88"]
                       for k in ("original","counterfactual")]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--reuse",default="v87-embeddings.pt")
    ap.add_argument("--out",default="v88-active-teacher.pt")
    ap.add_argument("--max-reason-tokens",type=int,default=28)
    args=ap.parse_args()
    old=torch.load(args.reuse,map_location="cpu",weights_only=True)
    assert old["source_fixture"]=="v87"
    assert old["fingerprint"]==dataset_fingerprint(old_rows())
    groups=all_pairs()
    unique={}
    for row in all_teacher_rows(groups):
        unique[teacher_key(row)]=row
    # 72 synthetic cases × 2 opposite policy settings; allow other partitions.
    print(f"ACTIVE TEACHER unique reasoning contexts={len(unique)}",flush=True)
    assert len(unique)>=70
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(BASE)
    model=AutoModelForCausalLM.from_pretrained(BASE,torch_dtype=torch.float32)
    model.eval();model.requires_grad_(False)
    ids={v:tok.encode(v,add_special_tokens=False) for v in (" A"," B")}
    if any(len(x)!=1 for x in ids.values()):
        raise RuntimeError("Teacher choice A/B must be single token for comparable logit scoring")
    a_id=ids[" A"][0];b_id=ids[" B"][0]
    initial=time.monotonic()
    traces={}
    for i,(key,row) in enumerate(sorted(unique.items()),1):
        prompt=advisor_prompt(row)
        seq=tok.apply_chat_template([{"role":"user","content":prompt}],
                                     tokenize=True,add_generation_prompt=True)
        tokens=torch.tensor([seq],dtype=torch.long)
        if len(seq)>1150:raise ValueError("Teacher prompt exceeds 1150 token limit")
        with torch.inference_mode():
            result=model.generate(tokens,max_new_tokens=args.max_reason_tokens,
                                  do_sample=False,pad_token_id=tok.eos_token_id)
        reason=tok.decode(result[0,len(seq):],skip_special_tokens=True).strip()
        reason=" ".join(reason.split())[:320]
        if not reason:reason="No usable explanation generated"
        # Deliberately compute forced-choice confidence AFTER generated rationale.
        # The teacher is free to disagree with synthetic oracle/gold.
        vote_prompt=prompt+"\nExplanation: "+reason+"\nConsidering ONLY the current approved preference and evidence, the better action is"
        vote_seq=tok.apply_chat_template([{"role":"user","content":vote_prompt}],
                                          tokenize=True,add_generation_prompt=True)
        with torch.inference_mode():
            logits=model(torch.tensor([vote_seq],dtype=torch.long)).logits[0,-1].float()
        choice=0 if logits[a_id]>=logits[b_id] else 1
        probability=float(torch.softmax(torch.stack((logits[a_id],logits[b_id])),dim=0)[choice])
        options=canonical(row)
        traces[key]={"reason":reason,"suggested_canonical":choice,
                     "choice_text":options[choice],
                     "choice_probability":round(probability,5),
                     "lang":row["lang"],"generated_length":len(result[0])-len(seq)}
        if i%20==0:print(f"ADVISOR generated {i}/{len(unique)} elapsed={time.monotonic()-initial:.1f}s",flush=True)
    # Keep all plain control representations, add only augmented source representations.
    features=dict(old["features"]);frozen=dict(old["frozen_yesno"])
    rows=rows_for_cache()
    for row in rows:
        for cand in row["task"]["choices"]:
            if key_for(row,cand) not in features:
                hidden,nt=representation(model,tok,row,cand,max_tokens=1200)
                features[key_for(row,cand)]=hidden.contiguous()
                frozen[key_for(row,cand)]=0.0 # NOT USED AS MATCHED SCORE FOR THIS EXTRA RAW BENCHMARK
    to_add={}
    for row in all_teacher_rows(groups):
        alt=augmented(row,traces[teacher_key(row)])
        for cand in alt["task"]["choices"]:
            k=key_for(alt,cand)
            if k not in features:to_add[k]=(alt,cand)
    print(f"ACTIVE ADVISOR augmented feature contexts={len(to_add)}",flush=True)
    for i,(k,(row,cand)) in enumerate(to_add.items(),1):
        feat,n=representation(model,tok,row,cand,max_tokens=1200)
        features[k]=feat.contiguous()
        if i%80==0:print(f"ADVISOR student features {i}/{len(to_add)} elapsed={time.monotonic()-initial:.1f}s",flush=True)
    out=Path(args.out)
    payload={
        "source_fixture":"v88","backbone":BASE,
        "source_fingerprint":old["fingerprint"],
        "teacher_contexts":len(traces),"teacher":traces,
        "features":features,"frozen_yesno":frozen,
        "new_augmented":len(to_add),
        "elapsed_seconds":round(time.monotonic()-initial,2)
    }
    torch.save(payload,out)
    public={k:v for k,v in payload.items() if k not in ("features","frozen_yesno","teacher")}
    public["sample_teacher_outputs"]=list(traces.values())[:4]
    out.with_suffix(".json").write_text(json.dumps(public,ensure_ascii=False,indent=2),encoding="utf8")
    print("SAVED",out,public,flush=True)
if __name__=="__main__":main()
