from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import torch

from config import ModelConfig
from model import PersonalNativeLM
from personal_model import PersonalLearningSystem
from personal_state import PersonalState
from longitudinal_data import make_profiles
from train_hebrew import load_tokenizer
import train_longitudinal as tl

DEFAULT_PROMPTS = [
    "ישראל היא מדינה",
    "המחשב יכול",
    "בשנים האחרונות",
    "המחקר מראה כי",
    "כאשר אנשים לומדים מיומנות חדשה",
]


def load_checkpoint(path, tokenizer_path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    tok, tokenizer_kind = load_tokenizer(tokenizer_path)
    tok.PAD = tok.pad_id
    lm = PersonalNativeLM(cfg).to(device)
    lm.load_state_dict(ckpt["model"], strict=False)
    lm.eval()
    return ckpt, cfg, tok, tokenizer_kind, lm


@torch.no_grad()
def generate(lm, cfg, tok, prompt, state, max_new_tokens=48, temperature=0.0, seed=17):
    ids = tok.encode(prompt, bos=True, eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=state.device)
    gen = torch.Generator(device=state.device)
    gen.manual_seed(seed)
    generated = []
    for _ in range(max_new_tokens):
        logits, _ = lm(x[:, -cfg.max_seq_len:], state)
        last = logits[0, -1]
        if temperature <= 0:
            nxt = int(last.argmax())
        else:
            probs = torch.softmax(last / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        generated.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=x.device)], dim=1)
        if nxt == tok.eos_id:
            break
    text = tok.decode(generated)
    unk_count = sum(i == getattr(tok, "unk_id", -999999) for i in generated)
    return {
        "text": text,
        "token_ids": generated,
        "replacement_char_count": text.count("\ufffd"),
        "unk_token_count": unk_count,
    }


@torch.no_grad()
def personal_demo(ckpt, cfg, tok, lm, device):
    if "personal_learning_system" not in ckpt:
        return None
    personal = PersonalLearningSystem(cfg).to(device)
    personal.load_state_dict(ckpt["personal_learning_system"], strict=False)
    personal.eval()
    profiles = make_profiles()[:4]
    tl.TOK = tok
    hids, htm, hem = tl.encode_histories([p.history for p in profiles], device)
    _, states = personal.from_history(hids, htm, hem)

    rows = []
    for i, profile in enumerate(profiles):
        user = {"user_id":profile.user_id,"targets":{"core":profile.core,"policy":profile.policy,"world":profile.world},"queries":[]}
        for kind, q in tl.PROMPTS.items():
            inp = torch.tensor([tok.encode(q,bos=True,eos=False)],dtype=torch.long,device=device)
            cond = torch.tensor([tok.encode(q,bos=False,eos=False)],dtype=torch.long,device=device)
            logits,_ = lm(inp,states[i:i+1],condition_ids=cond)
            pred=tl.predict_label(logits[0,-1])
            qemb=lm.token_embedding(cond)
            _,gates,relevance=lm.controller(states[i:i+1],qemb,return_routing=True)
            routes=[float(x) for x in gates[0]]+[float(1-relevance[0])]
            user["queries"].append({
                "kind":kind,"input":q,"output":pred,
                "routes":{"core":routes[0],"policy":routes[1],"world":routes[2],"routing":routes[3],"none":routes[4]}
            })
        rows.append(user)

    if "example_adapted_micro_latent" in ckpt:
        profile=profiles[0]
        hids,htm,hem=tl.encode_histories([profile.history],device)
        _,before_state=personal.from_history(hids,htm,hem)
        after_state=personal.state_decoder(ckpt["example_adapted_micro_latent"].to(device).unsqueeze(0))
        def preds(state):
            out={}
            for kind,q in tl.PROMPTS.items():
                inp=torch.tensor([tok.encode(q,bos=True,eos=False)],dtype=torch.long,device=device)
                cond=torch.tensor([tok.encode(q,bos=False,eos=False)],dtype=torch.long,device=device)
                logits,_=lm(inp,state,condition_ids=cond)
                out[kind]=tl.predict_label(logits[0,-1])
            return out
        adapted={"before":preds(before_state),"after":preds(after_state)}
    else:
        adapted=None
    return {"users":rows,"frozen_adaptation":adapted}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--out",default="demo_inference.json")
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()

    random.seed(17); torch.manual_seed(17)
    ckpt,cfg,tok,kind,lm=load_checkpoint(args.checkpoint,args.tokenizer,args.device)
    neutral=PersonalState("neutral").flatten().unsqueeze(0).to(args.device)

    language=[]
    for prompt in DEFAULT_PROMPTS:
        greedy=generate(lm,cfg,tok,prompt,neutral,temperature=0.0)
        sampled=generate(lm,cfg,tok,prompt,neutral,temperature=0.75,seed=17)
        language.append({
            "input":prompt,
            "greedy_output":greedy["text"],
            "sampled_output":sampled["text"],
            "greedy_replacement_chars":greedy["replacement_char_count"],
            "sampled_replacement_chars":sampled["replacement_char_count"],
            "greedy_unk_tokens":greedy["unk_token_count"],
            "sampled_unk_tokens":sampled["unk_token_count"],
        })
    report={
        "checkpoint":args.checkpoint,
        "stage":ckpt.get("stage"),
        "tokenizer_kind":kind,
        "language_examples":language,
        "replacement_chars_total":sum(x["greedy_replacement_chars"]+x["sampled_replacement_chars"] for x in language),
        "unk_tokens_total":sum(x["greedy_unk_tokens"]+x["sampled_unk_tokens"] for x in language),
        "personal_examples":personal_demo(ckpt,cfg,tok,lm,args.device),
    }
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__":
    main()
