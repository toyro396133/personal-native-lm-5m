"""Longitudinal synthetic-private-memory experiment on a frozen English LM.

Three staged releases of fictional personal facts; tests held-out question phrasings.
Reports base / trained adapter / full-context retrieval baseline each stage.
No real user data or remote telemetry. Base is never updated.
"""
import argparse
import json
import random
from pathlib import Path
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from run import PersonalAdapter, attach, BASE

ROOT = Path(__file__).resolve().parent
FACTS = [
    ("My preferred planning format is a three-item checklist.", "What planning format do I prefer?", "three-item checklist", "long essay", "Which planning format suits me?", "three-item checklist", "weekly calendar"),
    ("My fictional project codename is Copper Finch.", "What is my project codename?", "Copper Finch", "Silver Harbor", "Remind me of the codename I chose.", "Copper Finch", "Blue Lantern"),
    ("I prefer study sessions of 25 minutes.", "How long are my preferred study sessions?", "25 minutes", "90 minutes", "What is my usual study session length?", "25 minutes", "45 minutes"),
    ("The personal dashboard I am designing uses amber accents.", "Which accent color does my dashboard use?", "amber", "violet", "What accent did I select for my dashboard?", "amber", "turquoise"),
    ("I call my note archive the Lantern Vault.", "What is my note archive called?", "Lantern Vault", "Marble Shelf", "What name did I give my notes archive?", "Lantern Vault", "Atlas Box"),
    ("For status updates I prefer one short paragraph.", "How should status updates be formatted for me?", "one short paragraph", "five long sections", "What format should my progress report use?", "one short paragraph", "a table only"),
]
GROUPS = [FACTS[:2], FACTS[2:4], FACTS[4:]]
# Unseen phrasing per fact; candidates all paired with plausible alternatives.


def load():
    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return tok, model


def prompt(tok, question, context=""):
    user = (context + "\n" if context else "") + "Answer briefly: " + question
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=True, add_generation_prompt=True)


def nll(tok, model, question, answer, context="", max_length=192):
    head = prompt(tok, question, context)
    tail = tok.encode(answer, add_special_tokens=False)
    head = head[-(max_length - len(tail)):]
    ids = torch.tensor([head + tail])
    labels = torch.tensor([[-100]*len(head) + tail])
    output = model(input_ids=ids, labels=labels)
    # loss is average token CE; convert to sum to avoid candidate length bias
    return float(output.loss.detach()) * len(tail)


def evaluate(tok, model, seen, adapter_handle=None):
    items = [x for stage in GROUPS[:seen] for x in stage]
    def score(context):
        wins = 0
        details = []
        with torch.no_grad():
            for statement, _, _, _, question, right, wrong in items:
                a = nll(tok, model, question, " " + right, context)
                b = nll(tok, model, question, " " + wrong, context)
                win = a < b
                wins += win
                details.append({"question": question, "correct": right, "wrong": wrong,
                                "correct_nll": round(a, 3), "wrong_nll": round(b, 3), "pass": win})
        return {"accuracy": wins/len(items), "correct": wins, "total": len(items), "details": details}
    return score("")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps-per-stage", type=int, default=12)
    p.add_argument("--rank", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--output", default=str(ROOT / "longitudinal-results.json"))
    args = p.parse_args()
    torch.manual_seed(42)
    random.seed(42)
    torch.set_num_threads(2)
    tok, model = load()
    adapter = PersonalAdapter(model.config.hidden_size, args.rank)
    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    print("personal trainable parameters:", sum(p.numel() for p in adapter.parameters()), flush=True)
    report = {"model": BASE, "seed": 42, "steps_per_stage": args.steps_per_stage, "stages": []}
    for stage_num, newfacts in enumerate(GROUPS, 1):
        allfacts = [x for stage in GROUPS[:stage_num] for x in stage]
        # Evaluate base without hook and retrieve with full explicit personal facts.
        base = evaluate(tok, model, stage_num)
        context = "Personal notes (fictional):\n" + "\n".join(f"- {f[0]}" for f in allfacts)
        retrieval_pass = 0
        with torch.no_grad():
            for fact in allfacts:
                question, right, wrong = fact[4], fact[5], fact[6]
                retrieval_pass += nll(tok, model, question, " " + right, context) < nll(tok, model, question, " " + wrong, context)
        retrieval = {"correct": retrieval_pass, "total": len(allfacts), "accuracy": retrieval_pass/len(allfacts)}
        hook = attach(model, adapter, 1)
        losses = []
        try:
            # Cumulative rehearsal prevents intentional forgetting of earlier facts.
            for i in range(args.steps_per_stage):
                fact = allfacts[i % len(allfacts)]
                question = fact[1]
                answer = " " + fact[2]
                head = prompt(tok, question)
                tail = tok.encode(answer, add_special_tokens=False)
                ids = torch.tensor([head+tail])
                labels = torch.tensor([[-100]*len(head)+tail])
                opt.zero_grad(set_to_none=True)
                loss = model(input_ids=ids, labels=labels).loss
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()
                losses.append(round(loss.item(), 4))
            personal = evaluate(tok, model, stage_num)
        finally:
            hook.remove()
        entry = {"stage": stage_num, "facts_seen": len(allfacts), "train_losses": losses,
                 "base": base, "personal": personal, "retrieval": retrieval}
        report["stages"].append(entry)
        print(f"stage={stage_num} loss={losses[0]:.3f}->{losses[-1]:.3f} "
              f"base={base['correct']}/{base['total']} personal={personal['correct']}/{personal['total']} "
              f"retrieval={retrieval['correct']}/{retrieval['total']}", flush=True)
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("report:", args.output, flush=True)


if __name__ == "__main__":
    main()
