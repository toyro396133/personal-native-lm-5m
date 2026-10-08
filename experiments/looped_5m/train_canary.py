"""Controlled, from-scratch looped 5M training canary on a fixed text split."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from looped_model import LoopedPersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer


def evaluate(model, ids, seq_len, neutral, max_blocks=12):
    model.eval()
    total_nll, tokens = 0.0, 0
    with torch.no_grad():
        for i in range(0, min(len(ids) - 1, max_blocks * seq_len), seq_len):
            x = torch.tensor(ids[i:i + seq_len], dtype=torch.long).unsqueeze(0)
            y = torch.tensor(ids[i + 1:i + seq_len + 1], dtype=torch.long).unsqueeze(0)
            if x.shape != y.shape or x.numel() == 0:
                continue
            logits, _ = model(x, neutral)
            total_nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
            tokens += y.numel()
    model.train()
    return total_nll / tokens if tokens else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True)
    p.add_argument("--val", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--loops", type=int, default=1)
    p.add_argument("--injection", type=float, default=0.0)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--output", required=True)
    p.add_argument("--eval-every", type=int, default=0)
    p.add_argument("--checkpoint-dir", default="")
    args = p.parse_args()
    if args.steps < 1:
        p.error("--steps must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    tok, kind = load_tokenizer(args.tokenizer)
    cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    model = LoopedPersonalNativeLM(cfg, loops=args.loops, input_injection=args.injection)
    neutral = PersonalState("neutral").flatten().unsqueeze(0)
    train_ids = tok.encode(Path(args.train).read_text(encoding="utf-8"))
    val_ids = tok.encode(Path(args.val).read_text(encoding="utf-8"))
    starts = list(range(0, len(train_ids) - args.seq_len - 1, args.seq_len + 1))
    if len(starts) < args.batch_size or len(val_ids) <= args.seq_len:
        raise ValueError("Corpus too small for selected sequence length and batch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
    initial = evaluate(model, val_ids, args.seq_len, neutral)
    start_time = time.monotonic()
    losses = []
    milestones = []
    model.train()
    for step in range(args.steps):
        # Fixed independent RNG: identical sampled offsets for all loop configurations.
        selected = [starts[random.randrange(len(starts))] for _ in range(args.batch_size)]
        x = torch.tensor([train_ids[i:i + args.seq_len] for i in selected], dtype=torch.long)
        y = torch.tensor([train_ids[i + 1:i + args.seq_len + 1] for i in selected], dtype=torch.long)
        _, loss = model(x, neutral.expand(args.batch_size, -1), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if args.eval_every and ((step + 1) % args.eval_every == 0 or step + 1 == args.steps):
            val = evaluate(model, val_ids, args.seq_len, neutral)
            milestones.append({"step": step + 1, "tokens": (step + 1) * args.batch_size * args.seq_len, "val_nats_per_token": val})
            print(f"milestone loop={args.loops} step={step + 1} val={val:.6f}", flush=True)
            if args.checkpoint_dir:
                folder = Path(args.checkpoint_dir)
                folder.mkdir(parents=True, exist_ok=True)
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": cfg.__dict__, "loops": args.loops, "injection": args.injection, "step": step + 1, "seed": args.seed}, folder / f"step-{step + 1}.pt")
        elif step == 0 or (step + 1) % 25 == 0:
            print(f"loop={args.loops} step={step + 1}/{args.steps} loss={losses[-1]:.5f}", flush=True)
    final = evaluate(model, val_ids, args.seq_len, neutral)
    report = {
        "loop_passes": args.loops, "input_injection": args.injection,
        "seed": args.seed, "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_steps": args.steps, "train_tokens": args.steps * args.batch_size * args.seq_len,
        "train_first_loss": losses[0], "train_last_loss": losses[-1],
        "heldout_initial_nats_per_token": initial, "heldout_final_nats_per_token": final,
        "heldout_delta_nats_per_token": final - initial,
        "elapsed_seconds": time.monotonic() - start_time,
        "tokenizer_kind": kind, "milestones": milestones,
        "warning": "Small canary only. No SELF attribution; not equal compute.",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
