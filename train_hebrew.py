"""Stage-1 Hebrew pretraining using neutral PersonalState.

Supports:
- dependency-free hybrid Hebrew tokenizer;
- Unicode-safe SentencePiece tokenizer;
- optional external ByteLevel BPE tokenizer;
- controlled 5M/10M capacity profiles.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import random
import torch

from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState


def make_blocks(ids, seq_len):
    usable = len(ids) - (len(ids) % (seq_len + 1))
    for i in range(0, usable, seq_len + 1):
        b = ids[i:i + seq_len + 1]
        if len(b) == seq_len + 1:
            yield b[:-1], b[1:]


def load_tokenizer(path):
    path = str(path)
    if path.endswith(".model"):
        from sentencepiece_tokenizer import SentencePieceTokenizer
        return SentencePieceTokenizer.load(path), "sentencepiece-unicode"
    try:
        from hybrid_tokenizer import HybridHebrewTokenizer
        return HybridHebrewTokenizer.load(path), "hybrid-hebrew-v1"
    except Exception:
        from bpe_tokenizer import BPETokenizer
        return BPETokenizer.load(path), "bytelevel-bpe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--profile", choices=["5m", "10m"], default="5m")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="hebrew-lm.pt")
    ap.add_argument("--max-steps", type=int, default=0, help="0 means no explicit step limit")
    ap.add_argument("--seed", type=int, default=71)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    if args.profile == "10m":
        cfg = ModelConfig.hebrew_bpe_10m(tok.vocab_size)
    else:
        cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    text = Path(args.text_file).read_text(encoding="utf-8")
    ids = tok.encode(text)
    blocks = list(make_blocks(ids, args.seq_len))
    if not blocks:
        raise SystemExit("Corpus is too small for selected seq_len.")

    model = PersonalNativeLM(cfg).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"profile={args.profile} parameters={params:,}")
    print(f"tokens={len(ids):,} blocks={len(blocks):,} device={args.device} tokenizer={tokenizer_kind}")

    neutral = PersonalState("neutral").flatten().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    step = 0
    first_loss = None
    last_loss = None
    model.train()
    stop = False
    for epoch in range(args.epochs):
        random.shuffle(blocks)
        for j in range(0, len(blocks), args.batch_size):
            batch = blocks[j:j + args.batch_size]
            if len(batch) < 2:
                continue
            x = torch.tensor([b[0] for b in batch], dtype=torch.long, device=args.device)
            y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=args.device)
            s = neutral.unsqueeze(0).expand(x.size(0), -1)

            _, loss = model(x, s, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            last_loss = float(loss.detach())
            if first_loss is None:
                first_loss = last_loss
            if step == 1 or step % 50 == 0:
                print(f"epoch={epoch+1} step={step} loss={last_loss:.4f}")
            if args.max_steps and step >= args.max_steps:
                stop = True
                break
        if stop:
            break

    torch.save({
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "tokenizer_file": args.tokenizer,
        "tokenizer_kind": tokenizer_kind,
        "model_profile": args.profile,
        "parameter_count": params,
        "stage": "general-hebrew-pretraining",
        "steps": step,
        "first_loss": first_loss,
        "last_loss": last_loss,
    }, args.save)
    print(f"steps={step} first_loss={first_loss:.4f} last_loss={last_loss:.4f}")
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
