"""
Stage-1 Hebrew pretraining using neutral PersonalState.

Usage:
  python train_tokenizer.py corpus.txt
  python train_hebrew.py corpus.txt --tokenizer hebrew-bpe-4096.json

Stage-2 personalization should then mix:
  A) neutral general-language batches
  B) same-prompt/different-state contrastive batches
  C) different-state/same-answer irrelevance batches
"""
import argparse
from pathlib import Path
import random
import torch

from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState
from bpe_tokenizer import BPETokenizer

def make_blocks(ids, seq_len):
    usable = len(ids) - (len(ids) % (seq_len + 1))
    for i in range(0, usable, seq_len + 1):
        b = ids[i:i + seq_len + 1]
        if len(b) == seq_len + 1:
            yield b[:-1], b[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="hebrew-lm-5m.pt")
    args = ap.parse_args()

    tok = BPETokenizer.load(args.tokenizer)
    cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    text = Path(args.text_file).read_text(encoding="utf-8")
    ids = tok.encode(text)
    blocks = list(make_blocks(ids, args.seq_len))
    if not blocks:
        raise SystemExit("Corpus is too small for selected seq_len.")

    model = PersonalNativeLM(cfg).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters={params:,}")
    print(f"tokens={len(ids):,} blocks={len(blocks):,} device={args.device}")

    neutral = PersonalState("neutral").flatten().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    step = 0
    model.train()
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
            if step == 1 or step % 100 == 0:
                print(f"epoch={epoch+1} step={step} loss={loss.item():.4f}")

    torch.save({
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "tokenizer_file": args.tokenizer,
        "stage": "general-hebrew-pretraining",
    }, args.save)
    print(f"saved={args.save}")

if __name__ == "__main__":
    main()
