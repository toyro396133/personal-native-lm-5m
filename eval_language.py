from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import torch

from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer, make_blocks


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = PersonalNativeLM(cfg).to(device)
    load = model.load_state_dict(ckpt["model"], strict=False)
    if load.missing_keys or load.unexpected_keys:
        print(f"load missing={len(load.missing_keys)} unexpected={len(load.unexpected_keys)}")
    model.eval()
    return model, ckpt


@torch.no_grad()
def evaluate(model, tok, text, device, seq_len=128, max_blocks=256):
    ids = tok.encode(text)
    blocks = list(make_blocks(ids, seq_len))[:max_blocks]
    if not blocks:
        raise ValueError("validation corpus too small")
    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(device)
    losses = []
    for x_ids, y_ids in blocks:
        x = torch.tensor([x_ids], dtype=torch.long, device=device)
        y = torch.tensor([y_ids], dtype=torch.long, device=device)
        _, loss = model(x, neutral, y)
        losses.append(float(loss))
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20)), "blocks": len(losses), "tokens": len(blocks) * seq_len}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("validation_text")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--max-blocks", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--json-out")
    args = ap.parse_args()

    tok, kind = load_tokenizer(args.tokenizer)
    model, ckpt = load_model(args.checkpoint, args.device)
    text = Path(args.validation_text).read_text(encoding="utf-8")
    metrics = evaluate(model, tok, text, args.device, args.seq_len, args.max_blocks)
    metrics.update({"checkpoint": args.checkpoint, "tokenizer_kind": kind, "stage": ckpt.get("stage")})
    print(json.dumps(metrics, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
