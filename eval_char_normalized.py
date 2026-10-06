from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer


@torch.no_grad()
def evaluate(checkpoint, tokenizer_path, text, device, seq_len=128):
    tok, kind = load_tokenizer(tokenizer_path)
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    model = PersonalNativeLM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    ids = tok.encode(text, bos=True, eos=True)
    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(device)

    total_nll = 0.0
    target_tokens = 0
    # Non-overlapping contiguous blocks; each block predicts its next token.
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            continue
        x_ids = chunk[:-1]
        y_ids = chunk[1:]
        x = torch.tensor([x_ids], dtype=torch.long, device=device)
        y = torch.tensor([y_ids], dtype=torch.long, device=device)
        logits, _ = model(x, neutral)
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss_sum)
        target_tokens += len(y_ids)

    chars = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    nats_per_char = total_nll / max(1, chars)
    bits_per_char = nats_per_char / math.log(2)
    nats_per_utf8_byte = total_nll / max(1, utf8_bytes)
    return {
        "checkpoint": checkpoint,
        "tokenizer": tokenizer_path,
        "tokenizer_kind": kind,
        "raw_chars": chars,
        "raw_utf8_bytes": utf8_bytes,
        "encoded_tokens": len(ids),
        "evaluated_target_tokens": target_tokens,
        "tokens_per_100_chars": len(ids) / max(1, chars) * 100,
        "total_nll": total_nll,
        "nats_per_char": nats_per_char,
        "bits_per_char": bits_per_char,
        "nats_per_utf8_byte": nats_per_utf8_byte,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("tokenizer")
    ap.add_argument("text_file")
    ap.add_argument("--chars",type=int,default=300000)
    ap.add_argument("--seq-len",type=int,default=128)
    ap.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out")
    args=ap.parse_args()

    text=Path(args.text_file).read_text(encoding="utf-8")[:args.chars]
    result=evaluate(args.checkpoint,args.tokenizer,text,args.device,args.seq_len)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
