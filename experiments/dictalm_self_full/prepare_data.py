from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


def split_bucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 1000


def append_tokens(buf, ids, eos_id):
    buf.extend(ids)
    if eos_id is not None:
        buf.append(eos_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="dicta-il/DictaLM-3.0-1.7B-Base")
    ap.add_argument("--target-train-tokens", type=int, default=2_000_000)
    ap.add_argument("--target-val-tokens", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=217)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--out-dir", default="/kaggle/working/data")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    eos = tok.eos_token_id

    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        "heb_Hebr",
        split="train",
        streaming=True,
    ).shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    train = []
    val = []
    docs = train_docs = val_docs = 0

    for row in ds:
        text = (row.get("text") or "").strip()
        if not text:
            continue
        docs += 1
        ids = tok.encode(text, add_special_tokens=False)
        if not ids:
            continue

        # Deterministic document-level holdout.
        if split_bucket(text) < 10 and len(val) < args.target_val_tokens:
            append_tokens(val, ids, eos)
            val_docs += 1
        elif len(train) < args.target_train_tokens:
            append_tokens(train, ids, eos)
            train_docs += 1

        if len(train) >= args.target_train_tokens and len(val) >= args.target_val_tokens:
            break

        if docs % 500 == 0:
            print(
                f"docs={docs:,} train_tokens={len(train):,}/{args.target_train_tokens:,} "
                f"val_tokens={len(val):,}/{args.target_val_tokens:,}",
                flush=True,
            )

    train = np.asarray(train[:args.target_train_tokens], dtype=np.int32)
    val = np.asarray(val[:args.target_val_tokens], dtype=np.int32)
    train.tofile(out / "train.i32")
    val.tofile(out / "val.i32")

    report = {
        "model": args.model,
        "dataset": "HuggingFaceFW/fineweb-2/heb_Hebr",
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "split_rule": "sha1(text) % 1000 < 10 => validation",
        "train_tokens": int(train.size),
        "val_tokens": int(val.size),
        "documents_seen": docs,
        "train_documents": train_docs,
        "val_documents": val_docs,
        "tokenizer_vocab_size": len(tok),
        "eos_token_id": eos,
    }
    (out / "data-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
