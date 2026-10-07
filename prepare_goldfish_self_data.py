from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

def bucket(text: str) -> int:
    return int.from_bytes(hashlib.sha1(text.encode("utf-8")).digest()[:4], "big") % 1000

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="goldfish-models/heb_hebr_1000mb")
    ap.add_argument("--revision", default="425eee09d1b9ff32ffa21d6cfb300a74302a0799")
    ap.add_argument("--train-tokens", type=int, default=1_050_000)
    ap.add_argument("--val-tokens", type=int, default=64_000)
    ap.add_argument("--seed", type=int, default=1701)
    ap.add_argument("--out-dir", default="data/goldfish-self")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    if bos is None or eos is None:
        raise RuntimeError("Goldfish tokenizer must expose BOS/CLS and EOS/SEP IDs")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.i32"
    val_path = out / "val.i32"

    ds = load_dataset("HuggingFaceFW/fineweb-2", name="heb_Hebr", split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=5000)

    seen = set()
    train_n = val_n = docs = duplicates = 0
    train_docs = val_docs = 0
    train_hash = hashlib.sha256()
    val_hash = hashlib.sha256()

    with train_path.open("wb") as ft, val_path.open("wb") as fv:
        for row in ds:
            text = str(row.get("text", "")).strip()
            if len(text) < 120:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).digest()
            if digest in seen:
                duplicates += 1
                continue
            seen.add(digest)
            docs += 1

            ids = [int(bos)] + tok.encode(text, add_special_tokens=False) + [int(eos)]
            is_val = bucket("goldfish-self:" + text) < 60

            if is_val and val_n < args.val_tokens:
                take = ids[: args.val_tokens - val_n]
                if take:
                    raw = array("i", take).tobytes()
                    fv.write(raw)
                    val_hash.update(raw)
                    val_n += len(take)
                    val_docs += 1
            elif (not is_val) and train_n < args.train_tokens:
                take = ids[: args.train_tokens - train_n]
                if take:
                    raw = array("i", take).tobytes()
                    ft.write(raw)
                    train_hash.update(raw)
                    train_n += len(take)
                    train_docs += 1

            if docs % 2000 == 0:
                print(f"docs={docs:,} train={train_n:,} val={val_n:,}", flush=True)
            if train_n >= args.train_tokens and val_n >= args.val_tokens:
                break

    if train_n < args.train_tokens or val_n < args.val_tokens:
        raise RuntimeError(f"stream ended early train={train_n} val={val_n}")

    report = {
        "model_id": args.model,
        "revision": args.revision,
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "heb_Hebr",
        "shuffle_seed": args.seed,
        "strategy": "streamed shuffled Hebrew FineWeb2; global document dedupe; hash-separated validation; Goldfish BOS/EOS at document boundaries",
        "train_tokens": train_n,
        "val_tokens": val_n,
        "train_documents": train_docs,
        "val_documents": val_docs,
        "documents_seen": docs,
        "duplicates_rejected": duplicates,
        "tokenizer_vocab": len(tok),
        "bos_id": int(bos),
        "eos_id": int(eos),
        "train_sha256": train_hash.hexdigest(),
        "val_sha256": val_hash.hexdigest(),
    }
    (out / "data_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
