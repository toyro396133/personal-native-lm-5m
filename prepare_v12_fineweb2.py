from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from datasets import load_dataset
from train_hebrew import load_tokenizer

HEB = re.compile(r"[\u0590-\u05FF]")
LETTER = re.compile(r"[A-Za-z\u0590-\u05FF]")
SPACE = re.compile(r"[ \t]+")


def clean_chunk(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\x00", " ")
    text = SPACE.sub(" ", text).strip()
    return text


def good_chunk(text: str) -> bool:
    if len(text) < 80 or len(text) > 3000:
        return False
    letters = LETTER.findall(text)
    if len(letters) < 40:
        return False
    heb = HEB.findall(text)
    return len(heb) / max(1, len(letters)) >= 0.72


def chunks(text: str):
    for part in re.split(r"\n\s*\n|\n", text):
        part = clean_chunk(part)
        if good_chunk(part):
            yield part


def bucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--config", default="heb_Hebr")
    ap.add_argument("--train-tokens", type=int, default=10_000_000)
    ap.add_argument("--val-tokens", type=int, default=500_000)
    ap.add_argument("--val-per-mille", type=int, default=50)
    ap.add_argument("--seed", type=int, default=121)
    ap.add_argument("--shuffle-buffer", type=int, default=5000)
    ap.add_argument("--train-out", default="data/v12-fineweb2-train.txt")
    ap.add_argument("--val-out", default="data/v12-fineweb2-val.txt")
    ap.add_argument("--report", default="artifacts/v12_fineweb2_report.json")
    args = ap.parse_args()

    tok, kind = load_tokenizer(args.tokenizer)
    ds = load_dataset(args.dataset, name=args.config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    train_path = Path(args.train_out)
    val_path = Path(args.val_out)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    train_tokens = val_tokens = 0
    docs_seen = chunks_seen = accepted = duplicates = 0
    bad_language = unknown_rejected = 0
    seen = set()

    with train_path.open("w", encoding="utf-8") as ft, val_path.open("w", encoding="utf-8") as fv:
        for row in ds:
            docs_seen += 1
            text = str(row.get("text", ""))
            for chunk in chunks(text):
                chunks_seen += 1
                digest = hashlib.sha1(chunk.encode("utf-8")).digest()
                if digest in seen:
                    duplicates += 1
                    continue
                seen.add(digest)

                ids = tok.encode(chunk, bos=False, eos=False)
                if not ids:
                    bad_language += 1
                    continue
                unk = getattr(tok, "unk_id", -1)
                if unk >= 0 and any(i == unk for i in ids):
                    unknown_rejected += 1
                    continue

                n = len(ids) + 1
                is_val = bucket("v12:" + chunk) < args.val_per_mille

                if is_val:
                    if val_tokens >= args.val_tokens:
                        continue
                    fv.write(chunk + "\n")
                    val_tokens += n
                else:
                    if train_tokens >= args.train_tokens:
                        continue
                    ft.write(chunk + "\n")
                    train_tokens += n
                accepted += 1

                if train_tokens >= args.train_tokens and val_tokens >= args.val_tokens:
                    break
            if train_tokens >= args.train_tokens and val_tokens >= args.val_tokens:
                break

    report = {
        "version": "0.12",
        "dataset": args.dataset,
        "config": args.config,
        "tokenizer_kind": kind,
        "strategy": "streamed FineWeb2 Hebrew; Hebrew-heavy chunks; deduplicated; reject tokenizer UNK",
        "targets": {"train_tokens": args.train_tokens, "validation_tokens": args.val_tokens},
        "actual": {"train_tokens": train_tokens, "validation_tokens": val_tokens},
        "documents_seen": docs_seen,
        "chunks_seen": chunks_seen,
        "accepted_chunks": accepted,
        "duplicate_chunks": duplicates,
        "bad_language_chunks": bad_language,
        "unknown_token_chunks_rejected": unknown_rejected,
        "validation_per_mille": args.val_per_mille,
        "seed": args.seed,
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if train_tokens < args.train_tokens or val_tokens < args.val_tokens:
        raise SystemExit("FineWeb2 stream ended before token targets were reached")


if __name__ == "__main__":
    main()
