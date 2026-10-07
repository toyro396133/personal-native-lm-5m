from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

from datasets import load_dataset

from prepare_v12_fineweb2 import bucket, chunks
from train_hebrew import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--config", default="heb_Hebr")
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--tokens-per-shard", type=int, default=27_000_000)
    ap.add_argument("--val-tokens", type=int, default=500_000)
    ap.add_argument("--val-per-mille", type=int, default=10)
    ap.add_argument("--seed", type=int, default=217)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--out-dir", default="data/v17")
    ap.add_argument("--val-out", default="data/v17/val.txt")
    ap.add_argument("--report", default="artifacts/v17_data_report.json")
    args = ap.parse_args()

    tok, kind = load_tokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_path = Path(args.val_out)
    val_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, name=args.config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    shard_paths = [out_dir / f"train-shard-{i+1}.i32" for i in range(args.shards)]
    shard_files = [p.open("wb") for p in shard_paths]
    shard_hashers = [hashlib.sha256() for _ in shard_paths]
    shard_tokens = [0] * args.shards
    shard_chunks = [0] * args.shards

    docs_seen = 0
    chunks_seen = 0
    duplicates = 0
    unknown_rejected = 0
    val_tokens = 0
    val_chunks = 0
    seen = set()
    current_shard = 0

    def all_train_full():
        return all(n >= args.tokens_per_shard for n in shard_tokens)

    try:
        with val_path.open("w", encoding="utf-8") as fv:
            for row in ds:
                docs_seen += 1
                for chunk in chunks(str(row.get("text", ""))):
                    chunks_seen += 1
                    digest = hashlib.sha1(chunk.encode("utf-8")).digest()
                    if digest in seen:
                        duplicates += 1
                        continue
                    seen.add(digest)

                    ids = tok.encode(chunk + "\n", bos=False, eos=False)
                    if not ids:
                        continue
                    unk = getattr(tok, "unk_id", -1)
                    if unk >= 0 and any(i == unk for i in ids):
                        unknown_rejected += 1
                        continue

                    is_val = bucket("v17:" + chunk) < args.val_per_mille
                    if is_val:
                        if val_tokens < args.val_tokens:
                            fv.write(chunk + "\n")
                            val_tokens += len(ids)
                            val_chunks += 1
                        # Validation-bucket text is never allowed into training.
                        continue

                    while current_shard < args.shards and shard_tokens[current_shard] >= args.tokens_per_shard:
                        current_shard += 1
                    if current_shard >= args.shards:
                        if val_tokens >= args.val_tokens:
                            break
                        continue

                    buf = array("i", ids)
                    raw = buf.tobytes()
                    shard_files[current_shard].write(raw)
                    shard_hashers[current_shard].update(raw)
                    shard_tokens[current_shard] += len(ids)
                    shard_chunks[current_shard] += 1

                if docs_seen % 10_000 == 0:
                    print(
                        f"[prep] docs={docs_seen:,} chunks={chunks_seen:,} "
                        f"train={sum(shard_tokens):,} val={val_tokens:,} "
                        f"shards={shard_tokens}",
                        flush=True,
                    )

                if all_train_full() and val_tokens >= args.val_tokens:
                    break
    finally:
        for f in shard_files:
            f.close()

    report = {
        "version": "0.17",
        "dataset": args.dataset,
        "config": args.config,
        "tokenizer_kind": kind,
        "strategy": (
            "streamed shuffled FineWeb2 Hebrew; Hebrew-heavy chunks; global dedupe; "
            "deterministic validation holdout; four disjoint int32 training shards"
        ),
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "targets": {
            "shards": args.shards,
            "tokens_per_shard": args.tokens_per_shard,
            "total_training_reserve": args.shards * args.tokens_per_shard,
            "validation_tokens": args.val_tokens,
        },
        "actual": {
            "training_tokens": sum(shard_tokens),
            "validation_tokens": val_tokens,
            "shard_tokens": shard_tokens,
            "shard_chunks": shard_chunks,
            "validation_chunks": val_chunks,
        },
        "documents_seen": docs_seen,
        "chunks_seen": chunks_seen,
        "duplicates_rejected": duplicates,
        "unknown_token_chunks_rejected": unknown_rejected,
        "validation_per_mille": args.val_per_mille,
        "shards": [
            {
                "index": i + 1,
                "path": str(path),
                "tokens": shard_tokens[i],
                "bytes": path.stat().st_size,
                "sha256": shard_hashers[i].hexdigest(),
            }
            for i, path in enumerate(shard_paths)
        ],
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if not all_train_full() or val_tokens < args.val_tokens:
        raise SystemExit("FineWeb2 stream ended before all v0.17 targets were reached")


if __name__ == "__main__":
    main()
