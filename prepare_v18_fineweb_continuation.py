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
    ap.add_argument("--skip-documents", type=int, required=True)
    ap.add_argument("--shards", type=int, default=20)
    ap.add_argument("--tokens-per-shard", type=int, default=10_200_000)
    ap.add_argument("--seed", type=int, default=217)
    ap.add_argument("--shuffle-buffer", type=int, default=10_000)
    ap.add_argument("--val-per-mille", type=int, default=10)
    ap.add_argument("--out-dir", default="data/v18")
    ap.add_argument("--report", default="artifacts/v18_data_report.json")
    args = ap.parse_args()

    tok, kind = load_tokenizer(args.tokenizer)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)

    # Recreate exactly the same shuffled stream used by v0.17. We walk through
    # every document already consumed by the 100M study and rebuild its chunk
    # digest set. That gives the continuation a hard exclusion against exact
    # chunk duplicates from the old region, not merely a different random seed.
    ds = load_dataset(args.dataset, name=args.config, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    seen = set()
    skipped_docs = 0
    skipped_chunks = 0
    stream = iter(ds)
    for row in stream:
        skipped_docs += 1
        for chunk in chunks(str(row.get("text", ""))):
            skipped_chunks += 1
            seen.add(hashlib.sha1(chunk.encode("utf-8")).digest())
        if skipped_docs >= args.skip_documents:
            break

    if skipped_docs != args.skip_documents:
        raise SystemExit(
            f"stream ended while reconstructing v0.17 prefix: "
            f"{skipped_docs} != {args.skip_documents}"
        )

    shard_paths = [out_dir / f"train-shard-{i+1:02d}.i32" for i in range(args.shards)]
    shard_files = [p.open("wb") for p in shard_paths]
    shard_hashers = [hashlib.sha256() for _ in shard_paths]
    shard_tokens = [0] * args.shards
    shard_chunks = [0] * args.shards

    continuation_docs = 0
    chunks_seen = 0
    old_or_new_duplicates = 0
    validation_rejected = 0
    unknown_rejected = 0
    current_shard = 0

    def all_train_full():
        return all(n >= args.tokens_per_shard for n in shard_tokens)

    try:
        for row in stream:
            continuation_docs += 1
            for chunk in chunks(str(row.get("text", ""))):
                chunks_seen += 1
                digest = hashlib.sha1(chunk.encode("utf-8")).digest()
                if digest in seen:
                    old_or_new_duplicates += 1
                    continue
                seen.add(digest)

                # Preserve the exact v0.17 validation partition: anything in the
                # holdout bucket is excluded from continuation training too.
                if bucket("v17:" + chunk) < args.val_per_mille:
                    validation_rejected += 1
                    continue

                ids = tok.encode(chunk + "\n", bos=False, eos=False)
                if not ids:
                    continue
                unk = getattr(tok, "unk_id", -1)
                if unk >= 0 and any(i == unk for i in ids):
                    unknown_rejected += 1
                    continue

                while (
                    current_shard < args.shards
                    and shard_tokens[current_shard] >= args.tokens_per_shard
                ):
                    current_shard += 1
                if current_shard >= args.shards:
                    break

                buf = array("i", ids)
                raw = buf.tobytes()
                shard_files[current_shard].write(raw)
                shard_hashers[current_shard].update(raw)
                shard_tokens[current_shard] += len(ids)
                shard_chunks[current_shard] += 1

            if continuation_docs % 10_000 == 0:
                print(
                    f"[prep-v18] skipped_old_docs={skipped_docs:,} "
                    f"new_docs={continuation_docs:,} "
                    f"new_train={sum(shard_tokens):,} "
                    f"current_shard={current_shard+1}/{args.shards}",
                    flush=True,
                )
            if all_train_full():
                break
    finally:
        for f in shard_files:
            f.close()

    report = {
        "version": "0.18",
        "purpose": "strict continuation of v0.17 after its consumed document prefix",
        "dataset": args.dataset,
        "config": args.config,
        "tokenizer_kind": kind,
        "seed": args.seed,
        "shuffle_buffer": args.shuffle_buffer,
        "v17_documents_consumed": args.skip_documents,
        "reconstructed_old_documents": skipped_docs,
        "reconstructed_old_chunks": skipped_chunks,
        "continuation_documents_seen": continuation_docs,
        "continuation_chunks_seen": chunks_seen,
        "duplicate_chunks_rejected_against_old_and_new": old_or_new_duplicates,
        "validation_bucket_chunks_rejected": validation_rejected,
        "unknown_token_chunks_rejected": unknown_rejected,
        "validation_rule": f"bucket('v17:' + chunk) < {args.val_per_mille} excluded",
        "targets": {
            "shards": args.shards,
            "tokens_per_shard": args.tokens_per_shard,
            "total_training_reserve": args.shards * args.tokens_per_shard,
            "planned_consumption": 200_000_000,
        },
        "actual": {
            "training_tokens": sum(shard_tokens),
            "shard_tokens": shard_tokens,
            "shard_chunks": shard_chunks,
        },
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
        "disjointness_note": (
            "The same shuffled FineWeb-2 stream/seed as v0.17 is replayed through "
            "the exact old document count; old chunk digests are reconstructed and "
            "excluded from all new shards. New shards are also globally deduplicated."
        ),
    }
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    if not all_train_full():
        raise SystemExit("FineWeb-2 stream ended before all v0.18 continuation shards were full")


if __name__ == "__main__":
    main()
