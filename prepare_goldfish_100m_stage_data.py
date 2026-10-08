from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

MODEL = "goldfish-models/heb_hebr_1000mb"
REVISION = "425eee09d1b9ff32ffa21d6cfb300a74302a0799"


def val_bucket(text: str) -> int:
    return int.from_bytes(
        hashlib.sha1(("goldfish-self:" + text).encode("utf-8")).digest()[:4],
        "big",
    ) % 1000


def recover_stage1_train_hashes(tokenizer, target_tokens: int = 1_050_000) -> set[str]:
    """Replay canary #1 data selection and recover train-document hashes."""
    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="heb_Hebr",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=1701, buffer_size=5000)
    seen = set()
    used = set()
    train_n = 0
    for row in ds:
        text = str(row.get("text", "")).strip()
        if len(text) < 120:
            continue
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        if val_bucket(text) < 60:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        bos = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.cls_token_id
        eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id
        ids = [int(bos)] + ids + [int(eos)]
        take = min(len(ids), target_tokens - train_n)
        if take > 0:
            used.add(digest)
            train_n += take
        if train_n >= target_tokens:
            break
    if train_n != target_tokens:
        raise RuntimeError(f"could not recover stage1 prefix: {train_n}")
    return used


def load_hashes(path: str | None) -> set[str]:
    if not path:
        return set()
    p = Path(path)
    if not p.exists():
        return set()
    return {x.strip() for x in p.read_text(encoding="utf-8").splitlines() if x.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--prior-hashes")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-tokens", type=int, default=1_020_000)
    args = ap.parse_args()

    if not 2 <= args.stage <= 100:
        raise ValueError("stage must be 2..100")

    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    if bos is None or eos is None:
        raise RuntimeError("tokenizer lacks BOS/EOS-compatible IDs")

    used_before = load_hashes(args.prior_hashes)
    if args.stage == 2 and not used_before:
        print("Recovering exact canary #1 document set...", flush=True)
        used_before = recover_stage1_train_hashes(tok)
        print(f"Recovered {len(used_before):,} stage1 document hashes", flush=True)
    elif args.stage > 2 and not used_before:
        raise RuntimeError("prior cumulative document hashes are required after stage 2")

    # A stage-specific deterministic shuffle plus cumulative hash exclusion gives
    # fresh documents without scanning a 1/99 partition of the corpus.
    seed = 1701 + args.stage * 100_003
    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="heb_Hebr",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=5000)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.i32"
    hash_path = out / "used_doc_hashes.txt"

    local_seen = set()
    newly_used = []
    n = 0
    docs = 0
    skipped_prior = 0
    skipped_dupe = 0
    token_hash = hashlib.sha256()

    with train_path.open("wb") as fh:
        for row in ds:
            text = str(row.get("text", "")).strip()
            if len(text) < 120:
                continue
            digest = hashlib.sha1(text.encode("utf-8")).hexdigest()
            if digest in used_before:
                skipped_prior += 1
                continue
            if digest in local_seen:
                skipped_dupe += 1
                continue
            local_seen.add(digest)
            # Keep the original canary's validation partition permanently out of train.
            if val_bucket(text) < 60:
                continue

            ids = [int(bos)] + tok.encode(text, add_special_tokens=False) + [int(eos)]
            take = ids[: args.train_tokens - n]
            if not take:
                break
            raw = array("i", take).tobytes()
            fh.write(raw)
            token_hash.update(raw)
            n += len(take)
            docs += 1
            newly_used.append(digest)
            if docs % 1000 == 0:
                print(
                    f"stage={args.stage} docs={docs:,} tokens={n:,} "
                    f"prior_skips={skipped_prior:,}",
                    flush=True,
                )
            if n >= args.train_tokens:
                break

    if n != args.train_tokens:
        raise RuntimeError(f"stage {args.stage} produced only {n:,} tokens")

    cumulative = sorted(used_before.union(newly_used))
    hash_path.write_text("\n".join(cumulative) + "\n", encoding="utf-8")

    report = {
        "schema_version": 1,
        "experiment": "goldfish-124m-self-100m-100-stage",
        "stage": args.stage,
        "nominal_start_tokens": (args.stage - 1) * 1_000_000,
        "nominal_end_tokens": args.stage * 1_000_000,
        "model_id": MODEL,
        "model_revision": REVISION,
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "heb_Hebr",
        "shuffle_seed": seed,
        "shuffle_buffer": 5000,
        "training_tokens": n,
        "documents_used": docs,
        "prior_document_hashes": len(used_before),
        "new_document_hashes": len(newly_used),
        "cumulative_document_hashes": len(cumulative),
        "prior_documents_skipped": skipped_prior,
        "local_duplicates_skipped": skipped_dupe,
        "train_sha256": token_hash.hexdigest(),
        "tokenizer_vocab": len(tok),
    }
    (out / "data_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
