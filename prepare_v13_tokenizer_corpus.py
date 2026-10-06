from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from prepare_v12_fineweb2 import chunks, bucket


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset",default="HuggingFaceFW/fineweb-2")
    ap.add_argument("--config",default="heb_Hebr")
    ap.add_argument("--chars",type=int,default=8_000_000)
    ap.add_argument("--val-per-mille",type=int,default=50)
    ap.add_argument("--seed",type=int,default=121)
    ap.add_argument("--shuffle-buffer",type=int,default=5000)
    ap.add_argument("--out",default="data/v13-tokenizer-train.txt")
    ap.add_argument("--report",default="artifacts/v13_tokenizer_corpus_report.json")
    args=ap.parse_args()

    ds=load_dataset(args.dataset,name=args.config,split="train",streaming=True)
    ds=ds.shuffle(seed=args.seed,buffer_size=args.shuffle_buffer)

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)

    chars_written=0
    docs_seen=0
    chunks_seen=0
    accepted=0
    duplicates=0
    validation_skipped=0
    seen=set()

    with out.open("w",encoding="utf-8") as f:
        for row in ds:
            docs_seen+=1
            for chunk in chunks(str(row.get("text",""))):
                chunks_seen+=1
                # Tokenizer vocabulary is learned only from the train split.
                if bucket("v12:"+chunk) < args.val_per_mille:
                    validation_skipped+=1
                    continue
                digest=hashlib.sha1(chunk.encode("utf-8")).digest()
                if digest in seen:
                    duplicates+=1
                    continue
                seen.add(digest)
                f.write(chunk+"\n")
                chars_written+=len(chunk)+1
                accepted+=1
                if chars_written>=args.chars:
                    break
            if chars_written>=args.chars:
                break

    report={
        "version":"0.13",
        "dataset":args.dataset,
        "config":args.config,
        "target_chars":args.chars,
        "actual_chars":chars_written,
        "documents_seen":docs_seen,
        "chunks_seen":chunks_seen,
        "accepted_train_chunks":accepted,
        "validation_chunks_skipped":validation_skipped,
        "duplicates":duplicates,
        "validation_leakage":False,
        "seed":args.seed,
    }
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if chars_written<args.chars:
        raise SystemExit("FineWeb2 stream ended before tokenizer-corpus target")


if __name__=="__main__":
    main()
