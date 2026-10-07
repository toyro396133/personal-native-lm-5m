from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_hebrew import load_tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    started = time.time()
    print(f"[prep] reading_text path={args.text}", flush=True)
    text = Path(args.text).read_text(encoding="utf-8")
    print(f"[prep] text_loaded chars={len(text):,} elapsed={time.time()-started:.1f}s", flush=True)

    tok, kind = load_tokenizer(args.tokenizer)
    t0 = time.time()
    print(f"[prep] tokenization_started tokenizer={kind}", flush=True)
    ids = tok.encode(text)
    print(f"[prep] tokenization_complete tokens={len(ids):,} elapsed={time.time()-t0:.1f}s", flush=True)

    t0 = time.time()
    tokens = torch.tensor(ids, dtype=torch.int32)
    del ids, text
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(tokens, args.out)
    size = Path(args.out).stat().st_size
    print(f"[prep] tensor_saved path={args.out} bytes={size:,} elapsed={time.time()-t0:.1f}s", flush=True)

    report = {
        "tokenizer_kind": kind,
        "tokens": int(tokens.numel()),
        "dtype": str(tokens.dtype),
        "bytes": size,
        "total_elapsed_seconds": time.time()-started,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
