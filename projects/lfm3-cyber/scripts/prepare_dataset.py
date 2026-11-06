#!/usr/bin/env python3
"""Validate a source JSONL and emit split-specific conversational JSONL files."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_dataset import load_jsonl, validate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.jsonl)
    errors = validate_rows(rows)
    if errors:
        raise SystemExit("\n".join(errors))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        target = args.output_dir / f"{split}.jsonl"
        with target.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected:
                handle.write(json.dumps({"messages": row["messages"]}, ensure_ascii=False) + "\n")
        print(f"{split}: {len(selected)} examples -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
