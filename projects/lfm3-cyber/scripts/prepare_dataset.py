#!/usr/bin/env python3
"""Validate JSONL file/directory and emit SFT-ready split-specific messages JSONL."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from validate_dataset import load_dir,validate_rows

def main()->int:
 p=argparse.ArgumentParser(); p.add_argument("jsonl",type=Path); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
 rows=load_dir(a.jsonl); errors=validate_rows(rows)
 if errors: raise SystemExit("\n".join(errors))
 a.output_dir.mkdir(parents=True,exist_ok=True)
 for split in ("train","validation","test"):
  selected=[r for r in rows if r["split"]==split]; target=a.output_dir/f"{split}.jsonl"
  with target.open("w",encoding="utf-8",newline="\n") as h:
   for row in selected: h.write(json.dumps({"messages":row["messages"]},ensure_ascii=False)+"\n")
  print(f"{split}: {len(selected)} examples -> {target}")
 return 0
if __name__=="__main__": raise SystemExit(main())
