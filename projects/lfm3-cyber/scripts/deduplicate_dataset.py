#!/usr/bin/env python3
"""Report exact and high-similarity conversational duplicates; never mutates data."""
import argparse,json,re
from pathlib import Path
from difflib import SequenceMatcher

def text(row):
 return re.sub(r"\W+"," "," ".join(m["content"] for m in row["messages"] if m["role"] in {"user","assistant"}).casefold()).strip()
def main():
 p=argparse.ArgumentParser(); p.add_argument("path",type=Path); p.add_argument("--threshold",type=float,default=.88); a=p.parse_args()
 files=sorted(a.path.glob("*.jsonl")) if a.path.is_dir() else [a.path]
 rows=[]
 for f in files:
  for line in f.read_text(encoding="utf-8").splitlines():
   if line.strip(): rows.append(json.loads(line))
 exact={}; pairs=[]
 for i,r in enumerate(rows):
  t=text(r)
  if t in exact: pairs.append({"type":"exact","left":exact[t],"right":r["id"],"similarity":1.0})
  else: exact[t]=r["id"]
 for i,left in enumerate(rows):
  for right in rows[i+1:]:
   if left["split"]!=right["split"]:
    x,y=text(left),text(right)
    if min(len(x),len(y))>80:
     score=SequenceMatcher(None,x,y).ratio()
     if score>=a.threshold: pairs.append({"type":"cross_split_near_duplicate","left":left["id"],"right":right["id"],"similarity":round(score,4)})
 print(json.dumps({"examples":len(rows),"duplicate_pairs":pairs,"threshold":a.threshold},ensure_ascii=False,indent=2))
 return 1 if pairs else 0
if __name__=="__main__": raise SystemExit(main())
