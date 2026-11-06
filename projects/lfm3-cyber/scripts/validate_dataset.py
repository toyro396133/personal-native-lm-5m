#!/usr/bin/env python3
"""Validate LFM3-Cyber JSONL directories, metadata, duplicates and split isolation."""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict, Counter
from pathlib import Path
from difflib import SequenceMatcher

SPLITS={"train","validation","test"}
DOMAINS={"fundamentals","network_security","soc","threat_intelligence","incident_response","cloud_infrastructure","penetration_testing","application_security"}
DIFFICULTIES={"beginner","intermediate","advanced"}
ENVIRONMENTS={"defensive","authorized_lab","conceptual"}
REQUIRED={"id","scenario_id","split","domain","task","environment","difficulty","orientation","messages","expected_skills","verification","source","quality","authorization","provenance"}

def norm(text:str)->str:
    return re.sub(r"\W+"," ",text.casefold()).strip()

def validate_rows(rows:list[dict])->list[str]:
    errors=[]; ids=set(); scenario_splits=defaultdict(set); exact={}; texts=[]
    for n,row in enumerate(rows,1):
        p=f"row {n}"
        if not isinstance(row,dict): errors.append(f"{p}: not object"); continue
        missing=REQUIRED-row.keys()
        if missing: errors.append(f"{p}: missing {sorted(missing)}"); continue
        for key in ("id","scenario_id","task"):
            if not isinstance(row[key],str) or not row[key].strip(): errors.append(f"{p}: invalid {key}")
        if row["id"] in ids: errors.append(f"{p}: duplicate id {row['id']}")
        ids.add(row["id"])
        if row["split"] not in SPLITS: errors.append(f"{p}: invalid split")
        else: scenario_splits[row["scenario_id"]].add(row["split"])
        if row["domain"] not in DOMAINS: errors.append(f"{p}: invalid domain")
        if row["difficulty"] not in DIFFICULTIES: errors.append(f"{p}: invalid difficulty")
        if row["environment"] not in ENVIRONMENTS: errors.append(f"{p}: invalid environment")
        if row.get("orientation") not in {"defensive","authorized_offensive","conceptual"}: errors.append(f"{p}: invalid orientation")
        if row.get("orientation") != {"defensive":"defensive","authorized_lab":"authorized_offensive","conceptual":"conceptual"}.get(row["environment"]): errors.append(f"{p}: orientation/environment mismatch")
        msgs=row["messages"]
        if not isinstance(msgs,list) or len(msgs)<3: errors.append(f"{p}: messages require at least 3 turns"); continue
        roles=[]
        for m in msgs:
            if not isinstance(m,dict) or set(m)!={"role","content"} or m.get("role") not in {"system","user","assistant"} or not isinstance(m.get("content"),str) or not m["content"].strip():
                errors.append(f"{p}: malformed message"); break
            roles.append(m["role"])
        if not roles or "user" not in roles or roles[-1]!="assistant": errors.append(f"{p}: final turn must be assistant and user turn required")
        auth=row["authorization"]
        if not isinstance(auth,dict) or auth.get("scope")!=row["environment"] or auth.get("status") not in {"explicit","not_applicable"}: errors.append(f"{p}: authorization mismatch")
        elif row["environment"]=="authorized_lab" and (auth.get("status")!="explicit" or not auth.get("permitted_target")): errors.append(f"{p}: authorized_lab requires explicit permitted_target")
        src=row["source"]
        if not isinstance(src,dict) or not all(isinstance(src.get(k),str) for k in ("name","url","license","checked_at","usage")): errors.append(f"{p}: incomplete source metadata")
        elif not src["url"] and row["provenance"].get("content_origin")!="original_synthetic": errors.append(f"{p}: external source URL required")
        q=row["quality"]; ver=row["verification"]
        if not isinstance(q,dict) or q.get("review_status") not in {"pending","reviewed","rejected"}: errors.append(f"{p}: invalid quality metadata")
        if not isinstance(ver,dict) or ver.get("status") not in {"pending","automated","expert_review","verified"}: errors.append(f"{p}: invalid verification metadata")
        if q.get("review_status")=="reviewed" and ver.get("status") not in {"expert_review","verified"}: errors.append(f"{p}: reviewed requires expert review")
        if ver.get("status")=="verified" and q.get("review_status")!="reviewed": errors.append(f"{p}: verified requires reviewed quality")
        fp=norm(" ".join(m["content"] for m in msgs if m["role"] in {"user","assistant"}))
        if fp in exact and exact[fp]!=row["split"]: errors.append(f"{p}: exact duplicate crosses split")
        exact[fp]=row["split"]
        texts.append((n,row["split"],norm(" ".join(m["content"] for m in msgs if m["role"] in {"user","assistant"}))))
    for sid,ss in scenario_splits.items():
        if len(ss)>1: errors.append(f"scenario_id {sid} crosses splits {sorted(ss)}")
    # Conservative near-duplicate scan, only between different splits.
    for i,(ni,si,ti) in enumerate(texts):
        for nj,sj,tj in texts[i+1:]:
            if si!=sj and min(len(ti),len(tj))>80 and SequenceMatcher(None,ti,tj).ratio()>=0.88:
                errors.append(f"near duplicate across splits: rows {ni} and {nj}")
    return errors

def load_dir(path:Path)->list[dict]:
    files=sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    rows=[]
    for f in files:
        with f.open(encoding="utf-8") as h:
            for line_no,line in enumerate(h,1):
                if line.strip():
                    try: rows.append(json.loads(line))
                    except json.JSONDecodeError as e: raise ValueError(f"{f}:{line_no}: {e}")
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",type=Path); a=ap.parse_args()
    rows=load_dir(a.path); errs=validate_rows(rows)
    if errs: print("\n".join("ERROR: "+e for e in errs)); return 1
    counts=Counter(r["split"] for r in rows)
    print(f"PASS: {len(rows)} examples; "+", ".join(f"{s}={counts[s]}" for s in sorted(SPLITS)))
    return 0
if __name__=="__main__": raise SystemExit(main())
