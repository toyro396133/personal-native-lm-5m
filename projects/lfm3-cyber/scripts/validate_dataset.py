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


def validate_schema_value(value, schema, path="$"):
    """Validate the JSON Schema subset used by this dataset without third-party packages."""
    errors=[]
    expected=schema.get("type")
    types=expected if isinstance(expected,list) else ([expected] if expected else [])
    checks={"object":lambda x:isinstance(x,dict),"array":lambda x:isinstance(x,list),"string":lambda x:isinstance(x,str),"number":lambda x:isinstance(x,(int,float)) and not isinstance(x,bool),"null":lambda x:x is None}
    if types and not any(t in checks and checks[t](value) for t in types):
        return [f"{path}: expected {' or '.join(types)}"]
    if "enum" in schema and value not in schema["enum"]: errors.append(f"{path}: invalid enum value")
    if isinstance(value,str):
        if len(value)<schema.get("minLength",0): errors.append(f"{path}: below minLength")
        if schema.get("format")=="date":
            from datetime import date
            try: date.fromisoformat(value)
            except ValueError: errors.append(f"{path}: invalid date")
    if isinstance(value,list):
        if len(value)<schema.get("minItems",0): errors.append(f"{path}: below minItems")
        if isinstance(schema.get("items"),dict):
            for i,item in enumerate(value): errors.extend(validate_schema_value(item,schema["items"],f"{path}[{i}]"))
    if isinstance(value,dict):
        props=schema.get("properties",{})
        for key in schema.get("required",[]):
            if key not in value: errors.append(f"{path}: missing {key}")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props: errors.append(f"{path}: unexpected property {key}")
        for key,child in props.items():
            if key in value: errors.extend(validate_schema_value(value[key],child,f"{path}.{key}"))
    if isinstance(value,(int,float)) and not isinstance(value,bool):
        if value<schema.get("minimum",value) or value>schema.get("maximum",value): errors.append(f"{path}: outside numeric bounds")
    return errors

def validate_rows(rows:list[dict], schema:dict|None=None)->list[str]:
    errors=[]; ids=set(); scenario_splits=defaultdict(set); exact={}; texts=[]
    for n,row in enumerate(rows,1):
        p=f"row {n}"
        if not isinstance(row,dict): errors.append(f"{p}: not object"); continue
        missing=REQUIRED-row.keys()
        if missing: errors.append(f"{p}: missing {sorted(missing)}"); continue
        if schema: errors.extend(validate_schema_value(row,schema,p))
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


def load_schema_for_path(path:Path):
    p=Path(path); candidate=(p.parent if p.is_dir() else p.parent.parent)/"schema.json"
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.is_file() else None

def load_registry_for_path(path:Path):
    p=Path(path); candidate=(p.parent if p.is_dir() else p.parent.parent)/"sources"/"source-registry.json"
    return json.loads(candidate.read_text(encoding="utf-8")) if candidate.is_file() else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("path",type=Path); a=ap.parse_args()
    rows=load_dir(a.path)
    schema=load_schema_for_path(a.path); registry=load_registry_for_path(a.path)
    errs=validate_rows(rows,schema=schema,registry=registry)
    if errs: print("\n".join("ERROR: "+e for e in errs)); return 1
    counts=Counter(r["split"] for r in rows)
    suffix="; JSON Schema enforced" if schema else "; no adjacent schema found"
    print(f"PASS: {len(rows)} examples; "+", ".join(f"{s}={counts[s]}" for s in sorted(SPLITS))+suffix)
    return 0
if __name__=="__main__": raise SystemExit(main())
