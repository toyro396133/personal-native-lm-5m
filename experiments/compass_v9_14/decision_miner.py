"""COMPASS V9.14 — grounded decision claims from PRIVATE project documentation.

Stdlib only. Not a language model, not identity verification, not autonomous
approval. All candidates are PROJECT-only and PENDING_REVIEW.

CLI:
 python decision_miner.py --repo C:/path/to/chacham-lev --ref <pinned-sha> \
    --path docs/product/importance-map.md --path docs/... \
    --private-output C:/private/compass-decision-claims.json \
    --sanitized-metrics C:/private/compass-v914-metrics.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

APPROVED = re.compile(
    r"^(?:החלטה\s+מאושרת|הכרעה\s+מאושרת|עיקרון\s+מוצר\s+מאושר|"
    r"הכרעה\s+מחייבת|עיקרון\s+מאושר|נוהל\s+מחייב|"
    r"תיקון\s+סמכויות.*מחייב|approved\s+decision|confirmed\s+principle)\b",re.I)
PROPOSED = re.compile(
    r"^(?:הצעה|הצעת|מוצע|טיוטה|תכנון\s+מוצע|"
    r"proposal|draft|suggested)\b",re.I)
HISTORICAL = re.compile(
    r"^(?:מה\s+נעשה\s+בפועל|מה\s+היה|עד\s+היום|היסטור|"
    r"המצב\s+הקיים|existing\s+behavior|historical)\b",re.I)
OPEN = re.compile(
    r"^(?:שאל(?:ה|ות)\s+(?:פתוח|להמשך)|נותר\s+(?:פתוח|לבירור)|"
    r"טעון\s+(?:אימות|דיון)|לא\s+הוכרע|טרם\s+אושר|"
    r"open\s+question|unresolved)\b",re.I)
WITHDRAWN = re.compile(
    r"^(?:בוטל|בוטלה|הוחלף|הוחלפה|ביטול|נפסל|"
    r"revoked|superseded|withdrawn)\b",re.I)
DIRECT_REF = re.compile(r"(?<![a-zA-Z0-9])(?:T\d{1,3}|D-\d{3})(?![a-zA-Z0-9])",re.I)
REPLACEMENT = re.compile(
    r"(?:מחליף|מחליפה|מחליפים|מחליפות|מחליפ(?:ה|ים|ות)?|במקום|מתקנ(?:ת|ים|ות)?|מבטל(?:ת|ים|ות)?|"
    r"supersedes?|replaces?|corrects?|withdraws?)\s+(?:את\s+)?"
    r"(?:החלט(?:ת|ה)\s+)?((?:T\d{1,3}|D-\d{3}))",re.I)
HEADER = re.compile(r"^(#{1,6})\s+(.+)$")
SCOPE = "project"
LABELS = ("APPROVED_CLAIM","PROPOSED_CLAIM","HISTORICAL_EVIDENCE",
          "UNRESOLVED","REVOKED_OR_SUPERSEDED_CLAIM")
PATTERNS = [
    ("REVOKED_OR_SUPERSEDED_CLAIM",WITHDRAWN),
    ("UNRESOLVED",OPEN),
    ("PROPOSED_CLAIM",PROPOSED),
    ("APPROVED_CLAIM",APPROVED),
    ("HISTORICAL_EVIDENCE",HISTORICAL),
]

def _run(repo: Path,*args: str, optional: bool=False) -> str | None:
    done=subprocess.run(["git","-C",str(repo),*args],
        capture_output=True,text=True,encoding="utf8",errors="replace")
    if done.returncode:
        if optional:return None
        raise ValueError("git revision/path not accessible in local private clone")
    return done.stdout

def resolve_ref(repo: Path,ref: str) -> str:
    s=(_run(repo,"rev-parse","--verify",f"{ref}^{{commit}}") or "").strip()
    if not re.fullmatch(r"[a-f0-9]{40}",s):
        raise ValueError("invalid pinned Git commit")
    return s

def load(repo: Path,ref: str,path: str) -> str:
    if (not path.startswith("docs/") or ".." in Path(path).parts or
        not path.endswith((".md",".txt"))):
        raise ValueError("allow only docs/ markdown/text under the private repository")
    text=_run(repo,"show",f"{ref}:{path}",optional=True)
    if text is None:
        raise ValueError(f"not found at historical ref: {path}")
    return text

def normalize_heading(raw: str) -> str:
    raw=re.sub(r"^[\[\(]?(?:T\d{1,3}|D-\d{3})[\]\)]?\s*[-—:]?\s*","",raw.strip(),flags=re.I)
    return raw.strip("* \t—–:-")

def heading_status(raw: str) -> tuple[str,str]:
    title=normalize_heading(raw)
    for label,pattern in PATTERNS:
        if pattern.search(title):
            return label,"EXPLICIT_HEADING"
    if re.search(r"(?:\bהיסטורי\b|\bhistorical\b)",title,re.I):
        return "HISTORICAL_EVIDENCE","CONTEXTUAL_HEADING"
    return "UNRESOLVED","NO_EXPLICIT_STATUS"

def sections(markdown: str, path: str,ref: str) -> list[dict]:
    """Section has its OWN heading + body until the NEXT heading (any level).

    Parent headings are given as context only; they never authorize a child.
    The top-level title can be 'proposal' while its approved children remain
    distinct. Conversely, a parent 'approved' is not sufficient for children.
    """
    lines=markdown.splitlines()
    stack=[];out=[];current=None
    def flush(next_line: int):
        if current is None:return
        current["end_line"]=next_line-1
        current["evidence_excerpt"]=" ".join(x.strip() for x in current.pop("body") if x.strip())[:560]
        current["evidence_sha256"]=hashlib.sha256(
            (current["heading"]+"\n"+current["evidence_excerpt"]).encode("utf8")).hexdigest()
        out.append(current)
    for n,line in enumerate(lines,1):
        match=HEADER.match(line)
        if match:
            flush(n)
            level=len(match.group(1))
            while stack and stack[-1]["level"]>=level:
                stack.pop()
            status,basis=heading_status(match.group(2))
            current={
                "source_file":path,"source_commit":ref,
                "heading":match.group(2).strip(),"level":level,
                "start_line":n,"status":status,"status_basis":basis,
                "parents":[s["heading"] for s in stack],"body":[]
            }
            stack.append({"level":level,"heading":match.group(2).strip()})
        elif current is not None:
            current["body"].append(line)
    flush(len(lines)+1)
    return out

def baseline_document_title(all_sections: list[dict]) -> dict[int,str]:
    first=all_sections[0] if all_sections else None
    label=first["status"] if first else "UNRESOLVED"
    return {i:label for i in range(len(all_sections))}

def baseline_local_keyword(all_sections: list[dict]) -> dict[int,str]:
    results={}
    for i,s in enumerate(all_sections):
        combined=(s["heading"]+" "+s["evidence_excerpt"]).lower()
        # Ordinary first keyword regardless of heading hierarchy or precedence:
        if any(x in combined for x in ("מאושר","הוכרע","מחייב","approved")):
            results[i]="APPROVED_CLAIM"
        elif any(x in combined for x in ("הצעה","מוצע","טיוטה","proposal","draft")):
            results[i]="PROPOSED_CLAIM"
        elif any(x in combined for x in ("בעבר","היסטורי","historical","בפועל")):
            results[i]="HISTORICAL_EVIDENCE"
        else:results[i]="UNRESOLVED"
    return results

def mine(markdown: str,path: str,ref: str,project_id: str) -> list[dict]:
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}",project_id):
        raise ValueError("invalid canonical project ID")
    found=[]
    for item in sections(markdown,path,ref):
        refs=sorted(set(DIRECT_REF.findall(item["heading"])))
        supersedes=sorted(set(REPLACEMENT.findall(item["heading"]+" "+item["evidence_excerpt"])))
        # Exact heading status is a *claim*, not an approval decision.
        status=item["status"]
        confidence=0.90 if status!="UNRESOLVED" and item["status_basis"]=="EXPLICIT_HEADING" else 0.0
        uid=hashlib.sha256((ref+":"+path+":"+str(item["start_line"])).encode()).hexdigest()[:16]
        found.append({
            "candidate_id":uid,"status":"PENDING_REVIEW",
            "claim_status":status,"confidence_estimate_not_calibrated":confidence,
            "scope":"project","project_id":project_id,
            "source":f"git:{ref}:{path}#L{item['start_line']}-L{item['end_line']}",
            "source_sha256":item["evidence_sha256"],
            "heading":item["heading"],"context_path":item["parents"],
            "excerpt":item["evidence_excerpt"],
            "status_basis":item["status_basis"],
            "decision_ids":refs,"explicit_supersession_targets":supersedes,
            "approval_required":True,
            "review_reason":"Confirm identity/authority, applicability, and whether this is actual approved intent.",
        })
    return found

def inventory(repo: Path,ref: str,paths: list[str],project_id: str) -> list[dict]:
    sha=resolve_ref(repo,ref)
    if len(set(paths))!=len(paths):
        raise ValueError("duplicate allowlisted document")
    return [c for path in paths for c in mine(load(repo,sha,path),path,sha,project_id)]

def safe_summary(items: list[dict]) -> dict:
    from collections import Counter
    count=Counter(x["claim_status"] for x in items)
    nested=sum(x["status_basis"]=="EXPLICIT_HEADING" and x["context_path"] for x in items)
    return {
        "total_sections":len(items),"status_counts":dict(count),
        "explicit_heading_status":sum(x["status_basis"]=="EXPLICIT_HEADING" for x in items),
        "explicit_status_below_a_parent":nested,
        "explicit_supersession_links":sum(bool(x["explicit_supersession_targets"]) for x in items),
        "proposals_auto_approved":0,"human_independent_truth_labels":False,
        "warning":"Heading-derived silver labels are not independent human adjudications."
    }

def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument("--repo",type=Path,required=True)
    ap.add_argument("--ref",required=True)
    ap.add_argument("--project",default="chacham-lev")
    ap.add_argument("--path",action="append",required=True)
    ap.add_argument("--private-output",type=Path)
    ap.add_argument("--sanitized-metrics",type=Path)
    a=ap.parse_args()
    try:
        items=inventory(a.repo,a.ref,a.path,a.project)
        summary=safe_summary(items)
        # Full content stays only on the local private machine. Never publish
        # this file or upload it to a public CI artifact.
        if a.private_output:
            a.private_output.parent.mkdir(parents=True,exist_ok=True)
            a.private_output.write_text(
                json.dumps({"approved":False,"candidates":items},ensure_ascii=False,indent=2),
                encoding="utf8")
        if a.sanitized_metrics:
            a.sanitized_metrics.write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf8")
        print(json.dumps(summary,ensure_ascii=False,indent=2))
        return 0
    except (OSError,ValueError) as error:
        print(f"COMPASS V9.14 refused to run: {error}",file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
