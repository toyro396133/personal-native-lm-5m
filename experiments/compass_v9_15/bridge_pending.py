"""Safe V9.14 -> V9.12 COMPASS store importer for PRIVATE project evidence.

DEFAULT DRY RUN. Explicit --import-pending creates review proposals only;
NEVER approves them. No source documents are written to public GitHub.
Use a local SQLite file and review proposals in the V9.12 localhost UI.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"compass_v9_12"))
from compass_core import CompassStore,CompassError

MAX_CANDIDATES=1000
ALLOWED={"APPROVED_CLAIM","PROPOSED_CLAIM","HISTORICAL_EVIDENCE",
         "REVOKED_OR_SUPERSEDED_CLAIM"}
def import_candidates(file:Path,db:Path,*,commit:bool=False,
                      project_id:str="chacham-lev")->dict:
    raw=json.loads(file.read_text(encoding="utf8"))
    if raw.get("approved",True) is not False:
        raise ValueError("Refusing inputs that do not explicitly assert approved=false")
    rows=raw.get("candidates")
    if not isinstance(rows,list) or len(rows)>MAX_CANDIDATES:
        raise ValueError("Candidates must be a bounded list")
    # Prepare first so bad input does not partially import a batch.
    proposals=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError("Invalid candidate structure")
        if row.get("approval_required") is not True:
            raise ValueError("Every imported candidate must require approval")
        if row.get("project_id")!=project_id or row.get("scope")!="project":
            raise ValueError("Cross-project/personal candidate rejected")
        if row.get("status")!="PENDING_REVIEW":
            raise ValueError("Unexpected status; only pending evidence can be imported")
        if row.get("claim_status") not in ALLOWED:
            continue
        if row.get("status_basis")!="EXPLICIT_HEADING":
            continue
        uid=row.get("candidate_id","")
        if not isinstance(uid,str) or len(uid)!=16 or any(x not in "0123456789abcdef" for x in uid):
            raise ValueError("Invalid stable candidate ID")
        snippet=str(row.get("excerpt","")).strip()
        if len(snippet)<60:
            continue
        path=row.get("source","")
        if not isinstance(path,str) or not path.startswith("git:") or "#L" not in path:
            raise ValueError("Source must be pinned Git commit and line")
        decision_key="doc-"+uid
        proposals.append({
            "scope":"project","project_id":project_id,
            "key":decision_key,"operation":"set",
            "expected_revision":0,
            "content":snippet[:1400],
            "reason":"Source heading claims "+str(row["claim_status"])+
                "; not verified as authorized user decision. Please assess intent, "
                "rationale, applicability, and correction history before approval.",
            "author":"compass-v915-machine-candidate",
            "source":path[:400],
        })
    # Deduplicate within the same batch and across persistent pending/approved.
    unique={p["key"]:p for p in proposals}
    report={"project_id":project_id,"source_records":len(rows),
            "eligible_explicit_evidence":len(unique),
            "mode":"IMPORT_PENDING" if commit else "DRY_RUN",
            "proposals_created":0,"proposals_skipped_existing":0,
            "auto_approvals":0}
    if not commit:return report
    with CompassStore(db) as store:
        existing={(p["owner"],p["key"]) for p in store.proposals()}
        existing.update((p["owner"],p["key"]) for p in store.principles(include_history=True))
        for p in unique.values():
            if (project_id,p["key"]) in existing:
                report["proposals_skipped_existing"]+=1;continue
            store.propose(**p)
            report["proposals_created"]+=1
            existing.add((project_id,p["key"]))
        report["audit_ok"]=store.verify()["ok"]
        report["pending_total"]=len(store.proposals())
        # Never call store.approve(), even if a heading claims approval.
    return report

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--private-candidates",required=True,type=Path)
    ap.add_argument("--db",required=True,type=Path)
    ap.add_argument("--project",default="chacham-lev")
    ap.add_argument("--import-pending",action="store_true")
    a=ap.parse_args()
    try:
        print(json.dumps(import_candidates(a.private_candidates,a.db,
            commit=a.import_pending,project_id=a.project),ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError,CompassError) as e:
        print("COMPASS import refused: "+str(e),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
