#!/usr/bin/env python3
"""Create transparent coverage, quality and split audit reports."""
import argparse,json
from collections import Counter
from pathlib import Path
from validate_dataset import load_dir,validate_rows

def main():
 p=argparse.ArgumentParser(); p.add_argument("path",type=Path); p.add_argument("--output-dir",type=Path,required=True); a=p.parse_args()
 rows=load_dir(a.path); errors=validate_rows(rows); a.output_dir.mkdir(parents=True,exist_ok=True)
 coverage={"total":len(rows),"by_domain":dict(Counter(r["domain"] for r in rows)),"by_environment":dict(Counter(r["environment"] for r in rows)),"by_orientation":dict(Counter(r["orientation"] for r in rows)),"by_source_family":dict(Counter(r["provenance"]["source_family"] for r in rows)),"by_difficulty":dict(Counter(r["difficulty"] for r in rows)),"by_task":dict(Counter(r["task"] for r in rows))}
 quality={"review_status":dict(Counter(r["quality"]["review_status"] for r in rows)),"verification_status":dict(Counter(r["verification"]["status"] for r in rows)),"manual_reviewed":sum(r["verification"]["status"] in {"expert_review","verified"} for r in rows),"automated_only_or_pending":sum(r["verification"]["status"] in {"automated","pending"} for r in rows),"validator_errors":errors}
 split={"counts":dict(Counter(r["split"] for r in rows)),"scenario_groups":len({r["scenario_id"] for r in rows}),"method":"grouped by scenario_id; no scenario may cross splits; near-duplicate scan across splits"}
 for name,obj in (("coverage-report.json",coverage),("quality-report.json",quality),("split-report.json",split)):
  (a.output_dir/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
 print(json.dumps({"coverage":coverage,"quality":quality,"split":split},ensure_ascii=False,indent=2))
 return 1 if errors else 0
if __name__=="__main__": raise SystemExit(main())
