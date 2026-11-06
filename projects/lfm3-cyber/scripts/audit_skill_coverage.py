#!/usr/bin/env python3
"""Audit explicit expected-skill coverage against the project matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_dataset import load_dir

def build_report(rows: list[dict], matrix: dict) -> dict:
    target = int(matrix.get("target_min_examples_per_skill", 1))
    domains = {}
    gaps = []
    for domain, skills in matrix["domains"].items():
        skill_rows = {}
        for skill in skills:
            count = sum(
                row.get("domain") == domain and skill in row.get("expected_skills", [])
                for row in rows
            )
            skill_rows[skill] = {"examples": count, "target": target, "meets_target": count >= target}
            if count < target:
                gaps.append({"domain": domain, "skill": skill, "examples": count, "target": target})
        domains[domain] = {
            "skills": skill_rows,
            "skills_meeting_target": sum(x["meets_target"] for x in skill_rows.values()),
            "skills_total": len(skill_rows),
        }
    return {
        "dataset_examples": len(rows),
        "target_min_examples_per_skill": target,
        "priority_skills_total": sum(len(v) for v in matrix["domains"].values()),
        "priority_skills_meeting_target": sum(d["skills_meeting_target"] for d in domains.values()),
        "domains": domains,
        "gaps": gaps,
        "interpretation": "Coverage counts explicit expected_skills tags only. It is not a measure of answer correctness, model mastery, or human expert review."
    }

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--matrix", type=Path, default=Path(__file__).resolve().parents[1] / "dataset" / "skill-matrix.json")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_dir(args.dataset)
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    report = build_report(rows, matrix)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if report["gaps"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
