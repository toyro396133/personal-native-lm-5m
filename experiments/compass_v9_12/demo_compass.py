"""Runnable COMPASS V9.12 demonstration with public, invented project examples.

Usage (Windows / Linux):
  python experiments/compass_v9_12/demo_compass.py
  python experiments/compass_v9_12/demo_compass.py --db ./my-local-demo.sqlite3

Every record is fictitious and is explicitly approved only inside this demo.
"""
import argparse
import json
import tempfile
from pathlib import Path

from compass_core import CompassStore, CompassError


def run(path: Path) -> dict:
    with CompassStore(path) as c:
        def approved(**kwargs):
            proposal_id = c.propose(author="demo-author", source="demonstration-not-real",
                                    **kwargs)
            # Deliberately simulate explicit demo approval; not an AI action.
            c.approve(proposal_id, actor="demo-reviewer",
                      reason="Manual approval simulated for a public demo only")
            return proposal_id

        approved(scope="personal", project_id=None, key="work-method",
                 operation="set", expected_revision=0,
                 content="When the risk is low, try small reversible experiments",
                 reason="Example of a personal preference applicable across projects",
                 conditions={"risk": "low"})

        approved(scope="project", project_id="compass-lab", key="vision",
                 operation="set", expected_revision=0,
                 content="Keep the existing implementation unchanged",
                 reason="An invented initial description, to illustrate a later correction")

        approved(scope="project", project_id="another-project", key="vision",
                 operation="set", expected_revision=0,
                 content="This other project's independent aim is its own",
                 reason="This principle belongs to a different project")

        # Several later working notes do not modify the approved vision.
        for i in range(12):
            c.observe(f"Today implementation detail #{i} changed", author="demo-build-log")

        correction = c.propose(
            scope="project", project_id="compass-lab", key="vision",
            operation="set", expected_revision=1,
            content="Understand the project's enduring purpose, not just its changing code",
            reason="Clarification: implementation is not the authoritative long-term intent",
            author="demo-author", source="invented-user-correction")
        # Pending correction does not yet mutate an active principle.
        before = c.resolve(project_id="compass-lab", context={"risk": "low"})
        assert before["applicable"]["project"][0]["revision"] == 1

        c.approve(correction, actor="demo-reviewer",
                  reason="Explicit acceptance of the project vision correction")
        after = c.resolve(project_id="compass-lab", context={"risk": "low"})
        other = c.resolve(project_id="another-project", context={"risk": "high"})
        assert after["applicable"]["project"][0]["revision"] == 2
        assert after["applicable"]["project"][0]["content"].startswith("Understand")
        assert other["applicable"]["project"][0]["content"].startswith("This other")
        assert after["applicable"]["personal"][0]["key"] == "work-method"
        assert not other["applicable"]["personal"]  # conditional global preference
        audit = c.verify()
        result = {
            "status": "DEMO_PASS",
            "scope": "COMPASS_NOT_SELF",
            "approved_project_vision": after["applicable"]["project"][0],
            "applicable_personal_principles": [p["key"] for p in after["applicable"]["personal"]],
            "other_project_vision": other["applicable"]["project"][0]["content"],
            "other_project_inapplicable_personal_principles": len(other["applicable"]["personal"]),
            "project_vision_revisions": [
                {"revision": p["revision"], "status": p["status"],
                 "reason": p["reason"], "content": p["content"]}
                for p in c.principles(scope="project", project_id="compass-lab",
                                      include_history=True)
            ],
            "journal_events": audit["count"],
            "journal_consistent": audit["ok"],
            "warning": "Demo data only; approvals simulated. No autonomous language learning, real authentication or production deployment.",
        }

    with CompassStore(path) as reopened:
        assert reopened.verify()["head_hash"] == audit["head_hash"]
        assert reopened.resolve(project_id="compass-lab",
                                context={"risk":"low"})["applicable"] == after["applicable"]
        result["durable_reopen"] = "PASS"
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db",help="Optional location to preserve the fictional demonstration")
    args = p.parse_args()
    if args.db:
        path = Path(args.db)
        if path.exists():
            raise SystemExit("The requested demo database already exists; refusing to overwrite or reuse it")
        result = run(path)
    else:
        with tempfile.TemporaryDirectory() as folder:
            result = run(Path(folder)/"compass-demo.sqlite3")
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
