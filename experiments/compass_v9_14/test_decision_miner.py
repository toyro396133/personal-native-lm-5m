"""V9.14 invented-document tests, no private Chacham Lev source text in CI."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from decision_miner import (sections, mine, heading_status, safe_summary,
                            baseline_document_title,baseline_local_keyword,
                            inventory, load)

SHA="a"*40
DOCUMENT="""# הצעת מודל לתפעול

המסמך הזה הוא הצעת תכנון, וכולל הצעות לצד החלטות שכבר אושרו.

## החלטה מאושרת T45 — חמשת מצבי השיחה

**מעמד מאושר:** המבנה המצומצם נקבע במפורש.

### הצעה א — הרחבת הדוחות

הערכה לא מחייבת, מותנית בבדיקה בעתיד.

### הכרעה מאושרת T48 — תנאי סיום

נקבעה סגירה רק לאחר אירוע קבלה מפורש.

## מה נעשה בפועל

בעבר נרשמו דיווחים חלקיים; אין להסיק מכך שהנוהל מאושר.

## שאלות פתוחות

האם בכלל נרצה לפתח עוד הרחבות?

## בוטלה החלטה T13

החלטה ישנה כבר אינה תקפה, אין להחיל אותה.

## החלטה מאושרת T51 — תיקון שמחליף T37

החלטת T37 הוחלפה בהבהרה מפורשת.

## תכנון עומק

החומר כאן מתאר סקירה, לא הכרעה. דרוש אישור אנושי.
"""


def git(folder:Path,*args:str)->str:
    p=subprocess.run(["git","-C",str(folder),*args],capture_output=True,text=True)
    if p.returncode:
        raise RuntimeError(p.stderr)
    return p.stdout.strip()


class MarkdownDecisionTests(unittest.TestCase):
    def test_status_hierarchy_and_no_inherited_document_claim(self):
        parts=sections(DOCUMENT,"docs/synthetic.md",SHA)
        s={p["heading"]:p["status"] for p in parts}
        self.assertEqual(s["הצעת מודל לתפעול"],"PROPOSED_CLAIM")
        self.assertEqual(s["החלטה מאושרת T45 — חמשת מצבי השיחה"],"APPROVED_CLAIM")
        self.assertEqual(s["הצעה א — הרחבת הדוחות"],"PROPOSED_CLAIM")
        self.assertEqual(s["הכרעה מאושרת T48 — תנאי סיום"],"APPROVED_CLAIM")
        self.assertEqual(s["מה נעשה בפועל"],"HISTORICAL_EVIDENCE")
        self.assertEqual(s["שאלות פתוחות"],"UNRESOLVED")
        self.assertEqual(s["בוטלה החלטה T13"],"REVOKED_OR_SUPERSEDED_CLAIM")
        self.assertEqual(s["תכנון עומק"],"UNRESOLVED")

    def test_top_level_proposal_must_not_override_subsection(self):
        items=sections(DOCUMENT,"docs/x.md",SHA)
        baseline=baseline_document_title(items)
        self.assertEqual(baseline[1],"PROPOSED_CLAIM")
        self.assertEqual(items[1]["status"],"APPROVED_CLAIM")

    def test_keyword_in_body_is_not_verified_authority(self):
        items=sections(DOCUMENT,"docs/x.md",SHA)
        baseline=baseline_local_keyword(items)
        ix=next(i for i,p in enumerate(items) if p["heading"]=="מה נעשה בפועל")
        self.assertEqual(baseline[ix],"APPROVED_CLAIM")
        self.assertEqual(items[ix]["status"],"HISTORICAL_EVIDENCE")
        intro=next(i for i,p in enumerate(items) if p["heading"]=="תכנון עומק")
        self.assertEqual(items[intro]["status"],"UNRESOLVED")

    def test_evidence_has_verified_provenance_and_no_autoapprove(self):
        found=mine(DOCUMENT,"docs/synthetic.md",SHA,"demo-lab")
        self.assertEqual(len(found),len(sections(DOCUMENT,"docs/synthetic.md",SHA)))
        self.assertTrue(all(x["approval_required"] and x["status"]=="PENDING_REVIEW" for x in found))
        self.assertTrue(all(x["scope"]=="project" and x["project_id"]=="demo-lab" for x in found))
        self.assertTrue(all(x["source"].startswith("git:"+SHA+":docs/synthetic.md#L") for x in found))
        self.assertTrue(all(x["source_sha256"] for x in found))
        self.assertEqual(safe_summary(found)["proposals_auto_approved"],0)

    def test_explicit_identifier_and_correction_edges(self):
        found=mine(DOCUMENT,"docs/synthetic.md",SHA,"demo-lab")
        correction=next(x for x in found if "T51" in x["heading"])
        self.assertIn("T51",correction["decision_ids"])
        self.assertIn("T37",correction["explicit_supersession_targets"])
        history=next(x for x in found if x["heading"]=="בוטלה החלטה T13")
        self.assertEqual(history["claim_status"],"REVOKED_OR_SUPERSEDED_CLAIM")

    def test_normalized_heading_does_not_infer_unseen_approval(self):
        self.assertEqual(heading_status("פרק א — תכנון"),("UNRESOLVED","NO_EXPLICIT_STATUS"))
        self.assertEqual(heading_status("הצעת תכנון מפורטת")[0],"PROPOSED_CLAIM")
        self.assertEqual(heading_status("החלטה מאושרת T46 — סמכויות")[0],"APPROVED_CLAIM")
        self.assertEqual(heading_status("החלטה לא מאושרת")[0],"UNRESOLVED")

    def test_scope_and_source_validation(self):
        with self.assertRaises(ValueError):
            mine(DOCUMENT,"docs/x.md",SHA,"../other")
        for invalid in ("../docs/secret.md","src/auth.ts","docs/../../env.md"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    load(Path("."),SHA,invalid)

    def test_line_ranges_stable_and_no_body_quote_as_authority(self):
        one=mine(DOCUMENT,"docs/synthetic.md",SHA,"demo-lab")
        two=mine(DOCUMENT,"docs/synthetic.md",SHA,"demo-lab")
        self.assertEqual(one,two)
        self.assertTrue(all(x["source_sha256"] for x in one))
        self.assertTrue(all(x["source"].count("#L")==1 for x in one))
        self.assertGreater(one[-1]["source"].find("-L"),0)


class PrivateSnapshotGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        git(self.root,"init","-q")
        git(self.root,"config","user.email","v914@example.invalid")
        git(self.root,"config","user.name","V9.14 Synthetic CI")
        (self.root/"docs").mkdir()
        (self.root/"docs"/"plan.md").write_text(DOCUMENT,encoding="utf8")
        git(self.root,"add",".");git(self.root,"commit","-qm","historical")
        self.old=git(self.root,"rev-parse","HEAD")
        (self.root/"docs"/"later.md").write_text(
            "# החלטה מאושרת T81 — מאוחרת\n\nעיקרון חדש מאוחר.",
            encoding="utf8")
        git(self.root,"add",".");git(self.root,"commit","-qm","future")
        self.later=git(self.root,"rev-parse","HEAD")

    def tearDown(self):
        self.temp.cleanup()

    def test_real_git_read_at_pinned_version(self):
        old=inventory(self.root,self.old,["docs/plan.md"],"demo")
        self.assertGreater(len(old),4)
        self.assertEqual(old[0]["source"].split(":")[1],self.old)
        later=inventory(self.root,self.later,["docs/later.md"],"demo")
        self.assertEqual(len(later),1)
        self.assertEqual(later[0]["claim_status"],"APPROVED_CLAIM")
        with self.assertRaises(ValueError):
            inventory(self.root,self.old,["docs/later.md"],"demo")

    def test_paths_are_explicit_allowlist_and_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            inventory(self.root,self.old,["docs/plan.md","docs/plan.md"],"demo")
        with self.assertRaises(ValueError):
            inventory(self.root,self.old,["src/private.ts"],"demo")

    def test_no_mutation_private_repo(self):
        before=git(self.root,"status","--porcelain")
        _=inventory(self.root,self.old,["docs/plan.md"],"demo")
        self.assertEqual(git(self.root,"status","--porcelain"),before)


if __name__=="__main__":
    unittest.main(verbosity=2)
