"""Synthetic safety tests for transferring project evidence into pending review."""
import json
import tempfile
import unittest
from pathlib import Path
from bridge_pending import import_candidates
from compass_core import CompassStore


class PendingBridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.db=self.root/"memory.sqlite3"
        self.src=self.root/"synthetic-private.json"
        self.row={
            "status":"PENDING_REVIEW","claim_status":"APPROVED_CLAIM",
            "status_basis":"EXPLICIT_HEADING",
            "scope":"project","project_id":"chacham-lev",
            "candidate_id":"abcdef0123456789",
            "excerpt":"A confirmed section of the project decision record with a clearly stated explanation and evidence.",
            "source":"git:"+"a"*40+":docs/product/example.md#L10-L15",
            "approval_required":True
        }
        self.write([self.row])

    def tearDown(self):
        self.temp.cleanup()

    def write(self,rows,approved=False):
        self.src.write_text(json.dumps({"approved":approved,"candidates":rows}),encoding="utf8")

    def test_dry_run_does_not_create_database(self):
        r=import_candidates(self.src,self.db)
        self.assertEqual(r["eligible_explicit_evidence"],1)
        self.assertFalse(self.db.exists())
        self.assertEqual(r["auto_approvals"],0)

    def test_machine_approved_heading_not_approved_memory(self):
        r=import_candidates(self.src,self.db,commit=True)
        self.assertEqual(r["proposals_created"],1)
        self.assertEqual(r["auto_approvals"],0)
        with CompassStore(self.db) as s:
            self.assertEqual(s.principles(),[])
            self.assertEqual(len(s.proposals()),1)
            self.assertEqual([event["kind"] for event in s.timeline()],["PROPOSED"])
            self.assertIn("not verified",s.proposals()[0]["reason"])

    def test_repeated_ingestion_is_idempotent(self):
        self.assertEqual(import_candidates(self.src,self.db,commit=True)["proposals_created"],1)
        r=import_candidates(self.src,self.db,commit=True)
        self.assertEqual(r["proposals_created"],0)
        self.assertEqual(r["proposals_skipped_existing"],1)

    def test_unresolved_does_not_create_pending(self):
        self.write([{**self.row,"claim_status":"UNRESOLVED"}])
        r=import_candidates(self.src,self.db,commit=True)
        self.assertEqual(r["proposals_created"],0)

    def test_reject_cross_project_promotions(self):
        self.write([{**self.row,"scope":"personal"}])
        with self.assertRaisesRegex(ValueError,"Cross-project"):
            import_candidates(self.src,self.db,commit=True)
        self.assertFalse(self.db.exists())

    def test_reject_wrongly_preapproved_batches(self):
        self.write([self.row],approved=True)
        with self.assertRaisesRegex(ValueError,"approved=false"):
            import_candidates(self.src,self.db,commit=True)

    def test_reject_mixed_batch_atomically_before_db(self):
        malformed={**self.row,"candidate_id":"foo"}
        self.write([self.row,malformed])
        with self.assertRaisesRegex(ValueError,"stable candidate ID"):
            import_candidates(self.src,self.db,commit=True)
        self.assertFalse(self.db.exists())

    def test_skips_short_evidence(self):
        self.write([{**self.row,"excerpt":"too short"}])
        result=import_candidates(self.src,self.db,commit=True)
        self.assertEqual(result["eligible_explicit_evidence"],0)
        with CompassStore(self.db) as s:
            self.assertEqual(s.proposals(),[])


if __name__=="__main__":
    unittest.main(verbosity=2)
