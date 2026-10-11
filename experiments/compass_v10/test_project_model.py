"""Only invented fixtures; no private project material or GPT-6 teacher answer keys."""
import copy
import unittest
from project_model import assess,verified,EARLY

def fact(id,label):
    return dict(id=id,label=label,claim="Fictional test claim: "+id,
                source="git:"+EARLY+":docs/sample.md",source_lines=[3,7],
                status="TEACHER_CANDIDATE_NOT_VERIFIED")
def fixture():
    return dict(model="COMPASS_V10_PROJECT_UNDERSTANDING_PROTOTYPE",
        scope="chacham-lev",early_source_commit=EARLY,user_approved_memory_updates=0,
        claims=[fact("G6-001","APPROVED"),fact("G6-002","OPEN"),fact("G6-003","HISTORICAL")],
        goal_dependencies=[
          dict(id="R01",requires=["x","y"],outcome="SUPPORT",evidence=["G6-001"],rationale="fictional support"),
          dict(id="R02",requires=["z"],outcome="CONFLICT",evidence=["G6-001"],rationale="fictional conflict"),
          dict(id="R03",requires=["unknown"],outcome="ABSTAIN",evidence=["G6-002"],rationale="fictional unknown"),
        ])

class V10SourceAndReasoning(unittest.TestCase):
    def test_good_bundle(self):self.assertEqual(len(verified(fixture())[0]),3)
    def test_compound_support_requires_both_signals(self):
        self.assertEqual(assess(fixture(),["x","y"])["outcome"],"SUPPORT")
        self.assertEqual(assess(fixture(),["x"])["outcome"],"ABSTAIN")
    def test_explicit_conflict(self):self.assertEqual(assess(fixture(),["z"])["outcome"],"CONFLICT")
    def test_unknown_is_not_approved(self):
        result=assess(fixture(),["unknown"]);self.assertEqual(result["outcome"],"ABSTAIN")
        self.assertTrue(result["requires_human_review"]);self.assertFalse(result["autoapproved"])
    def test_open_policy_vetoes_generic_support(self):
        self.assertEqual(assess(fixture(),["x","y","unknown"])["outcome"],"ABSTAIN")
    def test_support_conflict_is_uncertain(self):
        self.assertEqual(assess(fixture(),["x","y","z"])["outcome"],"ABSTAIN")
    def test_no_cross_project_or_personal_transfer(self):
        for scope in ("other-project","personal"):
            with self.assertRaises(ValueError):assess(fixture(),["x"],project_id=scope)
    def test_reject_future_source(self):
        b=fixture();b["claims"][0]["source"]=b["claims"][0]["source"].replace(EARLY,"fe9863278a04dedf27a0c4fcb8cca56b91ec4544")
        with self.assertRaises(ValueError):verified(b)
    def test_reject_unverified_as_verified(self):
        b=fixture();b["claims"][0]["status"]="APPROVED"
        with self.assertRaises(ValueError):verified(b)
    def test_observation_not_policy(self):
        b=fixture();b["goal_dependencies"][0]["evidence"]=["G6-003"]
        with self.assertRaisesRegex(ValueError,"Normative"):verified(b)
    def test_abstain_rule_must_cite_open_issue(self):
        b=fixture();b["goal_dependencies"][2]["evidence"]=["G6-001"]
        with self.assertRaisesRegex(ValueError,"Uncertain"):verified(b)
    def test_no_claimed_autoapproval(self):
        b=fixture();b["user_approved_memory_updates"]=1
        with self.assertRaises(ValueError):verified(b)
    def test_citation_present_with_line_range(self):
        a=assess(fixture(),["z"])
        self.assertEqual(a["source_refs"][0]["lines"],[3,7])
        self.assertIn("git:"+EARLY,a["source_refs"][0]["source"])
    def test_deterministic(self):
        self.assertEqual(assess(fixture(),["y","x"]),assess(fixture(),["x","y"]))
    def test_reject_missing_source(self):
        b=fixture();b["goal_dependencies"][0]["evidence"]=["G6-999"]
        with self.assertRaises(ValueError):verified(b)

if __name__=="__main__":
    unittest.main(verbosity=2)
