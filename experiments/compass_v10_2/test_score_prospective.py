"""Fictional scorecards only. No real project questions or answers in CI."""
import copy
import unittest
from score_prospective import aggregate,exact_sign_p,LOCKED_SHA

METRICS=["goal_alignment","evidence_fidelity","continuation_value","uncertainty_calibration","unsupported_claims"]
CODES=["anon-q","anon-r","anon-s","anon-t"]
MAP=dict(zip(CODES,("R","C","N","F")))
def points(v):
    return {k:(0 if k=="unsupported_claims" else v) for k in METRICS}
def fixture(n=12,good=True):
    tasks=[]
    for i in range(n):
        by={}
        for code,arm in MAP.items():
            by[code]=points((4 if good else 1) if arm=="C" else 2 if arm=="R" else 1)
        tasks.append({"id":f"FAKE-{i:03d}","created_at":"2026-10-12T02:00:00+00:00",
           "workstream":("groups","routes","planning")[i%3],
           "fresh_independent_sessions":True,"task_and_evidence_adjudicated":True,
           "arm_id_hidden_until_ratings_finalized":True,
           "context_chars":{code:3300 for code in CODES},
           "ratings":[{"reviewer_id":"fictional-reviewer-1","scores":copy.deepcopy(by)},
                      {"reviewer_id":"fictional-reviewer-2","scores":copy.deepcopy(by)}]})
    return {"schema":"COMPASS_V10_2_PROSPECTIVE_V1",
            "preregistered_protocol_sha":LOCKED_SHA,
            "unblinding_arm_map":MAP.copy(),"tasks":tasks}

class V102Tests(unittest.TestCase):
    def test_exact_binomial_zero_non_ties(self):
        self.assertEqual(exact_sign_p([0,0])["two_sided_p"],1)
    def test_exact_binomial_all_wins(self):
        self.assertLess(exact_sign_p([1]*12)["two_sided_p"],0.05)
    def test_hypothetical_pass_needs_full_gate(self):
        result=aggregate(fixture())
        self.assertTrue(result["minimum_evaluable_tasks_reached"])
        self.assertTrue(result["locked_gate_passed"])
        self.assertEqual(result["verdict"],"PROVISIONAL_ONE_PROJECT_COMPASS_ADVANTAGE")
    def test_hypothetical_nonimprovement_fails(self):
        self.assertEqual(aggregate(fixture(good=False))["verdict"],"NO_PROVEN_ADDED_VALUE_OVER_RAG")
    def test_sample_size_limit_not_success(self):
        self.assertEqual(aggregate(fixture(n=10))["verdict"],"INSUFFICIENT_PROSPECTIVE_EVIDENCE")
    def test_no_cases_is_not_research_result(self):
        z=aggregate(fixture(n=0))
        self.assertEqual(z["task_count"],0)
        self.assertIsNone(z["paired_mean_C_minus_R"])
    def test_before_freeze_rejected(self):
        f=fixture();f["tasks"][0]["created_at"]="2026-10-11T23:00:00Z"
        with self.assertRaisesRegex(ValueError,"predates"):aggregate(f)
    def test_time_without_zone_rejected(self):
        f=fixture();f["tasks"][0]["created_at"]="2026-10-12T02:00:00"
        with self.assertRaisesRegex(ValueError,"Timezone"):aggregate(f)
    def test_duplicate_reviewer_rejected(self):
        f=fixture();f["tasks"][0]["ratings"][1]["reviewer_id"]="fictional-reviewer-1"
        with self.assertRaisesRegex(ValueError,"reviewer"):aggregate(f)
    def test_one_reviewer_rejected(self):
        f=fixture();f["tasks"][0]["ratings"].pop()
        with self.assertRaisesRegex(ValueError,"independent reviewers"):aggregate(f)
    def test_missing_arm_rejected(self):
        f=fixture();f["tasks"][0]["ratings"][0]["scores"].pop("anon-r")
        with self.assertRaisesRegex(ValueError,"all blind arms"):aggregate(f)
    def test_nonrandomized_shared_session_rejected(self):
        f=fixture();f["tasks"][0]["fresh_independent_sessions"]=False
        with self.assertRaisesRegex(ValueError,"independence"):aggregate(f)
    def test_context_budget_overrun_rejected(self):
        f=fixture();f["tasks"][0]["context_chars"]["anon-q"]=5001
        with self.assertRaisesRegex(ValueError,"budget"):aggregate(f)
    def test_score_domain_exceeded(self):
        f=fixture();f["tasks"][0]["ratings"][0]["scores"]["anon-r"]["unsupported_claims"]=5
        with self.assertRaisesRegex(ValueError,"0 to 4"):aggregate(f)
    def test_teacher_label_or_missing_protocol_rejected(self):
        f=fixture();f["preregistered_protocol_sha"]="not frozen"
        with self.assertRaisesRegex(ValueError,"hash mismatch"):aggregate(f)
    def test_severe_unsupported_fails_safety_gate(self):
        f=fixture()
        for row in f["tasks"]:
            for rat in row["ratings"]:
                rat["scores"]["anon-r"]["unsupported_claims"]=4
        result=aggregate(f)
        self.assertFalse(result["locked_gate_passed"])
    def test_requires_three_workstreams(self):
        f=fixture()
        for row in f["tasks"]:row["workstream"]="same"
        self.assertEqual(aggregate(f)["verdict"],"INSUFFICIENT_PROSPECTIVE_EVIDENCE")

if __name__=="__main__":
    unittest.main(verbosity=2)
