"""Public CI fixtures contain deliberately invented concepts only."""
import unittest
from chronological_study import replay

SHA="a"*40
def row(phase,facet,kind,claim,rev=0):
    source=f"git:{SHA}:docs/fictional.md#L1-L4"
    return {"phase":phase,"review_state":"GPT6_TEACHER_PROPOSAL_NOT_USER_APPROVED",
     "record":{"scope":"project","project_id":"chacham-lev","facet":facet,
       "kind":kind,"claim":claim,"why":"Invented test-only explanation",
       "source":source,"expected_revision":rev,"dependencies":[],"conditions":{}}}
def dataset():
    items=[
      row("A_EARLY","goal.reuse","goal","Save knowledge into next season"),
      row("B_INTERMEDIATE","phone.route_identity","open_question","Unknown route continuity"),
      row("B_INTERMEDIATE","phone.assignment_authority","decision","Either actor perhaps"),
      row("B_INTERMEDIATE","phone.daily_workflow","decision","Close after clearing all"),
      row("C_CORRECTIONS","phone.route_identity","decision","Yearly instance plus reusable reference",1),
      row("C_CORRECTIONS","phone.assignment_authority","decision","Planner prepares; operator activates",1),
      row("C_CORRECTIONS","phone.daily_workflow","decision","Next morning handles open work",1)
    ]
    return {"simulation_only":True,"autoapproval":False,"count":len(items),"candidates":items}

class RetrospectiveSandboxTests(unittest.TestCase):
    def test_engineering_continuity_with_disposable_review(self):
        outcome=replay(dataset())
        self.assertTrue(outcome["replay_invariants_passed"])
        self.assertEqual(outcome["disposable_synthetic_approvals"],7)
        self.assertEqual(outcome["approved_in_real_user_memory"],0)
        self.assertFalse(outcome["persisted_private_project_database"])
        self.assertEqual(outcome["context_delta"]["early_facets"],1)
        self.assertEqual(outcome["context_delta"]["later_facets"],4)
        self.assertEqual(outcome["stages"]["C_CORRECTIONS"]["historical_version_count"],7)
    def test_no_case_can_be_misrepresented_as_verified_approval(self):
        d=dataset()
        d["candidates"][0]["review_state"]="APPROVED"
        with self.assertRaisesRegex(ValueError,"Real user approvals cannot be simulated"):
            replay(d)
    def test_rejects_input_not_explicitly_marked_disposable(self):
        d=dataset();d["simulation_only"]=False
        with self.assertRaisesRegex(ValueError,"simulation-only"):
            replay(d)
    def test_future_facet_in_early_snap_fails_invariant(self):
        d=dataset()
        d["candidates"][0]["record"]["facet"]="phone.route_identity"
        d["candidates"][0]["record"]["kind"]="decision"
        # Later initial event cannot be an independent r0 and must not pass.
        with self.assertRaises(Exception):
            replay(d)

if __name__=="__main__":
    unittest.main(verbosity=2)
