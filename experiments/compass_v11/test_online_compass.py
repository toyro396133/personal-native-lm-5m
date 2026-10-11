"""Only fictional episodes: no private user data or teacher labels in public CI."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from online_compass import (ContinualCompass, validate_episode, explicit_reply,
                            features, evaluate, similarity)

SHA = "f" * 40
SRC = f"git:{SHA}:docs/fictional-review.md#L4-L18"


def episode(n, answer="ב", project="project-one", scope="project",
            question="Should this project preserve established knowledge or erase it?",
            options=None, source=SRC):
    return {
        "id":f"case.{n}","scope":scope,"project_id":project,
        "source":source,"question":question,"reply":answer,
        "options":options or [
            {"id":"א","text":"discard old lessons and overwrite history"},
            {"id":"ב","text":"preserve the historical knowledge and improve it"}
        ]
    }


class ContinualCompassTests(unittest.TestCase):
    def test_no_leakage_before_first_feedback(self):
        x=ContinualCompass().observe(episode(0))
        self.assertTrue(x["eligible"])
        self.assertEqual(x["prequential"]["status"],"ABSTAIN_COLD_START")
        self.assertEqual(x["model_learned"],1)

    def test_requires_more_than_one_prior_explicit_response(self):
        m=ContinualCompass()
        m.observe(episode(0))
        self.assertIsNone(m.predict(episode(1))["choice"])

    def test_learning_produces_real_changed_weights(self):
        m=ContinualCompass()
        m.observe(episode(0))
        m.observe(episode(1))
        a=m.predict(episode(2))
        self.assertEqual(a["choice"],"ב")
        self.assertGreater(a["margin"],0.03)
        self.assertTrue(m.scopes["project:project-one"]["weights"])
        self.assertEqual(len(a["support"]),2)

    def test_teacher_provided_gold_fields_forbidden(self):
        for key in ("gold","label","chosen_option_id","teacher_label","selected"):
            x=episode(0)
            x[key]="ב"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_episode(x)

    def test_short_option_exact_match_only(self):
        for answer in ("״ב״.", '"ב"', "ב"):
            self.assertEqual(explicit_reply(episode(0,answer=answer)),"ב")
        for answer in ("ב אבל יש הסתייגות","נראה לי ב","מסכים","כן","ב או א","לא"):
            self.assertIsNone(explicit_reply(episode(0,answer=answer)))

    def test_ambiguous_answer_is_never_training(self):
        m=ContinualCompass()
        x=m.observe(episode(0,answer="נראה לי ב, אבל ייתכן גם א"))
        self.assertFalse(x["eligible"])
        self.assertEqual(m.scopes["project:project-one"]["trained"],0)
        self.assertFalse(x["autoapproved"])

    def test_one_event_cannot_appear_twice(self):
        m=ContinualCompass()
        m.observe(episode(0))
        with self.assertRaises(ValueError):
            m.observe(episode(0))

    def test_project_isolation(self):
        m=ContinualCompass()
        for n in range(3):
            m.observe(episode(n))
        self.assertEqual(m.predict(episode(9,project="project-two"))["status"],
                         "ABSTAIN_COLD_START")
        self.assertNotIn("project:project-two", {k for k,v in m.scopes.items() if v["trained"]})

    def test_personal_scope_is_independent(self):
        m=ContinualCompass()
        for n in range(3):
            m.observe(episode(n,scope="personal",project=None))
        self.assertGreater(m.scopes["personal:*"]["trained"],1)
        self.assertEqual(m.predict(episode(8,project="project-one"))["status"],
                         "ABSTAIN_COLD_START")

    def test_cannot_misroute_personal_and_project_event(self):
        x=episode(0,scope="personal",project="project-one")
        with self.assertRaises(ValueError):
            validate_episode(x)

    def test_source_requires_pinned_git_document(self):
        for src in ("git:HEAD:docs/fictional.md#L1-L2",
                    "git:"+"f"*40+":docs/../secret.md#L1-L2",
                    "git:"+"f"*40+":private/secret.md#L1-L2",
                    "git:"+"f"*40+":docs/fictional.md#L5-L2"):
            with self.subTest(src=src),self.assertRaises(ValueError):
                validate_episode(episode(0,source=src))

    def test_checkpoint_round_trip(self):
        model=ContinualCompass()
        for n in range(3):
            model.observe(episode(n))
        snap=model.export()
        clone=ContinualCompass.restore(json.loads(json.dumps(snap)))
        self.assertEqual(model.predict(episode(10)),clone.predict(episode(10)))
        with self.assertRaises(ValueError):
            clone.observe(episode(0))

    def test_feedback_changes_learned_state_not_evidence_history(self):
        model=ContinualCompass()
        model.observe(episode(0,answer="ב"))
        old=copy.deepcopy(model.scopes["project:project-one"]["history"][0])
        old_weight=dict(model.scopes["project:project-one"]["weights"])
        model.observe(episode(1,answer="א"))
        self.assertEqual(model.scopes["project:project-one"]["history"][0],old)
        self.assertNotEqual(model.scopes["project:project-one"]["weights"],old_weight)

    def test_similarity_retrieval_is_independent_baseline(self):
        model=ContinualCompass()
        x=episode(0)
        result=model.observe(x)
        self.assertIsNone(result["retrieval_choice"])
        result2=model.observe(episode(1))
        self.assertIn(result2["retrieval_choice"],("א","ב"))

    def test_evaluate_all_cases_prequentially_without_fake_accuracy(self):
        results=evaluate([episode(n) for n in range(5)])
        self.assertEqual(results["episodes_seen"],5)
        self.assertEqual(results["eligible_direct_feedback"],5)
        self.assertEqual(results["research_verdict"],"INSUFFICIENT_REAL_DECISION_FEEDBACK")
        self.assertLess(results["by_method"]["learned"]["attempts"],5)
        self.assertGreater(results["by_method"]["learned"]["attempts"],0)

    def test_empty_eligibility_yields_no_score(self):
        results=evaluate([episode(n,answer="I need to rethink the premise") for n in range(3)])
        self.assertEqual(results["eligible_direct_feedback"],0)
        self.assertIsNone(results["by_method"]["learned"]["accuracy_when_answered"])
        self.assertEqual(results["by_method"]["learned"]["attempts"],0)

    def test_two_scopes_never_cross_reference_source(self):
        m=ContinualCompass()
        for n in range(3):
            m.observe(episode(n,project="project-one",source=SRC))
        for n in range(3):
            m.observe(episode(n,project="project-two",
                source=f"git:{'e'*40}:docs/other.md#L1-L8"))
        pred=m.predict(episode(7,project="project-two"))
        self.assertTrue(all(x["source"].startswith("git:"+("e"*40)) for x in pred["support"]))

    def test_deterministic_features(self):
        a=features("רציפות עבודה", "שימור תוצאות")
        b=features("רציפות עבודה", "שימור תוצאות")
        self.assertEqual(a,b)
        self.assertEqual(similarity("עבר עבר", "עבר"),1)

    def test_unknown_text_does_not_create_policy_approval(self):
        m=ContinualCompass()
        for n in range(3):
            x=m.observe(episode(n))
            self.assertFalse(x["autoapproved"])
        self.assertTrue(m.export()["no_autoapproval"])

    def test_unrelated_event_is_retained_without_supervised_update(self):
        m=ContinualCompass()
        m.observe(episode(0))
        before=copy.deepcopy(m.scopes["project:project-one"]["weights"])
        m.observe(episode(1,answer="Maybe; this was not decided"))
        self.assertEqual(before,m.scopes["project:project-one"]["weights"])
        self.assertEqual(len(m.scopes["project:project-one"]["history"]),2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
