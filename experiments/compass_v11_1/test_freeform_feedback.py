"""Synthetic engineering tests ONLY — do not infer real intent learning."""
import copy
import json
import unittest

from freeform_feedback import (CorrectionCompass, literal_correction,
                               replay, validate, SCHEMA)

SRC = "git:" + "1"*40 + ":docs/fictional-feedback.md#L8-L16"

def event(n=0, reply=None, *, project="project-one", scope="project",
          verified=True, proposal="מחק את כל הידע שנצבר מהשנים הקודמות",
          question="איך להתייחס למידע מתקופות קודמות?", source=SRC):
    if reply is None:
        reply = "לא למחוק את הידע ההיסטורי, אלא לשמור את הידע ולשפר אותו"
    return {"id":f"feedback.{n}", "scope":scope, "project_id":project,
            "speaker":"user", "source":source, "question":question,
            "proposal":proposal, "user_reply":reply,
            "source_verified":verified}

class CorrectionsTests(unittest.TestCase):
    def test_literal_contrast_extracted_without_teacher(self):
        p=literal_correction(event())
        self.assertEqual(p["style"],"EXPLICIT_NOT_BUT")
        self.assertIn("לשמור",p["preferred"])
        self.assertIn("למחוק",p["rejected"])
        self.assertFalse(p["is_policy_approval"])

    def test_explicit_error_with_written_prior_proposal(self):
        p=literal_correction(event(reply="טעות! חייבים לשמר את ההיסטוריה בכל שנה"))
        self.assertEqual(p["style"],"EXPLICIT_REPAIR")
        self.assertEqual(p["rejected"],event()["proposal"])

    def test_explicit_intended_correction(self):
        p=literal_correction(event(reply="התכוונתי לשמור את ההיסטוריה גם בשנה החדשה"))
        self.assertEqual(p["style"],"EXPLICIT_REPAIR")

    def test_simple_negative_has_no_positive_signal(self):
        for reply in ("טעות!", "לא בדקת נכון", "זה לא מה שסיכמנו"):
            with self.subTest(reply=reply):
                self.assertIsNone(literal_correction(event(reply=reply)))

    def test_ambiguous_hedges_rejected(self):
        for reply in ("נראה לי שכן",
                      "לא למחוק את הדוח, אלא אולי לשמור אותו",
                      "לא למחוק את התוצאות, אלא אולי לשמור אותן"):
            with self.subTest(reply=reply):
                self.assertIsNone(literal_correction(event(reply=reply)))

    def test_not_only_but_also_is_not_reversal(self):
        self.assertIsNone(literal_correction(event(
            reply="לא רק לשמור תוצאות אלא גם ללמוד מהן")))

    def test_missing_actual_source_verification_stores_only(self):
        m=CorrectionCompass()
        r=m.observe(event(verified=False))
        self.assertFalse(r["eligible"])
        self.assertEqual(r["training_events_after"],0)
        self.assertEqual(len(m.scopes["project:project-one"]["history"]),1)
        self.assertFalse(r["autoapproved"])

    def test_first_correction_is_pretrain_abstention(self):
        m=CorrectionCompass()
        r=m.observe(event(0))
        self.assertEqual(r["before"]["status"],"ABSTAIN_COLD_START")
        self.assertEqual(r["training_events_after"],1)
        self.assertEqual(r["comparison_type"],"RETROSPECTIVE_PAIR_RANKING_ONLY")

    def test_writes_trainable_weights_without_policy_approval(self):
        m=CorrectionCompass()
        m.observe(event(0))
        m.observe(event(1,reply="טעות! צריך לשמור נתונים ולעדכן את המידע כל שנה"))
        self.assertTrue(m.scopes["project:project-one"]["weights"])
        self.assertEqual(m.scopes["project:project-one"]["training_events"],2)
        self.assertTrue(m.export()["no_policy_approval"])

    def test_can_rank_future_predefined_options_without_reply(self):
        m=CorrectionCompass()
        for n in range(3): m.observe(event(n))
        later=m.rank("איך להתייחס למידע מתקופות קודמות?",
           ["למחוק את הידע ההיסטורי","לשמור את הידע ולשפר אותו"],
           scope="project",project_id="project-one")
        self.assertEqual(later["choice_index"],1)
        self.assertEqual(later["status"],"UNVERIFIED_PREFERENCE_RANKING")
        self.assertTrue(later["sources"])

    def test_candidate_order_invariance(self):
        m=CorrectionCompass()
        for n in range(3):m.observe(event(n))
        a=m.rank("איך להתייחס למידע מתקופות קודמות?",
                 ["למחוק את הידע ההיסטורי","לשמור את הידע ולשפר אותו"],
                 scope="project",project_id="project-one")
        b=m.rank("איך להתייחס למידע מתקופות קודמות?",
                 ["לשמור את הידע ולשפר אותו","למחוק את הידע ההיסטורי"],
                 scope="project",project_id="project-one")
        self.assertEqual(a["choice_index"],1-b["choice_index"])

    def test_new_project_has_no_cross_learning(self):
        m=CorrectionCompass()
        for n in range(3):m.observe(event(n,project="project-one"))
        p=m.rank("איך להתייחס למידע מתקופות קודמות?",
             ["למחוק ידע ישן","לשמור ידע מצטבר"],
             scope="project",project_id="project-two")
        self.assertEqual(p["status"],"ABSTAIN_COLD_START")

    def test_personal_not_promoted_from_project(self):
        m=CorrectionCompass()
        for n in range(3):m.observe(event(n))
        p=m.rank("איך להתייחס למידע מתקופות קודמות?",
             ["למחוק ידע ישן","לשמור ידע מצטבר"],
             scope="personal",project_id=None)
        self.assertEqual(p["status"],"ABSTAIN_COLD_START")

    def test_personal_event_must_not_have_project(self):
        with self.assertRaises(ValueError):
            validate(event(scope="personal",project="project-one"))

    def test_project_scope_can_have_same_event_id_independently(self):
        m=CorrectionCompass()
        m.observe(event(0,project="project-one"))
        m.observe(event(0,project="project-two"))
        self.assertEqual(len(m.scopes),2)

    def test_duplicate_same_scope_refused(self):
        m=CorrectionCompass();m.observe(event(0))
        with self.assertRaises(ValueError):
            m.observe(event(0))

    def test_no_external_gold_override(self):
        for key in ("gold","teacher_label","chosen_option","approved","preferred_label"):
            x=event();x[key]="APPROVED"
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate(x)

    def test_source_must_be_git_pinned_with_lines(self):
        for src in ("note-1","git:HEAD:docs/a.md#L1-L4",
                    "git:"+"2"*40+":docs/../private.md#L1-L4",
                    "git:"+"2"*40+":docs/a.md#L11-L9"):
            with self.subTest(src=src),self.assertRaises(ValueError):
                validate(event(source=src))

    def test_speaker_must_be_user(self):
        x=event();x["speaker"]="assistant"
        with self.assertRaises(ValueError):validate(x)

    def test_rejected_and_preferred_must_have_substance(self):
        for reply in ("לא א אלא ב","לא כן אלא לא", "לא לשמור מידע אלא לשמור מידע"):
            with self.subTest(reply=reply):
                self.assertIsNone(literal_correction(event(reply=reply)))

    def test_unlabelled_does_not_change_existing_weights(self):
        m=CorrectionCompass();m.observe(event(0))
        old=copy.deepcopy(m.scopes["project:project-one"]["weights"])
        r=m.observe(event(1,reply="אני לא בטוח שהשאלה מנוסחת נכון"))
        self.assertFalse(r["eligible"])
        self.assertEqual(old,m.scopes["project:project-one"]["weights"])

    def test_history_retains_original_user_words(self):
        e=event()
        m=CorrectionCompass();m.observe(e)
        self.assertEqual(m.scopes["project:project-one"]["history"][0]["user_reply"],
                         e["user_reply"])

    def test_checkpoint_round_trip_with_learning_state(self):
        m=CorrectionCompass()
        for n in range(3):m.observe(event(n))
        clone=CorrectionCompass.restore(json.loads(json.dumps(m.export())))
        task="איך להתייחס למידע מתקופות קודמות?"
        candidates=["למחוק את הידע ההיסטורי","לשמור את הידע ולשפר אותו"]
        self.assertEqual(m.rank(task,candidates,scope="project",project_id="project-one"),
                         clone.rank(task,candidates,scope="project",project_id="project-one"))

    def test_invalid_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            CorrectionCompass.restore({"schema":SCHEMA, "no_policy_approval":False,
                                       "training_source":"EXPLICIT_USER_REPAIRS_ONLY"})

    def test_replay_zero_cases_does_not_fake_accuracy(self):
        result, _=replay([event(0,reply="יש לעיין שוב בבקשה")])
        self.assertEqual(result["eligible_explicit_corrections"],0)
        self.assertEqual(result["scores"]["learned"]["attempts"],0)
        self.assertIsNone(result["scores"]["learned"]["accuracy_when_answered"])

    def test_replay_reports_small_real_data_gate(self):
        result, _=replay([event(i) for i in range(5)])
        self.assertEqual(result["research_status"],"INSUFFICIENT_REAL_CORRECTION_FEEDBACK")
        self.assertEqual(result["scores"]["frozen"]["attempts"],0)
        self.assertEqual(result["events_seen"],5)

if __name__=="__main__":
    unittest.main(verbosity=2)
