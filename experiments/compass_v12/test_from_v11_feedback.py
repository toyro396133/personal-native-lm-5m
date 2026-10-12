"""V12 adapter reading V11.1 review exports, with invented Hebrew feedback only."""
import copy
import unittest

from from_v11_feedback import compile_training_pairs, pair_from_feedback, fingerprint

SOURCE="chat:session0001:message0001"

def event(i=1,reply="לא למחוק את ההיסטוריה הקודמת, אלא לשמור את המידע משנים קודמות"):
    return {
        "id":f"feedback.{i}","scope":"project","project_id":"fake-project",
        "speaker":"user","source":SOURCE,"source_verified":True,
        "question":"מה הדרך להמשיך וללמוד משנים קודמות?",
        "proposal":"יש למחוק מידע היסטורי ולבנות מחדש הכול",
        "user_reply":reply
    }
def review(e):
    return {"event_id":e["id"],"event_hash":fingerprint(e),
            "action":"CONFIRM_CORRECTION","confirmed_by_user":True,
            "rejected":"למחוק את כל המידע שנלמד בשנים קודמות",
            "preferred":"לשמר את המידע ולשפר אותו לאורך השנים",
            "reviewer_note":"אישרתי את משמעות דברי המשתמש מול המקור"}

class V12ImportTests(unittest.TestCase):
    def test_direct_freeform_user_contrast_becomes_training_only(self):
        x=pair_from_feedback(event(),user_id="fictional-user")
        self.assertEqual(x["chosen_index"],1)
        self.assertEqual(x["schema"],"COMPASS_V12_TRAINING_PAIR_NOT_EVALUATION")
        self.assertEqual(x["feedback"]["source"],SOURCE)
        self.assertFalse(x["prospective_holdout_eligible"])
        self.assertFalse(x["project_policy_approved"])

    def test_unclear_feedback_never_becomes_label(self):
        x=pair_from_feedback(event(reply="נראה לי שכדאי אולי לשמור מידע קודם"),user_id="fictional-user")
        self.assertIsNone(x)

    def test_faithful_human_review_creates_training_signal(self):
        e=event(2,reply="היה צריך לבדוק אחרת")
        x=pair_from_feedback(e,user_id="fictional-user",review=review(e))
        self.assertEqual(x["candidates"][0],review(e)["rejected"])
        self.assertEqual(x["candidates"][1],review(e)["preferred"])
        self.assertEqual(x["review_status"],"USER_CONFIRMED")

    def test_stale_review_refused_after_original_user_reply_changes(self):
        e=event(3)
        signed=review(e)
        changed=copy.deepcopy(e);changed["user_reply"]="משתמש כתב הודעה אחרת"
        with self.assertRaisesRegex(ValueError,"not bound"):
            pair_from_feedback(changed,user_id="fictional-user",review=signed)

    def test_train_queue_reports_noneligible_and_no_evaluation(self):
        events=[event(1),event(2,"לא בטוח איך נשמר הידע")]
        q=compile_training_pairs(events,user_id="fictional-user")
        self.assertEqual(q["trainable_pairs"],1)
        self.assertEqual(q["evaluation_pairs"],0)
        self.assertEqual(q["unresolved_event_ids"],["feedback.2"])

    def test_user_rejected_or_unverified_source_stays_untrained(self):
        e=event()
        e["source_verified"]=False
        self.assertIsNone(pair_from_feedback(e,user_id="fictional-user"))
        e=event()
        e["speaker"]="assistant"
        with self.assertRaises(ValueError):pair_from_feedback(e,user_id="fictional-user")

    def test_no_orphan_human_review(self):
        with self.assertRaisesRegex(ValueError,"Orphan"):
            compile_training_pairs([event(1)],user_id="fictional-user",
                                   reviews=[review(event(5))])

    def test_no_duplicate_id_same_scope(self):
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            compile_training_pairs([event(),event()],user_id="fictional-user")

    def test_personal_scope_not_inherited_from_project(self):
        e=event();e["scope"]="personal"
        with self.assertRaises(ValueError):pair_from_feedback(e,user_id="fictional-user")

if __name__=="__main__":
    unittest.main(verbosity=2)
