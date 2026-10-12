"""Synthetic local-chat evidence ingestion tests; no real user chats uploaded."""
import unittest
from chat_ingest import from_chat
from freeform_feedback import CorrectionCompass, validate, literal_correction, event_fingerprint

def msg(n,role,text,session="session001"):
    return {"session_id":session,"id":f"message{n:03d}","role":role,"text":text}

def sample():
    return [
        msg(1,"user","איך ממשיכים ללמוד מפרויקטים לאורך זמן?"),
        msg(2,"assistant","נמחק את ההחלטות הקודמות ונשתמש רק בקוד החדש"),
        msg(3,"user","לא למחוק את ההחלטות הקודמות, אלא לשמר אותן עם הסברים"),
        msg(4,"assistant","אז אפשר להשתמש רק במידע העדכני"),
        msg(5,"user","טעות! צריך לשמור גם את המידע ההיסטורי עם ההקשר")
    ]

class IngestTests(unittest.TestCase):
    def test_two_actual_chat_feedback_pairs(self):
        events,audit=from_chat(sample(),project_id="sample-project")
        self.assertEqual(audit["emitted_events"],2)
        self.assertEqual([e["id"] for e in events],["message003","message005"])
        self.assertEqual(events[0]["source"],"chat:session001:message003")
        self.assertEqual(events[0]["proposal_source"],"chat:session001:message002")
        self.assertEqual(events[0]["user_reply"],sample()[2]["text"])
        self.assertEqual(events[0]["question"],sample()[0]["text"])
        self.assertEqual(audit["labels_extracted"],0)
        self.assertFalse(audit["autoapproved"])

    def test_extracted_original_chat_events_are_valid_compass_inputs(self):
        events,_=from_chat(sample(),project_id="sample-project")
        for event in events:
            validate(event)
            self.assertIsNotNone(literal_correction(event))
            self.assertEqual(event["source_authentication"],
                             "LOCAL_ROLE_TAGGED_EXPORT_NOT_CRYPTOGRAPHIC")

    def test_no_external_labels_but_online_compass_can_update_after_reply(self):
        events,_=from_chat(sample(),project_id="sample-project")
        model=CorrectionCompass()
        a=model.observe(events[0]);b=model.observe(events[1])
        self.assertEqual(a["training_events_after"],1)
        self.assertEqual(b["training_events_after"],2)
        self.assertFalse(a["autoapproved"])
        self.assertEqual(model.scopes["project:sample-project"]["training_events"],2)

    def test_conversation_corrections_need_not_be_one_letter(self):
        e,_=from_chat(sample(),project_id="sample-project")
        self.assertTrue(all(len(x["user_reply"])>20 for x in e))
        self.assertTrue(all(literal_correction(x) for x in e))

    def test_human_review_fingerprint_binds_original_chat_pair(self):
        e,_=from_chat(sample(),project_id="sample-project")
        old=event_fingerprint(e[0])
        e[0]["proposal"]="שינוי מאוחר של הפירוש הישן"
        self.assertNotEqual(event_fingerprint(e[0]),old)

    def test_boundary_between_sessions_is_not_assumed_continuous(self):
        ms=sample()
        ms[2]["session_id"]="session002"
        e,a=from_chat(ms,project_id="sample-project")
        self.assertEqual(len(e),0)
        self.assertGreater(a["skipped"]["session_boundary"],0)

    def test_tool_or_system_interruption_breaks_triplet(self):
        ms=[msg(1,"user","שאלה"),msg(2,"tool","תוצאת כלי"),
            msg(3,"assistant","הצעה"),msg(4,"user","טעות! לשמור מידע") ]
        e,a=from_chat(ms,project_id="sample-project")
        self.assertEqual(len(e),0)

    def test_freeform_unclear_is_evidence_without_weight_update(self):
        ms=[msg(1,"user","שאלה"),msg(2,"assistant","הצעה"),
            msg(3,"user","אני עדיין לא בטוח, צריך לעיין שוב")]
        e,_=from_chat(ms,project_id="sample-project")
        self.assertEqual(len(e),1)
        r=CorrectionCompass().observe(e[0])
        self.assertFalse(r["eligible"])
        self.assertFalse(r["autoapproved"])

    def test_too_long_messages_are_skipped_not_silently_truncated(self):
        ms=[msg(1,"user","שאלה"*800),msg(2,"assistant","הצעה"),
            msg(3,"user","טעות! צריך לשנות את ההצעה")]
        e,a=from_chat(ms,project_id="sample-project")
        self.assertEqual(len(e),0)
        self.assertEqual(a["skipped"]["text_too_long"],1)

    def test_duplicate_source_message_ids_rejected(self):
        ms=sample()
        ms[2]["id"]=ms[1]["id"]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            from_chat(ms,project_id="sample-project")

    def test_dont_infer_personal_scope_from_project_chat(self):
        with self.assertRaisesRegex(ValueError,"cannot silently"):
            from_chat(sample(),scope="personal",project_id="sample-project")

    def test_malformed_roles_rejected(self):
        ms=sample();ms[1]["role"]="admin"
        with self.assertRaisesRegex(ValueError,"Unsupported"):
            from_chat(ms,project_id="sample-project")

if __name__=="__main__":
    unittest.main(verbosity=2)
