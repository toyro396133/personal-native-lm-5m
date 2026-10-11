"""V9.15 synthetic-only tests: real-source classifier is NOT allowed to fake missing labels."""
import subprocess
import tempfile
import unittest
from pathlib import Path
from weak_classifier import (CharNB,grams,mask,keyword,evaluate,training_rows,CLASSES)


class WeakClassifierTests(unittest.TestCase):
    def test_headings_excluded_status_cues_masked(self):
        self.assertEqual(grams("החלטה מאושרת פרויקט מחייב"),
                         grams(" פרויקט "))
        self.assertNotIn("החלטה מאושרת",mask("החלטה מאושרת מסלול חדש"))

    def test_abort_when_training_has_no_proposal(self):
        cases=[{"truth":"APPROVED_CLAIM","body":"Some approved project reasons and actions"},
               {"truth":"HISTORICAL_EVIDENCE","body":"A past paper process and informal handover"}]
        with self.assertRaisesRegex(ValueError,"Not enough independent training classes"):
            CharNB(cases)

    def test_unseen_body_predict_never_autoapproves(self):
        cases=[]
        for name,body in [
          ("APPROVED_CLAIM","The committee confirmed stable ongoing interfaces with documented evidence."),
          ("PROPOSED_CLAIM","The proposed prototype is an idea awaiting a later discussion and review."),
          ("HISTORICAL_EVIDENCE","In the old workflow operators filled paper sheets without a unified archive."),
        ]:
            cases.append({"truth":name,"body":body})
        m=CharNB(cases)
        value,confidence=m.predict("Unknown future project context is not approved.")
        self.assertIn(value,(*CLASSES,"ABSTAIN"))
        self.assertGreaterEqual(confidence,0)
        self.assertLessEqual(confidence,1)
        self.assertEqual(m.predict("")[0],"ABSTAIN")

    def test_evaluation_preserves_abstentions_in_denominator(self):
        a=[{"truth":"APPROVED_CLAIM","body":"alpha"},
           {"truth":"PROPOSED_CLAIM","body":"beta"}]
        r=evaluate(a,lambda x:"ABSTAIN")
        self.assertEqual(r["covered"],0)
        self.assertEqual(r["correct"],0)
        self.assertEqual(r["n"],2)
        self.assertIsNone(r["selective_accuracy"])

    def test_test_labels_cannot_be_assumed_sufficient(self):
        a=[{"truth":"APPROVED_CLAIM","body":"alpha"},
           {"truth":"APPROVED_CLAIM","body":"beta"}]
        r=evaluate(a,lambda x:"APPROVED_CLAIM")
        self.assertEqual(r["correct"],2)
        self.assertEqual(r["macro_recall"],1)
        self.assertEqual(r["by_class"]["PROPOSED_CLAIM"],{})

    def test_keyword_control_receives_body_only(self):
        self.assertEqual(keyword("במקום הקוד הקיים זו טיוטה לבדיקה"),"PROPOSED_CLAIM")
        self.assertEqual(keyword("A new route system may need thought"),"ABSTAIN")


if __name__=="__main__":
    unittest.main(verbosity=2)
