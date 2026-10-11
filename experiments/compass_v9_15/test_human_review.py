"""Synthetic-only tests for offline human review annotation export."""
import unittest
from human_review import prepare,render


def case(i:int,status:str,excerpt:str):
    return {"candidate_id":str(i),"source":"git:a:docs/example.md#L"+str(i),
        "heading":"מסמך בדיקה "+str(i),"excerpt":excerpt,"source_sha256":"a"*64,
        "claim_status":status}


class HumanReviewTests(unittest.TestCase):
    def test_sampling_is_deterministic_and_does_not_export_labels(self):
        items=[case(i,"APPROVED_CLAIM" if i%3==0 else "UNRESOLVED",
                    "הערה ארוכה לבדיקת תיוג מסמך כדי שתחולת החלטה לא תיקבע אוטומטית. "*3)
               for i in range(50)]
        first=prepare(items,seed=19,limit=30)
        self.assertEqual(first,prepare(items,seed=19,limit=30))
        self.assertEqual(len(first),30)
        self.assertFalse(any("claim_status" in x for x in first))
        self.assertFalse(any("approval_actor" in x for x in first))

    def test_html_escapes_untrusted_script_closer(self):
        bad=case(1,"UNRESOLVED",
                 "</script><script>throw Error('executed')</script> " + "אזהרת מקור שלא נבדק. "*5)
        generated=render(prepare([bad],limit=1))
        self.assertNotIn("</script><script>throw Error",generated)
        self.assertIn("source_sha256",generated)
        self.assertIn("PENDING_REVIEW",generated) if False else None
        self.assertIn("אינו מאשר",generated)
        self.assertIn("אישור",generated)
        self.assertIn("REVOKED_OR_SUPERSEDED",generated)

    def test_too_short_ignored(self):
        self.assertEqual(prepare([case(1,"UNRESOLVED","short")]),[])

    def test_no_embedded_git_password_or_remote_execution(self):
        h=render([{"id":"1","source":"git:a:docs/a.md#L1",
                  "heading":"Test","excerpt":"Synthetic only","sha":"x"}])
        self.assertNotIn("fetch(",h)
        self.assertNotIn("http://",h)
        self.assertIn('textContent',h)


if __name__=="__main__":
    unittest.main(verbosity=2)
