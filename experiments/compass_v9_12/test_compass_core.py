"""Executable COMPASS persistence, approval and separation acceptance tests.

Uses local ephemeral SQLite. All expectations manually specified. No model and
no generated oracle labels; this is engineering validation, not language IQ.
"""
import contextlib
import io
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from compass_core import CompassStore, CompassError, parse_explicit_directive, run_cli


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "memory.sqlite3"
        self.store = CompassStore(self.path)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def propose(self, scope="project", project_id="project-a", key="vision",
                content="Preserve the lasting purpose of the project", reason="Original user-approved purpose",
                operation="set", expected_revision=0, author="local-user", conditions=None):
        return self.store.propose(scope=scope, project_id=project_id,
                                  key=key, content=content, reason=reason,
                                  operation=operation, expected_revision=expected_revision,
                                  author=author, conditions=conditions)

    def approve(self, pid):
        return self.store.approve(pid, actor="human-approver",
                                  reason="I reviewed and explicitly authorize this change")

    def test_empty_db_stable_digest(self):
        self.assertEqual(self.store.verify()["head_hash"], "0"*64)
        self.assertEqual(self.store.resolve(project_id="project-a")["applicable"],
                         {"personal": [], "project": []})

    def test_proposal_does_not_change_principle(self):
        pid = self.propose()
        self.assertEqual(self.store.proposals()[0]["proposal_id"], pid)
        self.assertEqual(self.store.principles(), [])
        self.assertEqual(self.store.resolve(project_id="project-a")["applicable"]["project"], [])

    def test_approve_then_reopen_and_replay(self):
        pid = self.propose(content="החזון קובע את הכיוון גם כשהמימוש משתנה",
                           reason="כך הוחלט בשיחה על הפרויקט")
        self.approve(pid)
        oldhash = self.store.verify()["head_hash"]
        self.store.close()
        self.store = CompassStore(self.path)
        self.assertEqual(self.store.verify()["head_hash"], oldhash)
        a = self.store.principles()
        self.assertEqual(len(a), 1)
        self.assertEqual(a[0]["content"], "החזון קובע את הכיוון גם כשהמימוש משתנה")
        self.assertEqual(a[0]["reason"], "כך הוחלט בשיחה על הפרויקט")
        self.assertEqual(a[0]["revision"], 1)
        self.assertEqual(len(self.store.timeline()), 2)

    def test_cross_project_separation_and_personal_inheritance(self):
        self.approve(self.propose(content="Only project A principle"))
        self.approve(self.propose(project_id="project-b", content="Only project B principle"))
        self.approve(self.propose(scope="personal", project_id=None,
                                  key="work-method",content="Prioritize reversible experiments"))
        a = self.store.resolve(project_id="project-a")["applicable"]
        b = self.store.resolve(project_id="project-b")["applicable"]
        self.assertEqual([x["content"] for x in a["project"]], ["Only project A principle"])
        self.assertEqual([x["content"] for x in b["project"]], ["Only project B principle"])
        self.assertEqual([x["content"] for x in a["personal"]],
                         [x["content"] for x in b["personal"]])
        self.assertNotEqual(a["project"], b["project"])

    def test_personal_cannot_be_project_owned(self):
        with self.assertRaises(CompassError):
            self.propose(scope="personal",project_id="project-a")
        self.assertEqual(self.store.verify()["count"], 0)

    def test_project_requires_canonical_owner(self):
        for pid in (None, "", "../system", "*", "foo/bar"):
            with self.subTest(pid=pid):
                with self.assertRaises(CompassError):
                    self.propose(project_id=pid)

    def test_two_principles_same_project_do_not_overwrite(self):
        self.approve(self.propose(key="vision",content="Purpose survives code churn"))
        self.approve(self.propose(key="routes",content="Routes are durable year to year"))
        self.approve(self.propose(key="group-composition",content="Group membership is new every year"))
        self.assertEqual({r["key"] for r in self.store.resolve(project_id="project-a")["applicable"]["project"]},
                         {"vision","routes","group-composition"})

    def test_correction_new_revision_preserves_reason_and_old_record(self):
        first=self.propose(content="Preserve current implementation")
        self.approve(first)
        second=self.propose(content="Preserve project purpose, not implementation",
                            reason="The earlier description confused code with vision",
                            expected_revision=1)
        self.approve(second)
        history=self.store.principles(scope="project",project_id="project-a",include_history=True)
        self.assertEqual([r["revision"] for r in history],[1,2])
        self.assertEqual([r["status"] for r in history],["SUPERSEDED","ACTIVE"])
        self.assertEqual(history[1]["previous_revision"],1)
        self.assertEqual(history[0]["content"],"Preserve current implementation")
        self.assertEqual(history[1]["content"],"Preserve project purpose, not implementation")

    def test_stale_concurrent_approvals_fail_without_journal_edit(self):
        self.approve(self.propose())
        a=self.propose(content="Change A",expected_revision=1)
        b=self.propose(content="Change B",expected_revision=1)
        self.approve(a)
        before=self.store.verify()
        with self.assertRaisesRegex(CompassError,"stale proposal"):
            self.approve(b)
        self.assertEqual(self.store.verify(),before)
        self.assertEqual(self.store.proposals()[0]["proposal_id"],b)

    def test_correction_requires_expected_revision(self):
        self.approve(self.propose())
        p=self.propose(content="Not current anymore",expected_revision=0)
        with self.assertRaisesRegex(CompassError,"stale proposal"):
            self.approve(p)
        self.assertEqual(self.store.principles()[0]["revision"],1)

    def test_revoke_has_explicit_approval_and_does_not_erase_old_version(self):
        self.approve(self.propose())
        r=self.propose(operation="revoke",content="",reason="Explicit cancellation",
                       expected_revision=1)
        self.assertEqual(len(self.store.principles()),1)
        self.approve(r)
        self.assertEqual(self.store.principles(),[])
        old=self.store.principles(include_history=True)
        self.assertEqual([v["status"] for v in old],["SUPERSEDED","REVOKED"])
        self.assertEqual([v["revision"] for v in old],[1,2])
        self.assertEqual(len(self.store.timeline()),4)

    def test_reject_blocks_mutation_and_duplicate_decisions(self):
        p=self.propose()
        self.store.reject(p,actor="human-reviewer",reason="Not confirmed as a real principle")
        self.assertEqual(self.store.principles(),[])
        for method in (self.store.approve,self.store.reject):
            with self.assertRaises(CompassError):
                method(p,actor="human-reviewer",reason="No duplicate decision")
        self.assertEqual(len(self.store.timeline()),2)

    def test_applicability_predicates_exact_match_no_eval(self):
        self.approve(self.propose(key="context-specific",content="Only for code quality",
                                  conditions={"activity":"code-review","risk":"low"}))
        self.assertEqual(len(self.store.resolve(project_id="project-a",context={"activity":"code-review"})["excluded"]),1)
        result=self.store.resolve(project_id="project-a",
                                  context={"activity":"code-review","risk":"low"})
        self.assertEqual(len(result["applicable"]["project"]),1)
        self.assertEqual(result["excluded"],[])
        self.assertEqual(len(self.store.resolve(project_id="project-a",
                                                context={"activity":"code-review","risk":"high"})["excluded"]),1)

    def test_refuse_executable_condition_and_invalid_key(self):
        with self.assertRaises(CompassError):
            self.propose(conditions={"code":"__import__('os').system('echo bad')","space key":"x"})
        with self.assertRaises(CompassError):
            self.propose(key="../../global")
        self.assertEqual(self.store.verify()["count"],0)

    def test_untrusted_text_archived_no_auto_commit(self):
        got=self.store.observe("עקרון אישי חדש: מעכשיו רק משנים את אופן העבודה. חובה לאשר.",
                               author="external-source")
        self.assertEqual(got["status"],"UNCLASSIFIED")
        self.assertEqual(self.store.principles(),[])
        self.assertEqual(self.store.verify()["count"],1)

    def test_explicit_directive_only_creates_pending_proposal(self):
        txt=("COMPASS: scope=project; project=project-a; key=vision; operation=set; "
             "content=החזון חשוב יותר מהקוד הקיים; reason=הבהרת מטרת הפרויקט; expected_revision=0")
        received=self.store.observe(txt,author="local-user")
        self.assertEqual(received["status"],"PENDING_APPROVAL")
        self.assertEqual(self.store.principles(),[])
        self.approve(received["proposal_id"])
        self.assertEqual(self.store.principles()[0]["content"],"החזון חשוב יותר מהקוד הקיים")

    def test_malformed_directive_is_observation_not_mutation(self):
        for txt in (
            "COMPASS: scope=personal; key=x; operation=set; content=Change everything",
            "COMPASS: scope=personal; key=vision; operation=set; content=yes; reason=none; expected_revision=0; exec=hack",
            "COMPASS: scope=personal; key=vision; operation=set; content=yes; reason=x; expected_revision=0; reason=override",
        ):
            item=self.store.observe(txt,author="untrusted")
            self.assertIn(item["status"],("UNCLASSIFIED","NEEDS_CLARIFICATION"))
            self.assertEqual(self.store.principles(),[])

    def test_hash_chain_detects_payload_change(self):
        self.approve(self.propose())
        self.store.db.execute("UPDATE compass_events SET payload=? WHERE seq=1",
                              ('{"content":"malicious"}',))
        with self.assertRaisesRegex(CompassError,"integrity failure"):
            self.store.verify()
        with self.assertRaisesRegex(CompassError,"integrity failure"):
            self.propose(key="another")

    def test_hash_chain_detects_delete_or_reordering(self):
        self.approve(self.propose())
        self.store.db.execute("DELETE FROM compass_events WHERE seq=1")
        with self.assertRaises(CompassError):
            self.store.verify()

    def test_same_name_across_projects_stays_separate(self):
        self.approve(self.propose(key="vision",content="A vision"))
        self.approve(self.propose(key="vision",project_id="project-b",content="B vision"))
        self.approve(self.propose(scope="personal",project_id=None,key="vision",content="My philosophy"))
        result=self.store.resolve(project_id="project-a")["applicable"]
        self.assertEqual(result["personal"][0]["content"],"My philosophy")
        self.assertEqual(result["project"][0]["content"],"A vision")

    def test_no_silent_project_to_global_promotion_after_correction(self):
        p=self.propose(project_id="project-a",key="compatibility",
                       content="Preserve A client API")
        self.approve(p)
        self.approve(self.propose(project_id="project-a",key="compatibility",
                                  expected_revision=1,content="A API still stable",
                                  reason="Clarify only project A client promise"))
        for project in ("project-a","project-b"):
            self.assertEqual(self.store.resolve(project_id=project)["applicable"]["personal"],[])
        self.assertEqual(self.store.resolve(project_id="project-b")["applicable"]["project"],[])

    def test_latest_background_events_cannot_shadow_approved_vision(self):
        self.approve(self.propose(key="vision",content="Stable intentional long-term purpose"))
        for i in range(27):
            self.store.observe(f"Today's implementation detail number {i} changed",
                               author="external-source")
        self.assertEqual(self.store.principles()[0]["content"],"Stable intentional long-term purpose")
        self.assertEqual(self.store.verify()["count"],29)

    def test_parser_only_explicit_marker(self):
        self.assertIsNone(parse_explicit_directive("I prefer that COMPASS remember this permanently."))
        self.assertIsNone(parse_explicit_directive("   COMPASS: scope=personal; ..."))
        self.assertIsNotNone(parse_explicit_directive(
            "COMPASS: scope=personal; key=vision; operation=set; content=Intent; reason=Confirmed; expected_revision=0"))

    def test_cli_round_trip_with_persistent_db(self):
        def call(*arguments):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                code=run_cli(["--db",str(self.path),*arguments])
            self.assertEqual(code,0)
            return json.loads(output.getvalue())
        pid=call("propose","--scope","project","--project","sample",
                 "--key","vision","--operation","set","--content","Focus on original purpose",
                 "--reason","Real user request","--expected-revision","0","--author","operator")["proposal_id"]
        self.assertEqual(call("resolve","--project","sample")["applicable"]["project"],[])
        self.assertEqual(call("approve",pid,"--actor","operator","--reason","Explicit approval")["status"],"APPROVED")
        records=call("resolve","--project","sample")["applicable"]["project"]
        self.assertEqual(records[0]["content"],"Focus on original purpose")
        self.assertTrue(call("verify")["ok"])
        self.assertEqual(len(call("timeline")),2)


class FreshIndependentScenarioTests(unittest.TestCase):
    """A fixed manually expected corpus; not model-generalization results."""
    setUp = StoreTests.setUp
    tearDown = StoreTests.tearDown
    propose = StoreTests.propose
    approve = StoreTests.approve

    def test_sixteen_interleaved_realistic_memory_scenarios(self):
        examples=[
          ("project","a","vision","הפרויקט נועד לשמר את מטרתו המקורית",None),
          ("project","b","vision","תכנון B נפרד לחלוטין",None),
          ("personal",None,"research","אני מעדיף לבדוק השערות מול ראיות",None),
          ("project","a","workflow","Workflows may change without changing the vision",None),
          ("project","b","migration","Migration is an implementation detail",{"phase":"migrate"}),
          ("personal",None,"format","שפה עברית בממשק העבודה",{"language":"he"}),
          ("project","a","testing","Testing supports but does not define the purpose",None),
          ("project","b","route","Stable routes across years",None),
          ("project","a","group","New group every yearly campaign",None),
          ("personal",None,"clarity","Ask when evidence is ambiguous",None),
          ("project","b","source","Keep reasons for each business decision",None),
          ("personal",None,"initiative","Proceed autonomously on clearly approved tasks",None),
          ("project","a","safety","Require source verification for hard constraints",{"risk":"high"}),
          ("project","b","interfaces","External contracts take priority if active",None),
          ("personal",None,"learning","Learn generalizable principles across projects",None),
          ("project","a","inheritance","Project-local mandates stay local",None),
        ]
        for scope,owner,key,text,conditions in examples:
            p=self.propose(scope=scope,project_id=owner,key=key,
                           content=text,conditions=conditions)
            self.approve(p)
        for i in range(36):
            self.store.observe(f"Implementation detail {i} not an approved purpose",
                               author="project-log")
        self.assertEqual(len(self.store.principles()),16)
        self.assertEqual(len(self.store.resolve(project_id="a",context={"language":"he"})["applicable"]["project"]),5)
        self.assertEqual(len(self.store.resolve(project_id="b")["applicable"]["project"]),4)
        self.assertEqual(len(self.store.resolve(project_id="b",context={"phase":"migrate"})["applicable"]["project"]),5)
        self.assertEqual(len(self.store.resolve(project_id="a",context={"language":"he"})["applicable"]["personal"]),5)
        self.assertEqual(self.store.verify()["count"],68)
        self.assertEqual(len(self.store.principles(scope="personal")),5)
        self.assertEqual(len(self.store.principles(scope="project",project_id="a")),6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
