"""Real loopback HTTP acceptance tests for the COMPASS V9.12b local UI."""
import contextlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request,urlopen

from compass_ui import make_server
from compass_core import CompassStore


class UIProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=Path(self.temp.name)/"compass-browser.sqlite3"
        self.server=make_server(self.db,0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.url=f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=4)
        self.temp.cleanup()

    def call(self,path,body=None,token=True,origin=None,host=None,
             mime="application/json",expect=200):
        url=self.url+path
        headers={}
        if host is not None:headers["Host"]=host
        if origin is not None:headers["Origin"]=origin
        if body is not None:
            headers["Content-Type"]=mime
            if token:headers["X-COMPASS-SESSION"]=self.server.session_token
            payload=json.dumps(body,ensure_ascii=False).encode()
            request=Request(url,payload,headers=headers,method="POST")
        else:
            request=Request(url,headers=headers)
        try:
            with urlopen(request,timeout=5) as response:
                status=response.status
                data=response.read()
                ctype=response.headers.get("Content-Type","")
        except HTTPError as error:
            status=error.code
            data=error.read()
            ctype=error.headers.get("Content-Type","")
        self.assertEqual(status,expect,(path,data[:1000]))
        return data.decode() if "text/html" in ctype else json.loads(data)

    def proposed(self,scope="project",project="a",key="vision",version=0,content="Original project purpose"):
        return {"scope":scope,"project_id":project if scope=="project" else None,
                "key":key,"operation":"set","content":content,
                "reason":"Human stated and explained this vision",
                "expected_revision":version,"author":"test-actor","source":"local-browser-test"}

    def test_local_binding_not_network(self):
        self.assertEqual(self.server.server_address[0],"127.0.0.1")

    def test_page_has_rtl_and_no_data_before_action(self):
        html=self.call("/")
        self.assertIn('lang="he" dir="rtl"',html)
        self.assertIn(self.server.session_token,html)
        self.assertIn("אין אישור אוטומטי",html)
        state=self.call("/api/state?project=a")
        self.assertEqual(state["audit"]["count"],0)
        self.assertEqual(state["resolved"]["applicable"]["personal"],[])

    def test_missing_and_incorrect_session_tokens_rejected(self):
        self.call("/api/ingest",{"text":"abc","author":"xx"},token=False,expect=403)
        good=self.server.session_token
        self.server.session_token="changed-token"
        self.call("/api/ingest",{"text":"abc","author":"xx"},
                  token=False,expect=403)
        self.server.session_token=good
        self.assertEqual(self.call("/api/state?project=a")["audit"]["count"],0)

    def test_untrusted_host_and_cross_origin_rejected(self):
        self.call("/api/state?project=a",host="attacker.example",expect=403)
        self.call("/api/propose",self.proposed(),host="attacker.example",expect=403)
        self.call("/api/propose",self.proposed(),
                  origin="https://evil.example",expect=403)
        self.assertEqual(self.call("/api/state?project=a")["audit"]["count"],0)

    def test_wrong_content_type_rejected(self):
        self.call("/api/propose",self.proposed(),mime="text/plain",expect=415)

    def test_propose_pending_approve_live_and_reopen(self):
        created=self.call("/api/propose",self.proposed())
        pid=created["proposal_id"]
        self.assertEqual(created["status"],"PENDING_APPROVAL")
        state=self.call("/api/state?project=a")
        self.assertEqual(len(state["pending"]),1)
        self.assertEqual(len(state["resolved"]["applicable"]["project"]),0)
        approved=self.call("/api/approve",{"proposal_id":pid,
              "actor":"local-test","reason":"Explicit approval with a rationale"})
        self.assertEqual(approved["status"],"APPROVED")
        s=self.call("/api/state?project=a")
        self.assertEqual(len(s["pending"]),0)
        self.assertEqual(s["resolved"]["applicable"]["project"][0]["content"],"Original project purpose")
        self.assertEqual(s["audit"]["count"],2)
        with CompassStore(self.db) as store:
            self.assertEqual(store.resolve(project_id="a")["applicable"]["project"][0]["key"],"vision")

    def test_correction_preserves_history_and_other_project_separation(self):
        first=self.call("/api/propose",self.proposed())["proposal_id"]
        self.call("/api/approve",{"proposal_id":first,"actor":"reviewer","reason":"yes"})
        change=self.call("/api/propose",self.proposed(version=1,content="Purpose survives code changes"))["proposal_id"]
        self.assertEqual(self.call("/api/state?project=a")["resolved"]["applicable"]["project"][0]["revision"],1)
        self.call("/api/approve",{"proposal_id":change,"actor":"reviewer","reason":"accepted correction"})
        other=self.call("/api/state?project=b")
        self.assertEqual(other["resolved"]["applicable"]["project"],[])
        now=self.call("/api/state?project=a")
        self.assertEqual(now["resolved"]["applicable"]["project"][0]["revision"],2)
        with CompassStore(self.db) as store:
            old=store.principles(scope="project",project_id="a",include_history=True)
            self.assertEqual([x["status"] for x in old],["SUPERSEDED","ACTIVE"])

    def test_reject_and_stale_revision_failure(self):
        old=self.call("/api/propose",self.proposed())["proposal_id"]
        self.call("/api/approve",{"proposal_id":old,"actor":"reviewer","reason":"yes"})
        p=self.call("/api/propose",self.proposed(version=1,content="Update A"))["proposal_id"]
        stale=self.call("/api/propose",self.proposed(version=1,content="Update B"))["proposal_id"]
        self.call("/api/approve",{"proposal_id":p,"actor":"reviewer","reason":"yes"})
        error=self.call("/api/approve",{"proposal_id":stale,"actor":"reviewer","reason":"yes"},expect=400)
        self.assertIn("stale",error["error"])
        self.assertEqual(self.call("/api/state?project=a")["resolved"]["applicable"]["project"][0]["content"],"Update A")
        separate=self.call("/api/propose",self.proposed(version=2,content="Update C"))["proposal_id"]
        self.call("/api/reject",{"proposal_id":separate,"actor":"reviewer","reason":"not approved"})
        self.assertEqual(self.call("/api/state?project=a")["resolved"]["applicable"]["project"][0]["content"],"Update A")

    def test_unstructured_ingest_never_autoapproves(self):
        event=self.call("/api/ingest",{"text":"תעד שהקוד אינו חזון הפרויקט","author":"external"})
        self.assertEqual(event["status"],"UNCLASSIFIED")
        state=self.call("/api/state?project=a")
        self.assertEqual(state["resolved"]["applicable"]["project"],[])
        self.assertEqual(state["pending"],[])

    def test_invalid_json_shape_and_unknown_action_route(self):
        self.call("/api/propose",{"not":"allowed"},expect=400)
        self.call("/api/resolve",{"project_id":"../other"},expect=400)
        self.call("/api/unknown",{"a":1},expect=404)

    def test_resolve_applicability_through_http(self):
        doc=self.proposed()
        doc["conditions"]={"risk":"low"}
        pid=self.call("/api/propose",doc)["proposal_id"]
        self.call("/api/approve",{"proposal_id":pid,"actor":"reviewer","reason":"Explicit"})
        default=self.call("/api/state?project=a")
        self.assertEqual(default["resolved"]["applicable"]["project"],[])
        conditional=self.call("/api/resolve",{"project_id":"a","context":{"risk":"low"}})
        self.assertEqual(conditional["applicable"]["project"][0]["key"],"vision")


if __name__=="__main__":
    unittest.main(verbosity=2)
