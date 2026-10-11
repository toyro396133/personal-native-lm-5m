"""COMPASS V9.12b local-only review UI. Python standard library only.

Run: python compass_ui.py --db ./compass-local.sqlite3 --port 8787
Loopback only. Anti-CSRF token + Host/Origin checks; NOT user authentication.
"""
from __future__ import annotations

import argparse
import json
import secrets
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

from compass_core import CompassStore, CompassError, validate_conditions

PAGE = r'''<!doctype html>
<html lang="he" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>COMPASS — זיכרון מאושר</title>
<style>
:root{color-scheme:dark;font-family:system-ui,-apple-system,"Segoe UI",Arial,sans-serif}
*{box-sizing:border-box}body{margin:0;background:#101827;color:#edf2fb;line-height:1.65}
header{padding:26px max(22px,5vw);border-bottom:1px solid #314051;background:#151f30}
h1{font-size:27px;margin:0;letter-spacing:.08em}h2{font-size:19px;margin:0 0 14px}
small,.muted{color:#afc0d4}.wrap{max-width:1200px;margin:0 auto;padding:24px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:20px}
article{border:1px solid #35465b;border-radius:14px;padding:22px;background:#172335}
.full{grid-column:1/-1}.item{border:1px solid #33435a;border-radius:9px;margin:12px 0;padding:14px;background:#122033}
.item strong{display:block;font-size:16px}.item .meta{font-size:13px;color:#9eb3cf;direction:auto}
button,input,textarea,select{font:inherit;border-radius:8px;border:1px solid #54718a}
input,textarea,select{width:100%;padding:10px;background:#101b2c;color:white}
textarea{min-height:74px}button{background:#1c7eaa;color:white;padding:9px 17px;cursor:pointer}
button.secondary{background:#314357}.row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.row input{flex:1;min-width:140px}label{display:block;margin:7px 0;font-size:14px}
.note{padding:11px;margin:10px 0;border-inline-start:3px solid #5eb0c8;background:#172b42;border-radius:6px}
#error{color:#ffb8ac;white-space:pre-wrap}#status{color:#a4e8c7;white-space:pre-wrap}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px;color:#cdd9e8}
@media(max-width:780px){.grid{grid-template-columns:1fr}.full{grid-column:auto}}
</style></head><body>
<header><div class="row" style="justify-content:space-between">
 <div><h1>COMPASS</h1><small>זיכרון אישי ופרויקטלי עם נימוקים ואישורים</small></div>
 <small>מקומי בלבד • אינו מחובר למודל שפה • אין אישור אוטומטי</small>
</div></header>
<main class="wrap">
 <div class="row"><label for="project">פרויקט פעיל</label><input id="project" value="compass-lab" placeholder="project-id">
 <button id="refresh">רענון</button><small id="integrity"></small></div>
 <p class="note">רק אישור מפורש משנה עיקרון פעיל. הודעה חופשית נשמרת כתיעוד, בלי להיחשב אוטומטית ככוונה מוסכמת.</p>
 <p id="error" role="alert"></p><p id="status" role="status"></p>
 <section class="grid">
 <article><h2>עקרונות אישיים</h2><div id="personal"></div></article>
 <article><h2>עקרונות הפרויקט</h2><div id="project-principles"></div></article>
 <article class="full"><h2>הצעות שממתינות לאישור</h2><div id="pending"></div></article>
 <article><h2>הצע עיקרון או תיקון</h2>
   <form id="proposal-form">
   <label>שיוך<select name="scope"><option value="project">הפרויקט הפעיל</option><option value="personal">אישי כללי</option></select></label>
   <label>מפתח עיקרון (באנגלית, למשל vision)<input name="key" required pattern="[a-z][a-z0-9_.-]*"></label>
   <label>פעולה<select name="operation"><option value="set">יצירה / תיקון</option><option value="revoke">ביטול</option></select></label>
   <label>נוסח העיקרון<textarea name="content"></textarea></label>
   <label>נימוק מפורש<textarea name="reason" required></textarea></label>
   <label>גרסה נוכחית (0 לחדש)<input name="expected_revision" type="number" min="0" value="0" required></label>
   <label>מזהה מקור<input name="source" value="local-browser-review"></label>
   <button type="submit">שמור כהצעה לאישור</button></form></article>
 <article><h2>תיעוד טקסט חופשי</h2>
   <form id="ingest-form"><label>אירוע עבודה<textarea name="text" required placeholder="פרט רקע, מחשבה, החלטה מוצעת..."></textarea></label>
   <button type="submit">שמור כמידע לא מאושר</button></form>
   <h2 style="margin-top:24px">בדיקת תחולה</h2>
   <form id="context-form"><label>תנאי הקשר (key=value, אחד בכל שורה)<textarea name="context" placeholder="risk=low"></textarea></label>
   <button type="submit" class="secondary">בדוק עקרונות רלוונטיים</button></form>
   <pre id="resolved"></pre></article>
 <article class="full"><h2>אירועים אחרונים ביומן</h2><div id="timeline"></div></article>
 </section>
</main>
<script>
"use strict";
const sessionToken=__TOKEN__;
const $=id=>document.getElementById(id);
const clear=el=>{while(el.firstChild)el.removeChild(el.firstChild)};
function line(tag,text,parent,cls){const x=document.createElement(tag);x.textContent=String(text);
 if(cls)x.className=cls;parent.appendChild(x);return x}
function clearMessages(){$("error").textContent="";$("status").textContent=""}
async function api(path,body){
 const options=body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json",
 "X-COMPASS-SESSION":sessionToken},body:JSON.stringify(body)};
 const r=await fetch(path,options),data=await r.json();
 if(!r.ok)throw new Error(data.error||("HTTP "+r.status));return data;
}
function displayRecord(parent,item){
 const box=line("div","",parent,"item");line("strong",item.key+" — "+item.content,box);
 line("div","נימוק: "+item.reason,box);
 line("div","מקור: "+item.source+" | גרסה "+item.revision+" | "+item.scope+" / "+item.owner,box,"meta");
 if(Object.keys(item.conditions||{}).length)
  line("div","תחולה: "+JSON.stringify(item.conditions),box,"meta");
}
async function refresh(){
 try{
  clearMessages();let project=$("project").value.trim();
  const data=await api("/api/state?project="+encodeURIComponent(project));
  $("integrity").textContent=data.audit.ok?"יומן תקין • "+data.audit.count+" אירועים":"יומן שגוי";
  for(const [id,records] of [["personal",data.resolved.applicable.personal],
    ["project-principles",data.resolved.applicable.project]]){
    const area=$(id);clear(area);
    if(!records.length)line("p","אין עקרונות מאושרים ותקפים בהקשר הזה.",area,"muted");
    for(const item of records)displayRecord(area,item);
  }
  const pending=$("pending");clear(pending);
  if(!data.pending.length)line("p","אין הצעות ממתינות.",pending,"muted");
  for(const item of data.pending){
    const box=line("div","",pending,"item");
    line("strong",item.operation+" • "+item.scope+" / "+item.owner+" • "+item.key,box);
    line("div",item.content||"ביטול עיקרון",box);
    line("div","נימוק: "+item.reason+" | גרסה צפויה: "+item.expected_revision,box);
    const actions=line("div","",box,"row");
    const approve=line("button","אישור",actions);
    const reject=line("button","דחייה",actions,"secondary");
    approve.onclick=()=>review(item.proposal_id,"approve");
    reject.onclick=()=>review(item.proposal_id,"reject");
  }
  const timeline=$("timeline");clear(timeline);
  for(const event of data.timeline){
    const box=line("div","",timeline,"item");
    line("strong",event.kind+" • "+event.timestamp,box);
    line("div","מספר אירוע "+event.seq+" • "+event.event_id,box,"meta");
  }
  $("resolved").textContent=JSON.stringify(data.resolved,null,2);
 }catch(e){$("error").textContent=e.message;}
}
async function review(id,kind){
 const reason=prompt(kind==="approve"?"מה הנימוק לאישור?":"מדוע לדחות?");
 if(!reason||!reason.trim())return;
 try{await api("/api/"+kind,{proposal_id:id,actor:"local-browser-operator",
   reason:reason.trim()});await refresh();$("status").textContent="הפעולה בוצעה ונרשמה ביומן."}
 catch(e){$("error").textContent=e.message;}
}
$("refresh").onclick=refresh;
$("proposal-form").onsubmit=async function(e){
 e.preventDefault();const d=Object.fromEntries(new FormData(this));
 const body={scope:d.scope,project_id:d.scope==="project"?$("project").value.trim():null,
 key:d.key,operation:d.operation,content:d.content,reason:d.reason,
 expected_revision:Number(d.expected_revision),source:d.source,author:"local-browser-operator"};
 try{const result=await api("/api/propose",body);await refresh();
 $("status").textContent="נוצרה הצעה בלבד: "+result.proposal_id+" — יש לאשר אותה בנפרד."}
 catch(err){$("error").textContent=err.message;}
};
$("ingest-form").onsubmit=async function(e){
 e.preventDefault();const d=Object.fromEntries(new FormData(this));
 try{const result=await api("/api/ingest",{text:d.text,author:"local-browser-operator"});
 await refresh();$("status").textContent="האירוע נשמר: "+result.status}
 catch(err){$("error").textContent=err.message;}
};
$("context-form").onsubmit=async function(e){
 e.preventDefault();try{
  const d=Object.fromEntries(new FormData(this));
  const context={};for(const row of d.context.split("\n")){
   if(!row.trim())continue;
   const index=row.indexOf("=");if(index<0)throw new Error("יש להקליד key=value");
   context[row.slice(0,index).trim()]=row.slice(index+1).trim();
  }
  const result=await api("/api/resolve",{project_id:$("project").value.trim(),context});
  $("resolved").textContent=JSON.stringify(result,null,2);
 }catch(err){$("error").textContent=err.message;}
};
refresh();
</script>
</body></html>'''


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, db_path: str | Path, port: int):
        self.db_path = str(db_path)
        self.session_token = secrets.token_urlsafe(36)
        super().__init__(("127.0.0.1", port), LocalHandler)


class LocalHandler(BaseHTTPRequestHandler):
    server: LocalServer

    def _allowed_host(self) -> bool:
        port = self.server.server_port
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        return self.headers.get("Host","") in allowed

    def _headers(self, status: int, mime: str, length: int):
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy",
                         "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()

    def _send(self, status: int, payload: dict | list):
        data=json.dumps(payload,ensure_ascii=False).encode("utf8")
        self._headers(status,"application/json; charset=utf-8",len(data))
        self.wfile.write(data)

    def _reject(self, status: int, why: str):
        self._send(status,{"error":why})

    def _security_gate(self, post=False) -> bool:
        if not self._allowed_host():
            self._reject(403,"Untrusted Host header")
            return False
        if post:
            origin=self.headers.get("Origin")
            valid={f"http://127.0.0.1:{self.server.server_port}",
                   f"http://localhost:{self.server.server_port}"}
            if origin is not None and origin not in valid:
                self._reject(403,"Cross-origin write rejected")
                return False
            if not secrets.compare_digest(
                    self.headers.get("X-COMPASS-SESSION",""),
                    self.server.session_token):
                self._reject(403,"Session token missing or invalid")
                return False
            if self.headers.get("Content-Type","").split(";")[0].strip().lower()!="application/json":
                self._reject(415,"Only application/json is accepted")
                return False
        return True

    def do_GET(self):
        if not self._security_gate():
            return
        uri=urlsplit(self.path)
        if uri.path=="/":
            page=PAGE.replace("__TOKEN__",json.dumps(self.server.session_token)).encode("utf8")
            self._headers(200,"text/html; charset=utf-8",len(page))
            self.wfile.write(page)
            return
        try:
            with CompassStore(self.server.db_path) as store:
                if uri.path=="/api/state":
                    project=parse_qs(uri.query).get("project",["compass-lab"])[0]
                    payload={"resolved":store.resolve(project_id=project),
                             "pending":store.proposals(),
                             "timeline":store.timeline(20),
                             "audit":store.verify()}
                elif uri.path=="/api/verify":
                    payload=store.verify()
                else:
                    return self._reject(404,"Not found")
            self._send(200,payload)
        except CompassError as error:
            self._reject(400,str(error))

    def do_POST(self):
        if not self._security_gate(post=True):
            return
        uri=urlsplit(self.path)
        if uri.path not in ("/api/propose","/api/approve","/api/reject",
                            "/api/ingest","/api/resolve"):
            return self._reject(404,"Not found")
        try:
            raw_size=self.headers.get("Content-Length","0")
            if not raw_size.isdigit() or not 0<int(raw_size)<=64*1024:
                return self._reject(413,"JSON body too large or missing")
            data=json.loads(self.rfile.read(int(raw_size)))
            if not isinstance(data,dict):
                return self._reject(400,"Expected JSON object")
            with CompassStore(self.server.db_path) as store:
                if uri.path=="/api/propose":
                    permitted={"scope","project_id","key","operation","content",
                               "reason","expected_revision","author","source","conditions"}
                    if set(data)-permitted:
                        raise CompassError("Unknown proposal fields")
                    result={"status":"PENDING_APPROVAL","proposal_id":store.propose(**data)}
                elif uri.path=="/api/approve":
                    result=store.approve(data["proposal_id"],actor=data["actor"],reason=data["reason"])
                elif uri.path=="/api/reject":
                    result=store.reject(data["proposal_id"],actor=data["actor"],reason=data["reason"])
                elif uri.path=="/api/ingest":
                    result=store.observe(data["text"],author=data["author"],
                                         source=data.get("source","local-browser"))
                else:
                    result=store.resolve(project_id=data["project_id"],
                                         context=data.get("context",{}))
            self._send(200,result)
        except (CompassError,KeyError,TypeError,ValueError,json.JSONDecodeError) as error:
            self._reject(400,str(error))

    def log_message(self, fmt, *args):
        # No raw request bodies or principle contents are logged.
        return


def make_server(db_path: str|Path, port: int=8787) -> LocalServer:
    return LocalServer(db_path,port)


def main():
    ap=argparse.ArgumentParser(description="COMPASS local browser UI (loopback only)")
    ap.add_argument("--db",required=True)
    ap.add_argument("--port",type=int,default=8787)
    ap.add_argument("--open",action="store_true",help="Open browser after startup")
    args=ap.parse_args()
    if not 0<args.port<65536:
        raise SystemExit("Port must be between 1 and 65535")
    with CompassStore(args.db) as store:
        store.verify()
    server=make_server(args.db,args.port)
    url=f"http://127.0.0.1:{server.server_port}"
    print(f"COMPASS local interface: {url}",flush=True)
    print("Only the current machine can connect. Ctrl+C to stop. Actor labels are NOT authenticated.",flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__=="__main__":
    main()
