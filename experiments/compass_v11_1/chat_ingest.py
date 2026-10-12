"""COMPASS V11.1: convert LOCAL role-tagged chat transcript into feedback events.

No LLM, external API, or inferred labels. Only consecutive USER -> ASSISTANT ->
USER triplets create a candidate event. This format does not certify that a
supplied export is genuine; source_verified confirms internal role/reference
consistency of the local input only. Keep raw transcripts and events private.
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

from freeform_feedback import validate

IDENT = re.compile(r"^[a-z][a-z0-9._-]{7,99}$")

def from_chat(messages: list[dict], *, project_id: str,
              scope: str = "project") -> tuple[list[dict], dict]:
    if not isinstance(messages,list):
        raise ValueError("Need chronologically ordered local message list")
    if scope not in ("project","personal"):
        raise ValueError("Explicit scope is required")
    if scope == "project":
        if not isinstance(project_id,str) or not re.fullmatch(r"[a-z][a-z0-9._-]{0,63}",project_id):
            raise ValueError("Explicit canonical project identifier required")
    elif project_id not in ("",None):
        raise ValueError("Personal chat input cannot silently carry project ownership")
    seen=set()
    for msg in messages:
        if (not isinstance(msg,dict) or
            not all(isinstance(msg.get(k),str) for k in ("session_id","id","role","text"))):
            raise ValueError("Each chat entry needs session_id, id, role, original text")
        sess,ident=msg["session_id"],msg["id"]
        if not IDENT.fullmatch(sess) or not IDENT.fullmatch(ident):
            raise ValueError("Session and message IDs must be stable identifiers >=8 chars")
        if msg["role"] not in ("user","assistant","system","tool"):
            raise ValueError("Unsupported message role")
        if not msg["text"].strip():
            raise ValueError("Empty message text")
        key=(sess,ident)
        if key in seen:
            raise ValueError("Duplicate source message")
        seen.add(key)
    events=[]
    reasons={"no_immediate_user_assistant_user_triplet":0,
             "session_boundary":0,"text_too_long":0,"valid":0}
    for i in range(2,len(messages)):
        a,b,c=messages[i-2:i+1]
        if c["role"]!="user":
            continue
        if len({a["session_id"],b["session_id"],c["session_id"]})!=1:
            reasons["session_boundary"]+=1
            continue
        if [a["role"],b["role"],c["role"]]!=["user","assistant","user"]:
            reasons["no_immediate_user_assistant_user_triplet"]+=1
            continue
        entry={
            "id":c["id"][:63],
            "scope":scope,"project_id":project_id,
            "speaker":"user",
            "source":f"chat:{c['session_id']}:{c['id']}",
            "question":a["text"],"proposal":b["text"],
            "user_reply":c["text"],
            "source_verified":True,
            "source_authentication":"LOCAL_ROLE_TAGGED_EXPORT_NOT_CRYPTOGRAPHIC",
            "proposal_source":f"chat:{b['session_id']}:{b['id']}"
        }
        try:
            validate(entry)
        except ValueError:
            reasons["text_too_long"]+=1
            continue
        events.append(entry)
        reasons["valid"]+=1
    return events, {"input_messages":len(messages), "emitted_events":len(events),
                    "skipped":reasons,
                    "data_status":"LOCAL_ROLE_TAGGED_SOURCE_ONLY_NOT_AUTHENTICATED",
                    "labels_extracted":0,
                    "autoapproved":False}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--messages",type=Path,required=True,help="PRIVATE local JSON array or JSONL")
    p.add_argument("--project",required=True)
    p.add_argument("--scope",choices=["project","personal"],default="project")
    p.add_argument("--events",type=Path,required=True)
    p.add_argument("--audit",type=Path,required=True)
    args=p.parse_args()
    content=args.messages.read_text(encoding="utf8")
    try:
        messages=json.loads(content)
    except json.JSONDecodeError:
        messages=[json.loads(line) for line in content.splitlines() if line.strip()]
    events,audit=from_chat(messages,project_id=args.project,scope=args.scope)
    args.events.write_text("".join(json.dumps(x,ensure_ascii=False)+"\n" for x in events),encoding="utf8")
    args.audit.write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding="utf8")
    print(f"Created {len(events)} private candidate feedback events, no training labels or external API.")

if __name__=="__main__":
    main()
