"""Convert V11.1 user-verified repairs to V12 neuro-adapter training PAIRS.

This preserves original user question and evidence, but IMPORTANT: after
a correction, the preferred alternative is known. These pairs are eligible
for training only, NOT valid prospective evaluation/accuracy examples.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from detachable_layer import check_id, check_source

# Small standalone parity reader for the V11.1 JSONL/review format.
# V11.1 lives on a DIFFERENT experiment branch; do not silently assume code
# is merged into main. No model-generated interpretation or hidden gold label.
HEDGE = re.compile(r"(?:אולי|כנראה|יתכן|ייתכן|נראה לי|אני לא בטוח)")
CONTRAST = re.compile(r"^\s*לא\s+(?!רק\b)(?P<rejected>.+?)\s*,?\s+אלא\s+(?!גם\b)(?P<preferred>.+?)\s*$",re.DOTALL)
REPAIR = re.compile(r"^\s*(?:טעות[!:.]+|התכוונתי\s+(?:ל|ש))\s*(?P<preferred>.+)\s*$",re.DOTALL)

def clean(value: str) -> str:
    return unicodedata.normalize("NFKC",value).strip().strip(" \t\n\r.!״\"'")

def substantial(text: str) -> bool:
    return len(text)>=9 and len(re.findall(r"[\w\u0590-\u05ff]+",text))>=2

def validate(event: dict) -> None:
    if not isinstance(event,dict) or any(
            k not in event for k in ("id","scope","project_id","speaker","source","question",
                                     "proposal","user_reply","source_verified")):
        raise ValueError("Malformed V11.1 source event")
    check_id(event["id"])
    if event["scope"] not in ("project","personal"):
        raise ValueError("Unknown event scope")
    if event["scope"]=="project":
        check_id(event["project_id"])
    elif event["project_id"] not in (None,""):
        raise ValueError("Project event cannot silently promote into personal state")
    if event["speaker"]!="user" or type(event["source_verified"]) is not bool:
        raise ValueError("Cannot infer authority or missing source attestation")
    check_source(event["source"])
    for key in ("question","proposal","user_reply"):
        if not isinstance(event[key],str) or not event[key].strip() or len(event[key])>2500:
            raise ValueError("Missing or oversized original user evidence")
    if any(k in event for k in ("gold","label","teacher_label","chosen_option","approved")):
        raise ValueError("Do not ingest teacher labels or project-policy approvals")

def fingerprint(event: dict) -> str:
    validate(event)
    payload=json.dumps(event,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def literal_correction(event:dict)->dict|None:
    validate(event)
    if not event["source_verified"]:
        return None
    reply=clean(event["user_reply"])
    if HEDGE.search(reply):
        return None
    m=CONTRAST.fullmatch(reply)
    if m:
        rejected=clean(m.group("rejected")).rstrip(" ,")
        preferred=clean(m.group("preferred"))
    else:
        m=REPAIR.fullmatch(reply)
        if not m:
            return None
        rejected=clean(event["proposal"])
        preferred=clean(m.group("preferred"))
    if (not substantial(rejected) or not substantial(preferred) or
        rejected.casefold()==preferred.casefold() or "?" in preferred):
        return None
    return {"rejected":rejected,"preferred":preferred}

def user_reviewed_correction(event: dict, review: dict)->dict|None:
    validate(event)
    if not isinstance(review,dict) or review.get("event_id")!=event["id"] or review.get("event_hash")!=fingerprint(event):
        raise ValueError("Review not bound to original source event")
    if review.get("action")=="KEEP_UNRESOLVED":
        return None
    if review.get("action")!="CONFIRM_CORRECTION" or review.get("confirmed_by_user") is not True:
        raise ValueError("No explicit direct user confirmation")
    if not event["source_verified"]:
        raise ValueError("Cannot train from unverified source")
    reason=review.get("reviewer_note")
    rejected=review.get("rejected")
    preferred=review.get("preferred")
    if (not isinstance(reason,str) or len(reason.strip())<9 or
        not isinstance(rejected,str) or not isinstance(preferred,str)):
        raise ValueError("Human source review requires rejected/preferred and reason")
    rejected,preferred=clean(rejected),clean(preferred)
    if not substantial(rejected) or not substantial(preferred) or rejected.casefold()==preferred.casefold():
        raise ValueError("Unclear approved user repair")
    return {"rejected":rejected,"preferred":preferred}

def pair_from_feedback(event: dict, *, user_id: str,
                       review: dict | None = None) -> dict | None:
    check_id(user_id)
    validate(event)
    if review is not None:
        correction = user_reviewed_correction(event,review)
    else:
        correction = literal_correction(event)
    if correction is None:
        return None
    if event["speaker"]!="user" or event["source_verified"] is not True:
        raise ValueError("Unverified original user reply is not suitable for training")
    check_source(event["source"])
    # V11.1 labels both values using DIRECT feedback / its human confirmation.
    # Option order is fixed for transparent audit. It is not a future outcome.
    return {
        "schema":"COMPASS_V12_TRAINING_PAIR_NOT_EVALUATION",
        "user_id":user_id, "scope":event["scope"],
        "project_id":event["project_id"],
        "event_id":event["id"], "source":event["source"],
        "question":event["question"],
        "candidates":[correction["rejected"],correction["preferred"]],
        "chosen_index":1,
        "review_status":"USER_CONFIRMED" if review is not None else "LITERAL_EXPLICIT_USER_CORRECTION",
        "feedback":{"event_id":event["id"],"source":event["source"],
                    "speaker":"user","confirmed_by_user":True,
                    "source_verified":True},
        "prospective_holdout_eligible":False,
        "project_policy_approved":False
    }

def compile_training_pairs(events: list[dict], *,
                           user_id: str, reviews: list[dict] | None = None) -> dict:
    check_id(user_id)
    reviews=reviews or []
    by_id={}
    for item in reviews:
        if not isinstance(item,dict) or not isinstance(item.get("event_id"),str):
            raise ValueError("Review records require an event_id")
        if item["event_id"] in by_id:
            raise ValueError("Duplicate reviewer confirmation")
        by_id[item["event_id"]]=item
    results=[]
    skipped=[]
    unique=set()
    for event in events:
        validate(event)
        key=(event["scope"],event["project_id"],event["id"])
        if key in unique:
            raise ValueError("Duplicate event in the provided chronology")
        unique.add(key)
        row=pair_from_feedback(event,user_id=user_id,review=by_id.get(event["id"]))
        if row is None:
            skipped.append(event["id"])
        else:
            results.append(row)
    if set(by_id) - {x["id"] for x in events}:
        raise ValueError("Orphan reviews are forbidden")
    return {
        "schema":"COMPASS_V12_PRIVATE_TRAINING_QUEUE_V1",
        "user_id":user_id,"pairs":results,
        "unresolved_event_ids":skipped,
        "total_events":len(events),"trainable_pairs":len(results),
        "evaluation_pairs":0,
        "status":"USER_CORRECTION_PAIRS_FOR_TRAINING_ONLY",
        "user_confirmed_review_count":sum(x["review_status"]=="USER_CONFIRMED" for x in results),
        "source_cryptographically_authenticated":False,
        "false_assistant_approval":False
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--events",required=True,type=Path,help="PRIVATE V11.1 input JSONL")
    p.add_argument("--reviews",type=Path,help="PRIVATE human review JSON array")
    p.add_argument("--user-id",required=True)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    events=[json.loads(x) for x in args.events.read_text(encoding="utf8").splitlines()
            if x.strip()]
    reviews=json.loads(args.reviews.read_text(encoding="utf8")) if args.reviews else []
    queue=compile_training_pairs(events,user_id=args.user_id,reviews=reviews)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(queue,ensure_ascii=False,indent=2),encoding="utf8")
    print(f"Prepared {queue['trainable_pairs']}/{queue['total_events']} PRIVATE V12 training pairs.")
    print("No prospective test labels, no outside model, no approved product policy.")

if __name__=="__main__":
    main()
