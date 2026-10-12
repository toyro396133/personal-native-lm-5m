"""COMPASS V11.1: independent online learning from explicit free-text corrections.

No external model, GPT-6 teacher, SELF checkpoint, API or automatic policy approval.
The text reader is intentionally high-precision/low-recall; direct user language
remains source evidence even when no learning label can safely be derived.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

from online_compass import (LR, DECAY, MIN_PRIOR, MARGIN, MAX_TEXT,
                            ID_RE, SOURCE_RE, scope_id, features, dot, similarity)

SCHEMA = "COMPASS_V11_1_LITERAL_CORRECTION_FEEDBACK"
HEDGE = re.compile(r"\b(?:אולי|כנראה|יתכן|ייתכן|נראה לי|אני לא בטוח|אבל אולי|אולי בכל זאת)\b")
CONTRAST = re.compile(
    r"^\s*לא\s+(?!רק\b)(?P<rejected>.+?)\s*,?\s+אלא\s+(?!גם\b)(?P<preferred>.+?)\s*$",
    re.DOTALL
)
REPAIR = re.compile(r"^\s*(?:טעות[!:.]+|התכוונתי\s+(?:ל|ש))\s*(?P<preferred>.+)\s*$", re.DOTALL)

def _clean(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip().strip(" \t\n\r.!״\"'")

def _substance(s: str) -> bool:
    return len(s) >= 9 and len(re.findall(r"[\w\u0590-\u05ff]+", s)) >= 2

def validate(event: dict) -> None:
    if not isinstance(event, dict):
        raise ValueError("Feedback must be an object")
    required = ("id", "scope", "project_id", "speaker", "source", "question",
                "proposal", "user_reply", "source_verified")
    if any(key not in event for key in required):
        raise ValueError("Missing original feedback, scope, or provenance")
    if any(key in event for key in
           ("gold", "teacher_label", "chosen_option", "approved", "preferred_label")):
        raise ValueError("Derived gold/approval fields are not allowed")
    if not isinstance(event["id"], str) or not ID_RE.fullmatch(event["id"]):
        raise ValueError("Invalid event ID")
    if event["scope"] not in ("project", "personal"):
        raise ValueError("Invalid memory scope")
    if event["scope"] == "personal":
        if event["project_id"] not in ("", None):
            raise ValueError("Project source cannot be silently promoted to personal")
    elif (not isinstance(event["project_id"], str)
          or not ID_RE.fullmatch(event["project_id"])):
        raise ValueError("Invalid project identifier")
    if event["speaker"] != "user":
        raise ValueError("Only direct user-authored corrections may teach this learner")
    src = event["source"]
    if not isinstance(src, str) or not SOURCE_RE.fullmatch(src):
        raise ValueError("Source must use pinned Git SHA and precise documentation lines")
    if "/../" in src or int(src.rsplit("#L", 1)[1].split("-L")[1]) < int(src.rsplit("#L", 1)[1].split("-L")[0]):
        raise ValueError("Invalid document source")
    if type(event["source_verified"]) is not bool:
        raise ValueError("Source verification flag must be explicit bool")
    for key in ("question", "proposal", "user_reply"):
        if (not isinstance(event[key], str) or not event[key].strip()
            or len(event[key]) > MAX_TEXT):
            raise ValueError("Empty or oversized: " + key)

def literal_correction(event: dict) -> dict | None:
    """Extract only narrowly explicit user repairs. Does not approve policy."""
    validate(event)
    if not event["source_verified"]:
        return None
    reply = _clean(event["user_reply"])
    if HEDGE.search(reply):
        return None
    m = CONTRAST.fullmatch(reply)
    if m:
        rejected = _clean(m.group("rejected")).rstrip(" ,")
        preferred = _clean(m.group("preferred"))
        style = "EXPLICIT_NOT_BUT"
    else:
        m = REPAIR.fullmatch(reply)
        if not m:
            return None
        rejected = _clean(event["proposal"])
        preferred = _clean(m.group("preferred"))
        style = "EXPLICIT_REPAIR"
    if not (_substance(rejected) and _substance(preferred)):
        return None
    if any(k in preferred for k in ("?", "؟", "אולי")):
        return None
    if rejected.casefold() == preferred.casefold():
        return None
    return {"rejected": rejected, "preferred": preferred, "style": style,
            "source": event["source"], "is_policy_approval": False}

def _vector_difference(question: str, preferred: str, rejected: str) -> dict[str, float]:
    good = features(question, preferred)
    bad = features(question, rejected)
    diff = {}
    for key in set(good) | set(bad):
        value = good.get(key, 0.) - bad.get(key, 0.)
        if value:
            diff[key] = value
    return diff

class CorrectionCompass:
    def __init__(self):
        self.scopes: dict[str, dict] = {}
        self.seen: set[str] = set()

    def _state(self, sid: str) -> dict:
        return self.scopes.setdefault(sid, {"weights": {}, "training_events": 0, "history": []})

    def rank(self, question: str, candidates: list[str], *,
             scope: str, project_id: str | None) -> dict:
        if scope not in ("personal", "project"):
            raise ValueError("Invalid scope")
        sid = "personal:*" if scope == "personal" and project_id in ("", None) else (
            "project:" + project_id if scope == "project" and isinstance(project_id, str)
            and ID_RE.fullmatch(project_id) else None
        )
        if sid is None or len(candidates) != 2 or not all(
                isinstance(x, str) and x.strip() and len(x) <= MAX_TEXT for x in candidates):
            raise ValueError("Pair must consist of two original candidate texts")
        if not isinstance(question, str) or not question.strip() or len(question) > MAX_TEXT:
            raise ValueError("Task must be supplied before feedback")
        state = self._state(sid)
        if state["training_events"] < MIN_PRIOR:
            return {"choice_index": None, "status": "ABSTAIN_COLD_START",
                    "margin": 0., "sources": []}
        scores = [dot(state["weights"], features(question, candidate)) for candidate in candidates]
        abs_margin = abs(scores[1] - scores[0])
        support = sorted(
            ((similarity(question, event["question"]), event["id"], event["source"])
             for event in state["history"] if event["eligible"]),
            reverse=True
        )
        refs = [{"event_id": i, "source": s} for sim, i, s in support[:3] if sim > 0]
        if abs_margin < MARGIN:
            return {"choice_index": None, "status": "ABSTAIN_LOW_MARGIN",
                    "margin": round(abs_margin, 6), "sources": refs}
        return {"choice_index": int(scores[1] > scores[0]),
                "status": "UNVERIFIED_PREFERENCE_RANKING",
                "margin": round(abs_margin, 6), "sources": refs}

    def retrieval(self, sid: str, question: str, candidates: list[str]) -> int | None:
        state = self._state(sid)
        history = [x for x in state["history"] if x["eligible"]]
        if not history:
            return None
        # Retrieval baseline sees the same earlier corrections but has no trained weights.
        ranked = sorted(
            ((similarity(question, h["question"]), h) for h in history),
            key=lambda row: row[0], reverse=True
        )
        best, past = ranked[0]
        if best == 0:
            return None
        values = [
            similarity(candidate, past["preferred"]) -
            similarity(candidate, past["rejected"]) for candidate in candidates
        ]
        return None if values[0] == values[1] else int(values[1] > values[0])

    def observe(self, event: dict) -> dict:
        validate(event)
        sid = scope_id(event)
        key = sid + ":" + event["id"]
        if key in self.seen:
            raise ValueError("Duplicate user event")
        correction = literal_correction(event)
        state = self._state(sid)
        if correction is not None:
            # This is a RETROSPECTIVE pair constructed after the user reply.
            # Model parameters are evaluated BEFORE that event updates them.
            # It is NOT a prospective answer to a naturally proposed option pair.
            pair = [correction["rejected"], correction["preferred"]]
            if hashlib.blake2b(event["id"].encode("utf-8"), digest_size=1).digest()[0] % 2:
                pair.reverse()
            gold = pair.index(correction["preferred"])
            before = self.rank(event["question"], pair,
                               scope=event["scope"], project_id=event["project_id"])
            prior_retrieval = self.retrieval(sid, event["question"], pair)
            frozen = None   # no learned state => abstain
            # Train only from this explicit correction, post-measurement.
            diff = _vector_difference(event["question"], correction["preferred"],
                                      correction["rejected"])
            weights = state["weights"]
            margin = dot(weights, diff)
            gradient = 1. / (1. + math.exp(max(-35., min(35., margin))))
            for k, val in diff.items():
                weights[k] = weights.get(k, 0.) * (1. - DECAY) + LR * gradient * val
            state["training_events"] += 1
        else:
            before = {"choice_index": None, "status": "ABSTAIN_NOT_EXPLICIT",
                      "margin": 0., "sources": []}
            prior_retrieval = frozen = gold = None
        state["history"].append({
            "id": event["id"], "source": event["source"], "question": event["question"],
            "proposal": event["proposal"], "user_reply": event["user_reply"],
            "eligible": correction is not None,
            "preferred": correction["preferred"] if correction else None,
            "rejected": correction["rejected"] if correction else None,
            "pattern": correction["style"] if correction else None,
            "verified_claim": event["source_verified"]
        })
        self.seen.add(key)
        return {
            "id": event["id"], "scope": sid, "eligible": correction is not None,
            "pattern": correction["style"] if correction else None,
            "before": before, "gold_index_from_original_user_reply": gold,
            "retrieval_index": prior_retrieval, "frozen_index": frozen,
            "training_events_after": state["training_events"],
            "comparison_type": "RETROSPECTIVE_PAIR_RANKING_ONLY",
            "proposed_principle_status": "REVIEW_REQUIRED", "autoapproved": False
        }

    def export(self) -> dict:
        return {"schema": SCHEMA, "scopes": self.scopes, "seen": sorted(self.seen),
                "no_policy_approval": True, "training_source": "EXPLICIT_USER_REPAIRS_ONLY"}

    @classmethod
    def restore(cls, obj: dict) -> "CorrectionCompass":
        if (not isinstance(obj, dict) or obj.get("schema") != SCHEMA or
            obj.get("no_policy_approval") is not True or
            obj.get("training_source") != "EXPLICIT_USER_REPAIRS_ONLY"):
            raise ValueError("Not an allowed COMPASS V11.1 checkpoint")
        model = cls()
        model.scopes = obj["scopes"]
        model.seen = set(obj["seen"])
        return model


def replay(records: list[dict]) -> tuple[dict, CorrectionCompass]:
    learner = CorrectionCompass()
    details = [learner.observe(event) for event in records]
    eligible = [x for x in details if x["eligible"]]
    scores = {}
    for name, accessor in [
        ("learned", lambda r: r["before"]["choice_index"]),
        ("retrieval", lambda r: r["retrieval_index"]),
        ("frozen", lambda r: r["frozen_index"])
    ]:
        evaluated = [(accessor(row), row["gold_index_from_original_user_reply"]) for row in eligible]
        attempts = [p == truth for p, truth in evaluated if p is not None]
        scores[name] = {
            "eligible": len(eligible), "attempts": len(attempts),
            "correct": sum(attempts), "wrong": len(attempts) - sum(attempts),
            "coverage": len(attempts) / len(eligible) if eligible else 0.,
            "accuracy_when_answered": sum(attempts) / len(attempts) if attempts else None
        }
    return ({
        "model": SCHEMA, "events_seen": len(details),
        "eligible_explicit_corrections": len(eligible),
        "noneligible_review_only": len(details)-len(eligible),
        "scope_names": sorted(learner.scopes),
        "scores": scores, "events": details,
        "research_status": "INSUFFICIENT_REAL_CORRECTION_FEEDBACK" if len(eligible) < 20
                           else "EXPLORATORY_RETROSPECTIVE_PAIR_RANKING_ONLY",
        "scientific_caveat": "Pair candidates derive retrospectively from user reply; NOT future intent inference."
    }, learner)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True, help="PRIVATE local JSONL")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--state", type=Path, help="Optional PRIVATE learned checkpoint")
    args = p.parse_args()
    records = [json.loads(line) for line in args.events.read_text(encoding="utf8").splitlines() if line.strip()]
    result, learner = replay(records)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    if args.state:
        args.state.write_text(json.dumps(learner.export(), ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps({k: v for k, v in result.items() if k != "events"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
