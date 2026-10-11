"""Independent COMPASS V11: conservative online project-choice learning.

No SELF code, external model, API, network, or automated policy approval.
All text and learnable weights remain in a user-local JSON state.
This is a deliberately WEAK char-gram learner, not semantic project understanding.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

LR = 0.1
DECAY = 0.0001
MIN_PRIOR = 2
MARGIN = 0.03
MAX_TEXT = 2500
SOURCE_RE = re.compile(r"^git:[0-9a-f]{40}:docs/[a-zA-Z0-9_./-]+[.]md#L[1-9][0-9]*-L[1-9][0-9]*$")
ID_RE = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
SCOPE_TYPES = {"project", "personal"}


def validate_episode(x: dict) -> None:
    if not isinstance(x, dict):
        raise ValueError("Episode must be a JSON object")
    for key in ("id", "scope", "project_id", "source", "question", "reply", "options"):
        if key not in x:
            raise ValueError("Missing required field: " + key)
    if not isinstance(x["id"], str) or not ID_RE.fullmatch(x["id"]):
        raise ValueError("Invalid episode id")
    if x["scope"] not in SCOPE_TYPES:
        raise ValueError("Invalid scope")
    if x["scope"] == "personal":
        if x["project_id"] not in ("", None):
            raise ValueError("Personal events cannot be project-owned")
    elif not isinstance(x["project_id"], str) or not ID_RE.fullmatch(x["project_id"]):
        raise ValueError("Project-owned event needs a project id")
    src = x["source"]
    if not isinstance(src, str) or not SOURCE_RE.fullmatch(src):
        raise ValueError("Source must be pinned Git documentation with exact line span")
    # Source validation checks syntax ONLY, not existence or authorship.
    part = src.split("#L")
    first, last = part[1].split("-L")
    if int(last) < int(first) or "/../" in src:
        raise ValueError("Invalid source line range or path")
    for key in ("question", "reply"):
        if not isinstance(x[key], str) or not x[key].strip() or len(x[key]) > MAX_TEXT:
            raise ValueError("Missing/oversized " + key)
    opts = x["options"]
    if not isinstance(opts, list) or not 2 <= len(opts) <= 4:
        raise ValueError("Require 2..4 original alternatives")
    ids = set()
    for option in opts:
        if not isinstance(option, dict) or set(option) != {"id", "text"}:
            raise ValueError("Option needs only id and text")
        if (not isinstance(option["id"], str) or not option["id"] or
            len(option["id"]) > 4 or option["id"] in ids):
            raise ValueError("Duplicate/invalid option ID")
        ids.add(option["id"])
        if not isinstance(option["text"], str) or not option["text"].strip() or len(option["text"]) > MAX_TEXT:
            raise ValueError("Invalid option text")
    # No curator/researcher/teacher supplied gold ID is accepted.
    if any(k in x for k in ("gold", "label", "chosen_option_id", "teacher_label", "selected")):
        raise ValueError("Labels must be derived exclusively from an explicit raw reply")


def explicit_reply(x: dict) -> str | None:
    """Accept ONLY whole one-option letter/name responses, without interpretation."""
    reply = unicodedata.normalize("NFKC", x["reply"]).strip().strip(" \"'״׳.! \t")
    matches = [o["id"] for o in x["options"] if reply.casefold() == o["id"].casefold()]
    return matches[0] if len(matches) == 1 else None


def grams(s: str, n: int = 3) -> set[str]:
    text = unicodedata.normalize("NFKC", s).casefold()
    words = re.findall(r"[\w\u0590-\u05ff]+", text)
    return {word[i:i+n] for word in words for i in range(max(0, len(word)-n+1))}


def features(question: str, option: str) -> dict[str, float]:
    """Pairwise hashed Unicode word n-gram features; no gold/choice metadata."""
    q = sorted(grams(question))
    o = sorted(grams(option))
    keys = {"o|" + v for v in o}
    # Bounded question-option conjunctions, avoiding option-ID shortcuts.
    keys.update("q|"+a+"|"+b for a in q[:12] for b in o[:12])
    if not keys:
        return {}
    norm = math.sqrt(len(keys))
    result = {}
    for k in keys:
        b = hashlib.blake2b(k.encode("utf8"), digest_size=8).hexdigest()
        result[b] = result.get(b, 0.0) + 1 / norm
    return result


def dot(weights: dict, vector: dict) -> float:
    return sum(weights.get(k, 0.0) * v for k, v in vector.items())


def similarity(a: str, b: str) -> float:
    left, right = grams(a), grams(b)
    union = len(left | right)
    return len(left & right) / union if union else 0.0


def scope_id(x: dict) -> str:
    return "personal:*" if x["scope"] == "personal" else "project:" + x["project_id"]


class ContinualCompass:
    def __init__(self):
        self.scopes: dict[str, dict] = {}
        self.seen: set[str] = set()

    def _state(self, scope: str) -> dict:
        return self.scopes.setdefault(scope, {"weights": {}, "history": [], "trained": 0})

    def predict(self, episode: dict) -> dict:
        validate_episode(episode)
        scope = scope_id(episode)
        state = self._state(scope)
        if state["trained"] < MIN_PRIOR:
            return {"choice": None, "status": "ABSTAIN_COLD_START", "margin": 0.0, "support": []}
        scores = [(dot(state["weights"], features(episode["question"], o["text"])), o["id"])
                  for o in episode["options"]]
        scores.sort(key=lambda t: (-t[0], t[1]))
        margin = scores[0][0] - scores[1][0]
        ranked_history = sorted(
            ((similarity(episode["question"], row["question"]),
              row["id"], row["source"])
             for row in state["history"] if row["eligible"]),
            reverse=True)
        support = [{"event_id": eid, "source": src}
                   for value, eid, src in ranked_history[:3] if value > 0]
        if margin < MARGIN:
            return {"choice": None, "status": "ABSTAIN_LOW_MARGIN",
                    "margin": round(margin, 6), "support": support}
        return {"choice": scores[0][1], "status": "PREDICTION_UNVERIFIED",
                "margin": round(margin, 6), "support": support}

    def retrieval_baseline(self, episode: dict) -> str | None:
        """Try matching previous chosen option text, no neural or oracle features."""
        state = self._state(scope_id(episode))
        past = [r for r in state["history"] if r["eligible"]]
        if not past:
            return None
        choices = [(similarity(opt["text"], prev["chosen_text"]), opt["id"])
                   for opt in episode["options"] for prev in past]
        if not choices:
            return None
        choices.sort(key=lambda a: (-a[0], a[1]))
        return choices[0][1] if choices[0][0] > 0 else None

    def recency_baseline(self, episode: dict) -> str | None:
        history = self._state(scope_id(episode))["history"]
        ids = {o["id"] for o in episode["options"]}
        for record in reversed(history):
            if record["eligible"] and record["choice"] in ids:
                return record["choice"]
        return None

    def observe(self, episode: dict) -> dict:
        validate_episode(episode)
        ident = (scope_id(episode), episode["id"])
        if repr(ident) in self.seen:
            raise ValueError("Duplicate event in same scope")
        before = self.predict(episode)  # always PRE-feedback
        retr = self.retrieval_baseline(episode)
        recent = self.recency_baseline(episode)
        chosen = explicit_reply(episode)
        state = self._state(scope_id(episode))
        if chosen is not None:
            chosen_opt = next(o["text"] for o in episode["options"] if o["id"] == chosen)
            pos = features(episode["question"], chosen_opt)
            options = [features(episode["question"], o["text"]) for o in episode["options"]
                       if o["id"] != chosen]
            weights = state["weights"]
            # Online pairwise feedback step, regularized. No future events.
            for neg in options:
                diff = defaultdict(float)
                for k,v in pos.items():
                    diff[k] += v
                for k,v in neg.items():
                    diff[k] -= v
                raw = dot(weights, diff)
                grad = 1 / (1 + math.exp(max(-35., min(35., raw))))
                for k,v in diff.items():
                    weights[k] = weights.get(k, 0.0)*(1 - DECAY) + LR*grad*v
            state["trained"] += 1
        else:
            chosen_opt = ""
        state["history"].append({
            "id": episode["id"], "source": episode["source"],
            "question": episode["question"],
            "reply": episode["reply"], "choice": chosen,
            "chosen_text": chosen_opt, "eligible": chosen is not None,
            "why": "Original user reply only; no teacher interpretation",
        })
        self.seen.add(repr(ident))
        return {"id": episode["id"],"scope":scope_id(episode),
                "eligible": chosen is not None,"observed_choice":chosen,
                "prequential": before,"retrieval_choice":retr,
                "recency_choice": recent,
                "first_option_choice":episode["options"][0]["id"],
                "model_learned":state["trained"], "autoapproved":False}

    def export(self) -> dict:
        return {"schema":"COMPASS_V11_LEARNER", "scopes":self.scopes,
                "seen":sorted(self.seen), "no_autoapproval":True}

    @classmethod
    def restore(cls, obj: dict):
        if obj.get("schema")!="COMPASS_V11_LEARNER" or obj.get("no_autoapproval") is not True:
            raise ValueError("Unknown or nonconservative checkpoint")
        learner = cls()
        learner.scopes = obj["scopes"]
        learner.seen = set(obj["seen"])
        return learner


def evaluate(records: list[dict]) -> dict:
    model = ContinualCompass()
    details = [model.observe(record) for record in records]
    eligible = [r for r in details if r["eligible"]]
    methods = {
        "learned":lambda r:r["prequential"]["choice"],
        "retrieval":lambda r:r["retrieval_choice"],
        "recency":lambda r:r["recency_choice"],
        "first_option":lambda r:r["first_option_choice"]
    }
    metrics={}
    for name,fn in methods.items():
        predicted = [(fn(x),x["observed_choice"]) for x in eligible]
        active = [p for p,gold in predicted if p is not None]
        correct = sum(p==gold for p,gold in predicted if p is not None)
        metrics[name] = {
            "eligible_events":len(eligible), "attempts":len(active),
            "correct":correct,"coverage":round(len(active)/len(eligible),4) if eligible else 0.,
            "accuracy_when_answered":round(correct/len(active),4) if active else None,
            "wrong_confident":len(active)-correct
        }
    return {
        "episodes_seen":len(details),
        "eligible_direct_feedback":len(eligible),
        "ambiguous_or_nonchoice":len(details)-len(eligible),
        "project_scopes":sorted(model.scopes),
        "by_method":metrics,
        "episodes":details,
        "research_verdict":"INSUFFICIENT_REAL_DECISION_FEEDBACK" if len(eligible)<20
                          else "EXPLORATORY_PREQUENTIAL_RESULT_NOT_INTENT_PROOF",
        "limits":"Raw one-letter user choices only. No independent semantic WHY/intent evaluation."
    }


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--episodes",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    p.add_argument("--state",type=Path,help="Optional local PRIVATE learned state checkpoint")
    args=p.parse_args()
    records=[json.loads(line) for line in args.episodes.read_text(encoding="utf8").splitlines()
             if line.strip()]
    result=evaluate(records)
    args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    if args.state:
        learner=ContinualCompass()
        for episode in records:learner.observe(episode)
        args.state.write_text(json.dumps(learner.export(),ensure_ascii=False,indent=2),encoding="utf8")
    print(json.dumps({k:v for k,v in result.items() if k!="episodes"},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
