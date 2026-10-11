"""COMPASS V9.12 standalone persistent source-scoped principle memory.

Python 3.11+ stdlib only. This is a cautious prototype and NOT a semantic
language model, user authentication system, secure signing or full policy engine.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ZERO_HASH = "0" * 64
OPERATIONS = {"set", "revoke"}
SCOPES = {"personal", "project"}
KEY_RE = re.compile(r"^[a-z][a-z0-9_.-]{1,63}$")


class CompassError(Exception):
    """Reject invalid provenance, ambiguous edits or corrupt journal safely."""


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def validate_conditions(value: Any) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or len(value) > 12:
        raise CompassError("conditions must be a JSON object of at most 12 exact string predicates")
    result: dict[str, str] = {}
    for key, match in value.items():
        if not isinstance(key, str) or not KEY_RE.fullmatch(key):
            raise CompassError("invalid applicability condition key")
        if not isinstance(match, str) or not match or len(match) > 160:
            raise CompassError("applicability values must be nonempty short strings")
        result[key] = match
    return result


def project_scope(scope: str, project_id: str | None) -> tuple[str, str]:
    if scope not in SCOPES:
        raise CompassError("scope must be personal or project")
    if scope == "personal":
        if project_id not in (None, ""):
            raise CompassError("personal principle cannot be owned by a project")
        return "personal", "*"
    if not isinstance(project_id, str) or not KEY_RE.fullmatch(project_id):
        raise CompassError("project principles require a canonical project identifier")
    return "project", project_id


def clean_text(value: Any, label: str, max_chars: int = 2000) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > max_chars:
        raise CompassError(label + " must be a nonempty bounded string")
    return value.strip()


def validate_proposal(p: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(p, dict):
        raise CompassError("proposal must be an object")
    scope, owner = project_scope(p.get("scope"), p.get("project_id"))
    key = p.get("key")
    if not isinstance(key, str) or not KEY_RE.fullmatch(key):
        raise CompassError("principle key must be stable: [a-z][a-z0-9_.-]{1,63}")
    operation = p.get("operation")
    if operation not in OPERATIONS:
        raise CompassError("operation must be set or revoke")
    expected = p.get("expected_revision")
    if type(expected) is not int or expected < 0:
        raise CompassError("expected_revision must be an explicit nonnegative integer")
    text = clean_text(p.get("content"), "content") if operation == "set" else ""
    reason = clean_text(p.get("reason"), "human rationale")
    if operation == "revoke" and expected == 0:
        raise CompassError("cannot revoke unknown principle")
    return {
        "scope": scope, "owner": owner, "key": key, "operation": operation,
        "expected_revision": expected, "content": text, "reason": reason,
        "conditions": validate_conditions(p.get("conditions")),
        "source": clean_text(p.get("source", "manual-proposal"), "source", 400),
        "author": clean_text(p.get("author"), "author", 120),
    }


class CompassStore:
    def __init__(self, db_path: str | Path):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), isolation_level=None, timeout=20)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA busy_timeout = 20000")
        self.db.execute("""
          CREATE TABLE IF NOT EXISTS compass_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            kind TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            payload TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            event_hash TEXT NOT NULL
          )
        """)
        self.db.execute("CREATE INDEX IF NOT EXISTS ix_compass_kind ON compass_events(kind)")

    def close(self):
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, typ, value, traceback):
        self.close()

    def _events(self) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM compass_events ORDER BY seq").fetchall()
        return [{**dict(x), "payload": json.loads(x["payload"])} for x in rows]

    @staticmethod
    def _digest(e: dict[str, Any]) -> str:
        fields = {k: e[k] for k in ("seq", "event_id", "kind", "timestamp", "payload", "prev_hash")}
        return hashlib.sha256(canonical(fields).encode("utf-8")).hexdigest()

    def verify(self) -> dict[str, Any]:
        events = self._events()
        prior = ZERO_HASH
        for count, event in enumerate(events, start=1):
            if event["seq"] != count or event["prev_hash"] != prior or self._digest(event) != event["event_hash"]:
                raise CompassError(f"journal integrity failure at event {count} (seq {event['seq']})")
            prior = event["event_hash"]
        return {"ok": True, "count": len(events), "head_hash": prior,
                "warning": "hash chain is unanchored and does not authenticate authors"}

    @contextlib.contextmanager
    def _transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.verify()
            yield
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise

    def _append(self, kind: str, payload: dict[str, Any]) -> str:
        row = self.db.execute("SELECT seq,event_hash FROM compass_events ORDER BY seq DESC LIMIT 1").fetchone()
        seq = row["seq"] + 1 if row else 1
        event = {"seq": seq, "event_id": str(uuid.uuid4()), "kind": kind,
                 "timestamp": now(), "payload": payload,
                 "prev_hash": row["event_hash"] if row else ZERO_HASH}
        event["event_hash"] = self._digest(event)
        self.db.execute("""
          INSERT INTO compass_events(seq,event_id,kind,timestamp,payload,prev_hash,event_hash)
          VALUES (?,?,?,?,?,?,?)
        """, (seq, event["event_id"], kind, event["timestamp"], canonical(payload),
               event["prev_hash"], event["event_hash"]))
        return event["event_id"]

    def _replay(self) -> dict[str, Any]:
        self.verify()
        proposals: dict[str, dict[str, Any]] = {}
        decisions: dict[str, str] = {}
        principle_history: dict[tuple[str, str], list[dict[str, Any]]] = {}
        observations = []
        for event in self._events():
            kind = event["kind"]
            payload = event["payload"]
            if kind == "OBSERVED":
                observations.append({"event_id": event["event_id"], **payload})
            elif kind == "PROPOSED":
                proposals[event["event_id"]] = {"status": "PENDING", **payload,
                                                  "proposal_id": event["event_id"],
                                                  "proposed_at": event["timestamp"]}
            elif kind in ("APPROVED", "REJECTED"):
                pid = payload["proposal_id"]
                if pid not in proposals or proposals[pid]["status"] != "PENDING":
                    raise CompassError("invalid proposal transition in journal")
                proposals[pid]["status"] = kind
                proposals[pid]["decision"] = {"actor": payload["actor"], "reason": payload["reason"],
                                               "timestamp": event["timestamp"],
                                               "event_id": event["event_id"]}
                decisions[pid] = kind
                if kind == "APPROVED":
                    proposal = proposals[pid]
                    identity = (proposal["owner"], proposal["key"])
                    history = principle_history.setdefault(identity, [])
                    old = history[-1] if history else None
                    expected = old["revision"] if old else 0
                    if expected != proposal["expected_revision"]:
                        raise CompassError("invalid stale-revision transition in journal")
                    if proposal["operation"] == "revoke" and not old:
                        raise CompassError("cannot revoke absent principle")
                    if old:
                        old["status"] = "SUPERSEDED"
                    revision = expected + 1
                    history.append({
                        "scope": proposal["scope"], "owner": proposal["owner"],
                        "key": proposal["key"], "revision": revision,
                        "content": proposal["content"] if proposal["operation"] == "set" else old["content"],
                        "reason": proposal["reason"],
                        "conditions": proposal["conditions"] if proposal["operation"] == "set" else old["conditions"],
                        "status": "ACTIVE" if proposal["operation"] == "set" else "REVOKED",
                        "source": proposal["source"], "author": proposal["author"],
                        "approval_actor": payload["actor"],
                        "approval_time": event["timestamp"],
                        "proposal_id": pid, "decision_event_id": event["event_id"],
                        "previous_revision": expected or None,
                    })
            else:
                raise CompassError("unknown event kind in journal")
        return {"proposals": proposals, "principles": principle_history,
                "observations": observations}

    def propose(self, *, scope: str, project_id: str | None = None,
                key: str, operation: str, content: str = "",
                reason: str, expected_revision: int, author: str,
                source: str = "manual-proposal", conditions: dict[str, str] | None = None) -> str:
        doc = validate_proposal(locals() | {"scope": scope})
        with self._transaction():
            self._replay()
            return self._append("PROPOSED", doc)

    def observe(self, raw: str, *, author: str, source: str = "external-message") -> dict[str, Any]:
        raw = clean_text(raw, "raw observation", 20000)
        author = clean_text(author, "author", 120)
        source = clean_text(source, "source", 400)
        parsed = parse_explicit_directive(raw)
        if parsed is None:
            with self._transaction():
                entry = self._append("OBSERVED", {"text": raw, "author": author,
                                                  "source": source, "classification": "UNCLASSIFIED"})
            return {"status": "UNCLASSIFIED", "observation_id": entry,
                    "note": "unstructured text did not mutate any active principle"}
        doc = {**parsed, "author": author, "source": source}
        try:
            result = self.propose(**doc)
        except CompassError:
            with self._transaction():
                entry = self._append("OBSERVED", {"text": raw, "author": author,
                                                  "source": source, "classification": "NEEDS_CLARIFICATION"})
            return {"status": "NEEDS_CLARIFICATION", "observation_id": entry,
                    "note": "invalid or ambiguous directive; no active principle changed"}
        return {"status": "PENDING_APPROVAL", "proposal_id": result,
                "note": "an explicit approver must commit the requested update"}

    def _decision(self, proposal_id: str, actor: str, kind: str, reason: str) -> dict[str, Any]:
        actor = clean_text(actor, "approver", 120)
        reason = clean_text(reason, "approval or rejection reason")
        with self._transaction():
            current = self._replay()
            proposals = current["proposals"]
            if proposal_id not in proposals or proposals[proposal_id]["status"] != "PENDING":
                raise CompassError("proposal is missing or not pending")
            p = proposals[proposal_id]
            if kind == "APPROVED":
                history = current["principles"].get((p["owner"], p["key"]), [])
                old = history[-1] if history else None
                revision = old["revision"] if old else 0
                if revision != p["expected_revision"]:
                    raise CompassError("stale proposal; expected_revision does not equal current revision")
                if p["operation"] == "revoke" and (not old or old["status"] != "ACTIVE"):
                    raise CompassError("nothing currently active to revoke")
            event_id = self._append(kind, {"proposal_id": proposal_id, "actor": actor,
                                           "reason": reason})
        return {"status": kind, "proposal_id": proposal_id, "decision_event_id": event_id}

    def approve(self, proposal_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        return self._decision(proposal_id, actor, "APPROVED", reason)

    def reject(self, proposal_id: str, *, actor: str, reason: str) -> dict[str, Any]:
        return self._decision(proposal_id, actor, "REJECTED", reason)

    def principles(self, *, scope: str | None = None, project_id: str | None = None,
                   include_history: bool = False) -> list[dict[str, Any]]:
        state = self._replay()
        if scope is not None:
            scope, owner = project_scope(scope, project_id)
        else:
            owner = None
        result = []
        for (identity, key), versions in state["principles"].items():
            if owner is not None and identity != owner:
                continue
            if include_history:
                result += copy_record(versions)
            elif versions[-1]["status"] == "ACTIVE":
                result.append(dict(versions[-1]))
        return sorted(result, key=lambda r: (r["scope"], r["owner"], r["key"], r["revision"]))

    def proposals(self, status: str = "PENDING") -> list[dict[str, Any]]:
        return [dict(p) for p in self._replay()["proposals"].values() if p["status"] == status]

    def resolve(self, *, project_id: str, context: dict[str, str] | None = None) -> dict[str, Any]:
        _, project_id = project_scope("project", project_id)
        context = validate_conditions(context)
        state = self._replay()
        resolved = {"personal": [], "project": []}
        excluded = []
        for (owner, _), versions in state["principles"].items():
            row = versions[-1]
            if row["status"] != "ACTIVE" or owner not in ("*", project_id):
                continue
            missing = {k: value for k, value in row["conditions"].items()
                       if context.get(k) != value}
            if missing:
                excluded.append({"scope": row["scope"], "owner": row["owner"], "key": row["key"],
                                 "revision": row["revision"], "reason": "APPLICABILITY_MISMATCH",
                                 "conditions_not_met": missing})
            else:
                resolved[row["scope"]].append(dict(row))
        for name in resolved:
            resolved[name].sort(key=lambda p: p["key"])
        return {"project_id": project_id, "context": context, "applicable": resolved,
                "excluded": excluded, "warning": "principles are guidance, not verified binding policy"}

    def timeline(self, limit: int | None = None) -> list[dict[str, Any]]:
        self.verify()
        result = self._events()
        if limit is not None:
            if limit < 1:
                raise CompassError("limit must be positive")
            result = result[-limit:]
        return result


def copy_record(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item) for item in history]


def parse_explicit_directive(raw: str) -> dict[str, Any] | None:
    """Conservative field syntax; not general intent recognition.

    COMPASS: scope=project; project=alpha; key=vision; operation=set;
    content=...; reason=...; expected_revision=0
    """
    if not raw.startswith("COMPASS:"):
        return None
    parts = raw[len("COMPASS:"):].strip().split(";")
    fields: dict[str, str] = {}
    allow = {"scope", "project", "key", "operation", "content", "reason", "expected_revision"}
    try:
        for part in parts:
            name, value = part.split("=", 1)
            name = name.strip()
            if name not in allow or name in fields:
                return None
            fields[name] = value.strip()
        if set(fields) - {"project"} != allow - {"project"}:
            return None
        expected = int(fields["expected_revision"])
        if str(expected) != fields["expected_revision"]:
            return None
        return {"scope": fields["scope"],
                "project_id": fields.get("project") or None,
                "key": fields["key"], "operation": fields["operation"],
                "content": fields["content"], "reason": fields["reason"],
                "expected_revision": expected}
    except (ValueError, KeyError):
        return None


def _context(raw: list[str]) -> dict[str, str]:
    result = {}
    for entry in raw:
        if "=" not in entry:
            raise CompassError("context must contain key=value")
        key, value = entry.split("=", 1)
        result[key] = value
    return validate_conditions(result)


def run_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="COMPASS V9.12 local memory (no model or cloud)")
    parser.add_argument("--db", required=True, help="SQLite database path")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("init")
    pp = actions.add_parser("propose")
    pp.add_argument("--scope", required=True, choices=sorted(SCOPES))
    pp.add_argument("--project")
    pp.add_argument("--key", required=True)
    pp.add_argument("--operation", required=True, choices=sorted(OPERATIONS))
    pp.add_argument("--content", default="")
    pp.add_argument("--reason", required=True)
    pp.add_argument("--expected-revision", type=int, required=True)
    pp.add_argument("--author", required=True)
    pp.add_argument("--source", default="manual-cli")
    pp.add_argument("--condition", action="append", default=[])
    o = actions.add_parser("ingest")
    o.add_argument("--text", required=True)
    o.add_argument("--author", required=True)
    o.add_argument("--source", default="external-message")
    for act in ("approve", "reject"):
        choice = actions.add_parser(act)
        choice.add_argument("proposal_id")
        choice.add_argument("--actor", required=True)
        choice.add_argument("--reason", required=True)
    li = actions.add_parser("list")
    li.add_argument("--scope", choices=sorted(SCOPES))
    li.add_argument("--project")
    li.add_argument("--history", action="store_true")
    resol = actions.add_parser("resolve")
    resol.add_argument("--project", required=True)
    resol.add_argument("--context", action="append", default=[])
    actions.add_parser("verify")
    ti = actions.add_parser("timeline")
    ti.add_argument("--limit", type=int)
    actions.add_parser("pending")
    args = parser.parse_args(argv)
    try:
        with CompassStore(args.db) as store:
            if args.command == "init":
                result = store.verify()
            elif args.command == "propose":
                result = {"proposal_id": store.propose(
                    scope=args.scope, project_id=args.project,
                    key=args.key, operation=args.operation, content=args.content,
                    reason=args.reason, expected_revision=args.expected_revision,
                    author=args.author, source=args.source,
                    conditions=_context(args.condition)),
                    "status": "PENDING_APPROVAL"}
            elif args.command == "ingest":
                result = store.observe(args.text, author=args.author, source=args.source)
            elif args.command == "approve":
                result = store.approve(args.proposal_id, actor=args.actor, reason=args.reason)
            elif args.command == "reject":
                result = store.reject(args.proposal_id, actor=args.actor, reason=args.reason)
            elif args.command == "list":
                result = store.principles(scope=args.scope, project_id=args.project,
                                          include_history=args.history)
            elif args.command == "resolve":
                result = store.resolve(project_id=args.project, context=_context(args.context))
            elif args.command == "verify":
                result = store.verify()
            elif args.command == "timeline":
                result = store.timeline(args.limit)
            else:
                result = store.proposals()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except CompassError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(run_cli())
