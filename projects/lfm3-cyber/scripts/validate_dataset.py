#!/usr/bin/env python3
"""Validate conversational JSONL and prevent scenario-level split leakage."""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

ROLES = {"system", "user", "assistant"}
SPLITS = {"train", "validation", "test"}
SCOPES = {"defensive", "authorized_lab", "conceptual"}


def validate_rows(rows: list[dict]) -> list[str]:
    errors: list[str] = []
    ids: set[str] = set()
    scenario_splits: dict[str, set[str]] = defaultdict(set)
    fingerprints: dict[str, str] = {}
    for line_no, row in enumerate(rows, 1):
        prefix = f"line {line_no}"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: row must be an object")
            continue
        required = {"id", "scenario_id", "split", "domain", "task", "source", "authorization", "messages"}
        missing = required - row.keys()
        if missing:
            errors.append(f"{prefix}: missing fields: {', '.join(sorted(missing))}")
            continue
        for key in ("id", "scenario_id", "domain", "task", "source"):
            if not isinstance(row[key], str) or not row[key].strip():
                errors.append(f"{prefix}: {key} must be a non-empty string")
        if row.get("split") not in SPLITS:
            errors.append(f"{prefix}: split must be one of {sorted(SPLITS)}")
        if isinstance(row.get("id"), str):
            if row["id"] in ids:
                errors.append(f"{prefix}: duplicate id {row['id']!r}")
            ids.add(row["id"])
        if isinstance(row.get("scenario_id"), str) and row.get("split") in SPLITS:
            scenario_splits[row["scenario_id"]].add(row["split"])
        auth = row.get("authorization")
        if not isinstance(auth, dict) or auth.get("scope") not in SCOPES or auth.get("status") not in {"explicit", "not_applicable"}:
            errors.append(f"{prefix}: invalid authorization metadata")
        elif auth.get("scope") == "authorized_lab":
            if auth.get("status") != "explicit" or not str(auth.get("permitted_target", "")).strip():
                errors.append(f"{prefix}: authorized_lab requires explicit status and permitted_target")
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) < 3:
            errors.append(f"{prefix}: messages must be a list with at least system, user and assistant turns")
            continue
        roles = []
        valid_messages = True
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in ROLES or not isinstance(message.get("content"), str) or not message["content"].strip():
                valid_messages = False
                break
            roles.append(message["role"])
        if not valid_messages or "user" not in roles or roles[-1] != "assistant":
            errors.append(f"{prefix}: messages need valid roles, a user turn, and a final assistant turn")
        else:
            fingerprint = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            previous = fingerprints.get(fingerprint)
            if previous and previous != row.get("split"):
                errors.append(f"{prefix}: identical conversation appears in multiple splits")
            fingerprints[fingerprint] = row.get("split")
    for scenario, splits in scenario_splits.items():
        if len(splits) > 1:
            errors.append(f"scenario_id {scenario!r} crosses splits: {', '.join(sorted(splits))}")
    return errors


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path)
    args = parser.parse_args()
    try:
        rows = load_jsonl(args.jsonl)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 2
    errors = validate_rows(rows)
    if errors:
        print("\n".join(f"ERROR: {error}" for error in errors))
        return 1
    counts = {split: sum(row.get("split") == split for row in rows) for split in sorted(SPLITS)}
    print(f"PASS: {len(rows)} examples; " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
