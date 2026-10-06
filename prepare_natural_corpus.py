from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HEB_RE = re.compile(r"[\u0590-\u05FF]")
LETTER_RE = re.compile(r"[A-Za-z\u0590-\u05FF]")
SPACE_RE = re.compile(r"\s+")


def normalize(line: str) -> str:
    line = line.strip().replace("\ufeff", "")
    line = SPACE_RE.sub(" ", line)
    return line


def acceptable(line: str, min_chars: int, max_chars: int, min_hebrew_ratio: float) -> bool:
    if len(line) < min_chars or len(line) > max_chars:
        return False
    letters = LETTER_RE.findall(line)
    if not letters:
        return False
    heb = HEB_RE.findall(line)
    if len(heb) / max(1, len(letters)) < min_hebrew_ratio:
        return False
    lowered = line.lower()
    noisy = (
        "קישורים חיצוניים", "ברוך בואך לוויקיפדיה", "רשימת ההמתנה",
        "בהצבעה", "ויקיפדיה האנגלית כאן", "הערך חוזר", "הקצרמרים",
    )
    if any(x in lowered for x in noisy):
        return False
    return True


def stable_bucket(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16) % 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("--train-out", default="data/hebrew-natural-train.txt")
    ap.add_argument("--val-out", default="data/hebrew-natural-val.txt")
    ap.add_argument("--report", default="natural_corpus_report.json")
    ap.add_argument("--val-per-mille", type=int, default=80)
    ap.add_argument("--min-chars", type=int, default=35)
    ap.add_argument("--max-chars", type=int, default=360)
    ap.add_argument("--min-hebrew-ratio", type=float, default=0.68)
    args = ap.parse_args()

    src = Path(args.source)
    train_out = Path(args.train_out); train_out.parent.mkdir(parents=True, exist_ok=True)
    val_out = Path(args.val_out); val_out.parent.mkdir(parents=True, exist_ok=True)

    seen = set(); train = []; val = []
    total = rejected = duplicates = 0
    for raw in src.read_text(encoding="utf-8", errors="ignore").splitlines():
        total += 1
        line = normalize(raw)
        if not acceptable(line, args.min_chars, args.max_chars, args.min_hebrew_ratio):
            rejected += 1
            continue
        if line in seen:
            duplicates += 1
            continue
        seen.add(line)
        if stable_bucket(line) < args.val_per_mille:
            val.append(line)
        else:
            train.append(line)

    train_out.write_text("\n".join(train) + "\n", encoding="utf-8")
    val_out.write_text("\n".join(val) + "\n", encoding="utf-8")
    report = {
        "source": str(src),
        "source_license": "CC BY-SA 3.0 (derived from Hebrew Wikipedia)",
        "raw_lines": total,
        "accepted": len(seen),
        "train_lines": len(train),
        "validation_lines": len(val),
        "rejected": rejected,
        "duplicates": duplicates,
        "validation_per_mille": args.val_per_mille,
        "filters": {
            "min_chars": args.min_chars,
            "max_chars": args.max_chars,
            "min_hebrew_ratio": args.min_hebrew_ratio,
        },
    }
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not train or not val:
        raise SystemExit("Corpus split is empty; adjust filters")

if __name__ == "__main__":
    main()
