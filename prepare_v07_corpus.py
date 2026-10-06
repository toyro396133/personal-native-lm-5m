from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

HEB_RE = re.compile(r"[\u0590-\u05FF]")
LETTER_RE = re.compile(r"[A-Za-z\u0590-\u05FF]")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://|www\.", re.I)


def normalize(line: str) -> str:
    line = line.replace("\ufeff", "").replace("\x00", " ").strip()
    line = SPACE_RE.sub(" ", line)
    return line


def acceptable(line: str, min_chars=28, max_chars=420, min_hebrew_ratio=0.62) -> bool:
    if len(line) < min_chars or len(line) > max_chars:
        return False
    if URL_RE.search(line):
        return False
    letters = LETTER_RE.findall(line)
    if len(letters) < 16:
        return False
    heb = HEB_RE.findall(line)
    if len(heb) / max(1, len(letters)) < min_hebrew_ratio:
        return False
    lowered = line.lower()
    wiki_noise = (
        "קישורים חיצוניים", "ברוך בואך לוויקיפדיה", "רשימת ההמתנה",
        "ויקיפדיה האנגלית כאן", "הערך חוזר", "הקצרמרים",
    )
    if any(x in lowered for x in wiki_noise):
        return False
    return True


def score(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], 16)


def read_lines(path: Path):
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def gather_wiki(path: Path):
    return [normalize(x) for x in read_lines(path)]


def gather_tree(root: Path):
    lines = []
    for path in sorted(root.rglob("*.txt")):
        try:
            lines.extend(normalize(x) for x in read_lines(path))
        except Exception:
            continue
    return lines


def cleaned_unique(lines):
    seen = set()
    out = []
    for line in lines:
        if not acceptable(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        out.append(line)
    return out


def deterministic_take(lines, n):
    if n <= 0 or len(lines) <= n:
        return sorted(lines, key=score)
    return sorted(lines, key=score)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--knesset-root", required=True)
    ap.add_argument("--wiki-cap", type=int, default=32000)
    ap.add_argument("--knesset-cap", type=int, default=110000)
    ap.add_argument("--validation-per-mille", type=int, default=60)
    ap.add_argument("--train-out", default="data/v07-train.txt")
    ap.add_argument("--val-out", default="data/v07-val.txt")
    ap.add_argument("--report", default="artifacts/v07_corpus_report.json")
    args = ap.parse_args()

    wiki_raw = gather_wiki(Path(args.wiki))
    kn_raw = gather_tree(Path(args.knesset_root))
    wiki = deterministic_take(cleaned_unique(wiki_raw), args.wiki_cap)
    kn = deterministic_take(cleaned_unique(kn_raw), args.knesset_cap)

    combined = []
    seen = set()
    for source, rows in (("wikipedia", wiki), ("knesset-public-domain", kn)):
        for line in rows:
            if line in seen:
                continue
            seen.add(line)
            combined.append((source, line))

    train, val = [], []
    source_counts = {"wikipedia": {"train":0,"val":0}, "knesset-public-domain":{"train":0,"val":0}}
    for source, line in combined:
        bucket = score("split:" + line) % 1000
        if bucket < args.validation_per_mille:
            val.append(line); source_counts[source]["val"] += 1
        else:
            train.append(line); source_counts[source]["train"] += 1

    Path(args.train_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.train_out).write_text("\n".join(train) + "\n", encoding="utf-8")
    Path(args.val_out).write_text("\n".join(val) + "\n", encoding="utf-8")

    report = {
        "version":"0.7",
        "sources":{
            "wikipedia":{"license":"CC BY-SA 3.0","raw_lines":len(wiki_raw),"selected_clean_lines":len(wiki)},
            "knesset-2004-2005":{"license":"Public Domain","raw_lines":len(kn_raw),"selected_clean_lines":len(kn)}
        },
        "combined_unique_lines":len(combined),
        "train_lines":len(train),
        "validation_lines":len(val),
        "source_split_counts":source_counts,
        "validation_per_mille":args.validation_per_mille,
    }
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if len(train) < 50000 or len(val) < 1000:
        raise SystemExit("v0.7 corpus unexpectedly small")

if __name__ == "__main__":
    main()
