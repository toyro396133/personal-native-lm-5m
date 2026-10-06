from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset

from prepare_v07_corpus import (
    acceptable,
    cleaned_unique,
    deterministic_take,
    gather_tree,
    normalize,
    score,
)
from train_hebrew import load_tokenizer


def split_source(lines, validation_per_mille):
    train, val = [], []
    for line in lines:
        if score("split:" + line) % 1000 < validation_per_mille:
            val.append(line)
        else:
            train.append(line)
    return train, val


def exact_tokens(tok, lines):
    if not lines:
        return 0
    return len(tok.encode("\n".join(lines), bos=False, eos=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--old-wiki", required=True)
    ap.add_argument("--knesset-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default="tomron87/hebrew-wikipedia-sentences-corpus")
    ap.add_argument("--knesset-cap", type=int, default=110000)
    ap.add_argument("--validation-per-mille", type=int, default=60)
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out", default="data/v10-unique-balanced-train.txt")
    ap.add_argument("--report", default="artifacts/v10_unique_balance_report.json")
    args=ap.parse_args()

    tok, kind = load_tokenizer(args.tokenizer)

    # Keep exactly the Knesset pool/split used in v0.8/v0.9.
    kn_all = deterministic_take(
        cleaned_unique(gather_tree(Path(args.knesset_root))),
        args.knesset_cap,
    )
    kn_train, kn_val = split_source(kn_all, args.validation_per_mille)
    kn_tokens = exact_tokens(tok, kn_train)

    # Exclude every sentence from the old 50K Wikipedia source, whether it was
    # train or validation. This prevents leakage and makes the added Wikipedia
    # material genuinely new relative to v0.8.
    old_wiki_lines = {
        normalize(x)
        for x in Path(args.old_wiki).read_text(encoding="utf-8", errors="ignore").splitlines()
        if normalize(x)
    }

    ds = load_dataset(args.dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    selected = []
    seen = set()
    approx_tokens = 0
    rejected = excluded_old = duplicate = 0

    # Target the same token mass as the Knesset training source, using only
    # unique new Wikipedia sentences.
    for row in ds:
        line = normalize(str(row.get("sentence", "")))
        if not acceptable(line, min_chars=28, max_chars=420, min_hebrew_ratio=0.62):
            rejected += 1
            continue
        if line in old_wiki_lines:
            excluded_old += 1
            continue
        if line in seen:
            duplicate += 1
            continue
        seen.add(line)
        selected.append(line)
        approx_tokens += len(tok.encode(line, bos=False, eos=False)) + 1
        if approx_tokens >= kn_tokens:
            break

    wiki_tokens = exact_tokens(tok, selected)
    if wiki_tokens < int(kn_tokens * 0.95):
        raise SystemExit(
            f"unique Wikipedia sample too small: wiki_tokens={wiki_tokens} target={kn_tokens}"
        )

    tagged = [("wiki-new", x) for x in selected] + [("knesset", x) for x in kn_train]
    tagged.sort(key=lambda p: score("v10mix:" + p[0] + ":" + p[1]))
    mixed = [x for _, x in tagged]

    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text("\n".join(mixed)+"\n",encoding="utf-8")

    total=wiki_tokens+kn_tokens
    report={
        "version":"0.10",
        "tokenizer_kind":kind,
        "strategy":"balance Knesset with genuinely new deduplicated full-Wikipedia sentences; no source oversampling",
        "hf_dataset":args.dataset,
        "old_wikipedia_sentences_excluded":len(old_wiki_lines),
        "unique_new_wikipedia_lines":len(selected),
        "knesset_train_lines":len(kn_train),
        "knesset_validation_lines":len(kn_val),
        "tokens":{"wikipedia_new":wiki_tokens,"knesset":kn_tokens},
        "wikipedia_token_share":wiki_tokens/max(1,total),
        "mixed_lines":len(mixed),
        "rejected_stream_rows":rejected,
        "excluded_old_wikipedia_matches":excluded_old,
        "duplicate_stream_rows":duplicate,
        "oversampling_factor":1.0,
    }
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
