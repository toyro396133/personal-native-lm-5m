from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from prepare_v07_corpus import gather_wiki, gather_tree, cleaned_unique, deterministic_take, score
from train_hebrew import load_tokenizer


def split_source(lines, validation_per_mille):
    train, val = [], []
    for line in lines:
        if score("split:" + line) % 1000 < validation_per_mille:
            val.append(line)
        else:
            train.append(line)
    return train, val


def encoded_token_count(tok, lines):
    if not lines:
        return 0
    return len(tok.encode("\n".join(lines), bos=False, eos=False))


def repeat_to_target(tok, lines, target_tokens):
    """Repeat a source deterministically until its encoded token mass reaches target."""
    base_tokens = encoded_token_count(tok, lines)
    if base_tokens <= 0:
        return [], 0, 0.0
    repeats = max(1, target_tokens // base_tokens)
    out = list(lines) * repeats
    tokens = encoded_token_count(tok, out)
    if tokens < target_tokens:
        remaining = target_tokens - tokens
        ordered = sorted(lines, key=lambda x: score("extra:" + x))
        extra = []
        running = 0
        for line in ordered:
            extra.append(line)
            running += max(1, len(tok.encode(line, bos=False, eos=False)))
            if running >= remaining:
                break
        out.extend(extra)
        tokens = encoded_token_count(tok, out)
    return out, tokens, len(out) / max(1, len(lines))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--wiki", required=True)
    ap.add_argument("--knesset-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--wiki-cap", type=int, default=32000)
    ap.add_argument("--knesset-cap", type=int, default=110000)
    ap.add_argument("--validation-per-mille", type=int, default=60)
    ap.add_argument("--out", default="data/v09-balanced-train.txt")
    ap.add_argument("--report", default="artifacts/v09_balance_report.json")
    args=ap.parse_args()

    tok, kind = load_tokenizer(args.tokenizer)

    wiki = deterministic_take(cleaned_unique(gather_wiki(Path(args.wiki))), args.wiki_cap)
    kn = deterministic_take(cleaned_unique(gather_tree(Path(args.knesset_root))), args.knesset_cap)

    wiki_train, wiki_val = split_source(wiki, args.validation_per_mille)
    kn_train, kn_val = split_source(kn, args.validation_per_mille)

    wiki_tokens = encoded_token_count(tok, wiki_train)
    kn_tokens = encoded_token_count(tok, kn_train)
    target = max(wiki_tokens, kn_tokens)

    wiki_weighted, wiki_weighted_tokens, wiki_repeat = repeat_to_target(tok, wiki_train, target)
    kn_weighted, kn_weighted_tokens, kn_repeat = repeat_to_target(tok, kn_train, target)

    tagged = [("wiki", x) for x in wiki_weighted] + [("knesset", x) for x in kn_weighted]
    tagged.sort(key=lambda p: score("mix:" + p[0] + ":" + p[1] + ":" + str(len(tagged))))
    mixed = [x for _, x in tagged]

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(mixed) + "\n", encoding="utf-8")

    report = {
        "version":"0.9",
        "tokenizer_kind":kind,
        "strategy":"equalize source token mass by deterministic oversampling; preserve all v0.8 unique training lines",
        "unique_train_lines":{"wikipedia":len(wiki_train),"knesset":len(kn_train)},
        "shared_validation_lines":{"wikipedia":len(wiki_val),"knesset":len(kn_val)},
        "pre_balance_tokens":{"wikipedia":wiki_tokens,"knesset":kn_tokens},
        "post_balance_tokens":{"wikipedia":wiki_weighted_tokens,"knesset":kn_weighted_tokens},
        "repeat_factor":{"wikipedia":wiki_repeat,"knesset":kn_repeat},
        "mixed_lines":len(mixed),
        "post_balance_wikipedia_share":wiki_weighted_tokens/max(1,wiki_weighted_tokens+kn_weighted_tokens),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
