from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from prepare_v07_corpus import acceptable, cleaned_unique, deterministic_take, gather_tree, normalize, score
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


def take_to_token_target(tok, ordered_lines, target_tokens):
    out = []
    running = 0
    for line in ordered_lines:
        out.append(line)
        running += len(tok.encode(line, bos=False, eos=False)) + 1
        if running >= target_tokens:
            break
    return out


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--old-wiki", required=True)
    ap.add_argument("--knesset-root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--dataset", default="tomron87/hebrew-wikipedia-sentences-corpus")
    ap.add_argument("--ratios", default="0.12,0.20,0.30,0.40")
    ap.add_argument("--knesset-cap", type=int, default=110000)
    ap.add_argument("--validation-per-mille", type=int, default=60)
    ap.add_argument("--shuffle-buffer", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--report", default="artifacts/v11_ratio_build_report.json")
    args=ap.parse_args()

    ratios=[float(x) for x in args.ratios.split(",")]
    tok,kind=load_tokenizer(args.tokenizer)

    kn_all=deterministic_take(cleaned_unique(gather_tree(Path(args.knesset_root))),args.knesset_cap)
    kn_train,kn_val=split_source(kn_all,args.validation_per_mille)
    kn_tokens=exact_tokens(tok,kn_train)

    old_wiki_lines={
        normalize(x)
        for x in Path(args.old_wiki).read_text(encoding="utf-8",errors="ignore").splitlines()
        if normalize(x)
    }

    max_ratio=max(ratios)
    max_wiki_target=int(kn_tokens*max_ratio/(1.0-max_ratio))

    ds=load_dataset(args.dataset,split="train",streaming=True)
    ds=ds.shuffle(seed=args.seed,buffer_size=args.shuffle_buffer)

    wiki_pool=[]
    seen=set()
    running=0
    rejected=excluded_old=duplicate=0
    for row in ds:
        line=normalize(str(row.get("sentence","")))
        if not acceptable(line,min_chars=28,max_chars=420,min_hebrew_ratio=0.62):
            rejected+=1
            continue
        if line in old_wiki_lines:
            excluded_old+=1
            continue
        if line in seen:
            duplicate+=1
            continue
        seen.add(line)
        wiki_pool.append(line)
        running += len(tok.encode(line,bos=False,eos=False))+1
        if running >= int(max_wiki_target*1.03):
            break

    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    mixes={}
    for ratio in ratios:
        target=int(kn_tokens*ratio/(1.0-ratio))
        wiki_lines=take_to_token_target(tok,wiki_pool,target)
        wiki_tokens=exact_tokens(tok,wiki_lines)
        tagged=[("wiki",x) for x in wiki_lines]+[("knesset",x) for x in kn_train]
        tagged.sort(key=lambda p:score(f"v11:{ratio:.3f}:"+p[0]+":"+p[1]))
        mixed=[x for _,x in tagged]
        key=f"r{int(round(ratio*100)):02d}"
        path=out_dir/f"v11-{key}.txt"
        path.write_text("\n".join(mixed)+"\n",encoding="utf-8")
        share=wiki_tokens/max(1,wiki_tokens+kn_tokens)
        mixes[key]={
            "requested_wikipedia_share":ratio,
            "actual_wikipedia_share":share,
            "wikipedia_lines":len(wiki_lines),
            "knesset_lines":len(kn_train),
            "wikipedia_tokens":wiki_tokens,
            "knesset_tokens":kn_tokens,
            "mixed_lines":len(mixed),
            "path":str(path),
        }

    report={
        "version":"0.11",
        "strategy":"unique-data source-ratio sweep; no oversampling",
        "tokenizer_kind":kind,
        "knesset_tokens":kn_tokens,
        "knesset_validation_lines":len(kn_val),
        "unique_wikipedia_pool_lines":len(wiki_pool),
        "old_wikipedia_sentences_excluded":len(old_wiki_lines),
        "stream_rejected":rejected,
        "stream_old_matches":excluded_old,
        "stream_duplicates":duplicate,
        "mixes":mixes,
    }
    Path(args.report).parent.mkdir(parents=True,exist_ok=True)
    Path(args.report).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
