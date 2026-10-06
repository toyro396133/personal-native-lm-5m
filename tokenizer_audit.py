from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from train_hebrew import load_tokenizer

HEB_WORD = re.compile(r"[\u05D0-\u05EA]+")


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("tokenizer")
    ap.add_argument("text")
    ap.add_argument("--out",default="tokenizer_audit.json")
    ap.add_argument("--sample-chars",type=int,default=500000)
    args=ap.parse_args()

    tok,kind=load_tokenizer(args.tokenizer)
    text=Path(args.text).read_text(encoding="utf-8")[:args.sample_chars]
    ids=tok.encode(text,bos=False,eos=False)
    roundtrip=tok.decode(ids)
    words=HEB_WORD.findall(text)
    report={
        "tokenizer_kind":kind,
        "vocab_size":tok.vocab_size,
        "sample_chars":len(text),
        "sample_tokens":len(ids),
        "tokens_per_100_chars":len(ids)/max(1,len(text))*100,
        "hebrew_words":len(words),
        "tokens_per_hebrew_word":len(ids)/max(1,len(words)),
        "replacement_chars_after_roundtrip":roundtrip.count("\ufffd"),
        "roundtrip_exact":roundtrip==text,
        "label_token_ids":{x:tok.encode(x,bos=False,eos=False) for x in ("1","2","5")},
        "unk_tokens":sum(i==getattr(tok,"unk_id",-999999) for i in ids),
    }
    Path(args.out).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if report["replacement_chars_after_roundtrip"] != 0:
        raise SystemExit("Tokenizer produced Unicode replacement characters")
    if any(len(v)!=1 for v in report["label_token_ids"].values()):
        raise SystemExit("Personal benchmark labels are not single tokens")

if __name__=="__main__":
    main()
