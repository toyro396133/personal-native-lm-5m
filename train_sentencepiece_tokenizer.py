from __future__ import annotations

import argparse
from pathlib import Path
import sentencepiece as spm

from sentencepiece_tokenizer import SentencePieceTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--model-type", choices=["unigram", "bpe"], default="unigram")
    ap.add_argument("--out", default="hebrew-unigram-4096.model")
    args = ap.parse_args()

    out = Path(args.out)
    if out.suffix != ".model":
        raise SystemExit("--out must end in .model")
    out.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(out.with_suffix(""))

    spm.SentencePieceTrainer.train(
        input=",".join(str(Path(x)) for x in args.files),
        model_prefix=prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=1.0,
        normalization_rule_name="nmt_nfkc",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        pad_piece="<pad>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        unk_piece="<unk>",
        user_defined_symbols=["<sep>", "1", "2", "5"],
        hard_vocab_limit=True,
        shuffle_input_sentence=False,
        input_sentence_size=0,
        split_digits=False,
        byte_fallback=False,
    )

    tok = SentencePieceTokenizer.load(out)
    for label in ("1", "2", "5"):
        print(f"label={label} token_id={tok.single_token_id(label)}")
    print(f"saved={out} vocab={tok.vocab_size} type={args.model_type}")


if __name__ == "__main__":
    main()
