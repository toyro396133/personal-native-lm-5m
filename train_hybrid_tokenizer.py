import argparse
from hybrid_tokenizer import HybridHebrewTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default="hebrew-hybrid-4096.json")
    args = ap.parse_args()
    tok = HybridHebrewTokenizer.train(args.files, args.vocab_size, args.min_frequency)
    tok.save(args.out)
    print(f"saved={args.out} vocab_size={tok.vocab_size} lexicon={len(tok.lexicon)}")


if __name__ == "__main__":
    main()
