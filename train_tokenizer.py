import argparse
from bpe_tokenizer import BPETokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="UTF-8 corpus files")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", default="hebrew-bpe-4096.json")
    args = ap.parse_args()

    tok = BPETokenizer.train(args.files, vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"saved tokenizer: {args.out}")
    print(f"vocab size: {tok.vocab_size}")

if __name__ == "__main__":
    main()
