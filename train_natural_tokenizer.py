from __future__ import annotations

import argparse
from hybrid_tokenizer import HybridHebrewTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--out", default="hebrew-natural-hybrid-4096.json")
    args = ap.parse_args()

    tok = HybridHebrewTokenizer.train(args.files, vocab_size=args.vocab_size, min_frequency=2)
    # Keep ASCII digits as raw byte IDs. The personalization benchmark uses
    # 1/2/5 as compact labels and older benchmark code expects those byte IDs.
    filtered = {piece: idx for piece, idx in tok.lexicon.items() if not piece.isdigit()}
    tok = HybridHebrewTokenizer(filtered, vocab_size=args.vocab_size)
    tok.save(args.out)
    print(f"saved={args.out} vocab={tok.vocab_size} lexicon={len(tok.lexicon)}")

if __name__ == "__main__":
    main()
