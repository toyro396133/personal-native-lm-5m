from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
import re

# IDs 0..255 are raw UTF-8 bytes. Special tokens follow.
PAD = 256
BOS = 257
EOS = 258
SEP = 259
BASE_VOCAB = 260

_TOKEN_RE = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)


class HybridHebrewTokenizer:
    """Dependency-free Hebrew-aware lexicon + byte fallback tokenizer.

    Frequent Hebrew words/punctuation get one token; everything else falls back
    losslessly to UTF-8 bytes. This gives us a ~4K vocabulary even in filtered
    environments where the optional Rust `tokenizers` package is unavailable.
    """

    def __init__(self, lexicon, vocab_size=None):
        self.lexicon = {k: int(v) for k, v in dict(lexicon).items()}
        self.inverse = {int(v): k for k, v in self.lexicon.items()}
        self.pad_id = PAD
        self.bos_id = BOS
        self.eos_id = EOS
        self.sep_id = SEP
        used = max([SEP] + list(self.inverse.keys())) + 1
        self.vocab_size = max(used, int(vocab_size or used))

    @classmethod
    def train(cls, files, vocab_size=4096, min_frequency=2):
        counter = Counter()
        for file in files:
            text = Path(file).read_text(encoding="utf-8")
            for piece in _TOKEN_RE.findall(text):
                if piece.isspace():
                    continue
                counter[piece] += 1
        room = max(0, vocab_size - BASE_VOCAB)
        items = [(t, c) for t, c in counter.most_common() if c >= min_frequency][:room]
        lexicon = {token: BASE_VOCAB + i for i, (token, _) in enumerate(items)}
        return cls(lexicon, vocab_size=vocab_size)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("type") != "hybrid-hebrew-v1":
            raise ValueError("not a hybrid-hebrew-v1 tokenizer")
        return cls(data["lexicon"], vocab_size=data.get("vocab_size"))

    def save(self, path):
        data = {
            "type": "hybrid-hebrew-v1",
            "base_vocab": BASE_VOCAB,
            "vocab_size": self.vocab_size,
            "lexicon": self.lexicon,
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def encode(self, text, bos=True, eos=True):
        ids = []
        for piece in _TOKEN_RE.findall(text):
            tok_id = self.lexicon.get(piece)
            if tok_id is not None:
                ids.append(tok_id)
            else:
                ids.extend(piece.encode("utf-8"))
        if bos:
            ids.insert(0, BOS)
        if eos:
            ids.append(EOS)
        return ids

    def decode(self, ids):
        out = []
        byte_buf = bytearray()

        def flush():
            if byte_buf:
                out.append(bytes(byte_buf).decode("utf-8", errors="replace"))
                byte_buf.clear()

        for i in ids:
            i = int(i)
            if 0 <= i < 256:
                byte_buf.append(i)
            elif i in (PAD, BOS, EOS, SEP):
                flush()
            else:
                flush()
                out.append(self.inverse.get(i, ""))
        flush()
        return "".join(out)
