from __future__ import annotations

from pathlib import Path
import sentencepiece as spm


class SentencePieceTokenizer:
    """Unicode-safe SentencePiece tokenizer wrapper.

    Unlike the hybrid byte fallback tokenizer, every normal generated piece is
    a Unicode string piece. Unknown characters use SentencePiece's <unk> token
    rather than emitting arbitrary partial UTF-8 byte sequences.
    """

    def __init__(self, model_file):
        self.model_file = str(model_file)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_file)
        self.pad_id = int(self.sp.pad_id())
        self.bos_id = int(self.sp.bos_id())
        self.eos_id = int(self.sp.eos_id())
        self.unk_id = int(self.sp.unk_id())
        self.vocab_size = int(self.sp.get_piece_size())
        self.PAD = self.pad_id

    @classmethod
    def load(cls, path):
        return cls(path)

    def encode(self, text: str, bos=True, eos=True):
        ids = list(self.sp.encode(text, out_type=int))
        if bos:
            ids = [self.bos_id] + ids
        if eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids):
        special = {self.pad_id, self.bos_id, self.eos_id}
        clean = [int(i) for i in ids if int(i) not in special and int(i) >= 0]
        return self.sp.decode(clean)

    def single_token_id(self, text: str) -> int:
        ids = self.encode(text, bos=False, eos=False)
        if len(ids) != 1:
            raise ValueError(f"{text!r} must encode to one token, got {ids}")
        return int(ids[0])

    def count_unk(self, ids) -> int:
        return sum(int(i) == self.unk_id for i in ids)
