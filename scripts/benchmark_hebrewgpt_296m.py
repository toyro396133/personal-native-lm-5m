#!/usr/bin/env python3
"""HebrewGPT-296M compatibility wrapper for the generic backbone benchmark.

The upstream model's documented tokenizer path is direct SentencePiece. Its
tokenizer_config currently makes recent Transformers try an invalid fast-tokenizer
conversion, so this wrapper deliberately follows the model author's documented
SentencePiece usage while reusing the exact same benchmark measurements.
"""
from __future__ import annotations

import sys

import sentencepiece as spm
import torch
from huggingface_hub import hf_hub_download

import benchmark_hebrew_backbone as bench


class SentencePieceAdapter:
    def __init__(self, model_file: str):
        self.sp = spm.SentencePieceProcessor(model_file=model_file)
        self.bos_token_id = int(self.sp.bos_id())
        self.eos_token_id = int(self.sp.eos_id())
        self.pad_token_id = int(self.sp.pad_id())
        if self.eos_token_id < 0:
            self.eos_token_id = 2
        if self.pad_token_id < 0:
            self.pad_token_id = self.eos_token_id
        self.sep_token_id = None
        self.model_max_length = 512

    def __len__(self):
        return int(self.sp.get_piece_size())

    def __call__(self, text, return_tensors=None, add_special_tokens=True, **kwargs):
        ids = list(self.sp.encode(str(text), out_type=int))
        if add_special_tokens and self.bos_token_id >= 0:
            ids = [self.bos_token_id] + ids
        if add_special_tokens and self.eos_token_id >= 0:
            ids = ids + [self.eos_token_id]
        if return_tensors == "pt":
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=True, **kwargs):
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().reshape(-1).tolist()
        else:
            ids = [int(x) for x in ids]
        if skip_special_tokens:
            special = {self.bos_token_id, self.eos_token_id, self.pad_token_id}
            ids = [x for x in ids if x not in special and x >= 0]
        return self.sp.decode(ids)


class DirectSentencePieceAutoTokenizer:
    @staticmethod
    def from_pretrained(model_id, revision=None, **kwargs):
        model_file = hf_hub_download(
            repo_id=model_id,
            filename="tokenizer.model",
            revision=revision,
        )
        return SentencePieceAdapter(model_file)


if __name__ == "__main__":
    bench.AutoTokenizer = DirectSentencePieceAutoTokenizer
    raise SystemExit(bench.main())
