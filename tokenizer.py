class ByteTokenizer:
    """
    Dependency-free UTF-8 byte tokenizer for the architectural prototype.
    It is intentionally simple. For serious Hebrew pretraining, replace it
    with a trained 4K-8K BPE/Unigram tokenizer while keeping the same model API.
    """
    PAD = 256
    BOS = 257
    EOS = 258
    SEP = 259
    vocab_size = 260

    def encode(self, text: str, bos=True, eos=True):
        ids = list(text.encode("utf-8"))
        if bos:
            ids = [self.BOS] + ids
        if eos:
            ids = ids + [self.EOS]
        return ids

    def decode(self, ids):
        raw = bytes([i for i in ids if 0 <= i < 256])
        return raw.decode("utf-8", errors="replace")
