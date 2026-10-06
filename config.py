from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int = 260
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = 1024
    max_seq_len: int = 512

    state_dim: int = 228
    personal_tokens: int = 4
    reader_hidden: int = 256
    dropout: float = 0.0

    @property
    def head_dim(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @classmethod
    def byte_prototype(cls):
        # ~5.26M params, zero tokenizer dependencies.
        return cls()

    @classmethod
    def hebrew_bpe_5m(cls, vocab_size=4096):
        # ~4.92M params at vocab=4096, including T_read.
        # 224 / 8 = 28 dimensions per attention head.
        return cls(
            vocab_size=vocab_size,
            d_model=224,
            n_heads=8,
            n_layers=6,
            d_ff=896,
            max_seq_len=512,
            state_dim=228,
            personal_tokens=4,
            reader_hidden=224,
        )
