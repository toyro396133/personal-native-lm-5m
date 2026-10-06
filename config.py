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
    controller_dim: int = 128
    dropout: float = 0.0

    @property
    def head_dim(self):
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @classmethod
    def byte_prototype(cls):
        return cls()

    @classmethod
    def hebrew_bpe_5m(cls, vocab_size=4096):
        return cls(
            vocab_size=vocab_size,
            d_model=224,
            n_heads=8,
            n_layers=6,
            d_ff=896,
            max_seq_len=512,
            state_dim=228,
            controller_dim=112,
        )

    @classmethod
    def hebrew_bpe_10m(cls, vocab_size=4096):
        """Capacity-control profile for v0.14.

        Keeps the same PersonalState ABI, tokenizer vocabulary size and
        conditioning design while increasing width/depth to roughly 10M params.
        """
        return cls(
            vocab_size=vocab_size,
            d_model=288,
            n_heads=8,
            n_layers=8,
            d_ff=1152,
            max_seq_len=512,
            state_dim=228,
            controller_dim=144,
        )
