from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.dropout = cfg.dropout

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class PersonalStateReader(nn.Module):
    """
    T_read: canonical state -> learned personal prefix tokens.

    The reader is shared by every user. The PersonalState itself is not
    serialized into natural-language prompt text.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.personal_tokens = cfg.personal_tokens
        self.d_model = cfg.d_model
        self.net = nn.Sequential(
            nn.LayerNorm(cfg.state_dim),
            nn.Linear(cfg.state_dim, cfg.reader_hidden),
            nn.GELU(),
            nn.Linear(cfg.reader_hidden, cfg.personal_tokens * cfg.d_model),
        )

    def forward(self, state):
        z = self.net(state)
        return z.view(state.shape[0], self.personal_tokens, self.d_model)

class PersonalNativeLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # Room for prefix + text.
        self.position_embedding = nn.Embedding(
            cfg.max_seq_len + cfg.personal_tokens, cfg.d_model
        )
        self.reader = PersonalStateReader(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying saves parameters and usually helps small LMs.
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, personal_state, targets=None):
        """
        input_ids: [B,T]
        personal_state: [B,state_dim]
        targets: [B,T] next-token targets or None
        """
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence too long")

        p = self.reader(personal_state)
        tok = self.token_embedding(input_ids)
        x = torch.cat([p, tok], dim=1)

        pos = torch.arange(x.shape[1], device=x.device)
        x = x + self.position_embedding(pos)[None, :, :]

        for block in self.blocks:
            x = block(x)
        x = self.final_norm(x)

        # Only text positions are language-model outputs.
        text_x = x[:, self.cfg.personal_tokens:, :]
        logits = self.lm_head(text_x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, personal_state, max_new_tokens=64, temperature=0.8):
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(x, personal_state)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids
