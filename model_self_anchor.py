from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import PersonalNativeLM


class SelfReferenceAdapter(nn.Module):
    """Small residual adapter that represents each hidden state relative to one
    shared, persistent model identity anchor.

    The final projection is initialized to zero, so adding this adapter changes
    *nothing* at initialization. Training must earn any use of SELF.
    """

    def __init__(self, d_model: int, rank: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.down = nn.Linear(4 * d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))

        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor, self_anchor: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        an = F.layer_norm(self_anchor, (self_anchor.shape[-1],))
        a = an.view(1, 1, -1).expand(xn.shape[0], xn.shape[1], -1)
        relative = torch.cat((xn, a, xn - a, xn * a), dim=-1)
        delta = self.up(F.gelu(self.down(relative)))
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * delta

    def diagnostics(self) -> dict:
        return {
            "gate": float(torch.sigmoid(self.gate_logit.detach()).cpu()),
            "up_norm": float(self.up.weight.detach().norm().cpu()),
            "down_norm": float(self.down.weight.detach().norm().cpu()),
        }


class SelfAnchoredPersonalNativeLM(nn.Module):
    """The existing LM plus one shared identity anchor visible at every layer.

    The ordinary language model is left intact inside `base`. After each
    Transformer block, a tiny adapter may express the current representation
    relative to the same shared SELF anchor.

    SELF is deliberately not a user identity or a memory store. It is a stable
    computational reference point for the general model.
    """

    def __init__(self, base: PersonalNativeLM, rank: int = 8):
        super().__init__()
        self.base = base
        self.cfg = base.cfg
        self.rank = rank
        self.self_anchor = nn.Parameter(torch.empty(self.cfg.d_model))
        nn.init.normal_(self.self_anchor, mean=0.0, std=0.02)
        self.self_adapters = nn.ModuleList(
            [SelfReferenceAdapter(self.cfg.d_model, rank=rank) for _ in range(self.cfg.n_layers)]
        )

    @property
    def token_embedding(self):
        return self.base.token_embedding

    @property
    def controller(self):
        return self.base.controller

    @property
    def blocks(self):
        return self.base.blocks

    def forward(
        self,
        input_ids,
        personal_state,
        targets=None,
        condition_ids=None,
        condition_mask=None,
    ):
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence too long")

        conditioning = None
        if condition_ids is not None:
            q_emb = self.base.token_embedding(condition_ids)
            conditioning = self.base.controller(personal_state, q_emb, condition_mask)

        x = self.base.token_embedding(input_ids)
        pos = torch.arange(T, device=x.device)
        x = x + self.base.position_embedding(pos)[None, :, :]

        for block, adapter in zip(self.base.blocks, self.self_adapters):
            x = block(x, conditioning)
            x = adapter(x, self.self_anchor)

        x = self.base.final_norm(x)
        logits = self.base.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        personal_state,
        condition_ids=None,
        condition_mask=None,
        max_new_tokens=64,
        temperature=0.8,
    ):
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(
                x,
                personal_state,
                condition_ids=condition_ids,
                condition_mask=condition_mask,
            )
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids

    def self_diagnostics(self) -> dict:
        return {
            "anchor_norm": float(self.self_anchor.detach().norm().cpu()),
            "layers": [a.diagnostics() for a in self.self_adapters],
        }
