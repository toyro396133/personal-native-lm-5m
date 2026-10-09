"""Parameter-tied depth experiment; baseline-compatible at loops=1."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from model import PersonalNativeLM


class LoopedPersonalNativeLM(PersonalNativeLM):
    """Repeat the *same* Transformer blocks, without creating extra weights.

    loops=1 is numerically identical to the base model given equal weights.
    Each pass visits every physical block in order. Reusing the weights is
    deliberate: effective depth grows while parameter count stays fixed.
    """

    def __init__(self, cfg, loops: int = 1, input_injection: float = 0.0):
        if loops < 1:
            raise ValueError("loops must be >= 1")
        if not 0.0 <= input_injection <= 1.0:
            raise ValueError("input_injection must be in [0, 1]")
        super().__init__(cfg)
        self.loops = loops
        self.input_injection = input_injection

    def forward(self, input_ids, personal_state, targets=None,
                condition_ids=None, condition_mask=None):
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("Sequence too long")
        conditioning = None
        if condition_ids is not None:
            q_emb = self.token_embedding(condition_ids)
            conditioning = self.controller(personal_state, q_emb, condition_mask)
        pos = torch.arange(t, device=input_ids.device)
        input_x = self.token_embedding(input_ids)
        input_x = input_x + self.position_embedding(pos)[None, :, :]
        x = input_x
        for iteration in range(self.loops):
            if iteration and self.input_injection:
                # Parameter-free convex injection, after each completed pass.
                x = (1.0 - self.input_injection) * x + self.input_injection * input_x
            for block in self.blocks:
                x = block(x, conditioning)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss
