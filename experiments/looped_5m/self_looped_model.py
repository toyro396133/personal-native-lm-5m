"""Experimental SELF-anchor interaction with tied-depth Transformer.

This is a *new, deliberately isolated proxy* for an iterative SELF anchor,
not a reproduction of the earlier diff_anchor / SELF-v2 implementation.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from looped_model import LoopedPersonalNativeLM


class SelfLoopedLM(LoopedPersonalNativeLM):
    def __init__(self, cfg, loops=1, self_mode="anchor", input_injection=0.0):
        super().__init__(cfg, loops=loops, input_injection=input_injection)
        if self_mode not in ("anchor", "capacity"):
            raise ValueError("self_mode must be 'anchor' or 'capacity'")
        self.self_mode = self_mode
        # Both modes have exactly the same trainable parameter budget.
        self.self_vector = nn.Parameter(torch.randn(cfg.d_model) * 0.02)
        self.anchor_intervention = "normal"

    def _self_delta(self, x):
        a = self.self_vector
        if self.anchor_intervention == "zero":
            a = torch.zeros_like(a)
        elif self.anchor_intervention == "shuffle":
            a = torch.roll(a, a.numel() // 2)
        elif self.anchor_intervention != "normal":
            raise ValueError("unknown anchor intervention")
        if self.self_mode == "anchor":
            # A reference direction that interacts with *current* state.
            # Centering prevents unconstrained mean-shift as a shortcut.
            a = F.layer_norm(a, (a.numel(),))
            delta = torch.tanh(F.layer_norm(x, (x.size(-1),)) - a)
        else:
            # Matched-capacity constant additive conditioning, no state comparison.
            delta = torch.tanh(a).view(1, 1, -1).expand_as(x)
        return delta * 0.02

    def forward(self, input_ids, personal_state, targets=None, condition_ids=None, condition_mask=None):
        _, t = input_ids.shape
        if t > self.cfg.max_seq_len:
            raise ValueError("Sequence too long")
        conditioning = None
        if condition_ids is not None:
            q_emb = self.token_embedding(condition_ids)
            conditioning = self.controller(personal_state, q_emb, condition_mask)
        pos = torch.arange(t, device=input_ids.device)
        input_x = self.token_embedding(input_ids) + self.position_embedding(pos)[None, :, :]
        x = input_x
        for iteration in range(self.loops):
            if iteration and self.input_injection:
                x = (1 - self.input_injection) * x + self.input_injection * input_x
            for block in self.blocks:
                x = block(x, conditioning)
            # Deliberately apply AFTER each complete shared-weight stack.
            x = x + self._self_delta(x)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), ignore_index=-100)
        return logits, loss
