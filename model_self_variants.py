from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


SELF_VARIANTS = {
    "self_v1",
    "self_v1_slow",
    "diff_only",
    "diff_anchor",
    "projected_diff",
    "projected_diff_slow",
}


def anchor_lr_scale_for_variant(variant: str) -> float:
    return 0.1 if variant.endswith("_slow") else 1.0


def canonical_architecture(variant: str) -> str:
    if variant in {"self_v1", "self_v1_slow"}:
        return "v1"
    if variant == "diff_only":
        return "diff_only"
    if variant == "diff_anchor":
        return "diff_anchor"
    if variant in {"projected_diff", "projected_diff_slow"}:
        return "projected_diff"
    raise ValueError(variant)


class SelfVariantAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int, architecture: str, projection_rank: int = 4):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.architecture = architecture
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)

        if architecture == "v1":
            in_dim = 4 * d_model
        elif architecture == "diff_only":
            in_dim = d_model
        elif architecture == "diff_anchor":
            in_dim = 2 * d_model
        elif architecture == "projected_diff":
            in_dim = d_model
            self.anchor_down = nn.Linear(d_model, projection_rank, bias=False)
            self.anchor_up = nn.Linear(projection_rank, d_model, bias=False)
            nn.init.normal_(self.anchor_down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(self.anchor_up.weight)
        else:
            raise ValueError(architecture)

        self.down = nn.Linear(in_dim, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def _anchor_for_layer(self, self_anchor: torch.Tensor) -> torch.Tensor:
        an = F.layer_norm(self_anchor, (self_anchor.shape[-1],))
        if self.architecture == "projected_diff":
            projected = an + self.anchor_up(F.gelu(self.anchor_down(an)))
            an = F.layer_norm(projected, (projected.shape[-1],))
        return an

    def forward(self, x: torch.Tensor, self_anchor: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        an = self._anchor_for_layer(self_anchor)
        a = an.view(1, 1, -1).expand(xn.shape[0], xn.shape[1], -1)

        if self.architecture == "v1":
            relative = torch.cat((xn, a, xn - a, xn * a), dim=-1)
        elif self.architecture == "diff_only":
            relative = xn - a
        elif self.architecture == "diff_anchor":
            relative = torch.cat((a, xn - a), dim=-1)
        elif self.architecture == "projected_diff":
            relative = xn - a
        else:
            raise RuntimeError(self.architecture)

        delta = self.up(F.gelu(self.down(relative)))
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * delta

    def diagnostics(self) -> dict:
        result = {
            "gate": float(torch.sigmoid(self.gate_logit.detach()).cpu()),
            "up_norm": float(self.up.weight.detach().norm().cpu()),
            "down_norm": float(self.down.weight.detach().norm().cpu()),
        }
        if self.architecture == "projected_diff":
            result.update({
                "anchor_projection_down_norm": float(self.anchor_down.weight.detach().norm().cpu()),
                "anchor_projection_up_norm": float(self.anchor_up.weight.detach().norm().cpu()),
            })
        return result


class SelfVariantPersonalNativeLM(nn.Module):
    def __init__(self, base, variant: str, rank: int = 8, projection_rank: int = 4):
        super().__init__()
        if variant not in SELF_VARIANTS:
            raise ValueError(variant)
        self.base = base
        self.cfg = base.cfg
        self.variant = variant
        self.architecture = canonical_architecture(variant)
        self.rank = rank
        self.projection_rank = projection_rank

        self.self_anchor = nn.Parameter(torch.empty(self.cfg.d_model))
        nn.init.normal_(self.self_anchor, mean=0.0, std=0.02)

        self.self_adapters = nn.ModuleList([
            SelfVariantAdapter(
                self.cfg.d_model,
                rank=rank,
                architecture=self.architecture,
                projection_rank=projection_rank,
            )
            for _ in range(self.cfg.n_layers)
        ])

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
        _, T = input_ids.shape
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
            "variant": self.variant,
            "architecture": self.architecture,
            "anchor_norm": float(self.self_anchor.detach().norm().cpu()),
            "layers": [a.diagnostics() for a in self.self_adapters],
        }
