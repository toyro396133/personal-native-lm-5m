from __future__ import annotations

import torch
import torch.nn as nn

from config import ModelConfig

# User-specific trainable state is deliberately much smaller than the main LM.
# It is decoded into the canonical 228-d PersonalState ABI.
CORE_Z = 32
POLICY_Z = 32
WORLD_Z = 32
ROUTING_Z = 16
CONF_Z = 4
PERSONAL_LATENT_DIM = CORE_Z + POLICY_Z + WORLD_Z + ROUTING_Z + CONF_Z  # 116


class PersonalStateDecoder(nn.Module):
    """Shared decoder: compact personal latent -> canonical 228-d state.

    The decoder is trained during system training. After the system is frozen,
    it can remain fixed while each user's tiny latent continues to learn.
    """

    def __init__(self):
        super().__init__()
        self.core = nn.Linear(CORE_Z, 64)
        self.policy = nn.Linear(POLICY_Z, 64)
        self.world = nn.Linear(WORLD_Z, 64)
        self.routing = nn.Linear(ROUTING_Z, 32)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.shape[-1] != PERSONAL_LATENT_DIM:
            raise ValueError(f"expected latent dim {PERSONAL_LATENT_DIM}, got {z.shape[-1]}")
        i = 0
        core_z = z[..., i:i+CORE_Z]; i += CORE_Z
        policy_z = z[..., i:i+POLICY_Z]; i += POLICY_Z
        world_z = z[..., i:i+WORLD_Z]; i += WORLD_Z
        routing_z = z[..., i:i+ROUTING_Z]; i += ROUTING_Z
        conf_z = z[..., i:i+CONF_Z]

        # Bounded canonical state keeps the ABI stable and portable.
        # Keep persistent state away from hard saturation so a long-lived user
        # model remains plastic and can revise an old belief/preference.
        scale = 0.8
        core = scale * torch.tanh(self.core(core_z))
        policy = scale * torch.tanh(self.policy(policy_z))
        world = scale * torch.tanh(self.world(world_z))
        routing = scale * torch.tanh(self.routing(routing_z))
        confidence = torch.sigmoid(conf_z)
        return torch.cat([core, policy, world, routing, confidence], dim=-1)


class PersonalHistoryEncoder(nn.Module):
    """Shared learner that turns a sequence of user interactions into an init latent.

    Shape:
      history_ids  [B, E, L]  (E events, L byte tokens per event)
      token_mask   [B, E, L]
      event_mask   [B, E]
      return       [B, 116]

    The encoder is shared/meta-learned. The returned latent can seed a persistent
    PersonalMicroModel for a particular user.
    """

    def __init__(self, cfg: ModelConfig, token_dim: int = 48, hidden_dim: int = 96):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, token_dim)
        self.event_proj = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim),
            nn.GELU(),
        )
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.to_latent = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, PERSONAL_LATENT_DIM),
        )

    def forward(self, history_ids, token_mask=None, event_mask=None):
        B, E, L = history_ids.shape
        emb = self.embedding(history_ids)  # [B,E,L,D]
        if token_mask is None:
            pooled = emb.mean(dim=2)
        else:
            w = token_mask.to(emb.dtype).unsqueeze(-1)
            pooled = (emb * w).sum(dim=2) / w.sum(dim=2).clamp_min(1.0)
        events = self.event_proj(pooled)

        if event_mask is not None:
            events = events * event_mask.to(events.dtype).unsqueeze(-1)
        out, _ = self.gru(events)

        if event_mask is None:
            last = out[:, -1]
        else:
            lengths = event_mask.long().sum(dim=1).clamp_min(1) - 1
            last = out[torch.arange(B, device=out.device), lengths]
        return self.to_latent(last)


class PersonalLearningSystem(nn.Module):
    """Global/shared part of the personal model stack."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.history_encoder = PersonalHistoryEncoder(cfg)
        self.state_decoder = PersonalStateDecoder()

    def from_history(self, history_ids, token_mask=None, event_mask=None):
        z = self.history_encoder(history_ids, token_mask, event_mask)
        state = self.state_decoder(z)
        return z, state


class PersonalMicroModel(nn.Module):
    """Persistent per-user trainable parameters.

    In v0.4 the user-specific model is 116 trainable scalars. It can be
    initialized by PersonalHistoryEncoder and then updated for that user while
    the main LM and shared personal-learning machinery stay frozen.
    """

    def __init__(self, initial_latent: torch.Tensor | None = None):
        super().__init__()
        if initial_latent is None:
            initial_latent = torch.zeros(PERSONAL_LATENT_DIM)
        initial_latent = initial_latent.detach().clone().reshape(PERSONAL_LATENT_DIM)
        self.latent = nn.Parameter(initial_latent)

    def forward(self, decoder: PersonalStateDecoder, batch_size: int = 1):
        state = decoder(self.latent.unsqueeze(0))
        if batch_size > 1:
            state = state.expand(batch_size, -1)
        return state

    @staticmethod
    def slot_gradient_mask(slot: str, device=None):
        """Mask used for routed online learning of one personal slot."""
        mask = torch.zeros(PERSONAL_LATENT_DIM, device=device)
        if slot == "core":
            mask[:CORE_Z] = 1
            mask[-CONF_Z] = 1
        elif slot == "policy":
            mask[CORE_Z:CORE_Z + POLICY_Z] = 1
            mask[-CONF_Z + 1] = 1
        elif slot == "world":
            a = CORE_Z + POLICY_Z
            mask[a:a + WORLD_Z] = 1
            mask[-CONF_Z + 2] = 1
        elif slot == "routing":
            a = CORE_Z + POLICY_Z + WORLD_Z
            mask[a:a + ROUTING_Z] = 1
            mask[-1] = 1
        else:
            raise ValueError(f"unknown personal slot: {slot}")
        return mask
