from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class ContextualPersonalController(nn.Module):
    """Build request-specific conditioning from a persistent PersonalState.

    The controller contains an explicit query-dependent router over the four
    canonical personal slots (core, policy, world, routing) plus a global
    relevance gate. This makes "ignore personal data for this request" a first-
    class operation rather than something the LM must discover accidentally.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c = cfg.controller_dim
        self.state_encoder = nn.Sequential(
            nn.LayerNorm(cfg.state_dim),
            nn.Linear(cfg.state_dim, c),
            nn.GELU(),
        )
        self.query_encoder = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, c),
            nn.GELU(),
        )
        self.slot_gate = nn.Linear(c, 4)
        self.relevance_gate = nn.Linear(c, 1)
        self.fuse = nn.Sequential(
            nn.Linear(2 * c, c),
            nn.GELU(),
            nn.LayerNorm(c),
        )

        # Backward-compatible starting point for v0.3 checkpoints: before the
        # router learns, almost all personal state is passed through.
        nn.init.zeros_(self.slot_gate.weight)
        nn.init.constant_(self.slot_gate.bias, 4.0)
        nn.init.zeros_(self.relevance_gate.weight)
        nn.init.constant_(self.relevance_gate.bias, 4.0)

    def _query_summary(self, query_embeddings, query_mask=None):
        if query_mask is None:
            pooled = query_embeddings.mean(dim=1)
        else:
            w = query_mask.to(query_embeddings.dtype).unsqueeze(-1)
            pooled = (query_embeddings * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)
        return self.query_encoder(pooled)

    def routing(self, query_embeddings, query_mask=None):
        q = self._query_summary(query_embeddings, query_mask)
        # Five-way competitive routing: core / policy / world / routing / none.
        # `relevance_gate` is reused as an inverse none-logit so v0.3/v0.4
        # checkpoints remain loadable.
        slot_logits = self.slot_gate(q)
        none_logit = -self.relevance_gate(q)
        route_probs = torch.softmax(torch.cat([slot_logits, none_logit], dim=-1), dim=-1)
        slot_gates = route_probs[:, :4]
        relevance = 1.0 - route_probs[:, 4:5]
        return q, slot_gates, relevance

    @staticmethod
    def _apply_slot_gates(personal_state, gates):
        core = personal_state[:, :64] * gates[:, 0:1]
        policy = personal_state[:, 64:128] * gates[:, 1:2]
        world = personal_state[:, 128:192] * gates[:, 2:3]
        routing = personal_state[:, 192:224] * gates[:, 3:4]
        confidence = personal_state[:, 224:228] * gates
        return torch.cat([core, policy, world, routing, confidence], dim=-1)

    def forward(self, personal_state, query_embeddings, query_mask=None,
                return_routing=False):
        q, slot_gates, relevance = self.routing(query_embeddings, query_mask)
        routed_state = self._apply_slot_gates(personal_state, slot_gates)
        s = self.state_encoder(routed_state)
        conditioning = self.fuse(torch.cat([s, q], dim=-1)) * relevance
        if return_routing:
            return conditioning, slot_gates, relevance
        return conditioning


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

        # Temporary activation modulation. These are shared global parameters;
        # the per-user value comes from the runtime conditioning code.
        self.cond_delta = nn.Linear(cfg.controller_dim, cfg.d_model, bias=False)
        self.cond_gate = nn.Linear(cfg.controller_dim, 1)

    def forward(self, x, conditioning=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        if conditioning is not None:
            delta = torch.tanh(self.cond_delta(conditioning)).unsqueeze(1)
            gate = torch.sigmoid(self.cond_gate(conditioning)).view(-1, 1, 1)
            x = x + gate * delta
        return x


class PersonalNativeLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.controller = ContextualPersonalController(cfg)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, personal_state, targets=None,
                condition_ids=None, condition_mask=None):
        """Autoregressive forward pass with optional request-specific conditioning.

        `condition_ids` must contain only information known before the answer
        is generated (normally the user's request). This keeps training causal.
        """
        B, T = input_ids.shape
        if T > self.cfg.max_seq_len:
            raise ValueError("Sequence too long")

        conditioning = None
        if condition_ids is not None:
            q_emb = self.token_embedding(condition_ids)
            conditioning = self.controller(personal_state, q_emb, condition_mask)

        x = self.token_embedding(input_ids)
        pos = torch.arange(T, device=x.device)
        x = x + self.position_embedding(pos)[None, :, :]

        for block in self.blocks:
            x = block(x, conditioning)
        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, personal_state, condition_ids=None,
                 condition_mask=None, max_new_tokens=64, temperature=0.8):
        # Conditioning is recomputed from persistent state + current query, but
        # it remains ephemeral; no LM parameter is modified by generation.
        for _ in range(max_new_tokens):
            x = input_ids[:, -self.cfg.max_seq_len:]
            logits, _ = self(
                x, personal_state,
                condition_ids=condition_ids,
                condition_mask=condition_mask,
            )
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, nxt], dim=1)
        return input_ids
