from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from core_goal_focus_hierarchical_lab import RelationState


class ContentProjector(nn.Module):
    def __init__(self, d_model: int, hidden: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(F.layer_norm(x, (x.shape[-1],)))


class AuthorityHead(nn.Module):
    def __init__(self, content_dim: int = 96, relation_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + relation_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, event, candidate, current_relation, candidate_relation):
        return self.net(torch.cat((
            event,
            candidate,
            current_relation,
            candidate_relation,
            current_relation * candidate_relation,
        ), dim=-1))


class JointCgfAuthority(nn.Module):
    """Rooted SELF -> CORE -> GOAL -> FOCUS authority module."""

    def __init__(self, d_model: int, content_dim: int = 96, relation_dim: int = 64):
        super().__init__()
        self.content = ContentProjector(d_model, content_dim)
        self.proposal = nn.Sequential(
            nn.Linear(content_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        self.self_core = RelationState(d_model, content_dim, relation_dim)
        self.core_goal = RelationState(d_model, content_dim, relation_dim)
        self.goal_focus = RelationState(d_model, content_dim, relation_dim)
        self.core_candidate = nn.Linear(d_model, relation_dim, bias=False)
        self.goal_candidate = nn.Linear(d_model, relation_dim, bias=False)
        self.focus_candidate = nn.Linear(d_model, relation_dim, bias=False)
        self.auth_core = AuthorityHead(content_dim, relation_dim)
        self.auth_goal = AuthorityHead(content_dim, relation_dim)
        self.auth_focus = AuthorityHead(content_dim, relation_dim)

    @staticmethod
    def _norm(x):
        return F.layer_norm(x, (x.shape[-1],))

    def _candidate(self, layer, x):
        return torch.tanh(layer(self._norm(x)))

    def hierarchy(self, root, core, goal, focus):
        return (
            self.self_core.initialize(root, core),
            self.core_goal.initialize(core, goal),
            self.goal_focus.initialize(goal, focus),
        )

    def forward(
        self,
        root,
        core, goal, focus, event,
        cand_core, cand_goal, cand_focus,
        authority_hierarchy=None,
    ):
        c = self.content(core)
        g = self.content(goal)
        f = self.content(focus)
        e = self.content(event)
        proposal = torch.sigmoid(
            self.proposal(torch.cat((c, g, f, e), dim=-1))
        ).clamp(1e-5, 1 - 1e-5)

        if authority_hierarchy is None:
            ac, ag, af = core, goal, focus
        else:
            ac, ag, af = authority_hierarchy

        current_rel = self.hierarchy(root, ac, ag, af)
        candidate_rel = self.hierarchy(root, cand_core, cand_goal, cand_focus)
        candidate = (
            self._candidate(self.core_candidate, cand_core),
            self._candidate(self.goal_candidate, cand_goal),
            self._candidate(self.focus_candidate, cand_focus),
        )

        authority = torch.sigmoid(torch.cat((
            self.auth_core(e, candidate[0], current_rel[0], candidate_rel[0]),
            self.auth_goal(e, candidate[1], current_rel[1], candidate_rel[1]),
            self.auth_focus(e, candidate[2], current_rel[2], candidate_rel[2]),
        ), dim=-1)).clamp(1e-5, 1 - 1e-5)

        local = proposal * authority
        p_core = local[:, 0:1]
        p_goal = 1.0 - (1.0 - p_core) * (1.0 - local[:, 1:2])
        p_focus = 1.0 - (1.0 - p_goal) * (1.0 - local[:, 2:3])
        final = torch.cat((p_core, p_goal, p_focus), dim=-1).clamp(1e-5, 1 - 1e-5)
        return proposal, authority, final


def proposal_target(kind: str, target, device):
    if kind == "goal_pressure":
        x = [0.0, 1.0, 1.0]
    elif kind == "core_pressure":
        x = [1.0, 1.0, 1.0]
    else:
        x = list(target)
    return torch.tensor(x, dtype=torch.float32, device=device)
