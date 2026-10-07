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


class CoreAuthorityHead(nn.Module):
    """CORE is input/content anchored and deliberately has no SELF input."""
    def __init__(self, content_dim: int = 96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, event, current_core, candidate_core):
        return self.net(torch.cat((
            event,
            current_core,
            candidate_core,
            current_core * candidate_core,
        ), dim=-1))


class GoalAuthorityHead(nn.Module):
    """GOAL depends on the relation between SELF and CORE, plus CORE↔GOAL."""
    def __init__(self, content_dim: int = 96, relation_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + relation_dim * 4, 160),
            nn.GELU(),
            nn.Linear(160, 80),
            nn.GELU(),
            nn.Linear(80, 1),
        )

    def forward(self, event, candidate_goal, self_core, core_goal):
        return self.net(torch.cat((
            event,
            candidate_goal,
            self_core,
            core_goal,
            self_core * core_goal,
        ), dim=-1))


class FocusAuthorityHead(nn.Module):
    """FOCUS is local and is referenced primarily to the current GOAL."""
    def __init__(self, content_dim: int = 96, relation_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + relation_dim * 3, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, event, candidate_focus, goal_focus):
        return self.net(torch.cat((
            event,
            candidate_focus,
            goal_focus,
            candidate_focus * goal_focus,
        ), dim=-1))


class DualReferenceCgfAuthority(nn.Module):
    """
    Two-reference topology:

        INPUT ---> CORE
                    |
        SELF <----> CORE  ---> GOAL ---> FOCUS

    SELF is not a parent of CORE. SELF and CORE are independent stable
    references whose relation constrains GOAL. CORE authority cannot see SELF
    by construction.
    """

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

        self.goal_candidate = nn.Linear(d_model, relation_dim, bias=False)
        self.focus_candidate = nn.Linear(d_model, relation_dim, bias=False)

        self.auth_core = CoreAuthorityHead(content_dim)
        self.auth_goal = GoalAuthorityHead(content_dim, relation_dim)
        self.auth_focus = FocusAuthorityHead(content_dim, relation_dim)

    @staticmethod
    def _norm(x):
        return F.layer_norm(x, (x.shape[-1],))

    def _candidate(self, layer, x):
        return torch.tanh(layer(self._norm(x)))

    def forward(
        self,
        root,
        core,
        goal,
        focus,
        event,
        cand_core,
        cand_goal,
        cand_focus,
        reference_root=None,
        reference_core=None,
        reference_goal=None,
        relation_mask=(1.0, 1.0, 1.0),
    ):
        c = self.content(core)
        g = self.content(goal)
        f = self.content(focus)
        e = self.content(event)
        cc = self.content(cand_core)

        proposal = torch.sigmoid(
            self.proposal(torch.cat((c, g, f, e), dim=-1))
        ).clamp(1e-5, 1 - 1e-5)

        rr = root if reference_root is None else reference_root
        rc = core if reference_core is None else reference_core
        rg = goal if reference_goal is None else reference_goal

        r_sc = self.self_core.initialize(rr, rc) * float(relation_mask[0])
        r_cg = self.core_goal.initialize(rc, rg) * float(relation_mask[1])
        r_gf = self.goal_focus.initialize(rg, focus) * float(relation_mask[2])

        cand_goal = self._candidate(self.goal_candidate, cand_goal)
        cand_focus = self._candidate(self.focus_candidate, cand_focus)

        # Crucial topology constraint: auth_core sees no root or relation state.
        a_core = self.auth_core(e, c, cc)
        a_goal = self.auth_goal(e, cand_goal, r_sc, r_cg)
        a_focus = self.auth_focus(e, cand_focus, r_gf)
        authority = torch.sigmoid(
            torch.cat((a_core, a_goal, a_focus), dim=-1)
        ).clamp(1e-5, 1 - 1e-5)

        local = proposal * authority
        p_core = local[:, 0:1]
        p_goal = 1.0 - (1.0 - p_core) * (1.0 - local[:, 1:2])
        p_focus = 1.0 - (1.0 - p_goal) * (1.0 - local[:, 2:3])
        final = torch.cat((p_core, p_goal, p_focus), dim=-1).clamp(1e-5, 1 - 1e-5)
        return proposal, authority, final
