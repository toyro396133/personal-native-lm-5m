"""COMPASS V12 — independent, detachable neural state coupled to a FROZEN LLM.

Uses the V9.0 live final-decoder-block hook shape, and a trainable per-user
and per-project state (not fake static owner bits). Does NOT read/write SELF.
No supervision invented from text: only source-linked direct user feedback.
No automatic authorization or promotion of project policy.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

STATE_SCHEMA = "COMPASS_V12_DETACHABLE_V1"
ID = re.compile(r"^[a-z][a-z0-9._-]{0,63}$")
GIT_SOURCE = re.compile(r"^git:[0-9a-f]{40}:docs/[a-zA-Z0-9_./-]+[.]md#L[1-9][0-9]*-L[1-9][0-9]*$")
CHAT_SOURCE = re.compile(r"^chat:[a-z][a-z0-9._-]{7,99}:[a-z][a-z0-9._-]{7,99}$")


def check_id(value: str) -> str:
    if not isinstance(value, str) or ID.fullmatch(value) is None:
        raise ValueError("User/project ID must be a nonempty safe stable identifier")
    return value


def check_source(source: str) -> str:
    if not isinstance(source, str) or not (GIT_SOURCE.fullmatch(source) or CHAT_SOURCE.fullmatch(source)):
        raise ValueError("Requires pinned local chat or Git evidence pointer")
    if source.startswith("git:"):
        if ".." in source.split(":docs/",1)[1].split("#L",1)[0].split("/"):
            raise ValueError("Source path traversal")
        lines = source.rsplit("#L",1)[1].split("-L")
        if int(lines[1]) < int(lines[0]):
            raise ValueError("Invalid source line range")
    return source


class FrozenResidualBridge(nn.Module):
    """V9.0-shaped 2-matrix neural layer; NO target-LLM weights altered.

    V9.0 owned a shared scalar->rank projection. Here that signal is replaced
    with separately learned user and project rank-dimensional neural states.
    The historical down/up weights may optionally be loaded, after validating
    dimensions, but historical V8.5 ranker weights are not compatible.
    """
    def __init__(self, hidden_size: int, rank: int = 16, gain: float = 1.0):
        super().__init__()
        if hidden_size < 2 or rank < 1 or gain <= 0:
            raise ValueError("Invalid bridge dimensions/gain")
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.normal_(self.up.weight, std=0.02)  # nonzero so latent has gradient
        self.hidden_size, self.rank, self.gain = hidden_size, rank, gain
        self.requires_grad_(False)
        self.eval()

    def forward(self, hidden: torch.Tensor, user_state: torch.Tensor,
                project_state: torch.Tensor) -> torch.Tensor:
        if hidden.shape[-1] != self.hidden_size:
            raise ValueError("Bridge/LM hidden sizes do not match")
        if tuple(user_state.shape) != (self.rank,) or tuple(project_state.shape) != (self.rank,):
            raise ValueError("Latent dimensions do not match bridge rank")
        normalized = F.layer_norm(hidden, (hidden.shape[-1],))
        u = user_state.to(device=hidden.device, dtype=hidden.dtype)
        p = project_state.to(device=hidden.device, dtype=hidden.dtype)
        correction = self.up(torch.tanh(self.down(normalized) + u + p))
        return hidden + self.gain * correction

    def load_v90(self, weights_path: Path) -> None:
        """Optional historical V9.0 safetensors bridge reuse.

        V9.0: down.weight, owner.weight, up.weight; owner scalar is NOT a
        trained per-user state and is intentionally not imported. Use the
        user's private, trusted existing checkpoint only; no auto download.
        """
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("Install safetensors to load V9.0 checkpoint") from exc
        tensors = load_file(str(weights_path), device="cpu")
        for name, parameter in (("down.weight", self.down.weight),
                                ("up.weight", self.up.weight)):
            if name not in tensors or tuple(tensors[name].shape) != tuple(parameter.shape):
                raise ValueError("V9.0 checkpoint shape mismatch: " + name)
        with torch.no_grad():
            self.down.weight.copy_(tensors["down.weight"])
            self.up.weight.copy_(tensors["up.weight"])
        self.requires_grad_(False)


class PersonalProjectState:
    """Separate personal/project state, exported as portable JSON (NO pickle)."""
    def __init__(self, rank: int):
        if rank < 1 or rank > 1024:
            raise ValueError("Invalid state dimension")
        self.rank = rank
        self.users: dict[str, list[float]] = {}
        self.projects: dict[str, dict[str, list[float]]] = {}
        self.events: list[dict] = []
        self.event_ids: set[str] = set()

    def latent(self, user_id: str, project_id: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        check_id(user_id)
        if project_id is not None:
            check_id(project_id)
        zero = [0.0] * self.rank
        personal = torch.tensor(self.users.get(user_id, zero), dtype=torch.float32)
        project = torch.tensor(self.projects.get(user_id, {}).get(project_id, zero), dtype=torch.float32)
        return personal, project

    def update(self, user_id: str, project_id: str | None, scope: str,
               values: torch.Tensor, *, event_id: str, source: str) -> None:
        check_id(user_id)
        check_id(event_id)
        check_source(source)
        if scope not in ("personal", "project"):
            raise ValueError("Unexpected feedback scope")
        if scope == "project":
            check_id(project_id)
        elif project_id is not None:
            raise ValueError("Personal updates must not carry project ownership")
        key = f"{scope}:{user_id}:{project_id}:{event_id}"
        if key in self.event_ids:
            raise ValueError("Duplicate original feedback event")
        if values.shape != (self.rank,) or not bool(torch.isfinite(values).all()):
            raise ValueError("Personal state must be finite and correctly sized")
        v = [float(x) for x in values.detach().cpu()]
        if scope == "personal":
            self.users[user_id] = v
        else:
            self.projects.setdefault(user_id, {})[project_id] = v
        self.events.append({"key": key, "source": source,
                            "scope": scope, "user_id": user_id, "project_id": project_id,
                            "approved_project_policy": False})
        self.event_ids.add(key)

    def to_data(self) -> dict:
        return {"schema": STATE_SCHEMA, "rank": self.rank,
                "users": self.users, "projects": self.projects, "events": self.events,
                "no_auto_policy_approval": True, "model_family": "MODEL_NEUTRAL"}

    @classmethod
    def from_data(cls, data: dict) -> "PersonalProjectState":
        if (not isinstance(data, dict) or data.get("schema") != STATE_SCHEMA or
            data.get("no_auto_policy_approval") is not True or
            data.get("model_family") != "MODEL_NEUTRAL"):
            raise ValueError("Unknown, corrupt or unsafe COMPASS state")
        result = cls(data["rank"])
        users, projects, events = data.get("users"), data.get("projects"), data.get("events")
        if not isinstance(users, dict) or not isinstance(projects, dict) or not isinstance(events,list):
            raise ValueError("Corrupt COMPASS state shapes")
        for uid, vals in users.items():
            check_id(uid)
            if not isinstance(vals, list) or len(vals) != result.rank or any(
                    type(x) not in (int,float) or not math.isfinite(x) for x in vals):
                raise ValueError("Invalid personal weights")
        for uid, records in projects.items():
            check_id(uid)
            if not isinstance(records, dict):
                raise ValueError("Invalid project state records")
            for pid, vals in records.items():
                check_id(pid)
                if not isinstance(vals, list) or len(vals) != result.rank or any(
                        type(x) not in (int,float) or not math.isfinite(x) for x in vals):
                    raise ValueError("Invalid project weights")
        result.users, result.projects = users, projects
        for event in events:
            if not isinstance(event, dict) or event.get("approved_project_policy") is not False:
                raise ValueError("Unapproved event claimed authority")
            key = event.get("key")
            check_source(event.get("source"))
            if not isinstance(key,str) or key in result.event_ids:
                raise ValueError("Duplicate or invalid event journal")
            result.event_ids.add(key)
        result.events = events
        return result

    def save(self, path: Path) -> None:
        """Atomic (not encrypted) personal memory checkpoint: keep PRIVATE."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf8", dir=path.parent,
                                         prefix=".compass-", suffix=".tmp", delete=False) as f:
            tmp = Path(f.name)
            json.dump(self.to_data(), f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: Path) -> "PersonalProjectState":
        return cls.from_data(json.loads(Path(path).read_text(encoding="utf8")))


def freeze_backbone(model: nn.Module) -> None:
    model.requires_grad_(False)
    model.eval()


@contextlib.contextmanager
def active_compass(model: nn.Module, bridge: FrozenResidualBridge,
                   user_state: torch.Tensor, project_state: torch.Tensor) -> Iterator[None]:
    """Genuine autograd-preserving forward hook at an LLM decoder block output."""
    if user_state.shape != (bridge.rank,) or project_state.shape != (bridge.rank,):
        raise ValueError("Rank mismatch")
    layers = model.model.layers
    if not len(layers):
        raise ValueError("Model does not expose decoder layers")
    def hook(_module, _inputs, out):
        hidden = out[0] if isinstance(out, (tuple,list)) else out
        if not isinstance(hidden,torch.Tensor) or hidden.ndim != 3:
            raise ValueError("Unexpected transformer decoder output")
        adapted = bridge(hidden[:, -1, :], user_state, project_state).unsqueeze(1)
        changed = adapted if hidden.shape[1] == 1 else torch.cat(
            (hidden[:, :-1, :], adapted), dim=1
        )
        if isinstance(out,tuple):
            return (changed, *out[1:])
        if isinstance(out,list):
            return [changed,*out[1:]]
        return changed
    handle = layers[-1].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def _logits(model: nn.Module, token_ids: torch.Tensor) -> torch.Tensor:
    result = model(input_ids=token_ids)
    logits = result.logits if hasattr(result, "logits") else result
    if logits.ndim != 3 or logits.shape[0] != 1:
        raise ValueError("Expected one token sequence and last-token logits")
    return logits[0,-1,:]


def pair_margins(model: nn.Module, bridge: FrozenResidualBridge,
                 user: torch.Tensor, project: torch.Tensor,
                 candidates: list[torch.Tensor],
                 yes_token_id: int, no_token_id: int) -> torch.Tensor:
    if len(candidates) != 2 or any(x.ndim != 2 or x.shape[0] != 1 for x in candidates):
        raise ValueError("Exactly two separately tokenized model prompts are required")
    if type(yes_token_id) is not int or type(no_token_id) is not int or yes_token_id == no_token_id:
        raise ValueError("Valid distinct Yes/No token IDs are required")
    with active_compass(model, bridge, user, project):
        return torch.stack([
            _logits(model, ids)[yes_token_id] - _logits(model, ids)[no_token_id]
            for ids in candidates
        ])


def personalized_update(model: nn.Module, bridge: FrozenResidualBridge,
                        bank: PersonalProjectState, *, user_id: str,
                        project_id: str | None, scope: str,
                        candidate_ids: list[torch.Tensor], chosen_index: int,
                        yes_token_id: int, no_token_id: int,
                        feedback: dict, steps: int = 8, lr: float = 0.05,
                        retention: float = 0.001) -> dict:
    """Train ONLY selected detachable latent, never update LLM/bridge weights.

    A label is accepted ONLY as a direct user confirmation; source_verified
    and confirmed_by_user are unverified local assertions, not authentication.
    For real work use authenticated review/authority upstream.
    """
    if (not isinstance(feedback,dict) or feedback.get("speaker") != "user" or
        feedback.get("confirmed_by_user") is not True or
        feedback.get("source_verified") is not True):
        raise ValueError("Cannot learn from unverified/assistant-derived feedback")
    if type(chosen_index) is not int or chosen_index not in (0,1):
        raise ValueError("Choice must be 0 or 1 from *previously presented* options")
    if scope not in ("personal","project"):
        raise ValueError("Unknown learning scope")
    if scope == "project":
        check_id(project_id)
    elif project_id is not None:
        raise ValueError("Project feedback cannot silently enter personal state")
    check_id(user_id)
    event_id=check_id(feedback.get("event_id"))
    source=check_source(feedback.get("source"))
    key = f"{scope}:{user_id}:{project_id}:{event_id}"
    if key in bank.event_ids:
        raise ValueError("Feedback already used for training")
    if bank.rank != bridge.rank or steps < 1 or steps > 200 or lr <= 0 or retention < 0:
        raise ValueError("Unsupported online learning hyperparameters")
    freeze_backbone(model)
    bridge.requires_grad_(False)
    bridge.eval()
    user, project = bank.latent(user_id,project_id)
    existing = project if scope=="project" else user
    trainable = nn.Parameter(existing.clone())
    opt = torch.optim.SGD([trainable],lr=lr)
    log=[]
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        chosen_user = trainable if scope=="personal" else user
        chosen_project = trainable if scope=="project" else project
        margin = pair_margins(model,bridge,chosen_user,chosen_project,candidate_ids,
                              yes_token_id,no_token_id)
        direction = margin[chosen_index]-margin[1-chosen_index]
        loss = F.softplus(-direction)+retention*(trainable-existing).square().sum()
        loss.backward()
        if trainable.grad is None or not bool(torch.isfinite(trainable.grad).all()):
            raise RuntimeError("No finite COMPASS state gradient (adapter may be disconnected)")
        opt.step()
        if not bool(torch.isfinite(trainable).all()):
            raise RuntimeError("Personal state diverged")
        log.append({"loss":round(float(loss.detach()),7),
                    "preference_margin":round(float(direction.detach()),7)})
    bank.update(user_id,project_id,scope,trainable.detach(),event_id=event_id,source=source)
    return {"status":"USER_STATE_UPDATED_ONLY","steps":steps,"scope":scope,
            "loss_first":log[0]["loss"],"loss_last":log[-1]["loss"],
            "margin_before":log[0]["preference_margin"],
            "margin_last_step_preupdate":log[-1]["preference_margin"],
            "state_delta_l2":float((trainable.detach()-existing).norm()),
            "source":source,
            "backbone_updated":False,"bridge_updated":False,"project_policy_approved":False}


def inspect_live(model: nn.Module, bridge: FrozenResidualBridge,
                 bank: PersonalProjectState, user_id: str, project_id: str | None,
                 candidate_ids: list[torch.Tensor], yes: int, no: int) -> dict:
    u,p=bank.latent(user_id,project_id)
    with torch.no_grad():
        before = torch.stack([
            _logits(model, ids)[yes]-_logits(model, ids)[no] for ids in candidate_ids])
        after = pair_margins(model,bridge,u,p,candidate_ids,yes,no)
    return {"frozen_no_compass":before.detach().tolist(),
            "with_compass":after.detach().tolist(),
            "margins_changed":bool((before-after).abs().max() > 1e-7)}
