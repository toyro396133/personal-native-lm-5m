from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import json
import torch

CORE_DIM = 64
POLICY_DIM = 64
WORLD_DIM = 64
ROUTING_DIM = 32

@dataclass
class StateSlot:
    vector: List[float]
    confidence: float = 0.0
    evidence_count: int = 0

    def validate(self, expected_dim: int):
        if len(self.vector) != expected_dim:
            raise ValueError(f"Expected vector dim {expected_dim}, got {len(self.vector)}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")

@dataclass
class PersonalState:
    """
    Canonical, model-independent per-user state (ABI v0.1).

    Facts/events should live in an explicit memory store. This structure holds
    learned/aggregated user state that can survive replacement of the LM.
    """
    user_id: str
    version: str = "0.1"
    core: StateSlot = field(default_factory=lambda: StateSlot([0.0] * CORE_DIM))
    policies: StateSlot = field(default_factory=lambda: StateSlot([0.0] * POLICY_DIM))
    worlds: Dict[str, StateSlot] = field(default_factory=dict)
    routing: StateSlot = field(default_factory=lambda: StateSlot([0.0] * ROUTING_DIM))

    def validate(self):
        self.core.validate(CORE_DIM)
        self.policies.validate(POLICY_DIM)
        self.routing.validate(ROUTING_DIM)
        for slot in self.worlds.values():
            slot.validate(WORLD_DIM)

    def flatten(self, scope: Optional[str] = None) -> torch.Tensor:
        """
        Produces a canonical vector independent of the language model.
        Layout:
          core[64], policy[64], selected_world[64], routing[32],
          four confidence scalars = 228 dims.
        """
        self.validate()
        world = self.worlds.get(scope or "", StateSlot([0.0] * WORLD_DIM))
        values = (
            self.core.vector
            + self.policies.vector
            + world.vector
            + self.routing.vector
            + [
                self.core.confidence,
                self.policies.confidence,
                world.confidence,
                self.routing.confidence,
            ]
        )
        return torch.tensor(values, dtype=torch.float32)

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(path: str) -> "PersonalState":
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)

        def slot(x):
            return StateSlot(
                vector=x["vector"],
                confidence=x.get("confidence", 0.0),
                evidence_count=x.get("evidence_count", 0),
            )

        obj = PersonalState(
            user_id=d["user_id"],
            version=d.get("version", "0.1"),
            core=slot(d["core"]),
            policies=slot(d["policies"]),
            worlds={k: slot(v) for k, v in d.get("worlds", {}).items()},
            routing=slot(d["routing"]),
        )
        obj.validate()
        return obj

@dataclass
class UpdateProposal:
    target: str  # "core", "policies", "routing", or "world:<name>"
    delta: List[float]
    confidence: float
    evidence_count: int = 1

class EvidenceConsolidator:
    """
    Safe update layer: the LM never writes directly to PersonalState.

    This first prototype deliberately uses deterministic evidence gating rather
    than a learned writer. A later T_write can produce UpdateProposal objects
    while this layer remains the final authority.
    """
    def __init__(self, min_confidence=0.70, min_evidence=2, max_step=0.08):
        self.min_confidence = min_confidence
        self.min_evidence = min_evidence
        self.max_step = max_step

    def apply(self, state: PersonalState, proposal: UpdateProposal) -> bool:
        if proposal.confidence < self.min_confidence:
            return False
        if proposal.evidence_count < self.min_evidence:
            return False

        if proposal.target == "core":
            slot, dim = state.core, CORE_DIM
        elif proposal.target == "policies":
            slot, dim = state.policies, POLICY_DIM
        elif proposal.target == "routing":
            slot, dim = state.routing, ROUTING_DIM
        elif proposal.target.startswith("world:"):
            name = proposal.target.split(":", 1)[1]
            slot = state.worlds.setdefault(name, StateSlot([0.0] * WORLD_DIM))
            dim = WORLD_DIM
        else:
            raise ValueError("Unknown proposal target")

        if len(proposal.delta) != dim:
            raise ValueError(f"delta must have dim {dim}")

        step = min(self.max_step, proposal.confidence * self.max_step)
        slot.vector = [
            max(-1.0, min(1.0, old + step * delta))
            for old, delta in zip(slot.vector, proposal.delta)
        ]
        slot.evidence_count += proposal.evidence_count
        slot.confidence = min(
            1.0,
            max(slot.confidence, proposal.confidence) * 0.9 + 0.1
        )
        return True
