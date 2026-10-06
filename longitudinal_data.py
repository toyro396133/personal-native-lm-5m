from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from personal_state import PersonalState, StateSlot, CORE_DIM, POLICY_DIM, WORLD_DIM, ROUTING_DIM


@dataclass(frozen=True)
class LongitudinalProfile:
    user_id: str
    core: int
    policy: int
    world: int
    history: List[str]


def _choice_text(slot: str, value: int) -> str:
    chosen = "1" if value else "2"
    if slot == "core":
        return f"המשתמש בחר אפשרות {chosen} למשימה מעשית."
    if slot == "policy":
        return f"המשתמש בחר פורמט {chosen} לתשובה."
    if slot == "world":
        return f"בפרויקט העבודה המשתמש בחר דרך {chosen}."
    raise ValueError(slot)


def _noise(i: int) -> str:
    bank = [
        "המשתמש פתח את המערכת ובדק את מצב הפרויקט.",
        "נשמרה אינטראקציה רגילה ללא בחירה חדשה.",
        "המשתמש קרא סיכום קצר והמשיך למשימה הבאה.",
        "המערכת הציגה מידע טכני כללי.",
        "המשתמש חזר לעבודה לאחר הפסקה קצרה.",
    ]
    return bank[i % len(bank)]


def build_history(core: int, policy: int, world: int, *, changed_core: bool = False) -> List[str]:
    """Create a longer history with repetition, noise and one contradiction.

    If changed_core=True, the older core preference is the opposite value, but
    the latest repeated evidence supports `core`. This gives the history encoder
    a recency/consistency problem instead of a flat lookup table.
    """
    old_core = 1 - core if changed_core else core
    h = [
        _noise(0),
        _choice_text("core", old_core),
        _noise(1),
        _choice_text("policy", policy),
        _noise(2),
        _choice_text("world", world),
        _noise(3),
        # One contradictory policy event should not dominate repeated evidence.
        _choice_text("policy", 1 - policy),
        _noise(4),
        _choice_text("policy", policy),
    ]
    if changed_core:
        h += [
            _noise(0),
            _choice_text("core", core),
            _noise(1),
            _choice_text("core", core),
        ]
    else:
        h += [_choice_text("core", core)]
    h += [
        _noise(2),
        _choice_text("world", world),
        _noise(3),
    ]
    return h


def make_profiles() -> List[LongitudinalProfile]:
    profiles = []
    idx = 0
    for core in (0, 1):
        for policy in (0, 1):
            for world in (0, 1):
                profiles.append(
                    LongitudinalProfile(
                        user_id=f"long-{idx}",
                        core=core,
                        policy=policy,
                        world=world,
                        history=build_history(core, policy, world, changed_core=(idx % 2 == 1)),
                    )
                )
                idx += 1
    return profiles


def target_state(profile: LongitudinalProfile):
    core = [0.0] * CORE_DIM
    policy = [0.0] * POLICY_DIM
    world = [0.0] * WORLD_DIM
    routing = [0.0] * ROUTING_DIM
    core[0] = 0.6 if profile.core else -0.6
    policy[0] = 0.6 if profile.policy else -0.6
    world[0] = 0.6 if profile.world else -0.6
    # Routing slot is not user preference here; keep one mild stable marker.
    routing[0] = 0.15
    s = PersonalState(
        user_id=profile.user_id,
        core=StateSlot(core, confidence=0.82, evidence_count=10),
        policies=StateSlot(policy, confidence=0.82, evidence_count=10),
        worlds={"work": StateSlot(world, confidence=0.82, evidence_count=10)},
        routing=StateSlot(routing, confidence=0.65, evidence_count=5),
    )
    return s.flatten("work")
