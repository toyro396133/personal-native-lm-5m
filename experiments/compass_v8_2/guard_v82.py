"""Deterministic action-execution guard for COMPASS V8.2 fictional cases.

Only externally supplied, provenance-tagged owner permission fields are consulted.
This is NOT learning and guarded outputs are always measured separately.
"""
class InvalidContext(ValueError):
    pass

def enforce(row,choice):
    task=row["task"];project=row["project"]
    if task["project_id"]!=project["id"] or not project.get("id"):
        raise InvalidContext("Mismatched project identity")
    if choice not in (0,1):
        raise ValueError("Pair choice must be 0 or 1")
    gate=task.get("authoritative_gate")
    if not isinstance(gate,dict) or gate.get("provenance")!="fictional-owner-record":
        raise InvalidContext("No trusted action-authorization source")
    intents=gate.get("choice_intents")
    if not isinstance(intents,list) or len(intents)!=2 or set(intents)!={"execute","hold"}:
        raise InvalidContext("Invalid execution/hold intents")
    kind=gate.get("governed_action")
    if kind not in ("approval","scope","revision",None):
        raise InvalidContext("Unrecognized governed action")
    allowed=True
    if kind in ("approval","revision"):
        allowed=gate.get("approval_granted") is True
    elif kind=="scope":
        allowed=gate.get("feature_in_scope") is True
    if not allowed and intents[choice]=="execute":
        new=intents.index("hold")
        return {"choice":new,"overridden":True,"reason":"owner_or_scope_not_authorized"}
    return {"choice":choice,"overridden":False,"reason":"no_override"}
