"""V8.1 authorization enforcement separate from learned prioritization.

These checks are explicitly external project state, not inferred labels and not
part of neural action-learning scores. Guarded accuracy must NEVER be reported
as raw model accuracy.
"""
ACTIONS=("DO_NOW","SCHEDULE","INVESTIGATE","DEFER","ASK_USER","REJECT")

class InvalidProjectContext(ValueError):
    pass

def guard_action(context,predicted_action):
    if predicted_action not in ACTIONS:
        raise ValueError("Unsupported action")
    project=context["project"];task=context["task"]
    if task["project_id"]!=project["id"]:
        raise InvalidProjectContext("Cross-project contamination: project_id mismatch")
    nodes={n["id"] for n in project["nodes"]}
    if task["node_id"] not in nodes:
        raise InvalidProjectContext("Task node is absent from authoritative project map")
    scope=task.get("scope_status")
    if scope not in ("inside","outside"):
        return {"action":"ASK_USER","overridden":True,"reason":"scope_unverified"}
    if scope=="outside":
        return {"action":"REJECT","overridden":predicted_action!="REJECT","reason":"explicit_out_of_scope"}
    needs=task.get("requires_owner_approval")
    approval=task.get("approval_status")
    if needs is not True and needs is not False:
        return {"action":"ASK_USER","overridden":True,"reason":"approval_requirement_unknown"}
    if needs and approval!="granted":
        return {"action":"ASK_USER","overridden":predicted_action!="ASK_USER","reason":"approval_not_granted"}
    return {"action":predicted_action,"overridden":False,"reason":"no_authorization_override"}
