"""Executor-side source authorization CONTRACT; no neural or text-side permission.

A real service must populate records from authenticated, versioned external
sources. Here only synthetic unit fixtures are used. No cryptographic
signature verification is implemented, and NO live enforcement is claimed.
"""
from datetime import datetime,timezone

def utc(value):
    if isinstance(value,datetime):
        if value.tzinfo is None:raise ValueError("timezone-aware time required")
        return value.astimezone(timezone.utc)
    if isinstance(value,str):
        d=datetime.fromisoformat(value.replace("Z","+00:00"))
        if d.tzinfo is None: raise ValueError("timezone-aware time required")
        return d.astimezone(timezone.utc)
    raise ValueError("UTC time required")

def authorize_action(*,proposal_project,source_id,intent,now,verified_lookup):
    """Fail CLOSED for unknown/mismatched/withdrawn/conflicted/expired sources.

    verified_lookup must be an independently authenticated trusted-source
    registry that is NOT copied out of model prompt, model prediction or gold.
    """
    reasons=[]
    if intent=="hold":return {"allowed":True,"reason":"nonexecuting_hold"}
    if intent!="execute":return {"allowed":False,"reason":"unknown_intent"}
    if not proposal_project or not source_id:
        return {"allowed":False,"reason":"missing_identity"}
    try:record=verified_lookup.get(source_id)
    except Exception:return {"allowed":False,"reason":"lookup_unavailable"}
    if record is None:return {"allowed":False,"reason":"unknown_source"}
    if not isinstance(record,dict):return {"allowed":False,"reason":"invalid_registry_record"}
    if record.get("project_id")!=proposal_project:
        return {"allowed":False,"reason":"wrong_project"}
    if record.get("verification")!="trusted_verified":
        return {"allowed":False,"reason":"not_independently_verified"}
    if record.get("authority")!="owner_or_delegate":
        return {"allowed":False,"reason":"unapproved_issuer"}
    if record.get("revoked") is not False:
        return {"allowed":False,"reason":"revoked_or_unknown"}
    if record.get("conflict_cleared") is not True:
        return {"allowed":False,"reason":"unresolved_conflict"}
    if record.get("latest_version")!=record.get("source_version"):
        return {"allowed":False,"reason":"superseded_version"}
    try:
        current=utc(now)
        since=utc(record["valid_from"])
        until=utc(record["valid_until"])
    except (KeyError,ValueError,TypeError):
        return {"allowed":False,"reason":"missing_or_invalid_validity"}
    if not (since<=current<until):
        return {"allowed":False,"reason":"expired_or_not_yet_valid"}
    return {"allowed":True,"reason":"externally_verified_current_source"}

def assert_contract():
    stamp="2026-10-11T04:00:00+00:00"
    known={"A":{
        "project_id":"P","verification":"trusted_verified",
        "authority":"owner_or_delegate","revoked":False,
        "conflict_cleared":True,"source_version":3,"latest_version":3,
        "valid_from":"2026-10-10T00:00:00Z","valid_until":"2026-10-12T00:00:00Z",
    }}
    kwargs={"proposal_project":"P","source_id":"A","intent":"execute",
            "now":stamp,"verified_lookup":known}
    assert authorize_action(**kwargs)["allowed"] is True
    assert authorize_action(**{**kwargs,"intent":"hold"})["allowed"] is True
    assert authorize_action(**{**kwargs,"source_id":"MISSING"})["allowed"] is False
    assert authorize_action(**{**kwargs,"proposal_project":"Q"})["allowed"] is False
    for field,value in (
        ("verification","unknown"),("authority",None),("revoked",True),
        ("revoked",None),("conflict_cleared",False),
        ("source_version",2),("valid_until","2026-10-11T03:00:00Z"),
        ("valid_from","2026-10-12T00:00:00Z")
    ):
        trial={"A":{**known["A"],field:value}}
        assert authorize_action(**{**kwargs,"verified_lookup":trial})["allowed"] is False,field
    assert authorize_action(**{**kwargs,"verified_lookup":{}})["reason"]=="unknown_source"
    return {"status":"PASS","scenarios":12,
            "warning":"No cryptographic signatures, external provider connection or production authorizations tested"}

if __name__=="__main__":
    import json;print(json.dumps(assert_contract(),indent=2))
