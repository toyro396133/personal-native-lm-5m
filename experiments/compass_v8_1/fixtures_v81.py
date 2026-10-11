"""COMPASS V8.1 fictional contrastive project fixtures.

Each project has paired scenarios which differ in one authoritative fact. Model
inputs contain evidence and permission status, never oracle actions. Gold files
are separate. Research-only English fictional data.
"""
import argparse
import json
from pathlib import Path

TRAIN_PROFILES=[
    ("juniper","defer"),("maple","schedule"),("linden","defer"),
    ("cedar","schedule"),("willow","defer"),("ash","schedule"),
    ("pine","defer"),("elm","schedule"),
]
TEST_PROFILES=[("saffron","defer"),("cobalt","schedule"),("umber","defer")]
TRAIN_PROJECTS=[
    ("records","Keep an accurate case-record workflow","saving records","record integrity"),
    ("analytics","Deliver reliable analytics","publishing dashboards","dashboard data"),
]
TEST_PROJECTS=[
    ("logistics","Deliver reliable dispatch plans","assigning delivery routes","route integrity"),
    ("language","Produce dependable language learning exercises","publishing lessons","lesson accuracy"),
]
CASES=(
    "confirmed_failure","uncertain_failure",
    "targeted_regression","optional_checks",
    "critical_dependency","routine_dependency",
    "approved_change","pending_change",
    "approved_expansion","out_scope_expansion",
    "later_cosmetics","unclear_suggestion",
)
ACTIONS=("DO_NOW","SCHEDULE","INVESTIGATE","DEFER","ASK_USER","REJECT")
GOLD_ACTIONS={
    "confirmed_failure":"DO_NOW","uncertain_failure":"INVESTIGATE",
    "targeted_regression":"DO_NOW","optional_checks":None,
    "critical_dependency":"DO_NOW","routine_dependency":"SCHEDULE",
    "approved_change":"SCHEDULE","pending_change":"ASK_USER",
    "approved_expansion":"SCHEDULE","out_scope_expansion":"REJECT",
    "later_cosmetics":"DEFER","unclear_suggestion":"INVESTIGATE",
}
NODE_FOR={
    "confirmed_failure":"C","uncertain_failure":"C",
    "targeted_regression":"A","optional_checks":"A",
    "critical_dependency":"S","routine_dependency":"S",
    "approved_change":"C","pending_change":"C",
    "approved_expansion":"X","out_scope_expansion":"X",
    "later_cosmetics":"A","unclear_suggestion":"A",
}
SPLIT_WORDS={
"train":{
"confirmed_failure":"Confirmed failure: {core} is broken, reproducible logs show corrupted results.",
"uncertain_failure":"A report alleges trouble with {core} but nobody has verified the problem.",
"targeted_regression":"A single focused regression check is essential to release the fix for broken {core}.",
"optional_checks":"Proposal to add broad routine tests not linked to any failing core deliverable.",
"critical_dependency":"The supporting setup for {support} blocks the repair of confirmed {core} breakage.",
"routine_dependency":"The planned, nonblocking {support} upgrade is ready for the next milestone.",
"approved_change":"The project owner approved revising the {core} milestone; schedule the authorized revision.",
"pending_change":"A colleague suggests replacing the core mission for {core} without owner approval.",
"approved_expansion":"An additional feature has been approved as in-scope for the next milestone.",
"out_scope_expansion":"A side product is outside the approved project scope.",
"later_cosmetics":"A decorative polish idea adds no essential support for {core} in the current phase.",
"unclear_suggestion":"A possible extra improvement has no known impact or verified benefit yet.",
},
"test":{
"confirmed_failure":"An independently reproduced fault in {core} blocks the accepted objective now.",
"uncertain_failure":"There might be a defect in {core}; the symptoms are vague and evidence is absent.",
"targeted_regression":"Before deploying a confirmed repair to {core}, one narrow verification case is mandatory.",
"optional_checks":"A hundred extra broad checks are proposed, unrelated to a known current incident.",
"critical_dependency":"A required {support} prerequisite stops the team fixing an already reproduced {core} incident.",
"routine_dependency":"Improve {support} during regular planning; current core operation is not obstructed.",
"approved_change":"The authorized owner signed off on adjusting {core}; book it into the planned work.",
"pending_change":"A team member requests a new core mission, yet the necessary owner sign-off is absent.",
"approved_expansion":"The project owner explicitly authorized this once-outside feature to enter project scope.",
"out_scope_expansion":"The suggested side business has no authorization and remains outside the project boundaries.",
"later_cosmetics":"Revise decorative presentation details that will not help the current {core} goal.",
"unclear_suggestion":"Someone offers an idea whose usefulness and relation to the approved goal remain unproven.",
}}
def build(split):
    profiles=TRAIN_PROFILES if split=="train" else TEST_PROFILES
    projects=TRAIN_PROJECTS if split=="train" else TEST_PROJECTS
    contexts=[];labels=[]
    for pi,(persona,qa_mode) in enumerate(profiles):
      for ji,(short,goal,core,support) in enumerate(projects):
        project_id=f"fiction-{split[0]}-{persona}-{short}"
        policy={
            "id":f"POL-{persona}","source":"fictional-approved-preference",
            "version":1,
            "qa_mode":qa_mode,
            "text":("Do optional broad quality checks later as scheduled work."
                    if qa_mode=="schedule" else
                    "Defer optional broad quality checks without a core dependency."),
        }
        nodes=[
            {"id":project_id+":C","layer":"core","intent":core},
            {"id":project_id+":S","layer":"supporting","intent":support},
            {"id":project_id+":A","layer":"ancillary","intent":"testing and optional polish"},
            {"id":project_id+":X","layer":"ancillary","intent":"proposed side feature subject to a separate scope decision"},
        ]
        map_data={
            "id":project_id,"version":1,"objective":goal,"phase":"foundation",
            "nodes":nodes,
            "decisions":[
                {"id":project_id+":D1","source":"fictional-owner","valid":True,
                 "statement":"Confirmed failures to the core objective outrank optional expansion."},
                {"id":project_id+":D2","source":"fictional-owner","valid":True,
                 "statement":"A core mission revision must be approved by the project owner."},
                {"id":project_id+":D3","source":"fictional-owner","valid":True,
                 "statement":"Work outside approved project scope cannot proceed."},
                {"id":project_id+":D4","source":"fictional-owner","valid":True,
                 "statement":"A targeted regression check needed to release a confirmed core fix may be urgent."},
            ]
        }
        for ci,case in enumerate(CASES):
            node=next(n for n in nodes if n["id"].endswith(":"+NODE_FOR[case]))
            is_pending=case=="pending_change"
            is_approved=case in ("approved_change","approved_expansion")
            is_out=case=="out_scope_expansion"
            requires_approval=case in ("pending_change","approved_change","approved_expansion")
            evidence_state=("confirmed" if case in
                ("confirmed_failure","targeted_regression","critical_dependency")
                else "unverified" if case in ("uncertain_failure","unclear_suggestion")
                else "not_applicable")
            proposal=SPLIT_WORDS[split][case].format(core=core,support=support)
            task={
                "id":f"ITEM-{split[0].upper()}{pi:02d}{ji:02d}{ci:02d}",
                "project_id":project_id,"node_id":node["id"],
                "proposal":proposal,
                "scope_status":"outside" if is_out else "inside",
                "requires_owner_approval":requires_approval,
                "approval_status":"granted" if is_approved else "pending" if is_pending else "not_required",
                "evidence_status":evidence_state,
                "blocked_core":case in ("confirmed_failure","targeted_regression","critical_dependency"),
                "evidence_source_id":f"SRC-{split[0]}{pi}{ji}-{ci:02d}",
            }
            action=(qa_mode.upper() if case=="optional_checks" else GOLD_ACTIONS[case])
            if action is None:raise RuntimeError(case)
            assert action in ACTIONS
            context={"profile_key":persona,"policy":policy,"project":map_data,"task":task}
            gold={"task_id":task["id"],"action":action,"case_kind":case,
                  "project_id":project_id,"layer":node["layer"],
                  "evidence_id":task["evidence_source_id"],
                  "source_family":split,
                  "template_id":f"{split}-T{ci:02d}"}
            contexts.append(context);labels.append(gold)
    return contexts,labels

def generate():
    inputs={};gold={}
    for s in ("train","test"):inputs[s],gold[s]=build(s)
    return inputs,gold

def save(root):
    root=Path(root)
    a,b=generate()
    for group,data in (("input",a),("gold",b)):
        folder=root/group;folder.mkdir(parents=True,exist_ok=True)
        for split,rows in data.items():
            (folder/(split+".jsonl")).write_text(
                "".join(json.dumps(x,ensure_ascii=False)+"\n" for x in rows),
                encoding="utf-8")
    return {s:len(a[s]) for s in a}

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--root",default="v8-1-fixtures")
    args=ap.parse_args();print(save(args.root))
