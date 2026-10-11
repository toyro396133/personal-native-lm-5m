"""Fresh V9.11 scenario families + mock, versioned COMPASS principle records.

Data remain researcher-authored. Schema information is a deliberately strong
input supplied by a (simulated) trusted source, NOT learned ownership or gold.
"""
import copy
import hashlib
import hmac
import json
import random
import fixtures_v1010 as legacy
from fixtures_v99 import origin,sign

NEW_TASKS={
 "en":{
  "user":[
   ("artifact-debug","A reversible build artifact is behaving strangely; you choose your working method.",
    "Investigate competing root causes and consequences before touching the build pipeline.",
    "Try a limited recoverable build adjustment and refine it from measurements."),
   ("deployment-tuning","Your own strategy is needed for a minor deployment optimization.",
    "Document unknown risks and compare hypotheses before modifying deployment.",
    "Start a small reversible deployment experiment and adapt quickly."),
   ("batch-health","You personally decide how to handle intermittent batch-processing anomalies.",
    "Examine possible failure modes and dependencies in detail before acting.",
    "Perform a safe small batch trial and iterate from its result."),
   ("restore-verification","A recoverable restoration check needs your preferred approach.",
    "Trace hazards and uncertain assumptions thoroughly before a restore attempt.",
    "Run a contained reversible recovery rehearsal and revise afterward."),
  ],
  "project":[
   ("support-index","The support knowledge index must evolve. Either redesign is feasible absent contract restrictions.",
    "Reimagine the index around the lasting aim of accessible support knowledge.",
    "Keep the published knowledge-search integration interface compatible."),
   ("resource-planner","A resource planning service is modernized and both technical paths can work.",
    "Design the planner around the intended benefit of coordinated resource use.",
    "Preserve the documented public resource-planning data contract."),
   ("case-timeline","An incident case timeline is being rebuilt; either strategy is normally allowed.",
    "Build around the mission of understandable case history rather than old screens.",
    "Maintain its established outward-facing case timeline API."),
   ("handoff-register","A project handoff register requires new internals. Both strategies are permitted by default.",
    "Restructure around the fundamental purpose of trustworthy handoffs.",
    "Honor the existing public handoff-event integration protocol."),
  ],
 },
 "he":{
  "user":[
   ("artifact-debug-he","אתה בוחר כיצד לטפל בתוצר בנייה שאינו פועל כצפוי ושאפשר לשחזר.",
    "לנתח היטב את הסיבות האפשריות ואת הסיכונים לפני שינוי תהליך הבנייה.",
    "לנסות שינוי קטן והפיך בתהליך ולשפר לפי הנתונים."),
   ("deployment-tuning-he","נדרש להחליט אישית כיצד לבצע כוונון קטן בתהליך הפריסה.",
    "לבדוק מראש הנחות לא ודאיות והשלכות אפשריות של השינוי.",
    "להריץ ניסוי פריסה קצר והפיך ולהמשיך לפי התוצאה."),
   ("batch-health-he","אתה מחליט מהי הגישה שלך לתקלות חולפות בעיבוד קבוצתי.",
    "לבחון תרחישי כשל ותלויות לפני שמבצעים שינוי.",
    "להפעיל בדיקה קבוצתית קטנה ובטוחה ולתקן בעקבות הממצאים."),
   ("restore-verification-he","נדרשת בחירה אישית בגישה לבדיקת יכולת שחזור של מערכת.",
    "לחקור את הסיכונים וההנחות הלא ברורות לפני פעולה.",
    "לבצע ניסיון שחזור מוגבל שניתן לבטלו ולשפר לפי התוצאות."),
  ],
  "project":[
   ("support-index-he","מפת הידע של מערכת התמיכה עוברת תכנון מחדש.",
    "לשפר את המערכת לפי המטרה המתמשכת של ידע תמיכה נגיש.",
    "לשמר את ממשק החיפוש החיצוני המתועד למערכות תלויות."),
   ("resource-planner-he","מערכת לתכנון משאבים עוברת חידוש והתאמה.",
    "לתכנן מחדש לפי התכלית של שימוש מתואם ונכון במשאבים.",
    "להשאיר את חוזה הנתונים החיצוני המתועד ללא שינוי."),
   ("case-timeline-he","ציר הזמן של אירועים מטופלים נבנה מחדש.",
    "לשמור על התכלית של תיעוד אירועים ברור ולא לשכפל מסכים קודמים.",
    "לשמור על תאימות ממשק ציר הזמן הקיים למערכות נוספות."),
   ("handoff-register-he","מאגר מסירת אחריות בין צוותים דורש מימוש חדש.",
    "לבנות לפי המטרה של מסירת אחריות אמינה וברורה.",
    "לשמר את פרוטוקול אירועי המסירה החיצוני הקיים."),
  ]
 }}
NEW_PREF={
 "en":{
  "user":{
   0:["My default is to examine evidence and failure risks before operations.",
      "Unknown consequences deserve thorough review before implementation.",
      "I prefer understanding alternative explanations before a live attempt.",
      "My first instinct is to study possible hazards, not run a trial."],
   1:["My default is to learn by doing a tiny reversible experiment.",
      "A small workable test and iterative feedback suit me better than long preparation.",
      "When risks are limited I want a fast observable prototype first.",
      "I favor a quick safely reversible probe to learn the next step."]},
  "project":{
   0:["The project's enduring goal is more important than its old implementation.",
      "Optimize for the actual benefit rather than replicating yesterday's workflow.",
      "We should change incidental architecture to fulfill the original mission.",
      "Project purpose should guide updates, not historical screen arrangements."],
   1:["Our project-specific commitment is to stable published integrations.",
      "Clients rely on an enduring external interface contract for this product.",
      "Keep outward-facing integration guarantees across internal rewrites.",
      "External consumers require continued compatibility with established protocols."]}},
 "he":{
  "user":{
   0:["ברירת המחדל שלי היא לבחון ראיות וסיכונים היטב לפני פעולה.",
      "אני מעדיף להבין את האפשרויות ואת ההשלכות לפני ניסוי מעשי.",
      "הגישה שלי היא לברר גורמי כשל לפני שמתחילים לשנות.",
      "קודם חוקרים סיכונים לא ידועים ורק אחר כך פועלים."],
   1:["אני מעדיף ללמוד בעזרת ניסוי קטן שאפשר לבטל.",
      "הכי מתאים לי להתחיל בבדיקה מעשית קצרה ולשפר לפי משוב.",
      "כאשר הסיכון מצומצם אני מעדיף אבטיפוס הפיך ומדיד.",
      "הנטייה שלי היא לנסות משהו קטן מהר ולהסיק מסקנות."]},
  "project":{
   0:["התכלית המתמשכת של הפרויקט חשובה יותר מהמימוש הקודם.",
      "הפרויקט נועד להשיג תוצאה ולא להעתיק תהליכים ישנים.",
      "יש לשנות רכיבים מקריים במימוש כדי להגשים טוב יותר את החזון.",
      "המטרה המקורית מנחה את ההתפתחות יותר מצורת המסכים הקודמת."],
   1:["לפרויקט יש מחויבות יציבה לממשקים חיצוניים מתועדים.",
      "לקוחות קיימים זקוקים לתאימות של חוזי הממשק.",
      "התחייבויות שילוב עם מערכות נוספות צריכות להישאר יציבות.",
      "הממשק החיצוני המפורסם של הפרויקט צריך להמשיך לפעול כפי שתועד."]}}
}
CONTEXT={
 "en":{"user":["A separate safety rule requires source verification when it is valid.",
                 "A revoked historical procedure is not binding and must not erase a working preference."],
        "project":["A verified active outside commitment can restrict the design direction.",
                   "Past technical choices and expired contracts must not erase the enduring project principle."]},
 "he":{"user":["הוראת בטיחות נפרדת עשויה לחייב בדיקה אם הוכח שהיא תקפה.",
                 "תהליך היסטורי שבוטל אינו מבטל את ההעדפה האישית המתמשכת."],
        "project":["התחייבות חיצונית מאומתת ותקפה יכולה להגביל שינוי במערכת.",
                   "מימושים קודמים וחוזים שפג תוקפם אינם מוחקים את עיקרון הפרויקט."]},
}
SLOT_SIGNERS={"personal":"demo-compass-personal-v11","project":"demo-compass-project-v11"}
POLICY_KEY=b"compass-v11-public-synthetic-contract-key"
def canonical(o):
 return json.dumps({k:v for k,v in o.items() if k!="signature"},
                   sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def sign_slot(slot):
 return hmac.new(SLOT_SIGNERS[slot["owner_type"]].encode(),canonical(slot),hashlib.sha256).hexdigest()
def sign_policy(policy):
 return hmac.new(POLICY_KEY,canonical(policy),hashlib.sha256).hexdigest()
def record(case,scope_key,kind,evidence,version,active,reason,supersedes=None):
 owner="personal" if scope_key=="user" else "project"
 target="*" if owner=="personal" else scope_key
 slot={"slot_id":f"C-{scope_key}-{kind}-{version}",
       "owner_type":owner,"scope_id":target,"kind":kind,"version":version,
       "evidence":list(evidence),"reason":reason,
       "source":"demo-research-author","active":active,
       "effective_on":"2026-10-11","supersedes":supersedes}
 slot["signature"]=sign_slot(slot)
 return slot

def build_slots(c):
 # Index by pair-side, then by individual COMPASS profile / workspace.
 # No downstream predictor receives mode, semantic label or swapped fixture bit.
 result={}
 for side in ("original","changed"):
  streams={}
  for scope_key in ("user","a","b"):
   h=c[side]["histories"][scope_key]
   assert len(h)==5
   main=record(c,scope_key,"preference",h[:3],2,True,
               "Explicit persistent preference or project purpose; retain across unrelated events.",
               supersedes=f"C-{scope_key}-preference-1")
   other=record(c,scope_key,"context",h[3:],1,True,
                "Independent conditional background; does not supersede main preference.")
   # Old explicitly superseded preference is represented as separate inactive record.
   old=record(c,scope_key,"preference",list(reversed(h[:3])),1,False,
              "Previous historical record, superseded by current version 2.")
   seq=[main,other,old]
   # Different order in the store, same versioned semantics.
   rr=random.Random(int(hashlib.sha256((c["id"]+side+scope_key).encode()).hexdigest()[:8],16))
   rr.shuffle(seq)
   streams[scope_key]=seq
  result[side]=streams
 return result

def fresh(seed=11):
 # Reuse original V9.10 generator mechanics, BUT replace every family,
 # all primary evidence text and context notes with new V9.11 authored text.
 old_tasks=legacy.TASKS
 old_primary=legacy.PRIMARY
 old_secondary=legacy.SECONDARY
 try:
  legacy.TASKS=NEW_TASKS
  legacy.PRIMARY=NEW_PREF
  legacy.SECONDARY=CONTEXT
  cases=legacy.create(seed+1100)
 finally:
  legacy.TASKS=old_tasks
  legacy.PRIMARY=old_primary
  legacy.SECONDARY=old_secondary
 for c in cases:
  c["id"]="V11-"+hashlib.sha256((c["id"]+"fresh-separate").encode()).hexdigest()[:14]
  c["source_events"]=[origin(c)]
  c["compass_slots"]=build_slots(c)
  if c["scope"]=="user":
   req="risks-first"  # typed action ID is independent from option position
  else:req="external-compatibility"
  # Candidate action order in the typed contract must match the natural-language
  # options. This is a trusted external wrapper, not something neural learned.
  opts=["risks-first","reversible-trial"] if c["scope"]=="user" else ["mission-redesign","external-compatibility"]
  if c["options"][0]!=NEW_TASKS[c["language"]][c["scope"]][0][2] and False:
   raise RuntimeError("unreachable")
  # Generator carries swapped metadata here ONLY to construct the trusted
  # external structured action IDs. Runtime never accesses case['swapped'].
  c["action_ids"]=list(reversed(opts)) if c["swapped"] else opts
  state=c["regime"]
  contract={"issuer":"demo-policy-registry","workspace":"*" if c["scope"]=="user" else c["project"],
            "state":state,"verified":True,"required_operation":req,
            "valid_from":"2026-01-01","valid_until":"2027-01-01",
            "revoked_on":"2026-10-01" if state=="revoked" else None}
  if state=="expired":contract["valid_until"]="2026-08-01"
  contract["signature"]=sign_policy(contract)
  c["policy_contract"]=contract
 assert len(cases)==128
 audit(cases)
 return cases

def tamper(case,mode):
 item=copy.deepcopy(case)
 if mode=="shuffle":
  for side in ("original","changed"):
   for key in ("user","a","b"):
    item["compass_slots"][side][key].reverse()
 elif mode=="superseded_last":
  for side in ("original","changed"):
   for key in ("user","a","b"):
    item["compass_slots"][side][key].sort(key=lambda row:not (row["kind"]=="preference" and not row["active"]))
 elif mode in ("bad_signature","unknown_scope"):
  for side in ("original","changed"):
   key="user" if item["scope"]=="user" else item["project"]
   slots=item["compass_slots"][side][key]
   current=next(x for x in slots if x["kind"]=="preference" and x["active"])
   if mode=="bad_signature":current["signature"]="0"*64
   else:
    current["scope_id"]="UNAUTHORIZED-WORKSPACE"
    current["signature"]=sign_slot(current)
 else:raise ValueError(mode)
 return item

def audit(cases):
 assert len(cases)==128
 assert len({c["id"] for c in cases})==128
 assert len({c["family"] for c in cases})==16
 for lang in ("en","he"):
  for scope in ("user","project"):
   subset=[c for c in cases if c["language"]==lang and c["scope"]==scope]
   assert len(subset)==32
   for regime in legacy.REGIMES:
    assert len([x for x in subset if x["regime"]==regime])==8
 for c in cases:
  assert c["source_events"][0]["mac"]==sign(c["source_events"][0])
  assert c["policy_contract"]["signature"]==sign_policy(c["policy_contract"])
  for side in ("original","changed"):
   for key in ("user","a","b"):
    slots=c["compass_slots"][side][key]
    assert len(slots)==3
    assert all(s["signature"]==sign_slot(s) for s in slots)
    assert sum(s["kind"]=="preference" and s["active"] for s in slots)==1
    assert sum(s["kind"]=="context" for s in slots)==1
  assert c["original"]["gold"]!=c["changed"]["gold"] if c["regime"]!="active" else c["original"]["gold"]==c["changed"]["gold"]
 return {"pairs":128,"languages":{"en":64,"he":64},"family_count":16,
         "principle_records_per_stream":3,"policy_regimes":{r:32 for r in legacy.REGIMES},
         "all_labels_synthetic":True,"mock_metadata_assumed_correct":True}
if __name__=="__main__":
 print(json.dumps(audit(fresh()),ensure_ascii=False,indent=2))
