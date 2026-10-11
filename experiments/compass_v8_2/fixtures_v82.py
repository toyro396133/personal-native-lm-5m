"""COMPASS V8.2 synthetic bilingual *pairwise* priority-reasoning dataset.

Task: choose between TWO proposed steps in a named fictional project.
For each decision family, an evidence flip changes which semantic step is best.
For each state the options are shown in both orders, in English and Hebrew.
The model's prompt does not include gold labels, dilemma type or state IDs.
Historical V8/V8.1 fixtures remain untouched.
"""
import json
import hashlib
from pathlib import Path

TRAIN_PROFILES=("birch","fern","juniper","larch","pine","cedar")
TEST_PROFILES=("agate","saffron","topaz")
TRAIN_PROJECTS=(
    ("archive","maintaining reliable inventory records","שמירת רישומי מלאי מדויקים"),
    ("sensor","delivering a reliable air-quality dashboard","הפעלת לוח נתוני איכות אוויר מהימן"),
)
TEST_PROJECTS=(
    ("dispatch","dispatching accurate service schedules","ניהול לוח שירות מדויק"),
    ("curriculum","publishing reliable training exercises","פרסום תרגילי לימוד מהימנים"),
)
KINDS=("confirmed","targeted","dependency","approval","scope","revision")
LANGUAGES=("en","he")
# option 0 and 1 have consistent meaning; 'state' controls which is right.
OPTIONS={
"confirmed":{
 "en":("Repair the alleged core defect immediately.","Verify the reported defect and gather proof before making changes."),
 "he":("לתקן מיד את התקלה המדווחת בליבת הפרויקט.","לבדוק את הדיווח ולאסוף ראיות לפני ביצוע שינוי."),
},
"targeted":{
 "en":("Write the focused regression check protecting the core repair.","Schedule a broad optional test-suite expansion."),
 "he":("לכתוב בדיקת רגרסיה ממוקדת שנדרשת לתיקון הליבה.","לתזמן הרחבה כללית של בדיקות שאינן דחופות."),
},
"dependency":{
 "en":("Complete the prerequisite work before attempting the blocked core milestone.","Continue the approved next core milestone without additional setup."),
 "he":("להשלים תנאי מקדים כדי לשחרר משימה חסומה בליבה.","להמשיך למשימת הליבה המתוכננת בלי היערכות נוספת."),
},
"approval":{
 "en":("Implement the proposed change to the project's core mission.","Seek owner approval before changing the core mission."),
 "he":("ליישם את השינוי המוצע במטרת הליבה של הפרויקט.","לבקש אישור מבעל הפרויקט לפני שינוי מטרת הליבה."),
},
"scope":{
 "en":("Begin the newly proposed project feature.","Reject the proposal because it falls outside approved scope."),
 "he":("להתחיל לפתח את היכולת החדשה שהוצעה.","לדחות את ההצעה משום שהיא מחוץ לתחום המאושר."),
},
"revision":{
 "en":("Apply the newly suggested replacement plan.","Continue the previously accepted plan."),
 "he":("להחיל את התוכנית החלופית שהוצעה לאחרונה.","להמשיך בתוכנית שהתקבלה קודם."),
},
}
# All state details are explicitly sourced in the prompt, without answer labels.
FACTS={
"confirmed":{
 "en":("The failure was reproduced twice and blocks the core outcome.",
       "The report is unverified and the claimed failure may not exist."),
 "he":("התקלה שוחזרה פעמיים והיא חוסמת את תוצאת הליבה.",
       "הדיווח אינו מאומת ולא ידוע אם התקלה בכלל קיימת."),
},
"targeted":{
 "en":("A confirmed core fix cannot be released until the targeted regression check passes.",
       "The old fault is already resolved and the targeted check is obsolete; routine QA planning is next."),
 "he":("אי אפשר לשחרר תיקון מוכח לליבה בלי בדיקת הרגרסיה הממוקדת.",
       "התקלה הישנה נפתרה והבדיקה הממוקדת כבר אינה נחוצה; נותר תכנון בדיקות שגרתי."),
},
"dependency":{
 "en":("The supporting setup is a proven blocker for today's core milestone.",
       "The proposed setup is optional and not needed for the next approved core milestone."),
 "he":("ההכנה התומכת חוסמת בפועל את השלמת אבן הדרך הנוכחית של הליבה.",
       "ההכנה שהוצעה היא רשות ואינה נחוצה לאבן הדרך המאושרת הבאה."),
},
"approval":{
 "en":("The project owner explicitly approved the proposed mission change.",
       "Only a teammate suggested the change; the owner has not approved it."),
 "he":("בעל הפרויקט אישר במפורש את השינוי המוצע במטרה.",
       "רק חבר צוות הציע את השינוי; בעל הפרויקט לא אישר אותו."),
},
"scope":{
 "en":("The owner explicitly added the feature to the approved project scope.",
       "The requested new feature is expressly excluded from the project scope."),
 "he":("בעל הפרויקט כלל במפורש את היכולת החדשה בתחום המאושר.",
       "היכולת החדשה הוחרגה במפורש מתחום הפרויקט."),
},
"revision":{
 "en":("The owner formally superseded the original plan with the new plan.",
       "The new plan is just an unapproved idea; the original plan is still current."),
 "he":("בעל הפרויקט החליף רשמית את התוכנית המקורית בתוכנית החדשה.",
       "התוכנית החדשה היא רק רעיון שלא אושר; התוכנית המקורית עדיין בתוקף."),
},
}
LAYERS={"confirmed":"core","targeted":"ancillary","dependency":"supporting",
        "approval":"core","scope":"ancillary","revision":"supporting"}
DOCUMENTED_SOURCE={"en":"Approved project ledger and attached incident/approval record.",
                   "he":"יומן החלטות הפרויקט המאושר ורישום האירוע או האישור הנלווה."}

def make_project(split,profile_idx,project_idx):
    name,english,hebrew=(TRAIN_PROJECTS if split=="train" else TEST_PROJECTS)[project_idx]
    person=(TRAIN_PROFILES if split=="train" else TEST_PROFILES)[profile_idx]
    pid=f"fiction-{split[0]}-{person}-{name}"
    return {"id":pid,"version":1,"phase":"core_build","objective":{"en":english,"he":hebrew},
            "nodes":[
                {"id":pid+":C","layer":"core","description":{"en":"Essential agreed product outcome","he":"התוצאה החיונית והמוסכמת"}},
                {"id":pid+":S","layer":"supporting","description":{"en":"Supporting work that can block the core","he":"שכבה תומכת שעשויה לחסום את הליבה"}},
                {"id":pid+":A","layer":"ancillary","description":{"en":"Checks and optional improvements","he":"בדיקות ושיפורים משלימים"}},
            ],
            "policy":{
                "id":f"POL-{person}","version":1,"source":"fictional-owner",
                "en":"Protect the real core before optional work. Respect explicit approvals and changes in evidence.",
                "he":"יש לקדם את הליבה לפני תוספות רשות, ולכבד אישורים מפורשים ושינויים בראיות."
            }}

def make_rows(split):
    projects=TRAIN_PROJECTS if split=="train" else TEST_PROJECTS
    profiles=TRAIN_PROFILES if split=="train" else TEST_PROFILES
    inputs=[];gold=[]
    for pi in range(len(profiles)):
      for pj in range(len(projects)):
        p=make_project(split,pi,pj)
        for lang in LANGUAGES:
          for ki,kind in enumerate(KINDS):
            for state in (0,1):
              for swap in (0,1):
                opt=list(OPTIONS[kind][lang])
                if swap:opt.reverse()
                # Core fact determines whether canonical option 0 or 1 is best.
                correct_option=state^swap
                taskid="ITEM-"+hashlib.sha256(f"compass82-task-{split}-{pi}-{pj}-{lang}-{ki}-{state}-{swap}".encode()).hexdigest()[:14]
                fact=FACTS[kind][lang][state]
                input_row={
                    "project":p,"lang":lang,
                    "task":{
                        "id":taskid,"project_id":p["id"],"layer":LAYERS[kind],
                        "proposal_id":f"REQ-{pi}{pj}{ki:02d}",
                        "record":fact,
                        "record_source_id":"SRC-"+hashlib.sha256(f"compass82-source-{split}-{pi}-{pj}-{ki}-{state}".encode()).hexdigest()[:12],
                        "record_source_description":DOCUMENTED_SOURCE[lang],
                        "choices":opt,
                        "authoritative_gate":{
                            "provenance":"fictional-owner-record",
                            "approval_granted":state==0 if kind in ("approval","revision") else None,
                            "feature_in_scope":state==0 if kind=="scope" else None,
                            "choice_intents":["hold","execute"] if swap else ["execute","hold"],
                            "governed_action":kind if kind in ("approval","scope","revision") else None,
                        },
                    }
                }
                label={"id":taskid,"project_id":p["id"],"kind":kind,
                       "state":state,"swap":swap,"language":lang,
                       "correct_digit":correct_option,
                       "correct_semantic_option":state,
                       "label_source":"fictional_authoring_oracle"}
                inputs.append(input_row);gold.append(label)
    return inputs,gold

def generate():
    rows={};labels={}
    for split in ("train","test"):rows[split],labels[split]=make_rows(split)
    return rows,labels

def save(root):
    root=Path(root);inputs,labels=generate()
    for dirname,records in (("inputs",inputs),("gold",labels)):
        folder=root/dirname;folder.mkdir(parents=True,exist_ok=True)
        for split,rows in records.items():
            (folder/(split+".jsonl")).write_text(
                "".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
    return {k:len(v) for k,v in inputs.items()}

if __name__=="__main__":
    import argparse
    a=argparse.ArgumentParser();a.add_argument("--out",default="v82-audit-data")
    args=a.parse_args();print(json.dumps(save(args.out)))
