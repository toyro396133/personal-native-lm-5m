"""COMPASS V9.15: real-project *weak supervision* of decision claims from BODY only.

IMPORTANT: Heading-derived label = silver, never independent user approval.
Train on allowlisted PRIVATE local project docs; no network or model packages.
Reports omit raw source text. No COMPASS memory changes or auto-approval.
"""
from __future__ import annotations
import argparse
import json
import math
import re
import sys
from collections import Counter,defaultdict
from pathlib import Path
from decision_miner import sections,load,resolve_ref

TRAIN = (
 "docs/product/importance-map.md",
 "docs/campaigns/telephone-fundraising-qa-log-2026-10-11.md",
 "docs/campaigns/telephone-assignment-and-relationship-unit-design-2026-10-11.md",
 "docs/campaigns/telephone-fundraising-operational-research.md",
 "docs/campaigns/operational-interview-followup-42-46.md",
 "docs/campaigns/operational-question-audit-index.md",
 "docs/campaigns/cha-pak-assembly-debrief.md",
 "docs/campaigns/operational-interview-live-notes.md",
)
HOLDOUT=(
 "docs/campaigns/telephone-call-outcomes-and-statuses-proposal-2026-10-11.md",
 "docs/product/interview-documentation-protocol.md",
)
PINNED_SHA="fe9863278a04dedf27a0c4fcb8cca56b91ec4544"
CLASSES=("APPROVED_CLAIM","PROPOSED_CLAIM","HISTORICAL_EVIDENCE")
CUE=re.compile(
 r"(?:החלטה\s+מאושרת|הכרעה\s+מאושרת|מעמד\s+מאושר|"
 r"הנחיה\s+מחייבת|עיקרון\s+מחייב|הוכרע|אושר|מאושר|מאושרת|"
 r"טיוטה|הצעה|הוצע|הצעת|מוצע|מוצעת|"
 r"מה\s+נעשה\s+בפועל|עד\s+היום|בעבר|היסטורי|"
 r"approved|proposed|proposal|draft|historical)",re.I)
LABEL_RULES=[
 ("HISTORICAL_EVIDENCE",re.compile(r"(?<!\w)(בעבר|היסטורי|עד\s+היום|מה\s+נעשה\s+בפועל|historical)(?!\w)",re.I)),
 ("PROPOSED_CLAIM",re.compile(r"(?<!\w)(הצעה|הצעת|מוצע|מוצעת|טיוטה|proposal|proposed|draft)(?!\w)",re.I)),
 ("APPROVED_CLAIM",re.compile(r"(?<!\w)(אושר|מאושר|מאושרת|מחייבת|הוכרע|החלטה\s+מאושרת|approved)(?!\w)",re.I)),
]
STOP_EXCERPT=re.compile(r"\s+")

def mask(text: str)->str:
    """Remove direct status-marker words so model cannot simply see its silver label."""
    return STOP_EXCERPT.sub(" ",CUE.sub(" ",text.lower())).strip()[:2000]

def grams(text: str) -> set[str]:
    s=mask(text)
    # Character grams, no heading and no path tokens.
    s=" ".join(re.findall(r"[\w\u0590-\u05ff]+",s,re.U))
    return {s[i:i+n] for n in (3,4,5) for i in range(max(0,len(s)-n+1)) if s[i:i+n].strip()}

def training_rows(repo:Path,commit:str,paths:tuple[str,...])->list[dict]:
    samples=[]
    for path in paths:
        for part in sections(load(repo,commit,path),path,commit):
            if part["status"] not in CLASSES or len(part["evidence_excerpt"])<60:
                continue
            samples.append({"file":path,"line":part["start_line"],
                            "truth":part["status"],"body":part["evidence_excerpt"]})
    return samples

class CharNB:
    def __init__(self,examples:list[dict],alpha:float=1.0):
        self.freq=Counter(row["truth"] for row in examples)
        self.total=len(examples)
        if not self.total or any(self.freq[c]==0 for c in CLASSES):
            raise ValueError("Not enough independent training classes in silver headings")
        self.counts={c:Counter() for c in CLASSES}
        for row in examples:
            self.counts[row["truth"]].update(grams(row["body"]))
        self.vocab=set().union(*[set(x) for x in self.counts.values()])
        self.denom={c:sum(self.counts[c].values())+alpha*len(self.vocab) for c in CLASSES}
        self.alpha=alpha
    def predict(self,text:str,threshold:float=.80)->tuple[str,float]:
        v=grams(text)
        if not v:
            return "ABSTAIN",0.0
        # Standard multinomial NB on unique per-sample ngrams, unseen docs held out.
        scores={}
        for c in CLASSES:
            score=math.log(self.freq[c]/self.total)
            score+=sum(math.log((self.counts[c].get(g,0)+self.alpha)/self.denom[c]) for g in v)
            scores[c]=score
        highest=max(scores.values())
        exp={c:math.exp(score-highest) for c,score in scores.items()}
        total=sum(exp.values())
        prediction=max(CLASSES,key=lambda c:scores[c])
        confidence=exp[prediction]/total
        return (prediction if confidence>=threshold else "ABSTAIN",round(confidence,5))

def keyword(text:str)->str:
    # Blind body-only cue lookup on equal input.
    for category,rule in LABEL_RULES:
        if rule.search(text):
            return category
    return "ABSTAIN"

def evaluate(samples:list[dict],predict)->dict:
    tally=Counter()
    per={c:Counter() for c in CLASSES}
    confusion=defaultdict(Counter)
    for row in samples:
        predicted=predict(row["body"])
        if isinstance(predicted,tuple):predicted=predicted[0]
        gold=row["truth"]
        confusion[gold][predicted]+=1
        per[gold]["n"]+=1
        per[gold]["correct"]+=int(predicted==gold)
        tally["total"]+=1
        tally["covered"]+=int(predicted!="ABSTAIN")
        tally["correct"]+=int(predicted==gold)
        tally["false_approved"]+=int(predicted=="APPROVED_CLAIM" and gold!="APPROVED_CLAIM")
    total=tally["total"]
    covered=tally["covered"]
    macro=sum(per[c]["correct"]/per[c]["n"] for c in CLASSES if per[c]["n"])/sum(bool(per[c]["n"]) for c in CLASSES) if total else 0.0
    return {
        "n":total,"covered":covered,"coverage":round(covered/total,5) if total else 0,
        "correct":tally["correct"],"accuracy_including_abstention":round(tally["correct"]/total,5) if total else 0,
        "selective_accuracy":round(tally["correct"]/covered,5) if covered else None,
        "macro_recall":round(macro,5),
        "false_approved":tally["false_approved"],
        "by_class":{c:dict(per[c]) for c in CLASSES},
        "confusion":{k:dict(v) for k,v in confusion.items()}
    }

def run(repo:Path,commit:str)->dict:
    commit=resolve_ref(repo,commit)
    if commit!=PINNED_SHA:
        raise ValueError("Must use the preregistered pinned commit")
    if set(TRAIN)&set(HOLDOUT):
        raise ValueError("Training and heldout documents overlap")
    tr=training_rows(repo,commit,TRAIN)
    te=training_rows(repo,commit,HOLDOUT)
    if len({x["truth"] for x in te})<2:
        raise ValueError("At least two test silver label classes required")
    clf=CharNB(tr)
    most=clf.freq.most_common(1)[0][0]
    methods={
        "char_3_to_5gram_nb":evaluate(te,clf.predict),
        "body_keyword":evaluate(te,keyword),
        "majority_training_class":evaluate(te,lambda _:most)
    }
    candidate=methods["char_3_to_5gram_nb"]
    baseline=methods["body_keyword"]
    useful=(candidate["coverage"]>=.5 and candidate["macro_recall"]>=.7
            and candidate["false_approved"]<baseline["false_approved"])
    return {
      "version":"9.15","component":"COMPASS_NOT_SELF",
      "data":"REAL_PRIVATE_PROJECT_DOCUMENTS",
      "private_source_copied_into_report":False,
      "pinned_commit":commit,
      "train_docs":len(TRAIN),"test_docs":len(HOLDOUT),
      "train_silver_counts":dict(clf.freq),
      "test_silver_counts":dict(Counter(r["truth"] for r in te)),
      "vocab_char_ngrams":len(clf.vocab),"methods":methods,
      "predeclared_exploratory_signal_met":useful,
      "scientific_verdict":"NO_PROVEN_AUTONOMOUS_INTENT_LEARNING",
      "limitations":[
        "All labels are weakly inferred from headings in the same corpus; they are not independent human ground truth.",
        "Small, topically overlapping October 2026 documents are not chronologically unseen future decisions.",
        "This model predicts decision-status heading types from text bodies, not vision/scope/supersession or truth.",
        "No approval or memory record changes ever happen during inference.",
        "Raw private documents, learned ngram features, and per-case text do not appear in this summary."
      ]
    }

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--repo",type=Path,required=True)
    a.add_argument("--commit",default=PINNED_SHA)
    a.add_argument("--output",type=Path,default=Path("compass-v915-private-summary.json"))
    args=a.parse_args()
    try:
        result=run(args.repo,args.commit)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError) as ex:
        print("COMPASS V9.15 refused:",str(ex),file=sys.stderr)
        return 2

if __name__=="__main__":
    raise SystemExit(main())
