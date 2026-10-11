# COMPASS V9.15 — Weak-supervised *REAL* Chacham Lev decision-type learning, with a locked independent document split

**Preregistered before running this V9.15 model on the private holdout documents.** This experiment is about COMPASS's ability to read source evidence, not SELF native language-model training.

## Hypothesis

A tiny language-derived classifier trained on **body text only**, with its decision-type labels weakly supplied by *explicit section headings* in historical project documentation, may recognize status patterns in held-out documentation. This is intentionally a modest first attempt at automated COMPASS interpretation. Success on heading-derived silver labels would **NOT** constitute independently verified human-intent learning.

## Fixed data, trust, privacy

- Private local/source repository: `toyro396133/chacham-lev`, fixed commit `fe9863278a04dedf27a0c4fcb8cca56b91ec4544` (not a temporal holdout here — DOCUMENT-level held-out split). No raw source docs, text features or trained weights containing project content in public COMPASS repo or CI.
- TRAIN documents fixed:
  1. `docs/product/importance-map.md`
  2. `docs/campaigns/telephone-fundraising-qa-log-2026-10-11.md`
  3. `docs/campaigns/telephone-assignment-and-relationship-unit-design-2026-10-11.md`
  4. `docs/campaigns/telephone-fundraising-operational-research.md`
  5. `docs/campaigns/operational-interview-followup-42-46.md`
  6. `docs/campaigns/operational-question-audit-index.md`
  7. `docs/campaigns/cha-pak-assembly-debrief.md`
  8. `docs/campaigns/operational-interview-live-notes.md`
- HELD-OUT documents fixed:
  1. `docs/campaigns/telephone-call-outcomes-and-statuses-proposal-2026-10-11.md` (critical mixed proposal-title and approved subheadings)
  2. `docs/product/interview-documentation-protocol.md` (explicit binding protocol)
- Samples: each Markdown section with a standalone heading meeting one of **three** labels `APPROVED_CLAIM`, `PROPOSED_CLAIM`, `HISTORICAL_EVIDENCE`, and at least 60 characters of *body* before the next heading. Sections labelled `UNRESOLVED` are *unlabelled*, never treated as true negatives. Other categories are out of scope.
- **Heading label is a SILVER label, not a true verified decision.** Gold and features are separated: train/eval examples use **body only**, not heading; strip leading status-marker labels (explicit 'מעמד מאושר', 'מוצע', 'הצעה' etc) from body before n-gram tokenization. The model cannot see heading or filename.
- No fine-tuning after holdout results. Locked `char ngram 3–5`, Multinomial Naive Bayes with Laplace alpha=1, lowercase, class priors from empirical TRAIN distribution (no hyperparameter search). No transfer of learned weights between project scopes.
- Predict only if maximum posterior >=0.80, else `ABSTAIN`. Report heldout coverage, selective correct, macro class recall and false `APPROVED_CLAIM`; **never allow predicted approval to mutate the store**. Posteriors from tiny NB are not calibrated truth likelihood.
- Equal-input baselines:
  - majority TRAIN class always (no abstention)
  - simple rule matching status cue occurrences in **the same body**; if no cue => ABSTAIN
  - no-heading document-title classifier is an unfair low baseline and should not be treated as scientific improvement.
- No external pretrained model, no internet inference or 360M model work. CPU and Python stdlib.
- Failure if missing any TRAIN label class or fewer than 2 test classes: scientific validity `INSUFFICIENT_REAL_SILVER_EVIDENCE`. Even if classification works on 12–20 weak labels, **scientific verdict remains `NO_PROVEN_AUTONOMOUS_INTENT_LEARNING`** pending independent human-judged transcripts and new approval/correction tasks.

## Tests and success interpretation

- Engineering: tests on purely fictitious documents for leakage (classifier never sees headings), standard held-out split, private-file allowlists, abstention, deterministic predictions, per-class reports, no autoapproval.
- Exploratory accuracy metric only: exact silver labels on the fixed two heldout documents and equal-information baselines. Report all classes, not cherry-picked.
- An interesting positive signal would be >=0.70 macro recall at >=0.50 coverage **and** lower false-approval rate than body-only keyword baseline. This is a postulated research target, not a claim of dependable real-world approval. Any model that fails should be reported as failed without retuning.
- Real breakthrough still needs: independent human labels (including statuses not signaled in headings), temporal unseen project decisions, detection of replacements and applicability, and meaningful win over simple baselines.
