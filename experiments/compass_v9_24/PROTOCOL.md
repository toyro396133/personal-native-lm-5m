# COMPASS V9.24 — Real decision relation learning and safe temporal graph

**Frozen before running the relation student.** Project: COMPASS, not SELF. GPT-6 provides *provisional teacher labels* grounded in user-decision records; never authenticate or autoapprove on its own.

## Data

Ten already-compiled paired historical records (V9.24, `COMPASS_V924_REAL_USER_DECISION_RELATIONS_PENDING.json`) from the PRIVATE Chacham Lev decision log, pinned Git commit `fe9863278a04dedf27a0c4fcb8cca56b91ec4544`. Each case supplies old/new human-readable explanations, `Tnn` source pointers and an unverified GPT-6 proposed relation. The explanations and teacher reasons were written with knowledge of the outcome; hence they can **never** be described as untouched blind prediction targets. No private raw user replies may be uploaded to this public repo.

Primary coarse classes, frozen:
- `REVISES`: `SUPERSEDES_INTERPRETATION`, `SUPERSEDES_TIMING` (2 cases).
- `QUALIFIES`: `STRENGTHEN_QUALIFIED`, `QUALIFIES_CLOSURE_CONDITION` (2).
- `EXTENDS`: other six add-next-stage, responsibility, separate-rule, status refinement and scoped design-gate cases.

## Learned student's true inputs

Only `old_explanation` + `new_explanation` separated by `[NEXT]`. **Do not use** `proposed_relation`, `teacher_reason`, `concept_key`, numeric event IDs, source path or test labels as features. The student sees model-generated paraphrases, not raw user speech. The full ten cases have been viewed by the teacher beforehand; no more blind claim.

**Predeclared method:** leave-one-**concept-key**-out cross-validation, 10 folds (each test item concept occurs in no training fold; there are ten unique keys). `TfidfVectorizer` character 3–5 grams on lowercased input, `LogisticRegression(C=1.0,class_weight=None,max_iter=1000,random_state=24)`, all folds fixed. One engineer-designed conservative lexical-rules baseline sees exactly the same two explanations; baseline only predicts revision on explicit "הוחלף/מחליף/במקום/מבוטל/טעות/מתקן" in NEW explanation; otherwise `EXTENDS`, with `QUALIFIES` if "מסויג/אבל/מבהיר/מחדד/בהתאם/תנאי" appears. Other baselines: ALWAYS_EXTENDS; TF-IDF 1-nearest-neighbor trained on nine pairs in each fold. Report 3-class correct/10, macro-F1, `REVISES` precision/recall and false `REVISES` alerts. `REVISES` is never an approved destructive action.

**No retrospective tuning:** Do not alter class grouping, examples, training features, rule keywords, methods or thresholds after seeing test outputs. This study is exploratory because labels were constructed by the same teacher that selected examples. A result above baselines may only be called `PROVISIONAL_TEACHER_LABEL_RETRIEVAL_SIGNAL`, never an independently verified scientific result.

## Persistent decision graph engineering

Build an auditable, offline `RelationProposalGraph` that emits **PENDING** edges with sources, relation category, confidence (unvalidated) and audit reason, without changing an active principle. Human review can approve an edge; only an explicitly approved `REVISES` edge marks an earlier claim as superseded **within its stated facet**. When either scope, source, earlier claim or approval evidence is absent, operation must abstain and not destructively update memory. `EXTENDS` and `QUALIFIES` never revoke the old claim. Every edit requires explicit expected-version CAS and approval actor; no acceptance for duplicate/stale edges. History replays across local SQLite restart. Connection to production V9.12 memory should be proposals-only until separately reviewed.

## Validation and honest scientific gates

- At least 10 invented engineering tests with negative cases: no auto-approval, traceability, scope isolation, stale revisions, duplicate approvals, exact facet correction, preserving older records, preserving distinct approved rules, source verification and reopen.
- Student LOCO test must disclose per-class performance and any false destructive recommendations; insufficient relation diversity for robust neural evidence.
- User-facing report must include **NO_PROVEN_INDEPENDENT_RELATION_LEARNING** absent separate human-adjudicated genuinely held-out change events and >=3 different projects. Never repeat V9.23 teacher-label p-value as real understanding.
