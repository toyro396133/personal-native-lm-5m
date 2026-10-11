# COMPASS V11 — Standalone lifelong project learning (first honest experiment)

**Frozen before implementation/evaluation, 2026-10-12.**

## The actual question

Can **COMPASS itself**, without SELF and without an external language-model teacher, incrementally learn from authentic user feedback during work on a project, preserve project WHY and prior corrections, then give useful support on the next **previously unseen** user decision? A durable journal or keyword classifier alone does NOT satisfy the question.

**Project boundary:** COMPASS and SELF are independent parallel research tracks; this work depends on no SELF checkpoints, and no GPT-6 or other hosted model is used for labelling, supervision, inference, or grading. An outside assistant can write the engineering code, but its opinions are NOT gold labels.

## V11A experiment (bounded proof-of-concept, not full product)

Use source-verified **decision episodes**, each containing a question, alternatives (exact text if the source provides them), a user reply/choice and its rationale as written or an explicit absence flag. Preserve the exact origin, ordering, original response, ambiguity, project scope, and historical context.

1. Observe incoming question **before its user answer**. Make an *out-of-sample chronological prediction* (or abstain), and log it.
2. User's **real answer** is feedback. Only an unambiguous reply that explicitly chooses an available alternative can train the supervised online learner. A “yes”, “no”, “maybe”, corrections to flawed question, or unclear one-character option reference without original alternatives must NOT be invented into a label.
3. Train a small independent CPU online learning module on **earlier answered episodes only** (pairwise choice features) and preserve separate personal and per-project state. Its weights are learned and persisted, unlike a static rules catalogue; raw evidence stays separate.
4. Next episode: predict before feedback, with source traces to earlier relevant evidence; “ABSTAIN” on insufficient support. No automatic change of approved project principle.
5. Baselines on identical input/history and temporal cutoff: no-memory majority / recency option, independent basic retrieval over previous explicit decisions, frozen-weight control, and learned COMPASS. Include shuffled feedback, swapped project and irrelevant-history stress.
6. Report coverage, conditional accuracy, wrong-confident predictions, reasons/source quality, project isolation, forgetting/correction handling, chronological cohort and computational cost. Each error remains in the report.

## Honest evidence constraints

- The PRIVATE Chacham Lev question/answer log has **52 T00–T51 units** at pinned Git commit `fe9863278a04dedf27a0c4fcb8cca56b91ec4544`. Some answers are “א/ב” while other episodes are open-ended or expressly correct the question. **52 episodes does NOT imply 52 valid supervised training labels.** Build a LOCAL-only manifest marking `eligible / ambiguous / open-ended / unanswered`. No public source text or donor/person data.
- No backfilling later Oct-11 outcomes into earlier inference. Never test on cases whose answers the learner already used.
- The loader must accept a simple user-local JSONL with explicitly curated multiple-choice episodes, no GPT-6 interpretations and no guessed rationale. A curator must record each direct source span and whether alternatives are actually provided. This annotation is **eligibility checking**, not making up labels.
- Until at least 20 eligible *direct-user* chronological prediction cases spanning multiple separate topics exist, declare the real-world result **INSUFFICIENT_REAL_DECISION_FEEDBACK**, without making an accuracy claim. Independent curation should ultimately validate applicability, meaning and WHY; narrow option prediction cannot by itself establish deep comprehension.
- Model selection / feature tuning only on training or separate development cases, never on test outcomes. Synthetic engineering tests are NOT scientific evidence.

## Predefined algorithm and minimal engineering

- Pure Python 3.11+ stdlib, no network, GPU, proprietary pretrained checkpoint or SELF imports.
- Stable hashed Unicode character 3–4-grams (blake2b), sparse linear pairwise feedback learner; project-scope isolations; a trainable project weight vector; raw decisions with source attached for read-only audit; optional explicit personal events are isolated (NOT automatically learned from projects).
- Online conservative logistic choice feedback with **fixed** learning rate 0.1, small L2 decay 0.0001, minimum **2 prior explicit labelled events** before emitting any recommendation; report coverage and exact loss. Prediction scores are uncalibrated margins, not probabilities or permission to act.
- Do not pretend feature strings are a "semantic" multilingual understanding engine. If retrieval, recency or majority beats learner on heldout decisions, accept the negative result.

## Outcome

A successful engineering implementation proves only source-audited sequential online updates and correct experimental no-leak evaluation. A research gain requires longitudinal verified real data, heldout performance exceeding strong baselines, and demonstrated correction/WHY transfer. **No claim of autonomous project-vision comprehension yet.**
