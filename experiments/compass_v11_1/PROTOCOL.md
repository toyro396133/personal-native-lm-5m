# COMPASS V11.1 — direct free-text user corrections, independent online feedback

**Protocol frozen 2026-10-12 before examining experimental results.** COMPASS is independent of SELF. No GPT-6 teacher, other hosted LLM, API, automatic grading by a foundation model, or SELF checkpoints.

## What is being tested

Can a *standalone trainable COMPASS learner*, using only a user's explicit free-text corrections to earlier interpretations, gradually prefer intent-consistent alternatives in a different later situation? The central product goal remains continuity of **project WHY**, not merely free-text status classification.

**This first iteration is intentionally narrow**. A deterministic high-precision reader converts only **literal explicit contrastive responses** `לא X אלא Y` (excluding `לא רק ... אלא גם`) and explicit `טעות! Y` replies referencing a prior written proposal into an eligible **correction pair**: rejected vs preferred. Corrections cannot confer authority/approval over project policy; they update an experimental personal or project-specific weight state only. Ambiguous statements, open questions, negation without a replacement, hedging, or implicit authority transfer remain evidence-only/REVIEW_REQUIRED and do not update learned weights. Every training text is either directly user-authored or the written rejected proposal; no teacher rewrites.

User inputs (private local JSONL) preserve `id`, project/scope, source Git SHA and precise line span, task/question, previous proposal, original user reply and optional source verification flag. Per-project weight vectors never cross to other projects or personal state without explicit review.

## Locked learning procedure

- Pure stdlib Python; reuse V11's UTF-8 Unicode hashed 3-gram features and pairwise SGD `LR=0.1`, `DECAY=0.0001`; keep one independent sparse weight vector per project/personal scope and separate raw evidence.
- Observe: score the potential correction pair against current weights **BEFORE** updating from the current reply, then update on the literal `preferred - rejected` pair. Never use the reply to manufacture the *current task* features at inference time; offline correction-pair probes created after a reply are explicitly labelled **RETROSPECTIVE_PAIR_RANKING**, not autonomous answer generation.
- At least two previous eligible corrections in same scope to emit a ranking; abstain on small-margin outcomes, missing source, unclear raw reply, or unknown scope.
- Controls: frozen initial vector, lexical retrieval over past corrections (same data), shuffled-correction-feedback negative control (only on lab synthetic), scope-swap and candidate-order invariance.
- No prior V11 choice-label-only episodes leak into the new correction learner unless explicitly imported as source-verifiable user feedback with a separate pre-registered mapping (not in V11.1).
- Rejected and preferred phrasing are not necessarily mutually exclusive entire project principles; changing learner preferences must NOT revoke an approved project fact. Human review is needed for that.

## Honest held-out criteria

- Dev toy examples are **engineering only**. Report prequential coverage, margin, accuracy on eligible later correction pairs, and retrieval/frozen baselines **with eligible denominator and zero-result handling**.
- Real project log is *not* suitable for automatic broad labelling. Re-scan a pinned private Chacham Lev Q/A log; **only exact user-authored excerpts** and their context are eligible. A source match does not imply that its recorded interpretation or approval is correct.
- **No cognitive research win** from 2–10 handcrafted corrections, or from detecting explicit "לא/אלא" wording. At least 20 independently source-reviewed correction pairs from multiple unrelated project topics, plus a separate future-decision evaluation of underlying WHY/rationale, are required before claiming longitudinal intent learning. No score manipulation after seeing outcomes.
- Project private text and local learned user weights must **never** be committed to public repo.

## Outcomes

Successful CI means `FREEFORM_FEEDBACK_ENGINEERING_VALIDATED`, not `UNDERSTANDS_USER_INTENT`. If insufficient real explicit correction episodes are found, record `INSUFFICIENT_REAL_CORRECTION_FEEDBACK`, stop claiming scientific gain, and prioritize user-guided evidence capture for open-ended replies.
