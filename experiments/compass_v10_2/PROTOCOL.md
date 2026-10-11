# COMPASS V10.2 — Prospective real-project continuation quality test

**Frozen protocol, 2026-10-12, before the first prospective case is collected.** Aim: directly test whether the evolving **project understanding model** improves GPT-6 responses versus equally-budgeted standard retrieval, not whether our SQLite code works. **NO OpenAI API**.

### Intervention and fair controls

Each actual NEW work request, created after protocol freeze, goes to **four independent new GPT-6 sessions** (to avoid prior project-contamination and prompt carryover) with identical base instructions, same current task, same model effort and at most **5,000 context characters** (counted, provenance included):
- `N` No project memory: only the current task; unknown intent must be admitted.
- `F` Flat memory: current human-approved project statements, no graph/WHY hierarchy.
- `R` RAG: relevant source snippets retrieved from the same authorized historical corpus, with paths and time boundaries.
- `C` COMPASS V10.1: current approved task-oriented WHY / principle / decision / unresolved / provenance brief.

**All four are compared only after independent fresh-session runs**. This experiment does **not** automate GPT-6 chats in the current session, invent independent calls or pay API charges. Manually supplied arm results are required. No arm can read responses from others before finishing. Project data stays private; files containing raw tasks and answers never go into public CI.

### True external targets

At least **12 genuinely later user-created real project decisions/problems**, with independently adjudicated user goals, correct scope, ground-truth evidence and required follow-up. A task whose "answer" was prewritten or already directly seen by GPT-6 teacher as a test answer is disqualified. No post-hoc cherry-picking of tasks. Mixed easy/hard cases must be reported; study spans at least 3 different substantive workstreams.

A reviewer must be blind to the model arm. Prefer **two independent human raters** with adjudication for disagreements. Scoring for each arm per task:
- `goal_alignment` 0..4 (does output preserve user WHY?)
- `evidence_fidelity` 0..4 (is factual grounding/provenance appropriate?)
- `continuation_value` 0..4 (does it advance work as requested?)
- `uncertainty_calibration` 0..4 (does it avoid inventing undecided things?)
- `unsupported_claims` 0..4 (higher = more unsupported confident claims)

**Primary utility** = goal_alignment + evidence_fidelity + continuation_value + uncertainty_calibration − 2*unsupported_claims (range −8 to 16). Primary contrast `C − R`, paired over distinct tasks (reviewer scores averaged within task, NOT counted as independent units).

### Locked research success gate

At least 12 valid independently assessed tasks from 3+ workstreams; mean `C-R` utility >= +2.0/16, **exact two-sided paired sign test p < 0.05** on non-ties, and COMPASS has no greater per-task severe unsupported-claim incidence (`unsupported_claims >=3`) than RAG. Context budgets and task timestamps must be valid for every arm. Reviewer inter-rater disagreement and exclusions logged, never silently suppressed.

Report all arms and every paired score, confidence/calibration failures, negative cases and evidence of tasks for which no memory is safer. Multiple exploratory comparisons are not used to promote the primary claim. A positive initial result is still **one-project provisional evidence**; require replication across other projects, long-term updates and real user feedback to claim broad COMPASS effectiveness.

### Falsification

If COMPASS merely recites goals, loses implementation facts, overtrusts teacher opinions, imports future policy, or triggers unsupported action, count failure. If user intent advantage is below RAG or not significant, conclude `NO_PROVEN_ADDED_VALUE_OVER_RAG` rather than altering thresholds. No model selection or optimization after the holdout labels are revealed.

### Implementation

Public research repo contains only (1) this protocol, (2) offline stdlib Python scorer with fake CI tests, and (3) a blank private-results schema/template. Reviewer submissions and all private project contents remain on user's local machine. No automatic policy approvals, production changes, or persistent task automation.
