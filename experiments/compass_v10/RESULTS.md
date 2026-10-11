# COMPASS V10 — Project Understanding Model (first working exploratory prototype)

**2026-10-12 — GPT-6 teacher in ordinary ChatGPT, no OpenAI API, no changes to SELF.**

## What actually exists

A local, source-linked **project-understanding context** was built from **31 pre-Oct-3 Chacham Lev teacher candidates** (at pinned commit `55b0ac175513`). It has:
- **5 WHY-level project goals** and the relationships between them.
- Source-cited documented principles, **unresolved policy facets**, proposed implementation, observed historical code and superseded statements kept **separate**.
- **12 explicit GPT-6-authored hypothesis rules** for structured proposed changes, returning SUPPORT / CONFLICT / ABSTAIN with evidence traces and a review-required flag.
- The prior tested V9.24 SQLite relation journal is included as **20 pending claims + 10 pending relation proposals, zero approvals**; later material is **not** backfilled into pre-Oct-3 rules.
- A generated **offline Hebrew RTL HTML explorer**, downloadable from the V10 conversation, with goal hierarchy, evidence browser and interactive *structured* product-change evaluation. This is *not* free-form autonomous natural-language comprehension.
- Offline local stdlib Python scripts `compass_v10.py`, `v10_context.py`, `build_explorer.py`, `test_v10.py`. The full local executable source and private model artifact are distributed to the user as separate conversation attachments, **not copied into this public branch**. No private donor/student/user text or other private raw project documents appear here.

## Reported measurements — separate engineering from science

| Method / test | Result | Valid interpretation |
|---|---:|---|
| 19 Python engineering tests | **19 PASS** | Program behavior and safety invariants run locally |
| Offline explorer JS syntax | **PASS** (node --check) | Parses, not an independent live-browser UX test |
| V10 teacher-built 16 rule-aligned cases | **16/16** | **Sanity test with answer/rule leakage by experimental design**, not generalization |
| Extra 12 *frozen-before-use* red-team prompts | **8/12** | GPT-6 teacher-created, NOT blind or independently scored |
| Retrieval only, TF-IDF top 3 | **7/12** locate at least one expected earlier teacher source | **Source retrieval only**; not decision accuracy and not a fair GPT-6 RAG comparison |
| No-memory baseline (abstain on all) | **8/16** | Shows that doing nothing is safe on unknown issues but misses known historical constraints |

Four extra stress errors **ST-01, ST-10, ST-11, ST-12**:
1. Missing priority between a confirmed constraint (students must not be reduced to one feature) and unresolved feature weights; abstained where teacher said CONFLICT.
2. Did not generalize the requirement to preserve decision rationale to a new student-assignment case.
3. Did not generalize the year-over-year knowledge goal to a new situation.
4. Did not generalize legacy code ≠ future product authority sufficiently.

**No post-stress tuning** was done to inflate 8/12. Both test sets were authored with knowledge of source documents by GPT-6; they do not establish independent competence.

**Important**: No four-arm GPT-6 trial was run. Same-model / same-budget blinded inference with (1) no memory, (2) retrieval, (3) structured flat facts, and (4) V10 project model remains a future experimental requirement. The V10 engine is **rule-authored by GPT-6**, not an autonomously neural-trained learner. It asks for review and never edits the application or approves a policy.

## A genuine capability demonstration

For a *structured* request to form new annual student groups based on current preferences **while keeping physical route identity and history**, V10 cites three independent earlier principles and explains why the two temporal views should coexist. For a question on later-only telephone handoff, V10 **abstains** rather than leaking Oct-11 decisions into Oct-2 training. Subsequent candidate updates are visible in pending history but not active.

The result is a tangible `PROJECT_UNDERSTANDING_PROTOTYPE` with traceable values and safe unknown boundaries. **Research verdict:** `NO_INDEPENDENT_COGNITIVE_GAIN_PROVEN`.

## What to test next — not more simple sentence classification

Have an independent task curator create fresh, real product decisions *after the context is frozen*, with explicit user-confirmed answer/rationale and scope; give the **same GPT-6 model** matched inference budgets and identical immediate tasks under four information setups. Evaluate design decision correctness, user-intent/rationale preservation, unsupported-claim rate, overconfident revision risk, and reduced re-explanation in a later work session. Keep projects separated and test personal-model promotion only with explicit review. No API spend is assumed; within-chat four independent blind model runs are **not available from the current session**.

[Locked protocol](PROTOCOL.md). Full results report (Hebrew) is a per-conversation local file; this public repository contains aggregate safe findings only.
