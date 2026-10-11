# COMPASS — research charter and source of truth

**Current project focus: COMPASS, NOT SELF model training.** Last reaffirmed 2026-10-11.

COMPASS is a system for persistent, adaptive understanding of a person and the evolving purpose of their projects. It is **not** merely long-context summarization, search over code, or a new training experiment on the underlying SELF language model. Experiments may use small frozen encoders and trainable readouts to validate the COMPASS architecture, but do not imply changes to any main LM weights.

## Product boundary

- **Personal COMPASS:** durable global preferences, decision principles, objectives and learning from work across projects. A project can contribute knowledge to the user's personal COMPASS, but only after verifying that it is applicable beyond that project.
- **Project COMPASS:** a separate persistent instance PER project, connected to the personal COMPASS. Learns project vision, intended outcomes, guiding principles, context, decisions and **why** decisions were made. A project's current code is a mutable implementation, NOT the authoritative or complete representation of its purpose.
- **Source and validity:** every learned principle should ultimately carry provenance, owner/scope, explanatory rationale, evidence, applicability, confidence, effective dates where relevant, and explicit supersession/revocation. Uncertain or contradictory observations require clarification/approval, not confident unverified overwriting.
- **Decisions:** choose actions aligned to active personal/project principles while keeping verified binding commitments in a **separate authority/constraint control plane**.
- **Historical improvement:** preserve useful work across session boundaries and years without allowing new irrelevant activity to erase old still-relevant principles; generalize between projects only when scope/conditions justify it.

## Research trajectory and concrete lessons

- V9.5–V9.6: text-only classification of binding requirements and dismissed instructions is not reliably valid/specific; do not treat wording as verified authority.
- V9.7: mocked signed provenance and explicit abstention reliably separate an external policy controller from learned preferences in synthetic cases. This is systems plumbing, not neural legal understanding.
- V9.8: isolated user/project numeric readouts help, but erroneous text-only ownership routing can still leak the wrong project's state.
- V9.9: explicit mocked task-origin metadata fixes synthetic routing, but its perfect score reflects author-supplied ownership and narrow binary tasks; not proof of human intent learning.
- V9.10: five-event histories with added contextual conditions collapse the short-history recurrent preference memory. Retrospective reordering recovers much performance, strongly implicating recency order.
- V9.11: explicit, versioned, provenance-labelled COMPASS concept slots prevent unrelated later context from overwriting the relevant preference, and preserve typed active/revoked obligations via a separate mock controller. **A trivial first-three-statement shortcut matches the slot accuracy on test fixtures**; reliable extraction, editing and applicability of real principles are still unsolved.

## Operational milestone — COMPASS V9.12 / V9.12b (2026-10-11)

- **First executable durable COMPASS component**, not only synthetic neural benchmarks: dependency-free Python/SQLite personal and per-project principle store, explicit approved revisions, source and rationale, applicability, journal replay, hash consistency checks. It is local and explicitly not production-authenticated.
- **Local Hebrew RTL review UI** with propose/review/approve/reject/resolve and local HTTP session/Origin/Host checks. Acceptance workflow [V9.12b](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38131221789): **36 distinct tests PASS across Python 3.11/3.12/3.13**, and end-to-end demo PASS on each.
- [Code and Windows instructions](https://github.com/toyro396133/personal-native-lm-5m/tree/experiment/compass-v9-12/experiments/compass_v9_12) · [Full engineering report](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v9-12/experiments/compass_v9_12/RESULTS_V9_12B_UI.md).
- **Crucial distinction:** operational durable memory is `PASS` on tested scenarios. Automatic autonomous understanding/extraction/updating of project vision and personal principles from real conversations remains `NOT_YET_LEARNED`; no independent human multilingual benchmark or real identity verification. This is **COMPASS**, not work on the SELF neural model.
- Until the cognitive learning gate is independently verified, do not label storage/UI success a scientific breakthrough or deploy as a trustworthy autonomous memory.

## First real private-project evidence pilot — COMPASS V9.13

- The research has moved beyond only synthetic cases: read-only analysis of a *private real project* at a fixed historical 2026-10-02 Git snapshot fitted a lightweight multilingual token TF-IDF retrieval index on **8 real planning/design documents (242 segments / 3,638 distinct terms)**. No private raw source is included in this public repository.
- Initial researcher-authored historic decision probes found the expected old source in top 3 for **6/6**, but the exact same probes also scored **6/6** with plain lexical overlap. This is a **retrieval result, NOT a demonstrated cognitive learning gain**. Later-decision novelty evidence remains exploratory and researcher-labeled.
- Nine project-principle summaries from earlier and later documents have been prepared **as review-pending candidates only**, with source refs; no project principle approved automatically, no production data mutated.
- [Experiment protocol and limits](../../experiments/compass_v9_13/EXPERIMENT_V9_13_REAL_PROJECT.md) remain on an experiment branch; reproducible local runner and manifest must run on the user's own private project checkout. Do not upload raw content or generated private local index/results publicly.
- Next threshold for an actual scientific breakthrough: independently judged withheld project decision episodes, robust discrimination of existing/new/revoked principles, correct rationale and applicability, plus strictly better accuracy than equal-source simple lexical/retrieval baselines. `NO_PROVEN_INTENT_LEARNING_GAIN` until passed.

## Real-project COMPASS learning attempt — V9.13–V9.15 (2026-10-11)

- Read private **Chacham Lev** project documents at pinned Git revisions without copying raw private documents, donor/student records or model features into the public COMPASS repository.
- V9.13: lexical and TF-IDF both recovered 6/6 obvious historic source documents on researcher-written probes. **No gain over lexical baseline**, no valid blind intent-learning claim.
- V9.14: observed 302 real Markdown headings in six private planning/interview documents, with only 16 explicit claim statuses (10 approved claims, 4 proposal claims, 2 historical; the rest uncertain). A document titled as a proposal held eight explicitly approved subsections. Built source/line-anchored conservative claim miner that never turns text into authorized policy or autoapproved memory. [V9.14 engineering run](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38133226742).
- V9.15: attempted simple supervised decision-status extraction from body text without leaking heading labels. Fixed eight-document TRAIN corpus had **5 approved, 0 proposal, 2 historical** weak-labelled examples; classifier correctly refuses to train. **`INSUFFICIENT_REAL_SILVER_EVIDENCE`**, not a success to extrapolate.
- Added offline Hebrew local private evidence-review page for independent human labels, and bridge from real private source evidence into the V9.12 COMPASS store as **PENDING_REVIEW** only (no autoapproval, isolated project scope, audit source). **29 synthetic engineering tests per Python 3.11/3.12/3.13 = 87 passed** in [V9.15 workflow](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38133941788).
- [V9.15 branch and local private workflow instructions](https://github.com/toyro396133/personal-native-lm-5m/tree/experiment/compass-v9-15/experiments/compass_v9_15).
- **Research truth:** COMPASS can now store reviewed historical project principles and safely prepare source-grounded candidate updates from genuine private documents. It **cannot yet be said to autonomously learn the user's project vision, reasoned corrections and validity conditions** on independently judged heldout real conversations. The next scientific blocker is acquisition/adjudication of a diverse *private human-labelled corpus*, evaluated beyond lexical baselines.
- Do not confuse externally structured metadata and manually approved policies with comprehension learned by a neural model. COMPASS remains the object of research; SELF weights and 360M adapters are out of scope.

## V9.24 — source-anchored decisions, GPT-6 teacher, first relation learner (2026-10-11)

- GPT-6 in ChatGPT (no API) annotated ten real source-anchored old/new decision pairs from Chacham Lev. All **20** source span anchors were checked against the private original log; teacher classifications were **not** independently human verified.
- A leave-one-concept-out small relation student (2 REVISES, 2 QUALIFIES, 6 EXTENDS teacher labels) scored **6/10** and **0/2 recall on crucial revisions**, exactly matching the majority EXTENDS baseline. A frozen simple keyword baseline scored **7/10**, 1/2 revision recall. **No demonstrated learning advantage**.
- An independent offline Python/SQLite **proposed relation graph** stored **20 pending claims + 10 pending relation edges, 0 approvals**; 18 tests passed including source and scope validation, stale version check, facet-specific correction, non-destructive extension, restart replay and chain corruption detection. This is an engineering proof-of-concept, **not** an authenticated production system.
- Prepared a **blind Hebrew offline review** of those pairs that hides all teacher labels, teacher rationale and student predictions. No independent reviewer has submitted results yet. Need independently labelled *real user statement* pairs, more actual corrections, scope and applicability, chronologically unseen material and equal-information baselines.
- [Predeclared protocol](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v9-24-relations/experiments/compass_v9_24/PROTOCOL.md) · [honest measured report](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v9-24-relations/experiments/compass_v9_24/RESULTS.md).
- Scientific status **`NO_PROVEN_INDEPENDENT_RELATION_LEARNING`**. Do **not** allow autonomous deletion or supersession of any approved user/project principle.

## Priorities after V9.11

1. **Learn/write/maintain concept records from actual event streams**, not only read gold-labelled concept slots: corrections, new principles, different scopes, reasons, conflicts and revocations.
2. **Interleaved multi-concept long-horizon evaluation**, with position shortcuts made insufficient and realistic project work, not only author-generated short three-line evidence.
3. **Hebrew and multilingual semantic competence**, heldout genuinely authored cases and independent human labels; report personal/project scores separately.
4. **Proper permission, source authentication and ambiguity handling** for production-grade use; test uncertain ownership and source reliability.
5. **Demonstrate real across-project personal learning** without promoting project-local commitments to global principles.
6. Measure equal-information baselines, cases and compute; preserve preregistration and honest NO-GO decisions, no premature integration with underlying SELF LM.

## Working convention

- Scientific code stays in `experiment/compass-v*` branches pending review; CI workflow triggers on main are not production deployment.
- Document protocol *before* first cold prediction, preserve all seeds and failures, compare baselines on the same cases and state explicit limits.
- Use the term **COMPASS** for this effort. Mentions of SELF should only clarify the boundary or a possible **future optional integration**, not redefine the current objective.
