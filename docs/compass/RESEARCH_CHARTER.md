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
