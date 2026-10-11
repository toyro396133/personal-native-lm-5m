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
