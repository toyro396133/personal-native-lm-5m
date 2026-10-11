> **SUPERSEDED / WRONG ARCHITECTURAL ASSUMPTION — 2026-10-12**
>
> The assertion below that COMPASS *is a subsystem of SELF* is **incorrect and explicitly retracted**. The owner clarified that **COMPASS and SELF are two independent, parallel development projects**. Their future connection is optional, not architectural ownership. Do **not** use the V11A SELF-native mandatory-integration plan below as the active research direction.
>
> **[Read the current canonical two-track charter](../PARALLEL_COMPASS_SELF_CHARTER.md).**
>
> Material below is retained **only as an audit trail of the superseded proposal**, not as instructions for research or implementation.

---

# SELF-native COMPASS — canonical research direction (2026-10-12)

**Owner decision / immediate correction:** COMPASS is a subsystem of the user's own SELF model research. It is not an add-on intended to make an outside hosted assistant better, nor a teacher-student project using a hosted language model. **No GPT-6, OpenAI API, or other proprietary external foundation-model dependency is part of the COMPASS training, inference, supervision, evaluation or deployment protocol.** External assistants can assist with writing/debugging repository code only; their generated judgments are not scientific training labels, reference answers or model inferences.

This document supersedes the V9.13–V10.2 line of inquiry that put GPT-6 at the centre of COMPASS, without erasing its historical negative results. The externally supported work is not the target; retain only reusable engineering pieces (source-scoped journal, revision checks, project identity, approval controls, test fixtures).

## Correct system boundary

```text
                    ONE USER
                  +----------+
                  | PERSONAL |
                  | COMPASS  |
                  +----+-----+
                       | selectively transferable abstractions
      +----------------+-----------------+
      |                                  |
 +----v------------------+     +---------v---------------+
 | PROJECT COMPASS A      |     | PROJECT COMPASS B         |
 | project-specific WHY   |     | different goals/rules     |
 | reasons, outcomes,     |     | corrections, trajectory   |
 | provenance, validity   |     | provenance, validity      |
 +---------+--------------+     +-------------+------------+
           | SELF-native conditioning/selection |
           +-----------------+------------------+
                             |
                +------------v---------------+
                | SELF MODEL (5M first, then  |
                | larger SELF checkpoints)   |
                | inference + trainable online|
                | adaptation where validated |
                +----------------------------+
```

COMPASS has THREE different functions and should not collapse them:
1. **Evidence memory**: raw events, transcripts/doc fragments and observed results, provenance, source/time/scope; not equivalent to beliefs or the user's intention.
2. **Persistent learned state**: a trainable compact personal state, with *separate* per-project COMPASS learned states; longitudinal online updates from project work. Stable cross-project abstractions may be proposed to personal state only after scope/permission review.
3. **SELF-native read/control pathway**: SELF receives request-conditioned relevant state and verified context; existing SELF personal state path must remain compatible.

Do not mandate that ALL COMPASS data fit in the model's main weights. A native learning system can use a trainable side state and auditable external evidence, provided no hosted model is required to do its thinking. The architecture should remain capable of direct coupling to SELF's internal representations, not merely generate a prose prompt.

## Existing compatible code, not speculative new infrastructure

- `README.md`: main LM and persistent per-user micro-model with runtime conditioning; frozen backbone + changing personal model is an established prototype lifecycle.
- `PERSONAL_STATE_ABI.md`: 228-dim state (core 64, policies 64, world 64, routing 32 and confidence values) and learned 116-dim per-user latent; adapter/controller pathway already exists in repository.
- `model_self_variants.py`: SELF adapter variants (e.g., self_v1, diff_anchor, projected_diff) are candidates for *measured* integration. Do not assume which wins in V11 without looking at completed runs.
- Prior COMPASS SQLite source-scoped memory is a review/event store; it is NOT itself the native trainable COMPASS and cannot stand in for model improvement.

**Architecture requirement:** first integrate with existing PersonalState ABI/controller and a project-specific learned state/mask, while controlling total parameter count, base LM quality and cross-project interference. Compare both standalone 5M and subsequent checkpoint capacity when feasible. Any new ABI requires a migration/compatibility benchmark rather than silent rewrites.

## Immediate falsifiable experiment — V11A (self-native, no hosted teacher)

1. Freeze a known reproducible SELF checkpoint and tokenizer; snapshot its weights, training split and usable CPU/GPU budget. Start at ~5M where possible. An old experimental model without useful language ability cannot meaningfully prove full vision comprehension; report structured consistency separately from open-ended reasoning.
2. Use ACTUAL user-authored decision events or independently human-labelled project changes with provenance. Learn on earlier sessions/time, test on genuinely later, unseen project work. Preserve contradictions, abstention, rationale and whether an apparent implementation change is not a change to user intent.
3. Compare at IDENTICAL SELF weights/budgets: **A** no persistent state, **B** ordinary retrieval from the same allowed history, **C** personal SELF micro-state only, **D** personal + separate learned project COMPASS. Include swapped-project state and shuffled-timeline controls to expose leakage and shortcut learning.
4. Train only necessary local SELF-native components (project state and a small state learner/readout, or interface with controlled frozen-backbone adaptation). No hosted-model annotations, target generation, inference, grading or API. Any labels mined by string heuristics must be explicitly described as weak labels.
5. At *minimum* measure: held-out project intent relevance, retention after unrelated sessions, correct rationale/validity, true correction vs implementation-only change, cross-project isolation, abstention/false confident policy claim, and base language quality/forgetting. Include counts, uncertainty and simple baselines.
6. Only claim a breakthrough on externally verified new events with statistically robust improvement over retrieval and micro-state baselines, and with no degradation in project isolation or base LM. Unit tests and evaluator labels alone do not establish cognitive ability.

## Non-negotiable constraints

- SELF model is the learner/decision maker. COMPASS is an ownable, persistent, SELF-native component and may train its own parameters.
- Personal COMPASS and each per-project COMPASS are separate but linked *selectively*. Never silently promote project-specific rules to personal policies.
- Sources, status and human approval remain outside the learned latent as auditable evidence/authority controls.
- Experimental imports cannot automatically approve or revoke user-authored project principles.
- No external LLM in the architecture, training or benchmark. This prohibition covers GPT-6 used as a teacher, the answerer in an A/B study, or an automatic judge.
- Keep historical V9/V10 material labelled historical, *not* the active roadmap. The next new COMPASS experiment must test **SELF itself**.
