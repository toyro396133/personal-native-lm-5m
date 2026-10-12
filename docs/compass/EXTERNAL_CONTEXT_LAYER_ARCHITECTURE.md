# COMPASS — External, portable learned context layer (canonical architecture)

**Authoritative owner clarification, 2026-10-12.**

COMPASS and SELF are **two independent parallel developments**. COMPASS is externally attached to a **capable large language model (LLM)**, which performs actual broad-language understanding and reasoning. COMPASS itself is a compact adaptive subsystem that learns/maintains/organizes **what the large model needs to know about this user and this project**. Its weights and durable data are **outside** the large model; it is neither merged into its backbone nor defined by the large model's identity. COMPASS should be compatible with **multiple interchangeable large models**, not tied to GPT-6 or SELF.

**Two different meanings of independent — do not confuse them:**
1. **Deployment/learning independence:** COMPASS has its own storage, state, versioning, updater and possibly compact trainable parameters; it can persist even when the large model is replaced. YES.
2. **Independent full language comprehension:** COMPASS must itself read arbitrary complex project dialogue and reason at the level of a large language model. NO — not the goal.

## Division of responsibility

| COMPASS (external smaller adaptive layer) | General capable large model (replaceable consumer) |
|---|---|
| Preserve personal + per-project durable experience across sessions | Understand natural-language task and interpret complex subtleties |
| Keep goals, decisions, rationales, versions, uncertainties, scope/conditions and provenance available | Combine retrieved project-specific knowledge with its general capability |
| Update compact learned state from feedback; retain auditable evidence separately | Explain implications, propose work plans, identify conflicts and reason over new situations |
| Select, rank, compress and expose **relevant context** for a particular request, with caveats | Judge whether a new proposal fits project goals using the provided context and original task |
| Receive observations, feedback and **proposed** model interpretations with source; route ambiguous ones for review | Supply potential structured observations when explicitly configured, without automatically declaring user-approved facts |
| Manage a stable model-neutral Context Bundle interface | Read that bundle through text or a documented adapter, without consuming COMPASS's model weights |

Avoid claims that all meaningful intent must be derivable from a narrow lexical classifier. COMPASS may include learned components for relevance, personalized selection, consolidation and update — **not a miniature general-purpose reasoner**.

## Architecture

```text
                        real work, history, user feedback
                                         |
                 +-----------------------v---------------------+
                 |  COMPASS — EXTERNAL, PERSISTENT, ADAPTIVE    |
                 |                                             |
                 | Personal COMPASS                            |
                 |  - transferable user patterns               |
                 |                                             |
                 | Project COMPASS A   Project COMPASS B  ...   |
                 |  - WHY / rules / decisions / context        |
                 |  - source, reasons, validity, versions      |
                 |                                             |
                 | Evidence store + versioned state            |
                 | Compact learner / relevance & consolidation |
                 | Context Builder + uncertainty gate          |
                 +-----------------------+---------------------+
                                         |
                               Context Bundle (portable)
                           data, not untrusted instructions
                                         |
                                +--------v---------+
                                | Large LLM        |
                                | INTERPRETS /     |
                                | REASONS / ACTS   |
                                +--------+---------+
                                         |
                               task / outcome / feedback
                                         |
                          source-grounded update proposal
                                         |
                                COMPASS review / update
```

**Not in the LLM's weights:** user/project memories, their durable versions, local learned representation and update ledger remain in COMPASS. The general model can be replaced or upgraded. SELF may **optionally** be used as a consumer if/when its language competence suffices, but **SELF development is not a prerequisite** for COMPASS.

## Minimum model-agnostic exchange contract

The Context Bundle must provide, with strict budgets and citations:
- `scope`: one user plus explicit project, not arbitrary projects or all global info.
- `project_why`: short stable aims; distinguish user-approved assertions from inferred hypotheses.
- `active_decisions`: only appropriately approved/validated decisions, with applicability and reasons.
- `relevant_history`: a few earlier events (corrections, counterexamples, outcomes) and exact evidence pointers.
- `uncertainty`: unanswered questions, competing interpretations, known gaps; prevent false claims of certainty.
- `source_provenance`: original document/message ID, date, revision, owner and approval/verification state.
- `omissions`: what was excluded because irrelevant, wrong-scope or context budget.
- `bundle_revision`: stable version/snapshot so the work can be reproduced.

A Bundle is evidence **for the large LLM to interpret**, not a ready-made final answer, nor an instruction that overrides the model's safety and task hierarchy.

## What to evaluate — correctly

The main COMPASS result is **the incremental improvement attributable to the external layer** on the **same fixed capable large model**, not whether its tiny learner can answer full project questions on its own:
- A) Big model, current task alone (no persistent user/project knowledge).
- B) Same big model + ordinary retrieval (same source documents, matched budget).
- C) Same big model + flat / static project notes (same budget).
- D) Same big model + COMPASS's evolving scoped Context Bundle.

Use genuinely later, withheld project tasks; compare decision alignment with the original user intent, rationale retention, correct validity/scope, false confidence, missed constraints, unnecessary re-explanation and source accuracy. Hold the large model, task, inference conditions, context budget and user permissions fixed. Include a frozen COMPASS-layer ablation to show whether **its own updates** add value beyond well-structured static notes.

Additional independent COMPASS metric: how accurately its compact model updates, selects and consolidates evidence and returns provenance, irrespective of the large model used. This is **not** evidence of autonomous full language understanding.

Do not make GPT-6 a required teacher or required runtime. The larger LLM is a replaceable **consumer**; any experiments using it to prepare candidate facts must disclose that role and separate user-confirmed authority from model suggestions. Do not leak private project sources into public repositories or external models without permission.

## Immediate next implementation (COMPASS track only)

**V12: model-neutral `ContextBundle` builder** from existing V9.12 source/approval journal and COMPASS personal/project scopes. Start with source ranking + hierarchical WHY + temporal state/corrections + uncertainty, budgets and citations. Reuse the small V11/V11.1 update experiments **only** where they measurably help ranking/consolidation; they are NOT the main language-understanding architecture.

One portable bundle, consumed as *data* by two different large models in local/manual tests when available; the system remains external to both. Any optional SELF adapter is a separate integration milestone, not a dependency. The scientific claim requires real continuation-quality evaluation above matched RAG and static memory, not successful unit tests alone.

**Canonical research target: `PORTABLE_EXTERNAL_ADAPTIVE_CONTEXT_LAYER`.** This supersedes assertions that (a) COMPASS must autonomously comprehend arbitrary project dialogue and (b) COMPASS must be integrated natively into SELF or taught by GPT-6.
