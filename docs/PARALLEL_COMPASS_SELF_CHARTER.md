# Two parallel research and development tracks: COMPASS and SELF

**Canonical clarification — 2026-10-12.** The owner has explicitly corrected a previous mistaken attempt to merge the projects into a single system. **COMPASS and SELF are TWO SEPARATE, PARALLEL DEVELOPMENTS.** Neither is defined as a subsystem of the other. Interoperability is an *optional interface*, not their identity, dependency, required training path or success criterion.

## A. COMPASS — independent adaptive understanding system

Purpose: build a continuously learning model of the user and of the **idea and purpose of each project**, including goals, guiding principles, reasons for decisions, historical evidence, constraints/validity, corrections, and the difference between desired intent and current implementation. It must preserve useful long-term knowledge, improve from actual work and feedback, and help future sessions/projects without repeatedly rebuilding understanding.

Architecture:
- **Personal COMPASS**: enduring patterns and goals specific to the user that can be applied across projects when genuinely transferable.
- **One Project COMPASS per project**: purpose, history, learned relations, decision rationale, valid conditions and unknowns specific to that project.
- Explicit, provenance-aware **selective knowledge transfer** between project and personal COMPASS, with separation of evidence vs interpretation vs approved decision.
- Own independent data format, memory/evidence mechanism, learning/update mechanism, evaluation, executable runtime and roadmap. No SELF checkpoint is required to operate or test the COMPASS research system.
- COMPASS may eventually interact with other systems through adapters, **including SELF**, but it is not defined as merely a prompt wrapper for an external hosted model; specifically GPT-6 is not a constituent of COMPASS, its teacher, or the target of its evaluations.

COMPASS's real learning and evaluation must be assessed **on COMPASS's own objectives**: source-backed continuing understanding, transfer of appropriate principles, correction handling, provenance and confidence, cross-project boundaries, held-out longitudinal work and value over memory/retrieval baselines. Don't call a word-classification benchmark or SQLite engineering test proof of deep project-intent learning.

## B. SELF — independent language model and model-research system

Purpose: design, train, improve and evaluate our own language model, including its architecture, language competence, SELF mechanisms, variants and scaling (e.g., 5M and larger models), training and inference.

SELF has its own:
- model, tokenizer, checkpoint series, training datasets, parameters and compute experiments;
- evaluation of language capability, generalization, identity/state mechanisms, model quality and training effects;
- personal-state modules already researched in SELF: these are **SELF's own work** and do not establish COMPASS as part of SELF;
- release gates and negative/positive research results.

No COMPASS training dataset or state format is mandatory for SELF to run, learn or make progress.

## Optional future interface (not a merger)

```text
   COMPASS: personal + project models     SELF: language model and SELF experiments
   independent research, learning, tests   independent training, checkpoints, tests
                         \                 /
                          \ [optional]    /
                           INTEROPERABILITY
                   (explicitly designed and evaluated)
```

A separate integration experiment can compare SELF alone to SELF assisted by COMPASS **if both sides are independently ready**, but this does not make COMPASS a SELF component or SELF a COMPASS component. Likewise COMPASS cannot depend for its validity on the quality of SELF 5M language generation.

## Current research status and corrections

- **V9.12–V10.2 COMPASS experiments:** useful *historical* provenance/journal/evaluation infrastructure and many known limits. Previous turns wrongly put GPT-6 into the COMPASS training/inference evaluation loop. Those model-centric experiments are **not** the active direction; retain accurate engineering results and negative research outcomes for reproducibility.
- **Mistaken 2026-10-12 SELF-native COMPASS assertion:** explicitly **RETRACTED**. Do not implement the previously proposed "V11A" as a mandatory COMPASS→SELF adapter, and do not redirect all COMPASS work to SELF 5M. The earlier file `SELF_NATIVE_ARCHITECTURE.md` is superseded and serves only as a record of that incorrect plan.
- **Next COMPASS work:** revisit the foundational COMPASS architecture and establish an **independent learning/evaluation plan** for progressively constructing and revising user/project understanding. Avoid both the GPT-6-centred detour and forced SELF coupling.
- **Next SELF work:** continue own model training, architecture and evaluations **independently**, using the main model experiment log and active SELF branches.

## Guardrails for all future work

1. Before naming a task, explicitly assign it to **COMPASS**, **SELF**, or **OPTIONAL INTEGRATION**.
2. Separate experimental datasets, checkpoints/artifacts, test results and success gates.
3. A COMPASS milestone must be about better COMPASS learning and long-term user/project understanding, not merely improving another model's outputs.
4. A SELF milestone must be about SELF's own model capability, not merely the quality of external memory.
5. No automatic elevation of a proposed/derived user or project principle to approved authority.
6. Source material from private projects remains private; public repository publishes only suitable code and aggregate findings.

This document supersedes conflicting statements in earlier COMPASS charters/reports. Historical records are preserved rather than rewritten as if their hypotheses had always been correct.
