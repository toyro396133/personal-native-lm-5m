# COMPASS — research charter and source of truth

**CURRENT CANONICAL DIRECTION — 2026-10-12:** COMPASS and SELF are **parallel independent developments**. COMPASS is an **external, portable, adaptive context layer** that learns/preserves a user's and each project's longitudinal state; a *replaceable capable large model* interprets that layer and performs the general language understanding and reasoning. COMPASS remains outside that model's weights, persistent across model replacement, with its own small learning/update/relevance mechanisms. It is **not** supposed to autonomously master arbitrary project language at its own limited model size; nor is it a native mandatory SELF component or a GPT-6-specific product. **Canonical architecture:** [External Context Layer](EXTERNAL_CONTEXT_LAYER_ARCHITECTURE.md) and [Parallel COMPASS / SELF tracks](../PARALLEL_COMPASS_SELF_CHARTER.md).

**What we must prove:** the SAME capable model, with a continuously updated COMPASS Context Bundle, performs genuinely later project work more faithfully than with equal-budget RAG or static notes. Independently assess COMPASS's relevance selection, source-trace retention, versioning, scope and update quality. Do not mistake an isolated small classifier's correctness for a requirement that COMPASS reason in full natural language.

## DIRECTION CORRECTION — External COMPASS layer + capable interpreter, not autonomous small-LLM comprehension

The user's clarified aim is **portable adaptive context**: COMPASS itself is a separate persistent smaller layer which stores and learns relevant personal/project state and hands a source-grounded, scoped context bundle to a replaceable capable larger model. The **large model** supplies deep project comprehension and reasoning, drawing on that learned context. It is not the target of COMPASS training and the COMPASS state is not trained inside the large model's weights. **SELF remains its own parallel research program; it is not a COMPASS prerequisite.**

Thus V11 / V11.1 lexical-choice/freeform-correction learners are **limited subcomponent engineering probes**, not a roadmap toward a standalone tiny full-language interpreter. Earlier experiments treating GPT-6 as an indispensable teacher/primary research subject are also not canonical; a capable LLM can be used as a **replaceable downstream consumer** for a FAIR context comparison. Next COMPASS milestone: model-neutral Context Bundle generation, relevant evidence consolidation and update; then controlled evaluation with the same large model and equal-information RAG/static controls.

[Authoritative architecture](EXTERNAL_CONTEXT_LAYER_ARCHITECTURE.md). Status: `PORTABLE_EXTERNAL_CONTEXT_LAYER_NOT_YET_EVALUATED`.

## COMPASS V11.1 — learning from authentic free-text user corrections (2026-10-12)

**Standalone COMPASS research; SELF remains a separate parallel development.** No hosted LLM, GPT-6-as-teacher, external API or SELF checkpoint was used in the learning engine or tests.

- Added a [literal free-text correction learning prototype](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-1-freeform-corrections/experiments/compass_v11_1/freeform_feedback.py): user-authored `לא X אלא Y` / `טעות! Y` feedback leads to an isolated learned preference update, with prior-state scoring, explicit ambiguity abstention and distinct per-project/personal state. The learner is a **small lexical online ranker**, not an autonomous understanding model.
- Added [local role-tagged chat ingestion](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-1-freeform-corrections/experiments/compass_v11_1/chat_ingest.py) and a [fully offline Hebrew user review page generator](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-1-freeform-corrections/experiments/compass_v11_1/offline_review.py). User-confirmed before/after interpretation and rationale can train the ranker; none can autoapprove/revoke project decisions. Chat-export roles and user approval identity are **not cryptographically authenticated**.
- [CI #38186804378](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38186804378) passed **47 distinct synthetic tests × 3 Python versions = 141 successful executions**. These are NOT real-user longitudinal understanding results.
- Examined the private historical Chacham Lev T00–T51 log: 52 stations, only **1 possible exact explicit repair pattern** found (T37), not a verified label-ready training pair. Most useful explanations/corrections are free-form and need careful interpretation, not mechanical string extraction.
- [Windows/local guide](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-1-freeform-corrections/experiments/compass_v11_1/README_HE.md) · [original protocol](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-1-freeform-corrections/experiments/compass_v11_1/PROTOCOL.md) · [transparent amendments and research verdict](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-1-freeform-corrections/experiments/compass_v11_1/AMENDMENTS_AND_RESULTS.md).
- **Next scientific step:** test future real project requests before their user answers, with independently human-confirmed intent/rationale and same-past-history retrieval/static controls. The prototype's currently logged pair rankings are **retrospective** because both options derive from a reply already observed by the experimenter. Status `FREEFORM_FEEDBACK_ENGINEERING_VALIDATED`, `NO_INDEPENDENT_LONGITUDINAL_INTENT_LEARNING_PROVEN`.

## Active next experiment — COMPASS V11 standalone lifelong learning (2026-10-12)

**COMPASS and SELF remain independent parallel developments.** The immediate COMPASS objective is no longer more static memory screens, GPT-6-assisted QA or integration with SELF, but a self-contained *experience-based updating learner*.

- Locked [V11 research protocol](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-standalone-longitudinal/experiments/compass_v11/PROTOCOL.md) and built [source-scoped CPU online learner](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-standalone-longitudinal/experiments/compass_v11/online_compass.py): chronological prediction-before-feedback, per-project and personal separate learned weights, abstention, provenance and same-past-history retrieval/recency/first-option controls.
- [CI 38186061247](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38186061247): **20 fictional engineering tests × Python 3.11/3.12/3.13 = 60 passing executions**. This establishes behavior of the updating module, **not** project-intent comprehension.
- Read-only audit of the **52** historical T00–T51 real private Chacham Lev planning episodes found **8** raw answers that *appear* to be direct single-letter choices. These are NOT yet confirmed as 8 training examples; the corresponding exact question alternatives and source authority must be inspected. Most of the corpus is free-form substantive user explanations/corrections.
- Next *actual scientific bottleneck*: turn open-ended real work events into reliable context-preserving learning signals **without external-model-generated labels**, anchored in direct user feedback and independent verification. Preserve the WHY, rationale, ambiguity, temporal and project scope. Only then run a genuinely chronological real holdout comparison with retrieval and non-learning baselines.
- Current scientific verdict: `V11_ONLINE_LEARNER_ENGINEERING_PASS` / `INSUFFICIENT_REAL_DECISION_FEEDBACK`; no evidence yet of autonomous deep understanding or better-than-retrieval decision quality.

[Detailed V11 evidence and limits](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v11-standalone-longitudinal/experiments/compass_v11/RESULTS.md).

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

> **Historical caveat for V10–V10.2 below:** the GPT-6-centric experimental line was discontinued as a research direction. Those results are retained for audit; they do not define COMPASS's next development or SELF's roadmap.

## V10 — First enduring project-understanding model, GPT-6 teacher in ChatGPT (2026-10-12)

**Pivot:** Stop optimizing isolated status classifiers. The research target is a persistently updated, source-grounded model of a project's WHY, intent, decisions, rationales, exceptions, validity and change history that can improve genuinely later work decisions.

- [V10 preregistered protocol](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-project-model/experiments/compass_v10/PROTOCOL.md) was committed before testing.
- Early private Chacham Lev material (pinned `55b0ac175513`, Oct 2, 2026): **31 GPT-6 teacher candidate claims**, partitioned as 11 source-asserted approved principles, 6 proposals, 6 open facets, 4 historical implementation observations and 4 superseded statements. Approval authentication is NOT independently established.
- Created **5 WHY-level goals** with source links and **12 GPT-6-authored causal/constraint hypotheses**, with SUPPORT / CONFLICT / ABSTAIN outcomes for **structured** design-change proposals and cited reasoning. A private offline Hebrew HTML explorer presents this first project understanding map.
- Subsequent private V9.24 SQLite memory graph remains **20 pending claims + 10 pending relations, 0 approvals**; later material is not backfilled into earlier policy. Cross-project/person scope promotions are prohibited automatically.
- Engineering: **19 local tests PASSED**, offline JS passed Node syntax check; GitHub Actions [run 38183466280](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38183466280) **15 distinct fictional tests x 3 Python versions = 45 PASSED executions** on the generic no-private-data engine.
- **Very important research boundaries:** teacher-aligned 16/16 **is only an engineered sanity check** (same GPT-6 wrote the rules and cases). Additional pre-run-frozen 12-case teacher-authored red team **8/12**; failures: unknown-vs-known precedence, applying requirement for decision rationale in new context, goal generalization, treating legacy implementation as non-authoritative. No after-test fixes to inflate it.
- Early-snapshot TF-IDF retrieval found one authored supporting source in top-three for **7/12 eligible** cases; this is **citation retrieval, not decision correctness**. A no-memory abstain always proxy matched 8/16, not a GPT-6 model evaluation.
- **NO valid four-arm blinded GPT-6 experiment was run:** the desired matching no-memory/RAG/flat facts/V10 context experiment still requires truly separate equivalent-context inference runs, future task questions and independent human scoring. No API was used and no neural SELF weights were trained.
- [Full aggregate-only V10 research report](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-project-model/experiments/compass_v10/RESULTS.md) · [generic V10 code](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-project-model/experiments/compass_v10/project_model.py). Full source-linked local model and exploratory cases remain conversation download artifacts, not raw public project documents.
- Verdict **`V10_PROJECT_MODEL_ENGINEERING_PROTOTYPE` / `NO_INDEPENDENT_COGNITIVE_GAIN_PROVEN`**. Do not interpret this as learned deep understanding. The next major task is externally scored **continuation quality on real decisions**, not another keyword classifier.

## V10.1 — Dynamic understanding without losing WHY (2026-10-12)

- **New operational capability:** the earlier V10 static project model now has an **incremental update loop** using the existing V9.12 versioned, append-only SQLite memory store. GPT-6 suggests source-linked `goal/principle/decision/open_question/implementation_observation` facets; none is activated unless independently reviewed and explicitly approved. Corrections are versioned per facet; old interpretations and their rationale survive.
- **Current-state retrieval:** approved WHY goals are shown first in a bounded, task-oriented brief. Pending proposals are excluded; source and revision are visible; unmet applicability and missing dependencies are flagged. No automatic personal/project knowledge promotion. Source excerpts are lower-trust data, not system instructions; this boundary is only a limited prompt-injection safeguard.
- **Measured technical results:** [GitHub Actions #38184257420](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38184257420): **29 fictional tests PASSED across Python 3.11/3.12/3.13 = 87 successful executions**, including three-phase replay of version corrections, source forging, malformed dependencies, stale revisions, project isolation, budget awareness, no autoapproval and preserved history.
- **Real-project evidence preparation:** 13 GPT-6 teacher-*proposed* source-linked Chacham Lev cases (4 early from pinned Oct 2, 3 intermediate, 6 later from pinned Oct 11), with three corrected facets. All 11 unique referenced source sections were read and checked against their Git heading/line range. Local source-sequence sanity check: **4 → 7 → 10 facets**, **3 two-version corrections** and **0 real approvals**; not a full real-data V9.12-backend run. The real bundle is only distributed to the user privately, never via public CI.
- [V10.1 Hebrew instructions](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-1-learning-loop/experiments/compass_v10_1/README_HE.md) · [protocol](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-1-learning-loop/experiments/compass_v10_1/PROTOCOL.md) · [detailed results](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-1-learning-loop/experiments/compass_v10_1/RESULTS.md).
- **Verdict:** `VERSIONED_PROJECT_STATE_LEARNING_LOOP_WORKING`, **`NO_INDEPENDENT_GPT6_REASONING_IMPROVEMENT_PROVEN`**. GPT-6 as teacher extracted the interpretations; the engine does not autonomously recognize deep user intent. A genuine breakthrough requires prospective human-judged continuation tasks, matching GPT-6 inference budget, and significant advantage over static/RAG baselines without unsupported-approval mistakes.

## V10.2 — Prospective independent decision-quality experiment is ready (2026-10-12)

- V10.1 demonstrates versioned project-understanding memory engineering, **not yet a cognitive advantage** over retrieval.
- Precommitted [V10.2 prospectively measured four-arm protocol](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-2-outcome-evaluation/experiments/compass_v10_2/PROTOCOL.md) at SHA `c7f972e385bf24489819402938dfbfa2baa988ef` (2026-10-11 23:30:16 UTC), before any eligible future task. Arms: same GPT-6 in independent chats with no memory, flat facts, ordinary RAG and V10.1 evolving context; max 5000 characters per context; no OpenAI API.
- **Primary externally meaningful score** is blind human-rated goal fidelity, source grounding, useful continuation, uncertainty handling and unsupported confident claims on 12+ newly arising user tasks from 3+ workstreams; 2 independent raters; mean COMPASS minus RAG >=2.0, exact paired sign p<0.05 and no greater severe unsupported-claim rate.
- Added offline [locked evaluator](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-2-outcome-evaluation/experiments/compass_v10_2/score_prospective.py) and [Hebrew guide](https://github.com/toyro396133/personal-native-lm-5m/blob/experiment/compass-v10-2-outcome-evaluation/experiments/compass_v10_2/README_HE.md). [GitHub Actions 38184617841](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38184617841) **17 fictional tests x 3 Python versions = 51 PASS executions**.
- **No real prospective GPT-6 comparisons have run yet.** No valid independent benefit has been established and the score template currently contains **zero** user tasks. Prevent test/teacher contamination; never invent other sessions or treat synthetic evaluator fixtures as empirical outcomes. Status `PROSPECTIVE_EXPERIMENT_PREREGISTERED_AWAITING_REAL_USER_TASKS`.

## CORRECTED RESEARCH BOUNDARY — 2026-10-12 (authoritative)

**Two parallel developments, not one:** COMPASS independently develops ongoing understanding of users/projects; SELF independently develops and trains its own language model. **Neither depends on the other** for its architecture, experiments or progress. Optional future integration requires its own explicit research plan and separate metrics.

The previous attempt to make SELF-native conditioning the *definition* of COMPASS and to redirect COMPASS's next experiment to SELF 5M is **retracted**. The earlier [SELF_NATIVE_ARCHITECTURE.md](SELF_NATIVE_ARCHITECTURE.md) is a preserved historical error, not a current instruction. The GPT-6-centred teacher/testing direction is also not part of the current COMPASS program.

**Next COMPASS goal:** establish independently how COMPASS can learn and revise a meaningful user/project model from continued work, preserving WHY, decisions, evidence, scope and uncertainty, with controlled evaluations against retrieval and static-state baselines. **Next SELF goal:** continue architecture, training and model evaluations from its own log independently. A later integration could be tested as a third, optional line.

[Canonical two-track development charter](../PARALLEL_COMPASS_SELF_CHARTER.md). Status: `COMPASS_INDEPENDENT_LEARNING_NOT_YET_PROVEN`, `SELF_TRAINING_SEPARATE`.

## Priorities after V9.11

1. **Learn/write/maintain concept records from actual event streams**, not only read gold-labelled concept slots: corrections, new principles, different scopes, reasons, conflicts and revocations.
2. **Interleaved multi-concept long-horizon evaluation**, with position shortcuts made insufficient and realistic project work, not only author-generated short three-line evidence.
3. **Hebrew and multilingual semantic competence**, heldout genuinely authored cases and independent human labels; report personal/project scores separately.
4. **Proper permission, source authentication and ambiguity handling** for production-grade use; test uncertain ownership and source reliability.
5. **Demonstrate real across-project personal learning** without promoting project-local commitments to global principles.
6. Measure COMPASS's own learning and continuity against equal-information memory/retrieval and static COMPASS baselines; keep independent evidence, preregistration and honest NO-GO. Treat SELF integration as optional and distinct.

## Working convention

- Scientific code stays in `experiment/compass-v*` branches pending review; CI workflow triggers on main are not production deployment.
- Document protocol *before* first cold prediction, preserve all seeds and failures, compare baselines on the same cases and state explicit limits.
- Use **COMPASS** for its own independent continuing user/project understanding system, and **SELF** for the independently researched language model. Any cross-project connection between COMPASS and SELF is optional and must not replace their separate roadmaps.
