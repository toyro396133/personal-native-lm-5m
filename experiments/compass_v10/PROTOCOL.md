# COMPASS V10 — Project understanding model and longitudinal decision support

**Preregistered 2026-10-12 before V10 evaluator execution.** COMPASS is the object of research, not SELF training. GPT-6 in ordinary ChatGPT is the provisional teacher. **No OpenAI API calls.**

## Target hypothesis

An explicitly sourced, evolving project model that distinguishes durable goals, decisions, evidence, open questions and implementation state will make more faithful recommendations on future project choices than a flat text/document store. This is about *useful maintained project-state reasoning*, not claiming independent neural understanding.

### Data boundaries

- TRAIN from existing 31 GPT-6 teacher-authored early candidates, each grounded in Chacham Lev documents pinned at `55b0ac175513` (Oct 2, 2026). Distinguish teacher label from authenticated user approval.
- Real later decision documents at `fe9863278a04dedf27a0c4fcb8cca56b91ec4544` (Oct 11, 2026) are **EVAL-ONLY**. Their rules must not be backfilled into an Oct 2 model. GPT-6 already read both periods, so all case construction and labels are **teacher-informed and exploratory, not human blind independent**.
- All raw private source content, local personal/tenant data and provenance-bearing per-case corpus files remain user-local. Only code, protocol and aggregate results may be pushed to public COMPASS repository.
- Project-specific knowledge cannot automatically promote to the personal COMPASS. A proposal cannot automatically become an approved decision.

### Operational model

Versioned project model = GOAL graph (why), committed PRINCIPLES/CONSTRAINTS with source, UNRESOLVED facets, time-scoped observations, supersession history, and teacher-inferred dependencies ("this change undermines this goal"). **Every inference states provenance and confidence limitations**.
- Separate at least `project_intent`, `product_decision`, `implementation_observation`, `unresolved`, `teacher_hypothesis`.
- Never treat observed code, old legacy system, a proposal or a guessed policy as authenticated user intent.
- Deterministic engine assesses structured **proposed product changes**, reports `SUPPORT`, `CONFLICT`, or `ABSTAIN`. Favor abstention on missing applicability. No execution/merge/approval.
- "Learning" means augmenting a tentative project model with source-linked memory and revising the candidate interpretation when a later decision is supplied and approved by human; **no neural weight training in V10**.

### Main experiment, frozen comparisons

Inputs: 12 researcher-authored scenarios representing future questions spanning physical routes, annual groups, priority, source quality, model separation and genuinely NEW telephone, interview and finance policy. All scenarios should have explicit `supported` vs `unknown` labels **decided from early source documents** before model predictions. They are **not blind** to the teacher.
- A: no memory, abstain on historical user intent.
- B: retriever, cosine/TF-IDF first matching early text paragraph; output may cite, but cannot synthesize assumptions from unretrieved documents.
- C: flat approved fact memory; return relevant facts, no dependency/judgment graph.
- D: V10 model reasoning over multiple early principles, scope and unresolved facets. Required output: recommendation, cited early supports, reasoning chain, and unresolved assumptions.
- **Do not claim a valid four-way GPT-6 experiment** without four independent, identical-budget blinded GPT-6 runs. If only code-based baselines are executed, label the results *deterministic surrogate comparisons* and do not call them cognitive superiority or a research breakthrough.
- Engineering tests include preserving previously approved facts during candidate proposals, early vs later isolation, refusing unverified approval, explicit version conflict, no cross-scope knowledge promotion, and correctness of source links.

### Research outcomes

Strict external success would require independent human scoring of real forward tasks, fixed rubric, identical GPT-6 context budget and cost, genuine out-of-sample holdout, and material gains over retrieval and static facts, including *false certainty* and work-repetition rate. This turn can at most deliver:
- `V10_PROJECT_MODEL_ENGINEERING_PROTOTYPE`;
- exploratory score vs deterministic controls, with qualitative failure trace;
- status `NO_INDEPENDENT_COGNITIVE_GAIN_PROVEN` until the human-controlled evaluation.

**Never** turn 100% unit tests or curated task-set results into proof of autonomous deep intent learning.
