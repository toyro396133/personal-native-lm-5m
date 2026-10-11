# COMPASS V8 — Project Hierarchy & Priority Reasoning
## Registered design; **no training run has been launched**

**Research component:** **COMPASS / מצפן**, the *personal decision-policy adapter*.  
**Different component:** **SELF** remains the model's internal "self" reference/mechanism. This V8 proposal does **not** test, change, or conflate the SELF self-model.  
**Repository research track:** experiments/compass_v8/ (separate from experiments/personal_smollm2/ and the native 5M SELF experiments).

### 1. User outcome and falsifiable central claim

Given two or more projects with different objectives, the model should:
1. Resolve which project a task belongs to.
2. Distinguish a project's non-negotiable **core outcome**, supporting **capabilities/layers**, **ancillary branches** (tests, polishing, optional ideas), and **out-of-scope** work.
3. Distinguish **structural importance** (where a task belongs) from **execution priority** (whether it is urgent now). A peripheral testing activity may become immediately necessary when it exposes or prevents a core blocker.
4. Consider **dependencies, evidence quality, decisions, constraints, current phase, deadlines, and validity conditions**, not superficial keyword matching.
5. Recognize when a task should be done, deferred, investigated, or explicitly escalated for a user decision — **without inventing policy**.
6. Carry the same user's prioritization principles across different projects while keeping each project's actual objectives, decisions, and private data separate.
7. Adapt when the user changes a rule or when evidence invalidates an earlier decision, without continuing to rely on stale assumptions.

**Main hypothesis H1:** A small *personal* learned policy (COMPASS), conditioned on a separate authoritative project graph, improves **correct prioritization of previously unseen situations** over simply supplying the graph and an explicit policy in the prompt. A change in answer format, length, or confidence alone does not support H1.

**Size hypothesis H2:** Increasing rank from 4 to 16 to 64 produces a measurable **held-out policy generalization** gain beyond matched compute, not just training memorization. Null results must be kept.

### 2. Three distinct sources of intelligence

**A. Project Memory / Ground Truth (external, versioned).** Structured per-project map of objective, core deliverables, supporting components, branches, decisions, sources, constraints, validity, current phase and dependencies. It is *not* embedded exclusively in personal weights. This map is changed when reality or the user changes. Every element has a stable ID, version and source.

**B. COMPASS (trainable adapter).** Learns transferable *personal criteria for deciding*: protect the core first, do not confuse activity with impact, identify prerequisite blockers, prefer reuse/continuity where justified, maintain scope, know when to ask for evidence and when to stop exploratory work. It must not memorize project-specific secrets or superseded project states.

**C. Frozen language model.** Interprets user requests and produces grounded analyses and recommendations. Initial experimental backbone: HuggingFaceTB/SmolLM2-360M-Instruct, with fully frozen backbone parameters and an isolated COMPASS adapter. The native 5M SELF architecture is unaffected.

Important: storing the literal map and decision reasons in external memory is a **feature** and a strong baseline, not an unfair trick. The scientific question is whether learned COMPASS policy adds anything on top of that.

### 3. Conceptual graph/schema

\`Project\`: id, title, version, phase, objective, success_conditions, non_goals.  
\`Node\`: id, project_id, type = core / supporting / ancillary / out_of_scope; intent; parent_id; required_for; state.  
\`Dependency\`: upstream_id, downstream_id, blocker?, severity, supporting evidence; possible core impact.  
\`Decision\`: id, project_id, claim, source, rationale, condition, valid_from, superseded_by, approval_status.  
\`Task\`: id, project_id or unknown, node_ref(s), proposal, urgency, impact, risk, reversibility, estimated effort, evidence.  
\`PersonalPolicy\`: stable general principles with provenance and version, **not the answer to a test task**.

The decision engine must output, separately:
- \`project_id\`, \`structural_layer\`, \`core_relation\`, \`blocked_core?\`;
- \`action\` = DO_NOW / SCHEDULE / INVESTIGATE / DEFER / ASK_USER / REJECT;
- \`priority_tier\`, supporting \`evidence_ids\` and \`policy_reason_ids\`;
- \`uncertainties\`, \`conditions_for_revisit\`, \`proposed_next_step\`.
Abstention is a valid success when evidence is insufficient. The output must not turn speculative evidence into confirmed facts.

### 4. Critical distinction: layer is NOT priority

Examples that must be classified correctly:
- "Add 100 more E2E checks" may be ancillary and deferred if the core currently works; not automatically high priority just because testing sounds responsible.
- "The basic save operation corrupts records, demonstrated by a failing E2E" makes **fixing that core defect** urgent and its targeted regression test a necessary prerequisite. The nature of "tests" alone does not decide priority.
- "Redesign the homepage gradients" may be ancillary to a product whose core workflow is still broken, but core to a *design-system* project explicitly scoped to visual accessibility.
- "The user changes a requirement" requires a versioned update and reassessment; repeating the old decision is a failure.
- "I found a possible defect, without reproduction or logs" can justify INVESTIGATE rather than rewriting core or asserting a confirmed bug.
- A task belonging to Project A must not inherit Project B's route, objective, assumptions, or permission.

### 5. Dataset: synthetic but independently authored and challenging

**Stage 0: protocol/data-quality pilot — no neural training.** Create 4 fictional users with different policies, each managing 2–3 fictional projects. Each project has 3–5 graph layers/nodes and 2–5 decision/dependency records. Build 20–30 scenarios per user with *adversarially similar task text* and at least one dependency that can elevate an ancillary task. Check that the graph schema and scoring can represent the scenarios unambiguously.

**Stage 1: training set.** Target ≥16 fictional policy profiles, each spanning ≥3 fictional projects and multiple phases. Generate around 800–1,200 candidate decisions, then manually or rule-audit every conflicting/ambiguous case. Balance core/supporting/ancillary, obvious blocker/non-blocker, update/stale state, and "insufficient evidence" decisions. Keep task labels **out** of the project-memory input.

**Stage 2: genuinely held-out evaluation.** A locked set of at least 240 scenarios from newly authored wording families, with entirely unseen *project maps* and combinations of dependencies. Include ≥80 "trap" examples designed to distinguish layer from urgency. Keep examples from the same project, authoring template and near-duplicate scenario **in the same split**; random row splits are forbidden. Hold back several entire fictional policy profiles as an additional cold-transfer test. Do not leak oracle ranks/rationales in retrieval results, file paths, metadata or prompts.

**Stage 3: revision tests.** Paired examples before/after an authoritative change (e.g. status turns from unverified concern to confirmed blocker; the project's core objective changes by explicit user decision). The model must update recommendations while retaining the new provenance and explicitly identifying what changed.

Have two independent reviewers (or a human reviewer plus a deterministic, **separately implemented** audit rule set) label the lock set. Log disagreement and adjudicate using the source project map, not model output. A deterministic ruleset alone cannot be called independent evidence if data were generated by the same ruleset.

### 6. Compared arms (same test contexts and answer schema)

| Arm | Frozen LM? | Project graph | Personal decision policy | Learned adapter |
|---|---|---|---|---|
| B0: no context | Yes | No | No | None |
| B1: graph only | Yes | Yes | None | None |
| B2: graph + written personal policy | Yes | Yes | Explicit current policy | None |
| B3: deterministic graph/rule engine | N/A | Yes | Explicit auditable rules | None |
| B4: zero-residual untrained adapter + graph | Yes | Yes | None | Untrained control |
| C4: COMPASS rank 4 + graph | Yes | Yes | Learned from past decisions | 7,680 weights |
| C16: COMPASS rank 16 + graph | Yes | Yes | Learned from same decisions | 30,720 weights |
| C64: COMPASS rank 64 + graph | Yes | Yes | Learned from same decisions | 122,880 weights |
| C16-P: rank 16 + graph + written policy | Yes | Yes | Learned + explicit | 30,720 weights |

The graph always contains **what the project is**, not the scenario's gold decision. C4/C16/C64's general decision preferences are learned; B2 receives those preferences as an explicit prompt. Including B2 and B3 prevents a false claim that parameter training is superior merely because the control lacked project guidance.

Architecture is *unchanged* for rank comparisons: a zero-initialized low-rank residual in the same frozen hidden layer (\`hidden_size=960\`): **trainable weights = 2×960×rank**, ranks 4, 16, 64. This is an adapter-capacity test, **not** training separate language models from scratch.

Use identical project context retrieval across language-model arms; log retrieved node IDs and completeness. An oracle-map control may separately measure upper-bound reasoning, but must be labeled as an oracle.

### 7. Training and compute discipline

- CPU-based GitHub Actions proof-of-concept (no dependence on user's local GPU); keep model download/torch installation caching reasonable and stop failing smoke tests before model load.
- Fix base checkpoint hash, tokenizer, layer attachment, learning rate, prompt format and output budget before running. Three seeds (11,19,37) for each capacity on the same data.
- **Primary matched-update comparison:** equal examples, order distribution and gradient steps per rank; log trainable-parameter count, optimizer settings, wall-time and token counts. Acknowledge that equal steps != equal floating-point compute.
- **Secondary compute-matched comparison:** limit large-rank runs to the same predeclared training compute or wall-time as rank 4 and report actual consumed compute; do not cherry-pick the better comparison.
- A pilot must verify schema parsing, no cross-project leakage, task-label isolation, exact output JSON syntax, memory revisions and abstention without invoking the base model. Then one small end-to-end smoke job; only after it passes, expand the rank × seed matrix.
- Use independent branch and artifacts with checkpoints per rank/seed; never overwrite V1–V7 metrics. Start at low budget, expand only if trial-level evidence indicates the benchmark is valid and informative.
- Rank growth is not automatic: skip C64 expansion if C4/C16 already match the graph+policy baseline or training primarily degrades basic reasoning.

### 8. Registered evaluation dimensions

**Primary (decision correctness):**
1. \`Action accuracy\`: choose what to do next (multi-class including abstention), not just which layer is named.
2. \`Joint project+layer+action\`: score success only if all three match the adjudicated label.
3. \`Priority regret\`: penalty for promoting low-value peripheral work over core blockers (asymmetric costs; penalties fixed before scoring).
4. \`Hard-constraint violations\`: cross-project data mixing, acting outside an explicit prohibition, accepting a superseded decision as current, or confident unsupported claims. These are blocking findings.
5. \`Revision consistency\`: before/after correct flip when facts change, stability when irrelevant wording changes.

**Secondary (explanation and generalization):**
- \`Evidence-grounded rationale\`: cited valid graph/policy IDs and the *true* blocker chain; no invented project details.
- \`Calibration/abstention\`: ask for missing evidence when needed, without reflexive blanket refusal.
- \`Multi-project interference\`: accuracy when a second unrelated project's context is added.
- \`Novel project transfer\`: performance on completely unseen projects and split-off wording families.
- \`Generic QA drift\`: unrelated reasoning/small knowledge probes with the personal adapter on/off.
- \`Efficiency\`: extra training/inference compute and memory per correct decision.

Include explicit *counterfactual pairs* in which changing only one blocking fact must change the chosen action, and distractor pairs in which a paraphrase must not change it. This is more informative than raw keyword accuracy.

All language-model arms must return a common structured decision first (validated against schema), followed by a brief rationale grounded in IDs. No ranking from answer verbosity. Human assessment of a locked subset checks whether recommended actions are *actually sensible*, not merely phrased confidently.

### 9. Success/failure gates (pre-commit before training)

**Pilot readiness:** ≥95% gold schema consistency after adjudication, and no overlap between train/test project IDs/templates. If not, fix the dataset first and record the change.

**Evidence of added COMPASS value:** on unseen-project test, the chosen trained rank must outperform **B2 (graph + written policy)** on joint decisions by at least **5 percentage points**, with interval estimation at the *project/profile cluster* level indicating a consistent direction. In addition it must not worsen cost-weighted priority regret, hard-rule violations, or generic performance. A result above B1 but below B2 **does not justify** learned policy in production.

**Capacity claim:** require rank 16 or 64 to outperform rank 4 under both primary matched-step and secondary compute-bounded analyses, across seeds and profiles, and justify added resource cost. Otherwise keep smaller rank.

**Safety and scope:** no confirmed project-mixing, outdated core decision, or forbidden-scope action on a prespecified high-risk canary set. Report ALL critical errors even when aggregate accuracy looks good.

**Null is a valid success for research:** If B2/B3 are as good or better, recommend explicit memory + auditable decision rules instead of a larger adapter. If results depend on hints only present in train templates, reject generalization claims.

### 10. Privacy and project separation

Do not train on user's real connected repositories, emails, confidential project maps or cross-user details for this test. All data is fictional. The live product must eventually support per-project permissions, provenance, update, expiry, and deletion (including audit / checkpoints / retraining policy). Experimental deletion of an active value **alone** is not full data erasure.

### 11. Deliverables and next execution sequence

1. **Planning (this commit):** freeze terminology, schema, competing hypotheses, baselines, data splits, rubric, and gates; no training.
2. Implement dataset generator, schema validator, oracle-independent review fixture and deterministic policy baseline. Commit an audit of source/task-label separation.
3. Implement COMPASS adapter rank parameterization and common evaluator; 2-project pilot with seed 11.
4. Once smoke and adversarial trap controls pass, run 3 ranks × 3 seeds with graph-only and explicit-policy baselines; preserve raw output, metrics, ranks, and cost.
5. Produce signed-off results comparing rank **4/16/64**, exact failure examples and a go/no-go decision **separate from SELF**.

**Current scientific decision:** No evidence yet that larger rank improves project-hierarchy reasoning. V8 is a prospective experiment, not an achieved result.
