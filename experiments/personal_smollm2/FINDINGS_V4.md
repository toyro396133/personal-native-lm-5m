# SELF V4 — completed three-seed result audit

**Execution:** [GitHub Actions #38091405388](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38091405388), completed **SUCCESS** (2026-10-10 22:40 UTC). The three training matrix jobs and the aggregate job all succeeded. Report artifact: \`self-v4-aggregate-results\` ID \`11684995073\`. Seed artifacts: 11=\`11684232966\`, 19=\`11684741580\`, 37=\`11684157774\`. Each seed artifact contains the per-step/per-fact JSON and separate replay/recent adapter checkpoints.

## Experiment overview

- Base: frozen \`HuggingFaceTB/SmolLM2-360M-Instruct\`.
- Per-user adaptation: residual adapter with 7,680 trainable weights, run in two independently trained versions per seed: \`recent\` (only newest 8 facts) and \`personal\` (rehearsal of all known facts).
- 24 *synthetic* user-specific items, injected across 6 staged conversational histories, two later preference changes, other conversation turns as distractors.
- Both adapter schedules use 32 optimizer steps per stage; 3 seeds (11,19,37).
- Retrieval is non-oracle **top-3 lexical keyword/bigram ranking** of user utterances, conditional on the question only. Hybrid = this same retrieval + replay-trained adapter.
- At every stage: choose between two candidate answers using conditional likelihood. At final stage: unrestricted greedy generations assessed via string normalization/phrase matching, including 24 personal queries and 8 general questions per seed.

## Final results, three-seed means (rounded)

| Condition | Binary candidate QA | Open-answer personal QA | Generic open QA |
|---|---:|---:|---:|
| frozen base | 54.2% | 4.2% | 87.5% |
| recent-data adapter | 81.9% | 18.1% | 62.5% |
| all-history replay adapter | 75.0% | 12.5% | 54.2% |
| frozen + top-3 retrieval | 79.2% | 54.2% | 87.5% |
| all-history adapter + top-3 retrieval (hybrid) | **93.1%** | **61.1%** | 54.2% |

These figures are ***automatic grading outputs***, not claims of exact semantic correctness. Retrieval \`hit@1=20/24 (83.3%)\` and \`hit@3=22/24 (91.7%)\`, same for all seeds (retriever is deterministic and seed-independent).

### Final stage by seed (counts, not percentages)

| Seed | Base binary | Recent binary | Replay binary | Retrieval binary | Hybrid binary | Recent free | Replay free | Retrieval free | Hybrid free | Generic base/retrieval | Generic recent | Generic replay/hybrid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11 | 13/24 | 20/24 | 17/24 | 19/24 | 22/24 | 4/24 | 3/24 | 13/24 | 14/24 | 7/8 | 3/8 | 4/8 |
| 19 | 13/24 | 19/24 | 19/24 | 19/24 | 23/24 | 5/24 | 4/24 | 13/24 | 14/24 | 7/8 | 6/8 | 4/8 |
| 37 | 13/24 | 20/24 | 18/24 | 19/24 | 22/24 | 4/24 | 2/24 | 13/24 | 16/24 | 7/8 | 6/8 | 5/8 |

The difference from prior 50-fact V3 is substantive: V4 uses fictional *chat-history messages*, top-3 retrieval, explicit revisions, and **open-ended answer generation**. Do not compare percentages as if same test.

## Detailed failure analysis

### Latest-fact replacement is NOT solved

The simulated user first establishes a preferred focus duration of **35 minutes**, later explicitly changes it to **25 minutes**. On the final **open-answer** question:

- Recent-data adapter: **25 minutes** in all three seeds (correct, though some answers include filler).
- Retrieval-only: **35 minutes** in all three seeds (outdated).
- Hybrid: **35 minutes** (seeds 11 and 37), or **three minutes** (seed 19); **0/3 correct**.
- All-history replay adapter: wrong in all three seeds (e.g. 30 minutes, two hours).

The other changed preference (dashboard accent from amber to teal) was answered \`teal\` by the hybrid in all three seeds, but not reliably by either adapter alone. Replacement behavior is asymmetric and context-dependent.

**Critical metric caveat:** Both old and correction utterances carry the same fact \`id\`, so \`retrieval.hit@1/hit@k\` can count a *superseded* note as the correct fact even though its value is wrong. Version-aware retrieval quality must treat revision and supersession separately. Do not call the 20/24 figure 'latest-version retrieval accuracy.'

### General-question regressions and pathological repetition

Across the 8-question general QA sanity set, automatic scores for frozen/retrieval are 7/8 per seed; replay/hybrid falls to 4/8, 4/8, and 5/8. This is an important regression signal, but the benchmark is too small to establish a population-level regression rate.

Several adapter answers contain clear repetitions of words/phrases, or wrong answers on elementary general questions; example: on **5 + 3**, the frozen model generates “Five plus three is eight,” which the string-based grader marks *incorrect* because expected is digit \`8\`. Some adapter outputs really are incorrect (\`five\`, or extraneous repeated text), so failures are not all scoring artifacts.

### Metric weaknesses

- **False negatives**: \`two weeks\` versus \`2 weeks\`; \`worked examples\` versus \`worked example\`; an answer describing a \`linear\` roadmap versus \`timeline\`; \`eight\` versus \`8\`.
- **Potential false positives**: a matching phrase inside a contradictory or nonsensical longer answer can pass the substring heuristic.
- Scores are deterministic automatic string matching; both an improved normalized evaluator and a fixed human-reviewed subset are needed.
- Three independent seeds on *the same* 24 facts do NOT equal 72 independent factual questions. No independence-based significance inference is justified from combining those observations.

## What we can and cannot conclude

Supported:
1. All three computational replications completed, and a true top-3, non-oracle retriever plus adapter achieves the highest **binary-choice** success on the synthetic task.
2. On scored open answers, retrieval (54.2%) and hybrid (61.1%) greatly exceed adapter-only (12.5%) and recent-only (18.1%).
3. The trained adapter causes serious general-answer quality issues on this narrow sanity check.
4. Rehearsal of all prior facts is not automatically superior to focusing on recent facts: recent wins in personal-only binary and free-answer tests under the same 32-update budget.
5. Updating/correcting previously recorded knowledge is still a blocking weakness; the strongest candidate architecture needs explicit versioned memory.

Not established:
- Reliable real-conversation personalized memory, safe deletion/forgetting, robust free-form generation, multi-population transfer, statistical advantage over retrieval, or preserved general capabilities.

## Proposed V5 gates (do not automatically launch)

1. **Versioned canonical memory:** extract user-authored facts with timestamps and supersession links. Retrieve only current values by default. Verify changes to 35→25 and amber→teal, report outdated-answer rate separately. The original conversation remains a source record, but not an equally weighted answer candidate.
2. **Selective adapter gating:** keep frozen backbone as default for general queries; enable personalized residual adapter only on user-specific tasks. Compare to retrieval-only, gated hybrid, and always-on hybrid.
3. **Conservative update strategy:** optimize preference/style or disambiguation effects; shrink LR and/or add KL penalty on general-purpose queries; do not rely on rehearsing all stale factual labels.
4. **Evaluation quality:** normalize digits and common variants, flag repetitions and contradictory answers, and manually review a small locked evaluation set. Report raw outputs, automatic scores, and human adjudication separately.
5. **Better independent test data:** more profiles, multiple paraphrase families, different synthetic histories per seed; include no-memory, outdated-note, irrelevant-note, and conflicting-note controls.
6. **Commit threshold:** prefer improved free-form factual recall *and* no material general-ability regression; never optimize candidate-ranking alone.

**Decision:** retain the frozen backbone + external memory as the current safer baseline; hybrid remains experimental due to free-form and generic regression failures. Keep V4 branch unmerged into production.
