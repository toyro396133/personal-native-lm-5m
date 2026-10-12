# SELF V7 — completed, audited six-run study

**GitHub Actions:** [Run #38095717406](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38095717406), **SUCCESS**. All six profile/seed trials and the aggregate review-artifact job succeeded. Aggregate artifact `self-v7-summary`: GitHub artifact ID `11686222212`. Trial JSON + per-user residual safetensors weights: profiles (0,1,2), seeds (11,19), all six artifact outputs retrieved and inspected.

## What was actually tested

Frozen `HuggingFaceTB/SmolLM2-360M-Instruct` with a 7,680-parameter rank-4 personal residual adapter. Three synthetic English style-preference profiles, two random seeds each, 48 adapter steps per trial. Eighteen held-out advice prompts per trial; **108 responses for each mode but only three distinct style targets**. Six conditions: frozen base, explicit style instruction, zero-residual untrained adapter+instruction, trained adapter alone, trained adapter+instruction and gated trained adapter+instruction. The last activates trained adaptation only for recognized practical advice, bypassing it on general knowledge questions.

The registered primary output budget was 160 new tokens; shorter 76-token ablation uses the same task prompts. Strict versus permissive format scorers were fixed before testing. A lexical topic-keyword test is a **weak relevance proxy**, not human-scored helpfulness, correct factual content, or task completion.

## Primary 160-token results

| Mode | Strict format | Permissive format | Topical keyword match | Permissive format **and** keyword | Hit 160-token limit |
|---|---:|---:|---:|---:|---:|
| Frozen base | 0/108 (0.0%) | 0/108 (0.0%) | 60/108 (55.6%) | 0/108 (0.0%) | 84/108 |
| Explicit preference prompt | 46/108 (42.6%) | 74/108 (68.5%) | 78/108 (72.2%) | 54/108 (50.0%) | 10/108 |
| Untrained zero-residual adapter + prompt | 46/108 (42.6%) | 74/108 (68.5%) | 78/108 (72.2%) | 54/108 (50.0%) | 10/108 |
| Trained adapter alone | 5/108 (4.6%) | 5/108 (4.6%) | 67/108 (62.0%) | 2/108 (1.9%) | 50/108 |
| **Trained adapter + prompt** | **87/108 (80.6%)** | **88/108 (81.5%)** | 68/108 (63.0%) | **57/108 (52.8%)** | **7/108** |
| **Gated adapter + prompt** | **87/108 (80.6%)** | **88/108 (81.5%)** | 68/108 (63.0%) | **57/108 (52.8%)** | **7/108** |

**Critical nuance:** improved style adherence is clear in this benchmark (+14/108 on permissive style; +41/108 on strict style), but topical keyword coverage **fell by 10/108** (from 78 to 68). The combined format+keyword measure rose by only **3/108** (54 to 57), which is far too small to assert reliable net response-quality improvement.

### Same-item paired outcome counts

Comparing the identical 108 queries across all six profile/seed runs, with trained adapter+prompt versus prompt alone:

| Outcome metric | Both succeed | Prompt-only wins | Trained-adapter wins | Both fail |
|---|---:|---:|---:|---:|
| Permissive style | 70 | 4 | 18 | 16 |
| Topical keyword | 57 | 21 | 11 | 19 |
| **Format + keyword** | 42 | **12** | **15** | 39 |

A simplistic two-sided exact McNemar/binomial calculation is approximately **p=0.0043** for style and **p=0.701** for format+keyword. **Do not interpret these as independent-user statistical guarantees**: repeated prompts, correlated style families, only 3 styles and only 2 seeds per style violate a naïve independence interpretation; these p-values are exploratory only.

### Within each style family: joint format+keyword (two seeds, 36 answers per family)

| Requested response style | Explicit preference prompt | Trained adapter + prompt |
|---|---:|---:|
| Two brief bullets | **18/36** | **18/36** |
| Three numbered steps | **26/36** | 24/36 |
| One concise sentence | 10/36 | **15/36** |

The apparent net benefit is **not uniform across styles**. Format gains are larger than the net keyword gains.

## Token budget sensitivity

Matched 76-token generation ablation:

| Mode | Permissive style | Keyword match | Combined | Hit 76-token limit |
|---|---:|---:|---:|---:|
| Prompt alone | 48/108 | 74/108 | 32/108 | 44/108 |
| Trained adapter + prompt | 77/108 | 66/108 | 47/108 | 22/108 |

At 160 tokens, combined scores become 54 versus 57 and token-limit hits drop to 10 versus 7. **Most of the apparently large advantage at short budgets shrinks as generation limits are relaxed.** Shorter adapter outputs sometimes complete the requested layout faster. This can be useful but it is not evidence of higher informational quality.

## Negative controls, gating and general QA

- **No-op control:** untrained zero-residual adapter+prompt produced **identical raw text** to prompt-only in all 108 tests (0 output mismatches). This validates that the hook itself was not responsible for the change observed after training.
- **Gating equivalence:** trained adapter+prompt and gated+prompt produced **identical raw personal-task outputs** (0 output mismatches). For all 48 generic questions the gated path invoked the adapter **zero times**.
- Generic trivia keyword-based accuracy: frozen **36/48 (75%)**; always-on trained adapter **41/48 (85.4%)**; gated **36/48 (75%)**. Gating reproduces frozen baseline behavior by design; the trivia suite is too small and fragile to prove non-regression or general reasoning gains.

## Qualitative spot audit of locked sample

The fixed review artifact includes original generated responses for two predetermined tasks per style, seed 19.

- **Two-bullet example:** On shared-folder organization, explicit prompt-only produced two numbered, topic-relevant points and passed permissive format but failed strict format. Adapter+prompt produced two short dash bullets with comparable relevance; strict scorer now passes. This is a format improvement, **not necessarily a better answer**.
- **Three-step example:** On bug-report writing, adapter+prompt produced `Identify the issue / Describe the problem / Provide the necessary details`. This meets the surface format but can be less useful than the longer prompt-only explanation which mentions specific issues and information needed. The weak keyword test marks the adapted answer as missing reference anchors.
- **Artifact-only example:** Without an explicit style prompt, one trained adapter reply degenerated into repeated `0.0.0...` tokens on a project archiving task, confirming that adapter-only behavior is not yet robust.
- Even plausible relevant answers may fail the keyword proxy due to synonymous wording. The keyword metric must not be treated as semantic-quality ground truth.

## Memory lifecycle

Deterministic synthetic memory controls passed: preference update, ignoring hypothetical or third-party utterances, active-state deletion and temporary preference expiration. **Historical audit events were retained** and weights were not erased/retrained on deletion; production privacy deletion is **not implemented**.

## Research decision and V8 gates

1. **Preserve the positive narrow finding:** supervised residual adaptation plus explicit preference prompting makes SmolLM2 follow a **style layout** more often than prompting alone under matched 160-token budgets.
2. **Do not claim independent acquired preference memory:** adapter-only style accuracy remains 5/108 (4.6%), worse than even prompt-only; the explicit style note remains necessary.
3. **Do not claim better overall response quality:** 57/108 vs 54/108 combined surface-format-plus-keyword, with a 10/108 topical keyword regression and style-family heterogeneity, does not establish meaningful utility gains.
4. **Keep frozen + explicit memory as the deployment-oriented baseline**, with the gated adapter as an isolated optional research arm; do not merge into production/core 5M SELF.
5. **Next controlled study:** assess semantic usefulness via human-reviewed locked samples with independent scoring; compare task-specific controlled decoding/length-limited prompting to learned adapter; add strong relevance constraints (e.g. required subtopics) and paired eval across genuinely diverse users and naturally phrased histories.
6. Expand generic task tests; pursue robust memory-update, expiration and deletion policies including adapter retraining where appropriate.

**Result status:** V7 complete and archived; no V8 run launched by this report.
