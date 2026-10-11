# COMPASS V8 rank-4 pilot — completed outcome and diagnostic

**Official GitHub Actions:** [Run 38098918328](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098918328) **SUCCESS**. Stage-0 revalidation passed; pilot job completed. Artifact `compass-v8-rank4-pilot-seed11` (ID `11687203549`) contains the JSON with all 16 predictions and candidate negative log-loss values plus the adapter weights. Results were verified against the worker logs **and** downloaded trial JSON.

## Setup

Frozen `HuggingFaceTB/SmolLM2-360M-Instruct` with a rank-4 7,680-parameter residual COMPASS adapter, random seed 11, 32 optimizer updates over 32 of the 64 authored train fixtures. Six-option forced-choice ranking based on *per-answer mean token loss including EOS*. The evaluation is 16 authored test scenarios from four fictional users, all on the test-only `planning` project kind. All language-model arms see the same project graph, with or without the explicit fictional personal policy. **No full multi-rank run was launched** in this pilot.

## Measured outcomes

| Condition | Exact action | Rate |
|---|---:|---:|
| Frozen graph only | 6/16 | 37.5% |
| Frozen graph + explicit personal policy | 5/16 | 31.25% |
| Trained adapter + graph | 8/16 | 50.0% |
| Trained adapter + graph + explicit personal policy | 8/16 | 50.0% |
| Deterministic template-rule sanity control | 16/16 | 100%, not evidence of model generalization |

### Category-level issue — the decisive finding

Gold-label distribution: `DO_NOW` 8, `INVESTIGATE` 4, `DEFER` 2, `SCHEDULE` 2. Thus a constant `DO_NOW` prediction already achieves **8/16 (50%)** with zero context-sensitive reasoning.

- Both trained COMPASS arms predict **DO_NOW in all 16 test cases**. Correct only for the eight `DO_NOW` gold labels, including four urgent targeted/core-repair cases. They fail **every** non-`DO_NOW` situation.
- Frozen graph-only predicts `INVESTIGATE` in 14 cases and `DO_NOW` in 2, producing 6/16. Frozen graph+policy predicts `INVESTIGATE` in 15 and `DO_NOW` in 1, producing 5/16.
- The eight trained-COMPASS errors cover every `broad_test` (`DEFER` or `SCHEDULE`) and every `suspected_issue` (`INVESTIGATE`). These distinctions are at the heart of the user-defined COMPASS objective.
- Recorded loss declined within the short run on several examples (e.g. step 8 1.307, step 16 1.160, step 24 1.024, step 32 0.941), but the full loss sequence is nonmonotonic. It does not establish a learned decision policy.
- Final 16-case evaluation used only four of eight authored case kinds. The 16/16 rule baseline is authored against this same synthetic grammar and is not a statistically independent oracle.

## Methodological problems to fix before a 3-rank × 3-seed matrix

1. **Constant-action baseline missing.** It should be shown next to model accuracy and balanced/macro accuracy. On this exact test, constant `DO_NOW` reaches 50%, so COMPASS adds **zero improvement over a trivial majority-action baseline**.
2. **Option-score calibration.** The six candidate answer strings differ in tokenization and length. The current `encode_pair` applies candidate-dependent context truncation (max_length=512) and includes EOS in each mean-token loss. The forced-choice score may favor lexical/token-length properties over project-specific evidence. Prefer computing the exact **common prefix once**, option token conditional log-probability on identical prefix, reporting length-normalized versus total log-probability, and applying any calibration on training/validation data only.
3. **The model often collapses to a single action.** Fix or explicitly measure per-class recall, balanced accuracy, macro-F1, action entropy, rejection/abstention, and paired evidence-flip tests.
4. **Very small and confounded holdout.** One seed, 16 authored English cases, four styles of case, and project-kind=planning exclusive to test. Train/test project split and wording split are not a sufficiently independent generalization test.
5. **Opaque scored behavior.** No validated structured free-form decision with grounded evidence/rationale was generated; a forced-choice score by itself does not demonstrate correct project hierarchy reasoning.

## Recommendation / sequencing

**Pilot pipeline worked, but decision-quality gate failed.** Do not spend nine jobs on ranks 4/16/64 using this scoring protocol yet. Before ranking capacity, implement a frozen common-prefix candidate scorer, a constant-action baseline and class-balanced / counterfactual diagnostics, then re-run the rank-4 smoke (at least two seeds) to verify there is information-sensitive classification rather than a collapsed preference. Once diagnostics and an independently audited held-out set exist, run all three ranks and seeds in parallel as the user suggested.

Do not merge this pilot adapter into production; keep COMPASS separate from model-internal SELF.
