# COMPASS V8.1 rank-4 contrastive pilot — completed and audited

**Execution:** [GitHub Actions run 38101573253](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38101573253), **SUCCESS**, completed 2026-10-11 01:26:17 UTC. Both data/safety validation and model-evaluation jobs passed. Retrieved and examined full `compass-v81-r4-s11.json` artifact, weights and per-item decisions; artifact **11688376472**.

**Scientific result: NO-GO for rank expansion.** The model trained correctly but the learned **personal COMPASS** still did not demonstrate conditional decision prioritization. The external guard did improve safety; this must not be credited to learned neural judgment.

## Setting and controls

- Frozen SmolLM2-360M-Instruct backbone; COMPASS low-rank residual rank 4 = **7,680 trainable weights**; seed 11, **48 paired optimizer steps**, two contrasting authored task examples per step.
- 192 fully fictional authored training tasks from 8 distinct synthetic policy profiles and 16 fictional project maps. A separately defined pool of 72 tasks from 3 other fictional policy profiles and 6 other projects; smoke evaluates **24** from **2** held-out projects. No overlap in literal profile IDs, project IDs or wording families.
- Six output labels were verified as distinct **single-token IDs**: `0..5` map to `DO_NOW,SCHEDULE,INVESTIGATE,DEFER,ASK_USER,REJECT`. Train and test use the same six-class next-token cross-entropy. No long-answer scoring, answer-length bias or candidate-dependent truncation. Nevertheless prior probabilities and model-vocabulary association still matter.
- Fixed shared neutral-prefix logit correction was an explicitly declared **secondary** diagnostic, not permission to cherry-pick.
- A strict **separate deterministic** scope/approval gate postprocesses proposals but does not alter learned classifications. All raw and post-guard scores kept distinct.

## Primary outcomes (24 same locked tasks)

| Method | Correct | Accuracy | Balanced class accuracy | Distinct actions predicted | Unsafe approvals/scope promotions |
|---|---:|---:|---:|---:|---:|
| Frozen backbone raw | 7/24 | 29.2% | 14.6% | 2/6 | 3 |
| **Trained COMPASS raw** | **7/24** | **29.2%** | **14.6%** | **2/6** | **4** |
| Trivial majority SCHEDULE always | **8/24** | **33.3%** | 16.7% | 1/6 | Not a safety policy |
| Frozen baseline + external permission/scope guard | 11/24 | 45.8% | 47.9% | 4/6 | 0 |
| **Trained COMPASS + external permission/scope guard** | **11/24** | **45.8%** | **47.9%** | **4/6** | **0** |
| Frozen neutral-logit calibrated | 4/24 | 16.7% | 16.7% | 1/6 | 0 |
| Trained COMPASS neutral-logit calibrated | 4/24 | 16.7% | 16.7% | 1/6 | 0 |

**Raw prediction histograms:** frozen `SCHEDULE=20, INVESTIGATE=4`; COMPASS `SCHEDULE=20, DO_NOW=4`. Calibration collapses *both* arms to `INVESTIGATE=24`, making the calibrated classifier no better than trivial constant prediction. The fixed calibration therefore does not rescue the study.

**Per-category recall, trained raw:** DO_NOW **0/6**, SCHEDULE **7/8**, INVESTIGATE **0/4**, DEFER **0/2**, ASK_USER **0/2**, REJECT **0/2**. Exactly **one of six actions** has any correctly answered cases. NOTE: the model *predicts* two categories (DO_NOW and SCHEDULE), but DO_NOW predictions are all mistakes, so it has no usable evidence-sensitive preference shift.

**Paired contrast performance:** ZERO successful pairs for all six contrast types, across two held-out projects (**0/12 total**) in every unguarded trained/frozen/calibrated arm. Pairs cover: confirmed vs unverified failure; targeted vs broad checks; core-blocking vs routine prerequisite; approved vs unapproved project mission revision; authorized vs forbidden scope expansion; cosmetics vs unverified suggestion. The guarded version only solves **4/12 pairs** (2 approval pairs + 2 scope pairs) due solely to explicit authoritative constraint enforcement; it still misses core-blocking/ancillary discrimination.

**External authorization guard effect:** Exactly **four** trained raw actions were overridden on the held-out test: two `SCHEDULE → ASK_USER` for unapproved core mission changes and two `DO_NOW → REJECT` for explicitly forbidden scope. The guard raises trained accuracy **7→11/24** without changing model weights. It is a sound prototype for rule-governed action gating on explicitly trusted fields, but does not prove policy learning; real deployment requires trusted authorization sources and executor-level enforcement.

**Optimization diagnostic:** cross-entropy averaged 1.887 over first 12 steps and 1.738 over last 12; `ln(6)=1.792` is uniform-choice benchmark. Individual losses were noisy, with last training step 1.905. This weak training trend does not imply successful out-of-project generalization.

## Locked gate decision

- Preflight integrity / independent fictional split: **PASS** (not independent human adjudication).
- Stable single-token labels and matching six-category train/eval objective: **PASS**.
- External deterministic project/scope/authorization guard tests: **PASS**.
- Learned raw action accuracy above frozen and trivial majority: **FAIL** (7/24 = 7/24, both below 8/24).
- Learned action diversity and recall for critical ASK_USER, REJECT, DO_NOW: **FAIL**.
- Core-vs-periphery counterfactual pair discrimination: **FAIL** (0/12).
- Neutral-context logit calibration: **FAIL** (all INVESTIGATE).
- Free-form grounded decision rationales or actual real-user policies: **NOT TESTED**.
- Rank 16 and 64 in V8.1: **NOT RUN**. No reason to assume more adapter parameters fix a dataset/objective issue.

**Decision: keep frozen project memory + explicit approved rules and guarded execution as present baseline. Do not promote the adapter or merge to SELF/production.**

## Proposed V8.2 study (NOT launched)

Before another parameter matrix, determine why synthetic categorization fails:
1. Test the frozen model's performance on *short, natural-language next-action explanations* versus fixed digit codes. Confirm the prompt has semantically meaningful examples and can discriminate on single synthetic case families.
2. Build an *independent* curated benchmark with explicit project core, evidence provenance and mutable approvals, in **Hebrew and English**, reflecting user intent but without using private project data without explicit authorization.
3. Include training and **development** sets for objective/label calibration plus a separate truly locked test set; report within-profile as well as cold-profile transfer, not just cold transfer after 48 steps.
4. Establish a deterministic graph-based decision executor as the safe baseline and use COMPASS only as a proposal ranker; train preference pairs with ranking loss or a small classification head rather than merely predicting code-token logits if warranted.
5. Run per-category balanced evaluation and enforce a model-side minimum: correct decisions in **all** six action categories and enough matched core-vs-ancillary contrast pairs. No expansion to ranks 16/64 until smaller experiments show stable improvement over explicit memory and rules across seeds.

No private data was used in these fictional benchmark runs; the historic V8 and independent model-internal SELF artifacts are unaffected.
