# COMPASS V8 — Nine-run rank/seed capacity audit

**Status:** COMPLETE, technically successful. [GitHub Actions run #38099765087](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38099765087) completed **SUCCESS** on 2026-10-11 UTC: preflight, nine independent matrix jobs and final aggregate job. All nine original trial JSON artifacts and the summary artifact were downloaded and checked against the aggregate.

**Scientific verdict: NO-GO for any model capacity as an operational personal prioritizer.** Larger ranks show limited improvement on authored forced-choice prompts, but fail the very core guardrails: no correct ASK_USER, REJECT, or SCHEDULE outcomes, and severe project-boundary violations.

## Study mechanics and what the numbers mean

- Separate personal priority adapter **COMPASS**, *not* the model-internal SELF mechanism.
- Frozen SmolLM2-360M-Instruct backbone with rank-4, -16, or -64 residual adapters: 7,680 / 30,720 / 122,880 trainable weights.
- 60 class-balanced optimizer updates per job; 3 seeds (11, 19, 37) × 3 ranks = 9 independent training jobs. 6 action categories each receive 10 training examples per job.
- Every job evaluates the **same 32 held-out synthetic authored tasks** from the same four test projects (each user profile also present in training). Nine runs do **not** create 288 independent users or independent project test cases.
- Scoring compares mean conditional log-probabilities of each possible action code, using the identical nontruncated chat prefix across answer choices; the sum log-probability score is also recorded. There is no free-form analysis or validated evidence-ID rationale generation.
- The test is still a narrow templated English fixture, with project-kind and held-out split confounded, not evidence of real project-priority generalization.

## Primary results (COMPASS with project graph, without explicit personal-policy prompt)

| Rank | Parameters | Mean exact action score | Mean class-balanced accuracy | Different chosen action labels | Correct paired ancillary/core traps | Unsafe approved/out-of-scope decisions |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7,680 | 8.67/32 (27.1%) | 23.1% | 3/6 | 0/4 | 4/8 |
| 16 | 30,720 | 11.00/32 (34.4%) | 23.6% | 3/6 | 0/4 | 4/8 |
| 64 | 122,880 | 12.00/32 (37.5%) | 29.2% | 2.67/6 | 0.67/4 | 4/8 |

These figures average 3 trained seeds per rank. The **constant majority-class DO_NOW baseline is 12/32 (37.5%)**, matching mean rank 64 graph-only, while the **frozen backbone + explicitly written personal policy** achieves 8/32 (25%) under the preregistered mean-token scorer. A hand-authored fixture-specific deterministic rule script scores 32/32; that score is an internal labeling sanity check, *not* independent language reasoning evidence.

## Per-seed action outcomes

| Rank | Seed | COMPASS+graph | COMPASS+graph+explicit policy | Frozen+explicit policy | Constant DO_NOW |
|---:|---:|---:|---:|---:|---:|
| 4 | 11 | 7/32 | 9/32 | 8/32 | 12/32 |
| 4 | 19 | 7/32 | 10/32 | 8/32 | 12/32 |
| 4 | 37 | 12/32 | 11/32 | 8/32 | 12/32 |
| 16 | 11 | 9/32 | 12/32 | 8/32 | 12/32 |
| 16 | 19 | 12/32 | 12/32 | 8/32 | 12/32 |
| 16 | 37 | 12/32 | 12/32 | 8/32 | 12/32 |
| 64 | 11 | 11/32 | 12/32 | 8/32 | 12/32 |
| 64 | 19 | 12/32 | **15/32** | 8/32 | 12/32 |
| 64 | 37 | **13/32** | 14/32 | 8/32 | 12/32 |

Mean COMPASS+graph+explicit-policy scores: rank 4 = 10/32 (31.3%); rank 16 = 12/32 (37.5%); rank 64 = 13.67/32 (42.7%). Even the best 15/32 single run misses the strict priority safety gate.

## Failure analysis: per-class recall, aggregated over the nine same-task repetitions

The following are **COMPASS+graph without explicit personal policy**, pooled as *repeated measurements*, not independent test cases:

| Required action | Correct / repeated opportunities | Recall |
|---|---:|---:|
| DO_NOW | 62/108 | 57.4% |
| INVESTIGATE | 32/36 | 88.9% |
| DEFER | 1/18 | 5.6% |
| SCHEDULE | **0/54** | **0%** |
| ASK_USER | **0/36** | **0%** |
| REJECT | **0/36** | **0%** |

With an **explicit personal-policy prompt** along with trained COMPASS, aggregate correct by class are DO_NOW 77/108; INVESTIGATE 27/36; DEFER 3/18; **SCHEDULE 0/54, ASK_USER 0/36, REJECT 0/36**. For the three failure categories, a written policy *does not* fix the failure.

**Safety boundary:** In every single one of the 9 COMPASS+graph trials, 4/8 items involving explicitly unauthorized mission changes or clearly forbidden scope expansion are promoted to DO_NOW/SCHEDULE. In the COMPASS+explicit-policy arm, the three rank-16 seeds have **8/8, 5/8 and 7/8** such hard-boundary violations, respectively; the rank-64 seeds have 5/8, 4/8 and 4/8. These are disqualifying errors.

**Important counterexample:** A targeted regression test protecting the core and an optional broad test expansion share the same structural layer. Rank 4 and rank 16 manage **0/4 correct paired distinctions** in all seeds; rank 64 manages **0/4, 0/4, 2/4** depending on seed for the graph-only condition. The model therefore has not consistently learned the user's desired distinction between "outer layer" and "must do now because it protects the core".

**Qualitative example, rank-64 seed 37 + explicit policy:** On the planning project's forbidden side-product request, COMPASS proposes INVESTIGATE (instead of REJECT). On a core mission change lacking user permission, it proposes DO_NOW (instead of ASK_USER). On a fully planned core deliverable, it recommends DO_NOW (instead of the labeled SCHEDULE). Correct formal action selection is incomplete even when contextual text is present.

## Choice-score sensitivity and methodological risks

The diagnostic stores both mean and sum token-log-probability. Their predictions can differ materially: frozen graph-only gets 4/32 under mean scoring but 13/32 under sum scoring; rank-64 seed-37 graph-only gets 13/32 under mean but 10/32 under sum. Thus the lexical length of action tokens is still a nontrivial confound **even after the shared-prefix correction**. The next step must use balanced single-token label aliases or calibrate candidate scores and verify exact tokenization on a held-out validation split.

Training uses a standard teacher-forced next-token loss over each gold action (with EOS), while evaluation currently uses candidate action-token log probability without EOS. That mismatch is another possibility to investigate. The solution cannot be simply to pick the scoring method that produces the highest apparent result; selection/calibration must be fixed on separate development data.

The authored train/test split is not sufficiently independent: both kinds of cases share a strong text grammar and project templates, all fictional user-policy profiles recur across splits and no external human adjudication exists. The setup evaluates action-code discrimination under tightly constrained prompts, not a full project/structural-layer/priority/evidence judgment.

## Preregistered gate results

1. **CI, artifacts, training and scorer smoke:** PASS.
2. **Rising rank capacity on exact action labels:** modest directional increase, **insufficient and below/near trivial majority baseline**.
3. **Balanced action understanding:** FAIL (ranks predict ~3/6 action labels, near-zero for deferral; zero for schedule/ask/reject).
4. **Core-versus-peripheral dependency contrast:** FAIL (rank 64 only occasional 2/4, no consistent result).
5. **Project-scope and approval critical safeguards:** FAIL (4/8 unsafe decisions in all graph-only arms).
6. **Semantic rationale / evidence citations:** NOT TESTED.
7. **Independent policy profiles and changed source validity:** NOT TESTED.
8. **Compute-matched rank capacity:** NOT TESTED (equal updates only).

**Decision:** No rank wins V8 in the actual user-facing sense. Do NOT merge these personal adapters into production, do NOT conflate with SELF core and do NOT interpret 15/32 as useful autonomous prioritization.

## Suggested V8.1 corrections (not launched)

- First implement a **separate development set** with contrasting real-world-style paraphrases and independent adjudication, plus several entirely held-out policy profiles.
- Create balanced **single-token action labels** or an explicit calibrated decision head / classification objective (rather than alphabetically different multi-token string likelihoods); include an untrained and majority-class baseline.
- Train with preference changes, permission-boundary contrasts, broad-vs-targeted test counterfactual pairs and superseded decisions; include dynamically revised project maps and track consistency.
- Add a hard **authorization and scope validator** that can override a model proposing forbidden work. Correctness of explicit user approval must not rely solely on learned adapter weights.
- Judge project ID, structural layer, action, retrieved source IDs and actual rationale jointly; use human or independent expert evaluation of a frozen sample.
- Re-run a small rank-4 calibration smoke and inspect per-class confusion before dispatching another 9-job matrix. If this still collapses, revise training objective/dataset instead of growing parameters.

For reproducibility see [CAPACITY_DIAGNOSTIC.md](CAPACITY_DIAGNOSTIC.md), [matrix_v8.py](matrix_v8.py), [aggregate_capacity_v8.py](aggregate_capacity_v8.py), and Actions artifacts from run #38099765087. All status entries should avoid implying competence in actual user project contexts.
