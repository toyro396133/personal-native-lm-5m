# COMPASS V8.5 — three EXACT personal model sizes, nine seeds: completed findings

**GitHub Actions run:** [38108048987](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38108048987), attempt **3**, ultimately **SUCCESS**, completed on 2026-10-11. All 9 trained personal heads, the frozen shared embeddings and the aggregate comparison succeeded. **Prior technical problems were fixed; these are genuine computed outputs, not extrapolated numbers.** Aggregated artifact `compass-v85-three-sizes-summary` ID **11691422538** contains JSON/Markdown; per-arm JSON/weights and reusable frozen embeddings remain downloadable in the same Actions run. See [registered design](EXPERIMENT_V8_5.md).

## Experimental question / exact personal model sizes

Does raising **the trainable size of the PERSONAL COMPASS component itself**, using the V8.4-style shared action scoring framework, improve decisions? The **shared frozen** SmolLM2-360M-Instruct backbone remains fixed in all arms. The personal scorer has two learned bias-free 960×rank matrices, so the trained parameter count is **exactly**:

| Personal rank | Learned personal weights |
|---:|---:|
| 4 | **7,680** |
| 16 | **30,720** |
| 64 | **122,880** |

This is a different scoring-head architecture from V8.4's former 3,848-weight `960→4→1` head. Old V8.4 numbers must NOT be presented as an apples-to-apples capacity baseline.

**Controls:** fixed set of expensive frozen candidate embeddings for all nine arms, matched train/development/evaluation splits, same 14 epochs and optimizer updates per seed, three seeds **11, 19, 37**, fixed project-scoped optimizer train pairs. Matched frozen single-option Yes-minus-No control was **64/144 (44.4%)** on this test — a weak baseline, below an always-position predictor at **72/144 (50%)**. Comparisons to more powerful rule/prompt/classifier baselines still needed.

## Synthetic study design

- Model learned only from synthetic fictional data: **640 train**, **128 development** for epoch selection; **144 locked test** decisions in two unseen fictional projects (72 English + 72 Hebrew), with 36 fact×option-order quadruples and 72 fact reversal pairs.
- Two *genuinely different fictional policy wordings*, rapid vs thorough, with separate policy preference families and binding evidence overrides. These are synthetic instructions, not actual user's personal decisions. Policy and test project are **confounded**: only one distinct test project was used per preference type, so changing only policy while keeping project constant remains untested.
- Six older dilemma families, two preference-sensitive families present in train, and one **new heldout-only** case family: an information source that later explicitly withdrew its evidence. Old family wording and semantic structure overlap between training and testing.
- All labels and fictional evidence authored by the experiment workflow; no independent person adjudicated gold correctness. Cases in state/order quartets are correlated rather than independent end-user choices.
- Deterministic scope/owner approval postprocessing (separate guard) is never counted as learned model accuracy.

## Complete rank×seed results

| Personal parameters | Seed 11 raw /144 | Seed 19 raw /144 | Seed 37 raw /144 | Three-seed mean raw /144 | Mean accuracy |
|---:|---:|---:|---:|---:|---:|
| **7,680** | 124 | 120 | 120 | **121.33** | **84.26%** |
| **30,720** | 122 | 124 | 122 | **122.67** | **85.19%** |
| **122,880** | 122 | 124 | 126 | **124.00** | **86.11%** |

The highest individual run is **122,880 / seed 37: 126/144 = 87.5%**, but the 3-seed ranges overlap heavily. The 16× parameter increase versus 7,680 yields just **+2.67/144** average correct items (+1.85 percentage points). **NO robust causal or statistically meaningful capacity advantage is established**; larger models entail more trainable weights and CPU cost.

## Other heldout metrics averaged over three seeds

| Learned personal weights | Fact reversals correct /72 | Complete four-way groups /36 | English correct /72 | Hebrew correct /72 | NEW source-withdrawal correct /16 |
|---:|---:|---:|---:|---:|---:|
| **7,680** | 50.00 | 25.00 | 64.00 | 57.33 | **8.00** |
| **30,720** | 50.67 | 25.33 | 64.67 | 58.00 | **8.00** |
| **122,880** | 52.00 | 26.00 | 65.33 | 58.67 | **8.00** |

Registered synthetic screening metrics such as ≥84/144 raw correct, ≥30/72 correct fact reversals, ≥16/36 full quadruples, and ≥44/72 per language are met by all nine arms. But high aggregate accuracy **conceals a critical generalization failure** in the genuinely unseen family.

### New family source withdrawal: **universally failed evidence reversal**

All nine heads produce exactly **8/16** on previously unseen `source_retracted` cases (50%; chance in these balanced opposites). Original per-decision JSON was inspected for rank4/seed11, rank16/seed11 and rank64/seed37. In **every** such inspected arm:

- **State 0**: source remains independently verified — **8/8** correct.
- **State 1**: source has *withdrawn* evidence and there is no verified replacement — **0/8** correct.
- The digits consistently reverse when the option order swaps, but **the underlying semantic choice does not flip when the source loses validity**.
- Aggregate 8/16 for all nine arms implies a consistent failure in this unseen family; the exact 8/8 vs 0/8 decomposition was explicitly verified in three original per-run artifacts, not separately checked at the per-row level for the other six.

This is a decisive **NO-GO for trusting the personal model to autonomously respond to withdrawn evidence or stale source validity**, regardless of model size. It is *not* fixed by the synthetic external guard, since the heldout source-withdrawal family has no independent enforced source-validity rule. Production requires hard source provenance/version checks and explicit revalidation.

### Familiar dilemma performance, and the genuine personalization caveat

For the best individual run (122,880, seed37), a per-family audit yields 16/16 for core-change approval, dependency blocking, scope, revision and targeted QA, 14/16 for confirmed fault and thorough-coverage preference, **10/16** for quick-vs-thorough rollout preference, and **8/16** for novel source withdrawal. Strong scores therefore remain uneven.

Average heldout accuracy on the eight *already trained* scenario families (128 items) is **113.33 /128**, **114.67 /128**, and **116/128** for the three personal sizes. The unfamiliar source family remains **8/16** for every size/seed. Identical templates and two projects severely limit any extrapolation to real-world projects.

There are two legitimately differing fictional policies, but every example from one heldout project contains only one policy mode. In order to claim context-sensitive personalization, a future evaluation must hold **project, evidence and candidate choices fixed** and flip **only** the authorized personal policy; measure how often the model correctly reverses its preference.

## Execution log and transparent repair history

Attempt 1 of Actions run 38108048987:
- Initial preflight PASS; CPU extraction of frozen option embeddings SUCCESS (artifact `compass-v85-frozen-embeddings`, ID **11691476708**).
- All nine training jobs failed in the inherited breakdown scorer due to **KeyError: `project_id`**. No result was fabricated.
- Investigation also found the borrowed V8.4 loss wrongly hard-coded **state0→action0** regardless of fictitious owner preference. Corrected to **state-specific gold label** plus factual-reversal consistency. Commit **43e08b12f02feb18cdec60495c3e8a5f72bf59e0**. The cached embeddings and ground truth were unchanged.
- Attempt 2 reran the failed matrix jobs, successfully completing **all nine personal arms** and preserving their individual JSON and safetensors checkpoints.
- Aggregation failed only because it checked `len(policy_stats_dict)` (4 keys) instead of the `total` item count (72). Fixed in commit **76faca2cfd4b459d4908e29059bb56f02c161411**. The matrix trained models were *not* retrained.
- Attempt 3 reran only the aggregation job, which succeeded. The final overall run is **SUCCESS**.

## Research decision

**Technical objective: COMPLETE.** All three **exact personal parameter capacities** and three seeds were run; no native model/SELF changes; reliable data and weight artifacts exist.

**Scientific verdict: POSITIVE ON FAMILIAR SYNTHETIC DECISION FAMILIES; NO-GO FOR GENERAL AUTONOMOUS PRIORITY INTELLIGENCE.** Gains above frozen and constant baselines are promising but do not establish external validity, individual personalization or retracted-evidence awareness. The largest head leads this artificial benchmark by a tiny amount that does **not** justify its 16-fold parameter cost today.

**Pragmatic next step:** keep **7,680 personal weights** as the efficiency-oriented experimental baseline, preserve 30,720 and 122,880 as comparison checkpoints, and build a new independent human-reviewed/externally authored cold decision-family validation set. Test source withdrawal, retracted authority, updated approvals, actual policy-only flips and mixed Hebrew/English evidence. Test against deterministic graph/authorization/source-validity rules and stronger calibrated frozen baselines. Repeat more seeds only after generalization is present. Do not merge research heads into production or native SELF.
