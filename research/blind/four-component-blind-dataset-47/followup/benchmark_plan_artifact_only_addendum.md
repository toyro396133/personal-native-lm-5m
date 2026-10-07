# Blind Follow-up Benchmark Plan: Artifact-Only Addendum

Frozen before execution of these analyses.

## Why this addendum exists
The preregistered causal benchmark plan requires anonymized access to checkpoint weights and a blind-safe execution adapter mapping V1–V7 and A/B/C/D without exposing their real identities. The repository tree contains no weight-like checkpoint blobs, the designated blind run contains only `four-component-blind-dataset-47`, and no safe blind manifest was found. Therefore the causal patching/intervention plan remains preregistered but **not executable without violating blindness**.

This addendum preregisters five independent falsification benchmarks that use only primitive measurements already present in the blind artifact. They do not claim to replace activation patching.

## F1. Direction-vs-magnitude residual benchmark

**Hypothesis being tested**: A has direction/sign-specific effects beyond generic perturbation magnitude.

**Competing explanation**: B effects under A intervention are fully explained by intervention effect norm.

**Prediction**: after controlling for primitive `mean_effect_norm` with checkpoint-layer fixed effects, `negate` and/or `shuffle` mode retains a non-zero association with B intervention accuracy relative to `zero`.

**Falsifier**: both mode coefficients are near zero and the magnitude-only model performs equivalently or better.

**Exact implementation**: pool V2–V7 layer records. One row per checkpoint-layer-mode. Outcome = `B_intervention_accuracy_by_A_mode[mode]`. Predictor = `A_intervention_geometry[mode].mean_effect_norm`, mode indicators, and fixed effects for checkpoint-layer. Fit OLS by least squares; cluster bootstrap by checkpoint-layer for coefficient CIs. Compare residual SSE and leave-one-group-out prediction between magnitude-only and magnitude+mode models.

**Raw metrics**: B accuracy, effect norm, axis consistency, mode, fixed-effect group, residuals, coefficients, bootstrap CIs, SSE.

## F2. Learned-reference specificity benchmark

**Hypothesis being tested**: the specific learned A reference sometimes matters independently of mere signal presence.

**Competing explanation**: learned, shuffled, and random references are interchangeable once in-distribution.

**Prediction**: learned minus mean(shuffle, random) margins show systematic non-zero effects, with variant-specific sign; V5 is predicted negative for C, V4/V6 more positive.

**Falsifier**: learned advantages are centered near zero with no sign persistence beyond shuffle-vs-random noise.

**Exact implementation**: use primitive `relative_B_margin_by_reference` and `relative_C_relation_margin_by_reference` at every eligible layer. Compute paired learned - mean(shuffle,random), learned-shuffle, learned-random, and shuffle-random. Bootstrap by checkpoint and count sign persistence per variant/token/layer.

**Raw metrics**: all three reference margins and paired differences.

## F3. Mechanism-parameter coupling and lag benchmark

**Hypothesis being tested**: A's downstream effect is associated with a structured internal mechanism rather than vector norm alone; later amplification may be preceded by layer-parameter changes.

**Competing explanation**: gates/path norms are incidental scale parameters unrelated to measured A effects.

**Prediction**: within-checkpoint centered gate/path norms correlate with per-layer A effect magnitude, and at least one lagged parameter at layer L predicts effect at L+1 better than chance/permuted layer order.

**Falsifier**: centered correlations are near zero and lagged prediction is no better than random layer permutations.

**Exact implementation**: V2–V7, layers 1–6 only. Join blind `layer_parameters` with primitive per-layer A effect magnitude. Center each variable within checkpoint. Compute Spearman/Pearson correlations. Fit standardized linear regression for same-layer and one-layer-lag prediction. Permute layer order within checkpoint 5000 times for lag null.

**Raw metrics**: gate, path_up_norm, path_down_norm, A_vector_norm, per-layer A effect magnitude, centered values, regression coefficients, permutation statistic.

## F4. C-versus-D challenge asymmetry benchmark

**Hypothesis being tested**: D is a broader/more disruptive challenge to B than C, not just an isolated outlier.

**Competing explanation**: the apparent D difficulty is driven by a few variants/token points or baseline difficulty.

**Prediction**: paired `B_under_D_accuracy - B_under_C_accuracy` is predominantly negative across eligible layer records and remains negative within multiple variants and token stages.

**Falsifier**: pooled difference is near zero, reverses after token/variant centering, or is concentrated in <=2 variants.

**Exact implementation**: compute paired difference at every layer/checkpoint. Report pooled mean/median, bootstrap CI by checkpoint, sign persistence, per-variant and per-token summaries, and centered regression with variant+token fixed effects.

**Raw metrics**: B_under_C_accuracy, B_under_D_accuracy, difference.

## F5. Early fingerprint stability benchmark

**Hypothesis being tested**: variant-specific behavior is substantially established by 25M/50M and predicts 120M family geometry.

**Competing explanation**: apparent families are late-emerging or metric-specific.

**Prediction**: pairwise variant-distance geometry at 25M and/or 50M correlates positively with 120M across multiple primitive metric subsets.

**Falsifier**: early-to-late distance-matrix correlations are unstable across metric subsets or not above label-permutation null.

**Exact implementation**: for each variant/token construct a standardized fingerprint from primitive-only fields: final A effect magnitude, final A axis consistency, final B_condition_1, B_under_C, B_under_D, final negate-zero B accuracy delta, final shuffle-zero B accuracy delta, learned-A B reference delta, learned-A C reference delta, layer-of-max A effect encoded 1..7, and final/peak A-effect retention. Standardize metrics within each token count. Compute Euclidean pairwise distance matrices across variants. Correlate upper triangles for 25->120 and 50->120 using Spearman and Pearson. Permute variant labels at early time 10000 times. Repeat on three subsets: intervention, reference, and depth/geometry-accessibility primitives.

**Raw metrics**: fingerprint table, standardized features, pairwise distances, correlations, permutation p-values.

## Interpretation guardrails
- These are falsification analyses on existing blind primitives, not newly generated interventions.
- No result here can establish causality for B/C/D interventions.
- The original causal plan remains open and must be run only if an anonymized checkpoint/runner interface becomes available.