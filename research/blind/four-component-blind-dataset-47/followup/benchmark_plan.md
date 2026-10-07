# Blind Follow-up Independent Benchmark Plan

Frozen before follow-up execution. Predictions below must not be edited after results are observed.

## General rules
- Use existing checkpoints only; no training and no weight updates.
- Operate only through anonymized IDs A/B/C/D and V1–V7.
- Never inspect real component/variant/checkpoint names.
- Store primitive measurements. Any composite is secondary and documented exactly.
- Primary causal effect = paired change against an unperturbed matched baseline on the same input.
- Report effect distributions, medians, bootstrap CIs, sign persistence, per-layer curves, and per-variant heterogeneity.
- Where possible, match perturbation L2 norm and angular displacement.
- Random donor/reference selections use frozen seeds 104729, 104759, 104761, 104773, 104779.
- Primary checkpoints for complete comparisons: 25M, 50M, 75M, 100M, 110M, 120M. 130M remains partial and descriptive only.

# Benchmark B1: Magnitude- and angle-matched component intervention matrix

**Hypothesis being tested**
A is direction/reference-sensitive rather than merely magnitude-sensitive; B is a stable anchor; C and D have distinct causal footprints.

**Competing explanations**
E1: all apparent effects are generic activation magnitude/noise.
E2: sign matters but identity/direction does not.
E3: each component has a selective causal footprint.

**Predictions**
- Under current interpretation, same-norm orthogonal/random perturbations to A should differ from same-norm opposite-direction perturbations in downstream readouts.
- B intervention should most directly alter B-related primitives.
- C and D intervention footprints should differ after perturbation magnitude is matched.

**Falsification**
A interpretation is falsified if downstream effects are explained by perturbation norm regardless of direction/sign/identity. B-anchor interpretation weakens if B intervention is not uniquely strongest on B-related readouts.

**Exact implementation**
At each selectable layer and checkpoint, estimate the native component vector for each sample. Apply five paired perturbations with identical target L2 displacement: +random orthogonal, -native/opposite, angular rotation by fixed theta, interpolation toward zero, and extrapolation past zero. Repeat for A/B/C/D. Record downstream layer displacements and blinded task/readout primitives. Use the same samples and displacement magnitudes across perturbation types.

**Raw metrics**
L2 displacement; cosine before/after; angular change; downstream component-aligned projection change; per-component probe/logit accuracy where available; error type counts; paired output divergence; layer-by-layer propagation.

# Benchmark B2: Reference substitution and donor identity test

**Hypothesis being tested**
A may be a reference/control signal whose specific learned value sometimes matters; B may be identity-like; C may be relational to B.

**Competing explanations**
E1: only signal presence/norm matters.
E2: any in-distribution donor works.
E3: donor identity and matching relation matter selectively.

**Predictions**
- For A, donor substitutions may preserve broad effect while changing learned-value-specific outcomes.
- For B, mismatched-B donors should produce larger B-related changes than matched-B donors.
- For C, donor effects should depend on whether the donor preserves the B relation.

**Falsification**
Reference interpretation weakens if all donors of matched norm are interchangeable. C-relational interpretation weakens if preserving/changing B relation has no differential effect.

**Exact implementation**
Patch each component from a donor example into a base example at matched layers. Donor strata are defined only by anonymized equality relations available to the blind runner: same blinded B/different C, same C/different B, same local context/different component value, and fully different context. If a stratum cannot be selected without semantic unblinding, omit it and record the limitation rather than infer names.

**Raw metrics**
Paired downstream displacement, component-projection changes, blinded task/readout accuracies, donor-base cosine, donor norm, output divergence, error categories.

# Benchmark B3: Intermediate-layer causal patching / retention test

**Hypothesis being tested**
The strong intermediate-layer geometry is functionally used, not merely decodable; layer 2 is a transient bottleneck and layers 3–5 carry causal leverage.

**Competing explanations**
E1: intermediate geometry is epiphenomenal.
E2: only final representation matters.
E3: causal leverage peaks in the middle layers and is transformed before final.

**Predictions**
Patching a controlled component representation at layers 3–5 should yield larger selective downstream changes than equally sized patches at layer 1 or final; layer 2 may show compression but not maximum leverage.

**Falsification**
Depth-chain interpretation is falsified if causal effect monotonically follows layer depth or only final has meaningful leverage, after magnitude matching.

**Exact implementation**
For each component, patch base-example activations with donor activations separately at each layer. Normalize patch displacement to a common L2 target. Measure propagation to subsequent layers and final outputs. Compare causal-effect layer-of-maximum with the first-study decode/geometry maxima without using the old composite scores as evidence.

**Raw metrics**
Layer-local displacement, downstream retained displacement, final paired output divergence, component-specific accuracy delta, effect layer-of-maximum, best-layer-to-final causal retention.

# Benchmark B4: Pairwise conditional/counterfactual composition matrix

**Hypothesis being tested**
B is upstream/stable relative to C; A modulates rather than generates B; D is more entangled/competitive.

**Competing explanations**
E1: all four factors are symmetric and independent.
E2: observed relations are probe geometry only.
E3: directed or conditional dependencies exist.

**Predictions**
B->C manipulation should have a different footprint from C->B manipulation; A->B should be weaker than B->A if B truly exists independently of A; D should cause broader off-target effects than C if it is more entangled.

**Falsification**
The proposed hierarchy fails if pairwise intervention effects are symmetric within uncertainty and conditioning on third components removes the apparent asymmetries.

**Exact implementation**
Construct all ordered pairs X->Y over A/B/C/D. Patch X while holding the other anonymized conditions constant when the blind runner can guarantee equality. Record Y-related and off-target changes. Repeat with a third component conditioned/fixed. Include counterfactual compositions from donors when equality constraints allow them.

**Raw metrics**
Directed paired effect sizes, off-target effect sum, conditional effect delta, symmetry ratio X->Y vs Y->X, failure/error counts.

# Benchmark B5: Cross-context stability and early-fingerprint prediction

**Hypothesis being tested**
B is the most cross-context-stable factor; A is dynamic; variant family fingerprints are established early enough to predict late causal profiles.

**Competing explanations**
E1: stability ordering is benchmark-specific.
E2: variant families emerge only late.
E3: early causal profiles predict later profiles robustly.

**Predictions**
B signatures should show higher within-component stability than A/C/D under blind-safe nuisance transformations. Rank/order of per-variant causal profiles at 25M/50M should predict 100M/120M better than chance across multiple primitive metrics.

**Falsification**
B-stability claim weakens if stability ranking changes across prompt/context banks. Early-family claim fails if early-to-late predictive agreement is unstable across metrics/seeds.

**Exact implementation**
Use only transformations that can be generated without learning component semantics: token-order-preserving distractor injection, controlled context-length padding, blind-safe paraphrase bank if supplied by anonymized runner, and repeated donor seeds. For early prediction, fit no free-form correlation fishing: use preregistered metric vector [directed intervention effect, off-target effect, causal layer-of-max, causal retention, donor-identity sensitivity]. Compare 25M and 50M profiles to 100M and 120M using Spearman, Kendall, cosine-profile similarity, and leave-one-variant-out nearest-family consistency.

**Raw metrics**
Within-condition variance, cross-context cosine, intervention effect stability, per-variant metric vectors, rank correlations, leave-one-out classification consistency.

## Decision rules
- Strong support: same qualitative direction in >=80% of eligible complete checkpoints and >=5/7 variants, with bootstrap CI excluding zero for pooled paired effect.
- Moderate support: >=65% eligible checkpoints and >=4/7 variants, or strong but variant-family-specific effect.
- Falsified: preregistered directional prediction reverses in >=65% of eligible checkpoints or the matched null/generic-magnitude explanation performs equivalently within uncertainty.
- Inconclusive: insufficient blind-safe samples, inaccessible checkpoints, or confidence intervals too broad.

## No post-hoc rule
Post-hoc analyses are allowed only in a clearly marked exploratory section and cannot retroactively alter these predictions or falsification criteria.
