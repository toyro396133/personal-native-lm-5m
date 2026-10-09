# 300M causal mechanism follow-up — consolidated results

This document consolidates the causal follow-ups after the 300M structural audit.

Runs:
- full residual activation patching: `37916251165`
- SELF adapter contribution causal patching v2: `37916891722`
- factorial SELF-by-context interaction v4 at 300M: `37918122058`
- factorial trajectory 150/170/200M: `37918632896`
- SELF single-layer causal trajectory 150/170/200/300M: `37922411679`

## Executive conclusion

The causal evidence now supports a substantially sharper mechanistic picture:

> The explicit SELF mechanism behaves more like a direction-specific
> **reference/modulation axis** than a content store. In the projected family,
> perturbing SELF at a single layer changes how contextual CORE information
> influences downstream GOAL/FOCUS while often preserving discrete CORE
> identity. In entangled controls, especially diff_anchor, the same type of
> intervention directly destabilizes CORE.

This is causal sensitivity evidence for the learned pathway. It is not evidence
of semantic selfhood.

## 1. Full residual-stream patching was intentionally downgraded

The first causal activation-patching assay replaced the entire residual state
with the source residual at layers 2–4.

Across baseline and SELF variants, targeted patches switched CORE/GOAL/FOCUS
source labels at or near 100%, while matched random perturbations did not.

This demonstrates that the relevant information is present and causally usable
in the residual stream, but the intervention is too strong to identify a
SELF-specific mechanism: once the complete source residual is substituted,
subsequent computation is largely source-derived.

Therefore the full-residual source-switch result is retained as a positive
assay sanity check, not as the primary mechanistic claim.

## 2. Surgical SELF-adapter contribution patching v2: modulation, not content storage

The v2 assay preserves the target base-transformer residual and changes only
the explicit SELF-adapter contribution.

At 300M, moving the source adapter delta into a target example produces:

- essentially **0% source-label switching** for CORE, GOAL and FOCUS;
- very small source-vs-target margin changes;
- nevertheless, ablating the adapter contribution causes a larger final
  representation effect than a random perturbation matched to the removed
  adapter-delta norm.

Ablation/random final-effect ratios:

| variant | L2 | L3 | L4 |
|---|---:|---:|---:|
| diff_anchor | 1.76x | 2.11x | 2.51x |
| diff_only | 2.62x | 2.08x | 1.07x |
| projected_diff | 1.47x | 1.52x | 1.85x |
| projected_diff_slow | 1.40x | 1.91x | 1.78x |

Discrete component accuracies remain essentially unchanged under these small
adapter-delta ablations.

Interpretation:

> The adapter contribution is functionally special, but it does not itself
> encode a transferable "this is CORE X / GOAL Y / FOCUS Z" payload.

That favors a reference/modulation interpretation.

## 3. Single-layer SELF interventions are strongly direction-specific

The strongest clean causal test changes the SELF anchor supplied to exactly
one adapter layer, while all other layers receive the learned anchor.

Random and orthogonal controls are matched to the same displacement magnitude.

### projected_diff_slow, layer 2, radial-2x / matched-control effect ratio

- 150M: **7.10x**
- 170M: **7.32x**
- 200M: **10.36x**
- 300M: **8.71x**

### projected_diff, layer 2

- 150M: **4.76x**
- 170M: **4.33x**
- 200M: **4.74x**
- 300M: **7.51x**

This is a persistent trajectory, not a one-checkpoint anomaly.

diff_anchor can show even larger direction-specific ratios at some layers
(e.g. ~12x at layer 2), which is why effect strength alone is not success:
the downstream factorization test is required.

## 4. Factorial SELF x context interaction is the strongest current evidence

The factorial assay asks whether the effect of SELF depends on the contextual
component, rather than merely producing a global representation displacement.

Eight random same-magnitude sham directions provide a reference distribution.

### 300M projected_diff_slow — layer 2

Normal component accuracy:
- CORE: 1.000
- GOAL: 0.991
- FOCUS: 0.995

Radial SELF intervention:
- CORE: **1.000**
- GOAL: **0.741**
- FOCUS: **0.843**

Eight-random-sham mean:
- CORE: 1.000
- GOAL: 0.991
- FOCUS: 0.990

Factorial RMS interaction:
- CORE -> GOAL: **4.52x** random mean, z=8.01, radial rank 1/9
- CORE -> FOCUS: **3.59x** random mean, z=5.74, radial rank 1/9

Thus the learned SELF direction changes the downstream contextual relation much
more than equal-magnitude arbitrary directions while discrete CORE identity
remains intact.

### 300M projected_diff — layer 2

Normal:
- CORE 1.000
- GOAL 1.000
- FOCUS 0.981

Radial:
- CORE **1.000**
- GOAL 0.949
- FOCUS 0.884

Random mean:
- CORE 1.000
- GOAL 0.999
- FOCUS 0.976

Factorial RMS:
- CORE -> GOAL: **4.19x**, z=4.57, rank 1/9
- CORE -> FOCUS: **4.85x**, z=4.78, rank 1/9

Again: large direction-specific downstream interaction without discrete CORE
collapse.

### diff_anchor negative control — layer 3

Radial:
- CORE **0.333**
- GOAL 0.653
- FOCUS 0.333

Random mean:
- CORE 1.000
- GOAL 0.995
- FOCUS 0.976

Factorial RMS:
- CORE -> GOAL: 1.61x random
- CORE -> FOCUS: 1.90x random

The learned SELF direction is special, but it is strongly entangled with CORE.
This distinguishes useful relational modulation from destructive coupling.

## 5. The factorial interaction persists over training

### projected_diff_slow, layer 2

CORE -> GOAL radial/random RMS ratio:
- 150M: **5.81x**
- 170M: **3.83x**
- 200M: **4.64x**
- 300M: **4.52x**

CORE -> FOCUS:
- 150M: **4.36x**
- 170M: **5.64x**
- 200M: **3.98x**
- 300M: **3.59x**

This is one of the strongest longitudinal findings in the project.

### projected_diff, layer 2

CORE -> GOAL:
- 150M: 2.87x
- 170M: 3.03x
- 200M: 2.38x
- 300M: **4.19x**

CORE -> FOCUS:
- 150M: 3.07x
- 170M: 4.56x
- 200M: 3.41x
- 300M: **4.85x**

The projected-family interaction therefore survives substantial continued
training and does not depend on one milestone.

## 6. Refined mechanism hypothesis

The evidence now favors:

`INPUT -> CORE`

plus a distinct learned SELF reference that modulates contextual computation:

`SELF <-> CORE -> dynamic GOAL / FOCUS organization`

The explicit adapter delta is not itself a content vector for CORE/GOAL/FOCUS.
Instead, the learned SELF direction changes how the network processes and
relates contextual content.

This explains why:

- SELF direction interventions can have large effects;
- source adapter-delta transplantation does not transfer component identity;
- CORE can remain discretely stable in projected variants;
- downstream GOAL/FOCUS interaction nevertheless changes selectively;
- diff_anchor can have stronger SELF effects yet worse factorization.

## 7. What is now supported

Supported by the single-seed 300M line:

1. the learned SELF direction is causally special relative to
   magnitude-matched random/orthogonal controls;
2. this direction-specific effect is localized differently by layer and
   architecture;
3. projected variants exhibit a repeatable SELF-by-context interaction;
4. in the projected family, large downstream interaction can coexist with
   preserved discrete CORE identity;
5. the explicit SELF adapter behaves more like a modulator/reference pathway
   than a transferable content store;
6. the key projected-family signatures persist across 150–300M.

## 8. Still not proven

- semantic selfhood;
- that the human labels SELF/CORE/GOAL/FOCUS uniquely identify the learned
  computation;
- multi-seed generalization of the 300M training line;
- robustness to entirely different prompt phrasings / assay templates;
- a unique mathematical operation equivalent to literal SELF x CORE -> GOAL.

## Next falsification gate

Before stronger claims, replicate the factorial causal signature on held-out
prompt templates that were not used by the current assay.

Success criterion:
- projected_diff/projected_diff_slow retain direction-specific SELF x context
  interaction above matched sham directions;
- CORE remains substantially more stable than the destructive diff_anchor
  control;
- results persist across more than one wording/template family.

Failure would downgrade the current result to assay-template-specific
organization.
