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


## 9. Held-out-template causal replication — PASSED

Workflow: `37925620017`

The strongest factorial result was replicated across four wording families with
leave-one-template-out decoding.

For each held-out wording family:

- decoder centroids were fitted only on the other three wording families;
- the held-out family was never used to construct the decoder;
- SELF radial-2x was compared with eight same-displacement random shams;
- the full CORE x GOAL x FOCUS factorial was evaluated on the held-out wording.

Templates included:

1. canonical: `הליבה היא ... היעד הנוכחי ... המוקד ...`
2. labeled-work phrasing;
3. declarative phrasing;
4. contextual phrasing.

### projected_diff — layer 2

Mean held-out clean accuracy:
- CORE: 0.980
- GOAL: 0.938
- FOCUS: 0.914

Radial SELF:
- CORE: **0.970**
- GOAL: 0.906
- FOCUS: **0.782**

Eight-random-sham mean:
- CORE: 0.983
- GOAL: 0.944
- FOCUS: 0.914

Aggregate factorial RMS:
- CORE -> GOAL: **4.83x** random mean, z=20.73, radial rank 1/9
- CORE -> FOCUS: **7.05x** random mean, z=21.53, radial rank 1/9

Crucially, radial SELF ranked **1/9 for both primary cells in every one of the
four held-out wording families**.

### projected_diff_slow — layer 2

Mean held-out clean accuracy:
- CORE: **0.994**
- GOAL: 0.935
- FOCUS: 0.898

Radial SELF:
- CORE: **0.964**
- GOAL: **0.684**
- FOCUS: **0.675**

Eight-random-sham mean:
- CORE: 0.994
- GOAL: 0.936
- FOCUS: 0.898

Aggregate factorial RMS:
- CORE -> GOAL: **4.42x**, z=12.15, rank 1/9
- CORE -> FOCUS: **3.41x**, z=10.13, rank 1/9

Again, radial SELF ranked **1/9 on both primary interactions for every held-out
template**.

This is especially informative because the downstream drop is much larger than
the CORE drop.

### diff_only — layer 3

- CORE remains essentially unchanged: 0.958 -> 0.961
- GOAL: 0.920 -> 0.894
- FOCUS: 0.925 -> 0.830
- CORE -> GOAL: **3.38x**
- CORE -> FOCUS: **6.32x**

The direction-specific contextual interaction is real here too, but the broader
longitudinal evidence still shows late SELF/CORE entanglement under stronger
global interventions.

### diff_anchor — layer 3 negative control

- CORE: **0.983 -> 0.332**
- GOAL: 0.910 -> 0.678
- FOCUS: 0.887 -> 0.334
- CORE -> GOAL: 1.71x
- CORE -> FOCUS: 1.86x

Thus diff_anchor again demonstrates that a special SELF direction is not enough:
the learned direction can be strongly causal while destroying the stable CORE
reference.

## 10. Updated causal conclusion

The held-out-template replication substantially weakens the explanation that
the projected-family result is a quirk of one synthetic sentence pattern.

The current strongest supported mechanism-level statement is:

> In the projected family, the learned SELF direction acts as a
> direction-specific contextual modulator. At layer 2, changing SELF alters
> how CORE-conditioned information propagates into downstream GOAL/FOCUS
> representations far more than equal-magnitude sham directions, while CORE
> identity is comparatively preserved.

The combination of evidence is important:

1. whole-residual patching confirms the information is causally usable but is
   too strong to localize the mechanism;
2. adapter-delta transplantation does **not** transfer component identity,
   arguing against SELF as a content store;
3. single-layer anchor intervention shows strong learned-direction
   specificity;
4. factorial interaction shows that the SELF effect depends on contextual
   component values;
5. the projected-family signature persists from 150M through 300M;
6. the same signature survives four held-out wording families.

This is the strongest evidence so far for the working dual-reference model:

`SELF <-> CORE`

with downstream GOAL/FOCUS organization modulated relative to those references.

It still does not establish semantic selfhood, uniqueness of this
interpretation, or multi-seed generalization.
