# 300M causal activation patching — results

Workflow run: **37916251165**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION.md`

All five planned arms completed successfully:

- baseline
- diff_anchor
- diff_only
- projected_diff
- projected_diff_slow

## Executive conclusion

The experiment produced two different classes of result.

### 1. Whole-residual component patching is causal but not architecture-specific

Replacing the complete residual state at layers 2–4 between equal-length
prompts differing in exactly one component switches CORE/GOAL/FOCUS almost
perfectly in **baseline as well as every SELF architecture**.

This verifies that the measured residual stream carries causal component
information, but it is too broad an intervention to identify a SELF-specific
mechanism.

Therefore this part of the assay is treated as a successful positive control,
not evidence for the SELF hypothesis.

### 2. Single-layer SELF intervention is strongly architecture-specific

The pre-registered matched-displacement SELF test is highly informative.

A radial 2x SELF intervention was applied at exactly one adapter layer while
all other layers used the normal learned SELF. It was compared with a random
anchor direction at exactly the same displacement.

The projected family shows the desired pattern most clearly:

> a direction-specific SELF intervention changes downstream GOAL/FOCUS while
> leaving CORE intact.

The negative control `diff_anchor` instead contains a sharply localized layer
where the same type of SELF intervention collapses CORE, GOAL and FOCUS
together.

This is the first direct causal localization result in the 300M line.

## Single-layer SELF causality

### Layer 2

| variant | radial/random effect | CORE radial | GOAL radial | FOCUS radial |
|---|---:|---:|---:|---:|
| diff_anchor | **14.82x** | 1.000 | 1.000 | 0.972 |
| diff_only | 4.18x | 1.000 | 0.944 | 0.889 |
| projected_diff | **4.77x** | **1.000** | 0.944 | 0.833 |
| projected_diff_slow | **7.63x** | **1.000** | **0.681** | **0.764** |

Matched-random controls preserve CORE at 1.000 in every variant.

For projected_diff:
- radial effect norm: 0.3016
- random effect norm: 0.0632

For projected_diff_slow:
- radial effect norm: 0.3779
- random effect norm: 0.0495

The important point is not merely that radial SELF has a larger effect.

In both projected variants, **CORE stays perfectly decoded while GOAL and
FOCUS are selectively disturbed**.

This supports a causal distinction between the stable CORE reference and
downstream organization influenced by SELF.

## diff_anchor reveals a localized entanglement bottleneck at layer 3

This is the strongest negative-control result.

### diff_anchor layer 3

- radial SELF effect norm: **0.8922**
- random matched effect norm: 0.2557
- effect ratio: **3.49x**

Under radial SELF at layer 3:

- CORE accuracy: **0.375**
- GOAL accuracy: **0.639**
- FOCUS accuracy: **0.361**

Under the random same-displacement intervention:

- CORE: **1.000**
- GOAL: **1.000**
- FOCUS: **1.000**

Thus the collapse is not explained by perturbation magnitude.

It is specifically associated with the learned SELF direction.

This causally localizes the previously observed diff_anchor SELF/CORE
entanglement primarily to the layer-3 path.

### Other diff_anchor layers

Layer 2:
- effect ratio 14.82x
- CORE 1.000
- GOAL 1.000
- FOCUS 0.972

Layer 4:
- effect ratio 3.98x
- CORE 1.000
- GOAL 0.931
- FOCUS 0.944

The severe CORE collapse is therefore not a generic consequence of touching
SELF anywhere. It is sharply layer-dependent.

## projected_diff shows a layer-2 downstream control point

### Layer 2
- radial/random effect ratio: **4.77x**
- CORE: **1.000**
- GOAL: 0.944
- FOCUS: **0.833**

### Layer 3
- ratio: 2.72x
- CORE: 1.000
- GOAL: 0.986
- FOCUS: 0.972

### Layer 4
- ratio: 1.81x
- CORE: 1.000
- GOAL: 1.000
- FOCUS: 0.931

The strongest downstream consequence occurs at layer 2.

This matches the older observation that early/middle layers contain richer
SELF-relative GOAL structure than the final representation.

## projected_diff_slow gives the strongest selective layer-2 effect

### Layer 2
- radial/random effect ratio: **7.63x**
- CORE: **1.000**
- GOAL: **0.681**
- FOCUS: **0.764**

Random matched:
- CORE: 1.000
- GOAL: 0.972
- FOCUS: 1.000

This is the cleanest causal result so far for the dual-anchor working model:

> changing SELF in a direction-specific way at layer 2 substantially changes
> downstream GOAL/FOCUS organization while the CORE identity remains stable.

### Layer 3
- ratio: 2.16x
- CORE: 1.000
- GOAL: 0.944
- FOCUS: 0.958

### Layer 4
- ratio: 2.18x
- CORE: 1.000
- GOAL: 0.958
- FOCUS: 0.972

Again, layer 2 is the strongest causal control point.

## diff_only suggests distributed, cumulative entanglement

Single-layer radial SELF perturbations:

### Layer 2
- ratio: 4.18x
- CORE: 1.000
- GOAL: 0.944
- FOCUS: 0.889

### Layer 3
- ratio: 3.21x
- CORE: 1.000
- GOAL: 0.931
- FOCUS: 0.875

### Layer 4
- ratio: 4.05x
- CORE: 1.000
- GOAL: 0.944
- FOCUS: 0.847

This is notable because the previous **all-layer** radial-2x intervention at
300M reduced final CORE accuracy to 0.333.

No single tested layer reproduces that CORE collapse.

The most plausible current interpretation is:

> diff_only SELF/CORE entanglement is distributed across layers and becomes
> destructive through accumulation/interactions, rather than being localized
> to one bottleneck.

This should be tested directly with multi-layer combinatorial interventions.

## Whole-residual component patching — why it is not the main result

Across baseline and SELF variants, targeted full-residual patches:

- switch the patched component toward the source at ~100%;
- outperform random matched-norm perturbations strongly;
- preserve the unchanged components almost perfectly.

For example, baseline itself gets source-switch advantage ~1.0 for CORE, GOAL
and FOCUS at layers 2–4.

This is useful confirmation that the assay can causally manipulate the
representations, but it cannot distinguish the explicit SELF mechanism from
ordinary transformer state.

We therefore **do not** use this result as evidence that SELF created
CORE/GOAL/FOCUS.

## Updated causal working model

The evidence now supports a more specific architecture-dependent picture.

### projected family

`SELF` has a causal downstream control point strongest around **layer 2**.

Perturbing it:
- leaves CORE stable;
- changes GOAL/FOCUS substantially;
- has much larger effect than a same-magnitude random direction.

This is consistent with:

`SELF <-> CORE`

followed by downstream relational organization rather than SELF replacing CORE.

### diff_anchor

The learned SELF direction enters a severe entanglement bottleneck around
**layer 3**, where perturbing SELF collapses CORE and downstream state together.

### diff_only

SELF effects are distributed. Single-layer interventions preserve CORE, while
the all-layer intervention destroys it, suggesting cumulative interaction.

## What is now supported more strongly

We can upgrade the previous statement:

> "projected models have geometry consistent with separated SELF and CORE"

to:

> **In the projected 300M models, direction-specific intervention on SELF at
> layer 2 causally changes downstream GOAL/FOCUS decoding while preserving
> CORE decoding, whereas an equal-magnitude random SELF perturbation does not
> produce the same effect.**

And for the negative control:

> **diff_anchor contains a layer-3 causal bottleneck where the learned SELF
> direction is entangled with CORE; matched random perturbation does not cause
> the collapse.**

These are causal sensitivity claims, not semantic selfhood claims.

## Important remaining limitation

The SELF intervention still changes the adapter computation broadly at one
layer.

The next assay should patch/ablate only the **adapter contribution**
`adapter(x, SELF) - x`, leaving the base transformer residual untouched.

That experiment can determine whether the SELF-specific adapter path itself
carries the selective causal effect, rather than the result being a broad
consequence of changing the adapter's reference input.

Priority targets:

1. projected_diff_slow layer 2;
2. projected_diff layer 2;
3. diff_anchor layer 3;
4. diff_only layers 2–4 for cumulative-interaction analysis.
