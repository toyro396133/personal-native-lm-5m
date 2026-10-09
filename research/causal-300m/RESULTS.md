# 300M causal follow-up — activation patching and SELF-adapter interventions

Runs:

- whole-stream activation patching: `37916251165`
- surgical SELF-adapter causal patching v2: `37916891722`

## Methodological correction

The first activation-patching assay replaced the **entire residual stream** of a
target prompt with the residual stream of a source prompt after layers 2/3/4.

Because source and target prompts differed in exactly one labeled component,
this produced near-trivial 100% source-label switching: once the entire hidden
state is replaced by the source state, all downstream computation is effectively
continued from the source state.

Therefore:

> the 100% CORE/GOAL/FOCUS switches from v1 are **not** treated as strong
> component-localization evidence.

The v1 result is retained as a sanity check showing that the downstream model
can carry source states through to final representation, but it is explicitly
downgraded as a causal localization assay.

## Valid causal result A: single-layer SELF intervention

A separate part of v1 changed the SELF anchor **only in one adapter layer**,
leaving every other layer and the input unchanged. Radial SELF perturbations
were compared with random same-displacement anchor perturbations.

This is a meaningful causal intervention on the explicit SELF pathway.

### projected_diff_slow

At layer 2, radial-2x SELF perturbation:

- final representation effect: **0.3779**
- matched random effect: **0.0495**
- effect ratio: **7.63x**
- CORE accuracy: **1.000**
- GOAL accuracy: **0.681** vs random **0.972**
- FOCUS accuracy: **0.764** vs random **1.000**

This is the cleanest causal dissociation observed so far:

> changing SELF specifically at layer 2 strongly disrupts downstream
> GOAL/FOCUS organization while preserving CORE identity.

At layers 3 and 4 the direction remains special, but the component-level
damage is much smaller.

### projected_diff

At layer 2:

- radial effect: **0.3016**
- random effect: **0.0632**
- ratio: **4.78x**
- CORE: **1.000**
- GOAL: 0.944 vs random 1.000
- FOCUS: 0.833 vs random 0.986

Again, SELF has a direction-specific downstream causal effect while CORE is
preserved.

### diff_anchor

Layer 3 is qualitatively different:

- radial effect: **0.8922**
- random effect: **0.2557**
- ratio: **3.49x**
- CORE: **0.375**
- GOAL: 0.639
- FOCUS: 0.361

This is causal evidence of the entanglement already inferred structurally.

The same kind of SELF intervention that selectively modulates downstream
structure in the projected family instead disrupts CORE itself in diff_anchor.

### diff_only

SELF direction is also causally special (roughly 3.2–4.2x random across
layers 2–4), while CORE remains 1.0 in this single-layer test. Its strongest
damage is mainly FOCUS rather than CORE.

This shows that the severe final all-layer radial-2x CORE collapse seen in the
structural audit can emerge from multi-layer accumulation rather than from any
single layer alone.

## Valid causal result B: adapter-path ablation

v2 removes only the learned SELF-adapter contribution at one layer and compares
that removal with random perturbation matched to the removed adapter-delta norm.

The learned adapter contribution is generally more consequential than matched
random noise:

### diff_anchor
- L2: 1.76x random
- L3: 2.11x
- L4: **2.51x**

### diff_only
- L2: **2.62x**
- L3: 2.08x
- L4: 1.07x

### projected_diff
- L2: 1.47x
- L3: 1.52x
- L4: **1.85x**

### projected_diff_slow
- L2: 1.40x
- L3: **1.91x**
- L4: 1.78x

Thus the adapter pathway is not equivalent to an arbitrary perturbation of the
same size.

However, coarse CORE/GOAL/FOCUS classification often remains intact after a
single-layer ablation. The causal role is therefore more subtle than simply
"this adapter stores the label."

## Important negative result: the SELF adapter does not directly carry component identity

v2 also swaps only the **adapter contribution** between source and target
prompts while keeping the base residual stream target-derived.

Across all four architectures and CORE/GOAL/FOCUS:

- source-label switch rate remains essentially **0**;
- component-margin shifts are tiny (generally around 1e-4 to 1e-3);
- unchanged components remain highly stable.

Therefore:

> the explicit SELF adapter is not acting as a content store for CORE, GOAL or
> FOCUS identity.

This is scientifically useful. It supports a **modulatory/reference** role for
SELF rather than a direct content-encoding role.

## Current causal interpretation

The combined evidence now supports a more specific model:

1. CORE/content representations exist in the base transformer pathway.
2. SELF adapters do not simply encode or copy CORE/GOAL/FOCUS labels.
3. The learned SELF direction modulates downstream computation in a
   direction-specific way.
4. In projected_diff/projected_diff_slow, early SELF intervention can strongly
   affect GOAL/FOCUS while preserving CORE.
5. In diff_anchor, SELF intervention can destroy CORE, demonstrating causal
   SELF/CORE entanglement.
6. Therefore the projected family is distinguished not merely by geometry but
   by **causal separation of reference and content**.

This is still not evidence of semantic selfhood. It is evidence about the
causal role of the explicit learned SELF mechanism.

## Next experiment

Run the same **single-layer matched SELF intervention** across:

- milestones: 150M, 170M, 200M, 300M
- variants: diff_anchor, diff_only, projected_diff, projected_diff_slow
- every transformer layer 1–6

Measure continuously:

- final representation displacement;
- CORE / GOAL / FOCUS classification;
- correct-vs-nearest-wrong centroid margin;
- radial vs random/orthogonal same-displacement selectivity;
- downstream GOAL/FOCUS damage relative to CORE damage.

This tests when causal SELF/CORE separation emerges, disappears or reorganizes
during training.
