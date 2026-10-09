# Internal SELF × CORE subspace intervention: 200M and 300M

Workflow run: `37923430126`

Status: **6/6 assays + comparison succeeded.**

Variants / active layers:
- projected_diff: layer 2
- projected_diff_slow: layer 2
- diff_anchor: layer 3

Milestones:
- 200M
- 300M

## Why this experiment matters

Previous factorial evidence directly intervened on SELF while independently
varying CORE in the prompt.

This experiment additionally manipulates an **internal learned CORE subspace**:

1. a rank-5 CORE subspace is learned from held-out training context
   combinations;
2. evaluation uses disjoint goal/focus context combinations;
3. only the coordinates inside the learned CORE subspace are changed;
4. the orthogonal target residual is left target-derived;
5. the CORE intervention is crossed with normal SELF, radial SELF, and eight
   random same-displacement SELF directions;
6. a random non-CORE, norm-matched residual direction is used as a CORE-patch
   control.

The primary criterion is therefore genuinely two-sided:

> does radial SELF × targeted CORE-subspace interaction on GOAL/FOCUS exceed
> both random-SELF controls and random-CORE-subspace controls?

## Main result

Yes — most clearly for **GOAL in the projected family**, especially
projected_diff_slow.

This is the strongest causal evidence in the project so far for a real
SELF × CORE interaction affecting GOAL.

## projected_diff_slow at 200M

Layer 2.

### CORE manipulation check

Under normal SELF:

- targeted CORE-subspace source-CORE switch: **12.96%**
- random non-CORE control: **0%**

Under radial SELF:

- targeted CORE-subspace switch: **33.33%**
- random non-CORE control: **0%**

The subspace intervention is therefore directionally meaningful, although it
does not fully replace CORE identity on most examples.

### GOAL interaction

RMS interaction for radial SELF × targeted CORE:

**0.004010**

Compared with eight random SELF directions:

- random-SELF mean RMS: ~0.000610
- ratio: **6.57x**
- z vs random SELF: **17.96**
- rank: **1/9**

Compared with radial SELF × random non-CORE patch:

- ratio: **1.82x**

This passes both primary controls.

### FOCUS interaction

- radial SELF × targeted CORE RMS: ~0.00130
- ratio vs random SELF: **4.85x**
- z: **9.39**
- rank: **1/9**
- ratio vs random CORE: **1.04x**

Thus FOCUS is clearly SELF-direction-specific but, at 200M, is **not strongly
CORE-subspace-specific** beyond the matched non-CORE perturbation.

This is an important distinction.

## projected_diff_slow at 300M

### CORE manipulation check

Normal SELF:
- targeted source switch: 2.78%
- random control: 0%

Radial SELF:
- targeted source switch: **18.52%**
- random control: 0%

### GOAL interaction

- target RMS: **0.003660**
- ratio vs eight random SELF directions: **5.83x**
- z: **9.19**
- rank: **1/9**
- ratio vs random CORE control: **2.12x**

The two-sided interaction therefore persists at 300M.

### FOCUS interaction

- ratio vs random SELF: **4.93x**
- z: **8.86**
- rank: 1/9
- ratio vs random CORE: **1.59x**

FOCUS becomes more CORE-subspace-specific than at 200M, but GOAL remains the
cleaner primary interaction.

## projected_diff at 200M

Layer 2.

CORE source-switch manipulation:

- targeted: 4.63%
- random CORE: 0%
- under radial SELF targeted rises to 7.41%

GOAL interaction:

- ratio vs random SELF: **3.70x**
- z: 10.91
- rank: 1/9
- ratio vs random CORE: **1.70x**

FOCUS interaction:

- ratio vs random SELF: **6.77x**
- z: 8.06
- rank: 1/9
- ratio vs random CORE: **1.76x**

So projected_diff also passes both controls, with a comparatively stronger
FOCUS signature.

## projected_diff at 300M

CORE source-switch:

- normal targeted: 1.85%
- random CORE: 0%
- radial targeted: **12.04%**
- radial random CORE: 0%

GOAL:

- ratio vs random SELF: **4.36x**
- z: 8.20
- rank: 1/9
- ratio vs random CORE: **2.11x**

FOCUS:

- ratio vs random SELF: **10.40x**
- z: 13.21
- rank: 1/9
- ratio vs random CORE: **2.66x**

This is a very strong 300M downstream interaction, especially for FOCUS.

## diff_anchor as entangled control

At 200M, layer 3:

### CORE
- SELF×CORE interaction ratio vs random SELF: 3.35x
- ratio vs random CORE: **7.53x**

### GOAL
- ratio vs random SELF: 2.62x
- ratio vs random CORE: 2.29x

### FOCUS
- ratio vs random SELF: 2.03x
- ratio vs random CORE: 2.49x

At 300M:

### CORE
- ratio vs random SELF: 4.12x
- ratio vs random CORE: 5.17x

### GOAL
- ratio vs random SELF: 2.27x
- ratio vs random CORE: 1.47x

### FOCUS
- ratio vs random SELF: 1.61x
- ratio vs random CORE: 1.59x

The dominant internal interaction remains CORE itself.

Together with direct layer-3 SELF perturbation collapsing CORE accuracy, this
continues to characterize diff_anchor as an entangled control rather than a
clean dual-reference mechanism.

## Strongest mechanistic reading

The combined evidence now supports a more precise working model.

### projected_diff_slow

At layer 2:

- SELF is directionally special;
- CORE has a learnable internal subspace;
- changing CORE-subspace coordinates interacts with the specific learned SELF
  direction;
- the interaction has a robust downstream effect on GOAL;
- the GOAL effect exceeds both random SELF directions and random non-CORE
  subspace perturbations;
- CORE identity can remain largely preserved despite large SELF effects.

This is consistent with:

`SELF × CORE -> GOAL`

as a **mechanistic working hypothesis**.

### projected_diff

Also shows a genuine two-sided interaction, with especially strong FOCUS
interaction by 300M.

### diff_anchor

Shows a strong interaction too, but the dominant target is CORE itself and
direct SELF perturbation destroys CORE. This is better described as
SELF/CORE entanglement.

## Important caveat: CORE manipulation is partial

The learned CORE-subspace intervention only switches final CORE identity on a
minority of examples.

Therefore we should not claim that the assay performs a complete causal CORE
replacement.

What is supported is narrower:

> a targeted perturbation restricted to a learned CORE subspace interacts with
> the learned SELF direction on downstream GOAL/FOCUS substantially more than
> matched random controls.

This is a strong two-sided internal causal result, but not yet a complete
causal rewrite of CORE state.

## Current strongest claim

The strongest defensible claim from the 5M research line is now:

> The projected SELF architectures learn a direction-specific SELF reference
> whose causal downstream effect depends on internal CORE-related coordinates.
> In projected_diff_slow, this interaction is most cleanly expressed on GOAL at
> layer 2 and persists from 200M to 300M. The effect survives controls for SELF
> displacement magnitude and for non-CORE residual perturbations.

This goes beyond correlation/geometry and supports a genuine interaction
mechanism.

It still does **not** establish semantic selfhood.

## Next validation priorities

The mechanism should now be challenged rather than expanded:

1. multi-seed replication of projected_diff_slow / projected_diff /
   diff_anchor;
2. stronger CORE-subspace intervention that increases CORE manipulation success
   without changing orthogonal residual magnitude;
3. transfer the same two-sided causal assay to the pretrained Goldfish line
   once its staged continuation reaches sufficiently mature checkpoints.
