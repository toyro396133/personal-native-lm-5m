# Preregistration — factorial SELF×context functional interaction v4

This document freezes the interpretation before the v4 workflow is launched.

## Motivation

v1 established that changing the learned SELF direction at one layer can
causally alter downstream GOAL/FOCUS while preserving CORE in projected
architectures.

v2 showed that the explicit adapter contribution is not a swappable semantic
payload.

v3 did **not** confirm a privileged CORE-structured SELF effect field relative
to eight random same-displacement anchor directions. It did show unusually
strong GOAL/FOCUS organization in the learned SELF effect.

v4 therefore tests interaction directly at the **functional response level**.

## Factorial design

For every full-factorial prompt:

CORE × GOAL × FOCUS

measure the change in correct-label decoder margin caused by a single-layer
SELF intervention.

For a given outcome, e.g. GOAL:

`Delta_GOAL(core, goal, focus) =
margin_GOAL(SELF_alt) - margin_GOAL(SELF_normal)`.

Then, holding GOAL and FOCUS fixed, compare Delta_GOAL across different CORE
values.

The pairwise difference:

`Delta_GOAL(core_B) - Delta_GOAL(core_A)`

is the direct CORE-conditioned SELF interaction term for that context.

No internal CORE patch is required: CORE is independently manipulated in the
input factorial grid.

## Models / layers

Primary:
- projected_diff @ layer 2
- projected_diff_slow @ layer 2

Controls:
- diff_anchor @ layer 3
- diff_only @ layers 2,3,4

Exact 300M checkpoints from source run `37678419157`.

## SELF intervention

Learned radial direction:
- SELF -> -SELF at the selected adapter layer only
- displacement = exactly 2x learned anchor norm

Controls:
- 8 independent random anchor directions
- each replacement has exactly the same 2x displacement

All other layers retain normal learned SELF.

## Outcomes

For each component decoder:
- CORE correct-label cosine margin
- GOAL correct-label cosine margin
- FOCUS correct-label cosine margin

The clean centroids are built from normal final representations.

## Full interaction matrix

For each SELF direction, calculate interaction magnitude for:

- CORE factor -> CORE / GOAL / FOCUS outcome
- GOAL factor -> CORE / GOAL / FOCUS outcome
- FOCUS factor -> CORE / GOAL / FOCUS outcome

Interaction summaries:

- mean absolute pairwise interaction
- RMS interaction
- mean within-context standard deviation
- mean within-context range

## Preregistered primary cells

The direct dual-anchor hypothesis is tested by:

1. **CORE factor -> GOAL outcome**
2. **CORE factor -> FOCUS outcome**

Primary statistic: RMS interaction.

## Strong evidence criterion

A primary cell is considered strong learned-direction interaction evidence if:

- radial RMS interaction > random-sham mean;
- z > 2 versus the eight random shams;
- radial rank is #1 among radial + 8 random directions;
- the qualitative result is supported by mean-absolute interaction.

Evidence is strongest if this occurs in projected variants while CORE decoding
itself remains high.

## Predicted patterns

### projected_diff_slow L2

Preferred result:
- strong CORE->GOAL and/or CORE->FOCUS radial interaction;
- stronger than random shams;
- CORE decoding remains ~1.0.

### projected_diff L2

Same pattern, expected weaker than projected_diff_slow.

### diff_anchor L3

Possible pathological interaction:
- CORE-conditioned SELF response may be large;
- but CORE itself may collapse under the radial intervention.

This would show interaction without healthy separation.

### diff_only

Expected distributed effects:
- primary interaction may be weak at any one layer;
- or appear across multiple layers without a single dominant bottleneck.

## Secondary interpretation

If primary CORE-factor cells fail but GOAL-factor or FOCUS-factor cells strongly
beat random controls, that supports a downstream-context control interpretation
rather than a direct SELF×CORE interaction.

## Falsification / weakening

The direct SELF×CORE functional-interaction hypothesis is weakened if:

- CORE->GOAL and CORE->FOCUS interaction magnitudes are not exceptional relative
  to random SELF directions;
- projected variants show strong SELF effects but those effects do not depend on
  CORE more than random controls.

## Guardrails

- No language-loss ranking.
- No semantic-selfhood claim.
- Large radial perturbation effects alone are not interaction evidence.
- Interaction must beat matched random SELF directions.
- Single-seed 300M checkpoints require later replication.
