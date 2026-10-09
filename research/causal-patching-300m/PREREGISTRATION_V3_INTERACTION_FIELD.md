# Preregistration — 300M SELF×CORE interaction field v3

This document freezes the interpretation before the v3 workflow is launched.

## Question

Does the *causal effect of changing SELF* depend systematically on which CORE
is present?

Define, at a selected adapter layer:

`delta_SELF(context) = rep(SELF_radial2, context) - rep(SELF_normal, context)`.

If SELF is merely a context-independent control knob, the direction of
`delta_SELF` should be similar across different CORE values.

If SELF and CORE form an interacting reference system, the SELF-induced effect
field should contain CORE-specific structure that generalizes across changes
in GOAL and FOCUS.

## Models / layers

Primary:
- projected_diff @ layer 2
- projected_diff_slow @ layer 2

Controls:
- diff_anchor @ layer 3
- diff_only @ layers 2,3,4

All checkpoints are exact 300M states from source run `37678419157`.

## Intervention

Radial SELF:
- replacement is `-SELF`
- displacement from normal is exactly 2x anchor norm

Controls:
- 8 independent random directions
- every random replacement is also exactly 2x anchor norm away from normal

Only the selected adapter layer receives the alternate SELF. All other layers
use the learned normal SELF.

## Effect fields

For every CORE × GOAL × FOCUS prompt, record:

1. local post-adapter effect vector;
2. final downstream effect vector.

The effect vector is the normalized-representation difference between alternate
SELF and normal SELF.

## Primary SELF×CORE metrics

For the radial effect field and each of the 8 random-sham fields:

1. **cross-context CORE decoding**:
   train CORE centroids from SELF-effect vectors in one set of GOAL/FOCUS
   contexts and test on disjoint GOAL/FOCUS contexts;

2. **CORE effect-field cluster margin**:
   same-CORE delta-direction similarity minus different-CORE centroid
   similarity.

The radial learned SELF direction is compared with the full random-sham
distribution.

## Strong interaction criterion

Strong evidence for a learned-direction SELF×CORE interaction requires, at the
final downstream field:

- CORE cross-context accuracy above chance;
- radial value > random-sham mean;
- and preferably z > 2 or radial rank #1 among radial + 8 random shams.

CORE cluster margin should show the same qualitative direction.

A result that is high locally but disappears downstream is evidence for a local
interaction that is later compressed, not a stable downstream mechanism.

## Secondary metrics

The same field analysis is run for GOAL and FOCUS.

These determine whether the SELF-effect field is organized primarily by:

- CORE identity;
- GOAL identity;
- FOCUS identity;
- or a mixture.

Final component decoding under radial SELF is retained to distinguish:

- interaction with preserved CORE;
- interaction accompanied by CORE collapse.

## Predictions

### projected_diff / projected_diff_slow @ L2

Preferred dual-anchor pattern:

- radial SELF effect field has stronger CORE structure than random shams;
- CORE content itself remains decodable;
- downstream GOAL/FOCUS may change substantially.

This would directly support:
`SELF × CORE -> downstream organization`.

### diff_anchor @ L3

Expected to show a strong but pathological interaction:

- radial effect may be highly CORE-conditioned;
- final CORE decoding may collapse simultaneously.

That would distinguish *interaction* from *healthy separation*.

### diff_only

Expected to be less localized:
- no single layer may fully explain the all-layer CORE entanglement;
- interaction evidence may be distributed across L2–L4.

## Falsification / weakening outcomes

The SELF×CORE interpretation is weakened if:

- radial CORE-field metrics are indistinguishable from random shams;
- effect-field structure is explained equally well by arbitrary
  same-displacement anchor directions;
- projected models show no CORE-conditioned SELF effect despite strong v1
  downstream sensitivity.

## Guardrails

- No language-loss ranking.
- No semantic-selfhood claim.
- Context-conditioned nonlinear effects are explicitly controlled by 8 random
  same-displacement directions.
- This remains a single-seed checkpoint experiment pending replication.
