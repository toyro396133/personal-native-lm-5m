# 300M factorial SELF×context functional interaction v4 — results

Workflow run: **37918122058**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION_V4_FACTORIAL_INTERACTION.md`

All four planned variants and the final comparison completed successfully.

## Executive conclusion

The preregistered **functional SELF×CORE interaction test is positive**.

For the projected family at layer 2, the effect of changing the learned SELF
direction on downstream GOAL and FOCUS margins depends strongly on which CORE
is present.

This dependence is:

- non-additive;
- much larger than eight random same-displacement SELF directions;
- ranked #1 among radial + 8 random controls;
- present while CORE decoding itself remains perfect.

This is the strongest evidence so far for the dual-anchor mechanism.

The key refinement relative to v3 is:

> The SELF×CORE interaction is expressed as a **functional modulation of
> downstream response**, not necessarily as a SELF-effect vector whose
> direction clusters by CORE.

In other words, CORE changes the *gain/response* of the SELF control field
without requiring the SELF-induced delta vector itself to become an explicit
CORE code.

## Preregistered primary cells

The primary hypothesis was tested with two cells:

1. CORE factor -> GOAL margin response
2. CORE factor -> FOCUS margin response

For each fixed GOAL/FOCUS context, the CORE value was changed independently.
For each CORE, the effect of radial SELF on the correct-label downstream margin
was measured.

The non-additive interaction term is the difference in SELF effect between
CORE values.

Primary statistic: RMS interaction.

Strong evidence required:

- radial > random-sham mean;
- z > 2;
- radial rank #1/9;
- matching qualitative support from mean-absolute interaction.

Both projected variants pass all criteria for both primary cells.

## projected_diff @ layer 2

Normal final decoding:

- CORE: 1.000
- GOAL: 1.000
- FOCUS: 0.981

Radial SELF final decoding:

- CORE: **1.000**
- GOAL: 0.949
- FOCUS: 0.884

Random-control mean:

- CORE: 1.000
- GOAL: 0.999
- FOCUS: 0.976

### CORE -> GOAL interaction

RMS:

- radial: **0.005609**
- random mean: 0.001340
- ratio: **4.19x**
- z: **+4.57**
- rank: **#1/9**

Mean absolute:

- radial: **0.004167**
- random mean: 0.001018
- z: **+4.61**
- rank: **#1/9**

### CORE -> FOCUS interaction

RMS:

- radial: **0.002898**
- random mean: 0.000598
- ratio: **4.85x**
- z: **+4.78**
- rank: **#1/9**

Mean absolute:

- radial: **0.002153**
- random mean: 0.000449
- z: **+4.93**
- rank: **#1/9**

This is direct evidence that the downstream effect of SELF is conditioned by
CORE identity while CORE remains robust.

## projected_diff_slow @ layer 2

Normal:

- CORE: 1.000
- GOAL: 0.991
- FOCUS: 0.995

Radial SELF:

- CORE: **1.000**
- GOAL: 0.741
- FOCUS: 0.843

Random-control mean:

- CORE: 1.000
- GOAL: 0.991
- FOCUS: 0.990

### CORE -> GOAL interaction

RMS:

- radial: **0.006575**
- random mean: 0.001456
- ratio: **4.52x**
- z: **+8.01**
- rank: **#1/9**

Mean absolute:

- radial: **0.005301**
- random mean: 0.001112
- z: **+8.43**
- rank: **#1/9**

### CORE -> FOCUS interaction

RMS:

- radial: **0.002577**
- random mean: 0.000717
- ratio: **3.59x**
- z: **+5.74**
- rank: **#1/9**

Mean absolute:

- radial: **0.001915**
- random mean: 0.000500
- z: **+6.45**
- rank: **#1/9**

This is the strongest healthy dual-anchor result in the entire 300M series.

The learned SELF direction interacts with CORE to modulate downstream
GOAL/FOCUS much more than arbitrary same-magnitude SELF changes, while CORE
identity remains fully recoverable.

## Why v3 and v4 are not contradictory

v3 asked:

> Does the **direction of the SELF-induced representation delta** itself form
> a privileged CORE-coded field?

Answer: not robustly.

v4 asks:

> Does changing CORE alter the **functional consequence** that SELF has on
> downstream GOAL/FOCUS?

Answer: strongly yes.

These are different mechanisms.

A nonlinear reference/control system can have:

- no simple CORE clustering in its perturbation-vector direction;
- but strong CORE-dependent gain, sensitivity or transformation of downstream
  state.

That is exactly the pattern observed.

Therefore the evidence now favors a **control-field / reference-frame
interaction** rather than a literal "CORE vector embedded inside the SELF
effect vector".

## diff_anchor @ layer 3 — interaction without healthy separation

diff_anchor also passes the primary interaction test.

Radial decoding:

- CORE: **0.333**
- GOAL: 0.653
- FOCUS: **0.333**

Random means:

- CORE: 1.000
- GOAL: 0.995
- FOCUS: 0.976

### CORE -> GOAL

RMS:
- radial: 0.007503
- random: 0.004663
- ratio: 1.61x
- z: **+4.33**
- rank: #1/9

### CORE -> FOCUS

RMS:
- radial: 0.004396
- random: 0.002320
- ratio: 1.90x
- z: **+6.80**
- rank: #1/9

Thus diff_anchor has a real SELF×CORE functional interaction too.

But it is pathological:

> the interaction is accompanied by collapse of the stable CORE state.

This cleanly separates two properties:

1. existence of SELF×CORE interaction;
2. whether that interaction preserves factorization.

The projected family has both interaction and separation.
diff_anchor has interaction without separation.

## diff_only — distributed SELF×CORE interaction

diff_only passes the primary criterion at every tested layer.

### Layer 2

CORE -> GOAL:
- ratio: 2.25x
- z: +5.44
- rank #1/9

CORE -> FOCUS:
- ratio: 2.51x
- z: +6.12
- rank #1/9

### Layer 3

CORE -> GOAL:
- ratio: 2.93x
- z: +3.83
- rank #1/9

CORE -> FOCUS:
- ratio: 4.31x
- z: +3.81
- rank #1/9

### Layer 4

CORE -> GOAL:
- ratio: 2.54x
- z: +6.04
- rank #1/9

CORE -> FOCUS:
- ratio: 2.73x
- z: +6.68
- rank #1/9

CORE decoding remains 1.000 for every single-layer radial intervention.

This explains the previous puzzle:

- no single layer destroys CORE;
- but all-layer radial SELF at 300M collapses CORE to 0.333.

The interaction is distributed across depth and can accumulate into
entanglement when all layers are perturbed together.

## Secondary factorial structure

The interaction matrix contains strong diagonal and cross-component effects.

For projected_diff L2, examples include:

- CORE -> CORE RMS z: +9.69
- GOAL -> GOAL z: +5.47
- GOAL -> CORE z: +6.80
- FOCUS -> CORE z: +5.62

For projected_diff_slow L2:

- GOAL -> GOAL z: +12.34
- CORE -> GOAL z: +8.01
- FOCUS -> GOAL z: +7.54

This indicates that SELF is not a one-edge mechanism.
It modulates a coupled relational state whose response depends on multiple
current components.

The preregistered primary result remains CORE-conditioned GOAL/FOCUS response.

## Combined v1–v4 mechanistic picture

### v1 — direction-specific causality

Changing SELF at one layer causally changes downstream computation.

Projected L2:
- CORE survives;
- GOAL/FOCUS change.

diff_anchor L3:
- CORE/GOAL/FOCUS collapse.

### v2 — SELF is not a content payload

The adapter contribution itself cannot simply be transplanted to move semantic
component identity.

Removing the adapter contribution is much less destructive than changing the
SELF reference.

Therefore SELF is better interpreted as a reference/control input to a
transformation.

### v3 — no simple CORE-coded SELF effect vector

The SELF-induced delta direction is not uniquely CORE-clustered relative to
random controls.

So the direct interaction is not a simple geometric "SELF effect points toward
CORE X" code.

### v4 — direct functional SELF×CORE interaction

Changing CORE strongly changes the downstream functional consequence of SELF,
far beyond random same-displacement directions.

This interaction is healthy in the projected family because CORE remains
stable.

## Updated evidence-backed model

The best current model is:

```
             SELF
               |
         reference/control
               |
               v
INPUT ---> CORE ----[interaction]----> GOAL / FOCUS
```

More precisely:

- CORE carries stable content/reference information;
- SELF supplies a learned control/reference coordinate;
- the transformation induced by SELF depends non-additively on CORE;
- that interaction shapes downstream GOAL/FOCUS;
- projected architectures preserve CORE while performing this interaction;
- diff_anchor performs a similar interaction but entangles/destroys CORE;
- diff_only distributes the interaction across several layers.

This is now stronger than the earlier purely geometric dual-anchor hypothesis.

## Strongest supported claim

The strongest scientifically defensible statement after the full 300M causal
series is:

> **In the projected 300M models, the learned SELF direction functions as a
> causal reference/control variable whose effect on downstream GOAL and FOCUS
> depends non-additively on the current CORE. This SELF×CORE interaction is
> substantially stronger than eight random same-displacement SELF controls
> while CORE identity remains stable.**

This is direct mechanistic evidence for a dual-reference interaction.

It is **not** evidence of semantic selfhood or subjective awareness.

## Next scientific gate

The highest-value next step is replication rather than inventing another
single-seed metric.

Priority:

1. repeat the v4 factorial interaction assay at 150M, 170M and 200M to map when
   the causal interaction emerges;
2. replicate the mechanism on the pretrained Goldfish trajectory;
3. obtain independent training seeds for the projected family before making a
   general architecture claim.
