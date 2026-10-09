# 300M SELF×CORE interaction field v3 — results

Workflow run: **37917508639**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION_V3_INTERACTION_FIELD.md`

All four planned reports and the comparison completed successfully.

## Executive conclusion

The preregistered **direct SELF×CORE interaction criterion was not confirmed**.

The radial learned-SELF intervention does create highly structured downstream
effect fields, but the structure that most clearly exceeds eight
same-displacement random anchor controls is generally **GOAL/FOCUS-oriented,
not CORE-oriented**.

The strongest updated interpretation is therefore:

> **SELF behaves as a causal reference/control variable for downstream
> relational processing, while CORE remains a distinct content-bearing
> reference in the residual stream.**

The data do not yet justify the stronger claim that the learned SELF direction
forms a privileged causal SELF×CORE interaction field.

This is an important refinement, not a failed experiment: v3 separates
"distinct SELF and CORE references" from the stronger and previously
unproven claim that their interaction itself has already been causally
localized.

## Preregistered primary test

For each prompt:

`delta_SELF(context) = rep(SELF_radial2) - rep(SELF_normal)`

was measured locally after the selected adapter and at the final downstream
representation.

The key question was whether the **direction of this SELF-induced delta field**
encodes CORE across disjoint GOAL/FOCUS contexts more strongly for the learned
radial SELF direction than for eight random anchor directions at the same 2x
displacement.

Two primary CORE metrics were preregistered:

1. cross-context CORE decoding from delta directions;
2. same-CORE vs different-CORE delta cluster margin.

Strong evidence required the radial learned direction to beat the random-sham
distribution on both in a coherent way.

That pattern did not appear.

## projected_diff @ layer 2

### Direct behavioral effect

Normal final decoding:

- CORE: 1.000
- GOAL: 1.000
- FOCUS: 0.981

Radial SELF at L2:

- CORE: **1.000**
- GOAL: 0.949
- FOCUS: 0.884

Eight random same-displacement directions, mean:

- CORE: 1.000
- GOAL: 0.999
- FOCUS: 0.981

This reproduces v1: learned SELF selectively affects downstream GOAL/FOCUS
while CORE remains intact.

### CORE interaction-field test

Local CORE cross-context accuracy:

- radial: 0.833
- random mean: 0.721
- z: +1.43
- radial rank: #1/9

But local CORE cluster margin:

- radial: -0.0004
- random mean: +0.0003
- z: -0.17
- radial rank: #7/9

Final CORE cross-context accuracy:

- radial: 1.000
- random mean: 0.993
- z: +0.47

Final CORE cluster margin:

- radial: 0.0078
- random mean: 0.0170
- z: -0.43

Therefore the direct CORE-interaction criterion is **not met**.

### GOAL/FOCUS organization of the SELF effect

Local GOAL cross-context decoding:

- radial: **0.667**
- random mean: 0.395
- z: **+7.18**
- rank: #1/9

Local FOCUS cross-context decoding:

- radial: **0.647**
- random mean: 0.435
- z: **+3.01**
- rank: #1/9

At final, discrete decoding differences compress, but GOAL delta-field cluster
margin remains unusually structured:

- radial GOAL margin: -0.0213
- random mean: -0.0491
- z: **+3.54**
- rank: #1/9

Thus the learned SELF direction produces an effect field that is much more
organized by downstream GOAL than arbitrary same-magnitude anchor changes.

## projected_diff_slow @ layer 2

### Direct behavioral effect

Normal:

- CORE: 1.000
- GOAL: 0.991
- FOCUS: 0.995

Radial SELF:

- CORE: **1.000**
- GOAL: **0.741**
- FOCUS: **0.843**

Random same-displacement mean:

- CORE: 1.000
- GOAL: 0.988
- FOCUS: 0.992

Again the learned SELF direction is highly selective for downstream structure
while preserving CORE.

### CORE field

Local CORE cross-context decoding looks strong:

- radial: 1.000
- random mean: 0.779
- z: +2.62
- rank: #1/9

However the independently preregistered CORE cluster-margin metric moves in
the opposite direction:

- radial: 0.0007
- random mean: 0.0077
- z: -1.32
- rank: #9/9

Final CORE:

- cross-context radial: 1.000
- random mean: 1.000
- cluster-margin radial: 0.0161
- random mean: 0.0897
- radial rank on margin: #9/9

This discordance means the strong-interaction criterion cannot be claimed.

### GOAL/FOCUS field

Local GOAL decoding from SELF-effect direction:

- radial: **0.824**
- random mean: 0.460
- z: **+4.86**
- rank: #1/9

Local FOCUS:

- radial: **0.765**
- random mean: 0.404
- z: **+4.65**
- rank: #1/9

This is one of the strongest v3 findings.

The learned SELF direction is not simply causing a larger generic
perturbation. Its causal effect field is unusually organized by the current
downstream GOAL/FOCUS state.

## diff_anchor @ layer 3

diff_anchor remains the pathological negative control.

Radial final decoding:

- CORE: **0.333**
- GOAL: 0.653
- FOCUS: **0.333**

Random same-displacement means:

- CORE: 0.999
- GOAL: 0.994
- FOCUS: 0.972

### CORE effect field

Final CORE cross-context decoding:

- radial: 1.000
- random mean: 1.000

But CORE cluster margin is dramatically **worse** for the learned radial
direction:

- radial: 0.0263
- random mean: 0.1720
- z: **-4.16**
- rank: #9/9

Thus even the architecture with catastrophic SELF/CORE entanglement does not
support the preregistered "privileged CORE-structured SELF field" hypothesis.

### GOAL/FOCUS field

Final GOAL:

- cross-context radial: 1.000
- random mean: 0.949
- z: **+4.04**
- GOAL cluster-margin z: **+7.26**
- rank: #1/9

Final FOCUS:

- cross-context radial: 0.951
- random mean: 0.913
- z: **+2.67**
- FOCUS cluster-margin z: **+4.05**
- rank: #1/9

So the pathological SELF direction also strongly reorganizes downstream
GOAL/FOCUS structure — but unlike the projected family, it simultaneously
destroys CORE.

This sharply separates:

- **downstream control strength**, from
- **healthy reference separation**.

## diff_only

diff_only was tested at layers 2, 3 and 4.

No single layer produces robust preregistered CORE-field evidence.

Notably, final CORE cluster margins for radial SELF are consistently no better
and often worse than random controls.

However downstream GOAL/FOCUS field structure appears repeatedly.

At layer 4 final:

GOAL cluster margin:
- radial: -0.0263
- random mean: -0.0576
- z: **+3.46**
- rank: #1/9

FOCUS cluster margin:
- radial: -0.0300
- random mean: -0.0660
- z: **+3.81**
- rank: #1/9

This fits the earlier conclusion that diff_only is distributed rather than
localized, but still uses SELF in downstream relational processing.

## What v3 rules out / weakens

The following strong statement is **not supported**:

> "The learned SELF direction creates a privileged CORE-specific causal effect
> field, proving a direct SELF×CORE interaction."

Random same-displacement directions can often carry as much or more CORE
structure in their effect fields.

CORE cross-context accuracy alone is insufficient because it can be high even
when the independent cluster-margin criterion fails.

## What v3 strengthens

The following statement is now substantially better supported:

> **The learned SELF direction causally modulates downstream relational
> processing in a way that is unusually organized by GOAL/FOCUS compared with
> arbitrary same-displacement anchor directions.**

In projected models this occurs while CORE remains intact.

In diff_anchor it occurs together with CORE collapse.

Therefore the architecture determines whether a strong SELF control field is
**disentangled** or **pathologically coupled** to the stable content state.

## Updated mechanistic picture

The evidence no longer justifies drawing an experimentally established edge:

`SELF <-> CORE`

as a direct causal interaction.

A safer evidence-backed diagram is currently:

```
INPUT ---> CORE --------------------+
                                    |
SELF ---- reference/control ------> downstream relational processing
                                    |
                               GOAL / FOCUS
```

The possibility that SELF and CORE interact remains plausible, but v3 did not
show that the learned SELF direction carries uniquely CORE-conditioned effects.

The strongest distinction is instead:

- CORE = stable content/reference structure;
- SELF = learned causal control/reference direction;
- GOAL/FOCUS = downstream relational state strongly affected by SELF;
- architecture determines whether that control preserves or destroys CORE.

## Combined v1–v3 conclusion

### v1
Changing SELF at one layer has direction-specific causal effects.
Projected L2 changes GOAL/FOCUS while preserving CORE; diff_anchor L3 can
collapse all components.

### v2
The adapter contribution is not a swappable semantic payload.
SELF behaves more like a control/reference input to a transformation.

### v3
The causal SELF effect field is **not uniquely CORE-structured** relative to
random controls. It is much more consistently and exceptionally organized by
GOAL/FOCUS.

Together:

> **SELF is best modeled at this stage as a learned control/reference axis
> acting on content-bearing residual representations, with a particularly
> strong role in downstream GOAL/FOCUS organization.**

This is a narrower and better-supported claim than the original direct
SELF×CORE interaction hypothesis.

## Next high-information experiment

The next test should be factorial rather than geometric:

1. independently manipulate CORE content and SELF reference;
2. measure downstream GOAL/FOCUS response;
3. compute the non-additive interaction term:
   `effect(SELF+CORE) - effect(SELF) - effect(CORE)`;
4. compare with matched random SELF directions and sham CORE patches.

That would test a SELF×CORE interaction directly at the **functional response**
level rather than trying to infer it from delta-field clustering.
