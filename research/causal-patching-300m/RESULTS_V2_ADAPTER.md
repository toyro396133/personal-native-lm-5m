# 300M SELF adapter contribution causal patching v2 — results

Workflow run: **37916891722**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION_V2_ADAPTER.md`

All four planned 300M SELF variants completed successfully:

- diff_anchor
- diff_only
- projected_diff
- projected_diff_slow

## Executive conclusion

v2 changes the mechanistic interpretation in an important way.

The explicit SELF adapter contribution itself does **not** behave like a
component-content channel that stores swappable CORE/GOAL/FOCUS information.

Instead, the strongest evidence is now consistent with:

> **SELF acts primarily as a reference/control variable that changes how the
> adapter transforms the current residual state, rather than as a payload
> containing CORE/GOAL/FOCUS content.**

This reconciles three previously separate observations:

1. learned SELF direction interventions can have very large causal effects;
2. those effects can selectively alter downstream GOAL/FOCUS while preserving
   CORE in projected architectures;
3. directly transplanting the adapter contribution from one semantic state to
   another does not transplant the semantic component.

## Test A — source adapter-delta patching is largely negative

The v2 source patch leaves the base-transformer residual from the target prompt
untouched and replaces only:

`d_target = adapter(x_target, SELF) - x_target`

with:

`d_source = adapter(x_source, SELF) - x_source`.

The intervention is therefore much more surgical than v1 whole-residual
patching.

Across all four architectures and all tested layers:

- discrete source-label switch advantage is **0.0** for CORE, GOAL and FOCUS;
- continuous source-vs-target margin advantages are tiny, generally
  approximately 1e-5 to 1e-3, and mixed in sign;
- unchanged components remain almost perfectly stable.

This is strong evidence against the hypothesis that the adapter contribution
itself contains a readily transplantable component representation.

### projected_diff

Representative margin advantages:

- CORE L2: -0.000034
- GOAL L3: +0.000044
- FOCUS L2: +0.000243

No source-label switches.

### projected_diff_slow

Representative margin advantages:

- CORE L3: +0.000812
- GOAL L4: +0.000186
- FOCUS L2: +0.000202

Again, no source-label switches.

### diff_anchor

Representative margin advantages:

- CORE L2: +0.000279
- CORE L3: -0.001159
- GOAL L4: +0.000078

No source-label switches despite the very strong v1 SELF-direction causality.

### diff_only

The same pattern holds. The largest adapter-delta intervention effects do not
produce semantic source transfer.

## Test B — adapter ablation is directionally non-random but semantically mild

Removing the learned adapter contribution at one layer produces a larger final
representation displacement than a matched random perturbation in many cases.

### diff_anchor

Ablation/random effect ratio:

- L2: 1.76x
- L3: 2.11x
- L4: **2.51x**

Yet CORE/GOAL/FOCUS decoding remains 1.000 across these ablations on the tested
subset.

This is particularly important at L3.

In v1, changing the SELF reference at diff_anchor L3 caused:

- CORE 0.375
- GOAL 0.639
- FOCUS 0.361

But **removing the adapter contribution itself** at L3 leaves all three at
1.000.

Therefore the diff_anchor L3 failure is not because the adapter stores CORE and
we deleted it.

The failure arises when the learned SELF reference drives the adapter
transformation in a specific wrong/entangled direction.

## projected_diff

Ablation/random effect ratio:

- L2: 1.47x
- L3: 1.52x
- L4: 1.85x

CORE remains 1.000 throughout.

Only L2 shows a tiny GOAL decoding reduction relative to matched random:

- ablation GOAL: 0.9896
- random GOAL: 1.0000

This is much smaller than the v1 radial-SELF intervention effect.

## projected_diff_slow

Ablation/random effect ratio:

- L2: 1.40x
- L3: **1.91x**
- L4: 1.78x

CORE remains 1.000 throughout.

At L2:

- ablation GOAL: 0.9792
- matched random GOAL: 0.9896
- FOCUS remains 1.000

Again, this is tiny compared with the v1 radial SELF intervention, where L2
GOAL/FOCUS fell to 0.681/0.764 while CORE stayed 1.000.

## diff_only

Ablation/random effect ratio:

- L2: 2.62x
- L3: 2.08x
- L4: 1.07x

CORE remains 1.000 at every layer.

This further supports the interpretation that the all-layer diff_only CORE
collapse is produced by distributed reference-conditioned interactions rather
than by deleting a content-bearing adapter channel.

## Combined interpretation of v1 + v2

### What v1 showed

Changing the SELF reference at a single layer can strongly and selectively
change downstream computation.

Most strikingly:

- projected_diff_slow L2: SELF radial/random effect ratio 7.63x, CORE 1.000,
  GOAL 0.681, FOCUS 0.764;
- projected_diff L2: ratio 4.77x, CORE 1.000, GOAL 0.944, FOCUS 0.833;
- diff_anchor L3: CORE/GOAL/FOCUS collapse under radial SELF but not matched
  random.

### What v2 now rules out / weakens

The causal effect is **not well explained by**:

> "the adapter output contains a semantic SELF/CORE/GOAL/FOCUS vector that can
> simply be copied between states."

Source adapter-delta transplantation does not transplant the semantic label.

Adapter removal also does not reproduce the severe radial-reference effects.

### Updated mechanistic model

The better interpretation is:

> **SELF is a control/reference coordinate that changes the transformation
> applied to the current content-bearing residual state.**

The residual stream carries CORE/GOAL/FOCUS content.

SELF conditions how that content is processed.

This is substantially closer to a reference-frame mechanism than to a memory
slot or semantic payload.

## Architecture-specific implications

### projected_diff / projected_diff_slow

The projected architectures appear to use SELF as a comparatively
**disentangled control reference**:

- changing SELF at L2 changes downstream GOAL/FOCUS;
- CORE remains stable;
- deleting/transplanting the adapter contribution alone has little semantic
  effect.

This is consistent with the user's dual-anchor picture:

`SELF <-> CORE`

where SELF does not contain CORE, but changes how the current CORE-centered
state is interpreted/organized downstream.

### diff_anchor

diff_anchor uses the SELF reference in a way that becomes catastrophically
entangled at L3.

Because deleting the adapter contribution is harmless while changing SELF is
destructive, the problem is specifically the **reference-conditioned mapping**,
not the mere presence of the adapter path.

### diff_only

The mechanism appears distributed and cumulative. No single adapter ablation
removes CORE, matching the earlier inference from v1.

## Scientific status

The strongest supported statement after v1+v2 is now:

> In the 300M projected models, the learned SELF variable causally controls
> downstream processing in an architecture- and layer-specific manner while
> CORE can remain stable. The explicit adapter contribution is not itself a
> swappable semantic content channel, supporting a control/reference
> interpretation of SELF rather than a storage interpretation.

This is stronger than correlational geometry, but still does not establish
semantic selfhood.

## Highest-information next test

The next experiment should test the **interaction field** directly:

> Does the causal effect of changing SELF depend systematically on which CORE
> is present?

At projected L2 and diff_anchor L3:

1. hold GOAL/FOCUS fixed;
2. vary CORE;
3. measure the SELF-induced adapter/output delta for each CORE;
4. test whether same-CORE SELF-effect directions cluster more strongly than
   different-CORE directions;
5. compare radial SELF with matched random anchor displacement;
6. test whether the SELF×CORE interaction predicts downstream GOAL/FOCUS
   changes.

A positive result would move the dual-anchor hypothesis from
"SELF is a causal control reference" toward direct evidence of a
**SELF-by-CORE interaction mechanism**.
