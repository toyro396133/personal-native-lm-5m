# Preregistration — 300M SELF adapter contribution causal patching v2

This document freezes the interpretation rules before the v2 workflow is launched.

## Why v2 exists

The v1 experiment showed:

- full residual CORE/GOAL/FOCUS patching is causally effective even in baseline,
  so that intervention is too broad to identify the explicit SELF mechanism;
- single-layer SELF anchor interventions are strongly direction-specific;
- projected_diff / projected_diff_slow preserve CORE while changing downstream
  GOAL/FOCUS;
- diff_anchor contains a layer-3 SELF/CORE entanglement bottleneck.

v2 isolates the explicit SELF adapter contribution:

`adapter(x, SELF) - x`

while leaving the base-transformer residual stream untouched.

## Models

- projected_diff
- projected_diff_slow
- diff_anchor
- diff_only

All are exact 300M checkpoints from source run `37678419157`.

## Test A — source adapter-delta patch

For equal-length source/target prompts differing in exactly one of:

- CORE
- GOAL
- FOCUS

At layer 2, 3 or 4:

1. compute the target adapter contribution `d_t`;
2. compute the source adapter contribution `d_s`;
3. leave the target base residual unchanged;
4. replace only `d_t` with `d_s`;
5. continue the target forward normally.

The intervention relative to normal target state is therefore:

`d_s - d_t`.

Matched random control:

- add a random tensor with exactly the same norm as `d_s - d_t`.

### Primary evidence

Positive evidence that the SELF adapter itself carries a component-specific
causal signal requires targeted adapter-delta patching to move the relevant
source-vs-target centroid margin more than the matched random control.

Discrete source-label switching is secondary because the adapter contribution
may be small relative to the base residual.

The two unchanged components must remain substantially more stable than the
patched component for a factorization claim.

## Test B — adapter ablation

At exactly one tested layer:

- normal state = `x + adapter_delta`;
- ablation state = `x`.

Matched control:

- perturb the normal state by a random tensor with norm exactly equal to the
  removed adapter contribution.

### Priority hypotheses

#### projected_diff_slow layer 2

Prediction:

- adapter ablation should affect GOAL/FOCUS more than CORE;
- any effect should exceed or differ systematically from the matched-random
  control;
- CORE should remain comparatively robust.

#### projected_diff layer 2

Same qualitative prediction, expected weaker than projected_diff_slow.

#### diff_anchor layer 3

Prediction:

- adapter-level intervention should produce broader cross-component damage,
  especially to CORE, consistent with the layer-3 entanglement bottleneck.

#### diff_only

Prediction:

- no single adapter ablation necessarily reproduces all-layer CORE collapse;
- effects may be distributed across layers.

## Interpretation rules

Strong adapter-specific causal evidence:

1. targeted adapter-delta margin shift > matched random;
2. effect repeats for multiple prompts and is not driven solely by a few
   decoder flips;
3. selective component preservation matches the architecture-specific v1
   result.

Weak/negative evidence:

- adapter-delta targeted patch indistinguishable from matched random;
- adapter ablation effects are tiny and non-selective;
- v1 anchor intervention effects cannot be localized to the explicit adapter
  contribution.

## Guardrails

- No language-loss ranking.
- No semantic-selfhood claim.
- No claim that a component is uniquely stored in the adapter.
- Single-seed 300M checkpoints remain a limitation.
- Continuous effect sizes are retained even if discrete decoding stays near
  ceiling.
