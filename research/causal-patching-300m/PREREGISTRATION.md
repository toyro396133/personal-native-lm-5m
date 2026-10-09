# Preregistration — 300M causal activation patching

Workflow run: `37916251165`

This document was committed after the assay code was fixed and the workflow
started, but before any arm produced a causal-patching result artifact.

## Scientific question

The 150M–300M structural audits showed repeatable SELF/CORE/GOAL/FOCUS
geometry, but geometry is not causality.

This experiment asks whether the internal representations measured at layers
2–4 actually carry component-specific causal information downstream.

## Models

Primary candidates:

- `projected_diff`
- `projected_diff_slow`

Controls:

- `baseline`
- `diff_anchor` — strong SELF effect / entangled negative control
- `diff_only` — learned CORE-reference signal with late 300M entanglement

All checkpoints are the exact 300M artifacts from source run
`37678419157`.

## Test 1 — component residual activation patching

For CORE, GOAL and FOCUS separately:

1. construct source and target prompts that differ in exactly one component;
2. require exactly equal token length;
3. run both to a selected layer;
4. patch the target residual state with the source residual state;
5. continue the target forward normally;
6. compare with a random perturbation matched to the full patch-delta norm.

Layers: **2, 3, 4**.

### Primary metrics

- source-label switch rate, restricted to examples where both clean source and
  clean target decode correctly;
- continuous source-vs-target centroid-margin shift;
- accuracy of the two components that were not changed.

### Positive causal evidence

A component is considered causally represented at a layer if:

1. targeted patching moves the source-vs-target margin more toward the source
   than the matched random control;
2. source-label switching is at least as strong as the matched random control
   and preferably strictly stronger;
3. the unchanged components remain substantially more stable than the patched
   component.

The continuous margin shift is primary when discrete switch counts are small.

### Factorization evidence

Evidence for CORE separation requires a targeted CORE patch to move CORE toward
the source while preserving GOAL and FOCUS substantially better than would be
expected from an indiscriminate perturbation.

The same logic applies to GOAL and FOCUS.

## Test 2 — single-layer SELF causality

For each SELF architecture and each layer 2–4:

- normal SELF is used at every adapter except the tested layer;
- at the tested layer only, use a radial 2x displacement (equivalent raw anchor
  negation);
- compare against a random direction at exactly the same 2x displacement.

### Primary metrics

- final representation effect norm;
- radial/random effect ratio;
- final CORE/GOAL/FOCUS decoding under each intervention.

### Positive SELF-layer evidence

A layer carries direction-specific SELF influence when the radial intervention
has a larger downstream effect than the matched random intervention.

A ratio above **1.5x** is treated as strong directional selectivity for this
screen, but the exact ratio is reported rather than collapsed to pass/fail.

For the projected family, the preferred pattern is:

- radial SELF effect > random;
- CORE remains comparatively robust;
- GOAL and/or FOCUS are affected without indiscriminate collapse.

For `diff_anchor`, stronger CORE damage or broader cross-component damage is
predicted from the 300M structural audit.

## Falsification / weakening outcomes

The dual-anchor causal interpretation is weakened if:

- targeted CORE/GOAL/FOCUS patches are no more effective than matched random
  perturbations;
- targeted patches indiscriminately destroy all components;
- projected variants do not show better separation than entangled controls;
- radial SELF perturbation is not distinguishable from random matched
  perturbation at layers 2–4.

## Guardrails

- No language-loss ranking is used.
- No result is described as semantic selfhood.
- A causal patch demonstrates sensitivity of the tested representation path,
  not that the human-readable label is uniquely encoded there.
- This remains a single-seed checkpoint study until replicated.
- Layerwise causal evidence is preferred over final-layer geometry when the two
  disagree.

## Decision after this experiment

If projected variants show component-specific patching plus selective
single-layer SELF effects, the next experiment is a more surgical
subspace/token-position patch and multi-seed replication.

If they do not, the structural geometry will be downgraded from a mechanistic
interpretation to a primarily correlational representation signature.
