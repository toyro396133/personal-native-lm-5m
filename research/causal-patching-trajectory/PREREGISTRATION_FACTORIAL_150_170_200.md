# Preregistration — factorial SELF×CORE causal trajectory, 150M/170M/200M

This study reuses **without modification** the v4 factorial functional
interaction assay that was preregistered and completed at 300M.

No new metric, threshold or interpretation rule is introduced.

## Checkpoints

Milestones:
- 150M
- 170M
- 200M

Variants:
- projected_diff
- projected_diff_slow
- diff_anchor
- diff_only

All checkpoints come from the same v0.18 fresh-continuation lineage used for
the 300M result.

## Layers

- projected_diff: layer 2
- projected_diff_slow: layer 2
- diff_anchor: layer 3
- diff_only: layers 2, 3, 4

These layer choices were fixed by the completed 300M v1–v4 causal series.

## Assay

Exactly the v4 assay:

- radial learned SELF intervention at selected layer;
- 8 random same-displacement SELF directions;
- full factorial CORE × GOAL × FOCUS contexts;
- downstream correct-label margin response;
- non-additive factorial interaction.

Primary cells:

1. CORE factor -> GOAL outcome
2. CORE factor -> FOCUS outcome

Primary statistic: RMS interaction.

Strong evidence uses the same preregistered v4 rule:

- radial > random mean;
- z > 2;
- radial rank #1/9;
- mean-absolute interaction agrees qualitatively.

## Questions

1. At what checkpoint does healthy projected-family SELF×CORE interaction
   become detectable?
2. Is it stable or non-monotonic from 150M -> 170M -> 200M -> 300M?
3. Does diff_anchor pathological interaction emerge before, during or after its
   known CORE-entanglement transition?
4. Is diff_only interaction distributed across layers throughout training or
   only at the late 300M state?

## Interpretation rules

- A 300M-positive result that disappears at earlier milestones means the
  mechanism is late-emerging.
- Repeated positive results at multiple milestones strengthen developmental
  replication but **do not substitute for independent training seeds**.
- Sign/magnitude fluctuations are treated as developmental reorganization,
  not automatically as noise.
- CORE preservation remains a separate axis from interaction strength.
- No language-loss ranking and no semantic-selfhood claim.
