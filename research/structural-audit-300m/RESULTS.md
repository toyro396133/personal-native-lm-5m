# Final structural checkpoint audit at 300M — seven variants

Source continuation run: `37678419157`  
Structural audit run: `37757999981`  
Status: **all seven training phase-20 jobs succeeded; all seven 300M checkpoints exist; structural audit + comparison succeeded.**

The source continuation workflow is marked `failure` only because the legacy
`compare300` job failed inside the old pinned commit. This is not a training
failure.

All variants reached:

`tokens_seen = 300,000,768`

## Final-layer structural snapshot

| variant | CORE cluster | GOAL relation | FOCUS relation | learned CORE ref | learned GOAL ref | learned FOCUS ref | CORE radial-2x |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.07134 | 0.09469 | 0.11900 | n/a | n/a | n/a | n/a |
| diff_anchor | 0.08068 | 0.05885 | 0.08420 | +0.00326 | -0.00975 | -0.00342 | 0.333 |
| diff_only | 0.08218 | 0.07787 | 0.09941 | +0.00789 | -0.00473 | -0.00509 | 0.333 |
| projected_diff | **0.08604** | 0.07256 | 0.12317 | **+0.00953** | -0.00692 | **+0.00881** | **1.000** |
| projected_diff_slow | 0.07294 | **0.10477** | 0.13369 | +0.00681 | -0.00519 | +0.00213 | **1.000** |
| self_v1 | 0.08053 | 0.08138 | 0.10678 | +0.00221 | -0.00469 | -0.01156 | **1.000** |
| self_v1_slow | 0.07042 | 0.04303 | **0.16107** | +0.00033 | **+0.00181** | -0.00190 | 0.833 |

No single scalar winner is scientifically justified.

## 1. projected_diff remains the cleanest learned SELF coordinate for CORE

At 300M, projected_diff again has the strongest final learned CORE-reference
advantage:

- CORE: **+0.00953**
- GOAL: -0.00692
- FOCUS: **+0.00881**

The important layerwise pattern is richer than the final number:

| layer | learned CORE | learned GOAL | learned FOCUS |
|---|---:|---:|---:|
| 1 | +0.0089 | +0.0039 | -0.0023 |
| 2 | +0.0079 | +0.0061 | +0.0008 |
| 3 | **+0.0105** | **+0.0078** | +0.0041 |
| 4 | +0.0088 | +0.0026 | **+0.0090** |
| 5 | +0.0092 | -0.0054 | +0.0048 |
| 6 | +0.0062 | -0.0058 | +0.0063 |
| final | +0.0095 | -0.0069 | +0.0088 |

Thus GOAL specificity is **present and positive in early/middle layers** and is
later reorganized/compressed into a negative final-layer value.

This reinforces the recurring conclusion:

> final hidden state alone is not a faithful summary of the internal
> SELF-relative organization.

## 2. projected_diff_slow remains the strongest balanced raw relational candidate, but GOAL learned-reference polarity is no longer positive at final

Final:

- strongest SELF-variant GOAL relation: **0.10477**
- FOCUS relation: 0.13369
- learned CORE: +0.00681
- learned GOAL: -0.00519
- learned FOCUS: +0.00213
- final radial-2x CORE: **1.000**

Layerwise learned GOAL reference is still positive at:

- layer 2: +0.0010
- layer 3: **+0.0044**

and then turns negative later.

Therefore the 200M state, where final learned CORE/GOAL/FOCUS were all positive,
was not a monotonic endpoint. Continued training reorganized the downstream
coordinate system.

This is strong evidence against a simple "more training = stronger SELF
hierarchy" story.

## 3. The projected family preserves the best SELF/CORE separation at final

Under magnitude-matched radial-2x SELF stress:

- projected_diff final CORE = **1.000**
- projected_diff_slow final CORE = **1.000**
- self_v1 final CORE = **1.000**
- self_v1_slow = 0.833
- diff_anchor = 0.333
- diff_only = 0.333

Random/orthogonal 2x perturbations remain much less damaging than radial SELF
perturbations.

### projected_diff
- radial-2x effect norm: 0.7930
- random-2x: 0.1958
- orthogonal-2x: 0.1700

### projected_diff_slow
- radial-2x effect norm: 0.8229
- random-2x: 0.1307
- orthogonal-2x: 0.2035

So the learned SELF direction remains strongly functionally privileged at 300M
while final CORE identity is preserved.

## 4. Final robustness can hide intermediate-layer entanglement

At 300M, even the projected family is not uniformly CORE-stable through depth.

### projected_diff radial-2x CORE accuracy
- layer 1: 1.000
- layer 2: 1.000
- layer 3: 1.000
- layer 4: 0.833
- layer 5: 1.000
- layer 6: 0.667
- final: **1.000**

### projected_diff_slow radial-2x CORE accuracy
- layer 1: 1.000
- layer 2: 0.833
- layer 3: 0.833
- layer 4: 1.000
- layer 5: 0.833
- layer 6: 0.667
- final: **1.000**

The model can therefore temporarily disturb CORE internally and reconstruct it
before final output.

This makes intermediate-layer activation patching even more important.

## 5. diff_only undergoes a late SELF/CORE entanglement transition

At 200M, diff_only final radial-2x CORE accuracy had recovered to 1.000.

At 300M it is:

**0.333**

Layerwise radial-2x CORE:

- layer 1: 0.833
- layer 2: 1.000
- layer 3: 1.000
- layer 4: 1.000
- layer 5: 0.500
- layer 6: 0.833
- final: 0.333

Yet learned CORE-reference advantage remains strong (+0.00789).

This is a key distinction:

> a learned SELF coordinate can become highly predictive of CORE while also
> becoming too causally entangled with CORE.

Reference specificity and disentanglement are separate axes.

## 6. diff_anchor remains the clearest negative control

At 300M:

- learned CORE: +0.00326
- learned GOAL: -0.00975
- learned FOCUS: -0.00342
- radial-1x CORE: 0.667
- radial-2x CORE: 0.333
- random-2x CORE: 1.000
- orthogonal-2x CORE: 1.000

Radial-2x representation effect:

- SELF direction: **1.3769**
- random same displacement: 0.2667
- orthogonal same displacement: 0.2889

The SELF direction is extremely important, but the architecture does not yield
the desired relational semantics or separation.

This is one of the strongest controls in the whole research program:

> "strong SELF" is demonstrably not the target.

## 7. self_v1 remains a powerful control direction without good learned downstream semantics

Final:

- CORE robustness radial-2x: **1.000**
- learned CORE: +0.00221
- learned GOAL: -0.00469
- learned FOCUS: **-0.01156**

SELF radial-2x effect remains much larger than random/orthogonal alternatives.

Thus self_v1 still behaves like a strong reusable modulation direction rather
than the desired learned relational coordinate.

## 8. self_v1_slow is a distinct late-state phenotype

At 300M:

- strongest final raw FOCUS relation: **0.16107**
- learned GOAL is the only positive final GOAL learned-reference advantage:
  **+0.00181**
- learned CORE is almost neutral: +0.00033
- learned FOCUS is slightly negative: -0.00190
- radial-2x final CORE: 0.833

This is not the clean two-anchor signature, but it suggests slow v1 develops a
different late organization centered more on FOCUS than CORE reference.

## 9. CORE and relation geometry still exist without explicit SELF

Baseline at 300M:

- CORE cluster: 0.07134
- GOAL relation: 0.09469
- FOCUS relation: 0.11900

Internal baseline peaks:

- layer 1 CORE: 0.1242
- layer 1 GOAL: 0.1685
- layer 3 FOCUS: 0.2484

Therefore explicit SELF is not responsible for the existence of CORE or
relation geometry. Its scientific value must be judged by how it organizes,
stabilizes or causally restructures those existing representations.

## 10. 150M -> 170M -> 200M -> 300M confirms multiple developmental clocks

The long continuation gives strong evidence that the structural metrics do not
move monotonically.

### projected_diff learned CORE
- 150M: +0.01172
- 170M: +0.00965
- 200M: +0.01046
- 300M: +0.00953

This is unusually stable.

### projected_diff learned FOCUS
- 150M: +0.00277
- 170M: +0.01351
- 200M: +0.00731
- 300M: +0.00881

### projected_diff learned GOAL
- 150M: -0.00140
- 170M: +0.00036
- 200M: -0.00330
- 300M: -0.00692

Yet layers 1–4 at 300M still show positive GOAL specificity.

### projected_diff_slow learned GOAL
- 150M: +0.00577
- 170M: +0.00867
- 200M: +0.00564
- 300M: -0.00519

The final-layer sign reversal while middle layers remain positive is especially
strong evidence for reorganization/compression rather than simple loss of the
feature.

### diff_only radial-2x CORE
- 150M: 0.833
- 170M: 0.667
- 200M: 1.000
- 300M: **0.333**

This is a dramatic non-monotonic entanglement trajectory.

## 11. Language health/control at 300M

Legacy `compare300` failed because it ran the old pinned comparison script.
The seven Phase-20 evaluation artifacts themselves are valid.

Final nats/character:

| rank | variant | nats/char | vs baseline |
|---|---|---:|---:|
| 1 | self_v1_slow | **1.504098** | 0.058% better |
| 2 | diff_anchor | 1.504732 | 0.016% better |
| 3 | baseline | 1.504973 | — |
| 4 | projected_diff_slow | 1.506287 | 0.087% worse |
| 5 | projected_diff | 1.506371 | 0.093% worse |
| 6 | self_v1 | 1.506749 | 0.118% worse |
| 7 | diff_only | 1.508689 | 0.247% worse |

These are health/control numbers only. They do not override the structural
results.

No catastrophic language degradation occurred.

## Final scientific reading of the 300M run

The strongest stable result over the full 100M->300M continuation is not a
single winning architecture.

It is the separation of several previously conflated properties:

1. **SELF direction selectivity**
2. **learned SELF-reference specificity**
3. **CORE robustness/disentanglement**
4. **GOAL/FOCUS relational organization**
5. **layerwise persistence vs final compression**

The projected family remains the most promising because it combines:

- strong and persistent learned CORE-reference specificity;
- positive FOCUS-reference specificity;
- large direction-specific SELF effects;
- final CORE robustness under severe matched SELF perturbation.

However the 300M result also shows that a fixed final-layer
SELF->CORE->GOAL->FOCUS hierarchy is too simple.

The better working model is:

> **SELF and CORE behave as distinct organizing references, while GOAL and
> FOCUS relations are dynamically constructed, redistributed and compressed
> across depth and training time.**

This is consistent with the user's dual-anchor formulation:

`SELF <-> CORE -> GOAL -> FOCUS`

but the exact downstream computation remains unproven and should now be tested
causally with activation patching.

## Highest-priority next scientific experiment

The long 300M screening line is complete.

The next high-information test is no longer more tokens on the 5M model. It is:

1. activation patching at layers 2–4;
2. direct matched CORE/GOAL/FOCUS interventions;
3. multi-seed replication of projected_diff / projected_diff_slow against
   baseline and negative controls;
4. compare these causal signatures with the ongoing pretrained Goldfish
   full-parameter transfer trajectory.
