# SELF-layer causal trajectory: 150M -> 170M -> 200M -> 300M

Workflow run: `37922411679`

Status: **16/16 milestone × variant assays succeeded; comparison succeeded.**

Variants:
- diff_anchor
- diff_only
- projected_diff
- projected_diff_slow

Milestones:
- 150M
- 170M
- 200M
- 300M

Every assay intervenes on the SELF anchor at **one adapter layer only**, over
all six transformer layers. Radial perturbations are compared with random and
orthogonal controls at exactly the same displacement magnitude.

Both discrete component accuracy and continuous correct-vs-nearest-wrong
centroid margin are measured at the final representation.

## Main conclusion

The causal trajectory strongly separates two different SELF mechanisms:

> **projected_diff_slow develops a stable early-layer SELF modulation of GOAL
> while largely preserving CORE identity, whereas diff_anchor develops a
> persistent and increasingly severe SELF/CORE entanglement centered on layer 3.**

This is stronger evidence than the earlier geometric audits because only the
SELF reference supplied to one adapter layer is changed.

## 1. projected_diff_slow has a persistent layer-2 causal signature

At layer 2 with radial-2x SELF perturbation:

| checkpoint | radial/control effect ratio | CORE acc | GOAL acc | FOCUS acc |
|---|---:|---:|---:|---:|
| 150M | 7.10x | 1.000 | 0.810 | 0.958 |
| 170M | 7.32x | 0.977 | 0.801 | 0.903 |
| 200M | **10.36x** | 0.995 | **0.481** | 0.935 |
| 300M | 8.71x | **1.000** | 0.741 | 0.843 |

The matched random/orthogonal controls preserve GOAL almost perfectly at these
checkpoints.

Relative to the clean classification margin, radial layer-2 SELF intervention
removes approximately:

### 150M
- CORE margin: 12%
- GOAL margin: **48%**
- FOCUS margin: 12%

### 170M
- CORE: 35%
- GOAL: **45%**
- FOCUS: 24%

### 200M
- CORE: 44%
- GOAL: **87%**
- FOCUS: 24%

### 300M
- CORE: 36%
- GOAL: **52%**
- FOCUS: 30%

Thus GOAL is repeatedly the component closest to the causal SELF modulation at
layer 2, especially at 150M, 200M and 300M.

CORE margin can move substantially without crossing the classification boundary,
which is why continuous margin and discrete accuracy must both be retained.

The 200M checkpoint is the strongest causal expression of this mechanism:
GOAL accuracy falls below 0.5 while CORE remains 0.995.

## 2. diff_anchor shows the opposite phenotype: layer-3 SELF/CORE entanglement

At layer 3, radial-2x SELF intervention:

| checkpoint | effect ratio | CORE acc | GOAL acc | FOCUS acc |
|---|---:|---:|---:|---:|
| 150M | 4.41x | 0.477 | 0.884 | 0.657 |
| 170M | 3.81x | **0.176** | 0.912 | 0.593 |
| 200M | 3.81x | 0.273 | 0.755 | 0.352 |
| 300M | 3.48x | 0.333 | 0.653 | 0.333 |

Relative CORE-margin damage becomes larger than the clean CORE margin itself:

- 150M: 89%
- 170M: **114%**
- 200M: **119%**
- 300M: **129%**

This directly supports the earlier structural interpretation:

> diff_anchor learns a highly influential SELF direction, but the direction
> becomes causally entangled with CORE rather than serving as a clean reference
> around which CORE remains stable.

The entanglement is not a one-off 300M artifact; it is already present by 150M
and persists throughout the trajectory.

## 3. projected_diff has strong SELF direction selectivity but a different causal role

At layer 2 the radial/control effect ratio is:

- 150M: 4.76x
- 170M: 4.33x
- 200M: 4.74x
- 300M: **7.51x**

CORE classification remains **1.000** at every milestone.

However its strongest component damage is not consistently GOAL. At layer 2:

- GOAL remains 0.972 / 0.991 / 0.986 / 0.949
- FOCUS is more sensitive: 0.838 / 0.912 / 0.926 / 0.884

This suggests projected_diff and projected_diff_slow are not simply fast/slow
versions of one identical causal circuit.

Working distinction:

- projected_diff: cleaner learned SELF-as-CORE coordinate, with modest
  downstream modulation;
- projected_diff_slow: stronger early causal modulation of GOAL while
  preserving CORE identity.

## 4. diff_only is weaker and more FOCUS-oriented in the single-layer assay

Layer-2 radial/control effect ratio grows modestly:

- 150M: 2.07x
- 170M: 2.38x
- 200M: 2.57x
- 300M: 2.59x

CORE remains 1.000 throughout this single-layer test.

FOCUS is the most consistently affected component at layer 2, while CORE and
GOAL are comparatively stable.

This is important because the all-layer structural assay showed severe
diff_only CORE collapse at 300M under global radial-2x intervention.

Therefore:

> diff_only's late CORE entanglement is likely an accumulated multi-layer
> effect rather than a single catastrophic layer.

## 5. Layer specificity is real

The architectures do not merely differ in overall intervention strength.

Recurring causal bottlenecks:

- diff_anchor: **layer 3** for CORE entanglement;
- projected_diff_slow: **layer 2** for GOAL modulation;
- projected_diff: layer 2 strong direction selectivity, with additional
  late-layer FOCUS sensitivity;
- diff_only: distributed weaker modulation with substantial FOCUS sensitivity.

This is exactly the type of evidence that pure final-layer geometry could not
supply.

## 6. Development is causal and non-monotonic

The earlier conclusion about multiple developmental clocks survives a direct
causal test.

For projected_diff_slow layer 2, GOAL accuracy under radial-2x moves:

- 150M: 0.810
- 170M: 0.801
- 200M: **0.481**
- 300M: 0.741

The mechanism becomes dramatically stronger by 200M and then partially
reorganizes by 300M.

Therefore the 200M checkpoint was not merely a noisy geometric optimum; it
corresponds to a real causal maximum of the tested SELF->GOAL modulation.

## 7. Updated mechanistic model

The strongest supported picture is now:

- CORE/content is carried primarily by the base transformer pathway;
- SELF adapters are modulatory/reference mechanisms rather than content stores;
- projected architectures can make SELF causally important without forcing
  CORE to depend on it;
- projected_diff_slow especially couples early SELF perturbation to GOAL;
- diff_anchor instead couples SELF strongly into CORE itself;
- downstream structure changes across depth and training time.

A useful working diagram is therefore:

`SELF <-> CORE`

with architecture-specific downstream modulation into GOAL/FOCUS, rather than
a fixed serial SELF -> CORE -> GOAL -> FOCUS chain.

## 8. What is now supported causally

Supported by direct intervention:

1. the learned SELF direction has architecture- and layer-specific downstream
   consequences;
2. those consequences are much larger than matched random/orthogonal
   perturbations;
3. projected_diff_slow can alter GOAL strongly while preserving CORE identity;
4. diff_anchor can causally destroy CORE from a single SELF intervention at
   layer 3;
5. the causal role changes non-monotonically across training checkpoints.

Not yet proven:

- a literal SELF × CORE -> GOAL computation;
- semantic selfhood;
- a direct causal role for an independently manipulated CORE coordinate;
- multi-seed generalization of these exact causal trajectories.

## Next scientific gate

The highest-information next test is a **factorial SELF × CORE intervention**:

- intervene on SELF at the causally active layer;
- manipulate only a learned CORE subspace rather than replacing the whole
  residual stream;
- measure an interaction term on GOAL/FOCUS downstream;
- compare projected_diff_slow against projected_diff and diff_anchor.

That experiment can test whether GOAL is genuinely computed relative to both
SELF and CORE rather than merely being separately sensitive to each.
