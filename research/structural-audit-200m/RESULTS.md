# Structural checkpoint audit at 200M — seven variants

Run: `37720011983`

Status: **completed successfully for all seven variants and comparison**.

This is the same v3 structural audit used at 150M and 170M. Language loss is
not the primary metric. The analysis focuses on SELF direction selectivity,
CORE robustness, learned-reference specificity, GOAL/FOCUS relation geometry,
and layerwise organization.

## Final-layer snapshot at 200M

| variant | CORE cluster | GOAL relation | FOCUS relation | learned CORE ref | learned GOAL ref | learned FOCUS ref | CORE radial-2x |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.07598 | 0.09612 | 0.12555 | n/a | n/a | n/a | n/a |
| diff_anchor | 0.07593 | 0.12164 | 0.10957 | +0.00502 | -0.01039 | -0.00544 | **0.167** |
| diff_only | 0.08682 | 0.07433 | 0.15664 | +0.00512 | +0.00043 | -0.00339 | 1.000 |
| projected_diff | 0.08088 | 0.10243 | 0.14665 | **+0.01046** | -0.00330 | **+0.00731** | 1.000 |
| projected_diff_slow | 0.07507 | **0.15496** | **0.21136** | +0.00700 | **+0.00564** | +0.00265 | 1.000 |
| self_v1 | **0.08826** | 0.10417 | 0.13583 | +0.00109 | -0.01101 | -0.01073 | 1.000 |
| self_v1_slow | 0.07186 | 0.11355 | 0.14127 | +0.00297 | -0.00505 | +0.00042 | 1.000 |

## 1. projected_diff_slow is now the clearest balanced structural candidate

At 200M it combines:

- strongest final GOAL relation: **0.15496**;
- strongest final FOCUS relation: **0.21136**;
- positive learned-reference advantage for CORE, GOAL and FOCUS;
- perfect final CORE preservation under radial-2x SELF stress;
- strong internal GOAL and FOCUS relation geometry.

Layerwise relation peaks:

- GOAL: layer 1 = 0.2006, layer 3 = 0.1965, layer 5 = 0.1976;
- FOCUS: layer 3 = 0.2725, layer 4 = 0.2736.

The final representation compresses these relations but preserves more of them
than the competing variants.

This is the strongest evidence so far for the desired pattern:

> SELF is functionally special, CORE remains stable, and downstream
> relational structure is organized around the two-anchor system without
> collapsing the anchors into one another.

## 2. projected_diff still has the cleanest learned CORE coordinate

At final:

- learned CORE-reference advantage: **+0.01046**, highest of all SELF variants;
- learned FOCUS-reference advantage: **+0.00731**, also highest;
- CORE remains perfect under radial-2x at final.

Layerwise, learned GOAL specificity is strongly positive in the middle:

- layer 1: +0.0041
- layer 2: +0.0100
- layer 3: **+0.0152**
- layer 4: +0.0090

It then becomes slightly negative in layers 5/6/final.

Interpretation: projected_diff builds a very clean SELF-relative coordinate
system early/mid-network, but some GOAL specificity is compressed or
reorganized late.

## 3. diff_anchor becomes an increasingly useful negative control

At 200M:

- learned CORE reference is positive: +0.00502;
- learned GOAL reference remains strongly negative: **-0.01039**;
- learned FOCUS reference remains negative: -0.00544;
- radial-2x SELF stress collapses final CORE accuracy to **0.167**;
- random/orthogonal perturbations of the same displacement preserve CORE at
  1.000.

Its radial-2x representation effect is **1.354**, versus only **0.213** for
random-2x.

This is strong evidence that the learned SELF direction is functionally
special, but in this architecture it becomes entangled with CORE while failing
to provide the desired downstream relational coordinate.

Thus:

> high SELF influence is not sufficient, and can be actively undesirable.

## 4. diff_only recovers CORE robustness but loses relational quality

From 170M to 200M:

- radial-2x CORE accuracy improves from 0.667 -> **1.000**;
- learned CORE reference remains positive: +0.00512;
- learned GOAL reference falls to near zero: +0.00043;
- learned FOCUS reference remains negative;
- final GOAL relation falls sharply to **0.07433**.

It remains evidence that a learned SELF coordinate can exist without clean
multi-component organization.

## 5. self_v1 remains a strong direction but weak learned reference

At final:

- radial-2x effect: **0.907**;
- random-2x effect: **0.091**;
- ratio: about **10x**;
- CORE remains 1.000 under radial-2x;
- learned GOAL: -0.01101;
- learned FOCUS: -0.01073.

This continues to separate two concepts:

1. the model is very sensitive to the learned SELF direction;
2. the specific learned SELF value is not necessarily a useful coordinate for
   downstream relational organization.

## 6. self_v1_slow recovers final CORE robustness but remains internally entangled

At final radial-2x CORE accuracy has improved:

- 150M: 0.50
- 170M: 0.667
- 200M: **1.000**

However the layerwise audit still shows strong intermediate fragility:

- layer 1: 0.667
- layer 2: 0.333
- layer 3: 0.50
- layer 4: 0.667
- layer 5: 0.833
- layer 6: 0.833
- final: 1.000

So the final output repairs/reconstructs CORE after substantial internal
interference. Final robustness alone would therefore hide important
architecture behavior.

## 7. FOCUS remains a stronger challenge than GOAL

GOAL leaves CORE classification at 1.000 in every measured layer of every
variant.

FOCUS creates intermediate-layer degradation in several variants, including:

- projected_diff_slow;
- projected_diff;
- self_v1;
- self_v1_slow;
- diff_only;
- diff_anchor.

This continues the independent blind finding that the fourth component behaves
as a stronger challenge dimension than the third.

Again, this is conditional/representational evidence; direct FOCUS intervention
is still required for a causal claim.

## 8. 150M -> 170M -> 200M is clearly non-monotonic

The trajectories do not support a single smooth maturation stage.

### projected_diff_slow

Final GOAL relation:

- 150M: 0.15194
- 170M: 0.15954
- 200M: 0.15496

Final FOCUS relation:

- 150M: 0.17621
- 170M: 0.15664
- 200M: **0.21136**

Learned CORE reference:

- 150M: +0.00537
- 170M: +0.00381
- 200M: **+0.00700**

The system reorganizes rather than simply strengthening monotonically.

### projected_diff

Learned CORE reference remains consistently strongest:

- 150M: +0.01172
- 170M: +0.00965
- 200M: +0.01046

Learned FOCUS reference:

- 150M: +0.00277
- 170M: **+0.01351**
- 200M: +0.00731

Again, a non-monotonic developmental clock.

### diff_anchor

Radial-2x CORE accuracy:

- 150M: 0.833
- 170M: 0.167
- 200M: 0.167

The SELF/CORE entanglement transition observed around 170M persists at 200M.

## 9. Updated working map at 200M

There is still no scientifically justified single scalar winner, but the
working interpretation is now clearer:

### projected_diff_slow
**Best balanced two-anchor / downstream organization.**

It currently provides the strongest combined evidence for stable CORE plus
SELF-relative GOAL/FOCUS organization.

### projected_diff
**Cleanest learned SELF coordinate for CORE.**

It also provides strong FOCUS specificity, with GOAL specificity strongest in
middle layers rather than final.

### diff_only
**Real learned reference, incomplete hierarchy.**

CORE reference persists, but GOAL/FOCUS organization is inconsistent.

### diff_anchor
**Strong SELF direction with damaging entanglement.**

Useful as a negative control demonstrating that SELF strength is not the target.

### self_v1
**Powerful control direction without useful learned-reference semantics.**

### self_v1_slow
**Internal entanglement followed by late repair.**

### baseline
CORE and substantial GOAL/FOCUS geometry exist without explicit SELF, so SELF
must be judged by the organization it adds or restructures, not by the mere
presence of those relations.

## 10. Current strongest claim

The evidence through 200M supports this formulation more strongly than the old
serial SELF->CORE story:

> The promising architectures treat SELF and CORE as distinct organizing
> references. The best current variant, projected_diff_slow, allows SELF to be
> functionally special while preserving CORE and supporting downstream
> GOAL/FOCUS structure around the reference system.

This remains a structural/mechanistic claim, not a claim of semantic selfhood.

## Next gate

The definitive checkpoint of this run is 300M. The same v3 audit should execute
automatically over all seven 300M checkpoints as soon as the long continuation
workflow finishes.
