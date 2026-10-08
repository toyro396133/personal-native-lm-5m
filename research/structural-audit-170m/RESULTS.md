# Structural checkpoint audit at 170M — seven variants

Run: `37709766525`

Status: **completed successfully for all seven variants and comparison**.

This audit uses the same v3 structural protocol introduced at 150M. Language
loss is not used as the primary ranking. The central measurements are CORE
robustness, GOAL/FOCUS relation geometry, learned SELF reference specificity,
and magnitude-matched SELF interventions.

## Final-layer snapshot at 170M

| variant | CORE cluster | GOAL relation | FOCUS relation | learned CORE ref | learned GOAL ref | learned FOCUS ref | CORE under FOCUS |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.05375 | 0.11252 | 0.15164 | n/a | n/a | n/a | 1.000 |
| diff_anchor | 0.05990 | 0.11225 | 0.14501 | +0.00289 | -0.01079 | -0.00228 | 0.981 |
| diff_only | 0.06424 | 0.11593 | 0.15355 | +0.00608 | +0.00642 | -0.00278 | 1.000 |
| projected_diff | 0.06793 | 0.09680 | 0.13215 | **+0.00965** | +0.00036 | **+0.01351** | 1.000 |
| projected_diff_slow | 0.05273 | **0.15954** | 0.15664 | +0.00381 | **+0.00867** | +0.00666 | 0.944 |
| self_v1 | **0.06916** | 0.11040 | 0.17729 | -0.00095 | -0.00350 | -0.00611 | 0.981 |
| self_v1_slow | 0.04906 | 0.09414 | **0.17859** | +0.00253 | +0.00002 | +0.00559 | 1.000 |

## Magnitude-matched SELF-direction stress at 170M

`radial_2x` is the magnitude-matched analogue of SELF negation.

| variant | CORE accuracy radial 2x | CORE accuracy random 2x | radial effect norm | random effect norm | radial/random |
|---|---:|---:|---:|---:|---:|
| diff_anchor | **0.167** | 1.000 | 1.153 | 0.177 | 6.51x |
| diff_only | 0.667 | 1.000 | 0.936 | 0.147 | 6.35x |
| projected_diff | 0.833 | 1.000 | 0.729 | 0.140 | 5.20x |
| projected_diff_slow | **1.000** | 1.000 | 0.769 | 0.150 | 5.14x |
| self_v1 | **1.000** | 1.000 | 0.880 | 0.078 | **11.21x** |
| self_v1_slow | 0.667 | 1.000 | 0.960 | 0.091 | **10.55x** |

Thus the learned SELF direction remains functionally special even after matching
perturbation magnitude. However SELF effect strength and CORE robustness remain
clearly distinct dimensions.

## Change from 150M to 170M

### projected_diff_slow

- final GOAL relation improves: 0.15194 -> **0.15954**;
- learned GOAL reference improves: +0.00577 -> **+0.00867**;
- learned FOCUS reference improves: +0.00264 -> **+0.00666**;
- learned CORE reference remains positive but weakens: +0.00537 -> +0.00381;
- final CORE-under-FOCUS drops from 1.000 -> **0.944**;
- radial-2x CORE robustness remains perfect at 1.000.

Interpretation: downstream SELF-relative organization continues to strengthen,
but FOCUS creates more conditional pressure on CORE in the normal path.

### projected_diff

- learned CORE reference remains the strongest: +0.01172 -> **+0.00965**;
- learned GOAL reference crosses from slightly negative to slightly positive:
  -0.00140 -> **+0.00036**;
- learned FOCUS reference rises strongly: +0.00277 -> **+0.01351**;
- final radial-2x CORE robustness weakens from 1.000 -> **0.833**.

Interpretation: learned downstream reference semantics improved substantially,
while very strong SELF-direction perturbation now leaks somewhat into CORE.

### diff_only

- CORE and GOAL learned-reference advantages remain positive;
- FOCUS specificity becomes more negative;
- radial-2x CORE accuracy drops from 0.833 -> **0.667**.

Interpretation: reference semantics remain real, but separation does not improve.

### diff_anchor

- learned CORE advantage weakens;
- GOAL remains strongly negative;
- radial-1x now already reduces CORE to 0.833;
- radial-2x collapses CORE to **0.167**.

Interpretation: SELF influence becomes increasingly entangled with CORE while
downstream learned-reference semantics remain misaligned.

### self_v1

- final CORE robustness under radial-2x remains perfect;
- radial/random effect ratio rises to >11x;
- learned CORE/GOAL/FOCUS specificity is negative at final;
- FOCUS relation rises strongly to 0.17729.

Interpretation: a highly special SELF-sensitive direction with strong
downstream geometry, but the learned SELF value itself is still not a good
reference coordinate.

### self_v1_slow

- learned GOAL reference reaches approximately zero;
- learned FOCUS reference becomes positive;
- radial-2x CORE accuracy improves from 0.50 -> 0.667;
- final FOCUS relation remains the strongest of all variants.

Interpretation: some recovery from the 150M entanglement, with late downstream
organization improving, but separation is still weaker than projected_diff_slow.

## Updated structural reading at 170M

There is still no single scalar winner.

- **projected_diff_slow**: best balanced GOAL-oriented dual-anchor organization
  and strongest learned GOAL reference, while preserving CORE under direct
  radial-2x SELF stress.
- **projected_diff**: strongest learned CORE reference and strongest learned
  FOCUS-reference advantage; downstream reference semantics improved sharply
  from 150M.
- **self_v1_slow**: strongest raw FOCUS relation and improved learned FOCUS
  specificity, but still more entangled under strong SELF stress.
- **diff_anchor**: increasingly clear counterexample that a powerful SELF
  channel is not sufficient; CORE becomes highly dependent on the SELF
  direction while GOAL/FOCUS reference polarity remains wrong.
- **baseline**: continues to demonstrate that CORE and substantial relational
  geometry exist without explicit SELF.

The 150->170M change is **non-monotonic and multi-axis**, consistent with the
blind follow-up's "multiple developmental clocks" interpretation rather than a
single smooth maturation process.
