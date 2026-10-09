# SELF × CORE factorial functional interaction: 150M -> 300M

Runs:

- 150/170/200M trajectory: `37918632896`
- 300M final interaction assay: `37918122058`

The assay changes SELF at one causally active adapter layer and asks whether the
effect of that SELF intervention on final component margins depends on the
independently varied CORE value in a full factorial prompt grid.

The radial SELF intervention is compared with **eight random SELF directions at
the same displacement magnitude**.

Primary cells:

- CORE factor -> GOAL margin response
- CORE factor -> FOCUS margin response

## Main result

The projected family, especially projected_diff_slow, shows a persistent
CORE-conditioned SELF effect on GOAL/FOCUS that is substantially larger than
matched random SELF directions.

This is the strongest evidence so far for a functional interaction between the
SELF reference and CORE context.

Important guardrail:

> SELF is directly intervened on, while CORE is independently varied in the
> factorial input grid. This demonstrates a causal SELF effect whose magnitude
> depends on CORE, but it is **not yet a direct internal causal intervention on
> a learned CORE coordinate**.

## projected_diff_slow: persistent CORE -> GOAL interaction under SELF perturbation

Layer 2, RMS interaction:

| checkpoint | radial | random mean | ratio | z vs random | rank |
|---|---:|---:|---:|---:|---:|
| 150M | 0.00783 | 0.00135 | **5.81x** | 18.44 | 1/9 |
| 170M | 0.00540 | 0.00141 | **3.83x** | 8.27 | 1/9 |
| 200M | 0.00897 | 0.00193 | **4.64x** | 6.91 | 1/9 |
| 300M | 0.00657 | 0.00146 | **4.52x** | 8.01 | 1/9 |

The effect is therefore:

- present at every audited milestone;
- several times larger than random same-displacement SELF directions;
- ranked first among the learned radial direction plus eight random shams every
  time;
- strongest at 150M/200M, again showing non-monotonic development rather than a
  simple monotonic growth curve.

### CORE -> FOCUS

The same layer also shows a strong CORE-conditioned SELF effect on FOCUS:

| checkpoint | radial/random RMS ratio |
|---|---:|
| 150M | 4.36x |
| 170M | **5.64x** |
| 200M | 3.98x |
| 300M | 3.59x |

Again, the radial SELF direction ranks first against all eight random controls.

## projected_diff also shows a persistent interaction

Layer 2 CORE -> GOAL RMS ratio:

- 150M: 2.87x
- 170M: 3.03x
- 200M: 2.38x
- 300M: **4.19x**

CORE -> FOCUS:

- 150M: 3.07x
- 170M: 4.56x
- 200M: 3.41x
- 300M: **4.84x**

Thus projected_diff also has a real CORE-conditioned SELF effect.

At 300M, its interaction strength becomes especially symmetric across:

- CORE -> CORE: 4.69x random
- CORE -> GOAL: 4.19x
- CORE -> FOCUS: 4.84x
- GOAL -> GOAL: 4.16x
- FOCUS -> FOCUS: 4.40x

This looks more like a broadly structured reference transformation than the
stronger GOAL-specific phenotype of projected_diff_slow at earlier checkpoints.

## diff_anchor: interaction exists, but CORE entanglement dominates

At layer 3, CORE -> GOAL interaction is above random at every milestone:

- 150M: 2.33x
- 170M: 1.56x
- 200M: 1.96x
- 300M: 1.61x

However its CORE -> CORE interaction is far larger:

- 150M: 5.32x
- 170M: **8.25x**
- 200M: 4.95x
- 300M: 5.73x

Together with the direct layer-3 intervention that collapses CORE accuracy,
this indicates that diff_anchor's SELF×context interaction is dominated by
SELF/CORE entanglement rather than a clean relational reference mechanism.

This is an important control:

> merely finding a SELF×CORE interaction is not sufficient; the interaction
> must be interpreted together with CORE preservation.

## diff_only: distributed interaction rather than a clean single bottleneck

diff_only shows significant CORE-conditioned SELF effects at layers 2–4.

At 300M CORE -> GOAL RMS ratios:

- L2: 2.24x
- L3: 2.93x
- L4: 2.54x

CORE -> FOCUS:

- L2: 2.51x
- L3: **4.31x**
- L4: 2.73x

This reinforces the earlier conclusion that diff_only is a more distributed
multi-layer mechanism. Its late global CORE entanglement is not localized to a
single catastrophic layer.

## Combining the factorial and direct-causal results

The two assays answer complementary questions.

### Direct one-layer SELF intervention

Shows **what breaks when SELF is changed**.

Key result:
- projected_diff_slow L2 can strongly disrupt GOAL while preserving CORE;
- diff_anchor L3 directly disrupts CORE.

### Factorial CORE-conditioned SELF effect

Shows **whether SELF's downstream effect depends on which CORE is active**.

Key result:
- projected_diff_slow CORE->GOAL interaction remains 3.8x–5.8x above random
  controls across 150–300M;
- projected_diff also shows persistent interaction;
- diff_anchor interaction is dominated by much stronger CORE self-dependence.

Together these support a stronger mechanistic statement than geometry alone:

> in the projected family, SELF is not merely a global additive vector. Its
> causal downstream effect changes as a function of CORE/context, while CORE
> can remain separately identifiable.

## Updated working model

The evidence now favors:

`SELF <-> CORE`

with a context-dependent interaction that modulates downstream GOAL/FOCUS.

For projected_diff_slow specifically, layer 2 is the clearest candidate
interaction site.

For diff_anchor, layer 3 is instead an entanglement site where SELF intervention
directly destabilizes CORE.

## What this does and does not establish

### Supported
- SELF has a direction-specific causal effect.
- The effect is architecture- and layer-specific.
- In projected_diff_slow, the SELF effect on GOAL depends strongly on CORE.
- The interaction persists across 150M–300M.
- The interaction is much larger than eight random same-displacement SELF
  controls.
- Clean reference behavior and harmful entanglement are separable mechanisms.

### Not yet established
- direct internal causal manipulation of CORE itself;
- a literal mathematical SELF × CORE gate;
- semantic selfhood;
- multi-seed replication of the complete causal pattern.

## Next strongest test

A direct **CORE-subspace intervention** should now be used:

1. learn a CORE subspace from held-out factorial contexts;
2. patch only that subspace while leaving the orthogonal residual target-derived;
3. cross that intervention with normal/radial/random SELF at the active layer;
4. measure the factorial interaction on GOAL/FOCUS;
5. compare projected_diff_slow, projected_diff and diff_anchor.

That would turn the current causal-SELF / conditional-CORE evidence into a
two-sided internal intervention test.
