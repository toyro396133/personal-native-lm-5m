# Structural checkpoint audit at 150M — seven variants

Run: `37706313613`

Status: **completed successfully for all seven variants and comparison**.

This audit intentionally does **not** rank models by language loss. Language
quality remains a health/control metric elsewhere. The primary question here is
whether internal representations become organized around distinct SELF and CORE
references, and how GOAL/FOCUS relations behave around them.

## Methodological update

The assay incorporates lessons from the independent blind follow-up:

- perturbation direction is compared at matched displacement magnitude;
- raw `negate` damage is not treated as sign-specific evidence by itself;
- SELF effect strength is separated from learned-value specificity;
- learned SELF is compared against multiple equal-norm sham references;
- measurements are layerwise, not only at final;
- no single composite "winner" is constructed;
- CORE/GOAL/FOCUS geometry is not called causal without direct component
  interventions.

Matched SELF interventions use four directions at 1x and 2x learned-anchor
displacement:

- radial;
- random;
- orthogonal;
- shuffle-derived.

`radial_1x` is effectively SELF=zero and `radial_2x` is effectively
SELF=negate.

## 1. CORE exists strongly without SELF

At 150M the baseline has:

- final CORE paraphrase accuracy: 1.0;
- CORE under GOAL: 1.0;
- CORE under FOCUS: 1.0;
- final CORE cluster margin: 0.05845.

This continues to support the high-confidence conclusion that CORE organization
does not require the explicit SELF mechanism.

The final CORE classification task is saturated across all seven variants, so
the important differences are in cluster margins, layerwise challenge, learned
reference specificity and intervention behavior.

## 2. Learned SELF specificity remains architecture-dependent

Final learned-reference advantage over one shuffled plus eight equal-norm random
shams:

| variant | CORE | GOAL | FOCUS |
|---|---:|---:|---:|
| projected_diff | +0.01172 | -0.00140 | +0.00277 |
| diff_only | +0.00703 | +0.00547 | -0.00065 |
| projected_diff_slow | +0.00537 | +0.00577 | +0.00264 |
| diff_anchor | +0.00454 | -0.01046 | -0.00462 |
| self_v1_slow | +0.00185 | -0.00352 | -0.00251 |
| self_v1 | +0.00068 | -0.01016 | -0.01000 |

The key result is not the scalar ranking. The polarity structure is the
important evidence.

### projected_diff

- learned CORE reference advantage is positive in every layer;
- learned GOAL reference advantage is positive through layer 5, then becomes
  slightly negative at layer 6/final;
- learned FOCUS reference becomes positive in layers 3/4, again in layer 6 and
  final.

This remains the cleanest early/middle two-reference geometry.

### projected_diff_slow

- learned CORE advantage is positive in every layer;
- GOAL specificity is weak/negative early, then becomes positive in layer 3,
  layer 5, layer 6 and final;
- FOCUS specificity is negative through most of the network but turns positive
  at layer 6 and final.

This is strong evidence for **late integration under the slow schedule** rather
than simply a weaker version of projected_diff.

At final it is the **only variant with positive learned-reference advantage for
CORE, GOAL and FOCUS simultaneously**.

### diff_only

- CORE specificity is positive in every layer;
- GOAL specificity is mostly positive;
- FOCUS specificity stays negative through the network and is only near-zero at
  final.

This is a strong learned-reference architecture, but the learned coordinate
does not cleanly extend to FOCUS.

### diff_anchor

- CORE specificity is positive;
- GOAL specificity is negative in every layer;
- FOCUS specificity is negative in every layer.

This reproduces the earlier distinction: a strong SELF channel and good
language behavior do not imply that the specific learned SELF value is the
right relational coordinate for downstream structure.

### self_v1 / self_v1_slow

Both remain weak on learned-value specificity. GOAL is negative throughout, and
FOCUS is mostly negative. The slow schedule does not repair this architecture's
reference semantics.

## 3. Magnitude-matched perturbations show strong direction selectivity

At **exactly the same displacement magnitude**, radial changes cause much larger
hidden-state effects than random or orthogonal changes.

Final mean representation effect norms:

| variant | radial 1x | random 1x | radial 2x | random 2x |
|---|---:|---:|---:|---:|
| self_v1 | 0.158 | 0.039 | 0.850 | 0.067 |
| self_v1_slow | 0.222 | 0.047 | 0.896 | 0.095 |
| diff_only | 0.320 | 0.080 | 0.890 | 0.151 |
| diff_anchor | 0.385 | 0.062 | 1.140 | 0.141 |
| projected_diff | 0.274 | 0.089 | 0.707 | 0.133 |
| projected_diff_slow | 0.318 | 0.086 | 0.739 | 0.133 |

Thus the SELF direction is not behaving like an arbitrary vector of the same
norm. At final, radial/random effect ratios are about 3–6x at 1x and 5–13x at
2x.

This refines the blind F1 correction. Earlier data correctly showed that raw
negate damage could not be interpreted as sign-only because magnitude was a
major confound. At 150M, after explicitly matching displacement magnitude,
**direction still matters strongly for internal representation displacement**.

That does not by itself establish semantic SELF. It does establish that the
learned direction is functionally special relative to random/orthogonal
directions at the same displacement.

## 4. CORE robustness separates architectures from SELF effect strength

At final:

- random 2x and orthogonal 2x preserve CORE accuracy at 1.0 for every SELF
  variant;
- radial 2x reduces CORE accuracy for:
  - self_v1_slow: 0.50;
  - diff_anchor: 0.833;
  - diff_only: 0.833;
- radial 2x leaves final CORE accuracy at 1.0 for:
  - self_v1;
  - projected_diff;
  - projected_diff_slow.

This is a particularly useful separation.

### self_v1

It has a very large radial-2x hidden effect (~0.850) and axis consistency
(~0.977), yet CORE remains fully preserved.

Therefore **large SELF effect is not the same thing as SELF/CORE entanglement**.

### projected family

Both projected variants are highly robust at final. projected_diff has one
intermediate layer-4 CORE drop under radial 2x (0.833) and then recovers;
projected_diff_slow remains CORE-perfect under radial 2x through all measured
layers.

This strengthens the interpretation of projection as a disentangling buffer.

### self_v1_slow

This is the most fragile architecture under radial 2x:

- layer 1: 0.667;
- layer 2: 0.833;
- layer 3: 0.50;
- layer 4: 0.50;
- layer 5: 0.667;
- layer 6: 0.333;
- final: 0.50.

The same 2x random and orthogonal perturbations preserve CORE at 1.0.

So the slow schedule has strongly amplified a **specific SELF-direction
dependence that interferes with CORE identity**, rather than generic sensitivity
to perturbation magnitude.

## 5. FOCUS remains the stronger challenge dimension

GOAL never reduces CORE classification in this audit: CORE-under-GOAL is 1.0 in
every layer of every variant.

FOCUS does reduce CORE in several architectures.

Mean layerwise `CORE_under_FOCUS - CORE_under_GOAL`:

- baseline: -0.0053
- self_v1: 0.0000
- diff_anchor: -0.0053
- projected_diff_slow: -0.0079
- projected_diff: -0.0185
- diff_only: -0.0265
- self_v1_slow: **-0.0476**

This independently extends the blind follow-up result that the fourth component
acts as a stronger challenge to CORE than the third component.

Caution: this remains conditional/representational evidence. Direct FOCUS
intervention is still needed for a causal claim.

## 6. Relation geometry is strongest internally and compressed at final

Peak GOAL relation layers:

- baseline: layer 3 = 0.2009 -> final 0.1297
- projected_diff_slow: layer 3 = 0.2204 -> final 0.1519
- most other SELF variants peak at layer 1 and decline toward final.

Peak FOCUS relation is usually around layer 4:

- self_v1_slow: 0.2679 -> final 0.1793
- projected_diff_slow: 0.2712 -> final 0.1762
- self_v1: 0.2666 -> final 0.1216
- diff_anchor: 0.2429 -> final 0.1266
- diff_only: 0.2652 -> final 0.1529
- projected_diff: 0.2061 -> final 0.1188

This again supports **internal relational organization followed by final-output
compression**.

## 7. Final structural profile by variant

### projected_diff_slow — best balanced late organization

- positive learned-reference advantage for CORE, GOAL and FOCUS at final;
- strongest final GOAL relation margin: 0.15194;
- near-strongest final FOCUS relation margin: 0.17621;
- complete final CORE robustness under all tested matched perturbations;
- learned GOAL/FOCUS specificity emerges late.

Current interpretation: **strongest balanced SELF/CORE/GOAL/FOCUS organization
at 150M**, especially if the desired behavior is stable dual anchors with
downstream relations.

### projected_diff — strongest CORE learned reference

- highest final learned CORE-reference advantage: +0.01172;
- positive CORE specificity across every layer;
- strong early/middle GOAL specificity;
- very robust CORE under interventions;
- downstream learned GOAL specificity fades near the final representation.

Current interpretation: **cleanest learned SELF-as-coordinate for CORE**, but
some downstream relational specificity is compressed late.

### diff_only — strong learned coordinate, weaker separation

- strong CORE learned-reference advantage;
- positive final GOAL learned-reference advantage;
- highest final CORE cluster margin: 0.07751;
- FOCUS learned specificity does not become positive;
- radial 2x can damage CORE.

Current interpretation: learned reference is real, but SELF/CORE separation is
less clean than the projected family.

### diff_anchor — strong channel, wrong downstream polarity

- positive learned CORE reference;
- negative learned GOAL/FOCUS reference at every layer;
- radial 2x damages CORE;
- representation effect is the strongest of all variants.

Current interpretation: **high SELF influence without the desired downstream
reference organization**.

### self_v1 — strong causal-looking direction, weak learned identity

- huge radial 2x representation effect;
- CORE remains robust;
- learned GOAL/FOCUS specificity is strongly negative.

Current interpretation: a powerful SELF-sensitive control direction, but the
specific learned SELF value is not organizing the desired relational
coordinates.

### self_v1_slow — late FOCUS relation but excessive entanglement

- strongest final FOCUS relation margin: 0.17932;
- highly fragile CORE under radial 2x;
- learned GOAL/FOCUS specificity remains negative.

Current interpretation: slow training strengthened a direction that became
functionally dominant but not well disentangled.

### baseline

CORE and substantial GOAL/FOCUS geometry exist without SELF. This remains the
critical control against attributing all relational organization to the SELF
mechanism.

## 8. Updated conclusions

High confidence from this audit:

1. CORE remains independently organized without SELF.
2. SELF intervention strength, CORE robustness and learned-value specificity are
   distinct axes.
3. Matching perturbation magnitude reveals strong direction-specific SELF
   effects that were hidden by the old zero/negate-only interpretation.
4. Projected architectures preserve CORE best under strong SELF-direction
   interventions.
5. The slow projected architecture develops learned downstream specificity
   later and is currently the most balanced final representation.
6. diff_anchor remains a strong example of "large SELF effect != correct learned
   SELF relational semantics".
7. FOCUS challenges CORE more than GOAL, consistent with the blind follow-up.
8. Internal relation geometry is richer than the final LM representation.

Still unproven:

- semantic selfhood;
- direct causal role of CORE/GOAL/FOCUS;
- literal SELF x CORE -> GOAL computation;
- whether the 150M patterns generalize across random seeds;
- whether the best 150M structure persists to 300M.

## 9. Next experiments

Priority order:

1. repeat the same audit at 160M, 180M, 200M, 250M and 300M as all seven
   checkpoints become available;
2. build an opaque V1–V7 causal runner so the blind analyst can perform direct
   activation patching without unblinding;
3. direct magnitude-matched CORE, GOAL and FOCUS interventions;
4. intermediate-layer activation patching around layers 2–4;
5. multi-seed replication of projected_diff and projected_diff_slow against
   baseline and a matched sham/capacity control;
6. transfer the same primitive audit design to the pretrained Goldfish
   seven-arm experiment once its checkpoints are available.
