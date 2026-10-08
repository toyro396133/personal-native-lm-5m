# Goldfish 124M structural transfer audit at 1M — results

Source canary: **37704611983**  
Structural audit: **37723626343**  
Status: **all seven arms + comparison completed successfully**

Backbone: `goldfish-models/heb_hebr_1000mb`  
Training regime: full-parameter continued training, 1M continuation tokens per arm.

## Executive conclusion

The transfer is **real but incomplete**.

After only 1M continuation tokens:

1. the added SELF direction becomes functionally special in every SELF arm;
2. this direction is more influential than random/orthogonal directions at the
   same displacement magnitude;
3. CORE remains completely robust under all tested SELF interventions;
4. some learned GOAL/FOCUS reference specificity is already emerging;
5. however, the strong learned SELF-as-CORE-coordinate signature seen in the
   small-model projected family has **not yet transferred**.

Therefore the current result is best described as:

> **functional SELF transfer before full two-anchor structural
> crystallization.**

This is a stronger and more informative result than either "it worked" or "it
did not work".

## 1. CORE is already strong in the pretrained backbone

At final layer, CORE-under-GOAL and CORE-under-FOCUS accuracy are 1.000 for all
seven arms, including baseline.

Final CORE cluster margins:

| arm | variant | CORE margin |
|---|---|---:|
| V1 | baseline | 0.08247 |
| V2 | self_v1 | **0.08316** |
| V3 | self_v1_slow | 0.08284 |
| V4 | diff_only | 0.08266 |
| V5 | diff_anchor | 0.08243 |
| V6 | projected_diff | 0.08248 |
| V7 | projected_diff_slow | 0.08245 |

The SELF mechanisms have not materially changed raw final CORE separability
after only 1M tokens.

This is expected for a pretrained backbone in which semantic/content structure
already exists before SELF is introduced.

## 2. Magnitude-matched direction selectivity transferred immediately

At final layer, radial SELF perturbations cause much larger representation
changes than random or orthogonal perturbations at exactly the same
displacement magnitude.

### 2x displacement

| arm | radial effect | random effect | orthogonal effect | radial/random |
|---|---:|---:|---:|---:|
| V2 | 0.09910 | 0.04160 | 0.03560 | 2.38x |
| V3 | 0.09269 | 0.04037 | 0.03372 | 2.30x |
| V4 | 0.08586 | 0.03164 | 0.03226 | 2.71x |
| V5 | 0.07661 | 0.03532 | 0.03239 | 2.17x |
| V6 | 0.08428 | 0.03137 | 0.02873 | 2.69x |
| V7 | 0.08415 | 0.03165 | 0.02714 | 2.66x |

At 1x displacement the selectivity is even clearer in several variants; for
example V6 radial/random is about 3.95x and V7 about 3.71x.

Thus:

> the learned SELF direction is not behaving like an arbitrary vector of the
> same norm.

This directly transfers one of the strongest small-model findings to a
pretrained 124M Hebrew backbone.

## 3. SELF effect is globally coherent across contexts

Final intervention-axis consistency is extremely high:

- V2 radial-1x: 0.9982
- V3 radial-1x: 0.9985
- V4 radial-1x: 0.9987
- V5 radial-1x: 0.9983
- V6 radial-1x: 0.9982
- V7 radial-1x: 0.9982

Even radial-2x remains ~0.987–0.997.

This means SELF perturbation does not merely inject arbitrary local noise. It
moves hidden representations in a highly reusable cross-context direction.

Caution: coherence alone does not establish semantic selfhood.

## 4. CORE is not yet entangled with SELF

Unlike several 150–200M small-model checkpoints, every Goldfish SELF arm keeps
CORE accuracy at 1.000 under:

- radial 1x;
- radial 2x;
- random 2x;
- orthogonal 2x;

and this remains true across every captured layer.

This is a very encouraging transfer property.

At 1M, the pretrained backbone has learned to use SELF without allowing strong
SELF perturbations to destroy CORE identity.

It is too early to know whether this robustness will remain after longer
co-adaptation.

## 5. Learned-reference specificity is only partially transferred

Final learned SELF advantage over equal-norm shuffled/random sham references:

| arm | CORE | GOAL | FOCUS |
|---|---:|---:|---:|
| V2 self_v1 | **+0.00036** | +0.00211 | +0.00110 |
| V3 self_v1_slow | -0.00117 | +0.00233 | +0.00250 |
| V4 diff_only | -0.00279 | ~0.00000 | +0.00267 |
| V5 diff_anchor | -0.00375 | +0.00139 | +0.00260 |
| V6 projected_diff | -0.00423 | **+0.00402** | **+0.00294** |
| V7 projected_diff_slow | -0.00279 | +0.00356 | +0.00266 |

This is the most important non-replication at 1M.

### What did transfer?

GOAL/FOCUS learned-reference specificity has begun to emerge, especially in the
projected family:

- V6 has the strongest final learned GOAL advantage;
- V6 also has the strongest final learned FOCUS advantage;
- V7 is second/near-second on both.

### What did not transfer yet?

The small-model signature in which projected_diff/projected_diff_slow learn a
privileged SELF reference for CORE is absent at 1M.

For V6 and V7, learned CORE-reference advantage is negative in every or nearly
every captured layer.

Therefore we should **not** claim that the full SELF+CORE two-anchor structure
has transferred at 1M.

## 6. V2 is the only arm with all three final learned-reference advantages positive

V2/self_v1 ends final with:

- CORE: +0.00036
- GOAL: +0.00211
- FOCUS: +0.00110

However these values are small, CORE specificity is not robustly positive
through depth, and the architecture did not show the strongest two-anchor
behavior in the long small-model study.

This is interesting early-transfer evidence, not enough to promote V2 over the
projected family.

## 7. V6/V7 show the clearest downstream-transfer signature

The projected family currently shows:

### V6 projected_diff
- strongest learned GOAL reference: +0.00402;
- strongest learned FOCUS reference: +0.00294;
- radial-2x effect ~2.69x random;
- complete CORE robustness.

### V7 projected_diff_slow
- learned GOAL reference: +0.00356;
- learned FOCUS reference: +0.00266;
- radial-2x effect ~2.66x random and ~3.10x orthogonal;
- complete CORE robustness;
- anchor norm stayed almost fixed during training despite growing functional
  importance.

This suggests that the projected architectures may first learn a useful
downstream SELF-relative coordinate before a privileged SELF/CORE coordinate
crystallizes.

That is a hypothesis to test with longer continuation, not a settled
developmental law.

## 8. The pretrained model already contains strong internal relations

Raw relation geometry is strongest early in the Goldfish network even in
baseline.

Baseline peaks:

- GOAL relation: layer 1 = 0.12289
- FOCUS relation: layer 1 = 0.19885

Final:

- GOAL relation: 0.00819
- FOCUS relation: -0.04394

The same compression pattern appears across all SELF arms.

This is a much stronger final-output compression than in the 5M model.

Interpretation:

> pretrained Goldfish already carries rich contextual relation geometry in
> early/internal layers, while its final language-model representation heavily
> compresses the particular assay geometry.

This means future Goldfish research must not rely on final hidden state alone.

## 9. Important difference from the 5M training regime

The tiny model learned language and SELF organization jointly from scratch.

Goldfish starts with:

- already-developed Hebrew representations;
- already-developed CORE-like/content geometry;
- already-developed contextual relation geometry.

The new SELF mechanism is entering an existing representational ecosystem.

It is therefore plausible that the transfer order differs:

1. SELF becomes functionally routed;
2. SELF becomes a coherent cross-context direction;
3. downstream relations begin to use it;
4. only later might the learned SELF value become a privileged coordinate for
   CORE.

The current 1M result supports steps 1–3 but not step 4.

## 10. Current scientific assessment

### Strongly supported
- full-parameter transfer is feasible;
- SELF becomes functionally active rapidly;
- SELF direction is magnitude-matched direction-selective;
- SELF effect is highly coherent across contexts;
- CORE remains robust;
- slow-anchor norm/function dissociation transfers;
- projected variants show early downstream learned-reference specificity.

### Not yet replicated
- projected-family learned SELF as privileged CORE reference;
- the full balanced SELF/CORE/GOAL/FOCUS signature seen around 150–200M in the
  5M study;
- architecture-specific SELF/CORE entanglement seen in long tiny-model
  training.

### Still unproven
- semantic SELF;
- direct causal CORE/GOAL/FOCUS roles;
- SELF x CORE -> GOAL computation;
- multi-seed generalization.

## Bottom line

Canary #1 is a successful transfer experiment.

But the correct conclusion is **not**:

> "the 5M result fully replicated."

The correct conclusion is:

> **the SELF mechanism and its directional selectivity transferred strongly to
> a pretrained 124M Hebrew backbone, while the deeper two-anchor
> SELF/CORE organization has not yet crystallized after only 1M continuation
> tokens.**

This creates a clean next scientific question: does longer full-parameter
co-adaptation cause the projected variants to develop the same privileged
SELF/CORE reference structure seen in the small-model study?
