# Unblinding comparison: independent four-component blind analysis

This document was created **after** the independent blind report had been frozen
and committed in:

- blind research commit: `b7b773dfafd1463157587e7e4170983d0bc736b7`
- blind ZIP blob: `7ab26fef4beb690e4bc304357ce2150df6a10471`
- reported ZIP SHA-256: `6de5427a6f585220dfb0f90fde4e483dc09ccb2749cad726672a0a6086b1bbeb`

The blind ZIP is not modified by this document.

## Hidden mapping

### Components
- A = SELF
- B = CORE
- C = GOAL
- D = FOCUS

### Variants
- V1 = baseline
- V2 = self_v1
- V3 = self_v1_slow
- V4 = diff_only
- V5 = diff_anchor
- V6 = projected_diff
- V7 = projected_diff_slow

## Independent rediscoveries

### 1. B/CORE is an input/content anchor that does not require A/SELF
Blind finding:
V1 had zero measurable A effect/axis while B remained highly recoverable and
stable.

Prior project finding:
baseline already had ~0.97-1.00 CORE structure; SELF was not needed to create
CORE.

Status: **strong independent replication**.

### 2. A/SELF has a clean early direction and stronger late effect
Blind finding:
A-axis consistency peaks early (layers 1-2), while A-effect magnitude peaks late
(layers 5-6), with consistency falling by final.

Prior project finding:
early SELF behaves like a reusable global coordinate; later layers amplify it
and make it more context-specific.

Status: **strong independent replication**.

### 3. Layer 2 is a repeated geometric bottleneck
Blind finding:
both B clustering and C-relative-to-B margin drop from layer 1 to 2 and recover
from layer 2 to 3 across active-A checkpoints.

Prior project finding:
the same layer-2 bottleneck was found across all original 35 checkpoints,
including baseline.

Status: **strong independent replication**.

### 4. Internal relational geometry is stronger than final-layer geometry
Blind finding:
C-B geometry recovers in layers 3-4, A amplifies near layer 5, then relational
geometry weakens toward final; only ~65.9% of peak C-B margin survives to final
on average.

Prior project finding:
final representation systematically compresses relational geometry and useful
SELF-relative structure is often stronger internally.

Status: **strong independent replication**.

### 5. Strong A/SELF influence is not the same as a meaningful learned A/SELF reference
Blind finding:
V5 has very large A effect, while learned-A advantage for C is negative at all
six complete checkpoints and all measured layers.

Unblinded:
V5 = diff_anchor.

Prior project finding:
diff_anchor can have a strong SELF channel and strong CORE->GOAL relation while
the specific learned SELF does not provide a privileged GOAL reference.

Status: **strong independent replication**.

### 6. Projected family preserves B/CORE unusually well under A/SELF interventions
Blind finding:
V6/V7 keep B accuracy under A interventions unusually clean.

Unblinded:
V6 = projected_diff
V7 = projected_diff_slow.

Prior project finding:
projected variants had the cleanest SELF/CORE separation and preserved
factorization through depth.

Status: **strong independent replication**.

### 7. diff_only learns a specific A/SELF reference but is more intervention-sensitive
Blind fingerprint:
V4 combines positive learned-A advantage for C with lower B preservation than
V6/V7.

Unblinded:
V4 = diff_only.

Prior project finding:
diff_only had the strongest learned-reference specificity but poorer clean
separation than projected_diff, especially under extreme negation.

Status: **independent replication with an important intervention caveat**.

## Partial / qualified convergence

### C/GOAL
The blind analysis inferred C as a relational dimension represented relative to
B. This is consistent with GOAL being organized relative to CORE.

However it did **not** establish that C is causally constructed from A x B, nor
that A is required for C.

Status: **supports relational GOAL interpretation; does not prove SELF x CORE -> GOAL**.

### D/FOCUS
The blind analysis identified D only as a more difficult, competing or local
condition, with low-to-medium confidence.

Status: **weak recovery of FOCUS role**.

### Two-anchor interpretation
The blind report explicitly considered the possibility that B is a stable anchor
and A a second reference axis, but correctly judged the semantic two-anchor
claim as not established by geometry alone.

Status: **functional support, semantic claim remains unproven**.

### Developmental-stage hypothesis
The blind report agrees that A-direction is already present early while A
effect grows later, but it did not find proof of a universal stage in which the
specific learned A value suddenly acquires meaning.

Status: **partial support; previous two-stage interpretation should remain a hypothesis**.

## New findings from the blind analysis

### 1. Composite-metric non-independence
The blind analysis recovered exact algebraic dependencies:
- B invariance is the mean of zero/negate/shuffle B accuracies;
- B structure is a deterministic weighted combination of component accuracies
  and invariance;
- A-B factorization is a deterministic function of A-axis consistency and B
  invariance.

Consequence:
correlations among these composite metrics are **not independent evidence**.
Future statistical analysis must prioritize primitive measurements.

Status: **important methodological correction**.

### 2. Negate is not a matched intervention
The blind report notes that A-negate usually changes norm/magnitude differently
from A-zero/shuffle. Therefore "sign/polarity sensitivity" cannot be isolated
without norm-matched interventions.

Status: **important methodological correction**.

### 3. Geometry can weaken while decodability remains high
The blind report emphasizes that final geometric margins can shrink while
classification accuracy remains saturated/perfect.

Consequence:
cosine geometry and functional accessibility must be tested separately.

Status: **important conceptual correction**.

### 4. Potential early predictor
Blind exploratory result:
learned-A advantage for B at 50M correlates strongly (Spearman ~0.943, n=6)
with learned-A advantage for C at 120M.

Status: **new but weak/exploratory due small n and multiple-comparison risk**.

### 5. V3/V5 parameter-norm inversion
The blind analysis observed that A-vector norm can decrease in V3/V5 while
functional A effect grows.

Unblinded:
- V3 = self_v1_slow
- V5 = diff_anchor

This extends the earlier conclusion that raw anchor norm is not a reliable proxy
for functional importance.

Status: **new supporting evidence**.

### 6. Robustness is multidimensional
V7 is highly robust to A perturbations yet can still show a marked B degradation
under D at 100M.

Unblinded:
V7 = projected_diff_slow.

Consequence:
"clean SELF/CORE separation" does not imply robustness to all contextual/local
conditions.

Status: **new trade-off / caution**.

## Where the blind analysis does NOT validate the stronger project story

The blind report does not establish:
- semantic selfhood;
- that A is specifically an autobiographical/agentive SELF rather than a generic
  learned reference/control axis;
- that C is causally generated by the relation between A and B;
- that D is definitively FOCUS;
- a proven cognitive hierarchy;
- a universal developmental phase transition;
- causal functional importance of intermediate-layer geometry.

These remain hypotheses requiring targeted intervention tasks.

## Updated confidence after unblinding

### High confidence
1. CORE/B exists robustly without SELF/A.
2. SELF/A behaves as an early coherent reference/control axis whose downstream
   effect is amplified later.
3. Layer 2 is a genuine repeated representational bottleneck in this assay.
4. Relational geometry is often strongest internally and compressed by final.
5. Channel strength and learned-reference specificity are distinct.
6. projected_diff-family preserves SELF/CORE separation better than the other
   active SELF families in these assays.

### Medium confidence
1. GOAL/C is organized relative to CORE/B.
2. projected_diff implements a useful two-reference geometry.
3. slow-reference learning changes developmental path/timing.
4. some early structural metrics predict later learned-reference behavior.

### Low / unproven
1. GOAL is literally computed from SELF x CORE.
2. FOCUS/D has been functionally identified by this assay.
3. intermediate geometry is behaviorally causal.
4. the observed training stages are universal rather than dataset/architecture
   specific.
5. any result demonstrates semantic selfhood.

## Priority follow-up experiments

1. Norm-match SELF/A zero, shuffle, negate, random and interpolation
   interventions.
2. Run primitive-metric statistics; stop counting composite correlations as
   independent evidence.
3. Intervene directly on CORE/B, GOAL/C and FOCUS/D, not only SELF/A.
4. Test whether reading/tapping layers 3-5 improves downstream structural tasks
   over using final.
5. Add checkpoints before 25M to resolve developmental ordering.
6. Repeat key variants across independent seeds.
7. Stress-test learned-reference advantages with multiple prompt banks,
   randomized-reference seeds and bootstrap confidence intervals.
