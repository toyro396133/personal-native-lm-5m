# Goldfish 124M seven-arm full-parameter SELF canary #1 — results

Source workflow run: **37704611983**  
Status: **SUCCESS**  
Backbone: `goldfish-models/heb_hebr_1000mb`  
Continuation budget: **1,000,000 tokens per arm**  
All pretrained backbone parameters remained trainable.

## Completion and archival

All seven arms completed successfully, as did the comparison job.

Final model + AdamW states were archived to the private Hugging Face repository:

`toyro967/personal-native-lm-goldfish-self`

under:

`goldfish-124m/V1..V7/1000k/training_state.pt`

with matching `metrics.json` files.

## Blind-safe code mapping used by this transfer

- V1 = baseline
- V2 = self_v1
- V3 = self_v1_slow
- V4 = diff_only
- V5 = diff_anchor
- V6 = projected_diff
- V7 = projected_diff_slow

## Infrastructure result

The experiment answers the first transfer question decisively:

> A pretrained 124.8M Hebrew causal LM can be trained end-to-end on a
> GitHub-hosted CPU runner with the existing SELF variants while keeping every
> backbone parameter unfrozen.

Measured cumulative full-training throughput at 1M:

| arm | tok/s | peak RSS MB |
|---|---:|---:|
| V1 baseline | 110.94 | 2942.6 |
| V2 self_v1 | 109.38 | 3077.4 |
| V3 self_v1_slow | 104.11 | 3113.7 |
| V4 diff_only | **143.95** | 3003.7 |
| V5 diff_anchor | 109.28 | 2981.6 |
| V6 projected_diff | 106.18 | 3037.9 |
| V7 projected_diff_slow | 107.80 | 3099.7 |

The memory headroom is comfortable. CPU time, not RAM, is the primary scaling
constraint.

## Language substrate health

Language loss is not the primary research metric, but no arm catastrophically
damaged the pretrained Hebrew substrate during this 1M-token canary.

Final validation NLL:

| arm | final NLL | change vs its initial |
|---|---:|---:|
| V1 | 4.560488 | -0.001427 |
| V2 | 4.559829 | -0.002086 |
| V3 | 4.560121 | -0.001793 |
| V4 | 4.559850 | -0.002073 |
| V5 | **4.559759** | -0.002156 |
| V6 | 4.560053 | -0.001862 |
| V7 | 4.560109 | -0.001806 |

Differences are tiny and are treated only as health/control evidence.

The validation trajectory is non-monotonic: all arms improve strongly around
250K and then return toward the initial validation level by 1M. This canary is
therefore not a language-optimization study.

## Key SELF transfer result: a dead path becomes functionally active

At initialization, every SELF intervention produced exactly zero effect because
the newly added adapter output path starts at zero.

After only 1M continuation tokens, replacing the learned SELF anchor causes a
clear functional effect in every SELF arm.

Final 1M validation NLL damage relative to normal learned SELF:

| arm | SELF=zero | SELF=negate | random same-norm | orthogonal same-norm |
|---|---:|---:|---:|---:|
| V2 | +0.02313 | +0.02470 | +0.02117 | +0.02226 |
| V3 | +0.02359 | +0.03013 | +0.02123 | +0.02169 |
| V4 | +0.02361 | +0.02549 | +0.02104 | +0.02210 |
| V5 | **+0.02691** | **+0.03313** | **+0.02445** | **+0.02725** |
| V6 | +0.02506 | +0.02934 | +0.02333 | +0.02465 |
| V7 | +0.02456 | +0.03039 | +0.02253 | +0.02564 |

Thus the learned anchor value is already functionally used by the pretrained
backbone after a relatively small continuation budget.

Caveat: these replacement modes do **not** have equal displacement magnitude.
The raw negate-vs-random ordering must not be interpreted as sign-specific
evidence. The magnitude-matched structural follow-up is required for that.

## Representation displacement also grows from zero

Final hidden-state mean L2 displacement at 1M:

| arm | zero | negate | random same-norm |
|---|---:|---:|---:|
| V2 | 2.158 | **3.275** | 2.117 |
| V3 | 2.147 | 2.848 | 2.081 |
| V4 | 2.004 | 2.628 | 1.947 |
| V5 | 2.021 | 2.471 | 1.955 |
| V6 | 1.996 | 2.645 | 1.973 |
| V7 | 1.946 | 2.605 | 1.913 |

This confirms that the SELF path has become embedded in the actual hidden-state
computation, not merely in a bookkeeping parameter.

## Slow-anchor replication on a pretrained backbone

All SELF arms started from the same anchor norm:

`~0.587276`

At 1M:

- V2: 0.64333
- V3 slow: **0.58763**
- V4: 0.66780
- V5: 0.64211
- V6: 0.64796
- V7 slow: **0.58778**

Despite almost no norm growth in V3/V7, their functional intervention effects
grew from exactly zero to levels comparable with the fast variants.

This independently transfers an important small-model conclusion to the
pretrained 124M backbone:

> SELF anchor norm is not a reliable proxy for SELF functional importance.

The system can learn routing/use of a nearly fixed-norm reference.

## Development over 250K -> 500K -> 1M

The SELF intervention effect grows strongly with continuation time in every
architecture.

For example, random same-norm NLL damage:

- V2: +0.00784 -> +0.01013 -> +0.02117
- V5: +0.00769 -> +0.01106 -> +0.02445
- V6: +0.00808 -> +0.01101 -> +0.02333
- V7: +0.00761 -> +0.01044 -> +0.02253

Thus functional SELF integration is already measurably developmental over the
first 1M pretrained-backbone continuation tokens.

## What this canary proves and does not prove

Supported:

1. Full-parameter SELF co-adaptation on the 124.8M pretrained Hebrew backbone
   is operationally feasible on GitHub CPU runners.
2. The added SELF path starts functionally neutral and becomes materially used
   during continuation.
3. The specific learned SELF anchor matters: replacing it with random or
   orthogonal same-norm anchors degrades held-out LM behavior and shifts hidden
   representations.
4. Slow-anchor variants reproduce the dissociation between anchor norm and
   functional importance.
5. No catastrophic language-substrate destruction occurred in 1M tokens.

Not yet supported by canary #1 alone:

- distinct SELF/CORE organization;
- learned-reference specificity for CORE/GOAL/FOCUS;
- magnitude-matched direction selectivity;
- causal SELF x CORE -> GOAL organization;
- direct FOCUS/GOAL causal roles;
- semantic selfhood.

Those require the dedicated layerwise structural assay.

## Immediate follow-up

Run a Goldfish-native version of structural audit v3 on all seven archived 1M
states:

- matched 1x/2x radial, random, orthogonal and shuffle-derived interventions;
- CORE robustness;
- learned SELF vs multiple equal-norm sham anchors;
- GOAL and FOCUS relation geometry;
- every transformer layer;
- no language-loss leaderboard.

This is the decisive scientific analysis of canary #1.
