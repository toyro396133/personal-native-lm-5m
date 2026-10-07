# Research Experiment Ledger

This file is the **source of truth for active experimental reasoning** in this
repository. The README keeps the historical implementation narrative; this
ledger records hypotheses, controls, results, confounds, conclusions and the
next decision.

Last updated: 2026-10-07.

## Research rules

1. Do not call a result a SELF effect merely because a SELF model beats a
   baseline.
2. Separate:
   - training advantage caused by adding SELF;
   - runtime dependence on SELF after training;
   - generic extra-capacity effects;
   - hierarchy/authority effects;
   - joint-training/co-adaptation effects.
3. Every important positive result should eventually have:
   - matched data/order/optimizer/budget;
   - capacity-matched control;
   - multiple seeds when margins are small;
   - causal ablation;
   - rescue/restore when practical;
   - evaluation on more than one text domain.
4. Record negative results and confounds explicitly. They are not erased when a
   later experiment succeeds.
5. Independent arms should run in parallel when scientifically valid.
6. The long-term conceptual target remains:
   **SELF is a stable reference point; CORE, GOAL and FOCUS are things related
   to SELF, not SELF itself.**
7. Key distinction:
   **focus may change without the center changing.**

---

## Historical language/personal-model line

| Version | Question | Result | Decision |
|---|---|---|---|
| v0.6 | Can natural Hebrew and personalization coexist? | unrestricted personalization caused forgetting; protected interface reached 100% longitudinal benchmark with language backbone bit-for-bit unchanged | freeze ordinary language path during later per-user integration |
| v0.7 | Does more natural Hebrew help? | quantitative improvement; generation still poor and byte-fallback artifacts remained | tokenizer became priority |
| v0.8 | Unicode-safe SentencePiece vs hybrid byte fallback | large character-normalized gain; no Unicode corruption; best old external-language checkpoint | Unicode-safe tokenizer adopted |
| v0.9 | 50/50 source balance by oversampling | negative; severe repetition/collapse | do not duplicate small sources to force balance |
| v0.10 | 50/50 using unique data | partial recovery but still below v0.8 | unique diversity matters |
| v0.11 | source-ratio sweep | ~12% Wikipedia arm won sweep but still below v0.8 external benchmark | source mix alone not enough |
| v0.12 | broader FineWeb2 data | broader data harder; old 5M budget undertrained | scale exposure before judging architecture |
| v0.13 | FineWeb-native tokenizer | small improvement but still data/capacity limited | continue controlled scaling |
| v0.14 | 10M model at ~10M exposure | only ~1% gain vs 5M; both undertrained | data exposure is major bottleneck |

Full older details remain in [README.md](README.md).

---

## SELF language line

### v0.15 — 50M SELF A/B

**Question:** Does a stable learned SELF reference help language modeling at
larger exposure?

At 50M, baseline and SELF were effectively tied on language likelihood.
However, post-training causal ablations were strong:

- adapter off worsened FineWeb and external evaluation;
- zero/random/shuffle anchor worsened performance;
- negating the anchor caused a much larger degradation;
- argmax predictions changed materially.

**Conclusion:** the trained network causally depended on a specific learned
internal reference. This is evidence for runtime dependence, **not yet proof
that SELF caused a training advantage**.

Anatomy showed that raw positive scaling and constant shifts had little/no
effect because LayerNorm removes them. Functional SELF behaves primarily like a
normalized pattern/direction.

### v0.16 / v0.16b — SELF v2 architecture screen

Variants:
- self_v1
- self_v1_slow
- diff_only
- diff_anchor
- projected_diff
- projected_diff_slow

At 8M all were effectively tied. At 15M, self_v1_slow led, but the old corpus
was being repeated after ~10M, so that result was not sufficient.

**Decision:** move to fresh disjoint data and much longer exposure.

### v0.17 — 100M seven-arm SELF study — ACTIVE

Run: **37572458350**

Design:
- seven arms in parallel;
- one continuous 100M run per arm;
- four disjoint ~25M shards;
- evaluation every ~2M;
- exact checkpoints at 25/50/75/100M.

#### 25M

Leader: **self_v1_slow**, about **0.145% better** than baseline.
Five of six SELF variants beat baseline; diff_only was worse.

#### 50M

Current ranking:

1. **diff_anchor** — 1.58110761 nats/char, ~0.138% better than baseline
2. **self_v1** — 1.581866, ~0.090% better
3. **projected_diff_slow** — 1.582532, ~0.048% better
4. baseline — 1.58329505
5. self_v1_slow — slightly worse than baseline
6. projected_diff — slightly worse
7. diff_only — clearly worse

**Important:** self_v1_slow lost its 25M lead. diff_anchor was already pulling
ahead around 40M, but the margin remains small.

**Current interpretation:** diff_anchor is the temporary language winner, but
no causal training-advantage claim is allowed until 75/100M plus
capacity-matched/multi-seed controls and post-training interventions.

Phase 3 (50→75M) is active.

---

## CORE / GOAL / FOCUS structural line

### v0.16c — one-step relational side lab

Frozen 15M checkpoints. Relational features produced small gains in some arms,
but the task was too easy and could be solved through shortcuts.

**Conclusion:** not evidence for the intended hierarchy.

### v0.16d — sequential drift lab

Long closed-loop sequences exposed a failure: zeroing/ablating the old
arithmetic relation representation could improve results dramatically.

**Conclusion:** the arithmetic relation encoder was a coordinate
distortion/shortcut, not a validated semantic relation mechanism.

### v0.16e — persistent hierarchical relations

Introduced explicit persistent edges:
- SELF→CORE
- CORE→GOAL
- GOAL→FOCUS

Content path was protected by residual design.

Result: safe but weak. self_v1_slow gained about +1.66 points over its matched
direct arm, but relation gates remained small and edge ablations had only tiny
effects.

**Conclusion:** protection worked; relations were too easy to ignore.

### v0.16f — authority decomposition

Changed the role of relations:

- content proposes **what** an event suggests changing;
- authority decides **whether** that event is allowed to mutate
  CORE/GOAL/FOCUS.

All six SELF variants improved strongly vs their direct arms. projected_diff
reached ~60.37% state exact vs ~39.98% direct.

But zeroing relation state improved results, while shuffling relation state
hurt.

**Conclusion:** Proposal→Authority decomposition is useful. The learned
RelationState was still noisy; the +20-point gain could not be attributed to
SELF.

### v0.16g — clean authority decomposition

Frozen projected_diff 15M checkpoint. Equal controller parameter count across
arms.

Approximate state-exact decomposition:

- direct reference: 39.98%
- authority + event only: 43.87%
- + candidate: 45.46%
- + neutral hierarchy: 49.27%
- + SELF hierarchy: 50.47%
- capacity control: 50.25%
- SELF hierarchy + counterfactual training: 54.27%

Counterfactual training made the SELF hierarchy strongly causally used:
shuffling hierarchy dropped ~54.27% → ~20.81%.

**Conclusion:** authority, candidate information, hierarchy and
counterfactual relational learning all help. SELF alone was only ~0.22 points
above a capacity control in this frozen setup, so SELF-specific benefit was not
established.

### v0.16g2 — 2×2 SELF × counterfactual completion

Run: **37592742981**

Raw cells:

| | no CF | CF |
|---|---:|---:|
| neutral hierarchy | 49.27% | **61.01%** |
| SELF hierarchy | 50.47% | 54.27% |

Raw interaction was negative.

**Critical confound:** the so-called neutral root was
`nn.Parameter(torch.zeros(d_model))` and was included in the optimizer.
Therefore it was **a trainable task-specific root**, not “no SELF”.

Further clue:
- neutral+CF: hierarchy shuffle produced essentially no change;
- SELF+CF: hierarchy shuffle caused a catastrophic drop.

**Conclusion:** g2 does **not** show that SELF hurts. It shows that a
task-trained root can score highly through a solution that appears much less
dependent on the specific hierarchy, while the frozen learned SELF makes the
hierarchy causally important.

This confound must be preserved in the record.

### v0.16h — joint LM + SELF + CGF + Authority, 15M — COMPLETED

Run: **37592167591**

Same 15M token budget and matching initialization within each pair.

#### projected_diff

- SELF-only structural state exact: **51.55%**
- joint structural state exact: **74.10%**
- delta: **+22.55 points**
- final state exact: 40.56% → **71.67%**
- language nats/char: 1.71236 → 1.71440 (~0.12% worse)

#### self_v1_slow

- SELF-only structural state exact: **46.77%**
- joint structural state exact: **84.61%**
- delta: **+37.84 points**
- final state exact: 38.89% → **80.56%**
- language nats/char: 1.71062 → 1.71169 (~0.063% worse)
- CORE false updates: 0
- CORE change recall: **100%**
- GOAL change recall: **100%**

Important weakness:
- GOAL false-update rate increased substantially in joint training
  (self_v1_slow ~29.1%).

**Conclusion:** this is strong evidence that **co-adaptation matters**. It is
not yet proof that SELF specifically caused the structural gain; the LM also
received structural gradients.

Current structural winner: **self_v1_slow + joint**.

### v0.16i — causal anatomy of joint winner — ACTIVE

Workflow: `.github/workflows/v16i-joint-causal-anatomy.yml`

Source checkpoint: v0.16h self_v1_slow + joint, 15M.

Interventions:
- whole SELF: normal / zero / random / shuffle / negate;
- LM SELF adapters off while learned SELF remains the authority root;
- authority SELF root zero while normal LM representations remain;
- hierarchy shuffle;
- all relations zero;
- drop SELF→CORE;
- drop CORE→GOAL;
- drop GOAL→FOCUS;
- language likelihood under SELF perturbations.

**Question:** is the 84.61% joint result causally dependent on SELF, and if so,
through the LM representations, the hierarchy root, or specific relation edges?

---

## Larger-model transfer

### DictaLM-3.0-1.7B full continued-training SELF A/B — SUBMITTED

GitHub orchestrates a private Kaggle GPU kernel:
`selfmodel/dictalm-self-full-canary`.

Pilot design:
- 2M tokens per arm;
- baseline: continued **full** training of all pretrained weights;
- SELF: same full training + diff_anchor SELF;
- matched data/seed/order/budget;
- post-training SELF-on vs SELF-off validation;
- not LoRA.

Purpose: test whether SELF co-adapts usefully inside a model that already has
strong Hebrew language ability.

This is a canary, not a final result. If stable, scale to 15M with stronger
causal controls.

---

## Current decision tree / next steps

1. **Finish v0.16i.**
   - If SELF perturbation strongly hurts while edge interventions are selective:
     treat joint SELF dependence as real and proceed.
   - If SELF perturbation barely matters:
     attribute most v0.16h gain to structural gradients/co-adaptation, not SELF.

2. **Repair GOAL stability.**
   - Increase penalty/weight for false GOAL mutation without sacrificing true
     GOAL-change recall.
   - Keep CORE and FOCUS metrics separately visible.
   - Do not simply optimize aggregate state exact.

3. **Run a matched joint control without semantic SELF.**
   Required before claiming SELF caused the joint-training gain:
   - same parameter count;
   - same structural gradients;
   - trainable sham/capacity reference;
   - ideally 3+ seeds for final comparison.

4. **Continue v0.17 to 75M and 100M.**
   Do not choose the language SELF winner from the 50M point alone.

5. **After 100M:** winner vs baseline vs capacity-matched sham on multiple
   seeds, then SELF zero/random/shuffle/negate, cross-seed SELF swap and
   restore/rescue.

6. **Neutral language evaluation:** maintain FineWeb holdout and add disjoint
   Wikipedia, Knesset and general-web heldouts with macro character-normalized
   metrics.

7. **DictaLM canary:** if 2M full-training pilot is stable, move to 15M before
   making conclusions.

## Current working hypotheses

- A stable learned internal reference is causally used by trained SELF models.
- The best language SELF form may differ from the best structural/cognitive
  SELF form.
- Slow SELF learning may help structural stability even when it is not the best
  pure language-loss choice.
- Proposal→Authority is a useful decomposition.
- Counterfactual relational learning is important for making hierarchy
  causally relevant.
- Joint training/co-adaptation appears much stronger than post-hoc attachment.
- None of the above yet proves that SELF, rather than matched structure/capacity
  plus co-adaptation, causes the large v0.16h structural gain.
