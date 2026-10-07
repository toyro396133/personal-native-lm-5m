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

### v0.17 — 100M seven-arm SELF study — COMPLETED

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

Ranking at 50M:

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

#### 75M

Ranking:

1. **diff_anchor** — **1.55502383 nats/char**
2. **projected_diff_slow** — 1.55617377
3. **self_v1_slow** — 1.55641981
4. **self_v1** — 1.55663523
5. baseline — 1.55795341
6. projected_diff — 1.55876802
7. diff_only — 1.56299276

diff_anchor remains the leader and is about **0.188% better than baseline** at
75M. This strengthens the 50M signal because the same variant leads at two
successive large milestones on fresh disjoint data.

Important: the margin is still small and does not yet establish SELF-specific
causation or capacity independence.

#### 100M — FINAL

Run **37572458350** completed successfully.

Final ranking:
1. **diff_anchor** — **1.5413327422 nats/char** (~0.111% better than baseline)
2. **self_v1_slow** — 1.5421087657 (~0.060% better)
3. **projected_diff_slow** — 1.5428534608 (~0.012% better)
4. baseline — 1.5430392239
5. projected_diff — 1.5430964447
6. self_v1 — 1.5434774596
7. diff_only — 1.5486498172

**Conclusion:** diff_anchor remains the language winner at 50M, 75M and 100M.
The margin is small but persistent across successive fresh-data milestones.
Second place is not stable across the whole run: projected_diff_slow was second
at 75M, while self_v1_slow is second at 100M. This is still not proof of
SELF-specific causation; matched capacity controls and multi-seed replication
remain required.

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

### v0.16i — causal anatomy of joint winner — COMPLETED

Run: **37600125673**

Workflow: `.github/workflows/v16i-joint-causal-anatomy.yml`

Source checkpoint: v0.16h self_v1_slow + joint, 15M.

Normal reference:
- structural state exact: **84.61%**
- language: **1.711694 nats/char**

Whole-SELF interventions:
- SELF=zero: structure -0.23 points; language +0.00137 nats/char worse
- SELF=random: structure -0.32 points; language +0.00171 worse
- SELF=shuffle: structure -0.23 points; language +0.00149 worse
- SELF=negate: structure **-1.88 points**; language **+0.01267 worse**
- LM SELF adapters off: structure -0.31 points; language +0.00120 worse

Hierarchy/root interventions:
- authority SELF root zero: **no aggregate structural change**
- drop SELF→CORE: **no aggregate structural change**
- drop CORE→GOAL: about **-0.10 points**
- hierarchy shuffle: **-5.81 points**
- zero all relations: **-10.49 points**
- drop GOAL→FOCUS: **-10.50 points**

**Conclusion:** the very large v0.16h joint-training gain is **not primarily
explained by runtime dependence on the learned SELF root**. The trained system
does use SELF in the LM path (especially visible under anchor negation and in
language likelihood), but structural accuracy is only weakly sensitive to
zero/random/shuffle SELF interventions.

The joint structural behavior is causally dependent on the hierarchy, with
**GOAL→FOCUS carrying almost all of the measured edge-level effect in this
benchmark**. SELF→CORE is effectively unused by the current task/controller.

Therefore v0.16h is strong evidence for **co-adaptation + structural hierarchy**,
not yet for SELF-specific causation. A matched joint sham/capacity root remains
required before any stronger SELF claim.

### v0.16j — independent CORE / GOAL / FOCUS causal 40M — COMPLETED

Goal: test whether the full rooted hierarchy becomes causally necessary when the
training objective explicitly prevents the shortcut found in v0.16i.

Design:
- 40M language-token exposure on fresh v0.17 shard-1 + shard-2 data;
- five resumable 8M phases;
- structural evaluation/logging every 4M (10 milestones: 4..40M);
- structural backbone: `self_v1_slow`, the v0.16h structural winner;
- two matched joint-training arms:
  - `self`: authority hierarchy rooted in the learned LM SELF anchor;
  - `sham`: same model/authority/parameter budget, but authority hierarchy rooted
    in a separate trainable sham vector;
- balanced structural batches across pressure/change/local/noise events;
- asymmetric penalty against false high-level mutation, especially GOAL drift;
- nested edge-specific counterfactual training so wrong CORE invalidates all lower
  authority, wrong GOAL invalidates GOAL/FOCUS, and wrong FOCUS invalidates FOCUS;
- closed-loop test sequences remain disjoint from training concepts/templates.

Every 4M milestone records:
- state exact, final exact, whole-sequence exact;
- CORE/GOAL/FOCUS state accuracy;
- false-update and true-change recall per level;
- pressure retention and GOAL retention across repeated FOCUS changes;
- causal interventions: root zero/shuffle/negate, whole-hierarchy shuffle,
  drop SELF→CORE, drop CORE→GOAL, drop GOAL→FOCUS;
- language loss and root/SELF diagnostics.

Interpretation rule:
- a high normal score is not sufficient;
- the desired result is selective degradation when each required edge is removed,
  while GOAL false-update improves without sacrificing GOAL-change recall;
- SELF-specific benefit requires the learned SELF-root arm to beat the matched
  sham-root arm, not merely to use a root causally.

#### v0.16j final 40M result

Run **37609514376** completed successfully.

At 40M:
- SELF-root state exact: **89.29%**
- matched sham-root state exact: **91.46%**
- SELF minus sham: **-2.17 points**
- SELF final exact: **86.36%**
- sham final exact: **88.64%**
- SELF whole-sequence exact: **22.27%**
- sham whole-sequence exact: **13.18%**
- GOAL false-update: SELF **9.11%** vs sham **23.21%**
- GOAL change recall: SELF **95.10%** vs sham **97.70%**

Causal edge anatomy at 40M in the SELF-root arm:
- hierarchy shuffle: **-51.21 points**
- drop CORE→GOAL: **-49.97 points**
- drop GOAL→FOCUS: **-35.98 points**
- drop SELF→CORE: **+1.89 points**
- root zero/shuffle/negate: roughly **+1.8 points**, not a degradation

**Conclusion:** the new counterfactual objective successfully made
CORE→GOAL and GOAL→FOCUS strongly causally necessary, and reduced GOAL drift in
the SELF-root arm. However SELF→CORE and the runtime root itself are still not
causally required. The matched sham arm scores higher on aggregate state exact,
so v0.16j does **not** establish SELF-specific structural benefit. It does show
that the middle/lower hierarchy can be made genuinely causal rather than merely
decorative.


---

## Larger-model transfer

### DictaLM-3.0-1.7B full continued-training SELF A/B — OFFLINE RESUBMITTED

Original online canary failed before model/data loading because Kaggle could not resolve
PyPI while `kaggle_entry.py` tried to install runtime dependencies. That failure
remains an infrastructure-only result: **0 training tokens**.

#### Offline infrastructure repair — 2026-10-07

GitHub run: **37610551523**

The canary was rebuilt to be network-independent on Kaggle:

- exact `dicta-il/DictaLM-3.0-1.7B-Base` snapshot downloaded outside Kaggle;
- fixed deterministic FineWeb-2 Hebrew inputs prepared outside Kaggle:
  **2,000,000 train tokens + 200,000 validation tokens**;
- both uploaded as private Kaggle datasets:
  - `selfmodel/dictalm-self-full-model`
  - `selfmodel/dictalm-self-full-data`;
- kernel metadata now has `enable_internet: false`;
- model/tokenizer loading uses local files only;
- requested accelerator: **NvidiaTeslaT4 (T4 ×2)** so the existing FSDP path can
  shard the full 1.7B model;
- experimental design remains unchanged:
  baseline full continued training vs full continued training + `diff_anchor`,
  2M tokens per arm, matched seed/data/order/budget, followed by SELF-on/off
  validation.

The offline resubmission completed successfully from GitHub and Kaggle reported:

`selfmodel/dictalm-self-full-canary = KernelWorkerStatus.ERROR`

A follow-up status collection (GitHub run **37611171847**) confirmed that the
Kaggle worker advanced from QUEUED to RUNNING after the offline inputs were
attached.

A later live status collection (GitHub run **37673261679**) found the kernel in
**ERROR**. The offline model/data upload path succeeded, but execution stopped
before model loading/training because `kaggle_entry.py` expected
`/kaggle/src/run_config.json`, which Kaggle did not place next to the executed
script. Exact exception: `FileNotFoundError: /kaggle/src/run_config.json`.

Therefore this second failure is again infrastructure-only and still represents
**0 training tokens**. The remaining fix is to embed/read the small run config
without relying on a sibling file in `/kaggle/src`; the large offline inputs
themselves are already attached successfully.

**Interpretation guardrail:** this ERROR is infrastructure-only, not an experimental result. Do not infer anything about DictaLM or SELF until actual training/evaluation output exists.

#### Embedded-config retry — 2026-10-07 evening

GitHub submission run: **37673856056**  
Immediate Kaggle status check: **37673908370**

Fix:
- removed runtime dependence on sibling `run_config.json`;
- embedded the unchanged canary settings directly in `kaggle_entry.py`;
- retained the same private offline model/data inputs and the same 2M-per-arm A/B design.

Current Kaggle status after the fix:

`selfmodel/dictalm-self-full-canary = KernelWorkerStatus.RUNNING`

Two independent post-submit checks, GitHub runs **37673908370** and
**37674053262**, both returned RUNNING. The second check occurred after the
immediate startup window, so the prior missing-config crash is definitively
passed. Kaggle does not expose the experiment's live stdout through this
collector while the worker is non-terminal, so an actual optimizer-step count
cannot yet be verified from outside the worker.

This is the first retry that passed the previous missing-config failure. It is
still not an experimental result until training/evaluation output is produced.




GitHub successfully submitted the private Kaggle GPU kernel:
`selfmodel/dictalm-self-full-canary`.

Pilot design:
- 2M tokens per arm;
- baseline: continued **full** training of all pretrained weights;
- SELF: same full training + diff_anchor SELF;
- matched data/seed/order/budget;
- post-training SELF-on vs SELF-off validation;
- not LoRA.

The Kaggle worker entered **ERROR before model/data loading or training**.
Exact failure: DNS resolution to PyPI failed repeatedly while the entrypoint
attempted to `pip install transformers/accelerate/datasets/...`.

Therefore this is **not an experimental result** and says nothing about
DictaLM, SELF, memory capacity or training stability.

Next infrastructure step: make the Kaggle run network-independent where
possible (preinstalled dependencies and/or attach mirrored model/data as
Kaggle inputs), then rerun the same 2M A/B canary unchanged.

---

## v0.16k — dual-reference SELF / CORE causal experiment — RUNNING

Run: **37677347342**

Conceptual correction being tested:

- **SELF** is the stable reference of the model/subject: "from where / as whom is
  this interpreted?"
- **CORE** is the stable reference of the current input/domain/object: "what is
  the central thing being dealt with?"
- SELF is therefore **not a parent of CORE**.
- The intended topology is:

```text
INPUT -> CORE
SELF <-> CORE -> GOAL -> FOCUS
```

Implementation constraints:
- CORE authority has **no SELF input by construction**;
- GOAL authority receives the SELF<->CORE relation plus CORE<->GOAL;
- FOCUS authority receives GOAL<->FOCUS;
- source language model is the v0.17 **diff_anchor 100M winner**;
- two matched arms: learned language SELF root vs trainable sham root;
- 20M new language tokens, in two 10M resumable phases;
- structural evaluation every 2M.

Counterfactual training:
- wrong SELF must leave CORE authority valid but invalidate GOAL/FOCUS;
- wrong CORE reference must leave CORE authority valid but invalidate GOAL/FOCUS;
- wrong GOAL reference must preserve CORE/GOAL but invalidate FOCUS.

Key causal tests:
- zero/shuffle/negate/decoy root;
- drop SELF<->CORE;
- drop CORE<->GOAL;
- drop GOAL<->FOCUS;
- shuffle CORE reference while holding the real CORE state fixed;
- shuffle GOAL reference while holding the real GOAL state fixed.

Success is **selective causality**, not just high score:
- SELF interventions should hurt GOAL more than CORE;
- CORE-reference interventions should hurt GOAL/FOCUS while CORE remains stable;
- GOAL-reference / GOAL<->FOCUS interventions should selectively hurt FOCUS;
- the learned SELF root must be compared against the matched sham root.

---

## v0.18 — seven-arm fresh continuation, 100M -> 300M — RUNNING

Run: **37676909277**

All seven v0.17 arms resume from their exact 100M checkpoints:
- baseline
- self_v1
- self_v1_slow
- diff_only
- diff_anchor
- projected_diff
- projected_diff_slow

Plan:
- **+200M additional training tokens**, ending at **300M cumulative**;
- 20 resumable phases of **10M** each;
- evaluation every **2M** inside every phase;
- checkpoint at every 10M boundary;
- full seven-arm comparisons at 150M, 200M, 250M and 300M.

Freshness / disjointness rule:
- replay the exact FineWeb-2 Hebrew shuffled stream used by v0.17
  (seed 217, shuffle buffer 10,000);
- reconstruct all chunk hashes from the exact **125,041 documents** consumed by
  the original v0.17 data-preparation run;
- reject every old chunk digest from the continuation;
- keep the same deterministic validation holdout exclusion;
- globally deduplicate the new continuation shards too.

This is intentionally stricter than merely changing the random seed: the
200M continuation is designed to contain no exact training chunks already seen
in the original 100M study.

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
