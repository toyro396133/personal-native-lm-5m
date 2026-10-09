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

#### Kaggle accelerator allocation diagnosis

The embedded-config retry later terminated before training because Kaggle ran it
on CPU: `cuda_available=false`, `gpu_count=0`, despite `enable_gpu=true`.

A T4-specific retry persisted both:
- `enable_gpu: true`
- `machine_shape: NvidiaTeslaT4`

Kaggle still allocated CPU, so this was not a metadata-loss problem. The T4
attempt again produced **0 training tokens**.

The next retry requests a single **NvidiaL4X1** instead. Submission run
**37678247682** succeeded; status checks show the kernel **RUNNING** and Kaggle
persists `machine_shape: NvidiaL4X1`. CUDA allocation has not yet been observed
in live stdout, so this remains infrastructure progress rather than an
experimental result.





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

Initial run: **37676909277**  
Active retry: **37678419157**

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

### v0.18 phase 1 result — 100M -> 110M

Run **37678419157**, phase 1 completed successfully for all seven arms.
Fresh-data evaluations were taken at 102/104/106/108/110M.

Per-milestone winners:
- 102M: **projected_diff** — 1.54008423 nats/char
- 104M: **self_v1** — 1.53982404
- 106M: **diff_anchor** — 1.53788209
- 108M: **diff_anchor** — 1.53915431
- 110M: **self_v1_slow** — 1.53847371

110M ranking:
1. **self_v1_slow** — 1.53847371 (~0.0421% better than baseline)
2. **projected_diff_slow** — 1.53901905 (~0.0067% better)
3. **projected_diff** — 1.53903486 (~0.0057% better)
4. baseline — 1.53912241
5. self_v1 — 1.53931069
6. diff_anchor — 1.53936315
7. diff_only — 1.54296862

Across all five phase-1 milestones (102..110M):
- best mean nats/char: **self_v1** — 1.53988112
- next: **diff_anchor** — 1.53995395
- best mean rank: **projected_diff_slow** — 2.8
- `diff_only` ranked 7th at every milestone.

Interpretation:
- the 100M leader `diff_anchor` remains highly competitive and won 106M/108M,
  but did **not** preserve a monotonic lead into the first fresh 10M continuation;
- `self_v1_slow` won the 110M endpoint;
- rankings fluctuate materially across 2M checkpoints, so 110M alone must not be
  treated as a new winner;
- the continuation therefore strengthens the need to judge by longer-window
  persistence (120M/150M+) and multi-seed controls, not a single endpoint.


### v0.18 infrastructure retry

The first run successfully built and uploaded all 20 fresh shards, but all seven
phase-1 arms failed before training because a literal `\\n` was accidentally
inserted between `RESUME_PATH` and `PREFIX` in `scripts/run_v17_phase.sh`.
No continuation training tokens were consumed in that failed attempt.

The phase-runner was repaired and a new workflow run (**37678419157**) reuses
the already-prepared/verified artifacts from run **37676909277** instead of
regenerating the corpus. Phase 1 now resumes the exact seven 100M checkpoints
toward 110M with evaluations at 102/104/106/108/110M.


---

## SELF / CORE representation-organization assay across saved language checkpoints — COMPLETED

Runs:
- geometry / causal-factorization pass: **37686685487**
- stricter learned-reference pass: **37687136668**

Scope: all **35 preserved seven-arm language checkpoints** currently available:
7 variants × {25M, 50M, 75M, 100M, 110M}.

The assay is frozen/post-hoc: checkpoints are not trained or modified.

### What is measured

1. **CORE structure** — whether the same CORE remains decodable across
   paraphrases and while GOAL/FOCUS content varies.
2. **CORE invariance to SELF** — whether CORE identity survives zero/negate/
   shuffle perturbations of SELF.
3. **SELF-axis consistency** — whether changing SELF causes a reusable hidden
   representation direction that generalizes across unseen COREs.
4. **Two-anchor factorization** — descriptive combination of a reusable SELF
   axis with preserved CORE identity.
5. **Learned SELF reference advantage** — on the *same fixed hidden states*,
   compare coordinates relative to the actual learned SELF against coordinates
   relative to shuffled/random anchors. This is the important control that
   separates learned reference geometry from an effect guaranteed by adapter
   wiring.
6. **GOAL-relative-to-CORE margin** — whether adding the same GOAL creates a
   consistent relation/displacement around different COREs.

### Sanity checks

- baseline has very strong CORE structure (~0.97–1.00) but exactly **0 SELF-axis
  score**, as intended;
- therefore CORE clustering by itself is mostly ordinary semantic organization
  and is **not evidence for SELF**;
- all SELF architectures create a strong causal SELF axis, but the first pass
  showed that this alone can be largely architectural, hence the stricter
  learned-anchor control.

### 110M snapshot

| variant | two-anchor factorization | learned CORE-reference advantage | learned GOAL-reference advantage | raw GOAL-relative-CORE margin |
|---|---:|---:|---:|---:|
| projected_diff | **0.9631** | +0.003970 | **+0.009122** | 0.1089 |
| projected_diff_slow | 0.9573 | **+0.003992** | -0.000654 | 0.1313 |
| diff_anchor | 0.8979 | +0.000937 | -0.011563 | **0.1406** |
| self_v1 | 0.8566 | +0.000857 | -0.004192 | 0.0908 |
| self_v1_slow | 0.8256 | -0.000291 | -0.007172 | 0.1202 |
| diff_only | 0.7469 | +0.003545 | **+0.014898** | 0.1208 |
| baseline | 0.0000 | N/A | N/A | 0.1101 |

### Trajectory conclusions, 25M -> 110M

**projected_diff**
- most balanced structural result;
- two-anchor factorization stays extremely stable (~0.955–0.963);
- learned CORE-reference advantage is positive at every saved point
  (mean **+0.003343**);
- learned GOAL-reference advantage is also positive at every saved point
  (mean **+0.009200**);
- current best candidate for "SELF and CORE are distinct anchors and the learned
  SELF is actually useful as a coordinate reference."

**projected_diff_slow**
- also has very stable high factorization (~0.948–0.962);
- learned CORE-reference advantage grows from -0.000140 at 25M to
  **+0.003992 at 110M**, the best 110M value;
- learned GOAL-reference advantage is mixed/near zero;
- suggests a slowly learned SELF may increasingly become a useful CORE-relative
  reference without yet showing the same GOAL benefit as projected_diff.

**diff_only**
- strongest average learned CORE-reference advantage (**+0.004558**) and
  strongest average learned GOAL-reference advantage (**+0.010784**);
- however two-anchor factorization falls from 0.9454 at 25M to **0.7469 at
  110M** because CORE becomes less invariant to SELF perturbations;
- interpretation: strong SELF-specific reference signal, but it increasingly
  entangles SELF with CORE rather than keeping the two anchors cleanly separate.

**diff_anchor**
- remains the language-loss winner from v0.17 and its raw GOAL-relative-to-CORE
  geometry becomes strong (0.1034 at 25M -> **0.1406 at 110M**, best at 110M);
- but the *specific learned SELF* gives only a tiny positive CORE-reference
  advantage (mean **+0.000680**) and a consistently negative GOAL-reference
  advantage (mean **-0.011025**);
- therefore diff_anchor currently looks like a good language / CORE->GOAL
  relation mechanism, **not the best learned SELF-reference mechanism**.

**self_v1 / self_v1_slow**
- weaker learned-reference specificity overall;
- self_v1 mean learned CORE advantage +0.001210, GOAL advantage -0.004487;
- self_v1_slow mean learned CORE advantage -0.000943, GOAL advantage -0.002060.

### Main scientific conclusion

The earlier suspicion is supported: **best language SELF and best structural
SELF need not be the same variant.**

At the current evidence level:
- language-loss leader: **diff_anchor**;
- best balanced two-anchor SELF/CORE organization: **projected_diff**;
- strongest but increasingly entangled learned-reference signal: **diff_only**;
- strongest emerging 110M CORE-reference specificity with clean factorization:
  **projected_diff_slow**.

The structural target is therefore not "beat baseline LM loss". It is:
- CORE remains stable as the input/object reference;
- SELF contributes a distinct reusable reference axis;
- the *specific learned SELF* is better than random/shuffled anchors;
- GOAL/other dynamic state can be organized relative to the two anchors without
  collapsing SELF and CORE into one representation.

### Deeper cross-checkpoint patterns (35-checkpoint analysis)

A layer-by-layer and language-vs-structure analysis of the same 35 checkpoints
reveals several additional patterns.

**1. SELF strength grows almost universally, but structural quality diverges.**
From 25M to 110M, the mean adapter gate increases monotonically in every SELF
variant:
- diff_anchor: 0.230 -> 0.501
- diff_only: 0.256 -> 0.485
- projected_diff: 0.248 -> 0.475
- projected_diff_slow: 0.245 -> 0.467
- self_v1: 0.241 -> 0.438
- self_v1_slow: 0.238 -> 0.422

The measured hidden-state effect of SELF interventions also increases strongly
for every architecture. Therefore the main problem is **not whether the network
uses the SELF channel more over time**. It does. The differentiator is where
that influence is routed: cleanly around CORE, washed out, or entangled into
CORE itself.

**2. Raw CORE->GOAL relation strength and learned-SELF specificity are distinct
axes.**
At 110M, relative to baseline raw CORE->GOAL geometry:
- diff_anchor: **+0.0305**, but learned-SELF GOAL advantage **-0.0116**;
- projected_diff: **-0.0012**, but learned-SELF GOAL advantage **+0.0091**;
- diff_only: **+0.0107** and learned-SELF GOAL advantage **+0.0149**;
- projected_diff_slow: **+0.0212** and learned-SELF GOAL advantage ~neutral
  (**-0.00065**).

This creates a useful conceptual map:
- **diff_anchor** learns a strong semantic CORE->GOAL organization that is not
  specifically anchored to the learned SELF;
- **projected_diff** learns a SELF-specific coordinate relation even when raw
  CORE->GOAL separation is not stronger than baseline;
- **diff_only** learns both, but increasingly mixes SELF polarity into CORE;
- **projected_diff_slow** learns strong clean CORE relation with delayed/weak
  SELF-specific GOAL geometry.

**3. diff_anchor shows a clear structural transition around 75M.**
Its raw GOAL-relative-to-CORE margin above baseline changes:
25M -0.0040, 50M -0.0048, 75M +0.0168, 100M +0.0215, 110M **+0.0305**.
Thus the language-loss winner appears to develop a strong CORE-centered
relational organization only after roughly 50M-75M. Its learned SELF does not
explain that relation: learned-SELF GOAL advantage is negative at every saved
checkpoint.

**4. projected_diff shows the cleanest "crystallization" of two anchors.**
At 25M its best factorization appears only by layer 4 and some early layers are
SELF-sensitive. From 50M onward:
- the best factorization is already present at **layer 1**;
- CORE invariance is perfect across all measured layers;
- final two-anchor factorization remains ~0.955-0.963;
- learned CORE-reference and GOAL-reference advantages remain positive at every
  saved checkpoint.

This looks like training moves the two-anchor coordinate system earlier in the
network rather than merely making it stronger at the output.

**5. projected_diff preserves structure through depth unusually well.**
Mean final/best-layer factorization ratio across checkpoints:
- projected_diff: **0.973**
- projected_diff_slow: **0.965**
- self_v1_slow: 0.964 (but collapses at 110M)
- diff_anchor: 0.911
- self_v1: 0.878
- diff_only: 0.864

The projected architectures therefore do not merely create a good intermediate
SELF/CORE geometry; they preserve most of it all the way to the final
representation.

**6. diff_only undergoes the opposite transition around 75M.**
At 25M/50M any CORE sensitivity to SELF is recoverable by the final layer. From
75M onward it persists to the output and worsens. Final CORE accuracy under
SELF negation is:
1.00 (25M), 1.00 (50M), 0.667 (75M), 0.667 (100M), **0.50 (110M)**.
Yet CORE accuracy under SELF **zeroing or shuffling remains 1.00 at every
checkpoint**. Thus the apparent entanglement is specifically **polarity/sign
sensitive**, not generic dependence on any precise SELF value. Negation is a
much harsher out-of-distribution intervention than zero/shuffle, so this is a
warning signal rather than proof that ordinary CORE content is corrupted.

**7. self_v1 variants contain a SELF channel but weak learned SELF identity.**
Both self_v1 and self_v1_slow have high causal SELF-axis consistency, but the
specific learned anchor is not consistently better than random/shuffled anchors
for organizing CORE/GOAL. In self_v1, early-layer GOAL-reference advantages are
sometimes positive but tend to become negative by the final representation.
This separates "architecture responds to an anchor" from "this learned anchor
has acquired a privileged reference role."

**8. Slow learning has architecture-dependent meaning.**
For projected_diff, the slow variant keeps the same clean CORE separation while
its learned CORE-reference advantage grows from -0.00014 at 25M to **+0.00399
at 110M**, essentially catching the fast variant (+0.00397). However its
SELF-specific GOAL advantage remains near zero. It also has better language loss
than fast projected_diff at every saved endpoint.
For self_v1, slow learning improves factorization over the fast variant through
50M-100M, but does not create stronger learned-anchor specificity and shows a
sharp 110M deterioration. "Slow SELF" is therefore not a universal benefit;
the projection architecture is what makes it useful.

**9. There is exploratory evidence of a language-vs-SELF-specificity tradeoff.**
After centering within each token checkpoint (so 25M is compared with 25M, etc.),
higher/worse LM nats correlate with stronger learned-reference specificity
across the 30 SELF checkpoints:
- learned CORE-reference advantage: Spearman rho **+0.63**
- learned GOAL-reference advantage: Spearman rho **+0.61**

Across the six architecture-level means the same relationship is even stronger
(rho ~+0.89 for CORE and +0.83 for GOAL), but n=6 and architectures are not
independent random samples, so this is exploratory rather than inferential
proof. It nevertheless supports treating LM loss as a health metric rather
than the primary SELF objective.

Importantly, this tradeoff appears weaker at 110M: projected_diff is now slightly
better than baseline LM loss while retaining the strongest balanced
SELF-specific geometry. Therefore the tradeoff may be transient rather than
fundamental.

**10. Raw anchor norm is not a useful proxy for SELF importance.**
At 110M projected_diff_slow has a much smaller anchor norm (~0.296) than
projected_diff (~0.721), yet its measured SELF intervention effect is actually
larger (~0.435 vs ~0.393) and factorization is nearly as strong. This is
consistent with LayerNorm/directional geometry: SELF is functioning primarily
as a learned direction/pattern, not through raw vector magnitude.

**New architectural hypothesis.**
The 35-checkpoint map suggests that the desirable system may need two distinct
functions that current variants split apart:
- projected_diff-like machinery for a clean learned **SELF reference axis**;
- diff_anchor-like machinery for strong **CORE->GOAL relational organization**.

A targeted hybrid should only be tested after the dual-reference causal assay
and multi-seed validation, to avoid architecture fishing.

### Additional recurring patterns from all 35 checkpoints

**11. Structural "fingerprints" are much more stable than LM rankings.**
Adjacent-checkpoint Spearman rank stability:
- LM rank: mean ~**0.64**
- two-anchor factorization: ~**0.89**
- SELF-axis consistency: ~**0.93**
- learned CORE-reference advantage: ~**0.83**
- learned GOAL-reference advantage: ~**0.87**

Thus architecture-specific SELF/CORE behavior is mostly visible early and persists,
while ordinary language ranking fluctuates much more. This means a 25M-50M
structural assay can be predictive of later structural behavior even when it is
not predictive of the final LM-loss winner.

**12. The network has a recurring "organize early, amplify late" pattern.**
Across the 30 SELF checkpoints:
- best two-anchor factorization is in layer 1/2/3 in **28/30** cases;
- maximal SELF representation effect is in layer 5/6 in **28/30** cases;
- layer 5 alone contains the maximum SELF effect in **23/30** cases.

This suggests two separate functions:
1. early/middle layers establish the SELF/CORE coordinate system;
2. late layers amplify the SELF signal for task/language use.

The strongest SELF magnitude is therefore usually *not* where the cleanest
SELF/CORE organization is formed.

**13. Learned-GOAL specificity has an extremely persistent architecture sign.**
Across all 7 measured representations × 5 checkpoints = 35 layer/checkpoint
points per SELF variant:
- **projected_diff:** learned SELF improves GOAL-relative geometry in **35/35**
  points;
- **diff_anchor:** learned SELF worsens it in **35/35** points;
- **diff_only:** positive in **33/35**;
- projected_diff_slow: positive in 21/35;
- self_v1: positive in 20/35;
- self_v1_slow: positive in only 5/35.

This is much stronger evidence than a single endpoint. It means the
projected_diff vs diff_anchor distinction is architectural and persistent,
not an accident of 110M.

**14. The same family separation appears for learned CORE-reference specificity.**
Positive learned CORE-reference advantage:
- diff_only: **35/35**
- projected_diff: **32/35**
- projected_diff_slow: 27/35
- diff_anchor: 23/35
- self_v1: 11/35
- self_v1_slow: only **2/35**

So the old v1 parameterization can react strongly to an anchor without making
the *specific learned anchor* privileged. Difference-based architectures are
much more likely to turn SELF into an actual coordinate reference.

**15. Plain subtraction vs projected subtraction exposes a recurring tradeoff.**
- `diff_only`: strongest learned-reference specificity, but increasing
  polarity sensitivity of CORE;
- `projected_diff`: slightly weaker reference advantage, but clean CORE
  invariance and positive learned GOAL advantage everywhere.

The projection therefore behaves like a **disentangling buffer**: it sacrifices
some raw SELF specificity to keep SELF and CORE as separate axes.

**16. Final CORE is not generically dependent on SELF; the vulnerability is almost
entirely SELF negation.**
Across all 30 SELF checkpoints:
- SELF zeroing damages final CORE classification in **0/30**;
- SELF shuffling damages final CORE classification in **0/30**;
- SELF negation damages it in **9/30**.

Therefore the earlier "CORE entanglement" diagnosis (especially for diff_only)
must be stated more narrowly: it is mainly **sign/polarity sensitivity to an
extreme out-of-distribution intervention**, not ordinary dependence on the
exact SELF vector. diff_only is still less clean than projected_diff, but CORE
does not collapse when SELF is merely absent or randomized.

**17. There is a shared data-driven relational rhythm across architectures.**
Final GOAL-relative-to-CORE margin:
- 50M -> 75M improves in **5/6** SELF variants;
- 75M -> 100M drops in **5/6**;
- 100M -> 110M rebounds in **6/6**.

Learned SELF->GOAL advantage shows a similar common pulse:
- 50M -> 75M improves in **6/6**;
- 75M -> 100M declines in **5/6**.

Because different architectures move together, part of the structural evolution
is caused by the training-shard/content distribution, not the SELF architecture.
`diff_anchor` is unusual in resisting the raw 75M->100M GOAL-relation drop,
which suggests greater CORE->GOAL semantic robustness.

**18. SELF-channel amplitude is saturating around 100M, while organization can
continue changing.**
Mean gate growth across the six SELF variants:
- 25->50M: +0.0957
- 50->75M: +0.0667
- 75->100M: +0.0455
- 100->110M: only +0.0139

Mean measured SELF effect growth similarly falls:
- +0.1117
- +0.0749
- +0.0528
- +0.0018

Thus by ~100M the model is no longer mainly learning "how much SELF to use".
The amplitude is near saturation; later training increasingly tests **how SELF is
organized and routed**. This makes the 100M->300M continuation especially useful.

**19. Slow-anchor training can produce a *stronger* functional SELF despite a much
smaller anchor vector.**
For projected_diff_slow, measured SELF effect is larger than fast
projected_diff at **all 5 saved checkpoints**, while its raw anchor norm is less
than half as large. The slow variant also beats the fast projected variant in LM
loss at **all 5 endpoints**.

The adapter gates and adapter weight norms are not larger in the slow model,
so the stronger effect cannot be explained by simple parameter magnitude.
A plausible hypothesis is timescale separation: a slowly moving anchor gives
the rest of the network a more stable coordinate system to align around.
This remains a hypothesis until anchor-direction drift is measured directly.

**20. SELF-specific information often exists internally and is later suppressed
by the LM output path.**
The layer where learned GOAL-reference advantage is maximal is the final
representation in only 9/30 SELF checkpoints.

Examples:
- self_v1 frequently has positive SELF-specific GOAL geometry in early/middle
  layers but negative geometry at the final representation;
- projected_diff remains positive at every layer/checkpoint, although its
  strongest effect is often internal;
- diff_only changes with training: from 75M onward its strongest learned-GOAL
  specificity survives all the way to the final representation.

This suggests another architectural objective: **preserve useful SELF-relative
structure through the final LM layers instead of merely creating it internally.**

**21. CORE semantics already exist without SELF.**
Baseline CORE-structure accuracy is ~0.97-1.00 throughout training and no SELF
variant consistently improves it. This supports the revised philosophy:
CORE should be input/content anchored; SELF's role is not to create CORE, but
to provide an independent coordinate reference around which relations such as
GOAL can be organized.

### Further recurring patterns: depth cascade, shortcut pressure, and two-stage SELF learning

**22. A remarkably consistent depth cascade appears across the 35 checkpoints.**
The representation geometry follows a repeated sequence:
- **layer 2 semantic bottleneck:** CORE margin and GOAL-relative-to-CORE margin
  are lower at layer 2 than at both layers 1 and 3 in **35/35 checkpoints**,
  including baseline;
- **layer 3 learned-CORE reference alignment:** learned SELF->CORE reference
  advantage is a local maximum at layer 3 in **25/30 SELF checkpoints**;
- **layer 4 GOAL relation formation:** raw GOAL-relative-to-CORE margin is a
  local maximum at layer 4 in **16/30**, and learned SELF->GOAL advantage is a
  local maximum there in **14/30**;
- **layer 5 SELF amplification:** measured SELF intervention effect is a local
  maximum at layer 5 in **25/30 SELF checkpoints**.

This is not proof of a hard-coded hierarchy, but it strongly suggests an
emergent processing rhythm:
`content -> compression/transform -> reference alignment -> relation building -> amplification`.

**23. The final LM representation systematically compresses relational geometry.**
Compared with layer 4:
- final GOAL-relative-to-CORE margin is lower in **35/35** checkpoints;
- final CORE margin is lower in **34/35** checkpoints;
- the best two-anchor factorization is never later than layer 4 and is already
  in layers 1-3 in **33/35** checkpoints.

The compression gap grows with training. Mean layer4-minus-final GOAL margin:
- 25M: +0.036
- 50M: +0.041
- 75M: +0.042
- 100M: +0.049
- 110M: **+0.062**

Thus longer LM training increasingly hides useful relational organization
inside the network rather than preserving it in the final token-prediction
representation. Structural controllers/readouts should therefore inspect or
tap intermediate layers, not assume the final layer is the best state space.

**24. SELF becomes stronger with depth while becoming less globally uniform.**
Within a fixed checkpoint, SELF-effect magnitude and SELF-axis consistency have
a negative Spearman relationship in **26/30** SELF checkpoints (mean rho about
**-0.48**).

Interpretation: early SELF behaves more like a clean global coordinate axis;
later layers amplify it but make its action more context/task-specific. This
fits the depth cascade: "reference first, use/amplify later."

**25. Adapter gate size is not a reliable map of where SELF is functionally used.**
Across training, larger gates correlate with larger SELF effects because both
grow over time. But *within a fixed checkpoint across layers*, gate size and
measured SELF effect have essentially no relationship (mean Spearman rho
~**-0.04**). The layer with the largest gate is also the layer with the largest
SELF effect in only **12/30** checkpoints; for diff_anchor and
projected_diff_slow this happens in **0/5** checkpoints each.

Therefore the ordinary transformer path strongly propagates/amplifies SELF:
local adapter gate magnitude must not be interpreted as local causal importance.

**26. Relative encoding is far more likely than v1 concatenation to make the
specific learned SELF a true coordinate reference.**
Across all layer/checkpoint observations:
- v1 family (self_v1 + self_v1_slow): learned SELF improves CORE-reference
  geometry in only **13/70 (~19%)** and GOAL-reference geometry in
  **25/70 (~36%)**;
- difference-based family (diff_only, diff_anchor, projected variants):
  CORE-reference improvement in **117/140 (~84%)**, GOAL-reference improvement
  in **89/140 (~64%)**;
- projected-difference family specifically: GOAL-reference improvement in
  **56/70 (80%)**.

Architectural hypothesis: v1 exposes raw `x` alongside `SELF`, `x-SELF`,
and products, so the LM can solve its objective through a direct-content
shortcut without privileging SELF. Difference-based variants force the
representation to be expressed relative to the reference. This supports
designing SELF as a coordinate transformation, not merely as extra features.

**27. Projection looks increasingly like a disentangling scaffold rather than
permanent extra capacity.**
From 25M to 110M:
- projected_diff anchor-projection up-weight norm shrinks **~59%** and down
  norm ~20%;
- projected_diff_slow up norm shrinks **~38%** and down norm ~14%.

Yet projected_diff retains/improves its two-anchor structure and learned
reference specificity. In the slow variant, learned CORE-reference advantage
actually grows while projection norms shrink.

A plausible interpretation is that the projection helps establish a favorable
training trajectory / coordinate system early, after which the rest of the
network internalizes much of that organization. Small projected residuals can
still matter directionally, so this is a scaffold hypothesis, not evidence that
the projection has become irrelevant.

**28. The "shape" of SELF is determined early; its learned identity matures later.**
Across the six SELF variants, rank correlation between 25M and 110M:
- SELF-axis consistency: Spearman rho **~0.93**;
- two-anchor factorization: ~0.68;
- measured SELF-effect magnitude: only ~0.07;
- learned CORE-reference advantage: only ~0.14;
- learned GOAL-reference advantage: ~0.43.

Thus by 25M the architecture has largely determined *how coherently SELF acts as
a channel*, but not how strong it will become nor whether the *specific learned
SELF vector* will acquire privileged semantic/reference meaning. This suggests
a two-stage developmental picture:
1. architecture establishes the coordinate mechanism early;
2. continued data teaches the coordinate what it means and how to use it.

**29. Slow and fast variants show transient divergence followed by partial
reconvergence.**
Using a joint structural-distance measure over final representation metrics:
- projected_diff vs projected_diff_slow: distance ~0.052 at 25M, peaks ~0.115
  at 75M, then returns to ~0.049 at 110M;
- self_v1 vs self_v1_slow: ~0.029 at 25M, grows to ~0.184 at 100M, then falls
  to ~0.083 at 110M.

This suggests anchor learning rate may often alter the *timing/path* of
organization more than define a wholly different eventual basin. The common
110M shard may contribute to reconvergence, so longer 120M-300M trajectories
are needed before treating this as an attractor claim.

### Guardrails / next validation

These are deterministic representation-geometry assays, not proof of semantic
selfhood. The learned-reference advantages are numerically small and should be
stress-tested with:
- multiple prompt sets / paraphrases;
- multiple random-anchor seeds;
- bootstrap confidence intervals;
- later 120M+ checkpoints from v0.18;
- the explicit dual-reference v0.16k causal task, where SELF and CORE roles are
  controlled by construction.

---

## v0.18 continuation status — 120M complete, 130M partial

Run: **37678419157**

### Phase 2 complete: 110M -> 120M

All seven arms completed successfully. Fresh-data winners:
- 112M: **diff_anchor** — 1.537544
- 114M: **diff_anchor** — 1.535211
- 116M: **diff_anchor** — 1.536221
- 118M: **projected_diff_slow** — 1.534339
- 120M: **projected_diff_slow** — 1.532278

120M ranking:
1. **projected_diff_slow** — 1.532278
2. diff_anchor — 1.532995
3. self_v1_slow — 1.533925
4. projected_diff — 1.533962
5. baseline — 1.534498
6. self_v1 — 1.535511
7. diff_only — 1.537047

Across 112..120M:
- best mean rank: **diff_anchor 1.6**
- second: **projected_diff_slow 1.8**
- wins: diff_anchor 3/5; projected_diff_slow 2/5
- diff_only remained 7th at all five checkpoints.

110M -> 120M nats/char improvement:
- projected_diff_slow: **0.006741**
- diff_anchor: 0.006368
- diff_only: 0.005922
- projected_diff: 0.005073
- baseline: 0.004624
- self_v1_slow: 0.004549
- self_v1: 0.003799

### Phase 3 current status: 120M -> 130M

Completed successfully:
- diff_anchor
- projected_diff_slow
- projected_diff
- self_v1

Still training at the current status check:
- baseline
- self_v1_slow
- diff_only

Among the four completed arms, provisional winners are:
- 122M: **diff_anchor** — 1.532343
- 124M: **projected_diff_slow** — 1.531995
- 126M: **projected_diff_slow** — 1.529796
- 128M: **diff_anchor** — 1.530120
- 130M: **diff_anchor** — 1.528674

These are provisional rankings until all seven phase-3 arms finish.

---

## Blind cross-session pattern dataset — 47 checkpoints

Workflow run: **37693109863**  
Artifact: **blind-self-core-pattern-dataset-47**  
Artifact id: **11514177355**

Purpose: provide a neutral dataset for an independent session to search for
patterns without inheriting the conclusions in this log.

Coverage:
- complete seven-arm checkpoints at 25M, 50M, 75M, 100M, 110M, 120M = **42**
- partial 130M coverage for:
  self_v1, self_v1_slow, diff_anchor, projected_diff, projected_diff_slow = **5**
- total = **47 checkpoints**
- baseline and diff_only are still absent at 130M at the time the dataset was
  frozen; 130M is explicitly marked partial and must not be used for a full
  seven-arm ranking.

The dataset contains:
- one normalized row per checkpoint;
- full per-layer assay reports;
- learned-reference metrics;
- two-anchor factorization metrics;
- CORE invariance / SELF perturbation metrics;
- SELF diagnostics;
- explicit coverage and caution metadata.

The independent-analysis prompt should instruct the other session **not to read
this EXPERIMENT_LOG.md before producing its first-pass findings**.

---

## Independent blind analysis — first cross-check findings

A separate account analyzed the anonymized 47-checkpoint four-component dataset
without unblinding A/B/C/D or V1-V7.

### Blind role inference vs actual hidden mapping

The blind analysis inferred:
- B = a stable identity/content anchor;
- A = an early coherent representational control/reference axis whose effect is
  amplified later;
- C = a relational dimension represented relative to B;
- D = a more local/challenging/competing condition, with lower confidence.

Hidden mapping used to construct the blind dataset:
- A = SELF
- B = CORE
- C = GOAL
- D = FOCUS

This is a meaningful independent convergence with the working SELF/CORE
interpretation, especially for A/B/C. D/FOCUS is less strongly recovered and
should remain lower-confidence.

### Independent recurring depth pattern

The blind analysis independently recovered:
- early A direction;
- layer-2 geometric bottleneck;
- recovery of B/C relational geometry at layers 3-4;
- A-effect amplification near layer 5;
- loss/compression of relational geometry toward final.

This closely matches the prior in-session layer-cascade analysis.

### Important methodological correction: several composite metrics are not independent evidence

The blind analysis discovered exact algebraic dependencies in all 329 layer
records:
- B invariance is the mean of zero/negate/shuffle intervention accuracies;
- B structure is a deterministic weighted combination of its component
  accuracies/invariance;
- A-B factorization is a deterministic function of A-axis consistency and B
  invariance.

Therefore correlations among structure, invariance, and factorization must not
be treated as multiple independent pieces of evidence. Future reports should
decompose them into primitive measurements before statistical interpretation.

### Other independent confirmations

- One variant (hidden V5 = diff_anchor) can have very large A/SELF intervention
  effect while learned-A/SELF reference advantage for C/GOAL remains negative.
  This independently supports the distinction between "SELF channel is used"
  and "the learned SELF value is a privileged reference".
- Hidden V6/V7 (= projected_diff / projected_diff_slow) preserve B/CORE under A
  interventions unusually well, independently recovering the projected-family
  disentanglement pattern.
- The blind analysis again finds that final-layer geometry is often weaker than
  useful intermediate-layer geometry.
- It explicitly warns that negate interventions are not magnitude-matched to
  zero/shuffle; sign-sensitivity claims therefore need matched-norm controls.

### New high-priority validation tasks from the blind report

1. Replace composite-metric correlations with primitive-metric analysis.
2. Add norm-matched A/SELF interventions to distinguish sign from magnitude.
3. Add earlier-than-25M checkpoints to test developmental ordering.
4. Add multi-seed replication.
5. Expand interventions to the other three components, not only A/SELF.
6. Directly test whether intermediate-layer geometry is functionally useful
   rather than merely linearly decodable.

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


---

## Structural checkpoint audit v3 at 150M — COMPLETED

Run: **37706313613**  
Results documentation: `research/structural-audit-150m/RESULTS.md`

All seven 150M checkpoints were audited in parallel with **language loss excluded
from the primary ranking**. The assay used magnitude-matched SELF interventions,
multiple equal-norm sham anchors, layerwise CORE/GOAL/FOCUS geometry and
separate measurements for SELF effect strength vs learned-value specificity.

Main conclusions:

- CORE remains strong without SELF; baseline is saturated on final CORE
  classification.
- magnitude-matched radial SELF perturbations produce much larger hidden-state
  effects than same-displacement random/orthogonal perturbations, so direction
  matters beyond magnitude at 150M;
- projected_diff and projected_diff_slow preserve CORE best under strong SELF
  perturbations;
- projected_diff has the strongest final learned CORE-reference advantage;
- projected_diff_slow is the only variant with positive final learned-reference
  advantage for CORE, GOAL and FOCUS simultaneously and is the best balanced
  structural candidate at 150M;
- diff_anchor has a very strong SELF effect but learned GOAL/FOCUS specificity
  remains negative, reinforcing that channel strength != correct reference
  semantics;
- self_v1_slow is highly fragile to radial-2x/negated SELF despite random and
  orthogonal 2x perturbations preserving CORE;
- FOCUS challenges CORE more than GOAL, extending the independent blind
  follow-up result;
- relational geometry is generally richer at internal layers than at the final
  LM representation.

No semantic SELF, direct CORE/GOAL/FOCUS causality or universal developmental
stage is claimed from this audit. Next priority is to repeat the same primitive
audit at later milestones and add direct activation-patching/component
interventions.


---

## Structural checkpoint audit v3 at 200M — COMPLETED

Run: **37720011983**  
Results: `research/structural-audit-200m/RESULTS.md`

All seven 200M checkpoints completed the same magnitude-matched structural audit
used at 150M and 170M.

Main conclusions:

- projected_diff_slow is the strongest balanced structural candidate at 200M:
  strongest final GOAL relation (0.15496), strongest final FOCUS relation
  (0.21136), positive learned-reference advantage for CORE/GOAL/FOCUS, and
  perfect final CORE preservation under radial-2x SELF stress;
- projected_diff retains the strongest learned CORE-reference advantage
  (+0.01046) and strongest learned FOCUS-reference advantage (+0.00731), with
  learned GOAL specificity concentrated in middle layers rather than final;
- diff_anchor remains a strong negative control: same-magnitude random and
  orthogonal perturbations preserve CORE, while radial-2x collapses final CORE
  accuracy to 0.167 and learned GOAL/FOCUS reference polarity remains negative;
- diff_only recovers final CORE robustness but loses GOAL relational quality;
- self_v1 remains a powerful SELF-sensitive direction with weak/negative
  learned downstream-reference semantics;
- self_v1_slow recovers final CORE robustness but still shows severe
  intermediate-layer entanglement followed by late repair;
- 150M -> 170M -> 200M trajectories are non-monotonic, strengthening the
  multiple-developmental-clocks interpretation.

Language loss remains a health/control metric and is deliberately not used as
the primary ranking in this audit.

The next decisive checkpoint for this continuation is 300M. A final seven-arm
v3 structural audit is configured to run automatically when the long v0.18
continuation workflow completes.


---

## Goldfish 124M seven-arm full-parameter SELF canary #1 — COMPLETED

Source training run: **37704611983**  
Structural transfer audit: **37723626343**  
Canary results: `research/goldfish-self-7arm/CANARY1_RESULTS.md`  
Structural results: `research/goldfish-self-7arm/STRUCTURAL_AUDIT_1M_RESULTS.md`

All seven full-parameter arms completed successfully on the pretrained Hebrew
Goldfish 124M backbone, and all final model+AdamW states were archived to the
private Hugging Face repository.

Main transfer conclusions:

- the added SELF path starts functionally neutral and becomes materially used
  after only 1M continuation tokens;
- magnitude-matched radial SELF perturbations produce substantially larger
  hidden-state effects than random/orthogonal perturbations of the same
  displacement, demonstrating that the learned SELF direction is functionally
  special rather than merely large;
- SELF perturbation effects are highly coherent across contexts;
- CORE remains perfectly robust under all tested SELF interventions across all
  captured layers at 1M;
- slow-anchor variants reproduce the dissociation between anchor norm and
  functional importance;
- projected_diff/projected_diff_slow already show the clearest learned
  GOAL/FOCUS reference specificity, but their learned CORE-reference advantage
  is still negative at 1M;
- therefore the transfer is real but incomplete: functional SELF integration
  transfers before the full two-anchor SELF/CORE organization crystallizes.

The correct scientific reading is not that the 5M structural result fully
replicated. The stronger supported statement is that a pretrained Hebrew
backbone can rapidly integrate a functionally special SELF direction while
preserving existing CORE structure, with early downstream-reference signals
appearing before a privileged learned SELF/CORE coordinate is established.

Next scientific gate: longer full-parameter co-adaptation, followed by the same
magnitude-matched structural audit, to test whether the projected family later
develops the small-model learned CORE-reference signature.


---

## v0.18 SELF continuation 100M -> 300M — TRAINING COMPLETE

Source run: **37678419157**  
Display title: `Resume 300M continuation from preserved fresh shards` (#2)  
Final structural audit: **37757999981**

All seven phase-20 training jobs reached **300,000,768 tokens** successfully and
all seven 300M checkpoints were uploaded.

The source workflow's overall conclusion is `failure` only because the legacy
`compare300` job used the old pinned comparison script. This is an
infrastructure/reporting failure, not a training failure.

The automatically triggered 300M structural audit completed successfully for
all seven variants plus comparison.

Results:
- `research/structural-audit-300m/RESULTS.md`
- `research/structural-audit-trajectory/150-170-200-300M.md`

Final high-level conclusions:

- projected_diff preserves the strongest learned SELF-as-CORE reference
  (+0.00953) and positive learned FOCUS reference (+0.00881);
- projected_diff_slow retains the strongest balanced long-run separation:
  final CORE survives radial-2x at every audited milestone 150/170/200/300M;
- diff_only develops severe late SELF/CORE entanglement by 300M despite having
  recovered at 200M, proving the trajectory is non-monotonic;
- diff_anchor remains the clearest negative control: very strong SELF
  influence, wrong downstream learned-reference polarity, and strong CORE
  dependence on radial SELF perturbation;
- internal layers contain richer and sometimes opposite-polarity GOAL
  organization than the final representation;
- the long run supports a dual-anchor working model, SELF <-> CORE, with
  downstream GOAL/FOCUS organization dynamically redistributed across depth
  and training time rather than a fixed serial hierarchy.

Language health at 300M remains near baseline for all variants; loss is retained
as a health/control measure only.

The 5M long-token screening line is now complete. Highest-value next work is
causal activation patching/direct component intervention, multi-seed
replication, and comparison with the ongoing pretrained Goldfish staged
trajectory.


---

## 300M causal activation patching v1 — COMPLETED

Workflow run: **37916251165**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION.md`  
Results: `research/causal-patching-300m/RESULTS.md`

Five 300M checkpoints were tested: baseline, diff_anchor, diff_only,
projected_diff and projected_diff_slow.

Main causal findings:

- whole-residual CORE/GOAL/FOCUS patching switches the corresponding component
  almost perfectly even in baseline, so this broad intervention is treated as a
  positive-control assay rather than SELF-specific evidence;
- projected_diff_slow layer 2 is the strongest selective SELF result:
  radial-2x SELF produces a 7.63x larger final representation effect than a
  random matched-displacement anchor perturbation, while CORE remains 1.000 and
  GOAL/FOCUS fall to 0.681/0.764;
- projected_diff layer 2 similarly gives 4.77x radial/random effect with CORE
  remaining 1.000 while GOAL/FOCUS change;
- diff_anchor reveals a sharply localized causal entanglement bottleneck at
  layer 3: radial SELF drops CORE/GOAL/FOCUS to 0.375/0.639/0.361, while the
  random same-displacement control leaves all three at 1.000;
- diff_only keeps CORE at 1.000 under every single-layer intervention tested,
  despite all-layer radial-2x CORE collapsing to 0.333 in the prior structural
  audit. This suggests distributed/cumulative SELF-CORE entanglement rather
  than a single bottleneck.

This upgrades the projected-family claim from correlational geometry to
direction-specific causal sensitivity: changing SELF at layer 2 selectively
changes downstream GOAL/FOCUS while preserving CORE.

It does not establish semantic selfhood.

Next assay: isolate the SELF adapter contribution itself
`adapter(x, SELF)-x`, using targeted ablation/source-delta patches and
matched-norm random controls, with priority on projected_diff_slow L2,
projected_diff L2, diff_anchor L3, and distributed diff_only controls.


---

## 300M SELF adapter contribution causal patching v2 — COMPLETED

Workflow run: **37916891722**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION_V2_ADAPTER.md`  
Results: `research/causal-patching-300m/RESULTS_V2_ADAPTER.md`

v2 isolated only the explicit SELF adapter contribution
`adapter(x, SELF)-x`.

Main findings:

- transplanting the source adapter delta into a target prompt does **not**
  transplant CORE/GOAL/FOCUS identity; source-label switch advantage is 0.0
  across all tested architectures/components/layers;
- continuous source-vs-target margin effects are tiny and mixed in sign;
- single-layer adapter ablation produces directionally non-random
  representation changes but almost no semantic decoding collapse;
- crucially, diff_anchor L3 adapter ablation leaves CORE/GOAL/FOCUS at 1.000,
  even though changing the SELF reference at that same layer in v1 collapses
  them to 0.375/0.639/0.361.

Combined v1+v2 interpretation:

**SELF behaves more like a control/reference coordinate that changes the
transformation applied to content-bearing residual states than like a semantic
payload or memory slot stored in the adapter output itself.**

This strengthens the control/reference interpretation and weakens a
content-storage interpretation.

Next high-information test: directly measure the SELF-by-CORE interaction
field — whether the causal effect vector induced by changing SELF depends
systematically on which CORE is present, compared against multiple
same-displacement random anchor directions.


---

## 300M SELF×CORE interaction field v3 — COMPLETED

Workflow run: **37917508639**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION_V3_INTERACTION_FIELD.md`  
Results: `research/causal-patching-300m/RESULTS_V3_INTERACTION_FIELD.md`

The preregistered direct SELF×CORE interaction criterion was **not confirmed**.

Key findings:

- projected_diff and projected_diff_slow preserve CORE under radial SELF, but
  their radial SELF-effect fields are not consistently more CORE-structured
  than eight random same-displacement anchor directions;
- CORE cross-context decoding can look high while the independent CORE
  cluster-margin test fails or reverses, so CORE decoding alone is not treated
  as interaction evidence;
- in contrast, the learned radial SELF effect field is strongly and unusually
  organized by GOAL/FOCUS:
  - projected_diff L2 local GOAL cross-context z=+7.18, FOCUS z=+3.01;
  - projected_diff_slow L2 local GOAL z=+4.86, FOCUS z=+4.65;
  - diff_anchor L3 final GOAL cluster-margin z=+7.26 and FOCUS z=+4.05 while
    CORE simultaneously collapses;
- diff_only also shows repeated downstream GOAL/FOCUS structure without a
  clean localized CORE-field signature.

Updated mechanistic interpretation:

**SELF is best modeled as a learned causal control/reference direction acting
on content-bearing residual representations, with particularly strong
GOAL/FOCUS organization. CORE remains a distinct stable content/reference
structure, but a privileged direct SELF×CORE interaction has not yet been
demonstrated.**

Next test: a factorial functional interaction assay that measures whether the
effect of changing SELF on downstream GOAL/FOCUS differs non-additively across
CORE values, compared with eight random same-displacement SELF directions.


---

## 300M factorial SELF×context functional interaction v4 — COMPLETED

Workflow run: **37918122058**  
Preregistration: `research/causal-patching-300m/PREREGISTRATION_V4_FACTORIAL_INTERACTION.md`  
Results: `research/causal-patching-300m/RESULTS_V4_FACTORIAL_INTERACTION.md`

The preregistered direct functional SELF×CORE interaction test is **positive**.

Primary projected-family results at layer 2:

- projected_diff CORE->GOAL RMS interaction:
  4.19x random mean, z=+4.57, rank #1/9;
- projected_diff CORE->FOCUS:
  4.85x, z=+4.78, rank #1/9;
- projected_diff_slow CORE->GOAL:
  4.52x, z=+8.01, rank #1/9;
- projected_diff_slow CORE->FOCUS:
  3.59x, z=+5.74, rank #1/9.

CORE decoding remains 1.000 under radial SELF in both projected variants.

Thus v4 resolves the apparent tension with v3:

- v3 showed the SELF-effect *vector direction* is not simply a privileged
  CORE-coded vector;
- v4 shows the *functional consequence* of SELF on GOAL/FOCUS depends strongly
  and non-additively on CORE.

This supports a nonlinear reference/control interaction rather than a literal
CORE code embedded in the SELF-effect direction.

Controls:

- diff_anchor also has real CORE-conditioned SELF interaction, but it is
  pathological because CORE collapses to 0.333 at L3;
- diff_only shows significant CORE->GOAL/FOCUS interaction at L2/L3/L4 while
  single-layer CORE remains 1.000, supporting a distributed interaction that
  becomes destructive only when accumulated across layers.

Strongest supported mechanistic claim:

**In the projected 300M models, learned SELF is a causal reference/control
variable whose effect on downstream GOAL and FOCUS depends non-additively on
the current CORE, beyond eight random same-displacement SELF controls, while
CORE identity remains stable.**

Next gate: replicate the exact preregistered v4 assay across earlier
150M/170M/200M checkpoints to map the developmental emergence of this causal
interaction, then compare with the pretrained Goldfish trajectory and obtain
independent seeds.
