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


---

## Causal SELF follow-up at 300M + trajectory 150M -> 300M — COMPLETED

Whole-stream activation patching run: **37916251165**  
Surgical SELF-adapter causal patching v2: **37916891722**  
Layerwise causal SELF trajectory: **37922411679**

Results:
- `research/causal-300m/RESULTS.md`
- `research/self-layer-causal-trajectory-150-300/RESULTS.md`

Methodological correction:
- the first whole-residual source patch produced near-trivial source-state
  transfer and is not treated as strong localization evidence;
- the valid causal evidence comes from one-layer SELF-anchor interventions and
  surgical adapter-path ablation/source-delta patching.

Main conclusions:
- SELF adapters behave as modulatory/reference pathways rather than direct
  stores of CORE/GOAL/FOCUS identity;
- projected_diff_slow has a persistent layer-2 causal signature across
  150/170/200/300M: radial SELF perturbation is 7.1x–10.4x stronger than
  matched controls and can sharply disrupt GOAL while preserving CORE identity;
- at 200M projected_diff_slow layer-2 radial-2x drives GOAL accuracy to ~0.48
  while CORE remains ~0.995, the strongest tested causal separation;
- diff_anchor shows the opposite phenotype at layer 3: CORE accuracy collapses
  to 0.18–0.48 across the trajectory, with CORE-margin damage eventually
  exceeding the clean margin itself;
- projected_diff keeps CORE at 1.0 under the layer-2 intervention at every
  audited milestone, but its downstream effect is weaker/more FOCUS-oriented;
- diff_only's global 300M CORE entanglement is not attributable to one single
  layer in this assay, suggesting multi-layer accumulation;
- the causal trajectory is non-monotonic and confirms the multiple
  developmental-clocks interpretation.

Next scientific gate: factorial SELF x CORE intervention using a learned CORE
subspace rather than whole-stream replacement, testing for an interaction on
GOAL/FOCUS downstream.


---

## SELF x CORE factorial functional interaction 150M -> 300M — COMPLETED

Runs:
- 150/170/200M trajectory: **37918632896**
- 300M final factorial assay: **37918122058**

Results:
- `research/self-core-factorial-interaction-150-300/RESULTS.md`

Main conclusions:
- projected_diff_slow layer 2 shows a persistent CORE-conditioned SELF effect on
  GOAL across 150/170/200/300M;
- CORE->GOAL RMS interaction under radial SELF is 5.81x / 3.83x / 4.64x / 4.52x
  the mean of eight same-displacement random SELF directions, ranking 1/9 at
  every milestone;
- CORE->FOCUS interaction is also consistently elevated (4.36x / 5.64x /
  3.98x / 3.59x);
- projected_diff also shows persistent CORE-conditioned SELF interaction, with
  especially strong 300M CORE->GOAL (4.19x) and CORE->FOCUS (4.84x);
- diff_anchor has a real interaction, but CORE->CORE dependence is much larger
  than CORE->GOAL and direct SELF intervention collapses CORE, confirming an
  entangled rather than clean-reference phenotype;
- diff_only shows distributed interaction across layers 2–4.

Interpretation guardrail:
SELF is directly intervened on; CORE is independently varied in the factorial
input grid. This is causal evidence for a SELF effect whose downstream impact
depends on CORE, but it is not yet a direct internal CORE intervention.

Next gate: learn a held-out CORE subspace and cross a surgical CORE-subspace
intervention with normal/radial/random SELF at the causally active layer.


---

## Internal SELF x CORE subspace causal interaction — COMPLETED

Run: **37923430126**

Results:
- `research/internal-self-core-subspace-200-300/RESULTS.md`

Design:
- learned rank-5 CORE subspace from held-out context combinations;
- evaluation on disjoint context combinations;
- targeted CORE-subspace patch preserves the orthogonal target residual;
- matched random non-CORE patch control;
- radial SELF crossed with targeted CORE;
- eight random same-displacement SELF controls.

Key results:

### projected_diff_slow, layer 2
200M:
- GOAL interaction = 6.57x random-SELF mean, z=17.96, rank 1/9;
- GOAL interaction = 1.82x radial-SELF x random-CORE control;
- FOCUS = 4.85x random SELF but only 1.04x random CORE.

300M:
- GOAL = 5.83x random SELF, z=9.19, rank 1/9;
- GOAL = 2.12x random CORE;
- FOCUS = 4.93x random SELF and 1.59x random CORE.

### projected_diff, layer 2
200M:
- GOAL = 3.70x random SELF and 1.70x random CORE;
- FOCUS = 6.77x random SELF and 1.76x random CORE.

300M:
- GOAL = 4.36x random SELF and 2.11x random CORE;
- FOCUS = 10.40x random SELF and 2.66x random CORE.

### diff_anchor, layer 3
The dominant two-sided interaction remains CORE itself rather than a clean
downstream GOAL/FOCUS relation, consistent with the direct CORE-collapse
phenotype.

Current strongest defensible mechanistic claim:
the projected architectures learn a direction-specific SELF reference whose
downstream causal effect depends on internal CORE-related coordinates.
projected_diff_slow expresses this most cleanly on GOAL at layer 2, surviving
both random-SELF and random-non-CORE controls.

Caveat:
CORE-subspace manipulation is partial rather than a complete CORE state rewrite.
This supports a two-sided internal interaction mechanism but not semantic
selfhood or a literal fully identified mathematical gate.

Next priorities:
multi-seed replication, stronger but still surgical CORE-subspace manipulation,
and transfer of this assay to sufficiently mature Goldfish checkpoints.


---

## 300M causal mechanism follow-up — COMPLETED THROUGH HELD-OUT-TEMPLATE REPLICATION

Runs:
- full residual activation patching: **37916251165**
- SELF adapter contribution causal patching v2: **37916891722**
- 300M factorial SELF-by-context interaction v4: **37918122058**
- factorial causal trajectory 150/170/200M: **37918632896**
- SELF single-layer causal trajectory 150/170/200/300M: **37922411679**
- held-out-template factorial causal replication: **37925620017**

Results:
- `research/causal-mechanism-300m/RESULTS.md`

Main conclusions:

- full residual source-state patching is a positive causal sanity check but is
  too strong to localize the SELF mechanism because it transfers essentially
  the whole source representation;
- surgical transfer of only the SELF-adapter delta does not transfer
  CORE/GOAL/FOCUS identity, while adapter ablation remains directionally
  non-random. This favors SELF as a reference/modulation pathway rather than a
  content store;
- single-layer SELF intervention is strongly direction-specific across the
  150M->300M trajectory;
- projected_diff_slow layer 2 radial-2x/matched-control final-effect ratios:
  7.10x, 7.32x, 10.36x, 8.71x at 150/170/200/300M;
- projected_diff layer 2 ratios:
  4.76x, 4.33x, 4.74x, 7.51x;
- at 300M, factorial SELF-by-context interaction for projected_diff_slow gives
  CORE->GOAL 4.52x and CORE->FOCUS 3.59x random-sham RMS while CORE discrete
  accuracy remains 1.000 in the canonical assay;
- projected_diff gives CORE->GOAL 4.19x and CORE->FOCUS 4.85x with CORE 1.000;
- diff_anchor remains the destructive-entanglement negative control, with CORE
  collapsing to 0.333 under the corresponding radial intervention;
- the factorial signature persists longitudinally from 150M through 300M;
- held-out-template replication passed across four wording families with
  leave-one-template-out decoders:
  projected_diff aggregate CORE->GOAL 4.83x, CORE->FOCUS 7.05x;
  projected_diff_slow 4.42x and 3.41x;
  radial SELF ranked 1/9 against eight same-displacement random shams for both
  primary cells in every held-out template;
- on the held-out wording replication, projected_diff preserved CORE much more
  strongly than FOCUS (0.980->0.970 vs 0.914->0.782), and projected_diff_slow
  preserved CORE substantially more than GOAL/FOCUS (0.994->0.964 vs
  0.935->0.684 and 0.898->0.675).

Current strongest mechanistic interpretation:

> The projected-family SELF mechanism acts as a learned, direction-specific
> contextual reference/modulator. Perturbing SELF at layer 2 changes how
> CORE-conditioned information propagates into downstream GOAL/FOCUS much more
> than matched sham directions, while CORE identity is comparatively preserved.

This does not establish semantic selfhood or multi-seed generalization.


## 2026-10-11 — English SmolLM2 per-user personal-memory research (branch-only)

Project: `experiments/personal_smollm2/`; experimental branch: `experiment/personal-adapter-smollm2`.
Frozen backbone: `HuggingFaceTB/SmolLM2-360M-Instruct`.
Per-user adapter: 7,680 trainable parameters. This is a separate research track from the native 5M SELF architecture and does not establish its quality.

- Initial 6-fact longitudinal pilot (GitHub Actions `38085893514`): final frozen 1/6, personal 5/6, explicit oracle facts 6/6. Very small and forced-choice.
- V1 50-fact run `38087143196`: frozen 27/50, personal 33/50, truncated all-notes 30/50; replay coverage and retrieval truncation confounds identified.
- V2 50-fact run `38088876024`: frozen 27/50, personal 32/50, oracle note 49/50. New facts received >=2 exposures; long-term rehearsal still sparse.
- V3 50-fact run `38090026196`: frozen 27/50, replay-personal 35/50, non-oracle lexical top-1 retrieval 46/50, hybrid 49/50, retrieval hit@1 45/50. Every old fact replayed, two exposures per new fact; 325 updates. The hybrid improvement over retrieval (3 additional correct, no lost correct) is not conclusive (two-sided exact paired p=0.25). Still two-answer ranking on structured fictional facts.
- V4 (GitHub Actions `38091405388`, launched; result **pending** when logged): mixed fictional chat history (24 facts across six stages, two explicit preference revisions and unrelated chatter), real top-3 note retrieval from user messages, frozen/recent-only/replay-personal/retrieval/hybrid comparisons; 32 steps per stage for each of two adapter strategies, three seeds (11/19/37), final unrestricted greedy answer generation plus 8 general non-personal sanity questions. See `experiments/personal_smollm2/EXPERIMENT_V4.md`. Full metrics plus per-seed checkpoints and aggregate artifacts are configured. **Do not report V4 improvements before receiving the actual metrics.**

Interpretation: V3 supports technical feasibility for hybrid personalized conditioning; it does not establish robust persistent memory in realistic conversations. Preserve the 5M native experiment track and the original V1–V3 code/results. Evaluate V4 outcomes and confounds before deciding subsequent variants.

## 2026-10-11 — SmolLM2 personal memory V4: completed three-seed chat-history evaluation

- [GitHub Actions run 38091405388](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38091405388) completed SUCCESS: seeds 11/19/37, plus aggregate job.
- Detailed artifact-verified audit: [FINDINGS_V4.md](experiments/personal_smollm2/FINDINGS_V4.md).
- Final 24-item two-choice accuracy (3-seed averages): frozen 54.2%, recent-only adapter 81.9%, full-replay adapter 75.0%, non-oracle top-3 memory 79.2%, replay+memory hybrid 93.1%.
- Final 24-item **open-answer string-scored** accuracy: frozen 4.2%, recent-only adapter 18.1%, full replay 12.5%, retrieval 54.2%, hybrid 61.1%. The high two-choice accuracy should NOT be treated as reliable open-form recall.
- Eight generic open QA items: base/retrieval 87.5% (automatic match; actually includes an obvious false negative on written-out `eight`), recent-only adapter 62.5%, replay and hybrid 54.2%. Sample is tiny but generation regressions and repetitive completions are plainly observable.
- The explicit preference revision 35→25 minutes was answered correctly by recent-only adapter in all three seeds, but **incorrectly** by retrieval-only, full replay, and hybrid in all three seeds. Retrieval version handling is a blocker. Retrieval top-1 20/24 and top-3 22/24 refer to matching *fact IDs*, and overstate up-to-date revision correctness because old and corrected statements share an ID.
- Decision: retain frozen backbone plus external memory as safer provisional baseline; hybrid remains experimental. Before a V5 run, implement versioned/superseding canonical memory, selectively gated adapters for personal vs generic queries, a robust free-answer grader plus fixed manual audit, general-capability preservation controls, and independent synthetic users. See FINDINGS_V4.md for locked recommendations. Do not merge V4 to production as a validated result.

## 2026-10-11 — SmolLM2 personal-memory V5 experiment launched (results pending)

- Why: V4 exposed outdated user-preference recalls (35→25), general-question regressions when an adapter is always on, and brittle automatic free-answer grading.
- Experimental branch `experiment/personal-adapter-smollm2`; protocol and validity controls: [EXPERIMENT_V5.md](experiments/personal_smollm2/EXPERIMENT_V5.md).
- Implemented deterministic revisioned canonical memory from *constrained user utterance grammar*; preserves supersession/provenance, rejects inconsistent correction, serves only active fact values.
- Implemented a transparent personal-query gate; general/unmatched queries use frozen SmolLM2 and no adapter. Kept an always-on hybrid arm and a **direct exact-memory arm** for a strong factual baseline.
- Trains a 7,680-parameter per-profile adapter on currently valid facts only; each stage rehearses all active slots with additional new/corrected exposure.
- Testing: three distinct fully fictional profiles with 12 slots each, two explicit updates per profile, 10 generic QA checks, open-ended generated answers, simple stale-value and repetition penalties. Profile and seed are not independently varied; extraction is **not** natural unrestricted chat parsing.
- Workflow: [SELF V5 run 38092858179](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38092858179). Three matrix jobs plus aggregate configured. **No V5 performance results were available at logging time**. Keep V4 findings and code intact; do not merge as a validated production design.


## 2026-10-11 — SELF V6 response-style study (implementation complete, result pending)

- Scope: frozen SmolLM2-360M-Instruct and rank-4 / 7,680-weight per-user adapter **for response style**, not factual recall. This experiment remains separate from native 5M SELF tracks.
- V5's frozen-LM + revisioned-memory baseline consistently outperformed factual adapter generation and generic always-on adapter output; V6 tests whether a small module can add value to *format behavior* instead.
- Three distinct fictional styles (two bullets, three numbered steps, one concise sentence), 12 synthetic training demos per style, 12 unseen-topic prompts, 48 gradient steps per run, two seeds per profile (11,19), total 6 independent runs.
- Five arms: frozen base, frozen + explicit remembered style prompt, adapter only, adapter + explicit prompt, selective gated adapter. **Explicit prompt is a mandatory comparator**; improving vs bare base alone is insufficient.
- Memory controls: constrained-language preference extraction, supersession, quoted/third-party statements, hypotheticals, explicit active deletion, two-request temporary overrides and expiry, confidence-safe clarification on unsupported statements. Stored audit source is not erased; privacy deletion is not yet production-grade.
- Held-out metric: literal response-*format* checks, not semantic correctness. Eight generic QA checks with raw generation preserved, and adapter bypass on generic prompts. Procedure and limitations: [EXPERIMENT_V6.md](experiments/personal_smollm2/EXPERIMENT_V6.md).
- GitHub Actions [run 38093850221](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38093850221) launched; **results not yet available at logging**. Do not imply V6 has validated personalization. Keep branch unmerged until results are examined.

### SELF V6 CI retry after preflight import typo

- The initial V6 workflow run [38093850221](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38093850221) failed before training in preflight due to a workflow-only symbol typo: imported `GENERIC` instead of the actual `GENERAL` benchmark question list. This was not a training result.
- Corrected the smoke-test import in `.github/workflows/self-v6-style.yml` on `main`, commit `218e56ac9a1dd913a7ff783f53acfc9748ad20d8`.
- Replacement run: [38093915224](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38093915224). Treat outcomes as pending until preflight, all six matrix jobs, aggregate job and artifacts pass. Do not selectively use an unsuccessful run.

### 2026-10-11 — SELF V6 style trial: corrected run completed and independently audited

- [Corrected GitHub Actions V6 run 38093915224](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38093915224): **SUCCESS**; all six style/seed jobs plus aggregate report. Original run 38093850221 failed preflight due to a workflow-only wrong import and was excluded.
- [V6 complete forensic report](experiments/personal_smollm2/FINDINGS_V6.md) and [reproducible alternate rubric script](experiments/personal_smollm2/audit_v6_results.py).
- Three fictional preference styles, two seeds each, 12 held-out prompts per run (72 outputs per condition), 48 training steps and 7,680 per-user trainable weights.
- Official surface-format scores: frozen 6/72, explicit style prompt 12/72, adapter only 8/72, **adapter + explicit prompt 51/72**, gated adapter without style prompt 8/72. The latter disables the adapter for generic questions but is not the combined gated+prompt condition.
- Exploratory score that also accepts two numbered items for the bullet preference and up to 50 words for one sentence: explicit prompt **30/72**, adapter+prompt **51/72**. Apparent improvement shrinks from 54.2 to 29.2 percentage points but persists on the same synthetic outputs.
- Generic QA in 8 probes x 6 runs: 36/48 frozen/gated, 41/48 adapter-enabled. Small and brittle checks; model outputs include refusal and nonsense, so do NOT infer enhanced or safely preserved general reasoning.
- Important confound: 76 generated tokens per response truncate long prompt-only answers more often; short learned style improves format completion but not necessarily semantic utility.
- Memory control (constrained parser) passed six runs for update, active deletion, temporary override and expiry; audits retain history, hence no production privacy deletion guarantee.
- Decision: **promising limited style effect conditional on an explicit preference prompt**, not spontaneous personalized output or a validated production feature. V7 needs a gated+prompt arm, non-trained adapter control, longer outputs, content-quality judgments and larger independent profiles. Existing SELF 5M core unaffected; branch remains experimental.

## 2026-10-11 — V7 gated style-conditioning falsification experiment launched

- Motivation: V6's trained residual adapter + explicit style prompt improved synthetic format compliance, but apparent advantage was sensitive to grading and output truncation; the gated+prompt configuration and untrained controls were not tested.
- Isolated experimental implementation: [benchmark_v7.py](experiments/personal_smollm2/benchmark_v7.py), [aggregate_v7.py](experiments/personal_smollm2/aggregate_v7.py), [EXPERIMENT_V7.md](experiments/personal_smollm2/EXPERIMENT_V7.md).
- Frozen SmolLM2-360M-Instruct, 7,680 learned per-profile adapter parameters, three fictional styles x two independent training seeds (11/19). Six arms: frozen, explicit style prompt, zero-residual untrained adapter+prompt, trained adapter only, trained adapter+prompt, and **gated trained adapter+prompt**. The style is from active memory; generic queries bypass adaptation and personalization prompts in the gated arm.
- Main: 160 generated tokens, 18 held-out advice questions per trial (12 prior and 6 new), locked strict and permissive structural scorers, fixed topic keyword anchors as **weak content proxy**, limit-hit and repetition checks. Additional matched 76-token budget comparison on explicit prompt and trained adapter+prompt, plus 8 generic QA questions and locked unreviewed qualitative sample.
- Preflight checks include constrained active-memory deletion, temporary override expiry, third-party and hypothetical rejection, task routing, non-trained zero-residual adapter identity. Active deletion does **not** erase historical audit events or already learned adapter data.
- Workflow: [V7 Actions run #38095717406](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38095717406) launched with six matrix jobs and one aggregate. First jobs passed preflight and entered the evaluation step; **results pending**. Retain this branch as research only; no production merge or performance claim pending artifact audit.

## 2026-10-11 — V7 completed style-and-content audit

- [GitHub Actions run 38095717406](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38095717406) completed **SUCCESS**: all six profile/seed jobs and the aggregate report. Per-trial outputs and checkpoints saved. Read [FINDINGS_V7.md](experiments/personal_smollm2/FINDINGS_V7.md) for detailed audit, negative controls and exact paired findings.
- On 108 held-out synthetic advice outputs at 160 tokens: **explicit preference prompt** permissive-style 74/108, content-keyword proxy 78/108, joint 54/108; **trained 7,680-parameter adapter+prompt** 88/108, 68/108, joint 57/108. Thus formatting improved substantially, keyword coverage fell and joint usefulness *proxy* barely improved.
- A zero-residual untrained adapter had **identical raw outputs** to explicit prompting across 108 cases. A gated adapter+prompt also produced identical personal outputs to always-on adapter+prompt, while entirely bypassing the adapter in 48 generic QA questions.
- At shorter 76-token budget: prompt-only joint 32/108 with 44/108 hitting token cap; trained adapter+prompt joint 47/108 with 22/108 hitting cap. At 160, joint scores converged to 54/108 and 57/108. **Truncation explains part of apparent V6 gain**; shorter answers often complete format earlier but may lose topical content.
- Locked qualitative review examples reveal additional limitations: explicit prompting can already produce acceptable numbered items for a two-bullet request; trained adapter can produce generic-sounding but formally correct steps; adapter-only can repeat nonsense. Format-only scores are **not** semantic evaluations.
- Gated generic scores matched frozen 36/48 by explicitly bypassing adapter; always-on adapter 41/48 on a tiny, lexically graded quiz. No broad non-regression claim.
- **Decision:** V7 supports limited style-conditioning with explicit memory prompt, but **does not show meaningful overall content-quality benefit or spontaneous retained style**. Maintain frozen+memory as default; keep experimental branch unmerged. Future research needs stronger semantic grading and fair comparisons to controlled prompting/decoding, with independent user histories and robust deletion controls.

## 2026-10-11 — COMPASS V8 planning (personal project hierarchy and prioritization)

**Nomenclature correction:** `SELF` is the separate model-internal self-reference research mechanism. The *personal prioritization adapter* is now **COMPASS (מצפן)**; prior SmolLM2 personal-adapter V1–V7 trials are historical predecessors, not SELF-mechanism experiments.

- Created isolated research branch `experiment/compass-v8`, under `experiments/compass_v8/`.
- Registered [EXPERIMENT_V8.md](experiments/compass_v8/EXPERIMENT_V8.md), [TERMINOLOGY.md](experiments/compass_v8/TERMINOLOGY.md) and [DOCUMENTATION_EXAMPLES.json](experiments/compass_v8/DOCUMENTATION_EXAMPLES.json).
- V8's question is *decision quality* across multiple independently defined project cores, supporting layers, ancillary work, blocker dependencies and source/validity rules — **not** style matching or factual memory reproduction.
- Architecture: versioned external Project Memory holds core definitions and authoritative decisions; optional learned COMPASS adapter learns cross-project personal weighting/triage policy; frozen LLM interprets and proposes actions, while the user remains authoritative on ambiguous choices.
- Key falsification: rank 4 (7,680 weights), 16 (30,720) and 64 (122,880), against frozen graph-only, **graph + explicit personal policy**, deterministic rule engine, and zero-residual untrained adapter. Locked unseen-project trap cases distinguish structural importance from immediate priority. Evidence grounding, project-isolation, revision, abstention and hard-rule violations are measured.
- **This is a preregistered design, not an experiment result. No V8 model trained, workflow dispatched or weights changed.** Implement validated fictional fixtures and an independent rules baseline before authorizing training matrix. Native SELF model and main branch unchanged.

## 2026-10-11 — COMPASS V8 Stage 0 completed and rank4 pilot launched

- [Stage-0 Actions #38098797553](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098797553) **SUCCESS**. Created `fixtures_v8.py`, `baseline_v8.py`, `validate_v8.py` in `experiments/compass_v8/` and documented findings in [STAGE0_FINDINGS.md](experiments/compass_v8/STAGE0_FINDINGS.md).
- Fixture scope: 4 fictional user profiles × 3 fictional projects each × 8 cases per project = **96** tasks (64 train / 32 unseen-project evaluation). Audits passed for opaque nonleaking task IDs, distinct train/test project and authoring-template IDs, evidence provenance, project isolation, and 12 core-versus-periphery paired traps. Gold and input kept in separate JSONL artifact files.
- Deterministic authored-rule baseline gives **96/96 on this templated fixture**, which is only a schema / internal-consistency sanity check, not valid generalization evidence.
- Launched [rank-4 pilot Actions #38098918328](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098918328) after Stage-0 revalidation; `pilot_v8.py` trains a 7,680-parameter COMPASS adapter on frozen SmolLM2-360M for 32 supervised updates, then checks 16 held-out test tasks via six-candidate action log-loss for graph-only/explicit policy with and without adapter.
- Pilot results **pending** at logging time. Full ranks 4/16/64 and multi-seed matrix NOT launched. Do not claim semantic decision reasoning, independent review, cold-user transfer or a validated model yet. COMPASS is separate from model-internal SELF experiments.

## 2026-10-11 — COMPASS V8 rank-4 pilot finished — action collapse

- [Actions 38098918328](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098918328) completed **SUCCESS**, including stage-0 validation; saved rank-4 adapter and full decision scores. [Forensic analysis](experiments/compass_v8/PILOT_RANK4_FINDINGS.md).
- 16 held-out cases, 32 supervised updates, seed 11, frozen SmolLM2-360M-Instruct + 7,680 trainable COMPASS weights. Accuracy: frozen graph 6/16; frozen graph+policy 5/16; adapter+graph 8/16; adapter+graph+policy 8/16; hand-authored deterministic sanity baseline 16/16.
- **Negative scientific result:** both trained-adapter modes predict `DO_NOW` on **every** test case. The gold set contains eight `DO_NOW` labels and eight other actions, so the adapter's 8/16 = **majority-class constant baseline**, not learned conditional prioritization. All four broad-QA tasks and four uncertain-defect cases are misclassified.
- **Decision:** working CI/training plumbing ≠ successful COMPASS policy learning. Do not yet launch 3 ranks × 3 seeds under this biased setup. Add a shared-prefix calibrated candidate score, majority baseline, per-class/balanced metrics, before/after scenario pairs and independent gold adjudication; then validate and parallelize the rank matrix.

## 2026-10-11 — V8 corrected 3-rank parallel matrix launched

- Previous [rank4 seed11 pilot](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098918328) trained successfully but emitted **DO_NOW on all 16** held-out decisions. Its 8/16 apparent accuracy was exactly a constant-majority-answer control, not learned strategic prioritization. Forensic evidence: [PILOT_RANK4_FINDINGS.md](experiments/compass_v8/PILOT_RANK4_FINDINGS.md).
- Implemented [matrix_v8.py](experiments/compass_v8/matrix_v8.py), [aggregate_capacity_v8.py](experiments/compass_v8/aggregate_capacity_v8.py) and [CAPACITY_DIAGNOSTIC.md](experiments/compass_v8/CAPACITY_DIAGNOSTIC.md), focusing on common-prefix option token log-probabilities, action-balanced training, per-class/balanced scores, trivial constant classifier baseline, ancillary/core contrast pairs and hard scope/approval violations.
- Workflow: [COMPASS V8 rank 4/16/64 × 3 seeds — GitHub Actions #38099765087](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38099765087).
- **Preflight succeeded**: project/test split, generated fixture audit, option-code token-length scoring on identical context, six-class balanced sampling, collapsed-answer detection. All nine matrix jobs entered execution; outcome pending at documentation time. Each trial runs 60 supervised updates on 64 authored train fixtures and evaluates 32 held-out examples.
- Sizes: rank 4 = 7,680 parameters; rank 16 = 30,720; rank 64 = 122,880; seeds 11/19/37. Frozen backbone: SmolLM2-360M-Instruct. This is a capacity diagnostic with shared synthetically authored English questions, not real-user policy adaptation or validated free-form project reasoning.
- Scientific guard: **do not call a rank successful because of nominal accuracy**. It must outperform the constant action and explicit-policy baselines on meaningful per-class and core-impact counterfactual discrimination, without critical out-of-scope decisions. All results retained even if negative. COMPASS remains separate from the model-internal SELF research.

## 2026-10-11 — COMPASS V8 nine-run rank matrix completed; critical no-go

- [GitHub Actions #38099765087](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38099765087): **SUCCESS**, preflight, nine rank/seed training-evaluation jobs and aggregate. All 9 trial artifacts and summary were checked; full audit: [CAPACITY_FINDINGS_V8.md](experiments/compass_v8/CAPACITY_FINDINGS_V8.md).
- Experimental setup: frozen SmolLM2-360M, 60 balanced supervised updates per adapter, ranks 4/16/64 (7,680 / 30,720 / 122,880 weights), seeds 11/19/37, **same 32 authored fictional test tasks** in every run. Mean graph-only action score grows 8.7/32 -> 11/32 -> 12/32, but the constant `DO_NOW` baseline is **12/32**.
- With learned adapter + explicit personal-policy prompt, mean scores by rank: 10/32, 12/32, **13.67/32**. Best single seed reached 15/32, still did not pass category or authorization gates.
- CRITICAL SCIENTIFIC FAILURE: across all nine graph-only adapter evaluations, **SCHEDULE 0/54, ASK_USER 0/36, REJECT 0/36**. In all nine graph-only runs, **4/8 approval/scope-boundary tasks were incorrectly promoted to action**. Mean balanced accuracy rank4/16/64: **0.231 / 0.236 / 0.292**. Ancillary-test vs core-blocker contrast rank4=0/4, rank16=0/4, rank64 mean 0.67/4.
- Score sensitivity remains important despite a common prefix: mean vs summed action-token likelihood can change exact rankings materially (frozen graph-only 4/32 vs 13/32). No claim of semantic priority judgement, real-user policy transfer, or free-form grounded rationale is warranted.
- **Decision: NO-GO for all capacity choices.** Trainability and apparent rank scaling do not establish competent project prioritization. Keep this isolated on `experiment/compass-v8`; SELF self-model and native 5M model unaffected.
- Proposed V8.1 correction: independent audited varied-project dataset with scope/approval counterfactuals, stronger equal-token/calibrated classifier, hard permission checks, held-out user policies and validity revisions, followed by a small calibration pilot before more nine-job launches.

## 2026-10-11 — COMPASS V8.1 paired-action pilot with external authorization enforcement

- Created branch `experiment/compass-v8-1`, preserving V8 and model-internal SELF research separately.
- Files: [plan](experiments/compass_v8_1/EXPERIMENT_V8_1.md), [synthetic dataset](experiments/compass_v8_1/fixtures_v81.py), [scope/approval guard](experiments/compass_v8_1/guard_v81.py), [data audit](experiments/compass_v8_1/validate_v81.py), [trainer/evaluator](experiments/compass_v8_1/pilot_v81.py).
- Test prep uses 192 fictional train and 72 heldout cases, with disjoint fictional user policies, project IDs and wording families. Six counterfactual task pairs per project require differing decisions. Gold decision labels never enter model prompt.
- Replaced unequal-length response choices with six verified single-token digit labels; training and inference optimize the same six-category cross-entropy. Calibration with frozen neutral-context logits is explicitly diagnostic, while raw neural classifier results remain the primary score.
- Scope/authorization guard is an **external deterministic check**; any guarded accuracy belongs to that check, not the trained adapter. Cross-project and missing-scope rejection are audited.
- [GitHub Actions V8.1 run 38101573253](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38101573253): structural/permission preflight PASS; rank-4 seed-11 48-pair-update training/evaluation entered its run step. **Model performance pending** as of this log update. No larger-rank expansion until raw action recalls and counterfactual discrimination are checked.
- This is a tiny synthetic experiment, with no human-adjudicated project priorities or free-form reasoning. The original V8 rank-scaling study was NO-GO due to failed ASK_USER/REJECT/SCHEDULE labels.

## 2026-10-11 — COMPASS V8.1 completed; neural policy fails independent-profile pilot

- [Actions run 38101573253](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38101573253) **SUCCESS**; Stage-0 validation and rank-4 seed-11 48-pair-update pilot both completed. Fetched and inspected all 24 decisions from original artifact **11688376472**. [Full findings](experiments/compass_v8_1/FINDINGS_V8_1.md).
- User/project cold-holdout scope: 192 synthetic train items, 72 synthetic test items, 24 test items across 2 truly unseen fictional projects and profile IDs used by pilot. Rank-4 learned adapter 7,680 weights; frozen SmolLM2-360M.
- **No neural generalization gain:** frozen raw **7/24 (29.2%)** and trained COMPASS raw **7/24 (29.2%)**, both below `SCHEDULE` constant-majority **8/24 (33.3%)**. Both raw model arms have **14.6% balanced accuracy** and **zero of twelve correctly distinguished contrast pairs**; trained COMPASS predicts only two distinct actions. Neural recalls: **DO_NOW 0/6, INVESTIGATE 0/4, DEFER 0/2, ASK_USER 0/2, REJECT 0/2, SCHEDULE 7/8**.
- External scope/authorization guard alone corrected **4** raw errors: two pending changes `SCHEDULE→ASK_USER`, two forbidden expansions `DO_NOW→REJECT`. Trained+guard accuracy **11/24 (45.8%)** matches frozen+guard **11/24**. This is **rules-based enforcement, not COMPASS learning**. Guarded contrast pairs only 4/12 (approval/scope pairs).
- Frozen-neutral logit calibration collapsed both to `INVESTIGATE` on all 24, yielding **4/24 (16.7%)**. Uniform-class cross-entropy controls and full logits archived in original artifact.
- **NO-GO for rank 16/64 expansion** while this classifier cannot reliably detect project-core blockers and permitted scope. Next: independent human-curated bilingual tasks, balanced development and locked test, natural-language decisions and a separate classification/ranking head or comparative loss, validate genuine context sensitivity before costly rank matrix.
- Historical V8 and separate model-internal SELF unaffected. No production merge.

## 2026-10-11 — COMPASS V8.2 bilingual pairwise action preference pilot

- Created isolated experimental branch `experiment/compass-v8-2` from V8.1, leaving separate model-internal SELF research untouched.
- Implemented [V8.2 protocol](experiments/compass_v8_2/EXPERIMENT_V8_2.md), [fictional EN/HE project scenarios](experiments/compass_v8_2/fixtures_v82.py), [validation](experiments/compass_v8_2/validate_v82.py), [external trusted-approval guard](experiments/compass_v8_2/guard_v82.py), and [rank-4 training/evaluation](experiments/compass_v8_2/pilot_v82.py).
- Technical shift: a **binary preference between two concrete proposed steps**, rather than prediction of one of six generic decision labels. Six decision families test confirmed vs uncertain core fault, urgent vs optional QA, blocking vs routine prerequisite, approved vs unapproved core revision, approved vs forbidden scope change, and valid vs superseded plans.
- Dataset: **576 synthetic authored train records** across six fictional profiles/12 projects; **288 synthetic held-out records** across three distinct fictional profiles/six projects, EN and HE with fact flip × reversed option order. Initial rank-4 pilot is **96 supervised updates** and **96 test instances across two held-out project profiles**. Constant-0/1 baseline is 50%; every complete fact-and-order contrast quadruple must succeed for structural evidence.
- <https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38102375388> workflow **Stage-0 preflight PASS** (train/test separation, balanced class labels, opposite-evidence and swapped-choice oracle checks, external permit/scope guard and project identity checks). Rank-4 training job queued at recording time.
- Scientific caution: fictional scripted labels and same generic preference wording across profiles do **not** demonstrate personalized policy learning, independent semantic generalization, full project planning or autonomous scope enforcement. Raw model outputs and post-guard outputs will be scored separately. No rank-16/64 matrix until raw paired-context success is established.

## 2026-10-11 — V8.2 technical rerun after Hebrew tokenizer cap

- First [Actions run #38102375388](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38102375388) passed the bilingual dataset/authorization preflight but **failed in the model job at the first supervised step** because its Hebrew prompt required 683 tokens and the provisional maximum was 650. **No V8.2 neural metrics were produced by that attempt**.
- Corrected the model-side prompt limit to 1,200 tokens, preserving the **entire** prompt without silent truncation. Added an early tokenizer pre-scan over all synthetic train and test contexts in both languages, before model allocation; any over-limit context now fails visibly with a per-language max report.
- Triggered **job rerun** on the same Actions workflow after committing the corrected trainer to `experiment/compass-v8-2`. The GitHub checkout explicitly follows that branch. Rerun model job is GitHub job ID **114361111019**; performance pending as documented.
- Distinguish external GitHub workflow activity from a completed result. Do not claim improved ranking ability until the full bilingual fact-flip, option-order and frozen-vs-learned comparison has finished and its artifact has been checked.

## 2026-10-11 — COMPASS V8.2 completed — language-conditioned constant choice, NO-GO

- [GitHub Actions run 38102375388 attempt 2](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38102375388) **SUCCESS**; verified both worker logs and original artifact `compass-v82-r4-s11` ID `11688651484`. Full audit and reproducible metrics in [FINDINGS_V8_2.md](experiments/compass_v8_2/FINDINGS_V8_2.md).
- Frozen SmolLM2-360M + isolated personal COMPASS rank4 (7,680 weights), seed11, 96 balanced paired-choice updates, 96 held-out fictional project decisions from 2 unseen fictional project profiles (48 English and 48 Hebrew). Train/test written fixture pools 576/288 items, but **only 96 test items evaluated**.
- **Raw result: frozen 48/96 (50%); COMPASS 48/96 (50%); exactly identical argmax for all 96.** Both frozen and trained select **option 1 for every English prompt (48/48)** and **option 0 for every Hebrew prompt (48/48)**. Thus apparent overall 50/50 label diversity is a language-conditioned constant predictor, not priority judgment.
- **Zero of 48 factual reversal pairs**, zero of 48 option-order reversal pairs, **zero of 24 complete four-way contrast sets** correct for either raw model. Every case family exactly 8/16 for both raw modes.
- External trusted-owner/scope guard modifies 12 decisions, raises both models to **60/96 (62.5%)**, permits 12/48 factual reversals but still **0/24 full contrast sets**. The gain comes entirely from deterministic enforcement, not neural COMPASS.
- English maximum prompt size 198 tokens, Hebrew maximum 726 tokens; earlier run attempt failed a 650-token ceiling (683 observed). A corrected 1200-token nontruncating preflight enabled success. Large language-conditioned token-prior and sequence-length differences are research confounders, not evidence of Hebrew understanding.
- **Decision: NO-GO for larger rank matrix; do not merge adapter into product/native SELF.** V8.3 candidate: measure token/position/language prior, explicit paired margin loss + full reversal consistency, stronger baselines, genuine distinct user-policy profiles and independent bilingual adjudication. Model-internal SELF remains a wholly separate experiment.

## 2026-10-11 — V8.3 preference margin and evidence/order reversal experiment launched

- New isolated branch `experiment/compass-v8-3` based on frozen V8.2 research, without touching native model-internal SELF or production.
- Added [paired margin trainer](experiments/compass_v8_3/train_v83.py) and [prespecified objective/gates](experiments/compass_v8_3/EXPERIMENT_V8_3.md). Six synthetic bilingual project dilemmas, each shown in both factual states and both possible choice orders. Train rank4=7,680 residual weights on frozen SmolLM2-360M; 48 updates with four simultaneous cases per update.
- Loss: softplus signed choice margin with target 0.75 plus 0.2 order-reversal and 0.2 fact-reversal consistency penalties. Heldout pilot uses 96 instances across two distinct fictional project/user IDs from pre-existing V8.2 train/test split (576/288 total fixtures).
- Frozen-base and learned-model raw predictions tested identically; **external permission/scope guard evaluated separately**. User-profile texts are not genuinely different in this synthetic generator; no personalized policy claim is possible. Labels are research-authored, not independently adjudicated.
- [GitHub Actions V8.3 run #38103592949](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38103592949): **preflight PASS** for synthetic benchmark, group pairing, mathematical loss ordering and differentiation; main model job launched. Research results **PENDING** when recorded. Do not claim generalization without full 96-case artifact audit.
- Gate: raw COMPASS >=55/96 and above frozen; at least 20/48 fact-reversal pairs, 20/48 order-reversal pairs, 8/24 full quadruples, and no degraded Hebrew/scope decisions; no automatic rank16/64 expansion unless this stage passes.

## 2026-10-11 — V8.3 complete training but report variable crash; patched and rerun

- [GitHub Actions run 38103592949 attempt 1](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38103592949) completed with **FAILURE** at the final report assembly, NOT at the model optimizer or evaluation.
- Preflight validation PASS. Model trial (rank4, seed11) completed **48/48 contrastive optimizer updates** and processed **96/96 heldout decisions**. No raw experiment JSON or adapter checkpoint was uploaded because the script raised `NameError: name 'begin' is not defined` while setting report `time_seconds`. The actual timer was named `beginning`. No scientific decision score is available from that failed attempt.
- Corrected the one-line reference `time.monotonic()-beginning` in `experiments/compass_v8_3/train_v83.py`, commit `c6cfc702d66823f09c03fe5b9788694513735cb9`.
- **GitHub Actions job rerun was requested successfully**, triggering new model job `114367003747` using the corrected experiment branch. Original validated data and locked protocol unchanged. Results must be rechecked from the new artifact before any rank-scaling claim.
- No work has been merged into native SELF or production. Do not call V8.3 a success until rerun job reports SUCCESS and raw decision metrics are inspected.

## 2026-10-11 — V8.3 completed; minor English gain but paired reasoning gates fail

- [GitHub Actions V8.3 #38103592949, attempt 2](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38103592949) **SUCCESS**. Rank4 seed11 completed all 48 quartet updates, evaluated all 96 held-out EN+HE prompts, saved raw per-decision JSON and 7,680-parameter checkpoint in artifact **11690110095**. Both worker logs and original JSON checked. [Full audit](experiments/compass_v8_3/FINDINGS_V8_3.md).
- Raw frozen model **48/96 (50.0%)** vs trained COMPASS **50/96 (52.1%)**: only **two** changed predictions, both successful English targeted regression-check cases at the *same state and choice order* in different synthetic projects. Other families unchanged; Hebrew **24/48 → 24/48**.
- Neural evidence/fact flips **0/48→2/48** and order flips **0/48→2/48**; complete four-way counterfactual groups **0/24→0/24**. Predeclared raw gate **55/96**, paired goals **20/48 each**, quadruple goal **8/24**: **ALL FAILED**. Larger Rank16/64 matrix **NOT RUN**.
- The original **language-conditioned near-constant selection remains**: frozen EN always 1, HE always 0; trained EN chooses 1 on **46/48**, HE chooses 0 on **48/48**. Reported overall 50%/52% is not evidence of genuine project-priority discrimination.
- Postprocessing guard: frozen+guard **60/96 (62.5%)**, trained+guard **62/96 (64.6%)**. Exactly **12** actions corrected by explicitly authored owner/scope rules for each model condition; this is **not** neural learning. Both guarded modes still **0/24** complete counterfactual groups.
- Training objective average first/last 12 updates **1.8845→1.2946**, but modest training-loss improvement does not establish generalization. Maximum nontruncated English/Hebrew prompt lengths 198/726 tokens; lexical and language-position bias remain open diagnostic confounds.
- **Research decision NO-GO for production and capacity scaling.** Next prioritize testing frozen prompt-position priors, bilingual option-code calibration, separate ranking head and independent policy profiles/labels before expensive training. Native SELF model and production code unchanged.

## 2026-10-11 — COMPASS V8.4 shared frozen-option ranking head pilot launched

- Context: V8.3 technically passed GitHub Actions but measured only **50/96** raw learned vs **48/96** frozen, 2/48 correct opposite-evidence pairs, **0/24** complete contrast sets; model still almost always chose option 1 in English and option 0 in Hebrew. Details: [V8.3 findings](experiments/compass_v8_3/FINDINGS_V8_3.md).
- Created isolated experimental branch `experiment/compass-v8-4` and implemented [shared option-ranker](experiments/compass_v8_4/ranker_v84.py) and [prespecified protocol](experiments/compass_v8_4/EXPERIMENT_V8_4.md). Frozen SmolLM2-360M encodes **each candidate separately** with the same project/evidence/policy context. A tiny shared `960→4→1` head (3,848 weights) learns a scalar fit score. Comparing two candidate scores structurally guarantees order-swap consistency; it does **not** establish semantic reasoning.
- Source dataset remains synthetic 576 train-pool / 288 held-out in EN+HE; hold out one fictional training profile for *development* (96 examples), train on 480 other examples and evaluate **96 locked items from two unseen test projects**. All three groups are project-disjoint. No real user data.
- Matched frozen **per-candidate Yes-vs-No scoring** is computed from the **same frozen contextual candidate embeddings** as the trainable head, and reported as a separate arm alongside unchanged historical V8.3 numbers (with prompt-change caveat). Deterministic permission/scope rules are a separate postprocessing arm, not trained intelligence.
- Gradient-based pairwise score loss explicitly trains both opposite fact states; option swapping uses the same frozen features, preventing superficial position learning by construction. Classifier head evaluated with complete state×order quadruples, per-language choice histograms and 48 factual reversal pairs. No parameter scaling without a genuinely better **factual** discrimination result.
- GitHub Actions run: [COMPASS V8.4 Frozen Embedding Ranker Pilot #38106163536](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38106163536), **launched**; do not claim model outcome while pending. Separate SELF and production remain unchanged.

## 2026-10-11 — V8.4 completed; first positive synthetic evidence-reversal gate

- [COMPASS V8.4 Actions run 38106163536](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38106163536) completed **SUCCESS**. Downloaded and inspected original artifact **11690043605** (ranker JSON + safetensors). Full audit: [FINDINGS_V8_4.md](experiments/compass_v8_4/FINDINGS_V8_4.md).
- New architecture: frozen SmolLM2-360M separate embeddings for each of two options and a **3,848-parameter** shared 960→4→1 ranker; 480 synthetic training choices, 96 development choices, **96 locked synthetic test choices** from two unseen fictional projects. Chose **epoch 11** by development (82/96 correct; 18/24 full groups).
- **PROMISING SYNTHETIC RESULT:** ranker **74/96=77.1%** correct vs prompt-matched frozen per-option Yes/No **52/96=54.2%**. Evidence-flip pairs improve **8/48→28/48**; complete fact×order quadruples **4/24→14/24**. Registered pilot criteria **all pass**.
- **BILINGUAL GAP:** EN 30/48→**44/48**; HE 22/48→**30/48**. Complete quartets EN 3/12→10/12; HE 1/12→4/12. The previously observed fixed language-specific answer-digit collapse is removed by architecture, but real bilingual priority understanding remains unproved.
- **REGRESSION/SAFETY:** dependency-blocking family drops **10/16 frozen→8/16 trained (chance)**. Raw learned outputs still incorrectly authorize some unapproved mission changes (4/8) and forbidden expansions (2/8). External owner/scope guard corrects six additional learned errors, raising trained score to **80/96**, and is NOT counted as learning. Frozen+guard 62/96.
- The frozen matched baseline uses English **Yes/No** output tokens for Hebrew cases and is not an exhaustive strong baseline. Option order invariance is built into the scorer; it **cannot** alone establish semantic competence. Gold labels and scenarios remain research-authored and templated, not independently judged; purported heldout user policies share the same generic preference text across identifiers. No real user personalization, source version changes, human-reviewed reasoning or autonomous planning established.
- **Decision: PASS-to-validation only; do NOT deploy or merge into model-internal SELF; do NOT immediately sweep higher ranks.** Proposed V8.5: entirely new independently authored bilingual project/policy evaluation with actual differing preference policies, varying evidence provenance, source validity and decision families; repeat multiple head seeds, investigate Hebrew/dependency, and compare to stronger explicit policy/graph controls without test-tuning.

## 2026-10-11 — COMPASS V8.5 corrected exact personal model sizes

- User clarified that the requested **three sizes refer specifically to trainable parameters in the PERSONAL COMPASS model**, not to the frozen LM backbone and not to a sweep over arbitrary MLP head widths. We preserved original personal capacity levels: **7,680, 30,720 and 122,880** trainable weights. Shared LM stays frozen `HuggingFaceTB/SmolLM2-360M-Instruct`.
- New experiment branch `experiment/compass-v8-5`, with [protocol](experiments/compass_v8_5/EXPERIMENT_V8_5.md), [fictional bilingual test cases](experiments/compass_v8_5/fixtures_v85.py), [dataset validator](experiments/compass_v8_5/validate_v85.py), [frozen shared embedding cache](experiments/compass_v8_5/cache_v85.py), [parameter-accurate personal model](experiments/compass_v8_5/matrix_v85.py), [full matrix aggregator](experiments/compass_v8_5/aggregate_v85.py).
- Exactly two bias-free 960×rank learned matrices implement one shared option scorer; rank 4/16/64 produces exactly `2 × 960 × rank` personal parameters; **shared 360M backbone does not count**. This differs from V8.4's earlier 3,848-weight head; do not use past V8.4 accuracy as a direct capacity control, though it motivated option-based scoring.
- New authored synthetic bilingual study contains **two genuinely different fictional approval policies** (rapid vs thorough), two preference-sensitive decision families, and a new source-retraction family held out of train/development. Train pool **768**, training **640**, development **128**; test pool **432**, evaluate **144** locked examples from two unseen projects with opposite approved policy modes. Test includes 72 fact-state pairs and 36 complete four-way groups.
- Single expensive CPU feature pass cached and shared across **3 sizes × 3 seeds (11,19,37), nine training jobs**, with equal epochs/updates, only development-driven checkpoint selection. External scope/approval guard scored separately. Gold labels are synthetic and are not independently human reviewed; exactly same group-family repetition remains a limitation.
- [GitHub Actions run 38108048987](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38108048987) launched via workflow on main, checking out V8.5 experiment branch. **Preflight PASSED**, frozen shared embedding job in progress at logging time; nine-arm results **not yet available**. Do not state a capacity winner or generalization success before all nine artifacts and full aggregate pass.
- Native SELF/5M and production remain untouched. Compute matched for optimizer steps, not exact FLOPs.

## 2026-10-11 — V8.5 failure in evaluation schema; fixed owner-preference target and reran nine arms

- [GitHub Actions V8.5 #38108048987, attempt 1](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38108048987) **FAILURE**: dataset preflight passed, expensive frozen 360M context-embedding cache succeeded and artifact `compass-v85-frozen-embeddings` (ID **11691476708**) was preserved; **all 9 personal-ranker matrix jobs failed at first development-set metric computation** before completing epoch 1, with `KeyError: 'project_id'`. Aggregator skipped; **NO model comparison results from attempt 1**.
- Confirmed all nine share the same coding problem: V8.5 per-case output had `project` but the inherited V8.2 paired-metric scorer required `project_id`. Added the missing canonical field, without altering fixtures or cached features.
- Additionally audited the new genuine fictional *rapid vs thorough* preference logic and found an insidious target mismatch. The borrowed V8.4 quartet loss assumed canonical **state0 always prefers choice0**, whereas for thorough profiles' newly introduced preference-sensitive families **state0 must prefer choice1**. Replaced the inherited loss/quadruple construction with `personal_train_quads` + `personal_pair_loss` using each canonical quartet's **actual oracle labels**. Enforces opposite labels under fact reversal and option swap. Prevents quietly teaching the model the wrong owner preference.
- Patched the isolated `experiment/compass-v8-5` branch at commit **43e08b12f02feb18cdec60495c3e8a5f72bf59e0**, then successfully requested GitHub **rerun of failed workflow jobs**. Attempt **2** reused earlier passed preflight/cache, started all nine arms and will aggregate only once all nine succeed.
- No V8.5 performance claim until JSON/checkpoint artifacts and the nine-arm aggregate exist. The user-provided three exact personal sizes remain **7,680 / 30,720 / 122,880** weights on the SAME frozen 360M LM; separate native SELF/5M and production unchanged.

## 2026-10-11 — V8.5 completed all nine exact personal sizes

- [GitHub Actions run 38108048987](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38108048987) **SUCCESS** at attempt 3, following technical repairs. The synthetic dataset/authorization preflight, frozen shared 360M encoder cache, all nine rank×seed head trainings, and the nine-arm result aggregator passed. Research report: [FINDINGS_V8_5.md](experiments/compass_v8_5/FINDINGS_V8_5.md). Original full nine-arm aggregate: GitHub artifact `compass-v85-three-sizes-summary` ID **11691422538**, plus nine separate raw JSON + personal-model weights artifacts.
- The user's **EXACT requested trainable PERSONAL model sizes** (not frozen LM weights) are **7,680 / 30,720 / 122,880**, ranks **4 / 16 / 64**, each with **3 seeds (11, 19, 37)**. Train 640 synthetic cases; dev 128; locked synthetic test 144 from two fictitious projects (72 EN/72 HE); all arms use one identical pretrained frozen representation cache. Two genuinely different fictional rapid/thorough policy texts but only one held-out test project for each policy.
- **Raw head mean /144 across 3 seeds:** rank4 121.33 (**84.26%**), rank16 122.67 (**85.19%**), rank64 124.00 (**86.11%**). All nine pass the predeclared synthetic accuracy, fact reversal and quartet thresholds; matched frozen control is only 64/144, below 72/144 constant-choice baseline and therefore a weak comparator. Largest-vs-smallest margin is only **2.67/144 = 1.85 percentage points**, overlapping seed outcomes and no established material parameter-size advantage.
- **Crucial generalization failure across ALL nine experiments:** entirely new heldout source-withdrawal family stays **8/16 (50%)** in every arm. Inspecting original individual decisions for rank4/seed11, rank16/seed11 and rank64/seed37 shows every source-valid state correct (**8/8**) and every withdrawn-source state wrong (**0/8**); model preserves choice under revoked evidence. Thus STOP any autonomous use without hard source validity enforcement. Novel action-family failure exists regardless of number of personal weights.
- Mean fact-reversal pairs/72: rank4 **50**, rank16 **50.67**, rank64 **52**. Mean complete four-way groups/36: **25**, **25.33**, **26**. Hebrew test mean/72: **57.33**, **58**, **58.67**; English **64**, **64.67**, **65.33**. Heavy bilingual and scenario-template limitations remain.
- Attempt 1: matrix training aborted at inherited metric-field `KeyError: project_id`. Fixed and discovered a second substantive error in original borrowed loss (`state0` assumed option0 even for `thorough` preferences); repaired to actual policy-specific gold (commit `43e08b12f02feb18cdec60495c3e8a5f72bf59e0`); all nine arms reran successfully at attempt 2 **without re-extracting the frozen cache**. Aggregator count typo fixed at `76faca2cfd4b459d4908e29059bb56f02c161411`, then only aggregate job reran and succeeded at attempt 3.
- **Decision:** Experiment complete; promising synthetic familiarity, **NO-GO for real decision autonomy or production**. **7,680 trainable personal weights is the efficiency reference** while 30,720/122,880 remain research controls. Do not merge into native model-internal SELF. Next: independent bilingual human-validated cold dilemma families, policy-only flips on same project, withdrawn authority/source provenance, stronger deterministic and language-calibrated frozen controls before choosing parameter size or deploying.

## 2026-10-11 — V8.6 source invalidation and policy-only cold pilot launched

- Context: all nine V8.5 exact personal-weight runs completed; **every one failed the previously unseen source-retraction family (8/16)**, irrespective of learned weights 7,680, 30,720 or 122,880. No evidence that simply increasing personal weights cures source revocation; synthetic familiarity scored 84–86% overall. Full results in [V8.5 findings](experiments/compass_v8_5/FINDINGS_V8_5.md).
- Implemented isolated `experiment/compass-v8-6` with [locked protocol](experiments/compass_v8_6/EXPERIMENT_V8_6.md), [fictional bilingual data](experiments/compass_v8_6/fixtures_v86.py), [validation](experiments/compass_v8_6/validate_v86.py), [V8.5 frozen-cache reuse](experiments/compass_v8_6/cache_v86.py), and [7,680 learned-weight pilot](experiments/compass_v8_6/pilot_v86.py).
- Two new source-validity training families: expired certificate and superseded report. Two NEW test-only source validity families: withdrawn delegated authority and conflict between sources; **32/176 locked test items**. Previously exposed V8.5 `source_retracted` remains a **diagnostic, not a clean holdout** (16 items).
- **16 policy-only paired interventions** on the same fictional project, objective, candidate actions, state=0 neutral evidence and option order, changing only the authorized `rapid`↔`thorough` priority text (plus hidden version/IDs). Important: do NOT invert state1 preferences when binding evidence already requires a choice.
- Frozen backbone stays `HuggingFaceTB/SmolLM2-360M-Instruct`; V8.5 archived features reused, not trained. Training pool 960, train **800**, development **160**, test **176** (88 EN, 88 HE), plus 16 separately evaluated policy intervention pairs. Initial pilot **rank4 = exactly 7,680 personal parameters**, seed11, 14 epochs using V8.5 optimizer/scheduler hyperparameters. Bigger 30,720/122,880 remain preserved V8.5 checkpoints; no blind new 3×3 matrix before cold gate passes.
- Compare raw neural scorer with frozen per-option control and **SEPARATE explicit owner/scope + source-provenance execution rules**. A positive postprocessing result cannot be credited to neural generalization.
- [GitHub Actions V8.6 run #38110217103](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38110217103) launched. **New scenario/policy cold-test preflight PASSED**. Shared frozen input-feature extension is in progress; metrics **PENDING**.
- All data and decision labels remain research-authored fiction rather than independent human-reviewed references; not licensed to claim real-world project expertise. COMPASS remains isolated from native SELF/5M and production.

## 2026-10-11 — V8.6 FINISHED: invalid-source and policy update tests

- [Actions #38110217103](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38110217103) completed SUCCESS. The rank-4 COMPASS personal head (7,680 weights, seed 11) trained on 800 fictional bilingual examples and evaluated 176 held-out examples plus 16 same-project policy-change pairs. Detailed source audit: [FINDINGS_V8_6.md](experiments/compass_v8_6/FINDINGS_V8_6.md). Raw result+weights artifact: 11690768329.
- Model-only accuracy **148/176** vs frozen **86/176**. New source-validity families **26/32**, but the invalid-source subset is only **10/16** (below target 12/16). The previously seen `source_retracted` family remains **8/16**, with **0/8 invalid sources** handled correctly.
- Changing only the authorized personal policy in the same fictional project: **8/16** correctly flipped decisions, below the preregistered 12/16 target; both versions are correct in **8/16**, below the 10/16 target. English 6/8 vs Hebrew 2/8.
- Separate deterministic source-validity rule improves model+guard 148/176 to **162/176** using authored provenance fields, which is **not neural learning nor authenticated production source checking**.
- **Decision: NO-GO to scaling V8.6** beyond 7,680 weights, no production/SELF merge. Preserve existing 30,720 and 122,880 parameter V8.5 heads. Next address reliable external source/provenance gating, same-project policy-pair learning and independently verified bilingual holdouts.

## 2026-10-11 — V8.7 owner-only policy pair learning launched

- User approved proceeding after V8.6 failed its raw policy-only counterfactual gate (8/16) and failed strong invalid-source constraints. V8.7 changes the **training objective**, not frozen language model, personal weight count or production.
- Implemented on isolated branch `experiment/compass-v8-7` the [preregistered protocol](experiments/compass_v8_7/EXPERIMENT_V8_7.md), [same-project paired fixture](experiments/compass_v8_7/fixtures_v87.py), [fixture/source contract validator](experiments/compass_v8_7/validate_v87.py), [fail-closed source lookup contract](experiments/compass_v8_7/trusted_sources_v87.py), [V8.6 cache extension](experiments/compass_v8_7/cache_v87.py) and [7680-weight policy-pair fine-tuner](experiments/compass_v8_7/train_v87.py).
- Restore EXACT `compass-v86-r4-s11.safetensors` rank4 / **7,680 personal parameters** from V8.6 as baseline. Train the identical head on 800 old authored bilingual task cases plus **80 new paired policy-only interventions** where same project, evidence, choices and order remain identical, while owner-authorized `rapid`↔`thorough` preference text and gold decision flip. Keep 160 development tasks and **16 separate development policy-only pairs**; choose epoch by development policy both-correct first, full contrast groups second, accuracy third. Frozen common 360M remains unchanged, and V8.5/V8.6 old test results stay exposed diagnostics only.
- Newly created COLD policy-only **maintenance_style** and **report_delivery** families: **16 counterfactual pairs**, not present in train/dev; show both-correct rate, actual policy-driven flips, Hebrew/English separately. Also compare on already-exposed V8.6 16 policy pairs and the old 176 tasks, with explicit diagnostic rather than pristine heldout labels.
- Separate execution-side source validity contract: simulated trusted registry identity, project, verified issuer, current version, revocation, unresolved conflicts and effective times; reject invalid/unknown source *before execution*. This is NOT trained model competence and does not implement cryptographic verification, live authenticated retrieval, or production authorization.
- [GitHub Actions #38111231513](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38111231513) launched: **preflight PASS**, frozen cache reuse+new embedding stage in progress when noted; NO model-learning results yet. Gate for larger rank/multiple seeds is cold >=12/16 both correct, >=12/16 choice flips, Hebrew >=5/8, no exposed-task regression below 140/176, and actual improvement relative to unchanged V8.6 weights. Synthetic author-created gold is not independent human adjudication. Native SELF/5M and production remain untouched.

## 2026-10-11 — V8.7 completed: policy-pair training regresses on new cold families

- [Actions run 38111231513](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38111231513), **attempt 2 SUCCESS** after a one-line-family bug fix. Attempt 1 failed before training: `eval_full` returned 3 results, but `train_v87.py` unpacked 2 in two development callsites. Fixed at commit `e1ad3ba5e494b074b504978157edc4d256146306`; reran **only failed training job** using the already passed reusable 360M embeddings. Original JSON and 7,680-weight checkpoint saved in artifact **11691891392**. [Full findings](experiments/compass_v8_7/FINDINGS_V8_7.md).
- Fine-tuned the SAME **7,680-weight** V8.6 rank4 personal scorer, seed11, on **80 train + 16 development** direct same-project policy-only preference pairs in addition to existing synthetic fact-state training. Selector chose **epoch 17** by development (policy pairs **16/16**, old tasks **160/160**). Test is two novel authored policy families (`maintenance_style`, `report_delivery`), **16 cold pairs**; all labels created by the research workflow, not external/human gold.
- **EXPOSED V8.6 diagnostics improved slightly:** old ordinary tasks **148/176 → 150/176**, old policy pairs both-correct **8/16 → 10/16**, old raw policy-choice flips **8/16 → 10/16**. Original withdrawn-source family still only **8/16**, with **0/8** invalid source cases correctly stopped.
- **NEW cold policy evaluation REGRESSED:** same original V8.6 head **4/16** both decisions correct and **8/16** policy flips; after V8.7 fine-tuning **2/16** both correct and **4/16** flips. English cold **0/8** pairs both correct, **0/8** flips before and after; Hebrew pairs both correct **4/8 → 2/8**, flips **8/8 → 4/8**. Targets ≥12/16 both and ≥12/16 flips, Hebrew ≥5/8: **FAIL**.
- **Research verdict NO-GO.** Despite perfect development and improving old decisions, broader policy generalization deteriorated. Do NOT automatically expand to 30,720 or 122,880 weights, deploy, or merge into native SELF/5M. Preserve checkpoint and work next on representation probes, stronger explicit owner-policy and source-provenance baselines, independently authored bilingual policy scenarios and controlled policy-only counterfactuals.

## 2026-10-11 — V8.8 completed — active advisor all-A bias

- [GitHub Actions run 38112408784](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38112408784) **SUCCESS** (3/3 jobs); scientific gate **FAIL / NO-GO**. New research-authored bilingual cold policy test: 16 same-project policy-only pairs (8 EN/8 HE). [Full audited findings](experiments/compass_v8_8/FINDINGS_V8_8.md).
- On new cold pairs, exact same 7,680-weight V8.6 checkpoint alone **6/16 both-correct, 8/16 flips**; injected active teacher text without weight training **0/16, 0/16**; 360M teacher's own vote **0/16, 0/16**; matched 18-epoch text-only paired training **4/16, 4/16**; matched teacher-assisted training **2/16, 2/16**. Hebrew both-correct **0/8** in either newly trained arm. Active-teacher arm regressed on previously inspected general benchmark **138/176 vs 150/176** for control.
- Independently inspected original teacher artifact **11692074929**: **all 144 teacher decisions selected canonical Action A**, regardless of user policy; all **72 Hebrew generated explanations identical**, just `אתה יועץ תיעדוף בפרויקט`. Teacher A/B confidence ~0.68–0.93 despite zero measured within-project policy flips. This exposes **teacher prompt/forced-choice interface positional bias, ineffective Hebrew explanation**, not a validated large-model reasoning benefit. Teacher reasoning/embeddings took ~1486 seconds on frozen 360M.
- Preserved artifact **11692412450** has JSON per-case decisions and both 7,680-weight checkpoints; **no 30,720/122,880 sweep or SELF merge**. Next must validate/calibrate teacher itself on fresh policy reversals, balanced A/B positions, validated structured answers/abstention and external source verification before trying a feedback loop or expanding personalized parameter count. All fixtures synthetic, not independently adjudicated.

## 2026-10-11 — V8.9 English-only active feedback gate launched

- User simplified the proof of concept: **English ONLY**; Hebrew and stronger base LMs can wait until the active large+small architecture proves itself. They also want a substantive small-model size sweep (7,680 / 30,720 / 122,880 / 491,520), but not before active advisor is diagnostically sound, and the last new size must be independently trained rather than evaluated randomly initialized.
- [Registered protocol](experiments/compass_v8_9/EXPERIMENT_V8_9.md), [English-only fictional holdout generator](experiments/compass_v8_9/fixtures_v89.py), [teacher + independent trained small-model feedback](experiments/compass_v8_9/teacher_v89.py), [label-independent evaluator](experiments/compass_v8_9/score_v89.py). Implemented on branch `experiment/compass-v8-9`.
- Correction to V8.8: no canonical A/B vote; compare **per-option Yes/No next-token logit margins**, check option-order swap invariance, produce a distinct English one-sentence analysis (max48 tokens), and measure teacher after **one bounded SMALL→LARGE feedback message** with unchanged V8.6 personal rank4 7,680 trained-weight checkpoint. **No backpropagation** and **no test labels in teacher prompt**. Inference and preference authority remain separate. Synthetic provenance/trust contract tested in preflight.
- New artificial EN decision pairs: 4 unseen scenario text families × 2 previously used fictitious test projects × 2 states (neutral policy decisive vs binding evidence overrides) × 2 option orders = **32 paired interventions**, 64 original+policy-changed situations. Neutral 16 pairs must flip; binding 16 must **not** flip. Rule to expand parameters pre-registered in protocol (both neutral and binding correct ≥12/16; order ≥28/32; bounded feedback raises neutral ≥2 without worsening binding; valid textual explanations ≥48/64). All gold created by research script rather than third party.
- [GitHub Actions V8.9 run 38114546492](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38114546492) started. English case audit **PASS**; CPU-only active teacher and feedback run in progress; scientific outcome **PENDING**. Not deployed, merged into native SELF/5M, or used on actual project/user data.

## 2026-10-11 — V8.9 completed — English active feedback NO-GO

- [Run 38114546492](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38114546492) **SUCCESS attempt 2** (attempt 1 fixture generator/reversed TypeError fixed). All four conditions ran on **32 synthetic English pairs** / 64 before-and-after policy states. [Full findings](experiments/compass_v8_9/FINDINGS_V8_9.md); artifact ID **11692493973** preserves each teacher generation, action-pair logits and small feedback.
- **Both decisions correct (neutral preference MUST flip / binding constraint MUST NOT):** unchanged 7,680 head **0/16, 16/16**; frozen 360M teacher direct **4/16, 12/16**; teacher after English explanation **2/16, 10/16**; teacher after one numerical untrusted personal-head feedback round **2/16, 11/16**. Choice-order semantic stability respectively **32/32**, **28/32**, **22/32**, **19/32**. Feedback worsened direct teacher on both neutral and binding; pre-registered research verdict **NO_GO_TO_SIZE_SWEEP**.
- Mechanical English reason non-echo check **64/64** but only **38 unique strings**, no proof of accuracy/faithfulness. Direct 360M format no longer all-A as in V8.8; still lacks reliable policy counterfactual flips, and feedback causes extra ordering sensitivity.
- Most important architectural correction from user: permanent **per-user trainable neural module attached to every running large-model invocation**, with *internal* conditioning of frozen backbone processing, not just one optional end-of-run ranking score or added chat-note. Future experimental implementation to test personal adapter/gated residual in selected frozen transformer blocks, with explicit owner-preference state, controlled ablation against text-only and deterministic priority-rule baselines. This user vision is not yet built in V8.9.
- No parameter-size sweep, no retraining, no native 5M SELF merge or production integration; V8.9 actual run is a diagnostic negative result. Next should isolate causal influence of personal module on hidden states and decisions before selecting personalized module capacity, with English-only new independent synthetic and eventually human-checked inputs.

## 2026-10-11 — V9.0 launched — conditional hidden-state personal adapter

- User reaffirmed the central research goal: REPLACE recurring textual personal context with a **genuine user-specific neural change inside a shared large LM**, without re-training its backbone for each user. The per-user module must be attached to the shared LM on every request and functionally affect its interpretation/inference. Previous COMPASS V8.5–V8.9 measured ranking and textual feedback, NOT this target.
- Implemented `experiment/compass-v9-0`: `personal_gate_v90.py` owner-state-conditioned **live forward hook on the last decoder block output before RMSNorm**; `train_v90.py` caches owner-blind block activations to learn a rank16 residual, and subsequently verifies the exact LIVE hooked model computes the same Yes/No output logit margins for both owner states as the cached path. Encoder/LM weights frozen, intended bit-identical before/after.
- Rank16 personal gate = `down(960→16) + owner(1→16) + up(16→960)` **30,736 shared learned parameters**. Individual users hold a **1D approved preference state** changed with zero per-user gradient steps. This pilot does NOT yet instantiate separate trained neural weights for each user. The owner priority string is ABSENT from the neural-gate prompt; it appears only in the independently measured text-injection baseline. A user-specific neural model, multi-layer attention coupling and full continuous personalization remain future architecture.
- Training uses English fictional owner-priority policy pairs and binding evidence from prior TRAINING synthetic profiles, with internal development selection; NEW V9.0 cold family texts `change_review`, `data_backfill`, `performance_investigation`, `handover_process` × 2 previously used fictional test projects × neutral/binding evidence × swapped option order = **32 paired interventions** (16 should flip, 16 should stay). Test gold excluded from optimization. Comparators: frozen owner-blind, frozen text-injected, transparent lexical deterministic baseline. Labels and examples are researcher-authored and not human-reviewed.
- Locked effectiveness gate: >=12/16 correct new neutral policy-change pairs, >=12/16 new binding unchanged pairs, > text-injection baseline on correct cold pairs; plus *mandatory* live-hook equivalence tolerance 0.001, nonzero owner-state effect, frozen backbone unchanged. No size sweep or native SELF merge until mechanism is validated.
- [GitHub Actions V9.0 run #38116261352](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38116261352) triggered; **preflight PASS**, live CPU LM experiment in progress at this writing, no performance results yet. [Pre-registered protocol](experiments/compass_v9_0/EXPERIMENT_V9_0.md).

## 2026-10-11 — V9.0 COMPLETED — neural hook works, zero cold preference flips

- [GitHub Actions #38116261352](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38116261352) **SUCCESS, attempt 1**, both jobs. Full source-authored result file: [FINDINGS_V9_0.md](experiments/compass_v9_0/FINDINGS_V9_0.md). Research artifact **11693970107** contains `compass-v90-personal-neural.json` and the shared 30,736-weight trained module `compass-v90-personal-neural.safetensors`.
- This was the **first genuine in-forward Transformer intervention** in the COMPASS V8–V9 comparison: a rank16 shared gate operating on **last decoder block output** before final norm/LM head of the frozen `SmolLM2-360M-Instruct` (361,821,120 backbone weights), with a hot-swapped explicit user preference scalar (`rapid=-1`, `thorough=+1`) **absent from the neural prompt**. It is NOT a separately learned per-user neural network yet; one shared adapter was trained once; per-user gradient steps 0.
- **Engineering PASSED:** Real live hook and cached block-vector + RMSNorm+LM-head margins numerically identical for BOTH personal states (within 0.001). For exact same tokenized prompt, tested candidate Yes/No margin: rapid **0.39679**, thorough **0.27843**, absolute effect **0.1183586**; backbone frozen (checked q_proj tensor exactly unchanged). The decision chosen between TWO actions remained the SAME for both states, though per-candidate logits differed.
- **New 32 research-authored synthetic cold paired interventions** over four previously unseen task TEXT families: neutral 16 require policy-switch decision reversal; binding 16 require staying with mandatory evidence. **Neural module 0/16 neutral both correct and 0 actual flips; 16/16 binding both correct; 16/32 overall**. Frozen base identical pair results (0/16, 16/16); simple injected preference text **3/16 neutral, 16/16 binding**; researcher-authored lexical rule **12/16 neutral, 16/16 binding**. Neural action-order consistency **28/32**, lower than frozen **32/32**. Training best development 14/24 at epoch6 of 60. No evidence of effective personal adaptation despite actual hidden/logit changes.
- **Scientific verdict NO-GO** for deployment, native SELF merge, and rank expansion. Next should investigate state×action *interaction*, explicitly contrastive preference-switch training, earlier-layer/multi-layer personal conditioning and richer user-specific state; keep frozen backbone + text + transparent rule controls and new heldout evaluation separate. A 1D user scalar is not the user's ultimate vision of a durable personalized neural model. All texts/labels authored by experimenter; no external human evaluation.

## 2026-10-11 — V9.1 launched — earlier neural conditioning and explicit paired preference contrast

- User approved next SELF/COMPASS research step after V9.0 **engineering pass but scientific 0/16 preference-flip failure**. User's target remains a continuously attached per-user neural component that changes the shared frozen LM's actual forward processing without repeatedly including owner context in text or retraining the full LM per user.
- Implemented on isolated branch `experiment/compass-v9-1`. [Registered V9.1 protocol](experiments/compass_v9_1/EXPERIMENT_V9_1.md), files `gate_v91.py`, `fixtures_v91.py`, `train_v91.py`, triggered [GitHub Actions run #38117696748](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38117696748). Scientific results pending at logging.
- Instead of last layer V9.0 owner residual, V9.1 injects a **rank16 4-slot state-conditioned real hook after third-from-last decoder block**, leaving **two downstream attention/MLP blocks** to process its effect. Shared gate learned parameters **30,784**, frozen SmolLM2 360M base. Hot-swap owner state `rapid=[1,0,0,0]` vs `thorough=[0,1,0,0]` WITHOUT owner policy in input tokens; 2 of 4 channels actually used, only 1 independent binary distinction. Not falsely claimed to be a trained personal model for each user.
- New supervised objective trains on identical facts/options under original and flipped explicit owner state, with signed correct choice margin plus **policy reversal contrast** for neutral situations and stable-choice penalty under binding evidence. 8 CPU epochs, 36 research-authored TRAIN pairs/epoch, best epoch selected by independent TRAIN-only development pairs. Frozen base has no trainable parameters; training gradients traverse 2 downstream decoder blocks to reach the personal gate.
- New cold synthetic English cases in **4 fictional project identities not previously examined**, 4 new task families `technical_release`, `catalog_consistency`, `support_handoff`, `quality_inspection`, 2 evidence regimes, swapped option order = **64 owner-policy intervention pairs**, **32 neutral MUST flip**, **32 binding MUST NOT flip**. All test data and labels still written by experimenter, NOT independent human adjudication.
- Matched controls: no-owner frozen base, English user-policy text in frozen prompt, transparent keyword priority/constraint baseline; check actual downstream hidden activation, logit change, original frozen projection unchanged, owner words absent from neural prompt, action-order invariance. Target >=24/32 neutral both correct, >=24/32 binding both correct, > both text and rule overall, >=56/64 semantic-order invariance. Failure is NO-GO to module-size sweep/native SELF merge/production.
- [V9.0 findings](experiments/compass_v9_0/FINDINGS_V9_0.md) archived unchanged. This experiment is a limited first neural interface study, not deployed continuous personalized service.

## 2026-10-11 — V9.1 COMPLETED — earlier neural gate, zero out-of-distribution owner preference reversals

- [Final successful GitHub run #38118071115](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38118071115), both preflight and training jobs PASS; training ran ~2120 seconds / 35.3 minutes on GitHub Actions CPU, after earlier infrastructure-only failed launches. [Full findings](experiments/compass_v9_1/FINDINGS_V9_1.md), artifact **11695360103** contains full per-case predictions and shared gate weights.
- V9.1 does cause TRUE early inner-Transformer change: injected after third-to-last decoder block, two subsequent blocks run, actual downstream hidden max difference **0.037725449**, live Yes/No logit margin owner-state difference **0.026943207**, frozen backbone spot-check unchanged. Shared learned rank16 gate **30,784** weights; per-user "4D" state actually carries **only one binary preference** (rapid vs thorough). Per-user gradient steps **0**; **not** user-specific learned weights.
- **32 new synthetic neutral owner-preference twin pairs both correct:** frozen **0**, internal neural gate **0**, text preference injection **3**, experimenter-authored lexical rule **24**. Actual choice flips with neural gate **0/32**. **32 binding evidence pairs both correct:** frozen **32**, neural **32**, text **32**, lexical **24**. Total 64 twin both-correct: **32 / 32 / 35 / 48**, respectively. Semantic choice-order consistency: neural **64/64**, frozen **62/64**, text **61/64**, rule **48/64**; neural consistency is unhelpful because it always picked the comprehensive choice across all new cases. Selected train epoch **3/8**, best developer pair both-correct **6/16**. **Scientific NO-GO**, registered neutral threshold 24/32 failed, neuro worse than text/rule. Frozen full model and native SELF main unchanged.
- Interpretation: engineering of a small internal personal neural connector is validated, but V9.1 does **not** learn user intent, governing project principles or preference-sensitive action choices. Hyperparameter/rank sweep not justified. No independent human-reviewed labels, and no full trained personal network.
- **Updated north star from user:** separate the question *what the personal/project model learns* (abstract purpose, rationale, corrections, global vs project principle) from *how that state affects the larger frozen LM*. Prior V8–V9 mainly optimized the second with only a 1-bit owner state. Next proposed **concept-transfer study V9.2**, before any further coupling changes: user-general COMPASS + project COMPASS learned from multiple fictional project histories, held-out novel dilemmas including implementation conflicting with intent, separate state update and approval/source-truth tracking, and isolation across projects. Matched frozen, textual retrieval, user-only, project-only, both, and deterministic symbolic baselines. **No V9.2 run has been launched by this note.**

## 2026-10-11 — V9.2 launched — hierarchical concept representations before LM coupling

- User approved V9.2 after recognizing V9.0 and V9.1 did prove in-transformer hidden/logit changes but **0** cold preference reversals. Change the research question to whether a small neural model learns **user-global principles and project-specific purposes** from corrections, rather than manipulating backbone logits before proving it holds useful representations.
- Code in `experiment/compass-v9-2/experiments/compass_v9_2/study_v92.py`, [full preregistration](experiments/compass_v9_2/EXPERIMENT_V9_2.md). Live reproducibility [GitHub Actions #38120988665](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38120988665) initiated. **At this writing scientific GitHub results pending; do not claim success.**
- Design: two explicitly different concept types (user working practice, project ultimate purpose/compatibility obligation), 1 user stream + 2 independently processed project streams per episode. Input = sequence of 3 English correction/history statements, with 45% of streams showing a newer correction overriding prior events. Shared 24D word embedding + two separate 28-hidden GRUs + classifiers; training 600 steps, synthetic 32-episode batches, best model selected only on TRAIN-template development examples. Different **held-out paraphrase templates** in 600 NEW cold synthetic episodes; synthetic oracle concept labels never inside text.
- Controls: constant no memory, user-only learned state, project-only learned state, both jointly, transparent full-history lexical keyword reader, known synthetic concept oracle. Also 128-case causal swap of project B history to test project A and user-global state isolation. Each inferred concept can drive a *trusted routed deterministic decision interface* for novel task descriptions in swapped order, but **the learner does not consume those dilemma descriptions**. This is a **concept classification probe**, NOT new-dilemma free-form reasoning or frozen-360M neural conditioning.
- Pre-registered GO only if cold overall concept ≥85%, corrected ≥80%, separating opposite project concepts ≥80%, hierarchical better than both single-source baselines and textual keyword reader, project-B intervention isolation 100%. A local draft test (not the exact GitHub preregistered script) yielded hierarchical ~78.4% vs transparent textual ~94.7%, suggesting substantial failure risk; GitHub version has STRICTER train/development vs cold paraphrase separation and its results govern decisions.
- No native SELF 5M, production, server, owner profile or existing COMPASS checkpoints modified; no LM backbone trained. A passing small scripted case would still need a proper text+retrieval frozen-LM baseline and independently adjudicated natural project histories before substantive claims of abstract intent learning.

## 2026-10-11 — V9.2 COMPLETED — hierarchical learning under cold paraphrases

- [GitHub Actions run #38121067104](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121067104) **SUCCESS**, all experiment steps complete. Initial run #38120988665 failed on **numpy** missing during safetensors serialization; dependency-only fix followed by full rerun, no changes to scientific test. [Complete audited findings](experiments/compass_v9_2/FINDINGS_V9_2.md); artifact **11695631053** includes full JSON, learned shared weights and preflight.
- V9.2 changed focus from *neural coupling* to *learning principles from past correction histories*. Shared tiny English-only hashed-word embedding plus two GRU streams (USER vs PROJECT) and separate heads, **107,492 total trainable shared weights**, 600 steps and 600 cold episodes. At 45% of event streams, a late correction replaces a principle. Held-out rewritten cold HISTORY sentences were excluded from train and dev selection. Only 2 researcher-scripted classes per dimension: global user's verification-vs-prototype work style; project purpose-vs-compatibility priority. No 360M model executed, no new LM internal coupling and no per-user weight training.
- **Cold concept identification accuracies**: no-memory 50.56%, user-only 46.22%, project-only 73.17%, **hierarchical 68.83%**, full-text transparent keyword parser **95.06%**, known-label oracle 100%. Cold after-correction neural **69.99%** vs parser **95.42%**. Opposed A/B project separation **67.84%** vs parser 100%. Global user branch **230/600 = 38.33%**, projects A **515/600=85.83%**, B **494/600=82.33%**. Thus the major weakness is **global user paraphrase generalization**, not merely isolation or compute.
- Program-level isolation **PASS 128/128** on counterfactual project-B history swaps: user and project A predictions unchanged, B follows copied A's history. But this is structural, not evidence of accurate learned concepts.
- Best dev **100% at training step 50** (!) but cold **68.83%**, exposing template memorization / lexical OOD. **Five of six preregistered effectiveness gates failed**; only isolation passed. Final research verdict **NO_GO_TO_LM_COUPLING**. Do not couple these weak latent states to the 360M model, deploy, or blindly sweep ranks/parameters.
- Critical caveat: new-dilemma texts were *not* read by neural module. A deterministic trusted-scope rule translated predicted 1-bit user/project concepts to actions; this experiment is ONLY a controlled latent-concept classifier and cannot demonstrate reasoning over new tasks, independent human intentions, durable online learning or strong memory-vs-RAG comparison. The textual comparator is a simple last-message keyword parser, not a strong LLM+retrieval baseline.
- Recommended research successor: use frozen **pretrained semantic encoder** to improve meaning-preserving paraphrase generalization, separate learned global/project memory updaters and corrective changes; test on independent intent-vs-implementation decisions and compare with strong text/RAG LM baseline. Measure per-user/profile privacy, cross-project transfer without leakage, corrections, and cost. Keep data split blind and preregister before running. Native SELF main remains unchanged.

## 2026-10-11 — V9.3 launched — frozen semantic pretrained encoder and incremental user/project memory

- User approved continuing from V9.2's cold-paraphrase NO-GO (**68.83% hierarchical vs 95.06% full-history lexical baseline, global user branch 38.33%**) to isolate semantic generalization rather than blind parameter scaling. Objective remains learning global-user and project-level principles from corrections, ultimately conditioning an otherwise frozen large language model through per-user neural modules.
- Implemented experimental branch `experiment/compass-v9-3`. [Study preregistration](experiments/compass_v9_3/EXPERIMENT_V9_3.md), [source](experiments/compass_v9_3/study_v93.py), [GitHub Actions run #38121831473](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121831473) launched. **No verified scientific outcome at log time.**
- NEW frozen English semantic encoder `sentence-transformers/all-MiniLM-L6-v2` (384D, no gradient), connected to a trainable common projection + separate user GRUCell and project GRUCell memory updaters and separate concept heads. Each feedback statement updates latent memory in a single forward call; project A and B share encoder/updater weights but have separate transient hidden states. No persistent user checkpoint, no personal per-user training, no frozen-360M LM calls or new actual Transformer hooks in this experiment.
- Input cases still synthetic, binary working-style / project purpose-vs-contract principles and simple late corrections. 600 supervised training steps seed 11, development uses only TRAIN statement templates; holdout **600 near** (V9.2-exposed) and **600 fresh far** episodes, using 16 new research-authored cold paraphrases, for each: global user, project A and B. Evaluates correct concept retrieval and revisions, opposed A/B discrimination, 128-case causal project B swap, exact equivalence of incremental RNN state updates.
- Registered controls: no-memory, user-only, project-only, trained hierarchical, transparent full-history latest-utterance keyword reader, frozen semantic latest-statement prototype retrieval, frozen semantic recency-weighted FULL-history prototype retrieval, and known-label oracle. Absolute/relative GO: near overall ≥85%, far overall ≥75%, far user ≥70%, far corrections ≥75%, far opposing projects ≥80%, far learned neural > best text/semantic baseline by **≥2 percentage points**, and perfect scope isolation and incremental update equivalence. Failure is NO-GO to wiring new memory into the big frozen LM, size sweep or production.
- Crucial limitation: NEITHER the new question nor project dilemma is semantically interpreted by the neural memory network. This test is explicitly only classifying already defined concepts from past statements, not a complete project-intent understanding system or text+RAG full frozen-LM comparison. Cold texts and gold are researcher-authored; no independent blind human test.

## 2026-10-11 — V9.3 COMPLETED — pretrained semantic features improve cold principle recovery

- [GitHub Actions successful #38121831473](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38121831473); both preflight and actual frozen-pretrained sentence encoder/online-memory training PASS. [Full study analysis](experiments/compass_v9_3/FINDINGS_V9_3.md), artifact **11695438682** with separate JSON and Safetensors.
- Frozen pretrained 384D `sentence-transformers/all-MiniLM-L6-v2` + shared **57,796** learnable projection/2 GRU memory update/2 binary concept heads, personal and project histories processed in distinct recurrent hidden states. 600 train steps, best train-template dev step50 (100% dev), 600 prior-exposed near episodes + 600 new researcher-authored far paraphrase episodes. The mini semantic encoder is pretrained and frozen; there is **NO frozen large LM call, no new neural integration into 360M, and no per-user weight training**. Numeric personal/project memory is temporary per episode, not durable profile data yet.
- **Official FAR evaluation** across 600 episodes/1,800 scoped concept labels: hierarchical neural **81.83%** (global **430/600 = 71.67%**, project A **522/600 = 87%**, project B **521/600 = 86.83%**), late corrections **81.89%**. Simple full-history text keyword reader **79.39%**, frozen last-statement embedding prototype retrieval **79.11%**, frozen weighted full-history semantic prototype **78.83%**. Gains mainly from **global user branch**: the project branches' 86.92% combined equals the frozen semantic last-message prototype baseline. Exposed prior V9.2 near templates got **94.78%**, not blind evidence.
- **Opposing A/B project principle separation FAR 73.13%**, below pre-registered **80%** threshold. Engineering stream isolation (swap project B history) **128/128** for no effect on user and A, B equals A when supplied identical history, incremental GRU scan matches state stepping. Thus pre-registered gate result: **6/7 PASS, ONE critical FAIL, final research verdict NO_GO_TO_BACKBONE_COUPLING**. Neural advantage over best simple non-LM baseline is **+2.44 percentage points**, no independent statistical or human review and no strong LLM+RAG comparison.
- Scope strictly synthetic two-class concept identification from histories; the model DOES NOT process new project dilemma questions, explain goals, distinguish real documents from inferred intentions or make free-form downstream decisions. Results are insufficient to call this genuine personal vision understanding or working long-term SELF agent.
- Next target: matched deliberately contradictory project goals and purpose-vs-implementation decisions, held-out independently authored tasks, multi-seed tests, richer cross-project contrastive state updater, plus strong frozen-LM and full-text retrieval baselines before large-model internal coupling or module-size sweep. Native SELF 5M code, old COMPASS weights and production unmodified.

## 2026-10-11 — V9.4 launched — semantic personal/project concept state to novel decision

- User approved V9.4 after V9.3 showed better held-out English concept recovery (**81.83% FAR overall; 71.67% global-user; 86.92% projects**) but failed contradictory-project separation (**73.13% <80% preregistration**) and did NOT consume new dilemma text. V9.4 directly addresses the missing causal link from *learned intention* to *correct choice in an unfamiliar natural-language problem*.
- Research branch `experiment/compass-v9-4`, [preregistered study](experiments/compass_v9_4/EXPERIMENT_V9_4.md), `fixtures_v94.py` + `study_v94.py`; [GitHub Actions run #38122865192](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38122865192) triggered. **Scientific results unknown at registration, do not claim success.**
- Stage 1: train V9.3-style tiny sequential hierarchical learned concept state from TRAIN-only English synthetic correction histories; freeze. Stage 2: new trainable SEMANTIC decision bridge receives 48D numeric user state, 48D numeric active-project state, pretrained frozen MiniLM 384D embedding of an actual NEW task, and 384D embeddings of TWO feasible action texts. No full personal history injected into this neural bridge; no task gold, principle enum, task type, binding metadata or oracle scope passed to bridge. Same task, option texts and unrelated histories under each matched intervention; only the relevant owner's history is changed. Unlike V9.3, the decision network truly sees the new TASK text and both action alternatives.
- Synthetic TRAIN 384 paired situations (8 training families); holdout **128 matched policy-change pairs** with 8 novel task families, new history paraphrases and four new project names, with equally represented neutral/binding and both action orders; changing principle must flip choice in neutral cases and must not flip under external binding facts. Train bridge 45 epochs; checkpoint from TRAIN-template development only.
- **Actual large model comparator** `HuggingFaceTB/SmolLM2-360M-Instruct`, base frozen and passes complete user and relevant project history as text, evaluates candidate Yes/No margins on a balanced **prespecified 32-pair subset**; neural scores the exact same subset separately. Full 128-pair neural results cannot be compared as if same cases as the 32-pair LM baseline. Also transparent keyword + synthetic authoritative scope/binding rule, frozen semantic concept-prototype + same rule, and no-memory baselines. Note that simple symbolic controls use oracle case scope and binding metadata (privileged, sanity upper bounds).
- Preregister gates: neutral BOTH-correct ≥60%, binding BOTH-correct ≥75%, option-order invariance ≥95%, neural beats frozen-360M full-text LM on the SAME 32 cases, and beats task-oracle-assisted keyword baseline on full 128. Failure is NO-GO to further in-LM neural coupling and parameter sweep. No change to native SELF main or production.
- The V9.4 bridge is a **separate trainable neural action scorer** working on numeric learned personal/project state + frozen task semantics; it is **NOT** the 360M model's own internal computation and does not complete the final COMPASS vision. All source texts/gold fictional and researcher-authored, no independent human holdout or genuine long-term user storage.
