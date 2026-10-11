# COMPASS V8.5 — three EXACT personal-model parameter counts

**Registered design (before results).** This V8.5 follows the **new shared option-ranker approach of V8.4**, NOT the failed late-layer LM residual approach from V8.1-V8.3. The user requested the original three personal-model sizes **7,680 / 30,720 / 122,880 trainable parameters**, not a sweep of LM backbone sizes (5M/50M/300M), nor an arbitrary 4/16/64-neuron MLP widening.

## Precise personal-model parameterization

The shared frozen language model stays **HuggingFaceTB/SmolLM2-360M-Instruct** (360M-ish **shared backbone weights not counted as personal parameters**). Each candidate is encoded independently from the project goal, approved preference and verified evidence. The **personal ranker** operates on its frozen 960-dimensional representation, computes a scalar score for each candidate and prefers the higher one.

This V8.5 implements exactly **TWO learned bias-free rank x 960 linear maps**, with scalar `score(candidate) = sum(tanh(U * LayerNorm(embedding)) * (V * LayerNorm(embedding))) / sqrt(rank)`. It yields:

| Personal rank | Trainable personal parameters | Frozen shared backbone |
|---:|---:|---|
| 4 | **7,680** | Same 360M |
| 16 | **30,720** | Same 360M |
| 64 | **122,880** | Same 360M |

Note: this ranker changes nonlinear scoring architecture compared with V8.4's original **3,848-weight** `960→4→1` head, so numeric results from V8.4 are **historic context, NOT a direct capacity control**. Fair direct capacity comparisons are the three V8.5 variants *within this study*, using the same architecture, features, data, number of optimizer updates and seeds.

## Data and holdout

- Reuse V8.2's six research-authored decision families, **add two new truly differing fictional preferences**: `rapid` favors focused quick pilot/targeted verification; `thorough` favors broader, systematic release and checks, but authoritative binding evidence/permissions override both. The new families explicitly invert the answer when the policy or evidence changes.
- Original 6 train profiles × 2 projects × 2 languages × 6 families × 2 truth states × 2 candidate orders = **576**. Add 2 preference-sensitive families = 192, giving **768 authored train+development examples**.
- Five original fictional profiles (10 projects, **640 examples**) train the heads; the sixth (2 projects, **128 examples**) is reserved for model checkpoint selection. Their projects are disjoint.
- Entire test pool has 3 different fictional profile IDs ×2 projects. Evaluate exactly **2 disjoint held-out projects**, one `rapid` and one `thorough`. Six old families + two new policy families + a **third, previously unseen source-withdrawal family** test whether withdrawn evidence flips a recommendation. **144 locked test decisions**, 72 English and 72 Hebrew, or 36 four-way state×order groups / 72 fact-flip pairs. The newly unseen source family has only **16** of the 144 test records; it appears nowhere in train or development.
- Source and labels are all written by the experimenter and are **NOT independently reviewed or human adjudicated**. This is a synthetic structured test, not validation on actual user account/work.
- The test contexts include fictional project goals, task evidence and approved preference, but the tokenizer prompts never include the oracle's gold choices or hidden case/state IDs.
- The external trusted-owner/approved-scope guard is a separate non-neural postprocessor, reported independently.

## Shared embedding compute and nine-arm trial

Frozen 360M feature extraction is the expensive CPU step. One **prepare** job extracts and saves embeddings and the *identical prompt frozen Yes-No scorer baseline* for all 640 train +128 development +144 locked test cases, duplicating no cached evidence/action pair. Save a frozen feature artifact with a dataset fingerprint, project split identities, language token maxima and options; train jobs verify fingerprint and use NO backbone inference.

A GitHub Actions matrix trains **3 ranks × 3 deterministic seeds (11,19,37) = 9 arms** (widths are rank of personal matrices, not number of LM parameters). Same optimizer schedule: AdamW, lr=0.001, batch=20 quadruples, 14 epochs. Use only the 128-case development split for best-epoch selection based on complete quadruple success and accuracy. Score a single locked test once per rank×seed, retain **every raw choice**, same frozen comparator, per-language and per-personal-policy results and all 72 fact-flip pairs; report isolated 16-example novel source-withdrawal family. Stop if any jobs fail rather than averaging incomplete matrix arms.

## Controls and criteria

- Baseline 1: constant option predictor, **72/144** due to fully reversed candidate order.
- Baseline 2: same-input **frozen LM per-candidate Yes-No scoring**; caveat: English Yes/No output tokens also used in Hebrew.
- Baseline 3: deterministic external owner/scope guard, always separate and never credited as learned neural skill.
- Register screening threshold, before looking at outcomes: trained head consistently improves over matched frozen on **all three seeds** at a given size; mean raw accuracy at least **58% (84/144)**, at least **30/72 fact-reversal pairs**, at least **16/36 full four-way quartets**, both language groups ≥**60% (44/72)**; genuine differing-policy families should exceed position baseline and never solely choose the same policy across user profiles.
- If only one size passes, do not infer a monotonic capacity advantage from 3 seeds; compare variation by project/family and validate on independent externally assessed data before production.
- Compute-cost fairness: per rank/seed exactly the same **epochs, batches, selected train groups**; parameterized models have different multiply-add costs, so *equal optimizer updates are not equal FLOPs*. Retain CPU timings.
- No changes to separate SELF model-internal research, native 5M, model checkpoints or production integrations.

**Conclusions await actual GitHub artifacts.**
