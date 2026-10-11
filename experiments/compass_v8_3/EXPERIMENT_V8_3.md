# COMPASS V8.3 — paired-margin, reversal-aware priority experiment

## Research objective

V8.2's rank-4 model obtained 48/96 decisions (50%), exactly the frozen SmolLM2-360M model. Both selected 1 for all English prompts, 0 for all Hebrew prompts, and got **0/48** factual reversal pairs and **0/24** complete fact/order quadruples. Therefore the experiment showed a language-conditioned position bias, not conditional project-priority learning. It also failed to prove user-specific policy learning.

In V8.3, retain the same **fictional bilingual authored** V8.2 train/test dataset and heldout split, but change the training objective to pairwise margins with explicit reversal-consistency penalties. Keep original V8.2 results as a fixed historic comparison; do not retrofit V8.2.

## Data, baseline and test

- Frozen backbone `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Train only COMPASS rank-4 residual 7,680 weights; seed 11, 48 contrastive steps (each step sees **four** prompts representing one project, language and dilemma kind, with both fact states and both candidate orderings).
- 576 authored train examples across 6 fictional profile IDs and 12 project maps; 288 authored heldout examples across 3 different profile IDs and 6 project maps. **Exactly 96 holdout cases in two projects** from different profile IDs and different project kinds evaluated in the pilot.
- EN/HE prompts are full-length without truncation (token-length scanner capped at 1,200). Labels are exactly two single-token digits 0 and 1, verified by tokenizer.
- The target is the **signed difference** of next-token logits for choice 0 vs 1. Correct sign reverses if factual evidence changes or if the two options are reversed. Explicit softplus margin loss (`target margin=0.75`) plus bounded consistency penalties (`order_weight=0.2`, `fact_weight=0.2`). These are fixed before running, not tuned on test cases.
- Two raw model arms: unchanged frozen backbone (direct control) and trained rank-4 adapter, both on identical project-context prompts.
- Two *separately reported* postprocessing arms: frozen/COMPASS plus authoritative external permission/scope guard. Never credit deterministic rules as COMPASS understanding.

## Preflight and validity tests

- Validate project/user group split; 48 training scenario types per project, balanced across two languages, two answer positions, two evidence states; no answer labels in model prompts.
- Verify group structure and exactly 144 training quadruples.
- Verify differentiable contrastive loss gradient using artificial logit margins, check sign flip mathematics and fixed digit tokens.
- Preserve the natural-language full context rather than candidate-dependent truncation.
- Record per-language prediction histograms and signed logit margins so a global 50/50 histogram cannot hide a language-specific constant answer.
- All six dilemma families and all evidence/option-order flips evaluated in original authored test groupings.

## Registered exploratory go/no-go gate

One rank-4 seed is a *diagnostic*, not reliable generalization. A necessary (not sufficient) signal for expanding capacity is:
1. Learned **raw** action accuracy at least 55/96 (57.3%) and strictly above unchanged frozen model, plus no collapse to one digit per language.
2. Correct **fact reversals at least 20/48**, correct **order reversals at least 20/48**, and **complete quadruples at least 8/24**.
3. No worsened correctness in both authorization/scope dilemma families; no worsening in Hebrew-specific accuracy.
4. A future run must also compare against written-policy prompting, a deterministic project-rule baseline and distinct policies before making claims of personal assistant reasoning.

These are prespecified **screening** thresholds only. This dataset is still authored by one agent, shares six templates between train and test, and gives every fake user the SAME transferable policy text: it does **not** establish personalization or independent human labeling. **Do not automatically promote model or run Rank16/64** based only on headline accuracy. If gate fails, investigate feature representation, language priors, calibrated embeddings and a dedicated ranking head, not parameter inflation.

## Project isolation

COMPASS is the distinct personal assistant priority module. SELF is the separate model-internal self-reference experiment and native 5M research. The V8.3 branch and workflow are isolated; no user private project information and no native SELF changes, no production merges.

Files: `experiments/compass_v8_3/train_v83.py`, this protocol document, and `.github/workflows/compass-v83-pilot.yml`.
