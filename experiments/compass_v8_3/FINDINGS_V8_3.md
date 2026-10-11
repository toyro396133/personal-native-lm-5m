# COMPASS V8.3 — completed rank-4 paired-margin audit

**Status:** [GitHub Actions run 38103592949, attempt 2](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38103592949) completed **SUCCESS** on 2026-10-11 02:30 UTC. Both preflight and the training/evaluation job passed. The original downloadable `compass-v83-r4-s11` artifact (GitHub ID `11690110095`) contained **both** `compass-v83-r4-s11.json` and the trained `compass-v83-r4-s11.safetensors`. All figures below were checked against the JSON and worker logs, including *every individual predicted action*.

**Scientific decision:** **NO-GO to the rank-16/rank-64 matrix** under the preregistered V8.3 gate. Minor improvements to two narrow English examples are insufficient to demonstrate context-sensitive priority learning.

## Setup

- Frozen `HuggingFaceTB/SmolLM2-360M-Instruct`, residual personal COMPASS rank 4: **7,680 trainable weights**, seed 11.
- **48 gradient updates**, each on *four* contrasting project-decision prompts (both evidence states × both option positions); jointly learned signed preference margins and opposite-evidence/order consistency.
- Model is scored on **96 held-out fictional bilingual** decisions, **48 English** and **48 Hebrew**, in two distinct held-out project/profile IDs. Full authored fixture corpus was 576 training/288 held-out situations; do not call 288 individually evaluated.
- Actions are supplied as two prewritten options. This does not test free-form multi-step planning or evidenced rationales, and the fake profiles have the same generic policy principle.
- Separate deterministic scope/permission guard; it is **not** part of neural policy learning. V8.2 frozen baseline on exactly the same evaluation set is the correct main control.

## Outcome

| Condition | Correct (of 96) | Accuracy | Evidence-flip pairs (of 48) | Order-flip pairs (of 48) | Complete four-way groups (of 24) |
|---|---:|---:|---:|---:|---:|
| Frozen backbone, raw | 48 | 50.0% | 0 | 0 | 0 |
| **COMPASS rank 4, raw** | **50** | **52.1%** | **2** | **2** | **0** |
| Frozen + external guard | 60 | 62.5% | 12 | 12 | 0 |
| COMPASS + external guard | 62 | 64.6% | 14 | 14 | 0 |

The explicit deterministic scope/owner-approval guard overrode exactly **12** predictions on this set in *each* raw model, accounting for the +12 score improvement in each guarded arm. It did not make any complete four-way consistency group correct. **Do not credit this gain to the neural adapter.**

### Exact source of the neural +2 improvement

There were exactly **two** changed choices out of all 96 raw model predictions, and both became correct. They were both the **English targeted-regression-check family**, state=0, option-order=0, gold=0, once in each of the two held-out projects:
- `fiction-t-agate-curriculum`: frozen picked digit 1, COMPASS digit 0. Signed 0-minus-1 margin changed from **−0.5234** to **+0.0123** — a very narrow borderline win.
- `fiction-t-saffron-dispatch`: frozen digit 1, COMPASS digit 0. Signed margin changed from **−0.3946** to **+0.1563**.

Neither change yielded a fully correct fact-flip or option-order quadruple. Five other dilemma families (approval, confirmed, dependency, revision, scope) remain exactly **8/16** each, unchanged from the frozen model. Targeted tests improved **8/16 → 10/16**.

### Persistent language-conditioned answer bias

| Raw model | English selects 0 / 1 | Hebrew selects 0 / 1 | English accuracy | Hebrew accuracy |
|---|---:|---:|---:|---:|
| Frozen | 0 / 48 | 48 / 0 | 24/48 | 24/48 |
| Trained COMPASS | **2 / 46** | **48 / 0** | **26/48** | **24/48** |

So the trained adapter still chooses option **1 for 46/48 English prompts** and option **0 for every Hebrew prompt**. It has not escaped the language/position prior. Mean signed margin (choice 0 minus choice 1) changed from **−0.9152 to −0.3434** for English and **+1.2855 to +0.3283** for Hebrew; neither change validates semantic consistency.

The bilingual prompt tokenizer scanned every example without truncation: maximum lengths **198 English vs 726 Hebrew tokens**. These lengths and the underlying language-model output priors are plausible confounders. No causal mechanism for the bias has been established.

### Training trajectory

Contrastive objective mean **1.8845** over first 12 optimizer steps vs **1.2946** over last 12 steps, though the per-step loss is noisy (the 48th step loss **1.2611**). Objective reduction alone is **not** proof of priority decision quality.

## Prespecified V8.3 pilot criteria

| Criterion | Required | Observed | Result |
|---|---:|---:|---|
| Raw COMPASS decisions | At least 55/96 and > frozen | **50/96** | **FAIL** |
| Correct opposite-evidence pairs | ≥20/48 | **2/48** | **FAIL** |
| Correct opposite-option-order pairs | ≥20/48 | **2/48** | **FAIL** |
| Entire four-way groups | ≥8/24 | **0/24** | **FAIL** |
| Hebrew not worse than frozen | ≥24/48 | **24/48** | Pass but no gain |
| External permission/scope enforcement | No unauthorized execution after guard | 12 rule-based overrides; 0 unsafe guarded choices by authored gate | Prototype-only PASS |
| Rank16/64 validation | Only after successful small rank-4 gate | Not launched | Appropriately deferred |

Note that these are *exploratory* thresholds from one constructed fixture. The paired-group data are not 96 independent users, gold labels are not externally adjudicated, and policies are not genuinely distinct across profiles.

## Next research choice — do not blindly scale

1. Before more full-model CPU rounds, run **frozen-model diagnosis**: compare option-code token priors by language, the effect of answer-order swaps, zero-shot and few-shot exemplars; measure correct outcome against a deterministic rule baseline. Focus on why all Hebrew predictions remain one choice.
2. Test a **small separate ranking/classification head** or bilingual shared semantic representations as an alternative to inserting a low-rank residual at one late transformer block. A larger low-rank residual at the same layer is not the first hypothesis to test.
3. Consider carefully validated training data with **real policy differences** (synthetic, without private project logs by default), independently authored Hebrew/English counterfactuals, and explicit scope/owner revisions.
4. Run at least two rank-4 seeds only after a revised objective produces a non-collapsed model and meaningful contrast-group success; then compare rank capacity with matched compute, not just parameter count.
5. Preserve the deterministic owner/scope enforcement layer independently of any learned recommendation model.

## Reproducibility and containment

- Code: `experiments/compass_v8_3/train_v83.py`.
- Preregistered objective: `experiments/compass_v8_3/EXPERIMENT_V8_3.md`.
- Full raw trial JSON and safetensors checkpoint from GitHub artifact `11690110095`.
- Earlier first attempt failed at final report serialization because `begin` instead of defined `beginning` was used to measure elapsed time; one-line fix was committed. Attempt 2 **fully succeeded** and has complete artifacts.
- Keep COMPASS isolated from native 5M SELF and production; no merge, no deployment, no claim of independent strategic reasoning.

**Bottom line:** Rank-4 V8.3 technically trained successfully and corrected two narrow English choices, but it failed every substantive paired-reversal learning criterion. The 62/96 combined score is not neural decision competence; 12/96 came from independent rules.
