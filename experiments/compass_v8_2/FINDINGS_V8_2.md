# COMPASS V8.2 — bilingual pairwise pilot results and scientific decision

**Status: GitHub Actions SUCCESS; learned priority policy NO-GO.** Run: [38102375388](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38102375388), attempt **2**, completed **2026-10-11 01:49:27 UTC**. Technical setup, preflight and training all completed successfully after fixing the Hebrew context-token ceiling. Original result/weights artifact: `compass-v82-r4-s11` GitHub artifact ID **11688651484**. Results were verified using both Actions worker logs and original unzipped JSON `compass-v82-r4-s11.json`.

## What was actually measured

- Frozen `HuggingFaceTB/SmolLM2-360M-Instruct`; COMPASS residual **rank 4 = 7,680 trainable weights**, seed **11**, **96 supervised steps**.
- 576 fully synthetic bilingual training situations from six fictional profiles, 12 fictional projects. 288 held-out synthetic bilingual situations from different profile *IDs* and project families. Only **96 held-out items** belonging to **two test projects** were evaluated in this pilot.
- Six authored dilemma families (core fault, targeted check, prerequisite, core mission approval, scope approval, plan revision); two factual states × both choice orders × English and Hebrew. Each group includes four items where reversal of evidence should change the answer and reversal of order should change the numerical code.
- Fixed **single-token binary choice** `0` or `1`; outputs use the same two-logit categorical cross-entropy in training and evaluation. Data labels are balanced 48/48 in 96 training steps.
- Separate deterministic owner/scope guard after classification, correctly audited on fixture; its outputs **must never be counted as learned neural performance**.

## Verified outcomes

| Method | Correct /96 | EN /48 | HE /48 | Correct fact-flip pairs /48 | Correct full quadruples /24 |
|---|---:|---:|---:|---:|---:|
| Frozen language model | 48 (50%) | 24 | 24 | 0 | 0 |
| **Trained COMPASS** | **48 (50%)** | **24** | **24** | **0** | **0** |
| Frozen + deterministic guard | 60 (62.5%) | 30 | 30 | 12 | 0 |
| COMPASS + deterministic guard | 60 (62.5%) | 30 | 30 | 12 | 0 |

Both raw models also achieve **0/48 correctly answered option-order reversal pairs**, and every one of six authored case families scores 8/16 in each model condition. Each model's predictions are **identical on all 96 examples** in spite of nonidentical underlying learned logits.

### The decisive diagnostic: language-conditioned constant answer

From the **original raw per-item JSON predictions**, the frozen model **and the trained adapter**:
- **English:** choose **option 1 in all 48/48** situations.
- **Hebrew:** choose **option 0 in all 48/48** situations.

Thus the apparent balanced overall action histogram `0:48, 1:48` is **NOT** substantive discrimination; it is a different *constant answer conditioned on prompt language*. In each language a constant predictor obtains 50% on this counterbalanced dataset. Neither model reacts to the project's core, to changed evidence, or to reversing choice order.

Relative raw logits differ after training, but **zero of 96 argmax decisions change**. On English examples the average logit(1)-logit(0) is ~0.915 frozen and ~2.068 adapted; Hebrew is ~-1.286 frozen and ~-0.576 adapted. These are *bias changes, not corrected decision behavior*.

### Explicit guard and training diagnostics

- Exactly 12 guarded decisions were changed by the owner/scope rule in each model condition; +12 correctly answered tasks, **48→60/96**, with **no** additional improvement attributable to training. The only fact-flip pairs corrected by the guard are 12/48; complete quadruples remain 0.
- Preflight checked all 576 train + 288 test context token lengths *before model allocation*: max English **198** tokens, Hebrew **726**, after raising the nontruncating input ceiling from 650 to 1,200. The first attempted job had failed before training with **683-token** Hebrew sample; attempt 2 succeeded. The EN/HE length difference is a potential confound and should be investigated.
- Training classification loss averaged ~**1.067** over first 24 updates vs ~**1.127** over last 24, nonmonotonic and not evidence of useful convergence; binary uniform-loss reference `ln(2)≈0.693`. Individual losses vary substantially and final step loss was 2.382.
- Bilingual profiles have different *IDs* but the same authored high-level preference text. The study tests no true user-specific policy difference. Dataset pairs are authored from the same limited grammar without independent human or external adjudication. Guard trusts explicit fictional source attributes and is not production-grade authorization.

## Registered objective gate

- GitHub workflow and bilingual fixture audit: **PASS**.
- Binary action tokens, matched training/evaluation and balanced dataset: **PASS**.
- Raw COMPASS better than frozen model or constant predictor by ≥5 percentage points: **FAIL** (50% vs 50%).
- Correct factual reversals ≥20/48: **FAIL** (0/48).
- Correct full quadruples ≥8/24: **FAIL** (0/24).
- Hebrew preservation / improvement: **NO IMPROVEMENT** (24/48 vs 24/48).
- Explicit external approval/scope enforcement: **PASS** only as a deterministic postprocessing prototype; not learned.
- Rank 16/64 or multiple seeds in V8.2: **NOT RUN**, because the rank-4 gate failed.
- Free-form grounded priority reasoning, actual distinctive user policies, long-term project-state revision: **NOT TESTED**.

**Decision:** Do not promote rank-4 adapter, do not launch a blind rank 16/64 matrix, do not merge any COMPASS findings into native 5M or model-internal SELF. The stronger immediate baseline remains explicit versioned project maps and auditable action gates.

## Next test proposal — V8.3 (not launched)

1. **Instrument why the model conditions on language rather than evidence:** inspect prompt-side 0/1 token priors, order labels, conditional logit deltas for each fact-flip and option-swap; run a trivial **controlled code permutation** and a neutral prompt sanity check.
2. **Change the learning objective** toward direct *pairwise margin ranking*: train logits to favor the correct action in both reversed-order presentations of the same situation, and train opposite outcomes for two evidence states. Penalize inconsistency explicitly instead of independently optimizing generic one-token classifications.
3. **Match length and semantics across English/Hebrew** or train/evaluate separately by language. Compare against frozen model prompted with an example, a simple source-rule engine and a learned *classification head* before attributing a gain to COMPASS.
4. **Make policies genuinely different** across fictional users and test new tasks within the same held-out policy; create a separate cold-policy transfer measurement where applicable. Add a separately authored Hebrew/English evaluation with independently reviewed gold.
5. Keep the **external** project-scope and approval enforcement; it must never become a learned-only safety guarantee.
6. **Pilot small first;** scale only if model changes decisions appropriately with evidence *and* both orderings, with paired success beating constant and strong explicit prompt/rule baselines.

No live private user data were included in training or evaluation. Existing V8, V8.1, native SELF and production code remain untouched.
