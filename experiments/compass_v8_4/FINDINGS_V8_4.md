# COMPASS V8.4 — completed bilingual shared-option ranking-head audit

**GitHub Actions:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38106163536
**Status:** SUCCESS, 2026-10-11 03:04 UTC. Both verification and model jobs completed. Original artifact 11690043605 contains compass-v84-ranker-w4-s11.json and compass-v84-ranker-w4-s11.safetensors. The JSON was downloaded, unpacked and independently inspected per case and per language.

**Research verdict: PASS for preregistered synthetic pilot screening; NOT validated for real personal project prioritization, deployment, or parameter scaling.**

## Setup

- Frozen HuggingFaceTB/SmolLM2-360M-Instruct. A shared candidate-action scorer 960→4→1 with **3,848 trainable parameters**, not a residual transformer adapter. Both candidate actions are encoded individually with the same project objective, observed evidence and approved generic policy.
- 576 synthetic bilingual training-pool examples, of which 480 from five fictional profiles train the scorer and 96 from the sixth fictional profile serve as the development set. Separate authored test pool 288. Evaluated **96** examples from **two** unseen fictional project/profile IDs, 48 English/48 Hebrew.
- Cached **672 distinct candidate-context embeddings**. Max input 161 English vs 581 Hebrew tokens. No truncation.
- AdamW learning rate 0.003, maximum 14 epochs; chosen **epoch 11** based only on development score: **82/96** correct and **18/24** fully correct project×language×dilemma quartets. Locked test evaluated once afterward.
- Direct frozen matched-input control: frozen LM next-token Yes-minus-No score for each separately encoded candidate. This is a meaningful but not necessarily strong control, especially because English Yes/No output tokens are used for Hebrew input. Historic V8.3 frozen 48/96 vs residual adapter 50/96 were measured on DIFFERENT prompts and are NOT the primary comparison.
- External owner/scope permission guard is postprocessing and is scored separately. A learned gain must be established without the guard.

## Locked test outcomes

| Arm | Correct /96 | Accuracy | Correct fact reversal pairs /48 | Correct order reversal pairs /48 | Complete fact×order quartets /24 |
|---|---:|---:|---:|---:|---:|
| Matched frozen candidate Yes/No score | 52 | 54.2% | 8 | 26 | 4 |
| **Trained shared option-ranker raw** | **74** | **77.1%** | **28** | **37** | **14** |
| Frozen + deterministic authorization guard | 62 | 64.6% | 14 | 31 | 7 |
| Trained head + deterministic guard | 80 | 83.3% | 32 | 40 | 16 |

The model alone improves **+22/96** over the matched frozen control. Among the same 96 decisions: 42 both correct; 32 changed from wrong to correct; 10 from correct to wrong; 12 both incorrect. A naive paired McNemar test gives p≈0.00094, but the 96 synthetically related prompts correspond to only **24 distinct four-way decision groups**. Conventional independent-item statistical interpretation is invalid; do not overclaim significance or generalization.

The external guard improves the head by **+6/96** (74→80) and the frozen control by **+10/96** (52→62). Those are rules-based corrections, NOT extra learning by COMPASS.

### By dilemma family (16 cases each)

| Dilemma | Frozen correct | Trained head correct |
|---|---:|---:|
| Core change approval | 6 | 12 |
| Verified vs unverified failure | 8 | 12 |
| Core-blocking vs routine prerequisite | **10** | **8** |
| Valid vs superseded plan | 12 | **16** |
| Authorized vs forbidden scope expansion | 8 | 12 |
| Essential targeted vs optional broad checks | 8 | **14** |

**Critical weakness:** Blocking-prerequisite discrimination actually DROPPED to chance (8/16). This is a core COMPASS requirement. For explicit unapproved core changes (state=1), raw trained head gets 4/8 wrong; for forbidden scope expansion (state=1), it gets 2/8 wrong. External execution guards remain essential.

### Language and position controls

| Language | Frozen correct /48 | Trained correct /48 | Fully correct four-way groups |
|---|---:|---:|---:|
| English | 30 | **44** | 3/12 frozen → **10/12** trained |
| Hebrew | 22 | **30** | 1/12 frozen → **4/12** trained |

In each language, both frozen and trained option scorers predict exactly 24 zeros / 24 ones over the order-counterbalanced test. The prior V8.2/V8.3 pattern of always choosing one particular number per language has been eliminated structurally. This does not itself prove semantic learning. There is a serious **English 91.7% vs Hebrew 62.5%** gap.

**Important distinction:** The shared scorer is order-equivariant by architecture. It flips its proposed digit when the two candidate options are swapped on ALL **48/48** pairs. The reported **37/48 correct order-reversal pairs** means both choices were actually correct on those 37 pairs; it does NOT imply the ranker learned to be order invariant. Eleven other pairs are consistently wrong despite correctly flipping the digit.

Fact-reversal pairs are the stronger scientific signal: improved **8/48 → 28/48**. Fully correct fact×order groups **4/24 → 14/24**. Raw model remains short of robust correctness, especially in Hebrew and dependencies.

## Preregistered go/no-go gates

| Gate | Required | Observed |
|---|---|---|
| Raw learned accuracy and greater than frozen matched control | ≥55/96 and higher than frozen | **PASS 74 > 52** |
| Correct fact-reversal pairs | ≥20/48 | **PASS 28** |
| Complete four-way groups | ≥8/24 | **PASS 14** |
| Each language accuracy | ≥26/48 and no language-digit collapse | **PASS English 44, Hebrew 30** |
| Raw approval/scope performance no worse than frozen | At least frozen score per family | **PASS approval 12>6; scope 12>8** |
| Independent adjudication, genuinely varied user policies and new case families | Not part of this pilot | **NOT TESTED** |

**Decision:** Promising synthetic feasibility and permission to proceed to stronger **V8.5 validation**, NOT promotion to production and NOT an automatic high-rank sweep.

## Risks, caveats, and next study

1. Train/test policy profile IDs are different, but their approved PERSONAL POLICY TEXT IS THE SAME. This experiment has **not proven personalization** to individuals with genuinely different priority values.
2. The evidence/action wording comes from the same limited set of authored bilingual templates across splits; the test project names differ. No independent human review, truly unseen decision-family holdout, changing source-validity history or real project evidence exists.
3. Hebrew remains weak (4/12 complete four-way groups). Dependency-blocking cases remain chance-level and regress from frozen. A frozen comparator using English Yes/No tokens on Hebrew input may be weak; test bilingual localized and calibrated controls and deterministic project graph dependency rules.
4. Next V8.5: freeze the present protocol and create an **independently authored** bilingual benchmark with changing approvals, conflicting or withdrawn source evidence, genuinely conflicting personal policies, and different dilemma families; keep test locked before trying new prompts. Repeat width-4 head over seeds 11/19/37 and report project/dilemma-level uncertainty. Do not claim gains based on repeatedly looking at the same 96 test items.
5. Separate scope/owner authorization must be enforced with trustworthy provenance by the action executor, not by weights.
6. The 360M CPU frozen embedding pass took ~1046 seconds; subsequent small-head training is very cheap. Reuse frozen embedding caches when justified, without allowing test-label contamination.

Original checkpoint and 55,078-byte JSON are in GitHub Actions artifact 11690043605. Experimental branch is experiment/compass-v8-4. No changes to native SELF/5M or production.
