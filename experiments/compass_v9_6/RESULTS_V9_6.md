# COMPASS V9.6 — observed results and next research decision

**Run:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38125425983
**Run attempt:** 2; first attempt failed at robust-fine-tuning because a copy of the frozen V9.4 bridge still had `requires_grad=False`. Fixed in commit `8d6e2d7`; rerun passed CI.
**Experiment artifact:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38125425983/artifacts/11697420883 — per-case predictions, JSON, small model weights.
**Scientific verdict:** `NO_GO_TO_LM_ADAPTER_INTEGRATION`.
**Status:** Completed. No in-model 360M adapter deployed or trained by this experiment.

## Headline finding

The extra TRAIN-only semantic perturbation augmentation improved robustness of *neutral* project/user choices against newly written nonbinding context, but caused over-reliance on the gate for truly binding cases and did not improve the all-case score. The separately trained detector achieved **92.19% binding sensitivity** at **75% specificity** on newly written families, with **16 false overrides out of 64 neutral** situations. The V9.5-type baseline hybrid still beats the V9.6 hybrid on the same 128 new paired cases.

### Paired BOTH-correct, matched, newly written V9.6 tasks

| Method | All (128) | Binding (64) | Neutral (64) |
| --- | ---: | ---: | ---: |
| Original V9.4 neural bridge alone | 86 | 39 | 47 |
| Robust V9.6 neural bridge alone | 69 | 18 | 51 |
| Original V9.4 bridge + newly fitted group-held-out V9.6 constraint gate (**matched baseline**) | **101** | **62** | 39 |
| Robust V9.6 bridge + SAME constraint gate (**candidate**) | **99** | **59** | **40** |
| Privileged researcher fixture rule (unfair metadata access) | 112 | 64 | 48 |

This shows the robustness fine-tuning marginally improved neutral behavior (39 -> 40 under the gate) but degraded binding choices (62 -> 59 under the same gate), for a net regression of two pairs. The pure robust bridge alone was especially fragile on binding tasks (18/64 versus 39/64). The proper scientific takeaway is not to claim a general improvement.

### Semantic perturbations: 64 NEW neutral paired tasks each

| Suite | Original bridge + same gate | Robust bridge + same gate | Delta |
| --- | ---: | ---: | ---: |
| Withdrawn / revoked obligations | 29/64 | **46/64** | **+17** |
| Benign context distraction | 44/64 | **47/64** | **+3** |

Pure neural model without the gate scored 29 -> 50 on revoked and 47 -> 51 on benign. However in the full hybrid the head itself falsely overrode on **8/64** revoked cases, illustrating why bridge-only gains do not necessarily translate to the composite.

### Recognition, invariance and project isolation

- Binding detector: **59/64 true binding detected** (92.19% sensitivity); **48/64 neutral correctly ignored** (75% specificity).
- Revocation stress: **8/64 false binding overrides**, above maximum registered 3/64.
- Exact option permutation symmetry: **128/128**.
- Alterations to unrelated project histories: **0 decision changes in 32** paired routing checks.
- Family-held-out development detector had 100% within its own TRAIN-family generator split but performed less well on completely new authored obligations: semantic shift remains significant.

### Conjunctive preregistered gates

| Gate | Result |
| --- | --- |
| New binding paired BOTH-correct >=48/64 | PASS (59) |
| Binding no regression vs same-gate baseline | **FAIL (59 vs 62)** |
| New neutral paired BOTH-correct >=45/64 | **FAIL (40)** |
| Neutral regressions <=3 | PASS |
| New binding sensitivity >=90% | PASS (92.19%) |
| New binding specificity >=90% | **FAIL (75%)** |
| Revocation paired BOTH-correct >=40/64 | PASS (46) |
| Revocation improvement >=12 pairs | PASS (+17) |
| Benign paired BOTH-correct >=40/64 | PASS (47) |
| Benign no regression | PASS (+3) |
| Revocation false overrides <=3/64 | **FAIL (8)** |
| Candidate swap equivalence >=95% | PASS (128/128) |
| Unrelated project history isolation 0 changes | PASS (0/32) |

9 passed, 4 failed. Strict verdict `NO_GO_TO_LM_ADAPTER_INTEGRATION`.

## Why this is useful despite NO-GO

It separates two weaknesses that earlier runs mixed:
1. **Decision-model brittleness:** irrelevant current-task text can distort intent-conditioned decisions. Augmented neutral training improved this specific aspect, from 29/64 to 46/64 on the new revoked-obligation suite.
2. **Authority recognition from words alone:** a separate classifier can still read 'obligatory' in the wrong context or miss genuinely active mandates. More paraphrase training alone is not a reliable definition of who has authority, which sources are trusted, which requirement is in force or which has been revoked.

The user/project state separation and option-symmetric scoring continue to behave as expected in these synthetic tests; neither establishes independent real-world use.

## Proposed V9.7 design (not executed in this run)

**Primary next move:** explicit provenance-aware policy validity *outside* the preference learner, not another round of piling word-based obligations into a classifier. A trustworthy decision pipeline should distinguish:

- Personal preference / project vision, learned from contextual history.
- Active project workspace identity and inherited user-level principles.
- External requirement claims and **authenticated source**, effective date, scope, expiry, and revocation.
- Hard constraint evaluator for candidate actions against active, verified requirements; unknown or conflicting policy requires abstention/human confirmation, not guessing from persuasive prose.
- Decision scorer using intent memories for permissible actions only, with irrelevance-consistency training retained where it proves beneficial.

To measure incremental effect rather than handing an oracle answer to the model, compare (a) claim-only text; (b) supplied structured source/validity evidence with realistic uncertainty, not fixture binding/gold labels; (c) independently audited human-written cold cases, including genuinely unverified, outdated and conflicting evidence. Allow *abstain* outcomes and report coverage/accuracy conditional on verified authority. Multi-seed repeatability and equal-access baselines are still essential before declaring any adapter ready.

Do not treat the research author's synthetic source descriptors as authenticated legal, safety, or production requirements. No V9.6 branch has been merged to main except the workflow trigger file. Existing production SELF components are unchanged.
