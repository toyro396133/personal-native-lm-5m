# COMPASS V9.5 — actual experiment results and V9.6 decision

**Run:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38124601285  
**Workflow status:** success (preflight and experiment finished).  
**Scientific verdict:** `NO_GO_TO_LM_ADAPTER_INTEGRATION`.  
**Detailed evidence:** GitHub Actions artifact `compass-v95-independent-policy-results` (artifact ID 11696528594), contains full per-case JSON, fitted bridge and policy head weights. This file is a readable record; artifact itself is stored by Actions.

## Frozen-before-cold methodology
- Train V9.3 numeric memory and V9.4 decision bridge anew, entirely on V9.4 TRAIN data.
- Train separately V9.5 binding detector + text-conditioned compliance option scorer using TRAIN data and explicitly withdrawn-obligation negative examples.
- Select classifier threshold (0.5) on repeated TRAIN-template development only, before looking at any cold data.
- Evaluate V9.4 reused 128 paired cold examples **not blind**; 128 newly written V9.5 held-out family paired examples (still same synthetic intent schema); and 64 neutral examples with a revoked-obligation text added to the question.
- The arbitration head does NOT receive `case.binding`, `case.scope`, `case.swapped`, gold labels or family ID during inference. Privileged synthetic rule/prototype baselines DO receive hidden fixture metadata.
- No native 360M adapter created, no 360M LM control rerun on new scenarios, no real permissions validated.

## Headline matched new-family performance

| Metric | V9.4 neural-only evaluated on V9.5 scenarios | V9.5 separate policy head + neural preferences |
| --- | ---: | ---: |
| Paired BOTH-correct, all | **104/128 (81.25%)** | **109/128 (85.16%)** |
| Paired BOTH-correct, binding | **55/64 (85.94%)** | **60/64 (93.75%)** |
| Paired BOTH-correct, neutral | **49/64 (76.56%)** | **49/64 (76.56%)** |
| Candidate order symmetry | — | **128/128** |
| Neutral regressions compared to V9.4 | — | **0** |

These are 128 PAIRS; each pair tests original and changed principle with identical task/options. Both choices must be correct for pair success. Do NOT present as independent human real-world decision accuracy.

## Binding detection
- Fresh case gate: **48/64** binding items recognized (sensitivity **75.0%**); **64/64** neutral items correctly rejected (specificity **100%**).
- Separate *policy-only* action chooser correctly selected **64/64** binding paired decisions on the fresh author-generated dataset. Therefore **recognizing when the override applies**, rather than selecting its direction, is the immediate bottleneck in this tiny synthetic setup.
- Threshold chosen on within-training-template dev achieved perfect scores, but generalized imperfectly to new textual situations. This demonstrates that the dev split was too easy.
- The reused V9.4 diagnostic dataset showed **no hybrid improvement** (both 101/128). A fresh-family improvement must not be portrayed as repeatable cross-domain benefit.

## Revoked-obligation stress — decisive failure
- 64 *neutral* pairs with semantically withdrawn binding-directive language added to the question.
- Gate correctly refrained from overriding on **64/64**.
- Nevertheless, both V9.4 and V9.5 neural preference bridge got **only 14/64 paired cases correct** (1/32 user and 13/32 project). Neutral unmodified new-family performance was 49/64.
- Because the gate never overrode, deterioration comes from the **base task-sensitive decision bridge** reacting to a semantically irrelevant or contradictory text augmentation, not from the constraint gate.
- Interpret as strong prompt/context-perturbation brittleness; avoid claiming robustness from the good classifier specificity.

## Preregistered pass criteria

| Criterion | Result |
| --- | --- |
| Binding >=48/64 | PASS (60/64) |
| Binding beats matched V9.4 | PASS (+5 pairs) |
| Neutral >=45/64 | PASS (49/64) |
| At most 3 neutral regressions | PASS (0) |
| New binding detector sensitivity >=90% | **FAIL (75%)** |
| Specificity >=90% | PASS (100%) |
| Revocation false overrides <=5% | PASS (0/64) |
| Option-order numerical symmetry >=95% | PASS (128/128) |

**Conjunctive verdict = NO-GO.** This is a scientific stop for LM adapter integration, not a CI error.

## Immediate recommended V9.6 priority
1. **Make intent choice invariant to nonbinding text perturbations.** During TRAIN only, augment neutral dilemmas with authentic-seeming *rescinded* constraints while maintaining the same policy pair gold; train paired task-conditioned regularization. Use new withheld revocation paraphrases at test. Do not train on V9.5 stress examples or select parameters from their results.
2. **Improve binding recognition across novel wording.** Introduce multiple independent obligation phrasings and challenging `not currently mandatory` counterexamples from TRAIN families, and group-split development by entire unseen task families rather than random cases of the same template. Fix threshold exclusively there.
3. **Distinguish trusted authority data from language.** Real mandatory requirements need externally authenticated source, active status, validity dates, provenance and revocation, not model guesses based only on prose. Do not substitute hidden fixture `binding` labels in decision runtime.
4. Evaluate against a **metadata-free** semantic baseline with access to the same question/options; separately disclose synthetic upper-bound oracle baselines. Add multiple random seeds and independent human-authored unfamiliar cases before a stronger claim.
5. Preserve per-project memory isolation and test changes to irrelevant project streams. Remain outside 360M LM weights until constraints and robustness demonstrably generalize.

These results are strictly from short English synthetic histories with a frozen MiniLM semantic encoder, trainable GRU history memory and compact decision scorers.
