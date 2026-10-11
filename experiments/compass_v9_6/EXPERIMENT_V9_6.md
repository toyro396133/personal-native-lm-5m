# COMPASS V9.6 — robust intent decisions and out-of-family obligation recognition

Protocol committed **before viewing any V9.6 test predictions**.
Parent experiment: V9.5 `NO_GO_TO_LM_ADAPTER_INTEGRATION`.
Run V9.5: https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38124601285

## Scientific issue

V9.5 reached 109/128 synthetic decision pairs correct, but its question-only constraint detector recognized only 48/64 new binding requirements (75%) and its preference decision network collapsed to 14/64 under *revoked-obligation text*, even when the detector correctly did not override. Favoring an automatic 360M integration here would be premature.

## Hypothesis and intervention

Can **training-time neutral-question invariance** plus diverse obligation phrasing improve robustness without reducing ordinary intent-sensitive choices or relying on case metadata at inference?

- Preserve user/project separate recurrent 48D states, trained on V9.3 TRAIN only using frozen MiniLM; project routing comes solely from workspace identity.
- Reproduce the **original V9.4 pure bridge** on V9.4 TRAIN-only with its existing within-train development, for a matched baseline.
- Copy its *training state*, then fine-tune the option-symmetric bridge using V9.4 TRAIN neutral cases and **new TRAIN-only withdrawn-obligation and irrelevant-context rewordings**. Train on original plus augmented questions, with identical gold labels, balanced paired history interventions, and symmetric option scores. Add consistency regularization between ordinary and modified question scoring. No test cases are used for optimization, model choice or threshold.
- Re-train V9.5 **separate obligation head** on V9.4 TRAIN families only using diverse freshly authored binding-authority paraphrases and nonbinding rescission variants. Hold entire TRAIN families out for threshold-selection development: no example with a family ID used to fit head weights enters its development set. Never use V9.5 cold tasks for training.
- At inference, head receives only frozen encoder vectors of question and two action texts; bridge receives numeric user+selected-project memory and frozen question/action vectors. Neither model receives `case.binding`, `case.scope`, `case.swapped`, expected flip, family ID, or gold label. The text itself states obligations. No fixture oracle may route predictions.
- Compare the original V9.4 bridge + V9.5-style gate trained under the same new group-split protocol to the new robust bridge + the **same frozen gate**. Report neural-only baseline, neural-only robust, and both hybrids on the exact same cases.
- Case study on V9.5 cold is **retrospective only** (already seen). Main outcome is a *newly authored* V9.6 cold suite constructed before testing.

## Registered V9.6 tests

- **V9.6 cold**: 128 paired dilemmas (64 neutral, 64 binding; 64 personal, 64 project; option orders balanced). Four never-trained, never-V9.5-tested task families per scope. Distinct task descriptions and obligation phrasings, synthetic researcher-written. Both original and changed intent state must be correct to score a pair. Active project selected by workspace key as in prior versions.
- **New rescinded obligation challenge:** 64 neutral paired cold dilemmas with *new*, postposed or embedded rescission clauses (some user, some project). No label changes. False overrides and pair accuracy both measured.
- **New benign-context distraction challenge:** same 64 neutral pairs with novel irrelevant extra procedural text, no label changes.
- **Retrospective** 128 V9.5 paired dilemmas (not considered blind).
- Option permutation equivalence on 128 decision instances and corruption of unrelated project history for paired isolated-project audit.

## Prespecified gates (all conjunctive)

1. V9.6 **new cold binding BOTH-correct >= 48/64** and no regression versus matched baseline hybrid on the same 64.
2. V9.6 **new cold neutral BOTH-correct >= 45/64**, no more than 3 matched regressions versus baseline hybrid.
3. V9.6 binding detection **sensitivity >= 90%** and **specificity >= 90%** on new cold questions.
4. Revocation-challenge neutral BOTH-correct **>= 40/64** and at least **+12 pairs** over matched baseline hybrid with same modified questions.
5. Benign distraction neutral BOTH-correct **>= 40/64** and **>= baseline hybrid** with same modified questions.
6. Revocation **false override <= 5%** (<=3/64).
7. Option-order symmetry **>=95%**.
8. Isolated unrelated-project-history changes affect **0/32** fixed-context decisions when active project + user histories are unchanged.

Passing these gates is **only provisional synthetic progress**: still no LM adapter integration until real human-authored and permission-verified evaluation. A failing gate remains NO-GO even if workflow completes.

## Limitations

All datasets, intentions, labels, synthetic authority and rescission clauses are researcher-authored. The problem remains a narrow binary semantic schema; no independently audited real-world human intent or legal validity, no live user memory updates, no actual in-transformer SELF coupling. Group-split dev is stronger than template-random dev, but same generator and common history phrases remain. Obligations are still recognized from prose only: **no authoritative source authentication, effective dates or revocation verification**. Metadata-aware symbolic baselines are not fair neural comparators and must be clearly labeled.

Report per-case predictions, paired effects, override traces, all gates, chosen dev threshold, trained model weights and aggregate metrics. No optimistic cherry-picking after viewing cold results.
