# COMPASS V9.5 — frozen history memory + separately trained constraint arbitration

**Protocol committed before viewing any V9.5 predictions.** V9.4 cold data have already been seen, so reused V9.4 cases are strictly diagnostic and cannot be relabeled as blind. V9.5 adds 128 *newly authored* task-family heldouts, with new project/task families but some reused history paraphrases and the same researcher-generated binary answer mechanism. That is NOT independent human evidence.

## Research hypothesis
V9.4 used one memory-conditioned decision scorer for preferences and obligations, scoring 46/64 binding pairs correctly, under its 75% requirement. An independent, **task-text-driven** learned policy gate may identify an explicit binding requirement and select a compliant option without relying on incidental user/project style. This is a narrow proof of concept, not a real legal/compliance validity verifier.

## Architecture
- Reuse V9.4 frozen pretrained MiniLM semantic encoder, trained 48D user/project GRU memory, and option-symmetric learned choice bridge, all retrained on **V9.4 train only**; no V9.4 or V9.5 cold data are used in gradient updates or threshold selection.
- Train separate `ConstraintHead` on TRAIN text embeddings of task question and candidate action embeddings, independently of personal/project history. A probability head predicts whether a question states an obligation; a shared symmetric option scorer estimates which action complies **only when obligation is detected**. Both are trained on synthetic TRAIN binding labels and binding action choices, not on cold gold labels. In particular, the head receives neither `case.binding`, `case.scope`, `case.swapped`, gold answers, task-family IDs nor any target project name at inference. A measured threshold is selected using train-template dev only.
- Hybrid routes to compliance scorer if probability >= development-selected threshold; otherwise to V9.4 neural style bridge. Never use fixture's binding metadata to route at inference. Policy head scores only text; obligations have no source authentication in this benchmark.
- Additional TRAIN-only *revoked-obligation negative* text augmentation to distinguish an explicit current mandate from mention of a withdrawn historical requirement.
- Evaluate with paired changed principles (both states must be correct), option swaps, independent project-routing and candidate-order symmetry.

## Evaluation
1. **V9.4 reused diagnostic cold:** 128 pairs, explicitly not blind.
2. **V9.5 new task families:** 128 pairs, 64 neutral, 64 binding, balanced user/project and swapped ordering. Different from both V9.4 train and cold families; still synthetic and same broad semantic axes. New scenario/task texts are written prior to predictions.
3. **Revoked requirement stress:** 64 additional neutral pairs with withdrawn historic-mandate text appended to the task; original neutral expected answers remain. New scenario texts should not be mistaken for current authoritative requirements.
4. Compare matched scores of V9.4-style memory+bridge, V9.5 hybrid, no-history, and privileged symbolic and prototype baselines. Privileged baselines get fixture-derived scope/binding and are *not* fair language-only models.
5. Preflight tests for action permutation symmetry and no gold/metadata argument to the arbitration head. Preserve decision-level results and weights.

## Prespecified success gates (all required for provisional synthetic progress)
- New-family binding paired BOTH-correct >= **48/64 (75%)**, and strict **positive gain** over the V9.4 pure neural baseline evaluated on those identical 64.
- New-family neutral paired BOTH-correct >= **45/64 (~70%)**, and no more than **3 paired neutral regressions** compared with the V9.4 neural baseline.
- New-family **obligation-detection** sensitivity and specificity >= **90%** each; classifier is audited separately from policy-choice score. This uses binding metadata only for EVALUATION.
- Revocation stress **false override rate <= 5%**, measured by policy gate decision rather than gold answer.
- Numerical candidate-order symmetry >= 95% on 128 swapped evaluations.
- No full-size LM adapter integration until an independent, human-authored and permission-aware evaluation.

Other limitations: two semantic principle classes, author-chosen actions, short histories; no authenticated-source verification or persistent per-user adapter. A success does not imply project intention is fully understood or that the component can adjudicate real contracts.
