# COMPASS V8.7 — controlled owner-policy preference learning: completed audit

**Final Actions status:** [run 38111231513, attempt 2](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38111231513) — **SUCCESS** on 2026-10-11 04:27 UTC. Preflight, frozen-feature cache extension and retried rank4 training completed. Raw individual predictions, development history and personal weights saved as artifact **11691891392** (`compass-v87-policy-r4-s11.json` + `.safetensors`). Inspected the original per-decision JSON and runner logs, not only the job conclusion.

**Scientific verdict: NO-GO** on preregistered *cold* policy-change generalization. Fine-tuning improved previously exposed synthetic tasks, but degraded new policy-change dilemmas. No promotion, no automated rank16/64 sweep and no native SELF/production merge.

## Technical incident

- Initial training job **FAILED**, before training, because `eval_full(...)` from V8.6 returns **three** outputs but two development calls in `train_v87.py` unpacked only two (`ValueError: too many values to unpack (expected 2)`).
- Corrected both callsites to unpack three values in commit `e1ad3ba5e494b074b504978157edc4d256146306`. Requested rerun of **only the failed training job**; V8.7 preflight/cache reused successfully without recomputing frozen 360M embeddings.
- Attempt 2 completed all 18 policy-pair fine-tuning epochs. The development-only selector chose **epoch 17**: development policy pairs **16/16** both correct, ordinary dev cases **160/160**, and **40/40** dev contrast groups. This is high development fitness, **not** out-of-project performance.

## Setup

- Frozen common `HuggingFaceTB/SmolLM2-360M-Instruct`. Personal model is **7,680 trainable weights**, same rank4 option scorer architecture and initialized from the actual preserved V8.6 checkpoint (seed 11). Optimizer: AdamW 0.0003, 18 epochs, joint loss from original fact-state pairs + 0.8 policy-only paired supervision.
- **80** synthetic policy-only training pairs, **16** policy-pair development examples, all from preexisting fictional training projects, with *same* project/objective/evidence/choices/order but authorized policy switching between rapid and thorough.
- Exposed V8.6 diagnostics: **176** already-inspected heldout cases and **16** policy-only twins; those cannot be described as pristine test data.
- NEW synthetic cold-policy families: `maintenance_style` and `report_delivery` in two fictional projects and two languages, with 2 candidate orders, making **16 pairs**. Their family texts/option statements are unseen in training/dev, but concepts and label logic overlap. All text and gold labels were authored by the same research workflow, without independent human adjudication.

## Result: unchanged V8.6 versus V8.7 policy-trained, SAME cold examples

| Metric | Unchanged V8.6 head | Fine-tuned V8.7 head | Direction |
|---|---:|---:|---|
| Exposed ordinary synthetic decisions correct | **148/176** | **150/176** | +2 |
| Exposed evidence reversal pairs correct | **60/88** | **62/88** | +2 |
| Exposed complete state×order groups | **30/44** | **31/44** | +1 |
| Previously inspected policy-only pairs: both versions correct | **8/16** | **10/16** | +2 |
| Previously inspected policy-only pairs: preference actually flips | **8/16** | **10/16** | +2 |
| **NEW cold policy families: both versions correct** | **4/16** | **2/16** | **−2** |
| **NEW cold policy families: preference actually flips** | **8/16** | **4/16** | **−4** |

The model remains strong on the synthetic source/evidence templates it already knows, but **cold policy generalization worsens despite the policy-specific objective and perfect development performance**.

### Language-specific cold policy change

| Language | Unchanged head: both correct | Fine-tuned: both correct | Unchanged head: raw flip | Fine-tuned: raw flip |
|---|---:|---:|---:|---:|
| English (8 pairs) | **0/8** | **0/8** | **0/8** | **0/8** |
| Hebrew (8 pairs) | **4/8** | **2/8** | **8/8** | **4/8** |

The English scorer **never changes its semantic preference** in either new family, regardless of what the authorized owner policy says. Hebrew was responsive but degraded under fine-tuning. With 16 related pairs these figures are diagnostic, not generalizable statistical estimates.

The frozen/unchanged exposed ordinary performance shifted from English **82/88**, Hebrew **66/88** to fine-tuned English **82/88**, Hebrew **68/88**. Exposed overall source-retraction diagnostic remains **8/16**, including **0/8** correct cases where the source has actually been withdrawn. Policy-only fine-tuning does **not** fix the separate evidence-revocation failure.

### Cold family detail (post-training)

- `maintenance_style`: correct both sides only on **2/8** policy pairs (both in Hebrew on the rapid-project representation); none in English.
- `report_delivery`: **0/8** policy pairs both correct. There is an option-choice flip in two Hebrew pairs, but both flips are **in the wrong direction**.
- The shared-option scoring architecture always cancels the numeric candidate order; this is not proof of semantic adaptation. The relevant tests are whether preference changes alone flip the action *correctly*.

## Preregistered gate

| Criterion | Registered | Observed | Result |
|---|---:|---:|---|
| NEW cold policy both correct | ≥12/16 | **2/16** | FAIL |
| NEW cold policy option flips | ≥12/16 | **4/16** | FAIL |
| Hebrew cold both correct | ≥5/8 | **2/8** | FAIL |
| Cold policy better than original unchanged V8.6 | Strictly improved | **2 vs 4 correct** | FAIL |
| No catastrophic regression on exposed ordinary tasks | ≥140/176 | **150/176** | PASS |
| Simulated trusted source registry fail-closed contract | Block invalid execute | PASS in synthetic unit checks | Software-only PASS |

**Research decision: NO-GO for training-size escalation (30,720 and 122,880), autonomous personal preference guidance and promotion to live projects.**

## What to do next

1. Address **representation, not rank size**. The frozen 360M last-token candidate embedding + nonlinear head may ignore decisive policy context while overfitting task-specific option wording. Perform controlled embedding probes: policy-text shuffled/removed, same task different owner preferences, cross-family/multilingual decodability, candidate-only vs context-only features, heldout family grouping.
2. Compare against **explicit symbolic policy matching** for truly equal safe options, and rule-based graph/source validity checks for mandatory blockers. Fine-tuning a tiny head may not be needed for deterministic owner-provided preferences.
3. New data must include varied policy-only interventions in MANY decision families and projects, paraphrases by independent authors/adjudicators, bilingual coverage and source timeliness/versioning. Lock a NEW independent test before tuning.
4. Recheck the training/evaluation data model: synthetic policy pairs change metadata source IDs (not visible to scorer) and the policy version, but only the authorization policy words should affect encoded inference; verify feature-cache keys and embeddings before attributing errors to reasoning.
5. Do not treat an untrusted literal `source_valid` field in a prompt as authenticated provenance. Maintain independent executor verification and fail closed; the V8.7 contract is **unit-only**, with no live cryptographic signature validation or provider trust connection.
6. Keep V8.6 original checkpoint, V8.7 fine-tuned checkpoint and original cold predictions reproducibly stored. No evidence a larger personal head would resolve cold generalization.

## Containment / original sources

- Research branch: `experiment/compass-v8-7`, protocol: `experiments/compass_v8_7/EXPERIMENT_V8_7.md`.
- Core: `train_v87.py`; fixtures: `fixtures_v87.py`; data audit: `validate_v87.py`; independent (mocked) gate: `trusted_sources_v87.py`.
- Actions run 38111231513 artifact **11691891392** preserves per-case JSON predictions and **7,680 personal weights**. Full frozen-cache artifact **11690889317** preserved.
- Native self-model 5M/SELF and production are unchanged. **No real user's project records or personal preferences were used**.
