# COMPASS V9.11 — versioned personal/project principle slots, not a SELF model

**Preregistered before any V9.11 cold predictions (2026-10-11).**

## Why
V9.10 train-only three-event English GRU collapsed from 8–10/128 to 49–57/128 when the same five old+new statements were reordered so that main preferences came last. This is *posthoc* evidence of last-event swamping. COMPASS V9.11 tests the smallest architectural fix: stable, provenance-labelled principle slots instead of one running state. **This project is COMPASS, not training native SELF or modifying a 360M language model.**

## Distinct components
- **COMPASS personal**: persistent user-wide principle slot(s) and their evidence; transferable when applicable, never replaced by a project-only instruction.
- **COMPASS project**: separate persistent per-project vision slots and evidence; changing implementation does not automatically revoke purpose.
- **Scope and provenance**: signed demo UI source declaration `personal` or `project`, independently validated; unknown => ASK.
- **Principle history**: separate versioned preference record with `slot_id`, `kind`, `author/scope`, `version`, `reason`, `supersedes`, `active`, `validity`, and 3 supporting statements. A new contextual note gets its own slot and **cannot silently overwrite the preference slot**. An explicitly superseded preference is inactive even if presented last.
- **Obligation validity**: optional separate MOCK policy controller with scope, effective period, cancellation state and typed action IDs. Both baseline and candidate receive the **same policy metadata**. It overrides a preference only when marked active; invalid sources must ABSTAIN, not use an oracle gold answer.

## Locked training
For each of seeds 11,19,23, train exactly the V9.3 48D personal and project GRU and V9.8 isolated action heads using original **V9.4 TRAIN** English synthetic data, with the same family-heldout TRAIN dev checkpoint selection. No V9.11 family, gold label, extra note, signature or test option enters optimization. Language model is the same frozen English-focused 384D MiniLM as V9.10, so Hebrew generalization is not assumed.

## Fresh V9.11 evaluation
- 128 fresh synthetic researcher-authored EN/HE counterfactual pairs: 64 EN + 64 HE; 64 personal + 64 project; 16 novel task-family names/disjoint natural-language task texts vs V9.4–V9.10; 32 pairs per policy status `open`, `active`, `revoked`, `expired`. Each pair changes ONLY the authoritative personal/project preference slot; other scope's history and five plain history events are unchanged.
- Existing v9.10 cases may be evaluated as **retrospective only** and are not included in cold acceptance.
- Slot ordering: all variants same records, but nonbinding context (reason, historical rule, and extra facts) may precede, follow, or interleave with the preference slot. Test deterministic order invariance on every pair under equivalent slot permutations.
- Old preference can be explicitly superseded by a new version. Retrieving current preference by `slot_id+version+supersedes` uses provided provenance, **not** autonomous language inference. Include 32 shuffled/superseded record challenges (derived from cold cases; not independent test samples). No slot is allowed to change simply because its mention is newest.
- Compare `old_full_history_gru`, `old_full_history_gru+same_mock_policy`, `new_slot_memory`, `new_slot_memory+same_mock_policy`, and `first_three_history+same_mock_policy` (position shortcut). Add privileged complete-gold check **only as fixture sanity**, not real model.
- Inference must not read fixture `gold`, `expected_flip`, `language`, `family`, `swapped` or `policy_status`. Policy controller reads explicit `policy_contract` which is a simulated trusted structured input, never fixture answer key.
- Maintain all 3 model seed runs and complete matched casewise predictions.

## Preregistered conjunctive gates on EACH seed
1. Full 128 test cases have valid mock owner routing, and all 32 missing/corrupt source cases return ASK.
2. Slot metadata stress: equivalent permutations and superseded-old-record insertions change **zero** of 256 paired original/changed decisions; wrong/ambiguous slot provenance triggers ASK on 32 manipulated cases.
3. On 128 fresh paired dilemmas, slot+same-policy earns **>=40/128** and at least **+20 paired** vs old full-history+same-policy.
4. For 64 English fresh pairs, slot+same-policy earns **>=36/64**. For 64 Hebrew fresh pairs, report separately; **>=16/64** is the provisional floor (not a passing human-language standard).
5. For 32 active MOCK obligations, both candidate and baseline-with-policy enforce **32/32 paired**; canceled/expired mock obligation never overrides the preference.
6. Source-dependent user/project isolation: **0/128 unexpected decision changes** when unrelated scope history changes.
7. Candidate option-order symmetry >=95%.
8. Structured slot representation must be explicitly logged with owner, source, version and reason (without logging real user PII); weights and per-case breakdown archived.

If all pass: `PROVISIONAL_SYNTHETIC_COMPASS_SLOT_MEMORY_PASS`, still **NO IN-MODEL SELF INTEGRATION**. A fail remains `NO_GO_COMPASS_REAL_WORLD`. English-focused encoder and synthetic binary categories are major limits. Do not retune after cold outcomes.

## Evidence and limitations
This demonstrates structured memory separation under **declared provenance**, not the model learning to create accurate source labels automatically, not independent human intent comprehension, and not secure signing. True durable COMPASS requires validation of the original user/project vision in real projects, explanation-backed updates, consent, uncertainty, a multilingual encoder and evaluation outside these authored one-bit dilemmas.
