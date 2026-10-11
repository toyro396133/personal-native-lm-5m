# COMPASS V9.7 — results: provisional structured-policy validation, NOT neural-authority learning

**Workflow:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38126570686
**Completion:** GitHub Actions `success`, run attempt 2, 2026-10-11.
**Per-case JSON + learned bridge weights:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38126570686/artifacts/11697162675
**Research verdict:** `PROVISIONAL_STRUCTURED_SYSTEMS_PASS_NOT_NEURAL_AUTHORITY`.
**No 360M native neural adapter integrated.**

## Reproducible history

1. The preregistered V9.7 protocol was committed BEFORE cold predictions.
2. Preflight attempt #1 found a **fixture-status naming mismatch**, expected `ENFORCED` vs policy runtime `ENFORCE` for 32 active cases. No neural cold decision was evaluated on this attempt. Fixed the status name only, commit `18777e84eb42c0486a8430074e9557156c9f083b`. Task texts, ground truth, registered numerical gates and model code were not changed.
3. Attempt #2 passed the preflight 256-case policy routing and adversarial checks, then trained frozen V9.3 memory + V9.4 bridge from TRAIN only and scored all 256 synthetic holdout pairs.
4. No test cases used to train, tune or select checkpoint weights. The policy controller itself is deterministic, does not learn, and uses externally supplied synthetic trusted keys and a revocation ledger.

## Overall 256-pair benchmark — both original and changed principle must be correct

| Method | BOTH-correct pairs | Percent |
| --- | ---: | ---: |
| Evidence-aware hybrid (frozen learned preference bridge + external policy verifier) | **225/256** | **87.89%** |
| Frozen learned bridge without policy evidence | **101/256** | **39.45%** |
| Same claim/action records, deliberately discarding validity/provenance (unsafe ablation) | **64/256** | **25%** |

The differences are **not a neural learning result**: the hybrid has a hard-coded, trusted (mock) source registry, authenticated policy records, revocation ledger and typed action IDs. The neutral cases are handled by the same learned bridge in all three systems. The naive ablation is intentionally wrong about trust and not a tuned safety baseline.

## Breakdown by policy scenario (32 pairs per condition)

| Condition | Evidence-aware BOTH-correct | Notes |
| --- | ---: | --- |
| No claim | 25/32 | Intent preference via neural bridge |
| Verified active obligation | **32/32** | Deterministic typed-action constraint |
| Authenticated obligation revoked by trusted ledger | 25/32 | Old record correctly ignored |
| Expired authentic obligation | 24/32 | Old requirement correctly ignored |
| Verified policy for unrelated project | 23/32 | Irrelevant requirement ignored |
| Unknown issuer / unverified claim | **32/32 abstain** | Never silently accepts dubious authority |
| Forged signed claim | **32/32 abstain** | Invalid MAC never becomes authority |
| Two contradictory valid active requirements | **32/32 abstain** | Never silently picks one conflicting obligation |

Policy routing: **256/256 expected policy states**, 0 false active requirements, 0 unnecessary abstentions, 0 missed uncertain abstentions.
Adversarial mutations: **6/6 pass** (tampered action, issuer, signature, scope, duplicate candidate types, unknown candidate types).
Project isolation test: **0 unexpected changes in 32** when only unrelated project history changes.

## Major newly exposed failure: personal/global preference generalization

The overall neutral-personal/project score **97/128 (75.78%)** conceals a large imbalance:

| Neutral user/project identity | BOTH-correct |
| --- | ---: |
| **Personal preference context:** 34/64 (**53.13%**) |
| **Project intention context:** 63/64 (**98.44%**) |
| Combined | 97/128 |

Derivation: 128 cases in each context, each includes 48 unverified/conflict abstentions + 16 verified active enforcement, all of which pass, leaving 64 truly preference-driven cases per context. Thus 98/128 user successes - 48 - 16 = **34/64**; 127/128 project successes - 48 - 16 = **63/64**.

This is **not** a claim that the model handles real projects with 98% accuracy: researcher-authored synthetic binary project tasks were much easier for these embeddings/this seed. In particular, **the personal COMPASS layer still fails almost half of unfamiliar neutral paired user dilemmas**, contrary to the long-term goal of understanding user preference across projects. Successful structured permission handling must not camouflage this deficiency. Treat as a warning beyond the original preregistered gates, NOT as a post-hoc change to pass/fail rules.

## All 9 preregistered gates

| Gate | Outcome |
| --- | --- |
| 96/96 unknown/forged/conflicting policy pairs abstain | PASS |
| 32/32 verified active binding pairs comply | PASS |
| 0/128 unnecessary abstentions on no claim/revoked/expired/other project | PASS |
| Neutral preferences >=70/128 BOTH-correct | PASS (97/128) |
| Neutral preference intent-flip agreement >=70/128 | PASS (97/128) |
| Unrelated-project isolation 0/32 | PASS |
| Correct verified policy routing, 0 unexpected enforcement | PASS (256/256) |
| Provenance verified vs unsafe same-records ablation gain >=60 uncertain pairs | PASS (+96/96) |
| All controller tampering tests pass | PASS (6/6) |

**Strict registered verdict:** provisional synthetic **structured systems** PASS. Do not rebrand it as native 360M SELF adapter, learned permission validity, independently validated project intention or safe production autonomy.

## Why this particular experiment matters

Previous V9.5 and V9.6 used a neural language detector to decide what was mandatory. This failed on genuine unseen phrasings, even after robustness training. V9.7 instead puts **validity outside the personal/project preference learner**: verify trusted signed records, current effective period, revoked state, project scope and candidate operation types, and refuse unsafe ambiguity. In a completely controlled synthetic store, this works. It is stronger *systems architecture*, not better evidence that an LLM can identify a real obligation from untrusted prose.

## Proposed V9.8, not started yet

**Priority A — personal intent:** hold project scope separate and improve transfer of **personal/global preference** to new tasks, using family-held-out development and new independently authored tasks. Target at least 75% BOTH-correct per scope, not merely pooled. Compare frozen prototype/keyword rule baselines with strictly identical inputs and count task-conditioned extra-data benefit. Multiple seeds and real corrections with provenance are necessary.

**Priority B — validated operations and trust:** current V9.7 assumes truthful typed operation IDs from trusted caller. Test malicious or mistaken action tagging and apply policy controls at action executor boundary; never equate the signed old record with the current real-world entitlement. Build real authenticated source integrations, signed revocation evidence, issuer authorization and user consent before production.

**Priority C — dynamic concept memory:** later test cross-project knowledge learned through actual sustained work, not just three synthetic English history statements. Project vision/intent should remain distinct from changing implementation; personal learning should generalize between projects without leaking project-specific binding commitments.

None of these justify native LM integration yet. Synthetic policy demo passes, **core personal adaptive intelligence remains far from established**.
