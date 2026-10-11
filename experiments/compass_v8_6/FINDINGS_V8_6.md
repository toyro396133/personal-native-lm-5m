# COMPASS V8.6 — completed source-validity and owner-policy counterfactual pilot

**Technical status: SUCCESS. Scientific gate: NO-GO to expanding sizes/production.**
[GitHub Actions run 38110217103](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38110217103), attempt 1, completed successfully 2026-10-11 04:09 UTC. Preflight, reusable frozen-feature extension and personal rank-4 training all passed. Original artifact **11690768329**, `compass-v86-r4-s11`, contains raw per-decision JSON (`compass-v86-r4-s11.json`) and the 7,680-weight `.safetensors` checkpoint. Individual predictions and the source/evidence splits were checked against the original JSON, not inferred from logs. The full shared frozen embedding artifact is ID **11691119625**.

## Precise experimental configuration

- Frozen common backbone `HuggingFaceTB/SmolLM2-360M-Instruct`. Train **only the personal COMPASS rank 4 = exactly 7,680 weights**, seed **11** for **14 epochs**, selecting **epoch 13** using the development set exclusively (152/160 correct; 36/40 all-four groups). V8.5 rank16=30,720 and rank64=122,880 remain preserved but NOT run in V8.6.
- Train: **800 synthetic bilingual decisions** from 5 fictional profiles. Development: **160** from a different fictional profile. Locked test: **176** authored bilingual decisions from two fictional held-out projects, **88 English / 88 Hebrew**. 44 counterfactual state×choice-order quartets, 88 fact-flip pairs.
- Add two source-expiry scenario families to training (certificate expires, report superseded). Add two **new, test-only** families (delegated authority revoked, conflicting verified sources), **32 total**. Preserve V8.5 **source_retracted** family as a *previously inspected diagnostic* (16), not cold independent validation.
- Separately evaluate **16 policy-only paired probes** in neutral evidence conditions: same fictional project, same objective, same candidate actions and same evidence; alter only the authorized rapid ↔ thorough policy text that is visible to the model. Policy-source IDs and versions are also changed in hidden metadata but not prompt context.
- All situations, authorizations, and golden labels are researcher-authored, not independently human-adjudicated. Matching real projects, source validity from actual provider records, autonomous explanation, and long-term evidence updating were NOT tested.
- Explicit deterministic authorization/scope rules and a synthetic `source_valid` executor-only guard are **external control conditions**, not weights or capabilities learned by the neural model.

## Raw and guarded scores

| Condition | Correct /176 | Accuracy | Correct fact reversals /88 | Complete quartets /44 |
|---|---:|---:|---:|---:|
| Frozen common LM with same-input per-candidate Yes/No logits | **86** | 48.9% | 12 | 6 |
| **Personal COMPASS rank4 raw** | **148** | **84.1%** | **60** | **30** |
| COMPASS + preexisting external approval/scope guard | **148** | **84.1%** | 60 | 30 |
| COMPASS + external approval/scope + synthetic source-validity guard | **162** | **92.0%** | 74 | 37 |

The external source-validity rule **overrode 14 actions**, improving apparent accuracy **148→162**. This gain is **NOT neural learning** and is grounded in authored oracle-linked fake `source_valid` metadata. In particular, for the previously examined source-retraction family this rule corrects all 8 invalid-state errors. Real source checking requires authenticated, current provenance, trust/version verification and enforced action authorization. Do NOT cite 92% as stand-alone model intelligence.

Raw language and fictional priority mode subgroups:

| Subgroup | Raw correct |
|---|---:|
| English | **82/88 (93.2%)** |
| Hebrew | **66/88 (75.0%)** |
| rapid fictional policy | **70/88 (79.5%)** |
| thorough fictional policy | **78/88 (88.6%)** |

Because the two heldout projects encode different baseline policies, their raw subgroup accuracy cannot by itself prove actual within-project personalization. Only the paired policy-only probe below isolates the preference intervention.

## Source-validity analysis: model fails on invalid cases

| Family | Raw correct /16 | Valid source state 0 /8 | Invalid/revoked source state 1 /8 | Status |
|---|---:|---:|---:|---|
| NEW delegated-authority revocation | **12** | 8 | **4** | Partially learned |
| NEW conflicting-source evidence | **14** | 8 | **6** | Partially learned |
| Previously inspected `source_retracted` | **8** | 8 | **0** | NO improvement |

**Two newly held-out families combined: 26/32 (81.25%), BUT validity-conditioned view is 16/16 valid vs only 10/16 invalid.** This **fails** the prespecified minimum **12/16 invalid**. Explicit authority revocation in Hebrew gets **0/4** incorrect-state responses right; English authority revocation gets **4/4**. For contradictory sources, Hebrew gets **2/4** invalid-state responses correct, English **4/4**.

The original previously failed source-retraction family remains **8/16 (50%)**: all eight valid-source cases correctly executed, **all eight withdrawn-source cases incorrectly executed**. Adding related expiration/version examples in training did **not** make raw COMPASS generalize reliably to this previously inspected form. Those cases should not be rebranded a blind test.

The shared personal scorer's invariance to swapping candidate positions is guaranteed by design, not proof that changing evidence is understood. Correct *fact reversal pairs* overall were 60/88; complete 30/44 groups.

## Controlled same-project personal-policy changes: 8/16, not yet reliable

**Policy-only probe results:**

| Policy-only intervention | Both old/new decisions correct | Raw choice actually flips |
|---|---:|---:|
| English (8 pairs) | **6/8** | 6/8 |
| Hebrew (8 pairs) | **2/8** | 2/8 |
| **Combined (16 pairs)** | **8/16 (50%)** | **8/16 (50%)** |

This misses both prespecified targets (≥10/16 both correct and ≥12/16 policy-induced switches). English rollout preference did not switch on the rapid-project policy twins; Hebrew rollout and one Hebrew coverage intervention did not switch correctly. Raw model often continues to choose according to the fixed project wording or task lexical features despite changed owner preference, especially in Hebrew. Policies are fictional and only priority-sensitive when evidence is neutral; binding state 1 must remain overriding.

## Prespecified go/no-go gate

| Gate | Threshold | Observed | Verdict |
|---|---:|---:|---|
| Raw test accuracy and matched frozen improvement | ≥130/176 and > frozen | **148/176 > 86/176** | PASS |
| Two NEW source-validity families combined | ≥24/32 | **26/32** | PASS |
| Within new families: revoked/invalid specifically | ≥12/16 | **10/16** | **FAIL** |
| Previously inspected V8.5 source withdrawal diagnostic | ≥12/16 | **8/16** | **FAIL** |
| Policy-only both original and changed decisions correct | ≥10/16 | **8/16** | **FAIL** |
| Policy-only raw preference switches | ≥12/16 | **8/16** | **FAIL** |
| Hebrew overall | ≥60/88 | **66/88** | PASS |
| External source status/authorization enforcement | Do not execute with known invalid provenance | Source validity prototype rectifies 14 predictions; actual secure provenance NOT tested | Prototype-only |

**Scientific verdict: NO-GO.** The general benchmark is promising, but the diagnostic objectives of reliable invalid-source recognition and policy-only update did not pass. In this experiment **none of the 30,720/122,880 variants were launched**, as they were gated on small-model success.

## What to do next without overclaiming

1. Keep **7,680** as the experimental efficiency baseline, while keeping all other V8.5 learned-weight checkpoints.
2. **Separate external source authority enforcement as a mandatory non-neural system contract**, not a text-only preference: track authoritative issuer, verification time, signature/version, revocation/supersession and effective decision time. Fail closed on unknown or invalid status. Synthetic `source_valid` from test oracle must not masquerade as real provider verification.
3. Build a new **policy-only matched training objective** (same project, same evidence, one policy change only) and test on independently authored policy pairs. Measure actual change of semantic action, not just label preference or global accuracy. This experiment had supervised examples of contrasting policies on DIFFERENT projects; paired same-project preference switch was only in testing.
4. Add explicit evidence validity feature representation and time/version reasoning, plus multilingual Hebrew source-invalidity examples with controlled paraphrase. Test separately by language and time state.
5. Require external/human-adjudicated, non-templated project scenarios and strong symbolic/graph comparison before claiming general-purpose personalization. Do not repeatedly revise or tune using the same exposed V8.5/V8.6 test cases as if they were blind holdouts.
6. Only then rerun multiple seeds and consider scaling personal-model parameter count; number of weights alone did not cure the source retraction case in V8.5.

## Research isolation and artifacts

- [V8.6 protocol](EXPERIMENT_V8_6.md), [fixture generator](fixtures_v86.py), [trainer and audit](pilot_v86.py), [preflight](validate_v86.py).
- GitHub Actions run 38110217103, artifacts: `compass-v86-preflight` (ID 11690699187), `compass-v86-shared-embeddings` (ID 11691119625) and `compass-v86-r4-s11` (ID 11690768329).
- Isolated experimental branch `experiment/compass-v8-6`, **no changes to native 5M SELF or production**.
