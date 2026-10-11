# COMPASS V9.9 — completed results and remaining scientific gap

**GitHub workflow:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128282105
**CI:** completed SUCCESS (preflight, seed 11, seed 19, seed 23, aggregate).
**Aggregate artifact:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128282105/artifacts/11698290236
**Per-seed JSON + separately trained readout weights:**
- Seed 11: https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128282105/artifacts/11698320190
- Seed 19: https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128282105/artifacts/11698430664
- Seed 23: https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38128282105/artifacts/11697990670

**Strict preregistered scientific verdict:** `PROVISIONAL_SYNTHETIC_SOURCE_ROUTING_PASS_NOT_SELF_MODEL` for **all 3 seeds**.

**CRITICAL INTERPRETATION:** This is NOT real-life accuracy, NOT neural discovery of intention ownership, NOT source authentication for a real user, NOT native SELF integration with frozen 360M. Source metadata is an input externally declared in the fixture and matched to gold by its researcher. This is partly a test of passing a label through a validated control plane. Do not use it to claim 100% human-project intention understanding.

## Why V9.9 exists

V9.8 had personal transfer 47/64,47/64,35/64 across seeds; question-only scope-router fell to 112/128 in seed23, and resulting project memory leaks affected personal decisions. V9.9 explicitly separates ownership declaration from the learned choice scorer. The UI source metadata identifies `PERSONAL_PROFILE_DECISION` versus `WORKSPACE_GOAL_DECISION`. A deterministic signed-record router selects a personal-only or active-project-only numeric readout, or returns ASK. Mock keys committed to the repository are deliberately public and provide no real authentication.

The numeric GRU memory and isolated neural readout still learn binary concept-to-option mapping from **V9.4 TRAIN only**; V9.9 new family test questions were not trained or used to tune hyperparameters. The task generator constructs origin metadata to agree with the research author's intended scope. That fact makes this test easier than autonomous routing from genuine unstructured conversations.

## Main results: matched 128 new synthetic pairs, BOTH original and changed principle correct

| Seed | Personal source-routed /64 | Personal question-routed same head /64 | Personal source-routed numeric prototype /64 | Project source-routed /64 | Total source-routed /128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 | **64** | 56 | 56 | **64** | **128** |
| 19 | **64** | 64 | 56 | **64** | **128** |
| 23 | **64** | 56 | 56 | **64** | **128** |
| All seed-runs (not independent human samples) | **192/192** | **176/192** | **168/192** | **192/192** | **384/384** |

On identical V9.9 cold tasks, explicit simulated source routing avoids lost personal decisions when the question-only scope classifier misroutes. The same isolated personal readout also performs better than the numeric prototype on V9.9's author-generated tasks. **This is not an apples-to-apples claim against V9.8's accuracy on its own different test distribution.** V9.9 new personal histories and action wordings happen to be easy for the learned head across seeds; the task source field is a constructed ground-truth-like label.

Control fixed-option A is 0/128 paired BOTH-correct because each neutral counterfactual requires the output to flip.

## Provenance and noninterference audit

- Verified signed mock request-origin selection: **128/128** for each seed.
- Missing/corrupted/conflicting/wrong-workspace source metadata: **96/96 pairs => ASK** in each seed.
- Personal decisions unchanged when other project histories were altered: **0 changes/128** in each seed.
- Personal decisions unchanged when active workspace A/B changed: **0 changes/128** in each seed.
- Project decisions unchanged when unrelated personal history changed: **0 changes/128** in each seed.
- Shared candidate scorer permutation equivariance: **128/128** in each seed.
- Four ASK scenarios each use 24 cases from the main set, transformed after primary case construction. They are NOT independent tasks.
- The test keys are not securely managed, no identity attestation is performed, and the mock UI event is selected to agree with research `case.scope`.

## Registered conditions (all PASS each seed)

| Gate | Seed 11 | Seed 19 | Seed 23 |
| --- | --- | --- | --- |
| Verified origin routes correctly 128/128 | PASS | PASS | PASS |
| Ambiguous/missing/bad origin triggers ASK 96/96 | PASS | PASS | PASS |
| Personal unrelated-project history changes 0/128 | PASS | PASS | PASS |
| Personal workspace switch changes 0/128 | PASS | PASS | PASS |
| Project unrelated personal history changes 0/128 | PASS | PASS | PASS |
| Personal same-head source-routed >=48/64 | PASS (64) | PASS (64) | PASS (64) |
| Personal gain over question-only route if wrong >=8; otherwise already strong with >=48 | PASS (+8) | PASS (question route already 64) | PASS (+8) |
| Project >=48/64 | PASS (64) | PASS (64) | PASS (64) |
| At least one same-origin personal readout or prototype >=48/64 | PASS (both 64,56) | PASS (64,56) | PASS (64,56) |
| Candidate option swap >=95% | PASS | PASS | PASS |

The scripted workflow correctly reports `PROVISIONAL_SYNTHETIC_SOURCE_ROUTING_PASS_NOT_SELF_MODEL`; passing V9.9 does **NOT** remove the broader `NO_GO_TO_LM_ADAPTER_INTEGRATION` recommendation.

## Strong conclusions we can support

1. **Explicit source and workspace separation prevent synthetic cross-talk**, given correctly issued source metadata. We should preserve this architecture regardless of how the neural preference weights later evolve.
2. A clean interface must treat missing, contradicting, revoked or otherwise untrusted identity/scope metadata as a reason to seek clarification, not quietly import a project-specific preference into the personal model.
3. The underlying small neural heads can choose different actions for opposed synthetic 3-message histories in these new task wordings. The unknown generalization performance for genuine users and long-running project goals remains open.
4. When comparing experiments, distinguish: (a) task ownership externally declared in a source field (V9.9), (b) autonomously inferred owner from new situation text (V9.8), and (c) semantic preference strength conditional on true owner. An intervention in one layer does not prove mastery of another.

## Scientific cautions about the near-perfect score

The score is **unexpectedly perfect** on an intentionally simple binary synthetic axis, and must motivate harder independent testing rather than declaring completion. The question descriptions all explicitly mark personal or project context; options repeat 'review risk vs small probe' or 'mission versus public contract'. Cold histories paraphrase the same conceptual categories as TRAIN and have only three sentences. V9.9 task labels and owner metadata are researcher-authored together. Three seeds do not create independent sampling of new people's projects.

The signed origin event is assumed generated by a trusted UI and its claim's scope matches the intended gold. In a real agent, an ambiguous task cannot safely be made personal or project-specific just because a convenient parameter was inserted. The source-control interface still needs actual user authorization, workspace membership, identity verification, policy revocation, auditability and applicability conditions.

## Next V9.10 research — recommendation, NOT launched

1. **Blind evaluator separate from training author**: real user/project scenario descriptions and ambiguous mixed personal/project goals. Require independent labels and explanations; use fully held-out scenario families and source-signing from real origin audit. At minimum add substantive **Hebrew** alongside English.
2. **Multiple concurrent principles and conditions**: personal global preference can be conditional on project type, risk, budget, quality, time; project purpose can override a personal habit but never override verified safety/contract obligations. Include contradictory, outdated and evolving principle histories spanning more than three updates. Test whether model preserves underlying project VISION distinct from implementation.
3. **No fabricated scope oracle**: only real source events issued through authenticated controls may select the active memory scope. Uncertain applicability should trigger ASK; report *selective risk/coverage* rather than forcing binary answers. Test spoofed scope from prompt and user-controlled action tags.
4. **Provenance-aware memory records**: source, author, timestamp, reason for decision, conditions of applicability, validity/revocation, evidence, explicit boundary of which projects may inherit a personal principle. Learn models only after handling these explicitly.
5. **Matched strict baselines** including full-history text prompt to actual frozen language model, numeric prototype and retrieval controls on the same human scenarios; record real compute/token cost. Reproducible multi-seed experiments with untouched cold splits before testing the actual in-model SELF adapter.

**Status:** V9.9 finished and documented; no production code or existing 360M language model modified. All experiment code stays on `experiment/compass-v9-9` (not merged to main); main has only the trigger workflow.
