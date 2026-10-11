# COMPASS V9.8 — completed three-seed report: better, but NO-GO

**Research protocol:** [EXPERIMENT_V9_8.md](EXPERIMENT_V9_8.md)
**GitHub Actions run:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38127430760
**Full three-seed aggregation artifact:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38127430760/artifacts/11697079032

Per-seed detailed JSON + separate head weights:
- **Seed 11:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38127430760/artifacts/11697234841
- **Seed 19:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38127430760/artifacts/11697518956
- **Seed 23:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38127430760/artifacts/11696879389

**CI verdict:** GitHub Actions success for the corrected attempt (valid artifacts and full evaluation).
**Scientific verdict:** `NO_GO_TO_LM_ADAPTER_INTEGRATION`, every seed fails some predeclared success gates.
**No native frozen 360M LM adapter was deployed, integrated or evaluated in this experiment.**

## Protocol and one preflight correction

The original preflight caught an error **before any V9.8 training or cold evaluation**: the selected neutral-index pattern, combined with a family-index expression, exercised only 4 of 8 heldout families in each scope. Fixed the test-family distribution in commit `64b866589b45df3bc7e7abf1c605748e0cb81ecd`, reran preflight, confirmed **all 16** new families and the option-order-symmetric trainable scorer, then launched three independent-seed jobs. This change was done before ANY new V9.8 cold prediction; no after-the-fact test tuning.

Source tasks and labels are still **researcher-authored synthetic binary concepts**. Three seeds vary random initialization and examples, but they are not independent human task domains.

## Matched new-task results — BOTH original and flipped principle must be right

| Seed | V9.4 personal /64 | V9.8 personal /64 | V9.4 project /64 | V9.8 project /64 | V9.4 all /128 | V9.8 all /128 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 11 | 39 | **47** (+8) | 62 | **64** (+2) | 101 | **111** (+10) |
| 19 | 40 | **47** (+7) | 56 | **64** (+8) | 96 | **111** (+15) |
| 23 | 29 | **35** (+6) | 62 | **64** (+2) | 91 | **99** (+8) |
| Total across 3 non-independent synthetic runs | **108/192** | **129/192** | **180/192** | **192/192** | **288/384** | **321/384** |

- Overall candidate is stronger than matched V9.4 joint bridge for every seed.
- Personal gains (+8,+7,+6) are positive but not consistently large, and NONE of the three candidate scores reaches the preregistered **48/64** minimum.
- Project scope is perfect in these synthetic examples, **not** evidence of real-life perfect project intent or natural-language task understanding.
- No-history fixed option A achieved **0/128 paired BOTH-correct** (counterfactual principle always demands a flip).
- Equal-input numeric prototype reference personal results: **42/64, 47/64, 41/64**. In seed 23 it beats the deployed V9.8 routed personal head (**41 vs 35**). This is important: a simple memory prototype sometimes beats the new learned component.

## Scope inference is a separate dominant failure

| Seed | Task-scope router | Personal with learned router | Personal with oracle scope (UNFAIR diagnostic) |
| --- | ---: | ---: | ---: |
| 11 | 128/128 | 47/64 | 47/64 |
| 19 | 120/128 | 47/64 | **54/64** |
| 23 | **112/128** | 35/64 | **47/64** |

The oracle diagnostic uses `case.scope` and cannot be shipped as a model result. It shows a measurable **loss caused by incorrect scope routing**. The controlled source identity (where a memory came from: personal or project) is a better candidate for reliable routing than incidental words in task descriptions. This does NOT make the natural-language problem of resolving mixed/ambiguous task ownership disappear.

## Separation and invariant controls

| Measure | Seed 11 | Seed 19 | Seed 23 |
| --- | ---: | ---: | ---: |
| Personal decisions altered by changing unrelated project histories (128 sides) | 0 | **8** | **16** |
| Project decisions altered by changing unrelated user histories (128 sides) | 0 | 0 | 0 |
| Personal decisions altered by switching active workspace A/B (128 sides) | 0 | **4** | **16** |
| Personal pairs BOTH-correct under opposed-project history /64 | 47 | 52 | **40** |
| Candidate option permutation symmetry | 128/128 | 128/128 | 128/128 |

Although the **personal-only head has no project-state input** and the **project-only head has no personal-state input**, a wrong scope-route can bring in the other state's scorer and thereby violate observed isolation. The architectural separation works only conditional on correct routing; the composed system must be tested as a whole.

## Strict preregistered gates by seed

| Gate | Seed 11 | Seed 19 | Seed 23 |
| --- | --- | --- | --- |
| Personal >=48/64 | FAIL 47 | FAIL 47 | FAIL 35 |
| Personal gain >=8 pairs vs matched V9.4 | PASS +8 | FAIL +7 | FAIL +6 |
| Project >=48/64 | PASS 64 | PASS 64 | PASS 64 |
| Project loss at most 4 pairs | PASS | PASS | PASS |
| Router >=120/128 | PASS 128 | PASS 120 | FAIL 112 |
| Personal unaffected by project-history noise | PASS | FAIL 8 changed | FAIL 16 changed |
| Project unaffected by user-history noise | PASS | PASS | PASS |
| Personal unaffected by workspace switch | PASS | FAIL 4 changed | FAIL 16 changed |
| Opposed-project personal score >=45/64 | PASS 47 | PASS 52 | FAIL 40 |
| Candidate order symmetry >=95% | PASS 128/128 | PASS 128/128 | PASS 128/128 |

**Conjunctive verdict: NO-GO (0/3 seeds pass all).** Do not raise success based on pooled averages or nearest seed.

## What was actually learned

1. TRAIN-only and family-heldout-dev optimization of small separate heads on frozen numeric personal/project GRU states improved the outcomes on the new researcher-authored synthetic dilemmas relative to the original joint bridge; personal concept transfer remains inconsistent across seeds and below the acceptance level.
2. Reliable cross-project isolation is not solved by disconnecting tensors in separate heads alone. **Ownership routing** must also be reliable, or an inference-time gate will leak irrelevant state through a different head.
3. The project head's strong performance masks ongoing problems in personal generalization. The V9.7 observed 34/64 personal results remain a useful warning. We should not average across scopes to claim adequate long-horizon self-model intelligence.
4. The numeric prototype is competitive and even stronger for personal intent in one seed. Any proposed extra complexity must keep beating cheap controls on genuinely novel human-created cases.

## Next V9.9 recommendation (not launched)

- **Source-aware scope policy**: use trusted metadata about origin and authority of a preference/principle (global user versus named project), and deliberately support **`UNKNOWN/ASK`** for ambiguous new tasks. This is not permission to pass gold fixture scope as an oracle; task ownership must be grounded in realistic available workflow data and explicit user/project instruction with uncertain cases.
- **Personal-memory learning**: compare direct learnable 48D state, simple state/option prototypes, and explicit concept/reason retrieval on independent unfamiliar personal task families. Preserve a separate *reason, provenance, confidence and validity conditions* record of why a preference was established and when it may not apply.
- **Adversarial cross-project evaluation**: prove a personal principle learned on one project can appropriately transfer to another when its applicability conditions hold, and **must not transfer** when it was only a project-specific requirement.
- **Independent human-written evaluation**, multiple languages (especially Hebrew), multiple independent people, longer history, multiple concepts, concept changes and contradictions over time. Current tests are three-message English synthetic histories.
- Keep V9.7 structured policy authority verification as a separate control plane, outside learned soft intent inference. Until an independent end-to-end realistic study, do not merge weak components into a frozen 360M LLM adapter.

Research scripts and results are on `experiment/compass-v9-8`; **not merged into main**. Workflow trigger exists on main solely to run that branch. Any further training using V9.8 cold cases makes them retrospective, not blind.
