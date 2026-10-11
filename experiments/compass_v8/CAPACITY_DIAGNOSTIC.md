# COMPASS V8 — balanced three-capacity diagnostic (registered before results)

**Triggered run:** [GitHub Actions 38099765087](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38099765087). The outcome is **pending** when this plan is authored.

## Why a new diagnostic?

The first rank-4 smoke trained successfully but predicted `DO_NOW` on **all 16** held-out tasks. That superficially yielded 8/16 correct, equal to a trivial constant-action majority predictor. Its six action codes were ranked by separate mean cross-entropies where the 512-token prefix window could differ with answer token length, and the evaluation used only four authored case kinds. This does **not** validate contextual priority decisions.

## Corrective changes

- **Identical full chat prefix** for all six answer-code candidates, with a hard maximum length check. No per-answer prefix truncation. The tokens of each action are scored at their true next-token positions. Calculate **mean token log probability (primary)** and **sum token log probability (diagnostic)**, excluding answer EOS to avoid candidate-specific termination bias.
- **Class-balanced training:** 60 successive supervised updates, ten examples of each of six actions. The same authored 64-example train corpus and sampling protocol are used for every rank and seed.
- **Evaluation:** all **32** held-out examples from four unseen planning-project graphs and all eight authored case kinds; no selection of only four interesting types.
- **Constant-action baseline:** report majority gold label, trivial accuracy, balanced accuracy, and prediction diversity. A model that emits one label can no longer be presented as evidence of conditional reasoning.
- **Counterfactual-like pairs:** per project, compare broad optional tests with a targeted regression test critical for fixing the core. Both share *ancillary structural layer* but require different immediate actions; record the number of pairs answered correctly.
- **Hard decision boundaries:** log cases where an unapproved core mission change or explicitly forbidden peripheral scope is promoted to action.
- **Per-class recall:** all six action types, prediction histogram, balanced accuracy, and action-code/token-length sensitivity.

## Nine independent optimizer runs

| Adapter rank | Weights (hidden=960) | Seeds | Training updates |
|---|---:|---|---:|
| 4 | 7,680 | 11, 19, 37 | 60 |
| 16 | 30,720 | 11, 19, 37 | 60 |
| 64 | 122,880 | 11, 19, 37 | 60 |

Frozen backbone is `HuggingFaceTB/SmolLM2-360M-Instruct`. Each job compares four conditions on identical contexts: frozen backbone+project graph, frozen+graph+explicit user policy, trained adapter+graph, and trained adapter+graph+explicit policy. The authored rule-engine sanity baseline is included.

The workflow runs shared dataset and scoring sanity checks before the matrix. All nine matrix jobs are eligible to run in parallel (subject to GitHub's actual available runners), and the final aggregation requires all nine successes. Each job retains its model-adapter weights and full action-code log probabilities in the artifact for later audit.

## Prespecified interpretation constraints

1. **Technical success is not research success.** If COMPASS still predicts one action, report failure regardless of headline accuracy.
2. A **5-percentage-point gain in joint decisions over frozen+explicit-policy** and no worsening on safety/scope traps would be a *preliminary* signal only; this matrix scores action codes rather than full project+layer+action explanations.
3. Same held-out projects across all seeds/ranks are **paired repeated measurements**, not nine user cohorts.
4. Each planning project is authored with a restricted synthetic text grammar and belongs to a project family unseen in training, but *all four fictional policies are shared across train and test*. No cold-user transfer is tested.
5. Authored rules attaining 100% here are not an independent evaluator. We need a separate gold-label adjudication exercise, naturally authored task histories and no superficial lexical shortcuts before using this in a product.
6. Equal update counts do **not** ensure equal FLOPs or wall-time. This is the matched-update phase only; proper compute-matched comparison remains a separate experiment.
7. `DO_NOW` bias may be addressed by sampling balance, but that does not prove the model learned conditional priorities. Counterexamples and per-class recall are the decisive checks.
8. Tests use entirely fictional data. COMPASS is **personal prioritization**, not SELF's model-internal self-representation. Nothing in the native 5M SELF research is changed.

**State:** matrix dispatched, outcomes and checkpoints not yet reviewed.
