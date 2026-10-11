# COMPASS V8.1 — contrastive priority policy pilot, registered plan

**Status:** implementation and validation, results pending. Distinct from model-internal SELF self-model; no native SELF changes.

## Problem from V8

In nine completed V8 runs (three adapter sizes × three seeds), the strongest rank 64 averaged 12/32 on a narrowly authored six-choice action benchmark. The **constant DO_NOW prediction** also scored 12/32. No rank correctly chose SCHEDULE, ASK_USER or REJECT on those evaluation sets, and unsafe promotion of forbidden-scope/unapproved work occurred. Six varying-length response strings remained sensitive to token length and label priors.

## V8.1 controlled changes

1. **Exactly six distinct single-token categorical IDs** `0,1,2,3,4,5`, mapped in a fixed order to `DO_NOW, SCHEDULE, INVESTIGATE, DEFER, ASK_USER, REJECT`. The program verifies each is one unique token for the frozen SmolLM2-360M-Instruct tokenizer before training. **Training and evaluation use the same six-logit softmax** (not training text/EOS with a different evaluation rule). Report raw argmax as the primary model score, and a preregistered frozen-backbone neutral-context calibration as a *secondary* diagnostic; do not cherry-pick.
2. **Paired contrasting training situations**: confirmed vs suspected failure; necessary targeted vs optional broad test; core-blocking vs routine prerequisite; authorized vs unapproved core change; authorized vs forbidden scope expansion; cosmetic deferral vs uncertain proposal. Each update sees two opposite-label tasks of one fictional project.
3. **Different fictional users and projects in the locked test**. Train has 8 fictional decision-policy profiles × 2 projects × 12 situations = **192**, test has 3 entirely disjoint fictional policy profiles × 2 disjoint projects × 12 = **72**. Pilot evaluates 2 of the six test projects = 24 items, including **12 paired contrasts**. All are English authored templates, and their gold labels and case IDs are separate from input examples. This increases split integrity but is not independent human validation.
4. **Authoritative permission and scope guard.** A separate deterministic program validates project node identity and task project ID, rejects explicit out-of-scope proposals, and asks the project owner for permission when required approval is not granted. The guard is applied **after** classification, and raw classifier errors are fully retained. *Guarded action accuracy must not be presented as evidence that COMPASS learned safety judgments*. Unknown scope/approval information is treated conservatively.
5. **Model comparison and evidence of genuine discrimination.** Compare frozen model vs learned rank-4 adapter, each with identical current graph+approved policy, on raw and calibrated single-token scores. Also show the constant-majority baseline, per-action recall, balanced accuracy, all paired counterfactual families and raw vs guarded scope violations. Save token IDs, neutral calibration vector, complete raw response probabilities, weights and every prediction for audit.

## Pilot scope and success gates

- Rank 4 = 7,680 weights; seed 11; **48 paired updates (96 labeled examples)**, with unchanged 360M backbone.
- Stage-0 structural validator MUST pass, including train/test user/project/template separation and real-source metadata, permission guard, pair validity, and cross-project rejection.
- Before rank expansion, require: **raw** test balanced accuracy above chance and constant-majority baseline; at least 4 of 6 action types predicted and measurable recall on `ASK_USER` and `REJECT`; at least half the counterfactual pairs correct as pairs, no claim of safety learned unless *raw* guard-category errors drop, and no hidden collapse.
- If a model still collapses to frequent labels or fails critical boundary understanding, repair objectives/data before any large rank-by-seed matrix. Passing deterministic authorization guard alone is not enough.
- Even if the pilot passes, this is only a synthetic classification result. External adjudication, full project version updates, validated decision rationales and production-grade deletion and permissions remain absent.

## Limits

- Authored rule labels and text templates are not independently assessed for clarity; no real user work, documents or private account data used.
- Tests are structured synthetic prompts, not free-form project-management tasks, and context evidence fields are authoritative by construction.
- The rank-4 pilot is diagnostic; **no claim about larger ranks** without a separate controlled run.
- The permission guard is a reference prototype only. Real product authorization requires trusted inputs and actual enforcement within the action executor.

## Files

- `fixtures_v81.py`: fictional contrastive inputs and independently stored gold.
- `guard_v81.py`: strict scope, approval and project integrity checks.
- `validate_v81.py`: no-leak, contrast and guard audit.
- `pilot_v81.py`: frozen 360M, matched categorical loss, paired updates, controlled evaluation.
- `.github/workflows/compass-v81-pilot.yml`: two-job guarded test and smoke study.

**Decision on V8.1:** run small diagnostic first; reserve larger-rank comparisons for evidence of semantic discrimination rather than simply increasing learned capacity.
