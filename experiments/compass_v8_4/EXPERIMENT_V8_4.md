# COMPASS V8.4 — shared semantic option ranker / bilingual diagnostic

**State:** protocol and implementation prepared; outcome pending. Separate from SELF (model-internal self-reference research).

## Motivation

V8.3's rank-4 residual updated only two of 96 decisions compared to frozen SmolLM2-360M (raw 50/96 vs 48/96), with just 2/48 correct fact-flip pairs, 2/48 correct order-flip pairs, **0/24 completely correct quartets**. The residual still picked option 1 in 46/48 English prompts and option 0 in 48/48 Hebrew prompts. Larger residual ranks are not justified without resolving this position/language collapse.

## Hypothesis

Instead of asking a causal language model to emit `0` or `1` after a bilingual two-option prompt, **encode each candidate action separately** with the frozen LM, jointly conditioned on the very same project objective, evidence and generic approved policy. Then train a **shared scalar option scorer** on differences between the two frozen representations. The scorer must not see the other candidate's order or numeric label.

That architecture automatically guarantees **order equivariance**: when options are exchanged, its signed score difference negates. This is a *structural property*, not evidence that the model learned semantics. The actual scientific gate is the harder question: does changing task evidence appropriately reverse the preferred action?

## Test protocol (prespecified)

- Backbone: frozen `HuggingFaceTB/SmolLM2-360M-Instruct` (all weights frozen; inference-only feature extraction).
- New learned component: shared two-layer MLP with width **4** (`hidden_size→4→1`, around 3.8k trainable parameters), instead of a late-layer residual adapter.
- Dataset: existing V8.2 authored synthetic bilingual fixture, 576 training-pool cases and 288 held-out cases from disjoint project and profile IDs. The **policies do not genuinely differ across profile IDs**, so this study cannot establish personalization.
- Reserve **one entire training profile (2 projects, 96 choices)** as a **development set** to select the epoch, train on the other **5 profiles (10 projects, 480 choices)**. Evaluate only once after selection on **96 locked examples from two held-out projects** (48 English, 48 Hebrew) to match V8.3's evaluated projects.
- Embed only the unordered `(project objective, evidence, policy, candidate action)` tuple; no gold decision, dilemma kind, chosen digit or answer order enters feature extraction.
- Cache frozen contextual features once per unique evidence/candidate combination. The *same* frozen representations are used for both orderings to rule out positional leakage. Do not truncate evidence; abort on long context.
- Train a two-state pairwise preference objective: `softplus(target-margin0)+softplus(target+margin1)`, with target 0.6 and symmetry penalty 0.15 on opposite-state scores. Optimize **14 epochs** with AdamW 0.003; report all epochs, select best by development-set fully correct quadruples, tie-break by development accuracy.
- **Matched frozen zero-shot comparator**: project-context and candidate-specific frozen LM `Yes` minus `No` next-token score (verified distinct one-token IDs), difference of the same frozen candidate contexts, with *no learned parameters or fit to gold*. Also show historical V8.3 raw 48/96 vs 50/96, explicitly noting the changed prompt format; the historical comparison is **not prompt matched**.
- Preserve authoritative external scope/permission enforcement but report it separately. Neither guard nor structurally enforced option-order consistency is evidence of neural semantic learning.

## Metrics and go/no-go gate

| Metric | Exploratory threshold before expansion |
|---|---|
| Raw learned head accuracy | ≥ 55/96 (57.3%) and exceeds prompt-matched frozen score |
| Correct **fact-flip pairs** | ≥ 20/48 |
| Correct complete 2-state × 2-order groups | ≥ 8/24 |
| English and Hebrew performance | Each ≥ 26/48 (no language collapses to an always-selected option) |
| Raw high-risk approval/scope cases | No worse than prompt-matched frozen condition |
| Control leakage | No optimizer gradients through backbone or held-out labels |

A correct option-order reversal is *guaranteed by construction* for any non-tied score; report it to audit implementation, but **never treat it alone as a learned capability**. If raw score fails the factual-reversal gate, do not scale hidden width or run a 3-rank grid. Study evidence representations, frozen model calibration and independent authored evaluation instead.

## Scientific caveats

All examples, labels, options and "approved" project evidence are entirely fictional and authored by the research workflow, not independently labeled. Project and user IDs differ but the scenario grammar and generic policy transfer across splits. This tests feasibility on a synthetic *fixed set of two provided candidate actions*, not autonomous planning, external document retrieval, real individual preferences, or verified rationale generation. Scores are descriptive; repeated paired examples are not independent users. An 82% score on this dataset, for example, would still not imply reliable real-world decision quality.

**Isolation:** all V8.4 work resides on `experiment/compass-v8-4`. No changes to original V8/V8.1/V8.2/V8.3 checkpoints, native 5M SELF, production code or deployment.
