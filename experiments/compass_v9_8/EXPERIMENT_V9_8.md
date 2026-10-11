# COMPASS V9.8 — personal intent transfer across projects

**Registered before any V9.8 cold predictions.** V9.7 passed 9/9 **mock provenance-controller** checks but its neural preference scorer managed only **34/64 (53.1%)** paired BOTH-correct personal contexts, versus **63/64** project intent. That overall 225/256 must never be taken as learned personal intelligence.

## Question and design

Does a separately trainable **personal intent readout**, connected to numeric global-user GRU state but not to project memory, improve decisions in NEW problem families where unrelated project principles conflict with global personal preferences? Can a **learned text-only scope router** select personal vs project reasoning without being handed the fixture's hidden `scope` label at inference?

- Retain V9.3 frozen pretrained MiniLM sentence encoder, V9.3 trained 48-D personal/project GRU states, and matched baseline V9.4 shared decision bridge. V9.3 memory trained using its original synthetic TRAIN corpus only. V9.4 joint scorer trained on the same V9.4 TRAIN examples, with baseline dev selection as published.
- New model: `PersonalReadout` receives **only global user state (48D)**, unseen question (384D) and two candidate action embeddings (2x384D). Trained from scratch using neutral (`binding=False`) personal TRAIN examples, counterfactual paired history flips, and consistency against **changes in inactive/active project histories**. Project state **cannot** enter this head. A single shared option scorer makes score permutation-equivariant.
- New `ScopeRouter` sees question embedding only (384D) and predicts whether personal or project intent applies. It is trained using scope labels from TRAIN questions but at inference receives no metadata, gold, binding status or hidden case scope. A second, separately trained `ProjectReadout` receives **only active project state**, the question and options. This prevents leakage from general user history into project-only decisions. The frozen V9.4 joint bridge remains the matched baseline; the two new scope-separated heads never see the other state. The model must not receive `case.scope`/gold at prediction time. Router and personal head are selected by **held-out TRAIN-family development only**, not new cold outcomes.
- TRAIN families remain the four original V9.4 TRAIN user families and four TRAIN project families. For router and both scope-specific readouts, keep `documentation` (personal) and `reporting` (project) entire families for development and remove from gradient training. The V9.4 matched baseline uses its original published V9.4 training schedule; report that its selected split is per-case, unlike new model's family holdout, instead of claiming identical optimization data.
- **No text of earlier histories at decision time**; learned GRU state, new question and option embeddings only. No non-text operation labels, simulated policy `binding` or authority oracle enters candidate scores.

## New researcher-authored heldouts

- 128 fresh NEUTRAL (`binding=False`) paired dilemmas, 64 personal and 64 project across **16 genuinely new family names**, disjoint from all V9.4, V9.5, V9.6 and V9.7 task families. Novel English wordings in candidate actions and question; 16 new personal/project-history statements instead of V9.4 heldout histories. Four previously unseen workspace IDs (from existing cold workspace namespace) and A/B option swaps balanced.
- In a pair, replace only the authoritative personal or active-project history; problem and choices identical; unrelated histories unchanged. Both states must be correct; single-decision accuracy is insufficient. Each scope contains both opposing preference principles, both workspaces and both option orders.
- Counterfactual distraction: for all 64 personal pairs, REPLACE histories of **both projects** with opposing/conflicting project principles; gold stays the same. For all 64 project pairs replace global user history with opposing personal preference sentences; active project's gold stays the same. Testing includes decision invariance vs original for all 256 original/changed decisions per perturbation suite.
- Additional *workspace-switch* personal stress: change selected workspace key from A to B, leaving user history and task/options fixed, with project histories contradictory; personal choice should not change. Test 64 pairs and both states.
- Retrospective V9.7 personal-neutral 64 pairs are **diagnostic only**, never count as blind or guide hyperparameter choices.

## Controls

1. V9.4 same-input original joint neural scorer.
2. New personal/project isolated readouts + learned router.
3. Same new readouts with **oracle scope routing** as an explicitly unfair upper-bound ablation (scope labels not provided to actual proposal).
4. Non-neural equal-input-class reference: question-text nearest TRAIN-family prototype router, plus mean frozen numeric-memory-state prototypes, with both action texts. This should not use hidden cold scope or action-order labels. Report even if it beats neural. This is a semantic comparator, not a complete retrieval-augmented LM.
5. No-memory fixed option control (option A), to detect low-information shortcuts.

## Registered success criteria

For **each of three seeds (11, 19, 23)** independently:
- Personal new-family paired BOTH-correct **>=48/64 (75%)** AND improves by >=8 pairs over matched frozen V9.4 scorer on same 64 personal cases.
- Project new-family paired BOTH-correct **>=48/64 (75%)** AND at most 4 fewer correct than V9.4 on same 64 project cases.
- Router correctly identifies scope **>=120/128 (93.75%)** without oracle tags at inference.
- For personal workspace-switch, 0/128 individual decision changes; for personal unrelated-project-history perturbation, 0/128 changes; for project unrelated-user-history perturbation, 0/128 changes.
- Personal opposed-project distraction retains **>=45/64 BOTH-correct**, not merely invariant incorrect answers.
- Shared-option scorer order reversal >=95% on both sides of 64 paired dilemmas (128 comparisons).
- Report every control including oracle-routing; **no acceptance claim based only on pooled seeds or cherry-picked winner**.

Three-seed all-gates PASS would only establish **provisional synthetic transfer**, not independent human-authorized enduring personal preferences and not integration inside a 360M Transformer. Synthetic creator labels and corpus pattern reuse remain major limitations. If any fails, explicit `NO_GO_TO_LM_ADAPTER_INTEGRATION`.

## Interpretation safeguards

The personal readout is trained from synthetic history-derived labels, NOT from real users' ongoing project conversations. Project-vision reasoning remains a separate concern; metadata routing is **learned only from text family cues**. Real implementations must handle ambiguous tasks, overlapping scopes and mandatory obligations through provenance-aware control plane. No project-specific obligation or private project history is allowed to become the user's global preference merely because it appeared in another project.

Do NOT call V9.8 a trained native in-LLM SELF adapter. Do NOT infer success beyond these binary synthetic axes and held-out authored examples.
