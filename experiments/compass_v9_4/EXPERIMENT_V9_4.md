# COMPASS V9.4 — from learned project intention to a novel actual decision

**Preregistered protocol, prior to viewing V9.4 cold model predictions.**
Branch: `experiment/compass-v9-4`. This deliberately addresses V9.3's major scientific limitation: the old concept learner only classified binary principles; it never consumed the *new technical problem* to decide what to do.

## Research question and true architecture scope

Does a **small numerical memory of past user/project correction histories** help a trainable **semantic decision network** choose one of TWO genuinely new project actions under new natural-language task descriptions, without re-injecting user/project memories as text into that decision network? Can it distinguish a goal-level project purpose from its historical implementation, respect explicit binding contracts and preserve the separation of global personal vs project states?

The long-term SELF/COMPASS design is a **persistently attached per-user neural module plus per-project neural modules inside a frozen general LLM**. **V9.4 does NOT implement that final coupling.** It first tests whether the memory can guide a previously unseen *task-conditioned choice* at all, while separately evaluating a real frozen 360M LLM with textual history as a baseline. The actual decision bridge is a smaller trainable neural scorer, not the 360M's own hidden layers.

## Implementation

1. Re-train the existing V9.3 user/project history memory from scratch on **TRAIN-only synthetic English correction histories**, using frozen pretrained 384D `sentence-transformers/all-MiniLM-L6-v2` embeddings. Training its 57,796 shared weights for **600** optimizer steps and choosing a checkpoint on TRAIN-template development histories only. **Freeze this learned memory before downstream bridge training**. One global user state and two isolated project states, 48D each, updated by GRUCell. A project key identifies the active workspace; this is environment routing, not a learned label.
2. New semantic decision bridge receives **only** (i) 48D user state; (ii) 48D active project state; (iii) frozen 384D pretrained embeddings of the NEW situation and both concrete action texts. No original history text, no oracle goal labels, no semantic task-family flags, no gold answer, no hidden preference mode or explicit scope field is passed to the bridge. It uses a **shared candidate scorer**, preventing artificial option-A position bias. Train with signed action margins for both states and paired contrastive loss: in neutral cases a corrected owner/project principle should flip the decision; binding constraint cases must remain stable.
3. Eight training task families: 4 project-purpose-vs-public-contract and 4 user verify-before-commit-vs-reversible-prototype. Four **different** held-out task families for each scope, four newly named project IDs never in training, and 16 newly authored owner/project history paraphrases. In *each paired cold case*, only ONE history stream is replaced by a conflicting history; all situation/choice text and unrelated streams remain identical.
4. Explicitly authorized binding requirement text overrides the learned prior. It appears in the new dilemma statement; gold is **never** inserted into neural or LLM input. By construction, neutral cases require a decision change and binding cases must NOT change. Both original and policy-changed versions are evaluated. Option-order swaps are present, plus an exact same-input reversed-option symmetry probe.
5. Train bridge on **384 synthetic pairs** (80% train, 20% within-TRAIN-template development), **45 epochs**, batch16, seed11. Best checkpoint selected ONLY on internal dev; test data never used to select hyperparameters or stop training. Cold: **128 synthetic policy intervention pairs**, 64 neutral and 64 binding balanced across user/project context, and two action orders. Task families, project names and far history texts differ from bridge training, but all labels, scenarios and language constructed by experimenter.
6. **Frozen actual 360M text-context comparator**: Run `HuggingFaceTB/SmolLM2-360M-Instruct` without fine-tuning, full relevant personal/project history supplied in a prompt alongside the new task/actions. It scores each candidate's *Yes vs No* next-token logits. This is an actual frozen 360M LLM inference baseline, not simply a keyword classification. CPU/time efficiency: score a **balanced prespecified subset of 32 of the 128 paired cold examples**; report comparison against the neural bridge on the **EXACT SAME 32** pairs, not the neural's full-128 number. This is a reasonably strong full-text comparison but **not** an optimized RAG system or latest frontier model.
7. Other controls: no-history constant-action, latest-message keyword + synthetic scope/binding rule, frozen semantic prototype memory + scope/binding rule. Note the deterministic rules DO have access to oracle synthetic task labels and are not fair pure-language baselines. They are upper-bound sanity checks for benchmark learnability.

## Preregistered effectiveness gates (all conjunctive)

- ≥**60%** BOTH-correct of **64 neutral** policy-switch pairs on entirely new test families.
- ≥**75%** BOTH-correct of **64 binding** cases.
- ≥**95%** option-order semantic invariance under actual reversed option vectors.
- Neural correct BOTH states on strictly more of the **same 32 subset** than the frozen 360M with **full relevant text history**.
- Neural wins over the synthetic keyword + privileged fixture-scope rule on the full 128 examples (conservative upper bound; failure means no claim of benefit over straightforward rules).
- Frozen semantic encoder and history memory weights remain **unchanged during bridge training**; user/project independent states and zero per-user gradient steps. No native SELF 5M or production code changed.

**GO** means only a preliminary synthetic concept-to-decision demonstration; further human independent cases and real-world evidence are mandatory before stronger conclusions. **NO-GO** means do not inject weak concepts into the big model or expand ranks just because some logits can be perturbed.

## Limitations to surface in every result

- All histories, tasks and gold labels **synthetic and researcher-authored**. These binary axes do not represent open-domain learned human intent. No independent judge or multiple-seed CI.
- Active project is selected from trusted workspace identity; user/project scopes are coded into different semantic contexts, and binding scenario semantics are explicitly written. The bridge DOES NOT receive oracle scope or binding flags, unlike some sanity baselines.
- Text vs neural baseline receives different representations (full history for 360M vs frozen semantic compressions for neural). Compare both *outcome and token/compute cost*, and do not overclaim causal superiority.
- No final in-Transformer native neural coupling of COMPASS to SmolLM2; the big model is called only in text-context baseline.
- All weight and state updates take place in a synthetic short 3-message stream. No cross-session persistent user model storage, online real-user consent/provenance, revocation, source authenticity, or long-horizon concept retention.

Artifacts planned: `compass-v94-concept-to-decision.json` includes results on every cold pair, development curve, test subset baseline and all gates; `compass-v94-concept-to-decision.safetensors` contains only decision bridge parameters; separate short artifact for preflight. No ZIP-only delivery to user.
