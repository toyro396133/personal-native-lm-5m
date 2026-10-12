# COMPASS as a repeatedly trained, detachable personal neural layer

**Architecture hypothesis refined from the owner's analogy, 2026-10-12.**
This document is a design interpretation to test, **not** evidence that the mechanism already works or a requirement to physically extract a Transformer block.

## The core idea

Treat COMPASS **as if one adaptation layer were separated from a large language model** and made user-owned: its small neural weights/state keep learning from long-term interactions and projects, while a capable large model supplies general reasoning when they are joined for inference.

**COMPASS remains independent from the parallel SELF research effort.** A large model is a replaceable *consumer*, not a teacher permanently entangled with COMPASS. We should prefer **genuinely neural adaptation** over treating all COMPASS knowledge as a static text bundle. However, source audit/event memory is still necessary so that mistaken updates can be reviewed, reversed and corrected.

```text
 human / project events and review
           |
 +---------v-------------------------------------+
 | COMPASS, detachable & user-owned             |
 |  raw evidence/provenance & validity ledger    |
 |  long-lived personal neural state P_user      |
 |  per-project neural state P_project           |
 |  continual update learner + replay/gating     |
 |  portable canonical latent Z                  |
 +--------------------+--------------------------+
                      |
             small model-specific bridge B_model
                      |
           +----------v----------+
           | frozen capable LLM  |
           | reason / interpret  |
           +---------------------+
```

## Two adapters, not one universal hidden layer

- **Permanent COMPASS core:** personal/project state and evidence; preferably a learned canonical latent that doesn't depend on the target LLM's vocabulary size, hidden width or parameter names. Updates repeatedly from trustworthy user feedback. This is the durable user-owned asset.
- **Small per-model bridge:** translates the canonical state to a mechanism a particular backbone understands. Possible bridges:
  1. `context_bundle` in normal text/JSON for black-box hosted LLMs; easiest portability and baseline, but weak evidence of direct neural effect.
  2. `soft_prefix` or a short trainable prefix embedding via a model-specific projector (requires model APIs or model internals, not generally available in black-box services).
  3. `activation_adapter` / residual gate / LoRA-style interface attached to a frozen open-weight model, giving stronger neuron-level conditioning but more architecture-specific. A raw Transformer block or LoRA matrix cannot generally be transplanted unmodified across unrelated backbones.

Therefore **portable COMPASS weights** and **universal plug-and-play native model insertion** are distinct aims. Portability can be achieved by retaining the central state and training a lightweight reader/adapter for each family, rather than retraining the user's entire history. This is a research hypothesis, not a guarantee.

## What exactly keeps learning?

For interaction `e_t`, separate user state `P_u` and relevant project state `P_{u,p}`, with update:
`(P_u',P_{u,p}') = Update(P_u,P_{u,p}, e_t, feedback_t, provenance_t)`
subject to user/project scope, protected prior preferences, confidence/validity, replay and update approval where needed.

For query `q` to capable model `M`:
`z = Select(P_u,P_{u,p}, q)`
`c_M = B_M(z, relevant_evidence, uncertainty)`
`response = M(q | c_M)` while `M` is frozen. **Continuous user-specific training happens only in COMPASS** and permitted bridge layers; evidence remains auditable outside learned weights.

Cross-project generalization must be selective; don't turn a project-specific rule into a universal user rule automatically.

## What V12 should actually test

1. **Fixed capable open-weight backbone** (if possible) and same evaluation tasks, base frozen. First compare no personalization, equal-budget RAG, static context and **continually trained COMPASS with text bundle**. This is the accessible full system baseline.
2. Compare the *same trained COMPASS state* with a **small model-specific neural bridge**, e.g. learned soft prefix or low-rank residual conditioning, to test the user's real idea against plain text injection.
3. Distinguish **frozen COMPASS state** from **updated COMPASS state** on later unseen user/project tasks. Test separate personal/project state, false user-authority claims, stale decisions, unknowns and long-horizon forgetting.
4. When first model works, freeze user COMPASS state, fit only a small **reader for a second different model**, and ask whether useful personalization transfers without relearning all user data. Compare to retraining on the second model and ordinary retrieval.
5. Keep **source-grounded traces** for every training change; do not claim that a model-specific adapter migrates automatically across all models.

**Scientific win conditions:** materially better later work quality than fixed-backbone RAG/static contexts, update benefit over frozen COMPASS, no unacceptable forgetting/false certainty, and demonstrable transfer of user state to an alternate backbone via low-cost adapter training. Engineering unit tests or a handful of teacher-created toy examples are not proof.

## Relation to literature (ideas, not proof of COMPASS)

- Soft prompt tuning with frozen language models (Philippy et al., 2024): https://aclanthology.org/2024.moomin-1.2/
- SPoT soft prompt transfer (Vu et al., 2022): https://aclanthology.org/2022.acl-long.346/
- PUMA personalized prompt migration across incompatible models (Zhao et al., AAAI 2026): https://ojs.aaai.org/index.php/AAAI/article/view/40805
- Cross-Model Memory Transfer via Target-Side Reader Adaptation (2026): https://arxiv.org/abs/2608.17050

These show relevant design patterns, but **do not establish that the proposed COMPASS can learn deep user intent**, and not every method is trained across multiple projects or users.

## Non-goals

- Do not require COMPASS's tiny model to reason at LLM scale or replace the main model.
- Do not merge COMPASS into SELF or require SELF checkpoints.
- Do not make GPT-6 a fixed teacher or mandatory runtime.
- Do not replace the learned state with a permanently static prompt/database dump.
- Do not assert that extracting *a literal layer* from one LLM can be plugged directly into all other architectures.
