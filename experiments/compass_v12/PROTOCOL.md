# COMPASS V12 — Detachable per-user neural layer attached to a frozen LLM

**Preregistered engineering protocol, 2026-10-12.** COMPASS and SELF are independent parallel projects. This experiment follows the owner's "as though one layer was taken out of the big model and retrained for the user" design. No OpenAI GPT-6 teacher, no SELF checkpoints, no private project text in public CI.

## Main question

Can a **portable persistent trainable state per user and per project** repeatedly update from verified owner feedback, while a **large LLM remains exactly frozen** and its **live internal decoder activation** is conditioned through a small separate V9.0-style adapter?

This is a *system integration/causality* test, not proof of long-horizon project understanding.

### Components and boundaries

1. **Frozen capable backbone**. Initial reference: `HuggingFaceTB/SmolLM2-360M-Instruct` used by V8/V9. This is NOT SELF. During personalization no backbone weights change.
2. **Model-family-specific COMPASS bridge**. V9.0-style `LayerNorm -> down(960→rank) -> tanh(+personal_state+project_state) -> up(rank→960) -> residual`. It can initialize from V9.0 `down.weight`/`up.weight` (optional user-local safetensors) or a research initialization. It is then frozen for per-user updates. The V8.5 ranking-head weights **cannot** be copied blindly: their matrices score options, rather than inject decoder activations. Record this noncompatibility.
3. **COMPASS-owned states**, dimension `rank`: `U_user` and `P_user,project`; both exist entirely separately from backbone and bridge weights. A project-scoped correction updates only `P`, a user-global correction updates only `U` and must be explicitly scoped. No project→personal promotion without permission.
4. **Feedback**: a verified direct user choice between two originally presented candidate actions; user approval to use it for training. Never produce supervision from an assistant's inference about intentions. `NO_AUTO_POLICY_APPROVAL`: training latent does not change authoritative principles.
5. **Detachable transfer**: export/import project+user vectors in JSON with architecture-neutral canonical rank/version, plus provenance pointers, outside model weights. This is *not yet* proven to work across different models; each will need an explicit bridge.

### Training protocol

- Freeze the bridge and backbone. Optimize **only a newly allocated per-user or per-project latent** with a fixed learning rate (initially 0.05), 8 gradient steps per feedback, L2 retention to old latent (0.001); persist after each reviewed feedback. The old shared tensors never enter optimizer.
- Use actual live hook on final decoder block; compare Yes/No next-token logit margins for each fully specified option. The preference loss is `softplus(-(chosen_score - other_score))`, where `score = YesLogit - NoLogit` on candidate-specific prompts. No training on heldout outcomes.
- CPU engineering test uses **small fictional transformer-shaped backbone** with deterministic full autograd and no external downloads. Separately provide an *optional* reproducible SmolLM2-360M smoke script that downloads model only when explicitly invoked. No claim of 360M end-to-end success from tiny tests.
- Pre/post checks: frozen-backbone byte-exact state; frozen bridge byte-exact; only intended user/project latent changes; pre/post same prompt logit shift; before/after paired choice success; state persistence; swapping users and projects; no accidental gradient/authoritative approval.
- Correct counterfactual: shuffled personal/project states and unrelated-project updates must not modify output for target user+project. An untouched user keeps identical inference.
- Report development/evaluation split explicitly. Toy test outcomes are *system mechanics*, not real user behavior. Real study needs source-audited owner feedback across time and matched RAG/static/context-only baselines.

### Acceptance gate

1. Synthetic end-to-end tests prove actual in-model hook and per-user latent gradients; backbone + bridge unchanged, no cross-project leakage, portable state round-trip, no autoapproval.
2. Optional 360M smoke must verify live hook changes actual model logits after **per-user update** with a genuine user-scoped supervision example (can start with synthetic test user; label as synthetic). Do not call a change in a logit a learned preference correctness result.
3. No promotion unless multiple genuinely new user/project decisions scored at a heldout temporal cutoff show additional quality over equal-context RAG AND frozen COMPASS, with a reliable scope/provenance gate.
4. If pretrained V9.0 bridge weight artifact is unavailable or incompatible, declare that explicitly. Random bridge proves only feasibility and may have arbitrarily poor language effects.

**Possible verdicts:** `V12_LAYER_HOOK_AND_PERSONAL_TRAINING_ENGINEERING_PASS` versus `V12_360M_REAL_BACKBONE_SMOKE_UNVERIFIED` and `NO_SEMANTIC_PERSONALIZATION_GAIN_PROVEN`. They must never be conflated.
