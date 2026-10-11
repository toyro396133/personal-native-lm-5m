# COMPASS / SELF V9.0 — real in-Transformer owner-conditioned neural residual

**Status: experiment registered; outcomes pending.** This is the first controlled attempt in the COMPASS series to condition a frozen shared language model **inside a Transformer block, without adding the user preference to the prompt**. It extends rather than reinvents the pre-existing `experiments/personal_smollm2/run.py` single-user adapter hook, with a shared pre-trained coupling and hot-swappable per-user preference state.

## User's actual target (important)

The desired system is: one continually running shared language model; it is **functionally coupled** to a distinct user-specific personalized neural model; preferences that persist over requests influence the large language model's actual neural processing rather than being periodically repeated as textual context. Backbone parameters stay fixed; a personal module and approved individual state can change cheaply without a complete new LM training run. Here "aware" is **functional architecture**, not consciousness.

In the **first V9.0 pilot** only, per-user state is an explicit one-dimensional approved owner-priority variable (rapid = -1 / thorough = +1). A **single shared low-rank adapter is trained once** using training-project examples, and different users can swap that 1D state **without further optimization**. This validates an inner-network personal-state interface but DOES NOT yet constitute a separately trained independent neural model for each individual. That broader requirement remains the north star.

## Exact architecture

- Frozen backbone: `HuggingFaceTB/SmolLM2-360M-Instruct`, CPU float32.
- Capture final output of the **last Transformer decoder block**, before frozen final RMSNorm and vocabulary head. In the live model, a real forward hook adds a personal residual to **the final token's hidden state**. The final-token attention has already run; this is **one block-output intervention**, not early-layer attention/cross-attention.
- Shared trainable adapter of rank **16**: `down:960→16`, `owner:1→16`, `up:16→960`:
  `h_personal = h + up(tanh(down(layernorm(h)) + owner(state)))`.
  **30,736 learned shared adapter parameters** (16 × (960+1+960)); frozen LM params not updated. Initial `up` zeroed to make disabled/initial adapter identical to backbone.
- Per-user dynamic state is `rapid=-1.0` or `thorough=+1.0`. Hot-switching state requires **0 per-user optimizer steps** and **0 personal text tokens**. We intentionally test causal effect with the SAME goal/options/evidence text and two owner states.
- For feasibility and compute, training caches each input's owner-blind final decoder-block hidden vector under `torch.no_grad()`; the real frozen norm + vocab head are reused with gradient flowing exclusively through the adapter. At test, an actual attached hook is exercised and its Yes/No logit margin is **numerically compared** with cached equivalent, for BOTH owner states, and the backbone weight tensor is byte-exactly unchanged.
- Both options are presented in the decision prompt but no personal preference is present unless running the **text baseline**. The personal state is passed directly to a tensor operation that changes an intermediate activation, not injected as text. This proves a path for model-level adaptation, not yet high-quality general preference reasoning.

## Synthetic training/evaluation isolation

- English only: no multilingual or Hebrew training.
- Training: previously prepared 40 owner-policy-only training pairs from distinct fictional train projects, plus supervised examples with a binding reversal from the V8.5 training split; development selection from another training-only policy profile (8 policy pairs plus binding examples). Same language model weights/seed and no cold labels in loss or epoch selection.
- **New V9.0 cold decision families**, not previously run in V8.9: `change_review`, `data_backfill`, `performance_investigation`, `handover_process`, × two previously used fictional test projects × neutral/binding evidential status × both option orders = **32 counterfactual preference pairs**: 16 neutral that MUST flip under changed personal priority; 16 binding that MUST NOT flip under owner priority changes. Only choices+evidence are in neural input; gold/meta/source flags are NEVER included.
- This is **researcher-authored synthetic gold**, not independently human reviewed; the test projects have been used before but the four task families and exact sentences are new. Such results cannot establish field readiness or broad person-level generalization.

## Locked ablation controls and gates

1. **Frozen no-personal-control**: same owner-blind candidate prompts / pretrained logits, no owner state.
2. **Shared inner-block neural personal adapter**: exact same prompts and frozen base, only personal state `-1` vs `+1` changes.
3. **Frozen text-injected baseline**: the approved owner policy is added to the prompt; no trained personal adapter, quantifying what old textual conditioning achieves.
4. **Transparent simple lexical rule** baseline: classify comprehensive vs focused action phrases and binding keywords, then pick according to explicitly approved user priority. This is tailored to synthetic data and is not a neural model, but establishes a strong cheap task-specific comparator.

**Functional gates** (must be checked before interpreting decision performance):
- Verified real live hook after decoder block and cached adapter produce matching Yes/No margins within **0.001** for both state values.
- At least one audited example shows changed internal/logit margin when only owner state changes.
- Frozen backbone weights unchanged; shared adapter training changes its own weights, not the LM.
- Personalized prompt has **no policy string**; activated state and 2 user profiles are kept separate.

**Research effectiveness gates:**
- ≥12/16 neutral test pairs BOTH correct (policy change should switch decision).
- ≥12/16 binding test pairs BOTH correct (priority should not override binding evidence).
- Must improve cold both-correct pairs over the explicit text baseline (strictly more, not just matching).
- Preserve positional invariance (same semantic action under option reversal) and require manual review before strong claims. A passing single-seed scripted holdout is only a smoke test, not proof of robust human personal reasoning.

**If gates fail, NO-GO** to adding more ranks or merging into native SELF 5M. Investigate adapter location, explicit latent preference representation, stronger independent data, fewer lexical artifacts, and training diversity before sweeping 7,680/30,720/122,880/491,520 personalized weights.

## What is explicitly NOT implemented yet

- No learned, independent per-user weights or rich multidimensional personal memory; only an externally approved 1D preference plus a single shared learned conditioned adapter.
- No persistent service that loads/saves personal modules across sessions; only proof of hot-swapping state at inference.
- No repeated multi-layer attention modifications; block output is altered after attention but before the model's final norm and LM head, which can influence next-token generation.
- No online owner-preference extraction/update from real untrusted messages, no authentication/authorization, no real provenance verification or encrypted per-user data store.
- No claim that neurons genuinely understand the user; "awareness" means the large forward pass is functionally conditioned on the user state.
- No direct evaluation of the separate native 5M SELF language model or production system.

## Planned output and provenance

Research branch `experiment/compass-v9-0`; GitHub Action workflow `compass-v90-personal-neural.yml`. Capture `compass-v90-personal-neural.json` with all cold per-pair choices, train/dev history, exact module parameter count, backbone unchanged assertion and actual hooked-model-vs-cached activation numerical test. Save `compass-v90-personal-neural.safetensors` only for experiment results; never promote based on successful technical execution alone.
