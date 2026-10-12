# COMPASS V12 — combined detachable user neural layer, frozen 360M and online updates

**Completed engineering study, 2026-10-12.** COMPASS remains independent of SELF. V12 builds on the general *shape* of V9.0's internal residual hook, the earlier V8.5 personal-weight experiments and V11.1 direct-user correction review. The V8.5 ranking weights are **not** falsely combined with V9.0's decoder residual: these are different parameterizations.

## Verified implementation

1. `detachable_layer.py`: user-owned `rank=16` neural state per user **and** separately per user+project; shared small model-specific V9.0-shaped frozen decoder adapter; live hook in a capable external frozen language model; repeated SGD updates **only to the selected personal/project state**, provenance, duplicate-event prevention and JSON checkpoint persistence outside model weights.
2. `smoke_360m_optional.py`: actual `HuggingFaceTB/SmolLM2-360M-Instruct` loaded with frozen parameters, real gradients through live final decoder block, optional **genuine preserved V9.0 pretrained shared `down.weight`/`up.weight`** from historical run 38116261352. V9.0 `owner.weight` encodes a scalar fictional owner state, not transferable continuously trained user weights, and is intentionally excluded.
3. `from_v11_feedback.py`: source-verified/direct-user or user-reviewed corrections converted to **training-only** positive-vs-negative pairs, preserving the original evidence and review hash. No invented assistant labels or automated product-policy approval.
4. `train_queue.py` and `--training-pairs` support: actual V11-style private feedback queue can be applied **in order** to the detachable state using live model gradients, with all-or-nothing update commitment and `--existing-state` for later sessions. No inference of fresh, unseen user intent is measured by these retrospective training pairs.

## Runs observed

| Run | Real model? | Result | Limit |
|---|---|---|---|
| [38190008600](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190008600) | PyTorch tiny fictional transformer | **18 engineering tests PASS** | Not language understanding |
| [38190351282](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190351282) | PyTorch tiny fictional transformer | **27 engineering tests PASS** | Added V11 source review validation |
| [38190589362](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190589362) | PyTorch tiny fictional transformer | **31 engineering tests PASS** | Full V11 review queue → V12 training, duplicate guard, repeated updates and atomic rollback |
| [38190117236](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190117236) | **Real frozen SmolLM2-360M** | **PASS: live hook, gradient and external state updated** | Random bridge, 6 steps, lr 0.2 |
| [38190341185](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190341185) | **Real frozen SmolLM2-360M** + preserved **V9.0 trained bridge** | **PASS: live hook and substantially nonzero update** | Still fake user, same final decision |
| [38190627871](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38190627871) | Real frozen SmolLM2-360M random bridge, **matched** 10 steps/lr 0.4 | Check result in original artifact before comparing quantitatively | Necessary matched control |

### First real 360M probe (not matched on hyperparameters)

**Random shared bridge, 6 steps, lr 0.2**:
- Source/feedback: fictional user and question; no actual user policy.
- Before candidate Yes−No logits: `[0.2847433090, 0.9049158096]`; after `[0.2847423553, 0.9049129486]`.
- Detachable project state delta norm: **0.0002227261**.
- Choice index remained **1**. Useful semantic improvement **not demonstrated**.

**Preserved historically TRAINED V9.0 shared bridge, 10 steps, lr 0.4**:
- Confirmed V9.0 `.safetensors` retrieved from **artifact 11693970107** of [V9.0 original real run](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38116261352).
- Same real base model frozen during personalization. Only new detachable project latent optimized.
- Before candidate Yes−No logits: `[0.4234046936, 0.7393836975]`; after `[0.3880338669, 0.7255182266]`.
- Pairwise chosen-minus-rejected margin **0.3159790 → 0.3374844**, an increase **+0.0215054**.
- Training loss **0.5475864 → 0.5395563**.
- Detachable project state delta norm: **0.1891083121**.
- Same chosen index **1** before/after. No novel user-decision test.

**Do not compare those two delta norms as a controlled advantage of pretraining:** learning rates/steps differ. The separate matched random-bridge control above uses the **same 10 steps, lr 0.4**; report its actual output before any controlled comparative statement. Also different initial bridge weights produce different starting margins, so the appropriate paired control metric is the within-run **margin improvement from training**, not raw logits alone.

### Real engineering claims justified

- **Yes**, V12 attaches an adaptive neural personal/project state at an *actual frozen large Transformer decoder output*. It is not only text memory, RAG or a post-hoc classifier.
- **Yes**, gradients from the real model's Yes/No logits modify a separately stored user's **project state**, and the model can consume that updated state later. Neither model backbone nor frozen shared bridge was in the optimizer; representative backbone tensors matched exactly in the 360M smoke.
- **Yes**, a *historically trained* V9.0 shared bridge can be reused with a NEW trainable user/project latent under its same hidden dimension and rank.
- **Yes**, V11.1-style user-reviewed raw correction input can now be translated to a source-linked training queue and continuously update the latent. That code path has passed synthetic PyTorch tests, but has **not** yet been run on independently adjudicated real user correction examples.

### Research claims NOT yet justified

- This is **not** proof of long-term project understanding or user-intent transfer from old sessions.
- No genuine future held-out human decisions were evaluated.
- No base model was taught the meaning of a user's project; the large model already handles language, COMPASS tries to adapt it.
- No cross-backbone user-state migration was tried; only one 960-dim model-specific bridge was tested.
- GitHub tests used **fake** source assertions and synthetic user labels; there is no cryptographic user-identity authentication or real source-authority integration.
- Comparison to equal-budget RAG, frozen COMPASS and plain-text context is still to be completed using actual future work tasks. No full benchmark or claim of personalization quality.

**Scientific verdict:** `V12_REAL_360M_DETACHABLE_NEURAL_LAYER_ENGINEERING_PASS`, **`NO_REAL_USER_SEMANTIC_PERSONALIZATION_PROVEN`**. A valid next test requires independently verified real user feedback plus truly later decisions, with the same frozen model and matched controls.
