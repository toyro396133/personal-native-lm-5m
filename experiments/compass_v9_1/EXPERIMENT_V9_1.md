# COMPASS / SELF V9.1 — earlier-layer personal-neural conditioning, causal paired learning

**Preregistered protocol — no V9.1 result has been inspected.** English-only research. Branch `experiment/compass-v9-1`. The user wants a frozen general language model functionally personalized **inside its processing**, avoiding repeated user-context text and avoiding retraining the general model for each user.

## New vs V9.0

- **Real earlier-layer injection:** attach the trainable state-conditioned residual after decoder layer **third from the end** in frozen `HuggingFaceTB/SmolLM2-360M-Instruct`. Two *subsequent Transformer decoder blocks* consume the altered representation and run normal attention/MLPs before final norm and output head. This differs from V9.0's last-block-output injection, which could only change the last normalization/logits.
- **Four-slot state interface:** input `state: R^4`, with pilot owner states `rapid=[1,0,0,0]`, `thorough=[0,1,0,0]`. Slots 3–4 are **reserved and untrained**, not independent personal preferences. Only ONE bit of personal information is tested; do not claim a trained rich user representation yet. The `state` is not present in user text.
- **One-time shared gate training, frozen backbone:** rank16 `down(960→16) + state(4→16) + up(16→960)` = **30,784 learned shared adapter weights**. All backbone gradients disabled. A user's state can be switched with no per-user optimizer step.
- **Explicit paired causal objective:** for the same prompt+facts+actions, change ONLY the personal state. Neutral cases should reverse the option ranking. Mandatory binding evidence cases must preserve the action. Loss combines supervised signed option margins, an opposite-policy pair reversal margin for neutral examples and a small difference penalty for binding cases.
- **Important compute trade-off:** unlike V9.0's cheap cached-last-block training, actual early-block gradients must propagate through two frozen-but-differentiable downstream Transformer layers. GitHub Actions CPU runner, seed11, **8 epochs**, **36 paired cases per epoch** (24 policy-neutral, 12 binding) selected solely from synthetic TRAIN families; best checkpoint selected by development-pair both-correct. No need to train the 360M backbone.

## Synthetic data isolation

Reuses V9.0 training/development subsets only; creates binding-policy twins in those splits without changing evidence. No gold labels, task IDs, user IDs, precomputed status or owner policy appear in owner-blind neural input tokens.

**Entirely NEW synthetic V9.1 cold test:** 4 new fictional project IDs with 4 unseen task families (`technical_release`, `catalog_consistency`, `support_handoff`, `quality_inspection`) × 2 evidence statuses × 2 order reversals = **64 same-project, same-evidence policy interventions**, **32 neutral MUST flip** and **32 binding MUST NOT flip**. All materials and labels written by the experimenter rather than independent reviewers; test result is not field reliability proof.

Project IDs and exact family text are new relative to V8.5–V9.0, rather than reused legacy TEST profile identities. Same scripted writing style remains a possible leakage/confound.

## Pre-registered controls and gates

For each NEW 64-pair holdout, measure both-correct before+after owner-policy change and order invariance, separately by neutral/binding. Arm comparisons:

1. **Frozen owner-blind baseline**, identical input for rapid and thorough, no neural gate.
2. **V9.1 early internal neural state conditioning**, identical language tokens, only personal state vector changes.
3. **Frozen model with full English personal preference in context**, no neural gate.
4. **Researcher-authored symbolic lexical priority/constraint rule**. It has direct access to approved owner policy mode and keywords; strong but task-specific, not a language model.

**Mandatory mechanical assertions:**
- Real installed hook alters downstream (-2) layer activation with *identical token IDs* under different personal states.
- The real full-model output Yes/No margin differs under the two personal states.
- The exact same owner-blind prompt text contains NO owner policy string.
- Frozen backbone remains unchanged (representative exact tensor checks; not all-weight cryptographic audit).
- Training labels never inspected on cold/test data until evaluation and never enter LM prompt.

**Effectiveness targets** (all required before claiming a useful result):
- **≥24/32 neutral paired examples both correct**, thus choosing differently for changed owner policy.
- **≥24/32 binding pairs both correct**, holding firm against owner preference.
- **strictly better than text-injected baseline** on total both-correct cold pairs.
- **strictly better than lexical baseline** on total both-correct cold pairs.
- **≥56/64 semantic action order invariance** (two user states across mirrored choice orders, 4 projects and 4 task families).
- Pre-registration is not a statistical power calculation. Seed11 only; replication, independent blinded human review and risk controls would be mandatory for real-world claims.

**If the neural module changes hidden states/logits but does not flip relevant choices correctly: engineering pass, scientific NO-GO, not "personalization solved".** Do not scale to other ranks, launch production, or merge into native SELF 5M without strong results.

## Honest scope and future direction

This is a **trained shared personal-state connector**, not a separately learned user-specific neural network yet. There is no per-user neural checkpoint storage, source authentication, real preference inference, recurrent per-token feedback, long-term user memory or multi-attribute independent preference learning. Future separate personalized models and continuously connected multi-layer/attention intervention require additional engineering and controlled testing, and no current results justify promising their success.

Artifacts planned: full `compass-v91-early-personal.json` (per-case predictions, development curve, causal hidden/logit probes), `compass-v91-early-personal.safetensors` (shared gate weights); independent preflight logs.
