# COMPASS / SELF V9.0 — completed results: functional in-block neural conditioning; failed owner-priority generalization

**Technical workflow:** [GitHub Actions run 38116261352](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38116261352), first attempt **SUCCESS**; both preflight and real-frozen-model training jobs completed, 2026-10-11 05:58:42 UTC. **Scientific verdict:** the adapter is a proven working neural intervention on the frozen model, but **FAIL / NO-GO on preference adaptation**. Do not promote it as a successful personal model.

## What V9.0 actually implements

Frozen `HuggingFaceTB/SmolLM2-360M-Instruct`, **361,821,120** backbone weights, all unchanged during training. A **rank-16 conditional residual with 30,736 learned SHARED weights** (`down:960→16`, `owner:1→16`, `up:16→960`) is attached by a genuine forward hook on the output of the LAST Transformer decoder block before final RMS normalization and language modeling head. It acts on the LAST token only. This is a true mid-forward activation perturbation, but **not** early attention steering or a multi-layer recurrent personal module.

Each fictional user is represented in this feasibility probe by exactly **one externally approved float owner state**, rapid = −1 and thorough = +1, hot-swappable between calls with **zero per-user optimizer steps**. It is **not yet a separately learned neural model per user**, nor a memory system. The shared adapter was trained once using fictional English projects. The adapted model receives the **same owner-blind text** for both owner states; no preference text is injected in the neural arm.

## Verified technical pass

- Preflight checked synthetic fixture isolation, exact parameter count, identity at zero initialization, and gradient-based learnability of a stand-alone rank16 gate.
- Live 360M Transformer hook after the final decoder block matched the separately cached frozen-layer activation path within **0.001 logit-margin tolerance** for both owner states. This is evidence that the effective neural transformation happened in the real model's inference graph, not merely in a downstream scorer.
- An identical prompt with no owner policy in it produced these two different Yes/No logit margins on a selected candidate: rapid **0.39679**; thorough **0.27843**, absolute difference **0.1183586**. Original backbone unmodified margin **−0.07120**.
- The exact same prompt changed the internal probability/logit signal under owner-state intervention, but **the full pairwise action choice did NOT change** in the tested state-swap example: both rapid and thorough picked option 0. The observed logit shift is NOT evidence of correct policy-induced decisions.
- Backbone retained exactly identical weights for the checked frozen projection tensor; `model.requires_grad_(False)` kept all backbone parameters frozen throughout. `backbone_unchanged` asserted true on that checked tensor; do not misrepresent this as a cryptographic checksum of every single weight.

## New, researcher-authored synthetic cold test results

New family texts `change_review`, `data_backfill`, `performance_investigation`, `handover_process` across **2 previously used fictional test projects**. **32 paired preference interventions**: 16 neutral where preference change MUST flip the action; 16 binding where an external mandatory constraint means preference MUST NOT flip the action. Each tested under both candidate presentation orders. Labels/scenarios synthetic author-created, no independent adjudication. Test-set outcomes were not optimization targets, but are not independently authored tasks.

| Method | Neutral both-correct /16 | Neutral actual flips /16 | Binding both-correct /16 | Overall both-correct /32 | Action-order semantic consistency |
|---|---:|---:|---:|---:|---:|
| Frozen base, no owner input | 0 | 0 | 16 | 16 | 32/32 |
| **Owner-state inner-block trained residual** | **0** | **0** | **16** | **16** | **28/32** |
| Frozen base + textual owner preference | 3 | 3 | 16 | 19 | 30/32 |
| Researcher-authored transparent lexical rule | 12 | 12 | 16 | 28 | 26/32 |

The rule is specifically tailored to these synthetic action words/constraints and not an independently generalizable AI model; nevertheless it illustrates that trivial structure can beat the neural baseline. The inner-block learned residual did not match even a weak textual owner preference injection.

Training/development: **60 epochs**, fixed seed **11**, rank **16**. Chosen epoch **6**, peak internal development **14/24** labeled decisions. At epochs 1, 10, 25 and 60, reported development correctness was **12/24**, and training batch loss about **1.02**. The model did not learn robust preference-sensitive choices.

### Registered gates

- **Functional real hooked 360M forward path:** PASS.
- **Nonzero state-conditioned logits with identical prompt:** PASS.
- **Frozen backbone not updated:** PASS (by design, spot-check exact tensor).
- **Cold neutral paired success ≥12/16:** **FAIL (0/16)**.
- **Cold binding paired success ≥12/16:** PASS (16/16).
- **Neural result strictly better than text-injected cold baseline:** **FAIL (16/32 < 19/32)**.
- Additional independent analysis finds some loss of order invariance, 28/32 vs frozen 32/32.

**Verdict: ENGINEERING YES, RESEARCH EFFECTIVENESS NO-GO.** Do NOT sweep 7,680 / 30,720 / 122,880 / 491,520 parameter modules, and do NOT merge this to native 5M SELF or production, just because the hook runs.

## What caused the gap? (hypotheses, not confirmed causes)

1. A one-dimensional state `rapid/thorough`, with one shared 30.7K adapter, is too narrow to count as the user's envisioned individually learned personal model. The shared encoder sees full evidence/options but the owner conditioning term may be almost uniform across candidate actions, yielding *similar* logit shifts for both actions rather than reversing action preference. On inspected example, two candidate scores shifted under user state without reversing score ordering.
2. Training exclusively at the last block's final-token output, **after** all attention computations, may lack expressive leverage to integrate user state with language-understanding in earlier layers.
3. The loss used scalar Yes/No logit differences from the frozen language model with very small trainable adapter; the frozen baseline can be biased toward a single semantic action. The current training objective doesn't explicitly penalize *failure to flip* both matched states or hold constraints fixed, although supervision contains those pairs.
4. Tiny and synthetic research-authored data: 40 training priority pairs, 8 development pairs, plus a limited number of binding cases, one training seed, four new but research-authored holdout task families.
5. Positional invariance decreased; any further model must report matched orders, frozen-vs-neural baselines and hard evidence constraints.

None of this disproves the bigger idea of user-specific neural conditioning. It demonstrates only that **this one-block, scalar-state, shared neural residual didn't learn the needed preferences**.

## Next research action

- Implement V9.1 as a **controlled factorized experiment**, not a blind parameter increase. Keep the backbone frozen and all synthetic data English-only. Introduce a distinct multi-dimensional **learned or explicitly validated personal state** and inject its *state-dependent interaction* into a **selected earlier block** of the decoder, optionally multiple carefully chosen layers, to allow attention downstream to condition on owner information. Use separate user modules and test isolation.
- Include explicit within-project owner-state pair loss (margin reversal for neutral cases; stability under binding constraints), plus safe approved state transition and no text policy injection in neural arm. Compare to an option-feature-weighting / deterministic rule model and a text-only baseline. Do not train on cold test labels. Preregister new untouched scenarios; obtain external review before real-world claims.
- Distinguish *per-user state update without training* (e.g. memory vector) from the shared interface's **one-time up-front training** and from actual per-user neural weight updates. State accurately which occurs in the implementation.
- Only after >12/16 both-correct on genuinely new neutral pairs AND >12/16 on binding, plus meaningful benefit vs text and symbolic baselines, compare trained module capacities on equal budgets. No native 5M/production merge yet.

## Raw artifacts

Run artifact [`compass-v90-personal-neural-results`](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38116261352/artifacts/11693970107) (ID **11693970107**) includes **`compass-v90-personal-neural.json`** with individual per-case predictions, training/development trace, numerical live-hook equality and state-swap logit probe, and **`compass-v90-personal-neural.safetensors`** containing the trained 30,736-weight SHARED adapter. Preflight artifact ID **11693408845**. Branch `experiment/compass-v9-0`; files under `experiments/compass_v9_0/`.

This is a scientific diagnostic result, not a fully personalized neural language model.
