# COMPASS V8.9 — English-only active advice and bounded student→teacher feedback

**Experiment registered before data from this run.** The user explicitly requested an English-only proof of concept, deferring Hebrew/multilingual and advanced backbone selection until the mechanism actually works. The user also asked for larger personal models: we will compare **7,680 / 30,720 / 122,880 / 491,520 learned personal weights**, but **only after** a useful reliable teacher/feedback architecture is established and controlled. Do NOT confuse larger parameter count with better source validation.

## Fundamental correction after V8.8

V8.8 teacher generated 144 recommendations, all for canonical Action A; 72 Hebrew generated reason strings were identical, and its injected advice degraded the personal scorer's heldout policy flips. That is a defect of the old **A/B next-token prompt interface and weak role format**, not evidence that a frozen language model cannot act as partner.

V8.9 excludes Hebrew completely and tests TWO improvements *before distillation*: independent per-action **Yes/No log-odds scores** instead of canonical A/B next-token completion, and an explicit **single bounded student→teacher feedback round**. The 360M teacher produces one short English contrastive explanation before scoring each candidate; when possible the new step conditions on it. We separately measure unconditioned initial scoring, explanation-conditioned scoring, and feedback-conditioned scoring. Invalid generated explanations are marked as abstention for the reasoned arm, not silently treated as valid decisions. Human-readable explanation is still not proof of faithful reasoning.

## Roles

**Model A: English-only frozen teacher, `HuggingFaceTB/SmolLM2-360M-Instruct`:** gets the goal, current authorized fictional owner preference, full evidence and both options. Generates an explanation (max 48 new tokens), and separately computes for each action the logits of "Yes" vs "No" as the best authorized action (not A vs B). This is still heuristic next-token scoring and must be audited for bias and calibration.

**Model B: frozen V8.6 personal rank4 checkpoint, exactly 7,680 weights:** independently scores each option using the same frozen 360M contextual embeddings as before; it contributes its *untrusted* preferred action and numeric scores to ONE follow-up prompt that asks the large teacher to reassess. No small/large gradient updates in V8.9 phase 1. The teacher may disagree; external authenticated facts/permissions and user approval always outrank either model.

**Evaluation:** gold is read ONLY by `score_v89.py` after `teacher_v89.py` has already generated and saved all logits, reasons and suggestions. No task family, gold answer, preference mode enum, or synthetic authorization metadata enters teacher/student text prompts. Each test input is fiction with an independently specified *current* authorized preference. Real preference changes still require a user-approved state update in a future application.

## New synthetic English-only research fixtures

Four completely new task *families* (incidents, data reconciliation, sprint planning, design validation) for the two previously used fictional heldout profiles; each covers:

- **Neutral:** both options authorized, equally adequate, current owner preference MUST control outcome and changing owner preference MUST flip the correct action.
- **Binding:** explicit documented constraint requires full thorough option, independent of owner's preference; changing policy MUST NOT flip outcome.
- **Choice-order swaps:** original option order and its reverse should lead to exactly the same *semantic* recommendation.
- 4 families × 2 projects × 2 evidence statuses × 2 swapped choice orders = **32 counterfactual pairs**, comprising **64 original+changed-policy decision contexts**, and both candidate scores.
- All gold and texts authored in this workflow; not external, human-graded or truly out-of-distribution to the synthetic author. The project profiles themselves were previously examined, although the family text is new. English only by design.

## Locked research gates before comparing model sizes

- Direct large teacher, neutral **≥12/16** pairs correct in both versions (correct flip).
- Direct large teacher, binding **≥12/16** both correct (correct NON-flip).
- Initial teacher semantic action order stability **≥28/32** when reversing order.
- One student→large feedback round must improve neutral pair correctness **by ≥2** over direct large teacher, **not lose** any binding pair correctness and preserve semantic order stability **≥28/32**.
- At least **48/64** English generated analyses pass simple nonempty/non-echo checks; qualitative review must follow even if text passes.
- No inference about utility for real people if this gate passes: need more seeds, independent human-reviewed new family holdouts and stronger non-neural baselines first.

Any failure => **NO-GO to blind rank sweep** until explicit teacher failure is diagnosed. A passing first gate only opens the *next* controlled capacity experiment: old 7,680 baseline, existing 30,720 and 122,880 checkpoints, a newly trained **491,520 rank256** model with matched supervised training and budget. A random 491,520 untrained head is NOT a valid capacity comparison. The next experiment must compare scores conditional on identical teacher messages and match training data/optimizer/seed.

## How V8.9 differs from V8.8

- NO Hebrew messages, labels or model generation.
- NO A/B token choice with canonical-sorted labels.
- Teacher evaluations based on per-candidate Yes/No margin in both choice orders; record margins and explain positional biases.
- Extra bounded second teacher pass **uses live small-head scores and preferred action**. This establishes a real causal feedback input path, even though neither model is updated through training.
- Track both preference flips and nonflips under binding facts, not just generic accuracy.
- Report original unchanged personal-head decisions as controls on exactly the same new inputs, not already exposed V8.6/V8.7 test scores.
- The external trusted-source registry contract from V8.7 is an independent fail-closed software property; it cannot be counted as neural reasoning.

**Native SELF 5M, real users, live project systems and production remain untouched.**
