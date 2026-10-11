# COMPASS V9.10 — bilingual, conditional, changing intent stress test

**Registered before any V9.10 cold predictions** (October 11, 2026). Previous V9.9 attained 384/384 pairs on researcher-authored one-bit English tasks only **when a simulated signed UI event correctly named the owner**. That is a structured routing test, NOT proof of human/project intent reasoning.

## Aim
Measure rather than assume whether the unchanged V9.9 numeric-memory + small intent-readout architecture handles:
1. Hebrew rather than English.
2. Longer **five-event** histories, where a broad preference coexists with a separate principle and reason.
3. **Conditional binding overrides** that take precedence over preferences.
4. An active/withdrawn/expired obligation in a new task.
5. Genuine changes to the primary preference across counterfactual histories.
6. Personal global principles and project-specific vision in separate streams, even if the non-authoritative stream disagrees.

## Locked input and prediction rule
- Train V9.3 48D numeric user/project GRU memory and V9.8 separate user/project readouts on the **original V9.4 English TRAIN only**, with unchanged family-heldout dev selection. Three deterministic seeds: 11,19,23. No V9.10 test text, status or labels in training. Frozen encoder: `sentence-transformers/all-MiniLM-L6-v2` (English-centric and **not a multilingual model**).
- Main **128 newly authored paired cold cases**: 64 English / 64 Hebrew, 64 personal / 64 project, 16 new test family names across language/scope. Eight pairs per family. A pair shares the question, options, source event and non-target histories; only its target personal/project preference history is reversed.
- Tasks have four regimes balanced in every scope x language: `open` (no binding requirement), `active` (current documented mandate), `revoked` (old mandate canceled), `expired` (old mandate out of force). The semantic obligation is stated in natural-language question text and in a trusted **mock policy manifest for structured-policy ablation only**. **V9.9 source-routed intent scorer receives NO regime, hidden gold, task family, or structured policy operation IDs.** It does receive the same source UI event declaring owner as in V9.9.
- Reuse V9.9 mock HMAC source resolver only to select which intent memory to read. A valid source event identifies owner but NEVER supplies the correct action/override. Trust keys are publicly embedded demo keys, NOT real authentication.
- Both user and project histories now have **five** textual events, including reasons/conditions for secondary principle; only the three primary preference observations change in the counterfactual, while both secondary-principle reason statements remain fixed. The temporal mandate is part of the new task, not an immutable global rule. The **canonical gold is computed by a separate deterministic specification**, not passed to the learned scorer.

## Controls
- `v99_source_routed_readouts`: same family of tiny learned semantic head + current source routing (the target architecture, not changed/tuned).
- `v99_question_routed_readouts`: identical heads with question-only learned owner routing, to expose owner confusion (no source evidence used by this arm).
- `source_routed_train_numeric_prototype`: same source and numeric states, TRAIN-only semantic prototypes.
- `structured_task_contract_oracle`: privileged upper bound that uses externally provided mock active-status and required operation ID; **not a fair NLP model**, and cannot be interpreted as model reasoning.
- `constant_action_zero`: no-history sanity check.
- A second **32-case source-integrity suite** (8 each: missing, bad signature, conflicting, wrong workspace) requires `ASK` (code 2), tested only with V9.9 source resolver.

## Prespecified evaluations
Metrics always count paired BOTH-correct (both original and changed history decisions correct), not optimistic single-decision accuracy. For each seed report personal/project x Hebrew/English x status and BOTH-flip agreement, override rule compliance, ASK, option order equivalence, source isolation. The candidate is not allowed to use task regime metadata at inference.
- Per seed: **overall >=88/128 paired**, English >=48/64, Hebrew >=40/64; and active conditional >=24/32. User and project each >=40/64.
- Scope routing based on mock valid origin 128/128, invalid origin ASK 32/32, option order >=95%, unrelated project/user-history isolation >=95% sampled decisions.
- Candidate must beat numeric prototype baseline on at least 2 of 3 independent seeds by >=5 pairs total (same test cases and language split).
- ALL three seeds must satisfy absolute gates. Otherwise **NO_GO_TO_LM_ADAPTER_INTEGRATION**. No revision of thresholds or cold fixtures based on outcomes.

## Interpretation controls
Because V9.9's frozen encoder is English-centric and all training is English, **poor Hebrew generalization is expected**. A good English score with weak Hebrew is a useful NEGATIVE result. The structured oracle can implement conditional mandates without language understanding because it is allowed to read typed ground-truth-like status metadata; it must never be marketed as neural reasoning. Synthetic examples written by the same research author are not a human-independent blind benchmark. Do not claim real-life intent, valid safety promises, real source signing, or 360M in-transformer SELF integration.

## Next after V9.10
If weak: multilingual frozen encoder + Hebrew training examples vs matched frozen English baseline, long-horizon multi-principle provenance/justification memory, and independently authored Hebrew/English scenarios. If unexpectedly strong: still require independent human judging, explicit applicability conflict and verified origin claims before any native 360M adapter.
