# COMPASS V9.9 — Source-aware routing without pretending metadata is learned intent

**Preregistered before any V9.9 cold predictions.** V9.8 three-seed result: personal BOTH-correct 47, 47 and 35 of 64; project 64/64 in all three; learned text-only router 128,120,112 of 128. Its misrouting propagated unrelated project histories into personal decisions. Scientific verdict V9.8 NO-GO.

## Research question

Does providing the controller an **explicit, authenticated (mock) request-origin event** reliably prevent personal/project memory cross-talk and permit useful human clarification when an event is missing, inconsistent or forged? What personal concept-to-new-action transfer remains after *removing routing ambiguity*, and does a simple same-input numeric prototype beat the separate learned personal head?

**Scope routing is a control-plane contract, not neural comprehension.** The test harness constructs signed source events matching the task assignment. The fact the signed event matches the researcher-authored answer key is a major **structured-label advantage**, not evidence the model inferred user intent. Never claim a V9.9 increase from adding trusted labels is a learned intelligence improvement.

## Components

1. Retrain V9.3 48-dimensional frozen user/project GRU numeric memory from original synthetic TRAIN only; do not feed history text directly at action time.
2. Retrain V9.8 `IntentReadout` personal head and project head on V9.4 TRAIN neutral cases with one personal and one project TRAIN family held out for development. Train a separate V9.8 question-only router on same split strictly for same-text baseline, not use it for source-aware decisions.
3. Train V9.8 numeric-state/action prototype reference on TRAIN. Project and personal heads only see their own 48D state. Candidate `source_routed_readouts` uses **externally supplied signed source manifest** selecting exactly one readout.
4. `source_routed_prototypes` similarly uses the same manifest to choose scope, and then selects a semantic option by nearest pretrained vector and numeric intent state. This is an equal-provenance baseline, although non-neural prototypes may use distinct inductive biases.
5. Distinct `ASK` action (index 2) is mandatory when origin event is missing, malformed, invalid signature, conflicting or explicitly for a different workspace. A signed source event is not evidence that its described project goal is true; it establishes only which input stream was selected in this simulated workflow. Dynamic consent and external authority verification are out of scope.
6. The source-aware controller NEVER receives `case.scope`, task family, `gold`, `swapped`, `expected_flip` or binding fixture tags at inference. It receives only the request origin manifest and trusted mock signer registry. The question-only router receives question embedding only. Gold is computed separately in the generator.

## Test design

- **Main fresh cold:** 128 newly written English neutral counterfactual pairs, 64 personal, 64 project across 16 new task families (not in V9.4–V9.8). Old and changed versions differ ONLY in the authoritative user's or active project's history. Options/questions and irrelevant history stay unchanged. All 128 carry signed origin events which are simulated real interface declarations like `PERSONAL_PROFILE_DECISION` versus `WORKSPACE_GOAL_DECISION` and are verified by HMAC (reproducible public fake secrets). Eight tasks/family per scope balanced with swaps/workspaces.
- **Uncertainty stress:** 96 disjoint *new* cases from main paired tasks, transformed (24 each): `missing_origin`, `bad_signature`, `conflicting_authoritative_sources`, `wrong_workspace`. Every one has gold `ASK` in both policy-states; neither neural nor prototype controller is permitted to guess. Note the stress set is constructed from existing cases and is NOT an independent sample.
- **Personal non-transfer guard:** replace other projects' entire histories with contradictory patterns and switch workspace key A/B; when valid global origin is signed the personal head must remain invariant. Same for project head under irrelevant user history. Each on 64 pairs (128 individual state comparisons).
- **Zero routing metadata ablation:** original V9.8 learned question-only scope router + separated readouts, scored on identical fresh 128 cases. A fair inference comparison requires noting that source routed version **gets extra source metadata**. It is a control-plane benefit, not an architectural language learning win.
- **Known V9.8 cold regression probe** is retrospective and not used for checkpoint selection; optional, never part of blind success gates.
- Run seeds **11, 19, 23** independently, choose checkpoints and a source-routed personal-head vs prototype option ONLY on held-out original TRAIN-family dev when a selection is necessary. Do not tune on any V9.9 cold results. Report raw readout and raw prototype alongside any composite.

## Prespecified gates (each seed; all conjunctive)

1. Verified correct origin -> 128/128 routed to intended signed source (`personal` or `project`), 0 unwarranted ASK on these clean tasks.
2. 96/96 uncertain/malicious origin paired situations -> **ASK** for both original+changed (0 silent guessed answer).
3. 0/128 individual personal decisions change when other-project histories change, and 0/128 when active project is switched; 0/128 project decisions change when irrelevant user history changes.
4. Personal source-routed learned head **>=48/64 paired BOTH-correct** and at least **+8** compared with the same-head question-only router on exact same personal cases, OR demonstrate the same-head router made fewer than 8 personal mistakes and source-routed personal accuracy >=48/64. The logical disjunction avoids penalizing perfect learned routing, but the 75% absolute threshold is mandatory.
5. Project source-routed head **>=48/64** paired BOTH-correct.
6. At least one of head or equal-provenance prototype is **>=48/64** personal paired correct, and report which is better (no post-hoc choosing without dev-based selection).
7. Option ordering invariance >=95% under actually swapped option vectors.
8. Full detailed per-case scores for both source-routed and question-routed head and equal-origin prototype; no missing seed.

**Even if all gates pass:** only `PROVISIONAL_SYNTHETIC_SOURCE_ROUTING_PASS_NOT_SELF_MODEL`; 360M native LM adapter remains NO-GO without independently authored Hebrew/English real user/project cases, permission audit, long-term multi-principle test and correct source tagging. Reusing the same binary synthetic axis cannot establish open-domain project intent.

## Important shortcomings disclosed in advance

- Metadata describing `PERSONAL_PROFILE_DECISION` or `WORKSPACE_GOAL_DECISION` is a **trusted explicit input by construction**, highly correlated with gold scope. This validates isolation conditional on an *external owner declaration*, not machine determination of owner.
- Synthetic HMAC with source keys checked into public repository does not authenticate real users.
- Task intents are two handcrafted global vs project dimensions; short synthetic history and frozen MiniLM. True user desires and project vision can conflict and require clarifying applicability conditions.
- In real systems origin and applicability can be uncertain or mixed; `ASK` behavior is necessary and support for real sources, permission scopes, history, revocation, trusted action tagging and consent is future work.
