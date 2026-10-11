# COMPASS V11 — independent online learning: first implementation and gate

**2026-10-12. COMPASS track only. SELF is a separate parallel model-development project. GPT-6 is not a model component, tutor/label source, answerer, or automatic evaluator. No API.**

## Implemented

- [`online_compass.py`](online_compass.py) is a fully standalone Python 3.11+ CPU learner with **learned sparse weight state per project**, explicit separate personal-only state, and never auto-promotes project observations into personal scope.
- **Prequential evaluation**: predict first from prior episodes; only then consume the original user answer as optional feedback and update weights. No future answers available to the current prediction.
- **Tightly conservative direct feedback eligibility**: options must be explicitly recorded from an original source, with a valid pinned-Git doc/line pointer, and the raw user's reply must equal an option ID after quote/punctuation normalization. Qualified, free-form or incomplete replies are **stored as evidence but never automatically converted to a gold label**. Researcher/generated labels are rejected by validation. Source pointer syntax is checked; correctness/authorship still requires external audit.
- **Transparent small learned mechanism:** hashed 3-gram lexical option and question×option features, online fixed-learning-rate pairwise logistic update. No semantic latent language understanding is asserted. `MARGIN` gate abstains when insufficient support; at least two prior explicit labelled experiences within the same scope are required.
- **Fair initial automated controls** computed from exactly the same past history: untrained first-option, most recent eligible option, and lexical-nearest previous chosen option. No retrieval model, hosted model, pretrained encoder or SELF coupling.
- **Private state checkpoint** can be saved/loaded locally, including learned weights and cited evidence. No changes to user-approved policy; feedback weights are never authorization.

## Engineering verification

[`GitHub Actions run 38186061247`](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38186061247): **20 distinct fictional-data tests passing** for Python 3.11, 3.12 and 3.13, **60 successful test executions**. Earlier run [38186034740](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38186034740) failed at the CI Python-version setup due to malformed GitHub workflow interpolation, **before the learner tests ran**; setup line corrected and rerun passed. Do not count the first failure as learning failure or omit it from reproducibility history.

## First authentic-data eligibility audit (READ ONLY, not model benchmark)

Directly inspected the *private* Chacham Lev telephone planning question/answer log at Git SHA `fe9863278a04dedf27a0c4fcb8cca56b91ec4544`, `docs/campaigns/telephone-fundraising-qa-log-2026-10-11.md`.

- Found **52** T00–T51 decision/interview sections.
- An elementary surface-only text check found just **8** first-line raw answers that look like a single stand-alone choice letter, in sections **T19, T21, T23, T29, T31, T36, T47, T48**. This is **a candidate eligibility count**, NOT 8 authenticated label-ready cases. Every original question and exact option alternatives must be checked before V11 could train from them; some original questions might not include the corresponding options verbatim.
- **44/52** did not match that narrow direct-letter pattern. Many are meaningful *free-form* statements, corrections, hedged preferences, requests for historical research, or explicitly unresolved decisions. Discarding these would defeat the user's central aim: learn project intent, WHY and decisions in context.
- The actual number of correctly source-verified and directly labelled real episodes is **not yet established**. **No V11 real-data prequential accuracy can be reported now.**
- If fewer than 20 curated, directly eligible chronological episodes are available across multiple topics, protocol verdict stays `INSUFFICIENT_REAL_DECISION_FEEDBACK`.

## Research truth and next action

The online learner is **engineering-valid**, but does not yet establish semantic lifelong user/project learning. Direct option choice is only a very narrow diagnostic. The **real next bottleneck** is a *source-grounded, independently reviewed event-to-learning-signal compiler* for free-form user answers and corrections: preserve exact question, reply, temporal context, source, rationale and uncertainty. No external LLM as teacher; use user-provided ground truth, deterministic unambiguous rules or explicit human review. Do not manufacture teacher labels.

Once reliable episodes exist, compare V11 online state to equal-data retrieval/recency/static alternatives on genuinely unseen later decisions with counterfactual project swap, irrelevant-context, correction and rationale-validity stress. Report any failure; only a stronger independently verified semantic result could claim meaningful understanding of project WHY.

**Scientific verdict: `V11_ONLINE_LEARNER_ENGINEERING_PASS`, `COMPASS_SEMANTIC_INTENT_LEARNING_NOT_YET_PROVEN`.** This is COMPASS work, independent from SELF.
