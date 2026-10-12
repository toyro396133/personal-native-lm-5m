# SELF V5 — canonical versioned personal memory and selective adaptation

Status: **implementation committed; evaluate actual GitHub Actions metrics before conclusions**.
Backbone: frozen HuggingFaceTB/SmolLM2-360M-Instruct; rank-4 per-user 7,680-parameter residual adapter.
Source files: `personal_v5_data.py`, `benchmark_v5.py`, `aggregate_v5.py`.

## Why V5 is necessary

In V4, the hybrid improved binary-choice scores but fell to 54.2% on a tiny generic-QA suite and used an outdated preference in all three seeds. V5 explicitly addresses **supersession**, **selective adapter invocation**, and **direct exact-memory baselines** rather than merely more optimizer steps.

## Deliberate scope: controlled extraction, not real general-purpose conversation parsing

Three **distinct synthetic profiles** each have 12 slots, learned in three stages. The last stage updates two previous slots (focus duration and dashboard accent). Input user messages use constrained grammar:

- `For focus duration, my choice is 35 minutes.`
- `Correction for focus duration: use 25 minutes instead of 35 minutes.`

A deterministic parser extracts slot/value/revision and saves the full provenance record, including superseded values. Any correction with a mismatched prior value is rejected. **It does not use the gold-answer table to populate memory**, but it assumes these authored templates; it is not a proof of generic chat fact extraction.

At answer time, retrieval ranks **current canonical slot labels** against question tokens and returns only active values; historical superseded values are not supplied as equally valid answers. The audit log retains prior versions for traceability.

## Six conditions

1. `frozen`: pretrained frozen model, no memory.
2. `adapter_only`: personal adapter for every question; no retrieved notes.
3. `retrieval`: frozen model with top-two current canonical notes.
4. `always_hybrid`: personal adapter always active + current notes.
5. `gated_hybrid`: same adapter + notes, but route generic/unmatched questions through the frozen backbone.
6. `direct_memory`: retrieve top matching current canonical value and return it *directly* for personal slot questions; general questions route through the frozen backbone. This is an intentionally strong comparator for schema-based factual queries.

All three profiles are evaluated separately with paired but different seeds (11,19,37). Each stage trains the adapter using up-to-date facts only; it replays all active facts and gives additional exposure to new and corrected facts. Older values are not trained after supersession.

## Evaluation and caveats

- At each stage, score binary candidate questions against current user facts; track latest-version top-1 retrieval, fact update count and training exposure.
- Final open-ended greedy generation: 12 personal questions, two updated-fact questions, 10 generic questions per profile. The direct-memory condition bypasses the LM for factual personal answers, demonstrating what storing an exact record can achieve.
- Normalization maps simple English number words to numerals and rejects mentions of obsolete values, explicit negation and repeated 3-word fragments. Raw outputs are preserved for manual review, and this scorer is not a full semantic judge.
- The gate is **hand-authored**, based on personal pronouns and lexical overlap with known slot names; generic-score preservation is an expected routing property, not preservation of the adapter itself.
- Profiles and seeds change together, so these are not independent noise seeds on each profile.
- This experiment does not test automatic unconstrained memory extraction, deletion/privacy lifecycle, hostile inputs, semantic retrieval, or large-scale generic regression.

## Results policy

Do not report outcomes from V5 until matrix run plus artifact aggregate completes successfully.
V4 baseline and notes: `FINDINGS_V4.md`. V5 is not a drop-in deployment candidate or an integration into the native 5M model.
