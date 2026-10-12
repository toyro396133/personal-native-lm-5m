# SELF SmolLM2 personal adapter: 50-fact longitudinal audit

## Reproducible run (2026-10-10)
- [GitHub Actions run 38087143196](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38087143196): completed SUCCESS.
- Base: HuggingFaceTB/SmolLM2-360M-Instruct, frozen.
- Personal interface: last-layer residual adapter, 7,680 learned weights (rank=4), 25 optimizer steps per stage, 10 stages, 50 synthetic profile facts (five new each stage).
- Scoring: forced-choice pairwise token negative log likelihood, NOT free-form generation.
- Final: **base 27/50 (54%), personal 33/50 (66%), naive full-history context 30/50 (60%)**.
- Longitudinal personal stage accuracies: 5/5, 10/10, 14/15, 17/20, 20/25, 24/30, 26/35, 29/40, 30/45, 33/50.
- Final per-cohort (chronological batches of 10): 8/10, 5/10, 10/10, 5/10, 5/10.
- Final base/personal paired outcomes: both correct 20, base-only 7, personal-only 13, both wrong 10 (exact paired two-sided p≈0.263).
- Final naive-retrieval/personal: both correct 21, retrieval-only 9, personal-only 12, both wrong 8 (exact paired p≈0.664).

## Critical validity limitations discovered during review
1. **Uneven training exposure in V1**: examples sampled by `(step*7+stage)%len(seen)` for 25 steps. In stage 7, NONE of the five newly introduced facts was sampled (gcd(7,35)=7). In stage 10, only ONE of five new facts was sampled; stage 9 sampled three of five. Thus V1 cannot establish saturation of 7,680 personal parameters.
2. **Truncated retrieval baseline**: naive all-notes text is left truncated by a 512-token context budget; stale early notes may drop out entirely. Therefore the 66% > 60% comparison is not evidence that parameter learning beats an appropriate retrieval system.
3. **Small and patterned synthetic data**: fixed formats, five paraphrases per category, single seed, repeated answer candidates; not a proof of user-level generalization or fact recall. Evaluation chooses between two candidates; it does not test producing the answer without candidates.
4. **Statistical uncertainty**: paired differences on 50 facts are not convincingly above chance at conventional significance. A negative-transfer and broader held-out-language test was not included.
5. **Training loss is per-different-example**: first-versus-last per-stage loss cannot be interpreted as a meaningful learning curve.

## Corrective follow-up
Added `benchmark_50_v2.py` retaining V1 for reproducibility. The new schedule guarantees at least two exposures to every new fact per stage, with the remaining steps rehearsing old facts. It compares against a deliberately **oracle** single-fact memory retrieval control (upper-bound, NOT a practical retriever), which avoids full-history truncation; each stage archives exposure counts and personal adapter checkpoints. It is a separate new run: results must not be described as successful until the run finishes.

Next evaluation priority: a real retriever without ground-truth matching, stronger held-out paraphrases, multiple seeds, free-form answer extraction, and explicit catastrophic-forgetting curves by fact age.
