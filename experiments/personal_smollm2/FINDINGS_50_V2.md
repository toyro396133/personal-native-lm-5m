# SELF SmolLM2 V2 — audited 50-fact longitudinal result

Run: https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38088876024
Status: SUCCESS; 2026-10-10 21:46–21:56 UTC.
Artifact: `self-50-v2-results`, ID `11683103325`; includes `benchmark-50-v2-results.json` and ten personal adapter safetensors checkpoints.
Source: `experiments/personal_smollm2/benchmark_50_v2.py` (experiment branch).

## Controlled experimental design

- Backbone: frozen HuggingFaceTB/SmolLM2-360M-Instruct.
- Per-user personal residual adapter: rank 4, 7,680 trainable parameters.
- Fifty synthetic facts in ten stages of five facts; 25 optimizer steps per stage, 250 total.
- V2 samples each new fact at least twice in the introduction stage and uses spare steps to rehearse older facts.
- Evaluation is **two-answer candidate conditional negative-log-likelihood ranking**, not open-ended chat generation.
- The V2 retrieval condition is **oracle retrieval:** the correct fact is manually selected from the ground truth for each question. It does not measure a real retrieval engine.

## Exact stagewise results

| Stage | Facts evaluated | Frozen correct | Personal correct | Oracle text correct | Personal old correct | Personal new correct |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 2 | 4 | 5 | — | 4/5 |
| 2 | 10 | 5 | 10 | 10 | 5/5 | 5/5 |
| 3 | 15 | 8 | 14 | 15 | 10/10 | 4/5 |
| 4 | 20 | 10 | 15 | 20 | 14/15 | 1/5 |
| 5 | 25 | 13 | 21 | 25 | 16/20 | 5/5 |
| 6 | 30 | 16 | 26 | 30 | 21/25 | 5/5 |
| 7 | 35 | 18 | 25 | 35 | 21/30 | 4/5 |
| 8 | 40 | 20 | 24 | 39 | 21/35 | 3/5 |
| 9 | 45 | 23 | 27 | 44 | 24/40 | 3/5 |
| 10 | 50 | 27 | 32 | 49 | 27/45 | 5/5 |

Final V2: frozen 27/50 (54%); personal 32/50 (64%); oracle-memory 49/50 (98%).

Final V1 (same 50 facts / test items): frozen 27/50 (54%); personal 33/50 (66%); **truncated full-history context** 30/50 (60%). V1 and V2 have **different training schedules and different memory comparisons**; the apparent 60% → 98% retrieval improvement is from changing the retrieval setup, not necessarily improving any retrieval algorithm.

## Paired comparisons at final stage

- Personal vs frozen: personal-only correct 9, frozen-only correct 4, both right 23, both wrong 14. Two-sided exact binomial McNemar p ≈ 0.26685; not statistically persuasive on this tiny synthetic set.
- Oracle memory vs personal: oracle-only correct 18, personal-only correct 1, both right 31, both wrong 0. Two-sided exact binomial McNemar p ≈ 0.0000763 (facts may not be independent; avoid broad statistical generalization).
- V2 vs V1 personal: V2-only 9, V1-only 10, both 23, neither 8. No final-accuracy improvement.

## Key limitations

1. **Incomplete long-term rehearsal** even in V2: stage 10 train schedule includes five new examples twice each (10 steps) and samples only 15 of the 45 previously introduced facts. Thus 30 old facts are not replayed in the last stage. With 25 fixed updates per stage, rehearsal coverage decays as history grows.
2. The V2 model's final old-fact accuracy is 27/45 (60%); introduction accuracy fluctuates (e.g. stage 4 new 1/5), so learning is unstable and should not be called established persistent memory.
3. Oracle fact retrieval includes the answer through perfect selection and is an upper bound, not deployable retrieval.
4. Small patterned synthetic dataset; only one random seed; pairwise answer ranking; no open-ended recall, privacy guarantees, or non-personal-task regression suite.
5. Training loss numbers are measured on different examples; first/last values do not constitute a per-sample learning curve.

## Research direction

The result is a useful demonstration of parameter-mediated personal conditioning, **not** evidence of superiority to retrieval or reliable 50-fact retention. Next controlled test should allocate rehearsal fairly (cover every known fact at least once per stage or use a larger replay buffer), compare a real retriever using only query-based selection, include multiple seeds and free-form answers, and preserve a separate held-out general capability suite. Consider an explicit hybrid: reliable external memory + low-rank personal preference/style controller.

Do not merge the experimental research branch as a validated production personalization solution.
