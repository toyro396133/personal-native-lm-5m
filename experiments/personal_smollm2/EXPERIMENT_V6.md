# SELF V6 — style conditioning and explicit-memory falsification

**Status:** Experimental code; evaluate actual GitHub Actions results before making empirical conclusions.

## Motivation from completed V5

V5 demonstrated that reliable canonical factual memory could beat a trained 7,680-parameter residual adapter for personal factual recall, and that always-on adaptation caused avoidable errors in generic questions. V6 asks a *different* question: can a tiny personal module add useful persistent **response-style conditioning** beyond simply prompting the frozen model with a known preference?

## Data and conditions

- Backbone: frozen `HuggingFaceTB/SmolLM2-360M-Instruct`.
- Three distinct fictional preferences: *exactly two short bullet points*, *three numbered steps*, and *one concise sentence*.
- For each preference, 12 synthetic supervised demonstrations (distinct topic prompts), 48 adaptation steps, learning rate 0.0003 and gradient clip 0.5, trainable 7,680 parameters. 12 **held-out topic prompts** are used for style-format evaluation.
- Six independent training jobs: 3 style profiles x seeds 11 and 19.
- Eight basic generic questions per job are an ancillary ability-regression signal.

The five test modes are:
1. `base`: frozen LM, no preference.
2. `memory_prompt`: frozen LM, **explicit style preference extracted from the user's fictional authored message**.
3. `adapter_only`: trained user adapter, no style prompt.
4. `adapter_plus_prompt`: trained adapter + the same explicit style prompt.
5. `gated_adapter`: adapter active for recognized practical advice requests, **disabled** for generic factual questions.

**Falsification rule:** Improvement over the `base` is insufficient. The adapter must show meaningful advantage over `memory_prompt` on unseen topic prompts with no material general knowledge regression before claiming worthwhile adaptation. An explicit preference prompt may prove superior, and such negative results must be preserved.

## Simulated memory lifecycle controls

`memory_v6.py` implements a transparent constrained-language extractor for user-style declarations, corrections, uncertainty/hypotheticals, third-party statements/quotes, explicit *active memory deletion*, temporary overrides for two response uses, and expiry. It logs provenance and routes practical advice separately from general trivia. It handles these **specific templates**, NOT unrestricted real user-language understanding. The persistent audit log is intentionally not deleted; a production erasure and privacy lifecycle would require deletion of source records, caches and possible retraining/replacement of personal weights.

## Measurement limitations

- Format-scoring counts bullets, numbered steps or short single-sentence structure; it **does not grade truth, content quality or user usefulness**. The outputs are saved for manual inspection.
- Generic factual keyword scores on only eight questions are inadequate to prove non-regression. They can misgrade answers.
- Repeating seeds on the same 12 held-out prompts is replication of initialization/training noise, not independent user-population evidence.
- The adapter is trained on artificial demonstration styles, not organically learned across thousands of natural user interactions.
- No hypothetical relationship to human identity or broader native SELF self-model is established. V6 is an isolated research lane.

## Files and run

- `memory_v6.py` — memory extraction, revision and temporary expiry tests.
- `benchmark_v6.py` — rank-4 adapter training, held-out generation and five baselines.
- `aggregate_v6.py` — six-run summary with counts and limitations.
- `.github/workflows/self-v6-style.yml` — GitHub Actions six-run matrix and aggregate.

Do not claim success or merge the experimental adapter into a production architecture until reviewing the completed run and output samples.
