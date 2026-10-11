# SELF V4 — mixed chat-history continual learning (pre-registration)

## Purpose

Test whether personal adapters learn sustained user-specific information from an interaction history beyond the previous templated 50-fact setting. Compare frozen SmolLM2 against two **compute-matched** personal-update schedules, query-only retrieval and the replay-adapter-plus-retrieval hybrid.

## Data

- English-only, **fictional** 24 user facts, introduced four at a time over six synthetic conversation stages.
- Each stage includes unrelated user messages and assistant acknowledgements. **Only user utterances** are searched by retrieval.
- Two corrections to earlier preferences appear at later stages; evaluation uses the latest value, not the old one.
- Training prompts and test questions are deliberately phrased differently, but remain hand-authored and small in number.
- There is **no real user's chat history**, telemetry, external personal data, or identity information.

## Conditions

1. `base`: frozen `HuggingFaceTB/SmolLM2-360M-Instruct`; no user context.
2. `recent`: per-user 7,680-weight adapter, trained only on last eight visible facts plus corrections.
3. `personal`: same architecture and **same 32 optimizer steps per stage**, trained with all known facts replayed at least once and each of four new facts twice.
4. `retrieval`: frozen backbone + top-3 notes ranked **only** from question vs user-utterance content with lexical IDF/bigrams.
5. `hybrid`: replay adapter + exactly the same top-3 retrieved notes.

All base LM weights are frozen; the personal adapters are saved separately. The two training strategies are update-count matched, but not compute-matched to retrieval or hybrid inference. No true answer or test fact ID is an input to the ranking algorithm. IDs are retained for post-hoc precision evaluation.

## Evaluation

- On each of six stages, compute pairwise candidate log-likelihood ranking across all known facts for five conditions; report old vs recent accuracy and retrieval hit@1 / hit@3.
- At final stage, perform **greedy open-answer generation** (24 personal questions per condition) without candidate answers in the prompt. Record every output verbatim and score conservatively via normalized literal answer matching.
- Evaluate eight generic non-personal QA items under each condition to detect obvious damage to general functionality; this is a tiny sanity check, not a no-regression guarantee.
- Repeat with independent random seeds **11, 19, 37**, keeping benchmark facts fixed; publish per-seed metrics and aggregate mean/range.
- Archive full metrics and personal adapter checkpoints; do not present unrun or failed experiments as empirical results.

## What would count as progress?

Prefer a stable open-answer advantage across the three seeds, with limited losses on generic QA, and explicit retrieval hit rates. A +1/+2 fact improvement alone is too weak to establish superiority.

## Known caveats

The language model is small and the synthetic data are hand-authored, so this cannot establish reliable personal memory in production. Literal answer matching may miss correct paraphrases or count some misleading phrasing. The two preference updates are limited; there is no user-controlled deletion or privacy lifecycle yet. A next iteration would add more varied free-form histories and larger independent test populations.

## Run

`.github/workflows/self-v4-mixed-chat.yml` on `main` runs three independent matrix jobs and a consolidated summary job. Each job checks out the experiment branch and never updates other experimental checkpoints.
