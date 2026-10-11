# SELF V6 — Completed six-run style-personalization audit

**Status:** GitHub Actions [corrected workflow #38093915224](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38093915224) completed **SUCCESS**, all six seed/style jobs and the final aggregate job.
**Prior failed preflight:** [run #38093850221](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38093850221) incorrectly imported `GENERIC` instead of `GENERAL`; excluded from evaluation. Fix: main commit `218e56ac9a1dd913a7ff783f53acfc9748ad20d8`.
**Artifacts:** 6 distinct `self-v6-profile-P-seed-S` JSON plus adapter checkpoints and `self-v6-summary` artifact ID `11685781478`. All six JSON generations independently inspected.

## Registered study design

- Frozen SmolLM2-360M-Instruct; 7,680-parameter rank-4 residual adapter.
- Three fictional user preference styles: exactly two bullets, three numbered steps, one concise sentence; training 12 authored demos, 48 optimizer updates, two seeds (11/19), **12 held-out topic prompts per trial**, 8 generic trivia prompts.
- Five conditions: `base`, `memory_prompt` (frozen + explicit remembered style), `adapter_only`, `adapter_plus_prompt`, `gated_adapter` (active for practical tasks; disabled for generic trivia).
- Automatic format-only rubric: checks surface layout, not answer factual correctness or usefulness.

## Original preregistered-rubric results

| Condition | Format compliance, 72 held-out outputs | General QA, 48 outputs | Repetitive style outputs |
|---|---:|---:|---:|
| Frozen base | 6/72 (8.3%) | 36/48 (75.0%) | 6 |
| Explicit preference prompt only | 12/72 (16.7%) | 36/48 (75.0%) | 0 |
| Adapter only | 8/72 (11.1%) | 41/48 (85.4%) | 4 |
| **Adapter + explicit preference prompt** | **51/72 (70.8%)** | **41/48 (85.4%)** | 0 |
| Selectively gated adapter without style prompt | 8/72 (11.1%) | 36/48 (75.0%) | 4 |

Adapter + explicit preference prompt beats prompt only in **all six trials**, but results are repetitions on **three preference/task families** (not 72 independent users), and scores only check format.

## Exploratory post-hoc alternate format rubric

As an audit of scorer sensitivity, we rescored exactly the **same 72 generated answers** (no extra training) with:
- Two bullets: also accept **two numbered list items** (not merely two hyphens).
- One sentence: accept <=50 words (rather than <=30), while retaining punctuation and single-sentence constraints.
- Three-step rubric unchanged.

| Condition | Official | Post-hoc alternate |
|---|---:|---:|
| Frozen base | 6/72 (8.3%) | 6/72 (8.3%) |
| Explicit preference prompt | 12/72 (16.7%) | **30/72 (41.7%)** |
| Adapter only | 8/72 (11.1%) | 8/72 (11.1%) |
| Adapter + preference prompt | **51/72 (70.8%)** | **51/72 (70.8%)** |
| Gated adapter | 8/72 (11.1%) | 8/72 (11.1%) |

**Interpretation:** the style advantage for adapter-plus-prompt remains positive, but shrinks from **+54.2 to +29.2 percentage points** (difference between official and exploratory measures). These are post-hoc observations, not prospective confirmatory estimates.

### Per-style post-hoc comparison

| User style | Prompt only | Adapter + prompt |
|---|---:|---:|
| Two concise bullet points, across 2 seeds | 16/24 | **22/24** |
| Three numbered steps, across 2 seeds | 4/24 | **15/24** |
| One concise sentence, across 2 seeds | 10/24 | **14/24** |

V6 generation used `max_new_tokens=76` for all methods. **Truncation is a major confound**: out of 72 prompt-only replies, 22 have no terminal punctuation; versus 17 for adapter+prompt. In the three-step profile, prompt-only frequently outputs long explanations that end mid-sentence or before step three. The adapted model usually writes shorter and thus completes the requested layout sooner. This is a possible functional advantage, but the benchmark cannot separate actual preference learning from response-length control and scorer sensitivity.

## Direct generation review

- Example (two-bullet preference, seed 11): frozen+prompt answers with two **numbered** items and is marked *failed* by original rubric; adapter+prompt answers with two short dash bullets and is marked *passed*. Both may be acceptable to a user.
- Example (three-step preference, seed 11): adapted+prompt writes `1. Choose a storage solution. / 2. Create folders. / 3. Organize files.` and passes. Prompt-only produces longer, sometimes unfinished advice; useful material may be present but format fails.
- For the one-sentence preference, prompt-only sometimes writes one reasonable sentence exceeding 30 words; the alternate scorer credits some.
- All six `memory_v6.memory_test` checks passed, including active preference update, active deletion, two-reply temporary override and expiry, ignoring third-party quotations and hypothetical statements. **The original audit records remain**: this is not GDPR-grade or production-grade memory erasure.
- In generic QA, the base model gives spurious refusals on basic questions (`days in a week`, `capital of Italy`), scoring 6/8. Adapter generic scores vary from 6/8 to 8/8 and can emit nonsense (e.g. `1.7777777777777778` as an answer to days in a week). No claim of preserved/improved underlying general capabilities is justified.
- `gated_adapter` protects generic queries by bypassing the adapter, but without an explicit preference prompt it does *not* reproduce the `adapter_plus_prompt` improvement on personal tasks. The current study therefore does **not** evaluate the complete combined path of adapter+prompt+gate in the same arm; this must be added before any deployment conclusion.

## Research decision

- **Positive limited evidence:** small learned adapter can sharpen a *specified formatting preference* when combined with an explicit preference prompt in this synthetic study; improvement over prompt-only persists with a more permissive rubric.
- **Not established:** spontaneous internalized style without reminding the model (adapter-only remains 11.1%); content utility; robust general skills; genuine natural-conversation preference extraction; long-term memory reliability; data erasure; benefits across a diverse user population.
- **Do not merge the adapter as a production feature.** Frozen LM + explicit remembered preference remains a simpler, credible default.

## V7 falsification requirements

1. Include `gated_adapter_plus_prompt` (personal tasks with **both** adapter and explicit instruction; generic queries without either) and compare against frozen+prompt and conditional inference with matched new-token budget.
2. Add a **non-trained adapter** + explicit prompt, to isolate optimization benefit from residual-forward-hook effects.
3. Evaluate with longer `max_new_tokens` (e.g. 160) and stop-at-format constraints as separate arms; report prompt truncations, semantic content usefulness and repetition separately.
4. Diversify tasks, styles, independently authored unseen profiles and seeds; avoid overinterpreting duplicates of the same prompts.
5. Retain raw response artifacts and a locked human-reviewed subset; report both strict and permissive format rubrics without picking only the best-looking one.
6. Ensure memory privacy deletion policy covers audit records and stored adapter training, not just active routing state.

**All V6 work remains isolated on the experimental branch** `experiment/personal-adapter-smollm2`; no core 5M SELF model or other tracks were changed.
