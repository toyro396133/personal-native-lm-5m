# SELF SmolLM2 V3 — audited 50-fact continual-personalization experiment

**Run:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38090026196 (SUCCESS)
**Artifact:** `self-50-v3-results`, GitHub artifact ID `11682998719`. Contains `benchmark-50-v3-results.json` plus ten `personal-v3-stage-*.safetensors` checkpoints.
**Verified:** GitHub Actions run and job statuses, job logs, and parsed JSON artifact.

## Setup

- Frozen backbone: HuggingFaceTB/SmolLM2-360M-Instruct.
- Per-user, last-layer residual adapter: 7,680 trainable parameters.
- 50 *fictional*, highly structured profile facts, introduced five per stage over ten stages.
- Rehearsal: each **old** fact once and each **new** fact twice per stage (training steps: 10, 15, …, 55; **325** total).
- One seed (19). Evaluation is **binary candidate likelihood ranking**, not free-form recall.
- `base`: no adapter, no external notes.
- `personal`: adapter only.
- `retrieval`: base model + lexical query-only top-1 note retrieval (no answer / fact-ID oracle).
- `hybrid`: adapter + *identical retrieved note*.

## Observed results

| Stage | Total facts | Base | Personal | Retrieval | Hybrid | Retriever hit@1 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5 | 2 | 2 | 5 | 5 | 5 |
| 2 | 10 | 5 | 10 | 9 | 9 | 9 |
| 3 | 15 | 8 | 15 | 14 | 15 | 14 |
| 4 | 20 | 10 | 15 | 19 | 20 | 18 |
| 5 | 25 | 13 | 19 | 24 | 25 | 23 |
| 6 | 30 | 16 | 24 | 28 | 30 | 27 |
| 7 | 35 | 18 | 22 | 33 | 33 | 32 |
| 8 | 40 | 20 | 26 | 36 | 38 | 36 |
| 9 | 45 | 23 | 31 | 41 | 44 | 41 |
| 10 | 50 | **27 (54%)** | **35 (70%)** | **46 (92%)** | **49 (98%)** | **45 (90%)** |

At final stage, top-3 retrieval is **50/50**; top-1 is 45/50. All five top-1 errors concern the `activity-1` through `activity-5` facts: the lexical retriever returns the corresponding `drink-N` fact because the question and both notes share “break N”. Of these, the hybrid recovers 4/5 answers and misses `activity-1`. Thus hybrid robustness to noisy retrieved context is the most interesting *specific* observation here.

## Paired final accuracy tests (two-sided exact McNemar / binomial discordants)

- Hybrid vs retrieval: **3 hybrid-only correct, 0 retrieval-only**, p=0.25. The +6 percentage point benefit on 50 synthetic questions **is not statistically compelling**.
- Hybrid vs personal: **14 hybrid-only correct, 0 personal-only**, p≈0.000122 (but synthetic examples are correlated; do not generalize broadly).
- Personal vs frozen: **10 personal-only, 2 frozen-only**, p≈0.0386, exploratory and without multiple-test correction.
- Personal V3 vs V2 on identical 50 final test items: **8 V3-only, 5 V2-only**, final accuracy 35/50 vs 32/50, not an established improvement.

## Interpretation

1. **Exhaustive replay was achieved**: JSON confirms stage 10 has all 50 training IDs, with once per old fact and twice per new fact; stage 10 personal old-fact accuracy was **30/45** vs **27/45** in V2. The higher final performance is modest and V3 uses **325 training updates** vs 250 in V2 (non-matched compute).
2. The 7,680-weight personal adapter helps in this narrow controlled setting, but **70% recall does not qualify as reliable personal-fact memory**.
3. Practical lexical retrieval alone reaches 92% on this constructed task. The hybrid gets 98%, but its margin over retrieval requires stronger evidence.
4. All testing is **forced choice between two answers**; no autonomous free-text fact recall, unrelated benchmark retention, multi-seed training, realistic chat history or privacy test was done.
5. The query-only retriever is intentionally simple and benefits from a templated question/fact domain. Top-1 retrieval is **not an oracle**, but the constrained data make the match easier than real-world personal history.
6. First and last logged training losses come from *different examples* and cannot establish consistent loss convergence.

## Research next step

Implement top-k query-based memory retrieval with disambiguation of overlapping `break N` intents, add held-out free-form questions, multiple seeds, realistic mixed/noisy conversations, explicit personalization-vs-factual-regression benchmarks, and matched-compute rehearsal comparisons (e.g. identical total updates). Preserve both successful and failed results without merging the experimental branch into production as if continual learning were solved.

**Do not claim** V3 proves that training a small adapter universally improves retrieval-based personalized assistants; it is evidence that such a hybrid is feasible and merits more rigorous study.
