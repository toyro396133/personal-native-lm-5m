# SELF V3: exhaustive rehearsal and query-only retrieval

V3 is the direct follow-up to V2 (see FINDINGS_50_V2.md). It is an English, synthetic-user memory experiment, **not** a deployed or verified personal assistant.

## Hypotheses

1. Every previously introduced fact receives at least one replay update in **every** later stage.
2. New facts receive two exposures in their introduction stage.
3. A query-only lexical retriever can find relevant facts from an indexed set of fictional notes without access to answer labels or ground-truth IDs.
4. A trained per-user residual adapter combined with retrieved evidence may perform differently from either alone.

## Four conditions

- `base`: frozen SmolLM2-360M-Instruct; no personal context.
- `personal`: frozen SmolLM2 + trainable ~7,680-parameter residual adapter; no notes.
- `retrieval`: frozen SmolLM2, with **top-1 text** from query-only lexical TF-IDF/bigram matching, no adapter.
- `hybrid`: same adapter + same retrieved text.

Retrieval ranking uses only evaluation question text and each note's *statement* (not evaluation labels, right/wrong answers, or true fact IDs). It reports hit@1 / hit@3. Notes are limited to facts introduced so far; no future-fact leakage. It is deliberately a simple lexical retriever, not a trained semantic retriever. Dataset patterns (e.g. "project 1") can make retrieval unnaturally easy and this must not be generalized.

## Training and scope

50 facts, released five at a time over 10 stages. At stage `s`, use `5s + 5` training updates: one per known fact and one additional per new fact. Total = 325 updates; V2 had 250, so this is an intentional rehearsal-coverage intervention **not a matched-compute comparison**. Each stage saves full per-fact result indicators, train exposure counts, retrieval selections, summary metrics (all/old/new), and a personal safetensors checkpoint.

Evaluation is still binary candidate conditional likelihood ranking of unseen question *wording*, not free-form recall. Held-out new content and multi-seed validation are not yet provided. The prior V2 oracle retrieval at 98% is an **upper bound**, not equivalent to V3's real query-only retrieval.

## Success criteria

- No old fact is omitted in the stage's training plan.
- Retrieval top-1 selection quality is measured separately from model answer quality.
- Compare final performance and old/new retention across all four conditions.
- Avoid claims of genuine continual memory without broader open-answer testing and repeated seeds.

Launch via `.github/workflows/self-50-facts-v3.yml` from `main`. Outputs are stored as GitHub Actions artifact `self-50-v3-results`.
