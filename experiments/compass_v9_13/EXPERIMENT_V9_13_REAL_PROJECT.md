# COMPASS V9.13 — Chacham Lev real historical documents (PRIVATE source, no copied data)

**Exploratory pilot executed 2026-10-11**, first-ever COMPASS analysis on the user's genuine project documentation rather than fabricated synthetic task histories.

## Governance and boundary
- **COMPASS** is the object of work, not SELF weight training.
- Source repository is privately accessible `toyro396133/chacham-lev`; it and its raw docs MUST remain private. This public COMPASS experiment contains **no source documentation**, donor/student rows, private conversations, credentials, or embedded learned term vectors.
- Source snapshot fixed to **`55b0ac175513` (2026-10-02 23:19:37 UTC)**, eight manually allowlisted product/architecture/dev documents. Evaluation documents are October 7–11 project decisions, loaded read-only from a later repository snapshot. Exact train/test refs must be pinned for reproducibility and code must assert no future docs appear at historical cutoff.
- Manual research queries and labels are developer-curated *after seeing relevant documents*: they are **not independent blinded human judgments**, and any novelty threshold fitted after inspecting scores must NOT count as a preregistered accuracy metric.

## What was genuinely computed on source documents

On the read-only private repository, a keyword/TF-IDF paragraph embedding was fitted from **8 historical files, 242 paragraphs and 3,638 unique lexical tokens**. Six researcher-written probes of historically established project rules were matched to their expected older source file: TF-IDF top-three **6/6**; identical-input lexical-overlap baseline top-three **6/6**. On six later-decision probes, similarities were lower, but two queries were subsequently found to be misgrounded/too vague; **the 6-probe novelty result is invalid and should not be reported as independently validated**. No binary "new principle" classifier was trained or tested blind. This is **information retrieval**, not learning project vision or reasoning.

The low similarity distribution on the exploratory late-decision probes and the perfect source retrieval of obvious historical rules may be useful for debugging. They do **not** show any advantage over a trivial baseline or ability to infer intentions.

## Protocol for a defensible next stage

1. Run only on the user's local clone of the private project or an access-restricted trusted runner. Read an allowlist of planning/product documents from a pinned **historical Git commit**, never `main` for TRAIN, and hash every source; do not ingest donor/student/customer personal data.
2. Freeze tests and train cutoff before any new predictions. Include difficult **ambiguous and contradictory** decisions, not just retrieval of obvious phrases, and label provenance/decision status by independent humans. At least 20 real long-running decision episodes, including later corrections and unreconciled drafts, is the target for a meaningful future scientific study.
3. Fit simple lexical/TF-IDF retrieval on the real historical docs; compare to equal-input token overlap and a method that retrieves merely recent paragraphs. Require margin over those controls. Compute source support separately from whether an inference about a decision is justified.
4. Candidate new project principles must be **PENDING_REVIEW** with exact file path, revision SHA, source span, reason and confidence. An unverified document excerpt never becomes an approved COMPASS principle. Do not merge project-specific obligations into a personal COMPASS automatically.
5. Keep **later** decisions out of training and ensure data-leak checks verify historical absence. Evaluate approval vs proposal vs supersession separately; avoid guessing from 'approved' words alone.
6. Record false updates, abstentions, source coverage and semantic errors. If the model only matches existing docs or performs equal to keyword baseline, report NO PROVEN COGNITIVE GAIN.
7. Preserve previous V9.12 operational SQLite and local browser UI as the *downstream* state holder; train or evaluate extraction separately. Full approval with user review is required before any mutation.

**Current scientific status:** `REAL_SOURCE_RETRIEVAL_PILOT_COMPLETED`; `NO_PROVEN_INTENT_LEARNING_GAIN`. No extra scores should be invented.

**Operational limitation:** GitHub connector allowed read access to the private project, but creating an experiment branch there was denied by the platform's safety controls. Do not attempt to circumvent that denial. This protocol stays in public COMPASS repo without reproducing private source contents, and the reproducible local runner is supplied as a separate user artifact rather than committing private data.
