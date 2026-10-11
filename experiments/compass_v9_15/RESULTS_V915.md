# COMPASS V9.14 + V9.15 — REAL Chacham Lev document-status pilot: status extraction works; unsupervised intent learning NOT established

**Source:** read-only, private `toyro396133/chacham-lev`, pinned SHA `fe9863278a04dedf27a0c4fcb8cca56b91ec4544`. Do NOT reproduce private source text or user/tenant data in the public COMPASS repository.

**Engineering CI:** V9.14 [run 38133226742](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38133226742) PASS after fixing a nested-counter summation bug; V9.15 [run 38133698893](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38133698893) PASS after fixing test-detected mask leakage. All Python 3.11/3.12/3.13 pass **21 distinct invented-source tests per version** (11 source miner + 10 weak-learner/reviewer tests) = **63 test executions**. No private source data went to CI.

## Real-source inspection, NOT independently labeled accuracy

Read six genuine project documents via GitHub private read access:
- importance map
- interview-documentation protocol
- telephone core closure review
- telephone call-outcomes-and-statuses proposal
- telephone core architecture
- operational interview live notes

Total **302 Markdown section headings**. A conservative heading-status interpreter found **16 explicit status headings**: **10 approved claims**, **4 proposal claims**, **2 historical claims**; **286 unspecified**. These are merely a document author's status *claims*, not authenticated approvals. The general document-level 'proposal' headline had **eight specifically approved subheadings**, demonstrating an important real structural error in classifying an entire document from its title alone. No actual authority was inferred from the headings. Note: aggregate counts were obtained using a parallel in-session implementation of the locked heading patterns on private document text; the Python miner's code was separately tested using invented Git documents. The private source excerpts were not exported.

A safe classifier must not blindly declare the other 286 headings approved. The ability to **preserve evidence source, line and project scope and explicitly abstain** is an operational improvement over naive blanket approval classification, but not autonomous understanding.

## V9.15 locked real-doc weak-supervision attempt: INVALID TRAIN SET

Preregistered TRAIN: eight private project docs; HOLDOUT: two other project docs. A section becomes a weak/silver-labeled sample only when its OWN heading explicitly states approved, proposed or historical and its body is sufficiently long. Body text alone is the model's feature; the status heading is not included.

| Class | Original TRAIN samples | Held-out silver samples |
| --- | ---: | ---: |
| APPROVED_CLAIM | **5** | **9** |
| PROPOSED_CLAIM | **0** | **4** |
| HISTORICAL_EVIDENCE | **2** | **0** |
| Total | **7** | **13** |

**Strict preregistered abort:** Missing the proposal class in training => **`INSUFFICIENT_REAL_SILVER_EVIDENCE`**, do not train or report predictive accuracy as an evaluation. A parallel diagnostic calculation on this invalid split was inspected, but it has no valid model-scientific meaning and is deliberately omitted from headline accuracy. Even if the split had all classes, 7 training examples and heading-derived 'truth' could not support a claim of robust intent comprehension.

## A concrete next step: independent private human annotation

Added [human_review.py](human_review.py): **offline Hebrew RTL review tool** that opens 1–200 locally selected source-backed sections from the user's own private Git checkout. It provides heading, short evidence excerpt, exact Git/ref/line source and a reviewer interface to assign one of:
- currently approved
- proposed but not approved
- historical description
- superseded/revoked
- uncertain/needs clarification
- not a principle/decision

The model's weak automatic heading label is NOT exposed as a suggestion; human reviewer must enter an evidence rationale for each annotation. The page uses textContent, escapes embedded JSON against script injection, uses no network and downloads private reviewed JSON to the user's local machine for later evaluation. No autoapproval. CLI examples in [README_HE.md](README_HE.md).

### What remains to demonstrate a true COMPASS scientific gain

1. A human-independent private data set with sufficiently many *different* kinds of decisions and real corrections, with rationale and verified owner/date/scope. Two annotators and adjudication wherever possible.
2. A chronological split: train only on early project decision histories and test on genuinely later approved corrections and new principles.
3. An extractor that predicts **which concept is changed, the proposed delta, why it applies, and what evidence** — not just the word 'approved' in its heading. Abstention on uncertainty; no autoapproval.
4. A win over equal-input retrieval, exact-rule and shallow classifier baselines in the same held-out human-judged cases, including false global-project leakage and cost of wrongful overwrites.

## Honest verdict

- `REAL_PRIVATE_SOURCE_STRUCTURE_EXTRACTOR_READY` — deterministic, auditable candidate mining prototype.
- `OFFLINE_HUMAN_EVIDENCE_LABELER_READY` — tests passing on fictional source docs, generated HTML not yet validated by a real human review.
- `INSUFFICIENT_REAL_SILVER_EVIDENCE_FOR_WEAK_TRAINING` — the intended learned class detector cannot be fitted validly on the locked data split.
- **`NO_PROVEN_AUTONOMOUS_INTENT_LEARNING`**. Neither Chacham Lev nor the underlying SELF language model were changed.

This is **COMPASS** project research, NOT self-model LM training and not release-ready cognitive memory.
