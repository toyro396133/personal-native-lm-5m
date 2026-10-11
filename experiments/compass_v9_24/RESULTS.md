# COMPASS V9.24 — actual measured results, honest NO-GO

**COMPASS, not SELF.** GPT-6 teacher operated in ordinary ChatGPT, **no OpenAI API**. Source was the private Chacham Lev decision log at commit `fe9863278a04dedf27a0c4fcb8cca56b91ec4544`. No raw user responses or private project documents have been committed to this public research branch.

## Before interpretation

- Ten **GPT-6 teacher-proposed** relation cases grounded in pairs of user decision-log sections. The 20 Git-source line anchors were independently checked against the correct `Tnn` source headings (20/20 valid anchors; **this does NOT independently verify labels**).
- Preregistered coarse classes: **2 REVISES**, **2 QUALIFIES**, **6 EXTENDS**. Teacher-generated old/new paraphrases are the only model features; teacher's class, rationale, key, numeric source IDs and file paths were hidden from models.
- **10 leave-one-concept-out** folds: nine concept-disjoint cases TRAIN, one TEST per fold, never retrained on test item.
- These labels were authored by GPT-6, and source paraphrases were created retrospectively. Hence this is not independent human-grounded learning.

## Results

| Same-input method | Teacher agreement | Macro-F1 | Revises recall | False-revision alerts |
|---|---:|---:|---:|---:|
| Frozen char 3–5 gram TF-IDF + logistic C=1 | **6/10** | 0.2500 | **0/2** | 0 |
| Frozen simple keyword rules | **7/10** | 0.4889 | **1/2** | 0 |
| TF-IDF 1-nearest-neighbor | 3/10 | 0.1667 | 0/2 | 0 |
| Always EXTENDS | 6/10 | 0.2500 | 0/2 | 0 |

**Student predicted EXTENDS for all ten items**, reproducing the majority baseline, and missed both known teacher-labelled corrections. The apparent 60% overall score conceals a 0% recall on the critical destructive revision class. There is **NO evidence of a learned advantage over simple controls**, and the sample (10, all from one project's discussion) is far too small.

**Research verdict: `NO_PROVEN_INDEPENDENT_RELATION_LEARNING`.**

## Concrete engineering progress

A separate **local, stdlib Python + SQLite `RelationGraph` component** was developed and executed with the ten private teacher cases; code/artifacts are available as per-conversation downloadable files, not yet integrated into production or committed here. Its ledger contains **20 proposed claims + 10 proposed edges, zero approvals** (30 append-only hash-linked events). It only authorizes an approved REVISES edge to supersede an approved earlier claim **within the specific facet**, after both claims have been individually approved and a matching revision and source pointer checked. EXTENDS and QUALIFIES **never** overwrite an old approved claim. Full reopening/replay, stale version rejection, project scoping, duplicate decision blocking, invalid source rejection and corruption detection passed **18 Python unit tests**. Hash chain is not cryptographically anchored and reviewer identity is not authenticated.

The exact original source positions were checked, but the relation classifications remain teacher-only proposals, not authority-verified truth.

## Independent review unblocks the next valid test

An offline Hebrew blind relation-review page was prepared. It has all ten old/new explanations and their pinned sources, **no teacher class, teacher rationale, or student prediction**. A person can inspect the real user statements and assign REVISES, QUALIFIES, EXTENDS, or UNCERTAIN with reasoning and a source-verification flag. Separate scoring code accepts that export without retraining or granting any memory approval. **No independent human review has occurred yet.**

Next: build a larger *independently human-annotated*, genuinely later set of corrections and completions from private multi-project work, then evaluate revision recall **and wrongful supersession risk** against equal-input keyword and retrieval baselines. Until then, don't claim a research breakthrough in autonomous project intent learning.

## Data boundaries

The source decision log, exact user messages and local SQLite graph must remain private. This GitHub branch contains only preregistration and aggregate findings. The experiment uses COMPASS components; **no SELF neural weights and no API** were used.
