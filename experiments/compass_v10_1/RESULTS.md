# COMPASS V10.1 — Longitudinal Project Understanding: measured results

2026-10-12 | No OpenAI API | No SELF model changes | Private Chacham Lev data never committed to this public repository.

## What is now implemented

- `learning_loop.py` uses the **existing V9.12 CompassStore SQLite append-only journal**. Each source-anchored `project` facet has a stable name, typed statement (`goal/principle/decision/open_question/implementation_observation`), WHY, dependencies, applicability conditions and optimistic expected revision.
- GPT-6 or a source reader can **PROPOSE** candidate updates but never automatically approve them. Only explicit review in the existing V9.12 local UI can make an update active. Recorded reviewer ID is an unverified identity claim, not authentication.
- `model_snapshot` uses only approved project-scoped statements. It excludes pending candidates, segregates legacy implementation observations from product intent, flags unsupported dependencies, honors applicability and preserves every prior version.
- `compile_brief` produces a task-oriented, source-linked snapshot within a character budget. WHY-goals are shown first, then other material. It labels source strings as untrusted **DATA**, quotes claims/WHY as JSON values (limited mitigation; it cannot guarantee LLM prompt-injection resistance), and exposes omitted facets due to budget.
- `chronological_study.py` can run retrospective event sequences in a temporary SQLite database and explicitly marks all approvals `SYNTHETIC_SIMULATION_NOT_REAL_USER`. Nothing is propagated to a real user/project database.

## Engineering measurements

| Evidence | Result | Limit |
|---|---:|---|
| Fictional local integration/unit cases | **29 distinct PASS** | Tests design invariants, not semantics |
| [GitHub Actions run 38184257420](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38184257420) | **29 PASS x Python 3.11/3.12/3.13 = 87 executions** | No private source data on public runner |
| Real private Chacham Lev source line spans checked against pinned Git | **11 unique referenced sections matched expected headings** | Verifies pointers, not correctness of GPT-6 interpretations |
| Local private 13-event teacher timeline *source-sequence sanity check* | **PASS** | Independent local script checks recorded sequence, **not** the V9.12 backing engine |
| Retrospective teacher context facets at stages A/B/C | **4 → 7 → 10** | Determined by staged input, not learned via ML |
| Explicit teacher corrections retained as two versions | **3** | Documentation-derived change proposals |
| Previously stated WHY goals lost through later changes | **0** | Structural invariant |
| Real approved user memory updates created | **0** | Required safeguard |

The local private 13-event source bundle contains four early project-purpose/temporal-identity/provenance principles from pinned Oct 2 documents, three intermediate telephone choices, and six later teacher proposals from an Oct 11 question-answer log. **Three later events revise the same facet as a preceding event** (annual telephone route concept, responsibility division, daily workflow timing). In the separate backend CI simulation, the same *shape* of event sequence retained old versions, kept original WHY, and showed the updated only-after-review value.

**Do not conflate:** the local 13-event source timeline was sequence-checked directly; the real V9.12 approval engine was exercised with fully **fictional** data in GitHub Actions. The full 13-event private bundle can be replayed on the user's own Windows machine using `chronological_study.py`; no such run on full private bundle was made through GitHub Actions.

## Crucial remaining research gap

This is **state maintenance**, not demonstrated independent **understanding**. All 13 source-specific descriptions were distilled by GPT-6 after reading sources; they are not independently judged human labels. The fact that a versioned database knows that revision 2 replaces revision 1 is expected behavior and does not prove it recognized a correction from raw speech.

No statistically valid same-model four-arm comparison (no context / RAG / flat facts / V10.1 evolving model) was completed. A fair comparison must involve **previously unseen user tasks**, exact matched GPT-6 budgets, independently assessed fidelity to user intent and rates of wrong certainty, repeated work and unwanted supersession. With ordinary ChatGPT and no API, this needs deliberately separated fresh sessions or independent human evaluation; don't simulate blindness inside a single already-informed teacher session.

**Scientific verdict:** `V10_1_VERSIONED_PROJECT_STATE_LOOP_ENGINEERED` and `NO_INDEPENDENT_COGNITIVE_GAIN_PROVEN`.

## Run locally

On Windows, select `experiment/compass-v10-1-learning-loop` in the research checkout; run the no-API `chronological_study.py` with the user's privately downloaded `COMPASS_V101_CHACHAM_PENDING_STAGES.json`. It creates and removes a disposable SQLite DB, writes only a user-local JSON study report, and cannot authorize live policies.

Use `learning_loop.py propose/snapshot/brief` for real **human-reviewed** project memory instead; do not copy synthetic approvals.
