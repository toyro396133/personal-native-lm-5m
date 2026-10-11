# COMPASS V9.12 — first runnable persistent COMPASS memory core (not an AI-learning breakthrough)

**Main acceptance workflow:** https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38130787162
**Passing acceptance run:** `success` for Python **3.11**, **3.12**, **3.13** and aggregation.
**Independent engineering tests:** 25 distinct tests per Python version = **75 successful test executions**. Every version also ran a complete persistent SQLite demonstration, including versioned correction, isolation and reopen (**3/3 demos PASS**).
**Code:** [compass_core.py](compass_core.py) · [demo_compass.py](demo_compass.py) · [tests](test_compass_core.py) · [Windows instructions](README.md)
**Preregistered protocol:** [PROTOCOL.md](PROTOCOL.md).

## Verifiable engineering result

V9.12 moves beyond the old question-answer toy benchmark by providing an actual dependency-free, Python 3.11+ program to **store and maintain** a personal COMPASS + per-project COMPASS memory:

1. Persistent local SQLite event journal, with original source text and timestamps; verified integrity chain from genesis (hash chaining only, no externally secured anchor).
2. Explicit approved principle proposals with provenance, human rationale, stable key, context conditions, author string and approval actor string; source claims are NOT independently authenticated.
3. Personal and project scope separated; project-specific facts cannot be silently promoted to global principles.
4. Approval-only updates and revocations; stale expected-version proposals fail atomically and preserve previous approved principles.
5. Each principle correction leaves its previous revision and rationale in the auditable history; later implementation notes do not overwrite it.
6. Reopen of the file reconstructs approved state from journal events, validates chain and provides a project+user scoped view filtered by exact-match context.
7. Conservative text ingestion: normal freeform Hebrew/English messages remain `UNCLASSIFIED`; an explicit `COMPASS:` structured instruction becomes a **pending proposal**, never an approved principle by itself.
8. CLI runs on Windows PowerShell and Linux, no pip package, GPU, Node or Docker required. Demo runs on isolated fictional data and simulates approval for demonstration, NOT actual human verification.

## Gate report

| Preregistered gate | Outcome |
| --- | --- |
| All acceptance tests pass on 3 Python versions | PASS (25/25 each) |
| No unapproved freeform text may change active memory | PASS |
| No project A -> project B or project A -> personal promotion | PASS |
| All stale revision conflicts rejected without partial commit | PASS |
| Approved revisions retain reason, source and prior revisions | PASS |
| SQLite close + reopen recovers identical state and audit hash | PASS |
| Hash-chain deliberately corrupted payload/deleted event detected | PASS |
| 16 manually specified varied project/personal principle cases | PASS |
| CLI and full SQLite demo on all 3 Python versions | PASS |
| Learning unknown human principles from unlabelled long conversations | **NOT YET TESTED, NOT LEARNED** |

**Engineering acceptance:** `COMPASS_OPERATIONAL_MEMORY_CORE_V1`.
**Scientific cognitive ability:** `NOT_YET_LEARNED`.
**Production authentication/security:** `NOT_READY` (open local database + untrusted actor strings).

### Development failures recorded

- Initial integration test used a one-character project key; validator required >=2 characters. Validator relaxed to accept a canonical single letter.
- Next test expected 5 applicable project-B principles, but one had an explicit `phase=migrate` condition and was properly excluded from an empty-context query; corrected the test to assert 4 normally and 5 when condition matches.
- Earlier test class duplicated inherited cases, removed duplication; 25 truly distinct assertions remain.

No test score was tuned on neural cold labels; these were ordinary programming tests. All passing CI and demo results were verified after corrections.

## Why this matters — and why it is not yet the user's requested ultimate COMPASS

Unlike the prior 9.x toy modules, V9.12 is **usable program code**: it can retain a user's approved long-term project vision, explicit reason and corrections across processes without allowing random new chat content to displace it. This is a tangible prerequisite for work over months and years.

However the difficult *learning* question remains **entirely unresolved**: raw conversations are not transformed into reliable candidate personal/project vision entries by a language-understanding model. Current `observe` intentionally abstains. Successful storage and human-controlled updates are NOT evidence of autonomous intent learning or real-world accuracy. Any claim of a breakthrough in understanding would be false.

## Immediate next COMPASS priorities

1. Independent evaluation with authentic and permission-cleared real project conversation histories, particularly Hebrew and changing scope/intent; do not push private chats into public repo.
2. Candidate extraction from new messages **as unapproved proposals** with supporting exact source span, context conditions, `correction-of` and confidence/ASK. Compare equal-input no-LLM/LLM/retrieval baselines with independent annotated examples and abstention cost.
3. Human review/approval workflow and source authorization in a real interface, with authentication, durable external audit anchor, backups, failure recovery and migrations.
4. Link project-specific acquired insights to global personal COMPASS only after explicit validated generalization, not silent copying.
5. Replace expensive O(N)-replay for each operation with verified immutable snapshots/materialized views and real-scale stress tests before production use.

**Project focus remains COMPASS; no underlying SELF model was trained, altered or integrated.** V9.12 experimental code remains on `experiment/compass-v9-12`, CI workflow trigger only on `main`.
