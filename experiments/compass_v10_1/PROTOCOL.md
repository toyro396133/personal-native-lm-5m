# COMPASS V10.1 — Longitudinal Project Understanding Loop

Pre-registered on 2026-10-12 before implementation tests. **No API, no SELF weight changes, no source/private records in the public repository.**

## Research question
Can an incrementally maintained, provenance-linked **project model** update its working understanding after later decisions *without losing the original WHY*, while excluding unverified, irrelevant, or obsolete information? Compare to frozen historical context (which cannot adapt) and an unsafe append-only fact list (which mixes superseded claims).

This tests **state maintenance and retrieval fidelity**, not neural cognition. GPT-6 in ChatGPT may propose interpretations. Only externally reviewed and manually approved project statements become active.

## Workflow
1. Observe new **project evidence**: user answer, outcome, planning document or implementation observation. Always retain raw authoritative source outside public repo; local metadata must include immutable Git SHA + file/line or explicit traceable local source.
2. Extract **proposed update**: facet (stable key), semantic kind (goal, principle, decision, open_question, implementation_observation), why/rationale, applicability conditions, dependencies (stable facets), and whether it is a change of version. This is **PENDING** only.
3. Explicit human review through existing COMPASS V9.12 UI: approve/reject. Approver string in local UI is a recorded claim, not cryptographic authentication. A submitted approval of the correct expected revision updates only its facet.
4. Compile the **current project brief** from approved facets only, grouping WHY, principles, decisions, unknowns and observations; show evidence, revision and conditions, previous superseded versions, and suggested questions. Pending updates shown separately *outside* the active context.
5. Answer tasks using the compiled brief in GPT-6 (ordinary chat) and log the resulting human correction as new evidence; repeat.

## Frozen engineering criteria
- Direct integration with existing `compass_core.CompassStore`, **not an independent competing memory store**.
- No autoapproval, no raw legacy code replacing user project intent, no cross-project/personal promotion.
- Pending proposals do **not** affect compiled decision context. Historical records remain accessible after correction. Revision CAS prevents stale corrections.
- Conditions restrict applicability; open questions remain open, never inferred as approvals.
- Goal WHY survives implementation observation updates. No dependency traversal with unchecked cycles.
- All output traces carry exact local source and current revision. Budget truncation is explicit.
- Unit tests use fictional sources/data only, independently tested Python 3.11–3.13.
- Real Chacham Lev case study uses only project-specific **source references and GPT-6 teacher interpretations** in a LOCAL-only batch; no real approval is claimed.
- Benchmark: historical snapshot vs approved-document-simulated newer snapshot; compare whether key concepts are represented; **do not label simulation as real human approval**.

**Strict scientific verdict:** this can at most be `VERSIONED_PROJECT_STATE_LEARNING_LOOP` (engineering). **`NO_INDEPENDENT_GPT6_DECISION_QUALITY_GAIN_PROVEN`** until genuinely new decisions are rated independently under identical model/context budget using static RAG vs incremental model.

Important privacy: never commit actual Chacham Lev text, decision exports, SQLite DB or private prompts to the public COMPASS repository.
