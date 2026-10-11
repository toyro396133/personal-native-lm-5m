# COMPASS V9.12 — From toy scores to an operational, auditable memory core

**Preregistered before implementation tests and before inspecting any V9.12 results.**

## Scope

This work is strictly **COMPASS**, not the underlying SELF model, training a large language model, or making a claim of general natural-language intent understanding. V9.11's versioned principle slots improved synthetic matched outcomes (82–86/128 vs 32/128 continuous GRU) but an ordinary first-three-items shortcut tied them. The critical missing functionality is writing/updating those slots from genuine work while preserving vision, reason, scope, corrections and uncertainties.

## Two V9.12 deliverables

**A. Executable end-to-end COMPASS core** that can be used from Python and the CLI without model weights, servers or external dependency:
- **SQLite persistent** event-sourced state with a personal ledger and a separate ledger per project, attached to canonical project identifiers. New events, proposals, authorizations, principle revisions, validity and reasons persist over restarts.
- An **immutable append-only event history at the application interface** with previous-event hash linking, verified from genesis. This is a corruption detector, NOT cryptographically authenticated or an externally anchored tamper-proof ledger. Protect database file with OS access controls; never treat user-supplied `actor_id` as authenticated.
- **Principles**: user vs project scope, owner ID, stable key, content/vision, reason, evidence/source event, applicability conditions, status ACTIVE/REVOKED/SUPERSEDED, revision, explicit approval actor/time, history of changes. The project COMPASS links to (but is not substituted by) the personal COMPASS.
- **Unstructured input becomes a PROPOSAL only**, never an automatically approved new global preference. In a deliberately conservative parser, recognize explicitly addressed `COMPASS:` change commands with a scope, key, content, reason and optional applicability; ambiguous text is archived as an `UNCLASSIFIED` event or flagged for human clarification without mutating the principle registry.
- Explicit `approve` action is required to commit an event (including corrections and revocations). Append-only correction preserves the earlier principle and the new version. Reject unknown slot, missing consent, empty reason, invalid project context, unrelated-project editing, stale revision and confused global-vs-project scope.
- `resolve(scope, project_id, context)` produces applicable ACTIVE project principles + eligible global principles (when conditions match), and always returns reasons/sources/versions. No silent promotion of project-only to global, no enforcement of binding source claims based only on text. Conditions are a tightly restricted exact-match JSON map, not code.
- Deterministic replay from events into snapshot; query must match replay and hash verification. SQLite transactions guarantee no half-committed event vs projection.
- A functional **CLI** for init, propose, approve, revoke, list, timeline, verify, and resolve with a durable local SQLite path. Demonstrate workflows using realistic Hebrew project-vision and personal-preference examples without any private identifiers.

**B. Honest evaluation** independent of the old binary V9.4–V9.11 toy task generator:
- Unit + integration tests test cross-project isolation, corrections, persistent reopen, out-of-order source timestamps vs event order, two independent concurrent principles, conditional applicability, one failed proposal plus intact state, corrupted audit hash detection, untrusted free-text import cannot mutate state, conflict/revision safety, CLI round trip.
- A fixture corpus with **manual expected outcomes** for at least 16 varied English/Hebrew mixed project history events, including interleaving principle/background/correction cases; shuffled arrival order must follow explicit revision/supersession instead of the newest text.
- Compare to **first/last-N context shortcut** on cases where principle evidence is interleaved. Be explicit that the new engine uses extra **structured approved metadata**. Its source trust is an administrative contract, not machine-learned text comprehension.
- Predeclared engineering gates: all core tests pass, no silent auto-approval 100%, no cross-project mutation 100%, replay consistency 100%, revision conflict check 100%, explicit audit reason for each state revision 100%; all 16 or more fixture checks match manual expectations, no test skipped.
- **Separate learning claim gate**: a test where a model autonomously discovers new vision/reason/supersession from freeform unseen user/project histories WITHOUT privileged annotations and beats equal-information baselines. V9.12 intentionally **does not** satisfy this yet. Its scientific status must be `NOT_YET_LEARNED` no matter how many structural tests pass.

## Acceptance and interpretation

A passing result is **`COMPASS_OPERATIONAL_MEMORY_CORE_V1`** — a tangible running program that persists, audits and respects user/project boundaries, not a COMPASS model that independently understands people. If anything fails, no such status. Do NOT integrate into the production site or any model automatically.

Next research phase, if core is solid: develop **proposal extraction and correction detection** on genuinely independently curated Hebrew/English transcripts, with calibrated abstention and human verification. Compare explicit-vocabulary rules, retrieval and language-model extractors on the SAME unlabeled incoming messages. Source ownership and approval boundaries remain non-negotiable.
