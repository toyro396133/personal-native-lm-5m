# COMPASS V8 — Stage 0 implementation and audit

**Status: PASS.** [GitHub Actions Stage-0 run #38098797553](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098797553) completed successfully on 2026-10-11 (UTC). Results artifact: `compass-v8-stage0-audited-fixtures` (artifact ID `11687068326`).

## Scope and constructed fixture

- 4 entirely fictional user-policy profiles and 12 entirely fictional projects (3 distinct project kinds per profile).
- 8 authored decision situations per project, including confirmed core breakage, broad optional tests, a narrowly necessary core regression test, an uncertain report, a supporting prerequisite, planned core work, forbidden scope expansion and a change requiring owner approval.
- 64 training inputs covering 8 projects; 32 test inputs covering 4 other projects.
- Input context includes project core objective, layer/type graph nodes, currently valid sourced decisions, user-approved general policy, task proposal and evidence. The gold action, case template label and priority are **not** included in the model-facing input.
- Train and test project identifiers, task IDs and authoring template IDs are disjoint. Case kinds are not encoded in opaque task IDs. Audits check provenance, node references and project/task consistency.
- 12 project-level adversarial pairs show that a *structurally ancillary* targeted regression test is urgent when blocking core repair, while a broad test expansion without a demonstrated blocker is deferred/scheduled. Other checks distinguish confirmed from suspected failures, unauthorized scope and missing approval.
- A separately coded **template-specific** deterministic decision baseline matches 64/64 train and 32/32 test labels. This is a **sanity check for authored miniature fixtures**, NOT genuine evidence of independent generalization or valid project-priority reasoning in the real world.

## Validity risks before any interpretation

1. Fixtures have authored, narrow English templates, and the test project kind (planning) is confounded with test split. Style may be easier to generalize than genuinely unseen tasks.
2. Gold decisions and the deterministic baseline were built by the same research implementation: perfect baseline score verifies fixture consistency, not independent human agreement.
3. All four fictional preference profiles appear in training and test, so cold user-policy transfer has not been measured.
4. Stage-0 includes no learned language model, no semantic response grading, no automatic extraction of real project maps, and no genuine memory erasure.
5. Future larger-rank tests MUST use stronger independently adjudicated benchmarks before claiming performance gains.

## Next isolated step

A gated rank-4 pilot (7,680 trainable adapter weights on frozen SmolLM2-360M) is defined at `pilot_v8.py` and dispatched in [GitHub Actions #38098918328](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38098918328). It uses 32 updates and 16 held-out project decisions; evaluates 6 exact action-token choices for frozen graph-only, frozen+explicit-policy, adapter graph-only and adapter+explicit-policy, versus rule baseline. This measures a limited candidate-ranking ability only — NOT independent free-form decision reasoning, rationale quality or production readiness.

The full rank × seed matrix remains **not launched** and no outcome is claimed until its own preregistered controls and audit pass.
