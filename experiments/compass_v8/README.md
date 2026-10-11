# COMPASS V8 planning record

**Status:** PLAN ONLY — no V8 training, GitHub Actions matrix or performance results.

- [Experiment design](EXPERIMENT_V8.md): hypotheses, sources of truth, ranking, capacity comparison, train/test rules, metric, failure traps, compute discipline and go/no-go gates
- [Terminology](TERMINOLOGY.md): sharp separation of COMPASS (personal priorities) from SELF (model's self-mechanism)
- [Fictional scenarios](DOCUMENTATION_EXAMPLES.json): illustrative counterexamples for layer-versus-priority, task dependencies and evidence uncertainty; not training/test split data

**Previous research:** V1–V7 under `experiments/personal_smollm2/`; last audited [V7 results](../personal_smollm2/FINDINGS_V7.md). The V7 formatting and topical-keyword metrics are not success criteria for this new priority-reasoning task.

**Implementation gate:** First approve/read the planned data contract and build project-hierarchy fixtures and a rule-based baseline. Do not train ranks 4/16/64 until the trap cases and leakage checks pass.
