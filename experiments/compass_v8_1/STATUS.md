# COMPASS V8.1 implementation

The [V8.1 pilot](https://github.com/toyro396133/personal-native-lm-5m/actions/runs/38101573253) passed its data and authorization preflight and entered rank-4 training.

Files: `fixtures_v81.py`, `guard_v81.py`, `validate_v81.py`, `pilot_v81.py`, `EXPERIMENT_V8_1.md`.

The pilot uses 192 fictional train examples and 72 held-out examples from disjoint profiles/projects (24 evaluated in this smoke), six identical-length digit labels, paired opposing examples and a separate external safety guard. Compare raw results to constant majority; do not credit the learned model for deterministic guard decisions.

**Research only:** No changes to model-internal SELF or production. Neural results not yet verified when recorded.
