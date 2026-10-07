# Reproducibility

## Blind source
- Repository: `toyro396133/personal-native-lm-5m`
- GitHub Actions run: `37694378705`
- Artifact: `four-component-blind-dataset-47`
- Artifact ID: `11514696554`
- Artifact ZIP SHA-256: `8522f3c5271cf00d1f5ecdfe98a57de4262a2ad6430f2d32086f38be036df285`
- Extracted dataset JSON SHA-256: `eccac8ede7c4701bdc96a5c4ca5b895b466dc117245b0a00ddc7eb41533c9ebb`

## Preregistration commits
- Hypothesis snapshot + causal benchmark plan: `67f6c12e923043066ae57b1f83d06a9286f4d78e`
- Artifact-only addendum: `2125b5ac26c5dc93a2f5e745c588dc0151969d96`

## Preregistration integrity
The first preregistration commit correctly persisted the hypothesis snapshot, but its intended causal `benchmark_plan.md` was accidentally replaced by a file-bridge error placeholder. The five analyses that actually ran were nevertheless fully preregistered in the second commit (`2125b5ac...`) before execution. See `preregistration_integrity_note.md` for details.

## Frozen preregistration hashes
- `hypothesis_snapshot.md`: `f8085d59b1987db593380192272352a488cb86aa1d7e92612619fa25a9a39ee3`
- `benchmark_plan.md`: `37737c712b9c522ba7504ec50d3f70631c6747e70b472209d9194a13802a3ddb`
- `benchmark_plan_artifact_only_addendum.md`: `c3b7fea7cbf3f0a2a5b544d8e7a9b084d4b187391a81cc357c62f87882ed165f`

## Randomness
Primary seed: `104729`.

- F1 cluster bootstrap: seed + 1, 2000 resamples.
- F2 checkpoint bootstrap: seed + 2, 3000 resamples.
- F3 lag permutation: seed + 3, 5000 permutations.
- F4 checkpoint bootstrap: seed + 4, 4000 resamples.
- F5 distance-matrix tests: 10000 label permutations per comparison with deterministic token-derived seeds.

## Execution

```bash
python scripts/run_followup_benchmarks.py
python scripts/run_exploratory_counterchecks.py
```

The scripts expect `dataset.json` extracted from the exact artifact above at the configured source path. No repository documentation or experiment source code is required.

## Execution notes
An initial implementation attempt aborted before benchmark results because the dataframe column `mode` collided with the pandas `.mode` method. A subsequent mathematically equivalent implementation used explicit column indexing. A second run hit CPU timeout in naive bootstrap/permutation loops; those loops were replaced by vectorized sufficient-statistic/permutation calculations without changing seeds, sample counts, metrics, predictions or falsification criteria.

## Blindness
No unblinding source was used. The causal checkpoint plan was not executed because no anonymized checkpoint manifest/runner was available.
