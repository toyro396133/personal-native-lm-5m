# Preregistration Integrity Note

## What happened
The first preregistration commit `67f6c12e923043066ae57b1f83d06a9286f4d78e` was intended to persist both:

- `hypothesis_snapshot.md`
- `benchmark_plan.md`

The hypothesis snapshot was persisted with the intended substantive content (with normal text-serialization differences such as final newline handling).

However, the file bridge used to transfer `benchmark_plan.md` returned an error string during one read, and that error string was accidentally committed as the file content. The Git blob in that commit is therefore **not** the intended causal benchmark plan.

## Why this does not alter the executed preregistration
The causal benchmark plan was not executed because no anonymized checkpoint/runner interface was available.

Before any benchmark results were generated, a second preregistration commit was made:

`2125b5ac26c5dc93a2f5e745c588dc0151969d96`

That commit correctly contains `benchmark_plan_artifact_only_addendum.md`, which specifies the five benchmarks F1–F5 that were actually executed, including their hypotheses, competing explanations, predictions, falsifiers, exact implementation, and raw metrics.

Therefore the analyses that produced results were preregistered correctly before execution.

## Frozen local hashes recorded before execution
- `hypothesis_snapshot.md`: `f8085d59b1987db593380192272352a488cb86aa1d7e92612619fa25a9a39ee3`
- intended causal `benchmark_plan.md`: `37737c712b9c522ba7504ec50d3f70631c6747e70b472209d9194a13802a3ddb`
- executed artifact-only addendum: `c3b7fea7cbf3f0a2a5b544d8e7a9b084d4b187391a81cc357c62f87882ed165f`

The final results commit restores the intended causal `benchmark_plan.md` from that frozen local copy. This restoration is post-results and must not be misrepresented as cryptographic proof that the causal plan file itself was correctly stored in the first preregistration commit.
