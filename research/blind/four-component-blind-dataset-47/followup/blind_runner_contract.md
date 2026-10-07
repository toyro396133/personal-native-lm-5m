# Blind Runner Contract Required for Causal Follow-up

This contract is intentionally semantic-free. A runner satisfying it would allow the preregistered causal benchmarks to execute without revealing real A/B/C/D or V1–V7 identities.

## 1. Checkpoint manifest
Return only:

```json
{"variant_id":"V1","tokens_m":25,"checkpoint_handle":"opaque:...","sha256":"..."}
```

No real architecture/checkpoint names may appear.

## 2. Sample interface
The runner should expose sample IDs and anonymous equality predicates only:

- `same_A`, `same_B`, `same_C`, `same_D`
- `different_A`, `different_B`, `different_C`, `different_D`
- blind-safe nuisance transforms such as paraphrase/distractor/context-length IDs

No semantic labels.

## 3. Activation interface

- `get_activation(checkpoint_handle, sample_id, layer_id)`
- `patch_component(component_id in A/B/C/D, layer_id, operation, seed, target_norm)`
- operations: `zero`, `opposite`, `random_same_norm`, `orthogonal_same_norm`, `interpolate(alpha)`, `donor_patch(donor_sample_id)`

The implementation may know real internals; the returned logs must not expose them.

## 4. Output interface
Return primitives only:

- raw logits or blinded task correctness
- activation displacement L2
- cosine/angle
- anonymous component projections
- error category IDs
- downstream layer displacement

## 5. Safety rule
If an operation cannot be executed without exposing a real component/variant/checkpoint name, fail closed with `BLINDNESS_BLOCKED` rather than printing the name.
