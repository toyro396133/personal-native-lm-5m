# Pilot JSONL shard layout

All JSONL files in this directory form one dataset. Load every `*.jsonl` shard and combine by the `split` field; do not assume one file per split.

The corpus contains **5,000 records**: 4,000 train, 500 validation and 500 test. The 2,000-to-5,000 expansion adds 3,000 records across 600 scenario groups, with five task-format variants per group.

- Training shards: `train.jsonl`, numbered `train-2.jsonl` through `train-37.jsonl`; new expansion shards `train-20.jsonl` through `train-37.jsonl`.
- Validation shards: `validation.jsonl`, numbered `validation-2.jsonl` through `validation-11.jsonl`.
- Test shards: `test.jsonl`, numbered `test-2.jsonl` through `test-11.jsonl`.

Each scenario group remains entirely within one split. Test is holdout-only. All 5,000 records remain pending independent expert review; none is approved training data.
