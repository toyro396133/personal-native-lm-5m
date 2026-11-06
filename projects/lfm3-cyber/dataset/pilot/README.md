# Pilot JSONL shard layout

All JSONL files in this directory form one dataset. Load every `*.jsonl` file and combine by the `split` field.

- Training: `train.jsonl`, `train-2.jsonl` through `train-6.jsonl` — 400 records.
- Validation: `validation.jsonl`, `validation-2.jsonl`, `validation-3.jsonl` — 50 records.
- Test: `test.jsonl`, `test-2.jsonl`, `test-3.jsonl` — 50 records.

Total: 500 records, split 80/10/10. The 60 newest scenario groups each have five format variants and remain within a single split. Test is holdout-only. All 500 records require independent expert review and remain pending.
