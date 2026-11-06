# Pilot JSONL shard layout

All JSONL files in this directory form one dataset. The loader and validator read every `*.jsonl` file in this directory and combine records by the `split` field.

- Training: `train.jsonl`, `train-2.jsonl`, `train-3.jsonl`, `train-4.jsonl`, `train-5.jsonl` — 160 records.
- Validation: `validation.jsonl`, `validation-2.jsonl` — 20 records.
- Test: `test.jsonl`, `test-2.jsonl` — 20 records.

Total: 200 records, split 80/10/10. Do not move rows between splits to improve results. Test is a holdout and must not be used for training or tuning. All records currently require independent expert review and remain marked pending.
