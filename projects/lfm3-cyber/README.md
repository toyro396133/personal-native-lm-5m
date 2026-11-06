# LFM3-Cyber — isolated training workspace

This project specializes `LiquidAI/LFM3-1.7B` for general cybersecurity work, with defensive analysis and explicitly authorized lab assessment. It is deliberately isolated from the repository's SELF and COMPASS research lines.

## Current stage

The dataset has reached **500 original synthetic examples** across eight domains: 400 train / 50 validation / 50 test. All 500 remain pending independent expert review and are not yet approved training data. No full training run has been started and no model weights have been downloaded for this dataset phase.

- Base checkpoint: `LiquidAI/LFM3-1.7B` (post-trained conversational/tool model).
- Pinned upstream revision: `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
- First adapter: `lfm3-cyber-lora-v1`.
- Method: supervised fine-tuning (SFT) with one LoRA adapter.
- Canonical dataset directory: `dataset/pilot/`; all `*.jsonl` shards are loaded together and reassembled by split by `prepare_dataset.py`.
- Current orientation: 488 defensive, 6 conceptual and 6 explicitly authorized isolated-lab examples.
- Retrieval (RAG) supplies changing references; LoRA should teach domain behavior and structured task performance, not memorize changing threat intelligence.

## Repository boundaries

All project code, configs, fixtures and workflow files live under `projects/lfm3-cyber/`, except the dedicated workflow at `.github/workflows/lfm3-cyber-ci.yml`. Existing SELF, COMPASS, Chacham Lev, model-mirroring and research workflows are not changed.

Training outputs, model weights, real logs, captures, credentials and private datasets must never be committed. Use GitHub Actions artifacts for short-lived reports only, or private external storage approved for the project.

## Environment

The model card demonstrates Transformers loading with `trust_remote_code=True` and identifies a tri-hybrid 24-layer architecture. The upstream code is pinned to an immutable Hugging Face revision because enabling remote code executes Python from that repository. The project does not assume Llama-style names; the module inventory is generated from the actual model class.

Install on a CUDA training host:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
# First install the PyTorch build matching your CUDA driver/runtime.
python -m pip install -r projects/lfm3-cyber/requirements.txt
```

The GitHub Actions CPU job attempts a bounded one-step LoRA smoke run and verifies that an adapter can be saved. This establishes technical feasibility only; it does not constitute a useful cybersecurity training run. CPU memory and runtime limits may still cause the attempt to fail.

## Confirmed LFM3 module inventory

The pinned architecture was instantiated on meta tensors in GitHub Actions; the 3.46 GB checkpoint weights were not downloaded. The inspection found 362 modules and 154 `torch.nn.Linear` modules across a 24-layer `Lfm3ForCausalLM`.

Linear suffix counts:

| Suffix | Count | Role / first-pass decision |
|---|---:|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` | 11 each | Attention projections; selected |
| `w1`, `w2`, `w3` | 24 each | Feed-forward projections; selected |
| `gate_proj` | 11 | Attention head gate; reserved for later ablation |
| `in_proj`, `out_proj` | 13 each | Gated short-convolution projections; reserved for later ablation |
| `lm_head` | 1 | Output head; excluded |

The first baseline targets `q_proj,k_proj,v_proj,o_proj,w1,w2,w3`. This is an informed starting configuration, not a proven optimum.

## Workflow

1. Validate the full Pilot directory (all JSONL shards) and check scenario-level split isolation:
   `python projects/lfm3-cyber/scripts/validate_dataset.py projects/lfm3-cyber/dataset/pilot`
2. Scan duplicates and generate audit reports:
   `python projects/lfm3-cyber/scripts/deduplicate_dataset.py projects/lfm3-cyber/dataset/pilot`
   `python projects/lfm3-cyber/scripts/audit_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/dataset/reports`
3. Inspect the architecture without downloading the 3.46 GB checkpoint:
   `python projects/lfm3-cyber/scripts/inspect_model.py --mode config --output projects/lfm3-cyber/reports/module-inventory.json`
4. Review the initial target suffixes in `configs/sft.yaml`; do not blindly add all linear modules.
5. Run adapter-injection smoke test:
   `python projects/lfm3-cyber/scripts/smoke_test_lora.py --inventory projects/lfm3-cyber/reports/module-inventory.json --target-modules q_proj,k_proj,v_proj,o_proj,w1,w2,w3`
6. Prepare split files:
   `python projects/lfm3-cyber/scripts/prepare_dataset.py projects/lfm3-cyber/dataset/pilot --output-dir projects/lfm3-cyber/data/prepared`
7. Start SFT only on a reviewed dataset and a suitable GPU:
   `python projects/lfm3-cyber/scripts/train_sft.py --train projects/lfm3-cyber/data/prepared/train.jsonl --validation projects/lfm3-cyber/data/prepared/validation.jsonl --inventory projects/lfm3-cyber/reports/module-inventory.json --target-modules q_proj,k_proj,v_proj,o_proj,w1,w2,w3`

The config-only inspection constructs the real model architecture on meta tensors: it fetches the pinned configuration and source code but not model weights. The PEFT injection smoke test likewise does not perform a forward/backward pass. A full model load, training step, save and reload still require a real GPU run.

## Language and evaluation caveat

The official LFM3-1.7B model card lists 16 supported languages and does not list Hebrew. Hebrew quality must not be assumed. Report Hebrew separately from English, and compare the base and adapted model on the same held-out prompts.

## Dataset contract

The canonical schema is `dataset/schema.json`. Every row includes provenance, source/license, authorization, difficulty, expected skills, verification and quality metadata in addition to conversational `messages`. All shards in the Pilot directory are loaded together; scenario IDs and near-duplicate prompts may not cross train/validation/test splits. All 500 examples remain pending independent expert review.

The dataset is predominantly defensive. The six authorized-lab examples retain explicit isolated-lab boundaries. Do not include real credentials, personal data, unredacted logs, or operational instructions for unauthorized compromise, persistence, credential theft, evasion, or exfiltration. For dual-use material, preserve authorization context and prefer detection, explanation, containment, remediation and bounded lab validation.

## Initial training defaults (provisional)

See `configs/sft.yaml`. Rank 16 / alpha 32 / dropout 0.05 are experiment starting points, not validated LFM3 settings. The first target-module set was selected from the real inventory. Start with short sequences and a small pilot; increase only after memory and quality checks.

## Acceptance gates

- dataset validation passes; scenario groups do not cross splits;
- source and license recorded for every non-synthetic source;
- module inventory saved and target modules explicitly reviewed;
- PEFT injects successfully and only adapter parameters are trainable;
- one tiny GPU smoke run completes and saves/reloads the adapter;
- held-out evaluation reports evidence-grounded accuracy, false-positive rate, safe refusal/authorization judgment, Hebrew-vs-English performance, and regression against the unadapted base;
- no claim of cybersecurity expertise without benchmark results.

## References

- Model card: https://huggingface.co/LiquidAI/LFM3-1.7B
- Pinned upstream revision: https://huggingface.co/LiquidAI/LFM3-1.7B/commit/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd
- Liquid AI LFM3 overview: https://www.liquid.ai/blog/lfm3
- Transformers PEFT integration: https://huggingface.co/docs/transformers/peft
- TRL PEFT integration: https://huggingface.co/docs/trl/peft_integration
