# LFM3-Cyber — isolated training workspace

This project specializes `LiquidAI/LFM3-1.7B` for general cybersecurity work, with both defensive analysis and authorized lab assessment. It is deliberately isolated from the repository's SELF and COMPASS research lines.

## Current stage

Infrastructure scaffold only. No full training run has been started and no paid compute is used.

- Base checkpoint: `LiquidAI/LFM3-1.7B` (post-trained conversational/tool model).
- Pinned upstream revision: `40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd`.
- First adapter: `lfm3-cyber-lora-v1`.
- Method: supervised fine-tuning (SFT) with one LoRA adapter.
- Initial pilot target: 500 reviewed examples; the checked-in pilot is only a tiny synthetic schema fixture, not a training corpus.
- Retrieval (RAG) supplies changing references; LoRA should teach domain behavior and structured task performance, not memorize changing threat intelligence.

## Repository boundaries

All project code, configs, fixtures and workflow files live under `projects/lfm3-cyber/`, except the dedicated workflow at `.github/workflows/lfm3-cyber-ci.yml`. Existing SELF, COMPASS, Chacham Lev, model-mirroring and research workflows are not changed.

Training outputs, model weights, real logs, captures, credentials and private datasets must never be committed. Use GitHub Actions artifacts for short-lived reports only, or private external storage approved for the project.

## Environment

The model card demonstrates Transformers loading with `trust_remote_code=True` and identifies a tri-hybrid 24-layer architecture. The upstream code is pinned to an immutable Hugging Face revision because enabling remote code executes Python from that repository. The project does not assume Llama-style names such as `q_proj` / `v_proj`.

Install on a CUDA training host:

```bash
python -m venv .venv
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
# First install the PyTorch build matching your CUDA driver/runtime.
python -m pip install -r projects/lfm3-cyber/requirements.txt
```

Do not assume the CPU GitHub-hosted runner is a viable training host.

## Workflow

1. Validate a JSONL dataset and check scenario-level split isolation:
   `python projects/lfm3-cyber/scripts/validate_dataset.py projects/lfm3-cyber/data/pilot/synthetic.jsonl`
2. Inspect the architecture without downloading the 3.5 GB checkpoint:
   `python projects/lfm3-cyber/scripts/inspect_model.py --mode config --output projects/lfm3-cyber/reports/module-inventory.json`
3. Review the inventory and choose exact target module suffixes. Do not blindly LoRA every linear layer.
4. Run adapter-injection smoke test:
   `python projects/lfm3-cyber/scripts/smoke_test_lora.py --inventory projects/lfm3-cyber/reports/module-inventory.json --target-modules <reviewed,module,suffixes>`
5. Prepare split files:
   `python projects/lfm3-cyber/scripts/prepare_dataset.py projects/lfm3-cyber/data/pilot/synthetic.jsonl --output-dir projects/lfm3-cyber/data/prepared`
6. Start SFT only on a reviewed dataset and a suitable GPU:
   `python projects/lfm3-cyber/scripts/train_sft.py --train projects/lfm3-cyber/data/prepared/train.jsonl --validation projects/lfm3-cyber/data/prepared/validation.jsonl --target-modules <reviewed,module,suffixes>`

The config-only inspection constructs the real model architecture on meta tensors: it fetches the pinned configuration and source code but not model weights. The PEFT injection smoke test likewise does not perform a forward/backward pass. A full model load, training step, save and reload still require a real GPU run.

## Dataset contract

Each JSONL row contains `id`, `scenario_id`, `split`, `domain`, `task`, `source`, `authorization` and conversational `messages`. Scenario IDs may not cross train/validation/test splits.

The pilot examples are synthetic and defensive. Future authorized-lab examples must document the lab boundary and permitted target. Do not include real credentials, personal data, unredacted logs, or operational instructions for unauthorized compromise, persistence, credential theft, evasion, or exfiltration. For dual-use material, preserve authorization context and prefer detection, explanation, containment, remediation and bounded lab validation.

## Initial training defaults (provisional)

See `configs/sft.yaml`. Rank 16 / alpha 32 / dropout 0.05 are experiment starting points, not validated LFM3 settings. Target modules remain unset until inspection. Start with short sequences and a small pilot; increase only after memory and quality checks.

## Acceptance gates

- dataset validation passes; scenario groups do not cross splits;
- source and license recorded for every non-synthetic source;
- module inventory saved and target modules explicitly reviewed;
- PEFT injects successfully and only adapter parameters are trainable;
- one tiny GPU smoke run completes and saves/reloads the adapter;
- held-out evaluation reports evidence-grounded accuracy, false-positive rate, safe refusal/authorization judgment, and regression against the unadapted base;
- no claim of cybersecurity expertise without benchmark results.

## References

- Model card: https://huggingface.co/LiquidAI/LFM3-1.7B
- Pinned upstream revision: https://huggingface.co/LiquidAI/LFM3-1.7B/commit/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd
- Liquid AI LFM3 overview: https://www.liquid.ai/blog/lfm3
- Transformers PEFT integration: https://huggingface.co/docs/transformers/peft
- TRL PEFT integration: https://huggingface.co/docs/trl/peft_integration
