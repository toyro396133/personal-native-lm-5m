# SELF personal micro-adapter on a proven English LM

An **isolated experimental pilot**, not a replacement for the existing Hebrew/SELF runs.

**Frozen base:** [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct), Apache-2.0. It is a compact English instruction model with published benchmarks, not a frontier-level reasoner.

**Personal model:** separate per-user low-rank residual weights injected through a forward hook inside the frozen Transformer's selected decoder layer. With hidden=960 and rank=4 the adapter has 7,680 trainable weights. The base parameters are not updated; gradients can pass through downstream frozen layers to train the personal adapter. The default hook is on the last decoder layer to minimize CPU training cost. Use `--layer-from-end 3` to test deeper effects, at a substantially higher memory/time cost.

## Local setup (Windows PowerShell; Python 3.10+)

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch>=2.2" "transformers>=4.44,<5" "safetensors>=0.4"
.\.venv\Scripts\python.exe experiments\personal_smollm2\run.py train --data experiments\personal_smollm2\example.jsonl --user demo --steps 20
.\.venv\Scripts\python.exe experiments\personal_smollm2\run.py chat --user demo --prompt "Help me organize my work."
```

First run downloads the base model from Hugging Face; that step requires access to the model and internet. CPU training may be slow on a 2-core machine. Use `--steps 2` for an initial smoke test.

## Real personal training data

Supply a UTF-8 JSONL file with `{"prompt": "...", "response": "..."}` on each line. These are **preferred target responses**, not raw historical assistant outputs. Curate or explicitly approve examples; don't blindly train on every chat turn because assistant errors and sensitive data can be memorized. The example dataset is fabricated, in English, and contains no private data.

The toy training loop rotates through examples and updates only one user adapter. This is *not yet* a background online-learning service; it is the smallest end-to-end prototype of the core mechanics. Production would require user-controlled data ingestion, opt-in, isolated storage, deletion, filtering, validated feedback, non-regression evaluation, checkpoint promotion, and safe scheduling.

## Research validity

- Compare **base only**, **retrieval memory baseline**, and **adapter with the same memory context** on held-out user-specific prompts.
- Evaluate general factual performance before and after personalization for negative transfer, plus performance on contradictory/stale preferences.
- Measure memory/latency for the final-layer and deeper-layer adapter; deep hooking through a frozen model is **not free** because backward still processes downstream operations.
- The personal adapter does not acquire all of the big model's capabilities. It is a small learned controller over computations already available in the base.
- Never commit data or checkpoints from real users.

## Current verification

Code committed for reproducible pilot, but **not yet run on downloaded model weights here**; no benchmark or learning gain is claimed.
