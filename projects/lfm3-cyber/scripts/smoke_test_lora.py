#!/usr/bin/env python3
"""Check PEFT injection into explicitly reviewed modules on the pinned LFM3 architecture."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--target-modules", required=True, help="Comma-separated exact module suffixes from the inventory")
    parser.add_argument("--model-id", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--revision", default="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    args = parser.parse_args()
    targets = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    if not targets:
        raise SystemExit("At least one reviewed target module is required.")

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    available = {item["name"].split(".")[-1] for item in inventory["linear_modules"]}
    unknown = sorted(set(targets) - available)
    if unknown:
        raise SystemExit(f"Targets are not linear module suffixes in inventory: {unknown}")

    from accelerate import init_empty_weights
    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    with init_empty_weights(include_buffers=True):
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    peft_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    wrapped = get_peft_model(model, peft_config)
    trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
    total = sum(p.numel() for p in wrapped.parameters())
    if trainable <= 0 or trainable >= total:
        raise SystemExit(f"Unexpected trainable parameter count: {trainable}/{total}")
    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "target_modules": targets,
        "rank": args.rank,
        "alpha": args.alpha,
        "total_parameters": total,
        "trainable_parameters": trainable,
        "trainable_percent": round(100 * trainable / total, 6),
        "status": "PEFT injection passed on meta-initialized architecture; no forward/backward pass was run.",
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
