#!/usr/bin/env python3
"""Inspect pinned LFM3 module structure without assuming Llama-style projection names."""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--revision", default="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd")
    parser.add_argument("--mode", choices=("config", "weights"), default="config")
    parser.add_argument("--output", type=Path, default=Path("module-inventory.json"))
    args = parser.parse_args()

    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    if args.mode == "config":
        from accelerate import init_empty_weights
        with init_empty_weights(include_buffers=True):
            model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_id,
            revision=args.revision,
            trust_remote_code=True,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
        )

    modules = []
    for name, module in model.named_modules():
        if not name:
            continue
        modules.append({
            "name": name,
            "class": module.__class__.__name__,
            "is_linear": isinstance(module, torch.nn.Linear),
            "in_features": getattr(module, "in_features", None),
            "out_features": getattr(module, "out_features", None),
        })
    linear_modules = [item for item in modules if item["is_linear"]]
    suffix_counts = Counter(item["name"].split(".")[-1] for item in linear_modules)
    report = {
        "model_id": args.model_id,
        "revision": args.revision,
        "mode": args.mode,
        "model_type": getattr(config, "model_type", None),
        "architectures": getattr(config, "architectures", None),
        "num_hidden_layers": getattr(config, "num_hidden_layers", None),
        "module_count": len(modules),
        "linear_module_count": len(linear_modules),
        "linear_suffix_counts": dict(sorted(suffix_counts.items())),
        "linear_modules": linear_modules,
        "all_modules": modules,
        "warning": "Candidate inventory only. Review target modules and run PEFT injection smoke test before training.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(modules)} modules ({len(linear_modules)} linear) to {args.output}")
    print("Linear suffix candidates:")
    for suffix, count in sorted(suffix_counts.items()):
        print(f"  {suffix}: {count}")
    if not linear_modules:
        raise SystemExit("No torch.nn.Linear modules found; inspect architecture before configuring LoRA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
