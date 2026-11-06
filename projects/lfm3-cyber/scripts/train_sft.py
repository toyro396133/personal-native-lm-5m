#!/usr/bin/env python3
"""Small, genuine CPU/GPU SFT runner for the pinned LFM3 checkpoint."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--target-modules", required=True)
    parser.add_argument("--model-id", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--revision", default="40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd")
    parser.add_argument("--output-dir", default="projects/lfm3-cyber/model/adapters/lfm3-cyber-lora-v1")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive value overrides epochs; use 1 for a smoke-training run.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if not args.train.is_file() or not args.validation.is_file():
        raise SystemExit("Train and validation JSONL files must both exist.")
    if not args.inventory.is_file():
        raise SystemExit("Run inspect_model.py first and provide its module inventory.")
    targets = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    if not targets:
        raise SystemExit("Reviewed --target-modules are required; automatic target guessing is disabled.")
    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    available = {item["name"].split(".")[-1] for item in inventory.get("linear_modules", [])}
    unknown = sorted(set(targets) - available)
    if unknown:
        raise SystemExit(f"Target suffixes are absent from the inspected LFM3 inventory: {unknown}")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    has_cuda = torch.cuda.is_available()
    if args.device == "cuda" and not has_cuda:
        raise SystemExit("CUDA was requested but is unavailable.")
    if args.device == "cpu" and has_cuda:
        device = "cpu"
    else:
        device = "cuda" if has_cuda and args.device != "cpu" else "cpu"
    if device == "cpu":
        # CPU training is intentionally conservative: full-precision frozen base,
        # checkpointed activations, one-example batches and no mixed precision.
        dtype = torch.float32
        bf16 = fp16 = False
        print(f"CPU training selected; logical CPUs={torch.get_num_threads()}")
    else:
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        bf16, fp16 = dtype == torch.bfloat16, dtype == torch.float16

    dataset = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.validation)})
    if len(dataset["train"]) == 0:
        raise SystemExit("Training split is empty; refusing to start.")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=targets,
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1 if device == "cpu" else 8,
        warmup_steps=0,
        logging_steps=1 if args.max_steps > 0 else 10,
        save_strategy="no" if args.max_steps > 0 else "steps",
        save_steps=100,
        save_total_limit=2,
        eval_strategy="no" if args.max_steps > 0 else "steps",
        eval_steps=100,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=bf16,
        fp16=fp16,
        optim="adamw_torch",
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=(device == "cpu"),
    )
    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"TRAINING_COMPLETE device={device} train_examples={len(dataset['train'])} output={args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
