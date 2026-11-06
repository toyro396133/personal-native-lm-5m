#!/usr/bin/env python3
"""SFT entry point. Requires a CUDA-capable training host and reviewed targets."""
from __future__ import annotations
import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--target-modules", required=True, help="Reviewed comma-separated module suffixes")
    parser.add_argument("--model-id", default="LiquidAI/LFM3-1.7B")
    parser.add_argument("--output-dir", default="projects/lfm3-cyber/model/adapters/lfm3-cyber-lora-v1")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    args = parser.parse_args()
    if not args.train.is_file() or not args.validation.is_file():
        raise SystemExit("Train and validation JSONL files must both exist.")
    targets = [value.strip() for value in args.target_modules.split(",") if value.strip()]
    if not targets:
        raise SystemExit("Reviewed --target-modules are required; automatic target guessing is disabled.")

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise SystemExit("Full SFT is disabled on CPU in this scaffold. Use a suitable CUDA host.")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("This initial config expects BF16 support. Select/validate an FP16 config explicitly for other GPUs.")

    dataset = load_dataset("json", data_files={"train": str(args.train), "validation": str(args.validation)})
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
    )
    training_args = SFTConfig(
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        gradient_checkpointing=True,
        bf16=True,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": False},
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved adapter to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
