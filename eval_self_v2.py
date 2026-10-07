from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from config import ModelConfig
from eval_v15 import char_eval, generate, PROMPTS
from model import PersonalNativeLM
from model_self_variants import SelfVariantPersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer


def load_self_v2(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    base = PersonalNativeLM(cfg)
    model = SelfVariantPersonalNativeLM(
        base,
        variant=ckpt["variant"],
        rank=int(ckpt.get("self_rank", 8)),
        projection_rank=int(ckpt.get("projection_rank", 4)),
    )
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    return ckpt, cfg, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fineweb-val", required=True)
    ap.add_argument("--chars", type=int, default=50000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_self_v2(args.checkpoint, args.device)
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    text = Path(args.fineweb_val).read_text(encoding="utf-8")[:args.chars]

    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(args.device)
    examples = []
    for prompt in PROMPTS:
        examples.append({
            "input": prompt,
            "greedy": generate(model, cfg, tok, prompt, neutral, 0.0),
            "sampled": generate(model, cfg, tok, prompt, neutral, 0.75),
        })

    result = {
        "version": "0.16",
        "variant": ckpt["variant"],
        "tokens_seen": ckpt["tokens_seen"],
        "steps": ckpt["steps"],
        "parameters": ckpt["parameter_count"],
        "base_parameters": ckpt["base_parameter_count"],
        "extra_parameters": ckpt["parameter_count"] - ckpt["base_parameter_count"],
        "initial_base_sha256": ckpt["initial_base_sha256"],
        "anchor_lr_scale": ckpt.get("anchor_lr_scale", 1.0),
        "tokenizer_kind": tokenizer_kind,
        "fineweb2": char_eval(model, tok, text, args.device),
        "language_examples": examples,
        "replacement_chars_total": sum(
            e["greedy"]["replacement_chars"] + e["sampled"]["replacement_chars"]
            for e in examples
        ),
        "self_diagnostics": ckpt.get("self_diagnostics"),
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
