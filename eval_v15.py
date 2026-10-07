from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import PersonalNativeLM
from model_self_anchor import SelfAnchoredPersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer


PROMPTS = [
    "ישראל היא מדינה",
    "המחשב יכול",
    "בשנים האחרונות",
    "המחקר מראה כי",
    "כאשר אנשים לומדים מיומנות חדשה",
]


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    base = PersonalNativeLM(cfg)
    variant = ckpt.get("variant", "baseline")
    if variant == "self":
        model = SelfAnchoredPersonalNativeLM(base, rank=int(ckpt.get("self_rank") or 8))
    else:
        model = base
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device)
    model.eval()
    return ckpt, cfg, model


@torch.no_grad()
def char_eval(model, tok, text: str, device: str, seq_len: int = 128):
    ids = tok.encode(text, bos=True, eos=True)
    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(device)

    total_nll = 0.0
    target_tokens = 0
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits, _ = model(x, neutral)
        total_nll += float(
            F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                y.reshape(-1),
                reduction="sum",
            )
        )
        target_tokens += y.numel()

    chars = len(text)
    nats = total_nll / max(1, chars)
    return {
        "raw_chars": chars,
        "encoded_tokens": len(ids),
        "evaluated_target_tokens": target_tokens,
        "tokens_per_100_chars": len(ids) / max(1, chars) * 100,
        "total_nll": total_nll,
        "nats_per_char": nats,
        "bits_per_char": nats / math.log(2),
        "nats_per_utf8_byte": total_nll / max(1, len(text.encode("utf-8"))),
    }


@torch.no_grad()
def generate(model, cfg, tok, prompt: str, state, temperature: float, seed: int = 17):
    ids = tok.encode(prompt, bos=True, eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=state.device)
    gen = torch.Generator(device=state.device)
    gen.manual_seed(seed)
    generated = []

    for _ in range(48):
        logits, _ = model(x[:, -cfg.max_seq_len:], state)
        last = logits[0, -1]
        if temperature <= 0:
            nxt = int(last.argmax())
        else:
            probs = torch.softmax(last / temperature, dim=-1)
            nxt = int(torch.multinomial(probs, 1, generator=gen))
        generated.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=x.device)], dim=1)
        if nxt == tok.eos_id:
            break

    text = tok.decode(generated)
    adjacent_repeats = sum(a == b for a, b in zip(generated, generated[1:]))
    max_run = 0
    run = 0
    prev = None
    for token in generated:
        if token == prev:
            run += 1
        else:
            run = 1
            prev = token
        max_run = max(max_run, run)

    return {
        "text": text,
        "tokens": len(generated),
        "unique_token_ratio": len(set(generated)) / max(1, len(generated)),
        "adjacent_repeat_ratio": adjacent_repeats / max(1, len(generated) - 1),
        "max_identical_token_run": max_run,
        "replacement_chars": text.count("\ufffd"),
        "unk_tokens": sum(i == getattr(tok, "unk_id", -999999) for i in generated),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fineweb-val", required=True)
    ap.add_argument("--external-val", required=True)
    ap.add_argument("--chars", type=int, default=300000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_model(args.checkpoint, args.device)
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    fine_text = Path(args.fineweb_val).read_text(encoding="utf-8")[:args.chars]
    external_text = Path(args.external_val).read_text(encoding="utf-8")[:args.chars]

    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(args.device)
    examples = []
    for prompt in PROMPTS:
        examples.append({
            "input": prompt,
            "greedy": generate(model, cfg, tok, prompt, neutral, 0.0),
            "sampled": generate(model, cfg, tok, prompt, neutral, 0.75),
        })

    result = {
        "version": "0.15",
        "checkpoint": args.checkpoint,
        "variant": ckpt.get("variant", "baseline"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "steps": ckpt.get("steps"),
        "parameters": ckpt.get("parameter_count"),
        "base_parameters": ckpt.get("base_parameter_count"),
        "initial_base_sha256": ckpt.get("initial_base_sha256"),
        "tokenizer_kind": tokenizer_kind,
        "fineweb2": char_eval(model, tok, fine_text, args.device),
        "external": char_eval(model, tok, external_text, args.device),
        "language_examples": examples,
        "replacement_chars_total": sum(
            e["greedy"]["replacement_chars"] + e["sampled"]["replacement_chars"]
            for e in examples
        ),
    }
    if ckpt.get("variant") == "self":
        result["self_diagnostics"] = ckpt.get("self_diagnostics")

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
