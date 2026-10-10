"""Isolated English pilot: frozen SmolLM2 + per-user trainable residual adapter.

Usage:
  python experiments/personal_smollm2/run.py train --data experiments/personal_smollm2/example.jsonl --user demo
  python experiments/personal_smollm2/run.py chat --user demo --prompt "Give me a study plan."
No training data is uploaded by this program; Hugging Face model download is separate.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM2-360M-Instruct"
ROOT = Path(__file__).resolve().parent
STORAGE = ROOT / "local_users"


class PersonalAdapter(nn.Module):
    """Zero-initialized low-rank residual, ~2*hidden*rank parameters."""
    def __init__(self, hidden: int, rank: int = 4):
        super().__init__()
        self.down = nn.Linear(hidden, rank, bias=False)
        self.up = nn.Linear(rank, hidden, bias=False)
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.down(x))


def load_backbone():
    tokenizer = AutoTokenizer.from_pretrained(BASE)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float32)
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model


def attach(model, adapter, layer_from_end):
    layers = model.model.layers
    idx = len(layers) - layer_from_end
    if idx < 0 or idx >= len(layers):
        raise ValueError("Invalid --layer-from-end")
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            return (adapter(output[0]), *output[1:])
        return adapter(output)
    return layers[idx].register_forward_hook(hook)


def encode_pair(tokenizer, prompt, response, max_length):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    answer = tokenizer.encode(response, add_special_tokens=False) + [tokenizer.eos_token_id]
    available = max_length - len(answer)
    if available < 1:
        raise ValueError("Response longer than max_length")
    prefix = prefix[-available:]
    ids = prefix + answer
    labels = [-100] * len(prefix) + answer
    return torch.tensor([ids]), torch.tensor([labels])


def user_path(user):
    if not user or not all(c.isascii() and (c.isalnum() or c in "-_") for c in user):
        raise ValueError("User ID must contain only ASCII letters, digits, '_' or '-'")
    return STORAGE / user


def train(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer, model = load_backbone()
    examples = [json.loads(line) for line in Path(args.data).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not examples or any(not isinstance(ex.get("prompt"), str) or not isinstance(ex.get("response"), str) for ex in examples):
        raise ValueError("JSONL must contain nonempty prompt/response strings")
    hidden = model.config.hidden_size
    adapter = PersonalAdapter(hidden, args.rank)
    handle = attach(model, adapter, args.layer_from_end)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr)
    print(f"Base: {BASE}; personal trainable parameters: {sum(p.numel() for p in adapter.parameters()):,}; frozen base: {sum(p.numel() for p in model.parameters()):,}")
    try:
        for step in range(args.steps):
            ex = examples[step % len(examples)]
            ids, labels = encode_pair(tokenizer, ex["prompt"], ex["response"], args.max_length)
            optimizer.zero_grad(set_to_none=True)
            loss = model(input_ids=ids, labels=labels).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            print(f"step={step+1}/{args.steps} loss={loss.item():.4f}", flush=True)
    finally:
        handle.remove()
    target = user_path(args.user)
    target.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in adapter.state_dict().items()}, str(target / "adapter.safetensors"))
    (target / "config.json").write_text(json.dumps({"base": BASE, "rank": args.rank, "hidden": hidden, "layer_from_end": args.layer_from_end}, indent=2), encoding="utf-8")
    print(f"Saved personal adapter in {target}")


@torch.no_grad()
def chat(args):
    tokenizer, model = load_backbone()
    target = user_path(args.user)
    config = json.loads((target / "config.json").read_text(encoding="utf-8"))
    if config["base"] != BASE or config["hidden"] != model.config.hidden_size:
        raise ValueError("Incompatible adapter")
    adapter = PersonalAdapter(config["hidden"], config["rank"])
    adapter.load_state_dict(load_file(str(target / "adapter.safetensors")))
    adapter.eval()
    handle = attach(model, adapter, config["layer_from_end"])
    try:
        ids = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], return_tensors="pt", add_generation_prompt=True)
        generated = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        print(tokenizer.decode(generated[0, ids.shape[-1]:], skip_special_tokens=True))
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--user", required=True)
    t.add_argument("--steps", type=int, default=20)
    t.add_argument("--lr", type=float, default=0.001)
    t.add_argument("--rank", type=int, default=4)
    t.add_argument("--layer-from-end", type=int, default=1, help="1=last layer; 3=third from end (costlier)")
    t.add_argument("--max-length", type=int, default=128)
    t.add_argument("--seed", type=int, default=7)
    c = sub.add_parser("chat")
    c.add_argument("--user", required=True)
    c.add_argument("--prompt", required=True)
    c.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()
    if args.command == "train":
        if args.steps < 1 or args.rank < 1 or args.max_length < 8:
            parser.error("steps/rank/max-length must be positive")
        train(args)
    else:
        chat(args)


if __name__ == "__main__":
    main()
