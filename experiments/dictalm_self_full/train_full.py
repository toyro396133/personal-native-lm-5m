from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Adafactor,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    set_seed,
)


class TokenBlocks(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.tokens = np.memmap(path, dtype=np.int32, mode="r")
        self.width = seq_len + 1
        self.n = len(self.tokens) // self.width

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        start = i * self.width
        arr = np.asarray(self.tokens[start:start + self.width], dtype=np.int64)
        return torch.from_numpy(arr.copy())


class DiffAnchorSelfAdapter(nn.Module):
    """Same conceptual SELF form that leads v0.17 at 50M: [a, x-a]."""

    def __init__(self, hidden_size: int, rank: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.down = nn.Linear(2 * hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        self.gate_logit = nn.Parameter(torch.tensor(-2.0))
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, x, self_anchor):
        xn = self.norm(x)
        an = F.layer_norm(self_anchor, (self_anchor.shape[-1],))
        a = an.view(1, 1, -1).expand(xn.shape[0], xn.shape[1], -1)
        rel = torch.cat((a, xn - a), dim=-1)
        delta = self.up(F.gelu(self.down(rel)))
        return x + torch.sigmoid(self.gate_logit) * delta


def decoder_layers(model):
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "base_model", None), "layers", None),
    ]
    for layers in candidates:
        if layers is not None and isinstance(layers, nn.ModuleList):
            return layers
    raise RuntimeError(
        "Could not find decoder layers. Expected a Qwen-style model.model.layers ModuleList."
    )


def hidden_size_of(model):
    for name in ("hidden_size", "d_model", "n_embd"):
        value = getattr(model.config, name, None)
        if value:
            return int(value)
    raise RuntimeError("Could not determine hidden size from model config")


def inject_diff_anchor_self(model, rank=8):
    hidden = hidden_size_of(model)
    layers = decoder_layers(model)

    model.self_anchor = nn.Parameter(torch.empty(hidden))
    nn.init.normal_(model.self_anchor, mean=0.0, std=0.02)
    model.self_adapters = nn.ModuleList(
        [DiffAnchorSelfAdapter(hidden, rank=rank) for _ in layers]
    )
    model._self_enabled = True
    model._self_hook_handles = []

    for i, layer in enumerate(layers):
        adapter = model.self_adapters[i]

        def make_hook(adapter_ref):
            def hook(_module, _inputs, output):
                if not getattr(model, "_self_enabled", True):
                    return output
                if torch.is_tensor(output):
                    return adapter_ref(output, model.self_anchor)
                if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
                    return (adapter_ref(output[0], model.self_anchor), *output[1:])
                raise TypeError(f"Unsupported decoder-layer output type: {type(output)!r}")
            return hook

        model._self_hook_handles.append(
            layer.register_forward_hook(make_hook(adapter))
        )

    return {
        "hidden_size": hidden,
        "layers": len(layers),
        "rank": rank,
        "self_parameters": (
            model.self_anchor.numel()
            + sum(p.numel() for p in model.self_adapters.parameters())
        ),
    }


def make_loader(path, seq_len, batch_size, seed, shuffle):
    ds = TokenBlocks(path, seq_len)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )


@torch.no_grad()
def evaluate(model, loader, accelerator, self_enabled=True, max_batches=0):
    unwrapped = accelerator.unwrap_model(model)
    previous = getattr(unwrapped, "_self_enabled", None)
    if previous is not None:
        unwrapped._self_enabled = self_enabled

    model.eval()
    local_nll = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    local_tokens = torch.zeros((), device=accelerator.device, dtype=torch.float64)

    for i, ids in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        ids = ids.to(accelerator.device, non_blocking=True)
        out = model(input_ids=ids, labels=ids, use_cache=False)
        target_tokens = ids.shape[0] * (ids.shape[1] - 1)
        local_nll += out.loss.detach().double() * target_tokens
        local_tokens += target_tokens

    nll = accelerator.reduce(local_nll, reduction="sum")
    count = accelerator.reduce(local_tokens, reduction="sum")
    mean = (nll / count.clamp_min(1)).item()
    model.train()

    if previous is not None:
        unwrapped._self_enabled = previous

    return {
        "nats_per_token": mean,
        "perplexity": math.exp(min(mean, 20.0)),
        "evaluated_tokens": int(count.item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["baseline", "self"], required=True)
    ap.add_argument("--model", default="dicta-il/DictaLM-3.0-1.7B-Base")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--target-tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--self-rank", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="fp16",
    )
    set_seed(args.seed)
    random.seed(args.seed)

    started = time.time()
    if accelerator.is_main_process:
        print(
            f"arm={args.arm} model={args.model} processes={accelerator.num_processes} "
            f"device={accelerator.device} target_tokens={args.target_tokens:,}",
            flush=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    self_info = None
    if args.arm == "self":
        self_info = inject_diff_anchor_self(model, rank=args.self_rank)

    # This is intentionally FULL continued training: every pretrained parameter
    # remains trainable, together with SELF in the experimental arm.
    for p in model.parameters():
        p.requires_grad_(True)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = Adafactor(
        model.parameters(),
        lr=args.lr,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
        weight_decay=args.weight_decay,
    )

    train_loader = make_loader(
        args.train, args.seq_len, args.micro_batch, args.seed, shuffle=True
    )
    val_loader = make_loader(
        args.val, args.seq_len, args.micro_batch, args.seed, shuffle=False
    )

    tokens_per_microstep_global = (
        args.micro_batch * args.seq_len * accelerator.num_processes
    )
    estimated_microsteps = math.ceil(args.target_tokens / tokens_per_microstep_global)
    estimated_optimizer_steps = math.ceil(estimated_microsteps / args.grad_accum)
    warmup_steps = max(1, int(round(estimated_optimizer_steps * 0.05)))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(estimated_optimizer_steps, warmup_steps + 1),
    )

    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    initial_eval = evaluate(model, val_loader, accelerator, self_enabled=True)

    model.train()
    seen = 0
    microstep = 0
    optimizer_steps = 0
    first_loss = None
    last_loss = None

    epoch = 0
    while seen < args.target_tokens:
        epoch += 1
        for ids in train_loader:
            ids = ids.to(accelerator.device, non_blocking=True)

            with accelerator.accumulate(model):
                out = model(input_ids=ids, labels=ids, use_cache=False)
                loss = out.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            microstep += 1
            if accelerator.sync_gradients:
                optimizer_steps += 1

            step_tokens = ids.shape[0] * (ids.shape[1] - 1) * accelerator.num_processes
            seen += int(step_tokens)
            last_loss = float(loss.detach().float().item())
            if first_loss is None:
                first_loss = last_loss

            if accelerator.is_main_process and (
                microstep == 1 or microstep % 50 == 0 or seen >= args.target_tokens
            ):
                print(
                    f"arm={args.arm} epoch={epoch} microstep={microstep} "
                    f"optimizer_steps={optimizer_steps} tokens={seen:,} "
                    f"loss={last_loss:.4f}",
                    flush=True,
                )

            if seen >= args.target_tokens:
                break

    accelerator.wait_for_everyone()
    final_eval = evaluate(model, val_loader, accelerator, self_enabled=True)

    self_off_eval = None
    if args.arm == "self":
        self_off_eval = evaluate(
            model, val_loader, accelerator, self_enabled=False
        )

    elapsed = time.time() - started
    report = {
        "experiment": "DictaLM 1.7B continued FULL training + SELF canary",
        "arm": args.arm,
        "model": args.model,
        "all_pretrained_weights_trainable": True,
        "self_architecture": "diff_anchor" if args.arm == "self" else None,
        "self_info": self_info,
        "parameters": total_params,
        "trainable_parameters": trainable_params,
        "target_tokens": args.target_tokens,
        "tokens_seen": seen,
        "microsteps": microstep,
        "optimizer_steps": optimizer_steps,
        "seq_len": args.seq_len,
        "micro_batch": args.micro_batch,
        "gradient_accumulation": args.grad_accum,
        "world_size": accelerator.num_processes,
        "optimizer": "Adafactor",
        "lr": args.lr,
        "seed": args.seed,
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "initial_validation": initial_eval,
        "final_validation": final_eval,
        "self_adapter_off_validation": self_off_eval,
        "elapsed_seconds": elapsed,
    }

    if accelerator.is_main_process:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
