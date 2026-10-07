from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import psutil
import torch

from config import ModelConfig
from model import PersonalNativeLM
from model_self_variants import (
    SELF_VARIANTS,
    SelfVariantPersonalNativeLM,
    anchor_lr_scale_for_variant,
)
from personal_state import PersonalState
from train_hebrew import load_tokenizer


def now():
    return time.strftime("%H:%M:%S", time.gmtime())


def log(msg):
    print(f"[{now()}] {msg}", flush=True)


def fingerprint(model):
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(str(t.dtype).encode())
            h.update(bytes(t.untyped_storage()))
    return h.hexdigest()


def resources():
    p = psutil.Process(os.getpid())
    vm = psutil.virtual_memory()
    return {
        "rss_mb": round(p.memory_info().rss / 1024**2, 1),
        "system_ram_percent": vm.percent,
        "cpu_percent": psutil.cpu_percent(interval=None),
    }


def make_model(cfg, variant, self_rank=8, projection_rank=4):
    base = PersonalNativeLM(cfg)
    model = SelfVariantPersonalNativeLM(
        base,
        variant=variant,
        rank=self_rank,
        projection_rank=projection_rank,
    )
    return base, model


def make_optimizer(model, variant, lr):
    scale = anchor_lr_scale_for_variant(variant)
    if scale == 1.0:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

    anchor_id = id(model.self_anchor)
    main = [p for p in model.parameters() if id(p) != anchor_id]
    return torch.optim.AdamW(
        [
            {"params": main, "lr": lr, "weight_decay": 0.1},
            {"params": [model.self_anchor], "lr": lr * scale, "weight_decay": 0.1},
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--variant", choices=sorted(SELF_VARIANTS), required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--resume")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--self-rank", type=int, default=8)
    ap.add_argument("--projection-rank", type=int, default=4)
    ap.add_argument("--save", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok, kind = load_tokenizer(args.tokenizer)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("variant") != args.variant:
            raise ValueError(f"resume variant mismatch: {ckpt.get('variant')} != {args.variant}")
        cfg = ModelConfig(**ckpt["config"])
        self_rank = int(ckpt.get("self_rank", args.self_rank))
        projection_rank = int(ckpt.get("projection_rank", args.projection_rank))
        base, model = make_model(cfg, args.variant, self_rank, projection_rank)
        model.load_state_dict(ckpt["model"], strict=True)
        base_sha = ckpt["initial_base_sha256"]
        base_params = int(ckpt["base_parameter_count"])
        step = int(ckpt["steps"])
        seen = int(ckpt["tokens_seen"])
        first_loss = ckpt.get("first_loss")
        last_loss = ckpt.get("last_loss")
        epoch = int(ckpt.get("data_epoch", 1))
        order_pos = int(ckpt.get("data_order_pos", 0))
        prior_elapsed = float(ckpt.get("total_elapsed_seconds", 0.0))
    else:
        cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
        base, model = make_model(cfg, args.variant, args.self_rank, args.projection_rank)
        base_sha = fingerprint(base)
        base_params = sum(p.numel() for p in base.parameters())
        step = 0
        seen = 0
        first_loss = None
        last_loss = None
        epoch = 1
        order_pos = 0
        prior_elapsed = 0.0

    model = model.to(args.device)
    params = sum(p.numel() for p in model.parameters())
    extra = params - base_params
    log(
        f"variant={args.variant} parameters={params:,} extra={extra:,} "
        f"anchor_lr_scale={anchor_lr_scale_for_variant(args.variant)} "
        f"base_sha256={base_sha}"
    )

    tokens = torch.load(args.tokens, map_location="cpu", weights_only=True)
    width = args.seq_len + 1
    usable = tokens.numel() - tokens.numel() % width
    blocks = tokens[:usable].view(-1, width)
    n_blocks = blocks.shape[0]

    neutral = PersonalState("neutral").flatten().to(args.device)
    opt = make_optimizer(model, args.variant, args.lr)
    if args.resume:
        opt.load_state_dict(ckpt["optimizer"])
        for state in opt.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(args.device)

    start_seen = seen
    start_step = step
    last_heartbeat = time.time()
    model.train()

    while seen < args.target_tokens:
        g = torch.Generator(device="cpu")
        g.manual_seed(args.seed + 1000 * epoch)
        order = torch.randperm(n_blocks, generator=g)

        while order_pos < n_blocks and seen < args.target_tokens:
            idx = order[order_pos:order_pos + args.batch_size]
            order_pos += args.batch_size
            if idx.numel() < 2:
                continue

            batch = blocks[idx]
            x = batch[:, :-1].to(args.device, dtype=torch.long)
            y = batch[:, 1:].to(args.device, dtype=torch.long)
            state = neutral.unsqueeze(0).expand(x.size(0), -1)

            _, loss = model(x, state, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            seen += int(y.numel())
            last_loss = float(loss.detach())
            if first_loss is None:
                first_loss = last_loss

            if step == 1 or step % 100 == 0 or time.time() - last_heartbeat >= 60:
                elapsed = time.time() - started
                log(
                    f"variant={args.variant} step={step} tokens_seen={seen:,} "
                    f"loss={last_loss:.4f} chunk_min={elapsed/60:.1f} "
                    f"resources={resources()}"
                )
                last_heartbeat = time.time()

        if order_pos >= n_blocks:
            epoch += 1
            order_pos = 0

    chunk_elapsed = time.time() - started
    total_elapsed = prior_elapsed + chunk_elapsed

    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "tokenizer_kind": kind,
        "model_profile": "5m",
        "variant": args.variant,
        "self_rank": args.self_rank,
        "projection_rank": args.projection_rank,
        "anchor_lr_scale": anchor_lr_scale_for_variant(args.variant),
        "parameter_count": params,
        "base_parameter_count": base_params,
        "initial_base_sha256": base_sha,
        "stage": "v0.16-self-v2-screen",
        "steps": step,
        "tokens_seen": seen,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "data_epoch": epoch,
        "data_order_pos": order_pos,
        "seed": args.seed,
        "total_elapsed_seconds": total_elapsed,
        "self_diagnostics": model.self_diagnostics(),
    }
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.save)

    report = {
        "variant": args.variant,
        "target_tokens": args.target_tokens,
        "start_tokens": start_seen,
        "tokens_seen": seen,
        "chunk_steps": step - start_step,
        "steps": step,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "chunk_elapsed_seconds": chunk_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "base_sha256": base_sha,
        "base_parameters": base_params,
        "parameters": params,
        "extra_parameters": extra,
        "anchor_lr_scale": anchor_lr_scale_for_variant(args.variant),
        "self_diagnostics": model.self_diagnostics(),
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(f"saved={args.save} tokens={seen:,} loss={last_loss:.4f}")


if __name__ == "__main__":
    main()
