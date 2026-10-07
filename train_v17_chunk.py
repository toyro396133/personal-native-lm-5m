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


ALL_VARIANTS = {"baseline", *SELF_VARIANTS}


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
    if variant == "baseline":
        return base, base
    model = SelfVariantPersonalNativeLM(
        base,
        variant=variant,
        rank=self_rank,
        projection_rank=projection_rank,
    )
    return base, model


def make_optimizer(model, variant, lr):
    if variant == "baseline":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

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


def load_tokens(path: str):
    p = Path(path)
    if p.suffix == ".i32":
        count = p.stat().st_size // 4
        tokens = torch.from_file(str(p), shared=False, size=count, dtype=torch.int32)
    else:
        tokens = torch.load(p, map_location="cpu", weights_only=True)
    if tokens.ndim != 1:
        raise ValueError(f"expected 1D token tensor, got {tuple(tokens.shape)}")
    return tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--variant", choices=sorted(ALL_VARIANTS), required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--resume")
    ap.add_argument("--reset-data-cursor", action="store_true")
    ap.add_argument("--data-seed-offset", type=int, default=0)
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
        self_rank = int(ckpt.get("self_rank", args.self_rank) or args.self_rank)
        projection_rank = int(ckpt.get("projection_rank", args.projection_rank) or args.projection_rank)
        base, model = make_model(cfg, args.variant, self_rank, projection_rank)
        model.load_state_dict(ckpt["model"], strict=True)
        base_sha = ckpt["initial_base_sha256"]
        base_params = int(ckpt["base_parameter_count"])
        step = int(ckpt["steps"])
        seen = int(ckpt["tokens_seen"])
        first_loss = ckpt.get("first_loss")
        last_loss = ckpt.get("last_loss")
        prior_elapsed = float(ckpt.get("total_elapsed_seconds", 0.0))

        if args.reset_data_cursor:
            epoch = 1
            order_pos = 0
        else:
            epoch = int(ckpt.get("data_epoch", 1))
            order_pos = int(ckpt.get("data_order_pos", 0))
            old_offset = int(ckpt.get("data_seed_offset", args.data_seed_offset))
            if old_offset != args.data_seed_offset:
                raise ValueError(
                    f"data seed offset changed without --reset-data-cursor: "
                    f"{old_offset} -> {args.data_seed_offset}"
                )
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
        prior_elapsed = 0.0
        epoch = 1
        order_pos = 0

    if seen >= args.target_tokens:
        raise ValueError(f"checkpoint already at {seen:,}; target={args.target_tokens:,}")

    model = model.to(args.device)
    params = sum(p.numel() for p in model.parameters())
    extra = params - base_params
    anchor_scale = 1.0 if args.variant == "baseline" else anchor_lr_scale_for_variant(args.variant)

    log(
        f"variant={args.variant} parameters={params:,} extra={extra:,} "
        f"anchor_lr_scale={anchor_scale} base_sha256={base_sha} "
        f"reset_data_cursor={args.reset_data_cursor} data_seed_offset={args.data_seed_offset}"
    )

    tokens = load_tokens(args.tokens)
    width = args.seq_len + 1
    usable = tokens.numel() - tokens.numel() % width
    blocks = tokens[:usable].view(-1, width)
    n_blocks = blocks.shape[0]
    if n_blocks < args.batch_size:
        raise RuntimeError("token shard too small")
    log(
        f"loaded shard={args.tokens} tokens={tokens.numel():,} usable={usable:,} "
        f"blocks={n_blocks:,} dtype={tokens.dtype} resources={resources()}"
    )

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
        g.manual_seed(args.seed + args.data_seed_offset + 1000 * epoch)
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
                chunk_seen = seen - start_seen
                log(
                    f"variant={args.variant} step={step} tokens_seen={seen:,} "
                    f"chunk_tokens={chunk_seen:,} loss={last_loss:.4f} "
                    f"chunk_tok_s={chunk_seen/max(elapsed,1):.1f} "
                    f"chunk_min={elapsed/60:.1f} resources={resources()}"
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
        "self_rank": args.self_rank if args.variant != "baseline" else None,
        "projection_rank": args.projection_rank if args.variant != "baseline" else None,
        "anchor_lr_scale": anchor_scale if args.variant != "baseline" else None,
        "parameter_count": params,
        "base_parameter_count": base_params,
        "initial_base_sha256": base_sha,
        "stage": "v0.18-self-300m" if args.target_tokens > 100_000_000 else "v0.17-self-100m",
        "steps": step,
        "tokens_seen": seen,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "data_epoch": epoch,
        "data_order_pos": order_pos,
        "data_seed_offset": args.data_seed_offset,
        "seed": args.seed,
        "total_elapsed_seconds": total_elapsed,
    }
    if args.variant != "baseline":
        payload["self_diagnostics"] = model.self_diagnostics()

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
        "chunk_tokens_per_second": (seen - start_seen) / max(chunk_elapsed, 1),
        "cumulative_tokens_per_second": seen / max(total_elapsed, 1),
        "base_sha256": base_sha,
        "base_parameters": base_params,
        "parameters": params,
        "extra_parameters": extra,
        "anchor_lr_scale": anchor_scale if args.variant != "baseline" else None,
        "data_epoch": epoch,
        "data_order_pos": order_pos,
        "data_seed_offset": args.data_seed_offset,
        "token_shard": args.tokens,
        "resources_final": resources(),
    }
    if args.variant != "baseline":
        report["self_diagnostics"] = model.self_diagnostics()

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    log(
        f"saved={args.save} tokens={seen:,} loss={last_loss:.4f} "
        f"chunk_min={chunk_elapsed/60:.1f} total_train_min={total_elapsed/60:.1f}"
    )


if __name__ == "__main__":
    main()
