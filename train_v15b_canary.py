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
from model_self_anchor import SelfAnchoredPersonalNativeLM
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--variant", choices=["baseline", "self"], required=True)
    ap.add_argument("--target-tokens", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--self-rank", type=int, default=8)
    ap.add_argument("--save", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    log(f"variant={args.variant} phase=load_tokenizer")
    tok, kind = load_tokenizer(args.tokenizer)

    log(f"variant={args.variant} phase=init_base_model")
    cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    base = PersonalNativeLM(cfg)
    base_sha = fingerprint(base)
    base_params = sum(p.numel() for p in base.parameters())

    if args.variant == "self":
        model = SelfAnchoredPersonalNativeLM(base, rank=args.self_rank)
    else:
        model = base
    model = model.to(args.device)
    params = sum(p.numel() for p in model.parameters())
    log(
        f"variant={args.variant} phase=model_ready parameters={params:,} "
        f"base_parameters={base_params:,} base_sha256={base_sha} resources={resources()}"
    )

    log(f"variant={args.variant} phase=load_shared_token_ids path={args.tokens}")
    tokens = torch.load(args.tokens, map_location="cpu", weights_only=True)
    if tokens.ndim != 1:
        raise ValueError(f"expected 1D token tensor, got shape={tuple(tokens.shape)}")
    log(
        f"variant={args.variant} phase=token_ids_loaded tokens={tokens.numel():,} "
        f"dtype={tokens.dtype} resources={resources()}"
    )

    width = args.seq_len + 1
    usable = tokens.numel() - tokens.numel() % width
    blocks = tokens[:usable].view(-1, width)
    n_blocks = blocks.shape[0]
    if n_blocks < args.batch_size:
        raise RuntimeError("shared token tensor too small")

    neutral = PersonalState("neutral").flatten().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    step = 0
    seen = 0
    first_loss = None
    last_loss = None
    last_heartbeat = time.time()
    model.train()

    log(
        f"variant={args.variant} phase=training_started blocks={n_blocks:,} "
        f"target_tokens={args.target_tokens:,}"
    )

    epoch = 0
    while seen < args.target_tokens:
        epoch += 1
        g = torch.Generator(device="cpu")
        g.manual_seed(args.seed + 1000 * epoch)
        order = torch.randperm(n_blocks, generator=g)

        for j in range(0, n_blocks, args.batch_size):
            idx = order[j:j + args.batch_size]
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
            batch_tokens = int(y.numel())
            seen += batch_tokens
            last_loss = float(loss.detach())
            if first_loss is None:
                first_loss = last_loss

            if step == 1 or step % 100 == 0 or time.time() - last_heartbeat >= 60:
                elapsed = time.time() - started
                rate = seen / max(elapsed, 1)
                log(
                    f"variant={args.variant} phase=train epoch={epoch} step={step} "
                    f"tokens_seen={seen:,} loss={last_loss:.4f} "
                    f"tokens_per_sec={rate:.1f} elapsed_min={elapsed/60:.1f} "
                    f"resources={resources()}"
                )
                last_heartbeat = time.time()

            if seen >= args.target_tokens:
                break

    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "tokenizer_file": args.tokenizer,
        "tokenizer_kind": kind,
        "model_profile": "5m",
        "variant": args.variant,
        "self_rank": args.self_rank if args.variant == "self" else None,
        "parameter_count": params,
        "base_parameter_count": base_params,
        "initial_base_sha256": base_sha,
        "stage": "v0.15b-canary",
        "steps": step,
        "tokens_seen": seen,
        "first_loss": first_loss,
        "last_loss": last_loss,
    }
    if args.variant == "self":
        payload["self_diagnostics"] = model.self_diagnostics()

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.save)

    report = {
        "variant": args.variant,
        "target_tokens": args.target_tokens,
        "tokens_seen": seen,
        "steps": step,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "elapsed_seconds": time.time() - started,
        "tokens_per_second": seen / max(time.time() - started, 1),
        "base_sha256": base_sha,
        "base_parameters": base_params,
        "parameters": params,
        "extra_parameters": params - base_params,
        "resources_final": resources(),
    }
    if args.variant == "self":
        report["self_diagnostics"] = model.self_diagnostics()
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    log(
        f"variant={args.variant} phase=checkpoint_saved path={args.save} "
        f"tokens_seen={seen:,} loss={last_loss:.4f} total_min={(time.time()-started)/60:.1f}"
    )


if __name__ == "__main__":
    main()
