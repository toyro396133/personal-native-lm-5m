from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from goldfish_checkpoint_codec import (
    save_bf16_model_checkpoint,
    save_int4_model_checkpoint,
)
from goldfish_self_variants import GoldfishSelfLM, build_variant, variant_name
from train_goldfish_self_arm import (
    eval_loss,
    intervention_probe,
    load_i32,
    make_optimizer,
    rss_mb,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}", flush=True)


def build_from_resume(payload: dict, device: str = "cpu"):
    variant_code = payload["variant_code"]
    model_id = payload["model_id"]
    revision = payload["model_revision"]
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(True)

    model = build_variant(
        base,
        variant_code,
        rank=int(payload.get("rank") or 8),
        projection_rank=int(payload.get("projection_rank") or 4),
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)

    base_lr = float(payload.get("base_lr") or 1e-5)
    adapter_lr = float(payload.get("adapter_lr") or 1e-4)
    optimizer = make_optimizer(model, base_lr, adapter_lr)
    optimizer.load_state_dict(payload["optimizer_state"])
    if payload.get("torch_rng_state") is not None:
        torch.set_rng_state(payload["torch_rng_state"])
    return model, optimizer, base_lr, adapter_lr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-code", required=True, choices=[f"V{i}" for i in range(1, 8)])
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--resume-state", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--save-state", required=True)
    ap.add_argument("--save-int4", required=True)
    ap.add_argument("--save-bf16")
    ap.add_argument("--report", required=True)
    ap.add_argument("--stage-tokens", type=int, default=1_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--data-seed-base", type=int, default=8841)
    args = ap.parse_args()

    if not 2 <= args.stage <= 100:
        raise ValueError("stage must be 2..100")

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    payload = torch.load(args.resume_state, map_location="cpu", weights_only=False)
    if payload.get("variant_code") != args.variant_code:
        raise RuntimeError(
            f"resume variant mismatch: {payload.get('variant_code')} != {args.variant_code}"
        )

    previous_stage = int(payload.get("stage") or 1)
    if previous_stage != args.stage - 1:
        raise RuntimeError(
            f"resume stage mismatch: previous={previous_stage}, requested={args.stage}"
        )

    model, optimizer, base_lr, adapter_lr = build_from_resume(payload)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable != total_params:
        raise RuntimeError(f"not fully trainable: {trainable:,}/{total_params:,}")

    train_tokens = load_i32(args.train)
    val_tokens = load_i32(args.val)

    width = args.seq_len + 1
    usable = train_tokens.numel() - train_tokens.numel() % width
    blocks = train_tokens[:usable].view(-1, width)
    if blocks.shape[0] < 2:
        raise RuntimeError("training shard too small")

    initial_eval = eval_loss(model, val_tokens, args.seq_len, max_tokens=16_384)
    initial_interventions = intervention_probe(model, val_tokens, args.seq_len)

    seen = 0
    step = 0
    epoch = 0
    order_pos = 0
    order = None
    started = time.perf_counter()
    max_rss = rss_mb()
    last_log = started
    data_seed = args.data_seed_base + args.stage * 10_007

    model.train()
    while seen < args.stage_tokens:
        if order_pos == 0:
            g = torch.Generator(device="cpu")
            g.manual_seed(data_seed + 1000 * epoch)
            order = torch.randperm(blocks.shape[0], generator=g)

        while order_pos < order.numel() and seen < args.stage_tokens:
            idx = int(order[order_pos])
            order_pos += 1
            batch = blocks[idx]
            x = batch[:-1].long().unsqueeze(0)
            y = batch[1:].long().unsqueeze(0)

            logits = model(input_ids=x, use_cache=False).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            seen += int(y.numel())
            max_rss = max(max_rss, rss_mb())

            now = time.perf_counter()
            if now - last_log >= 60:
                log(
                    f"V={args.variant_code} stage={args.stage} step={step} "
                    f"stage_seen={seen:,} loss={float(loss):.4f} "
                    f"tok_s={seen/max(now-started,1e-9):.1f} rss_mb={rss_mb():.1f}"
                )
                last_log = now

        if order_pos >= order.numel():
            epoch += 1
            order_pos = 0

    elapsed = time.perf_counter() - started
    final_eval = eval_loss(model, val_tokens, args.seq_len, max_tokens=16_384)
    final_interventions = intervention_probe(model, val_tokens, args.seq_len)
    diagnostics = model.self_diagnostics() if isinstance(model, GoldfishSelfLM) else None

    previous_actual = int(payload.get("cumulative_actual_tokens") or payload.get("tokens_seen") or 1_000_000)
    cumulative_actual = previous_actual + seen
    nominal_total = args.stage * 1_000_000

    metadata = {
        "schema_version": 2,
        "experiment": "goldfish-124m-self-100m-100-stage",
        "variant_code": args.variant_code,
        "variant_name": variant_name(args.variant_code),
        "stage": args.stage,
        "nominal_total_tokens": nominal_total,
        "cumulative_actual_tokens": cumulative_actual,
        "stage_tokens_seen": seen,
        "stage_steps": step,
        "total_parameters": total_params,
        "base_lr": base_lr,
        "adapter_lr": None if args.variant_code == "V1" else adapter_lr,
        "seq_len": args.seq_len,
        "data_seed": data_seed,
        "source_stage": previous_stage,
        "model_id": payload["model_id"],
        "model_revision": payload["model_revision"],
    }

    report = {
        **metadata,
        "initial_validation": initial_eval,
        "final_validation": final_eval,
        "initial_interventions": initial_interventions,
        "final_interventions": final_interventions,
        "self_diagnostics": diagnostics,
        "elapsed_seconds": elapsed,
        "stage_tokens_per_second": seen / max(elapsed, 1e-9),
        "max_process_rss_mb": max_rss,
        "language_metric_role": "health/control only; not the primary SELF score",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    exact_payload = {
        "schema_version": 2,
        "experiment": metadata["experiment"],
        "variant_code": args.variant_code,
        "variant_name": metadata["variant_name"],
        "stage": args.stage,
        "model_id": payload["model_id"],
        "model_revision": payload["model_revision"],
        "base_parameter_count": payload.get("base_parameter_count"),
        "parameter_count": total_params,
        "rank": payload.get("rank"),
        "projection_rank": payload.get("projection_rank"),
        "base_lr": base_lr,
        "adapter_lr": payload.get("adapter_lr"),
        "anchor_lr_scale": payload.get("anchor_lr_scale"),
        "tokens_seen": cumulative_actual,
        "cumulative_actual_tokens": cumulative_actual,
        "nominal_total_tokens": nominal_total,
        "steps": int(payload.get("steps") or 0) + step,
        "seed": payload.get("seed"),
        "data_seed": data_seed,
        "seq_len": args.seq_len,
        "initial_base_sha256": payload.get("initial_base_sha256"),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }

    Path(args.save_state).parent.mkdir(parents=True, exist_ok=True)
    torch.save(exact_payload, args.save_state)

    archive_meta = {
        k: exact_payload.get(k)
        for k in (
            "schema_version",
            "experiment",
            "variant_code",
            "variant_name",
            "stage",
            "model_id",
            "model_revision",
            "parameter_count",
            "rank",
            "projection_rank",
            "nominal_total_tokens",
            "cumulative_actual_tokens",
            "seq_len",
        )
    }
    state = model.state_dict()
    save_int4_model_checkpoint(args.save_int4, state, archive_meta, block_size=256)
    if args.save_bf16:
        save_bf16_model_checkpoint(args.save_bf16, state, archive_meta)

    log(
        f"complete V={args.variant_code} stage={args.stage} nominal={nominal_total:,} "
        f"actual={cumulative_actual:,} tok_s={report['stage_tokens_per_second']:.1f} "
        f"val={final_eval['nats_per_token']:.6f}"
    )

    del exact_payload, optimizer, model
    gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
