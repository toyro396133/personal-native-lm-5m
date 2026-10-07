from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import time

import psutil
import torch
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM

from goldfish_self_variants import (
    GoldfishSelfLM,
    anchor_lr_scale_for_variant,
    anchor_replacement,
    build_variant,
    intervention_geometry,
    variant_name,
)

INTERVENTION_MODES = (
    "zero",
    "negate",
    "shuffle",
    "random_same_norm",
    "orthogonal_same_norm",
)

def now() -> str:
    return time.strftime("%H:%M:%S", time.gmtime())

def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)

def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2

def load_i32(path: str) -> torch.Tensor:
    p = Path(path)
    n = p.stat().st_size // 4
    return torch.from_file(str(p), shared=False, size=n, dtype=torch.int32)

def model_fingerprint(model: torch.nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            h.update(name.encode())
            t = tensor.detach().cpu().contiguous()
            h.update(str(tuple(t.shape)).encode())
            h.update(bytes(t.untyped_storage()))
    return h.hexdigest()

@torch.no_grad()
def eval_loss(model, ids: torch.Tensor, seq_len: int = 128, max_tokens: int = 16_384) -> dict:
    model.eval()
    ids = ids[: min(ids.numel(), max_tokens + 1)]
    total = 0.0
    predicted = 0
    started = time.perf_counter()
    for start in range(0, ids.numel() - 1, seq_len):
        chunk = ids[start : start + seq_len + 1]
        if chunk.numel() < 2:
            continue
        x = chunk[:-1].long().unsqueeze(0)
        y = chunk[1:].long().unsqueeze(0)
        logits = model(input_ids=x, use_cache=False).logits
        loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total += float(loss_sum)
        predicted += int(y.numel())
    elapsed = time.perf_counter() - started
    model.train()
    return {
        "nll_sum": total,
        "predicted_tokens": predicted,
        "nats_per_token": total / max(predicted, 1),
        "ppl_per_token": math.exp(min(total / max(predicted, 1), 30.0)),
        "tokens_per_second": predicted / max(elapsed, 1e-9),
        "elapsed_seconds": elapsed,
    }

@torch.no_grad()
def final_hidden(model, ids: torch.Tensor, seq_len: int = 128) -> torch.Tensor:
    model.eval()
    x = ids[:seq_len].long().unsqueeze(0)
    out = model(input_ids=x, use_cache=False, output_hidden_states=True, return_dict=True)
    hidden = out.hidden_states[-1].detach().float().cpu()
    model.train()
    return hidden

def cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    aa = F.normalize(a.reshape(-1, a.shape[-1]), dim=-1)
    bb = F.normalize(b.reshape(-1, b.shape[-1]), dim=-1)
    return float((aa * bb).sum(-1).mean())

@torch.no_grad()
def intervention_probe(model: torch.nn.Module, val_ids: torch.Tensor, seq_len: int = 128) -> dict | None:
    if not isinstance(model, GoldfishSelfLM):
        return None
    model.eval()
    anchor = model.self_anchor.detach().clone()
    normal_loss = eval_loss(model, val_ids, seq_len=seq_len, max_tokens=2048)
    normal_hidden = final_hidden(model, val_ids, seq_len=seq_len)
    result = {
        "normal_nats_per_token": normal_loss["nats_per_token"],
        "modes": {},
    }
    for i, mode in enumerate(INTERVENTION_MODES):
        replacement = anchor_replacement(anchor, mode, seed=7719 + i * 101)
        geometry = intervention_geometry(anchor, replacement)
        with model.temporary_anchor(replacement):
            alt_loss = eval_loss(model, val_ids, seq_len=seq_len, max_tokens=2048)
            alt_hidden = final_hidden(model, val_ids, seq_len=seq_len)
        delta = alt_hidden - normal_hidden
        result["modes"][mode] = {
            **geometry,
            "nats_per_token": alt_loss["nats_per_token"],
            "delta_nats_per_token": alt_loss["nats_per_token"] - normal_loss["nats_per_token"],
            "final_hidden_mean_l2": float(delta.norm(dim=-1).mean()),
            "final_hidden_rms": float(delta.square().mean().sqrt()),
            "final_hidden_cosine_to_normal": cosine_mean(normal_hidden, alt_hidden),
        }
    model.train()
    return result

def make_optimizer(model: torch.nn.Module, base_lr: float, adapter_lr: float):
    if not isinstance(model, GoldfishSelfLM):
        return torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.01)
    base_params = list(model.base.parameters())
    anchor = [model.self_anchor]
    adapter_params = list(model.self_adapters.parameters())
    scale = anchor_lr_scale_for_variant(model.variant)
    return torch.optim.AdamW(
        [
            {"params": base_params, "lr": base_lr, "weight_decay": 0.01, "name": "base"},
            {"params": adapter_params, "lr": adapter_lr, "weight_decay": 0.0, "name": "self_adapter"},
            {"params": anchor, "lr": adapter_lr * scale, "weight_decay": 0.0, "name": "self_anchor"},
        ]
    )

def save_training_state(path: str, model, optimizer, payload: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        **payload,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }, path)

def upload_to_hf(path: str, report_path: str, repo_name: str, variant_code: str, target_label: str) -> dict:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for checkpoint archival")
    api = HfApi(token=token)
    who = api.whoami()
    namespace = who.get("name") or who.get("fullname")
    if not namespace:
        raise RuntimeError("could not resolve Hugging Face namespace")
    repo_id = f"{namespace}/{repo_name}"
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)
    base = f"goldfish-124m/{variant_code}/{target_label}"
    api.upload_file(path_or_fileobj=path, path_in_repo=f"{base}/training_state.pt", repo_id=repo_id, repo_type="model")
    api.upload_file(path_or_fileobj=report_path, path_in_repo=f"{base}/metrics.json", repo_id=repo_id, repo_type="model")
    return {
        "repo_id": repo_id,
        "checkpoint_path": f"{base}/training_state.pt",
        "metrics_path": f"{base}/metrics.json",
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-code", required=True, choices=[f"V{i}" for i in range(1, 8)])
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--target-tokens", type=int, default=1_000_000)
    ap.add_argument("--milestones", default="250000,500000,1000000")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--base-lr", type=float, default=1e-5)
    ap.add_argument("--adapter-lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20261008)
    ap.add_argument("--data-seed", type=int, default=8841)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--projection-rank", type=int, default=4)
    ap.add_argument("--model", default="goldfish-models/heb_hebr_1000mb")
    ap.add_argument("--revision", default="425eee09d1b9ff32ffa21d6cfb300a74302a0799")
    ap.add_argument("--save", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--hf-repo", default="personal-native-lm-goldfish-self")
    args = ap.parse_args()

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    torch.manual_seed(args.seed)
    variant = variant_name(args.variant_code)

    log(f"code={args.variant_code} loading pretrained backbone revision={args.revision}")
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False
    for p in base.parameters():
        p.requires_grad_(True)

    base_parameter_count = sum(p.numel() for p in base.parameters())
    base_sha = model_fingerprint(base)
    model = build_variant(base, args.variant_code, rank=args.rank, projection_rank=args.projection_rank)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params != total_params:
        raise RuntimeError(f"not all parameters trainable: {trainable_params:,}/{total_params:,}")

    # Reset after adapter initialization so matched base dropout starts from the
    # same RNG state in every arm.
    torch.manual_seed(args.seed)

    train_tokens = load_i32(args.train)
    val_tokens = load_i32(args.val)
    width = args.seq_len + 1
    usable = train_tokens.numel() - train_tokens.numel() % width
    blocks = train_tokens[:usable].view(-1, width)
    if blocks.shape[0] < 2:
        raise RuntimeError("training shard too small")

    optimizer = make_optimizer(model, args.base_lr, args.adapter_lr)
    milestones = sorted(set(int(x) for x in args.milestones.split(",") if x.strip()))
    if args.target_tokens not in milestones:
        milestones.append(args.target_tokens)
        milestones.sort()

    log(
        f"code={args.variant_code} params={total_params:,} base={base_parameter_count:,} "
        f"extra={total_params-base_parameter_count:,} train_tokens={train_tokens.numel():,} "
        f"val_tokens={val_tokens.numel():,} rss_mb={rss_mb():.1f}"
    )

    report = {
        "schema_version": 1,
        "experiment": "goldfish-124m-self-seven-arm-1m",
        "variant_code": args.variant_code,
        "model_id": args.model,
        "model_revision": args.revision,
        "base_parameter_count": base_parameter_count,
        "parameter_count": total_params,
        "extra_parameter_count": total_params - base_parameter_count,
        "all_parameters_trainable": trainable_params == total_params,
        "seq_len": args.seq_len,
        "base_lr": args.base_lr,
        "adapter_lr": None if variant == "baseline" else args.adapter_lr,
        "anchor_lr_scale": None if variant == "baseline" else anchor_lr_scale_for_variant(variant),
        "rank": None if variant == "baseline" else args.rank,
        "projection_rank": None if variant == "baseline" else args.projection_rank,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "initial_base_sha256": base_sha,
        "milestones": [],
    }

    initial_eval = eval_loss(model, val_tokens, args.seq_len, max_tokens=16_384)
    report["initial_eval"] = initial_eval
    report["initial_self_diagnostics"] = model.self_diagnostics() if isinstance(model, GoldfishSelfLM) else None
    report["initial_interventions"] = intervention_probe(model, val_tokens, args.seq_len)
    log(f"code={args.variant_code} initial_val={initial_eval['nats_per_token']:.6f}")

    model.train()
    seen = 0
    step = 0
    epoch = 0
    order_pos = 0
    order = None
    started = time.perf_counter()
    max_rss = rss_mb()
    milestone_index = 0
    last_log = time.perf_counter()

    while seen < args.target_tokens:
        if order_pos == 0:
            g = torch.Generator(device="cpu")
            g.manual_seed(args.data_seed + 1000 * epoch)
            order = torch.randperm(blocks.shape[0], generator=g)
        while order_pos < order.numel() and seen < args.target_tokens:
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

            if time.perf_counter() - last_log >= 60:
                elapsed = time.perf_counter() - started
                log(
                    f"code={args.variant_code} step={step} seen={seen:,} loss={float(loss):.4f} "
                    f"tok_s={seen/max(elapsed,1e-9):.1f} rss_mb={rss_mb():.1f}"
                )
                last_log = time.perf_counter()

            while milestone_index < len(milestones) and seen >= milestones[milestone_index]:
                target = milestones[milestone_index]
                elapsed = time.perf_counter() - started
                val = eval_loss(model, val_tokens, args.seq_len, max_tokens=16_384)
                interventions = intervention_probe(model, val_tokens, args.seq_len)
                diagnostics = model.self_diagnostics() if isinstance(model, GoldfishSelfLM) else None
                entry = {
                    "target_tokens": target,
                    "tokens_seen": seen,
                    "step": step,
                    "train_last_loss": float(loss.detach()),
                    "elapsed_seconds": elapsed,
                    "cumulative_tokens_per_second": seen / max(elapsed, 1e-9),
                    "process_rss_mb": rss_mb(),
                    "max_process_rss_mb": max_rss,
                    "validation": val,
                    "self_diagnostics": diagnostics,
                    "interventions": interventions,
                }
                report["milestones"].append(entry)
                Path(args.report).parent.mkdir(parents=True, exist_ok=True)
                Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                log(
                    f"code={args.variant_code} milestone={target:,} seen={seen:,} "
                    f"val={val['nats_per_token']:.6f} tok_s={entry['cumulative_tokens_per_second']:.1f}"
                )
                milestone_index += 1

        if order_pos >= order.numel():
            epoch += 1
            order_pos = 0

    elapsed = time.perf_counter() - started
    report["final"] = {
        "tokens_seen": seen,
        "steps": step,
        "elapsed_seconds": elapsed,
        "tokens_per_second": seen / max(elapsed, 1e-9),
        "max_process_rss_mb": max_rss,
        "final_model_sha256": model_fingerprint(model),
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    payload = {
        "schema_version": 1,
        "experiment": report["experiment"],
        "variant_code": args.variant_code,
        "model_id": args.model,
        "model_revision": args.revision,
        "base_parameter_count": base_parameter_count,
        "parameter_count": total_params,
        "rank": None if variant == "baseline" else args.rank,
        "projection_rank": None if variant == "baseline" else args.projection_rank,
        "base_lr": args.base_lr,
        "adapter_lr": None if variant == "baseline" else args.adapter_lr,
        "anchor_lr_scale": None if variant == "baseline" else anchor_lr_scale_for_variant(variant),
        "tokens_seen": seen,
        "steps": step,
        "epoch": epoch,
        "order_pos": order_pos,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "seq_len": args.seq_len,
        "initial_base_sha256": base_sha,
    }
    save_training_state(args.save, model, optimizer, payload)
    log(f"code={args.variant_code} saved_state={args.save} bytes={Path(args.save).stat().st_size:,}")

    target_label = f"{args.target_tokens // 1000}k"
    hf = upload_to_hf(args.save, args.report, args.hf_repo, args.variant_code, target_label)
    report["archive"] = hf
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.upload_file(path_or_fileobj=args.report, path_in_repo=hf["metrics_path"], repo_id=hf["repo_id"], repo_type="model")
    log(f"code={args.variant_code} archived={hf['repo_id']}/{hf['checkpoint_path']}")

    del optimizer, model, base
    gc.collect()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
