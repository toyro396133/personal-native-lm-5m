#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path

import psutil
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def collect_eval_text(max_chars: int, seed: int) -> tuple[str, dict]:
    """Deterministically collect a modest Hebrew FineWeb2 slice.

    We keep this deliberately small because this is a backbone selection
    canary, not the final language benchmark.
    """
    ds = load_dataset(
        "HuggingFaceFW/fineweb-2",
        name="heb_Hebr",
        split="train",
        streaming=True,
    )
    ds = ds.shuffle(seed=seed, buffer_size=2000)
    parts: list[str] = []
    total = 0
    docs = 0
    for row in ds:
        text = str(row.get("text", "")).strip()
        if len(text) < 200:
            continue
        # Prefer chunks that are visibly Hebrew-heavy.
        heb = len(re.findall(r"[\u0590-\u05FF]", text))
        letters = len(re.findall(r"\w", text, flags=re.UNICODE))
        if letters and heb / max(letters, 1) < 0.45:
            continue
        take = text[: min(len(text), 3000)]
        parts.append(take)
        total += len(take)
        docs += 1
        if total >= max_chars:
            break
    joined = "\n\n".join(parts)
    joined = joined[:max_chars]
    if len(joined) < max_chars // 2:
        raise RuntimeError(f"FineWeb2 stream yielded only {len(joined)} chars")
    meta = {
        "dataset": "HuggingFaceFW/fineweb-2",
        "config": "heb_Hebr",
        "shuffle_seed": seed,
        "shuffle_buffer": 2000,
        "documents": docs,
        "chars": len(joined),
        "utf8_bytes": len(joined.encode("utf-8")),
        "sha256": sha256_text(joined),
    }
    return joined, meta


def model_context_limit(model, tokenizer) -> int:
    candidates = []
    for name in ("max_position_embeddings", "n_positions", "n_ctx", "seq_length"):
        x = getattr(model.config, name, None)
        if isinstance(x, int) and 8 <= x <= 1_000_000:
            candidates.append(x)
    tmax = getattr(tokenizer, "model_max_length", None)
    if isinstance(tmax, int) and 8 <= tmax <= 1_000_000:
        candidates.append(tmax)
    return min(candidates) if candidates else 512


@torch.no_grad()
def eval_nll(model, ids: torch.Tensor, chunk_len: int) -> dict:
    model.eval()
    total_nll = 0.0
    predicted = 0
    chunks = 0
    started = time.perf_counter()
    for start in range(0, ids.numel() - 1, chunk_len):
        chunk = ids[start : start + chunk_len + 1]
        if chunk.numel() < 2:
            continue
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:].unsqueeze(0)
        logits = model(input_ids=x).logits
        loss_sum = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction="sum",
        )
        total_nll += float(loss_sum)
        predicted += y.numel()
        chunks += 1
    elapsed = time.perf_counter() - started
    return {
        "nll_sum": total_nll,
        "predicted_tokens": predicted,
        "nats_per_token": total_nll / max(predicted, 1),
        "ppl_per_token": math.exp(min(total_nll / max(predicted, 1), 30.0)),
        "eval_seconds": elapsed,
        "eval_tokens_per_second": predicted / max(elapsed, 1e-9),
        "chunks": chunks,
    }


def generation_diagnostics(text: str) -> dict:
    chars = [c for c in text if not c.isspace()]
    heb = sum("\u0590" <= c <= "\u05FF" for c in chars)
    replacement = text.count("\ufffd")
    words = re.findall(r"\S+", text)
    grams = [tuple(words[i:i+4]) for i in range(max(0, len(words)-3))]
    repeated_4gram_ratio = 0.0
    if grams:
        repeated_4gram_ratio = 1.0 - len(set(grams)) / len(grams)
    return {
        "chars": len(text),
        "hebrew_char_ratio": heb / max(len(chars), 1),
        "replacement_chars": replacement,
        "words": len(words),
        "repeated_4gram_ratio": repeated_4gram_ratio,
    }


@torch.no_grad()
def generate_samples(model, tokenizer, prompts: list[dict], max_new_tokens: int, seed: int) -> list[dict]:
    model.eval()
    out = []
    for i, p in enumerate(prompts):
        torch.manual_seed(seed + i)
        encoded = tokenizer(p["text"], return_tensors="pt", add_special_tokens=False)
        before = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.8,
            top_p=0.92,
            repetition_penalty=1.05,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        new_ids = generated[0, before:]
        continuation = tokenizer.decode(new_ids, skip_special_tokens=True)
        out.append({
            "id": p["id"],
            "prompt": p["text"],
            "continuation": continuation,
            "diagnostics": generation_diagnostics(continuation),
        })
    return out


def full_train_benchmark(model, ids: torch.Tensor, seq_len: int, steps: int, lr: float) -> dict:
    """Actually update every model parameter with AdamW on CPU."""
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in trainable)
    if trainable_params != params:
        raise RuntimeError(f"Not fully trainable: trainable={trainable_params:,} total={params:,}")

    seq_len = min(seq_len, ids.numel() - 1)
    x = ids[:seq_len].unsqueeze(0)
    y = ids[1:seq_len+1].unsqueeze(0)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)

    times = []
    losses = []
    max_rss = rss_mb()

    # First step is an explicit warm-up and allocates Adam state.
    all_steps = steps + 1
    for s in range(all_steps):
        started = time.perf_counter()
        logits = model(input_ids=x).logits
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        elapsed = time.perf_counter() - started
        max_rss = max(max_rss, rss_mb())
        if s > 0:
            times.append(elapsed)
            losses.append(float(loss.detach()))

    total_measured_tokens = seq_len * len(times)
    total_measured_seconds = sum(times)
    result = {
        "all_parameters_trainable": True,
        "total_parameters": params,
        "trainable_parameters": trainable_params,
        "seq_len": seq_len,
        "batch_size": 1,
        "optimizer": "AdamW",
        "measured_steps": len(times),
        "step_seconds": times,
        "median_step_seconds": statistics.median(times) if times else None,
        "tokens_per_second": total_measured_tokens / max(total_measured_seconds, 1e-9),
        "losses": losses,
        "max_process_rss_mb": max_rss,
    }
    del opt
    model.zero_grad(set_to_none=True)
    gc.collect()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-chars", type=int, default=24000)
    ap.add_argument("--eval-seed", type=int, default=913)
    ap.add_argument("--eval-seq-len", type=int, default=256)
    ap.add_argument("--train-seq-len", type=int, default=128)
    ap.add_argument("--train-steps", type=int, default=2)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--generation-prompts", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))

    prompts_payload = json.loads(Path(args.prompts).read_text(encoding="utf-8"))
    prompts = prompts_payload["prompts"][: args.generation_prompts]

    text, data_meta = collect_eval_text(args.eval_chars, args.eval_seed)

    api = HfApi()
    info = api.model_info(args.model)
    revision = info.sha

    load_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        revision=revision,
        trust_remote_code=True,
    )
    if tokenizer.eos_token_id is None and tokenizer.sep_token_id is not None:
        tokenizer.eos_token = tokenizer.sep_token
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=revision,
        trust_remote_code=True,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    model.to("cpu")
    load_seconds = time.perf_counter() - load_started

    params = sum(p.numel() for p in model.parameters())
    model_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    context_limit = model_context_limit(model, tokenizer)

    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    if encoded.numel() < 32:
        raise RuntimeError("Evaluation text tokenized to unexpectedly few tokens")

    chars = len(text)
    utf8_bytes = len(text.encode("utf-8"))
    tokenizer_stats = {
        "eval_tokens": int(encoded.numel()),
        "chars": chars,
        "utf8_bytes": utf8_bytes,
        "chars_per_token": chars / encoded.numel(),
        "bytes_per_token": utf8_bytes / encoded.numel(),
        "vocab_size": int(len(tokenizer)),
    }

    eval_chunk = min(args.eval_seq_len, max(8, context_limit - 1))
    nll = eval_nll(model, encoded, eval_chunk)
    nll["nats_per_char"] = nll["nll_sum"] / max(chars, 1)
    nll["bits_per_char"] = nll["nats_per_char"] / math.log(2)
    nll["nats_per_utf8_byte"] = nll["nll_sum"] / max(utf8_bytes, 1)

    generations = generate_samples(
        model,
        tokenizer,
        prompts,
        args.max_new_tokens,
        args.seed,
    )

    train_seq = min(args.train_seq_len, max(8, context_limit - 1))
    training = full_train_benchmark(
        model,
        encoded,
        train_seq,
        args.train_steps,
        lr=1e-5,
    )

    result = {
        "schema_version": 1,
        "model_id": args.model,
        "resolved_revision": revision,
        "library_name": getattr(info, "library_name", None),
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "parameters": params,
        "model_parameter_bytes_fp32": model_bytes,
        "load_seconds": load_seconds,
        "context_limit_detected": context_limit,
        "process_rss_after_all_mb": rss_mb(),
        "data": data_meta,
        "tokenizer": tokenizer_stats,
        "language_modeling": nll,
        "full_training": training,
        "generations": generations,
        "environment": {
            "torch": torch.__version__,
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "platform": os.uname().sysname if hasattr(os, "uname") else os.name,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": args.model,
        "revision": revision,
        "parameters": params,
        "nats_per_char": nll["nats_per_char"],
        "bits_per_char": nll["bits_per_char"],
        "train_tokens_per_second": training["tokens_per_second"],
        "max_rss_mb": training["max_process_rss_mb"],
        "output": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
