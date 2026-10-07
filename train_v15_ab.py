from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from config import ModelConfig
from model import PersonalNativeLM
from model_self_anchor import SelfAnchoredPersonalNativeLM
from personal_state import PersonalState
from train_hebrew import load_tokenizer


def base_fingerprint(model: PersonalNativeLM) -> str:
    """Hash the initial ordinary LM without requiring NumPy.

    Serialize each tensor's raw bytes through PyTorch storage. Both A/B jobs
    construct the same base model before the SELF-only parameters are created,
    so matching hashes prove identical language-backbone initialization.
    """
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode("utf-8"))
            h.update(str(tuple(t.shape)).encode("ascii"))
            h.update(str(t.dtype).encode("ascii"))
            h.update(bytes(t.untyped_storage()))
    return h.hexdigest()


def save_checkpoint(
    path: Path,
    model,
    cfg: ModelConfig,
    tokenizer_path: str,
    tokenizer_kind: str,
    variant: str,
    rank: int,
    params: int,
    base_params: int,
    initial_base_sha256: str,
    step: int,
    tokens_seen: int,
    first_loss: float,
    last_loss: float,
    epoch: int,
):
    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "tokenizer_file": tokenizer_path,
        "tokenizer_kind": tokenizer_kind,
        "model_profile": "5m",
        "variant": variant,
        "self_rank": rank if variant == "self" else None,
        "parameter_count": params,
        "base_parameter_count": base_params,
        "initial_base_sha256": initial_base_sha256,
        "stage": "v0.15-hebrew-scaling-self-ab",
        "steps": step,
        "tokens_seen": tokens_seen,
        "epoch": epoch,
        "first_loss": first_loss,
        "last_loss": last_loss,
    }
    if variant == "self":
        payload["self_diagnostics"] = model.self_diagnostics()
    torch.save(payload, path)
    print(
        f"checkpoint={path} variant={variant} step={step} "
        f"tokens_seen={tokens_seen:,} loss={last_loss:.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--variant", choices=["baseline", "self"], required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--self-rank", type=int, default=8)
    ap.add_argument(
        "--checkpoint-tokens",
        default="10000000,25000000,50000000",
        help="Cumulative target-token exposures at which to save checkpoints.",
    )
    ap.add_argument("--out-dir", default="artifacts/v15")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    milestones = sorted({int(x) for x in args.checkpoint_tokens.split(",") if x.strip()})
    if not milestones or milestones[0] <= 0:
        raise ValueError("checkpoint token targets must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)

    # Build the ordinary LM first in both A/B jobs. This guarantees that the
    # language-backbone initialization is byte-identical before SELF exists.
    base = PersonalNativeLM(cfg)
    initial_base_sha256 = base_fingerprint(base)
    base_params = sum(p.numel() for p in base.parameters())

    if args.variant == "self":
        model = SelfAnchoredPersonalNativeLM(base, rank=args.self_rank)
    else:
        model = base
    model = model.to(args.device)

    params = sum(p.numel() for p in model.parameters())
    print(
        f"variant={args.variant} profile=5m parameters={params:,} "
        f"base_parameters={base_params:,} initial_base_sha256={initial_base_sha256}"
    )

    text = Path(args.text_file).read_text(encoding="utf-8")
    ids = tok.encode(text)
    del text

    block_width = args.seq_len + 1
    usable = len(ids) - (len(ids) % block_width)
    if usable < block_width * args.batch_size:
        raise SystemExit("Corpus is too small for selected sequence/batch size.")

    # Tensorized blocks avoid millions of Python list objects for the 50M run.
    token_tensor = torch.tensor(ids[:usable], dtype=torch.long)
    del ids
    blocks = token_tensor.view(-1, block_width)
    n_blocks = blocks.shape[0]

    print(
        f"encoded_tokens={usable:,} blocks={n_blocks:,} "
        f"target_tokens={milestones[-1]:,} device={args.device} tokenizer={tokenizer_kind}"
    )

    neutral = PersonalState("neutral").flatten().to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step = 0
    tokens_seen = 0
    first_loss = None
    last_loss = None
    next_i = 0
    model.train()

    for epoch in range(1, args.epochs + 1):
        # Independent RNG keeps batch order identical in baseline and SELF jobs,
        # even though the SELF job creates extra Torch parameters.
        g = torch.Generator(device="cpu")
        g.manual_seed(args.seed + 1000 * epoch)
        order = torch.randperm(n_blocks, generator=g)

        for j in range(0, n_blocks, args.batch_size):
            idx = order[j:j + args.batch_size]
            if idx.numel() < 2:
                continue

            batch = blocks[idx]
            x = batch[:, :-1].to(args.device)
            y = batch[:, 1:].to(args.device)
            state = neutral.unsqueeze(0).expand(x.size(0), -1)

            _, loss = model(x, state, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            step += 1
            tokens_seen += int(y.numel())
            last_loss = float(loss.detach())
            if first_loss is None:
                first_loss = last_loss

            if step == 1 or step % 100 == 0:
                print(
                    f"variant={args.variant} epoch={epoch} step={step} "
                    f"tokens_seen={tokens_seen:,} loss={last_loss:.4f}"
                )

            while next_i < len(milestones) and tokens_seen >= milestones[next_i]:
                target = milestones[next_i]
                label = f"{target // 1_000_000}m"
                save_checkpoint(
                    out_dir / f"v15-{args.variant}-{label}.pt",
                    model, cfg, args.tokenizer, tokenizer_kind, args.variant,
                    args.self_rank, params, base_params, initial_base_sha256,
                    step, tokens_seen, first_loss, last_loss, epoch,
                )
                next_i += 1

            if next_i >= len(milestones):
                break

        if next_i >= len(milestones):
            break

    if next_i < len(milestones):
        raise SystemExit(
            f"Training ended at {tokens_seen:,} target-token exposures before "
            f"the final milestone {milestones[-1]:,}. Increase corpus reserve."
        )

    report = {
        "version": "0.15",
        "variant": args.variant,
        "base_parameters": base_params,
        "parameters": params,
        "extra_parameters": params - base_params,
        "initial_base_sha256": initial_base_sha256,
        "encoded_corpus_tokens": usable,
        "blocks": n_blocks,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "milestones": milestones,
        "final_steps": step,
        "final_tokens_seen": tokens_seen,
        "first_loss": first_loss,
        "last_loss": last_loss,
    }
    if args.variant == "self":
        report["self_diagnostics"] = model.self_diagnostics()

    report_path = out_dir / f"v15-{args.variant}-train-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
