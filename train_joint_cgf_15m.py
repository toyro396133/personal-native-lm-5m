from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from personal_state import PersonalState
from train_hebrew import load_tokenizer
from train_v17_chunk import load_tokens, make_model
from model_self_variants import anchor_lr_scale_for_variant
from core_goal_focus_hierarchical_lab import (
    make_hard_sequences,
    TRAIN_CORES, TEST_CORES,
    TRAIN_GOALS, TEST_GOALS,
    TRAIN_FOCI, TEST_FOCI,
    TRAIN_TEMPLATES, TEST_TEMPLATES,
)
from joint_cgf_model import JointCgfAuthority, proposal_target


ALLOWED_VARIANTS = {"projected_diff", "self_v1_slow"}


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


def make_optimizer(model, authority, variant, lr):
    anchor_id = id(model.self_anchor)
    main = [p for p in model.parameters() if id(p) != anchor_id]
    main += list(authority.parameters())
    return torch.optim.AdamW(
        [
            {"params": main, "lr": lr, "weight_decay": 0.1},
            {
                "params": [model.self_anchor],
                "lr": lr * anchor_lr_scale_for_variant(variant),
                "weight_decay": 0.1,
            },
        ]
    )


def tokenize_texts(tok, sequences, max_len=112):
    texts = set()
    for seq in sequences:
        texts.update(seq["initial"])
        for e in seq["steps"]:
            for k in (
                "core", "goal", "focus", "event",
                "candidate_core", "candidate_goal", "candidate_focus",
                "next_core", "next_goal", "next_focus",
            ):
                texts.add(e[k])
    return {
        text: tok.encode(text, bos=True, eos=True)[:max_len]
        for text in sorted(texts)
    }


def encode_ids(model, ids, lengths):
    if hasattr(model, "base"):
        base = model.base
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block, adapter in zip(base.blocks, model.self_adapters):
            x = block(x, None)
            x = adapter(x, model.self_anchor)
        x = base.final_norm(x)
    else:
        base = model
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block in base.blocks:
            x = block(x, None)
        x = base.final_norm(x)

    mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)


def encode_text_batch(model, tokenized, texts, device):
    unique = list(dict.fromkeys(texts))
    seqs = [tokenized[t] for t in unique]
    max_len = max(len(x) for x in seqs)
    ids = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
    lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long, device=device)
    for i, seq in enumerate(seqs):
        ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
    reps = encode_ids(model, ids, lengths)
    return {t: reps[i] for i, t in enumerate(unique)}


def stack(rep, rows, key):
    return torch.stack([rep[r[key]] for r in rows])


def structural_loss(model, authority, tokenized, rows, wrong_rows, device, cf_weight=0.25):
    texts = []
    keys = (
        "core", "goal", "focus", "event",
        "candidate_core", "candidate_goal", "candidate_focus",
    )
    for r in rows:
        texts.extend(r[k] for k in keys)
    for r in wrong_rows:
        texts.extend((r["core"], r["goal"], r["focus"]))

    rep = encode_text_batch(model, tokenized, texts, device)
    core, goal, focus, event = [stack(rep, rows, k) for k in ("core", "goal", "focus", "event")]
    cc, cg, cf = [stack(rep, rows, k) for k in ("candidate_core", "candidate_goal", "candidate_focus")]
    wc, wg, wf = [stack(rep, wrong_rows, k) for k in ("core", "goal", "focus")]

    root = model.self_anchor.unsqueeze(0).expand(len(rows), -1)
    proposal, auth, final = authority(root, core, goal, focus, event, cc, cg, cf)

    y_final = torch.tensor([r["target"] for r in rows], dtype=torch.float32, device=device)
    y_prop = torch.stack([
        proposal_target(r["kind"], r["target"], device) for r in rows
    ])
    y_auth = y_final

    loss = (
        0.8 * F.binary_cross_entropy(proposal, y_prop)
        + 1.2 * F.binary_cross_entropy(auth, y_auth)
        + 1.5 * F.binary_cross_entropy(final, y_final)
    )

    _, bad_auth, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        authority_hierarchy=(wc, wg, wf),
    )
    loss = loss + cf_weight * F.binary_cross_entropy(
        bad_auth, torch.zeros_like(bad_auth)
    )
    return loss


@torch.no_grad()
def build_rep_cache(model, tokenized, device, batch_size=64):
    model.eval()
    texts = list(tokenized)
    out = {}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        out.update({
            k: v.detach().cpu()
            for k, v in encode_text_batch(model, tokenized, chunk, device).items()
        })
    return out


def posthoc_train_authority(model, authority, rows, rep_cache, aux_steps, batch_size, device, lr, seed):
    for p in model.parameters():
        p.requires_grad_(False)
    authority.train()
    opt = torch.optim.AdamW(authority.parameters(), lr=lr, weight_decay=0.1)
    rng = random.Random(seed)
    root = model.self_anchor.detach().to(device)

    for _ in range(aux_steps):
        idx = [rng.randrange(len(rows)) for _ in range(batch_size)]
        wrong_idx = [(i + 97) % len(rows) for i in idx]
        batch = [rows[i] for i in idx]
        wrong = [rows[i] for i in wrong_idx]

        def s(rs, key):
            return torch.stack([rep_cache[r[key]] for r in rs]).to(device)

        core, goal, focus, event = [s(batch, k) for k in ("core", "goal", "focus", "event")]
        cc, cg, cf = [s(batch, k) for k in ("candidate_core", "candidate_goal", "candidate_focus")]
        wc, wg, wf = [s(wrong, k) for k in ("core", "goal", "focus")]

        proposal, auth, final = authority(
            root.unsqueeze(0).expand(batch_size, -1),
            core, goal, focus, event, cc, cg, cf,
        )
        y_final = torch.tensor([r["target"] for r in batch], dtype=torch.float32, device=device)
        y_prop = torch.stack([proposal_target(r["kind"], r["target"], device) for r in batch])

        loss = (
            0.8 * F.binary_cross_entropy(proposal, y_prop)
            + 1.2 * F.binary_cross_entropy(auth, y_final)
            + 1.5 * F.binary_cross_entropy(final, y_final)
        )
        _, bad_auth, _ = authority(
            root.unsqueeze(0).expand(batch_size, -1),
            core, goal, focus, event, cc, cg, cf,
            authority_hierarchy=(wc, wg, wf),
        )
        loss = loss + 0.25 * F.binary_cross_entropy(
            bad_auth, torch.zeros_like(bad_auth)
        )

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(authority.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def eval_structure(model, authority, sequences, rep_cache, device):
    model.eval()
    authority.eval()
    root = model.self_anchor.detach().to(device)

    total = exact = final_exact = 0
    first_errors = []
    false = [0, 0, 0]
    false_den = [0, 0, 0]
    true = [0, 0, 0]
    true_den = [0, 0, 0]

    for seq in sequences:
        pc, pg, pf = seq["initial"]
        tc, tg, tf = seq["initial"]
        first = None

        for t, e in enumerate(seq["steps"]):
            def one(text):
                return rep_cache[text].unsqueeze(0).to(device)

            _, _, probs = authority(
                root.unsqueeze(0),
                one(pc), one(pg), one(pf), one(e["event"]),
                one(e["candidate_core"]), one(e["candidate_goal"]), one(e["candidate_focus"]),
            )
            bits = (probs[0] >= 0.5).tolist()
            truth = [bool(x) for x in e["target"]]

            for s in range(3):
                if truth[s]:
                    true_den[s] += 1
                    true[s] += int(bits[s])
                else:
                    false_den[s] += 1
                    false[s] += int(bits[s])

            pc = e["candidate_core"] if bits[0] else pc
            pg = e["candidate_goal"] if bits[1] else pg
            pf = e["candidate_focus"] if bits[2] else pf
            tc, tg, tf = e["next_core"], e["next_goal"], e["next_focus"]

            total += 1
            ok = (pc, pg, pf) == (tc, tg, tf)
            exact += int(ok)
            if not ok and first is None:
                first = t + 1

        final_exact += int((pc, pg, pf) == (tc, tg, tf))
        first_errors.append(first if first is not None else len(seq["steps"]) + 1)

    div = lambda a, b: a / b if b else None
    return {
        "state_exact_per_step": div(exact, total),
        "final_state_exact": div(final_exact, len(sequences)),
        "mean_steps_until_first_error": statistics.mean(first_errors),
        "core_false_update_rate": div(false[0], false_den[0]),
        "goal_false_update_rate": div(false[1], false_den[1]),
        "focus_false_update_rate": div(false[2], false_den[2]),
        "core_change_recall": div(true[0], true_den[0]),
        "goal_change_recall": div(true[1], true_den[1]),
        "focus_change_recall": div(true[2], true_den[2]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--variant", choices=sorted(ALLOWED_VARIANTS), required=True)
    ap.add_argument("--mode", choices=["self_only", "joint"], required=True)
    ap.add_argument("--target-tokens", type=int, default=15_000_000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--structure-every", type=int, default=16)
    ap.add_argument("--structure-batch", type=int, default=4)
    ap.add_argument("--structure-weight", type=float, default=0.15)
    ap.add_argument("--save", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    started = time.time()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    tok, kind = load_tokenizer(args.tokenizer)
    cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
    cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
    base, model = make_model(cfg, args.variant, 8, 4)
    authority = JointCgfAuthority(cfg.d_model)

    initial_base_sha = fingerprint(base)
    initial_model_sha = fingerprint(model)
    base_params = sum(p.numel() for p in base.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    authority_params = sum(p.numel() for p in authority.parameters())

    model = model.to(args.device)
    authority = authority.to(args.device)
    opt = make_optimizer(model, authority, args.variant, args.lr)

    train_sequences = make_hard_sequences(
        320, 18, 30, 7101,
        TRAIN_CORES, TRAIN_GOALS, TRAIN_FOCI, TRAIN_TEMPLATES,
    )
    test_sequences = make_hard_sequences(
        180, 36, 52, 11103,
        TEST_CORES, TEST_GOALS, TEST_FOCI, TEST_TEMPLATES,
    )
    train_rows = [e for seq in train_sequences for e in seq["steps"]]
    tokenized = tokenize_texts(tok, train_sequences + test_sequences)

    tokens = load_tokens(args.tokens)
    width = args.seq_len + 1
    usable = tokens.numel() - tokens.numel() % width
    blocks = tokens[:usable].view(-1, width)
    n_blocks = blocks.shape[0]
    neutral = PersonalState("neutral").flatten().to(args.device)

    struct_rng = random.Random(args.seed + 9091)
    step = 0
    seen = 0
    aux_steps = 0
    first_loss = None
    last_loss = None
    first_aux_loss = None
    last_aux_loss = None
    epoch = 1
    order_pos = 0

    model.train()
    authority.train()

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

            _, lm_loss = model(x, state, y)
            loss = lm_loss
            aux_loss = None

            if (step + 1) % args.structure_every == 0:
                aux_steps += 1
                if args.mode == "joint":
                    ids = [struct_rng.randrange(len(train_rows)) for _ in range(args.structure_batch)]
                    wrong_ids = [(i + 97) % len(train_rows) for i in ids]
                    rows = [train_rows[i] for i in ids]
                    wrong = [train_rows[i] for i in wrong_ids]
                    aux_loss = structural_loss(
                        model, authority, tokenized, rows, wrong,
                        args.device, cf_weight=0.25,
                    )
                    loss = loss + args.structure_weight * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(authority.parameters()), 1.0
            )
            opt.step()

            step += 1
            seen += int(y.numel())
            last_loss = float(lm_loss.detach())
            if first_loss is None:
                first_loss = last_loss
            if aux_loss is not None:
                last_aux_loss = float(aux_loss.detach())
                if first_aux_loss is None:
                    first_aux_loss = last_aux_loss

            if step == 1 or step % 250 == 0:
                print(
                    f"mode={args.mode} variant={args.variant} step={step} "
                    f"tokens={seen:,} lm_loss={last_loss:.4f} "
                    f"aux_steps={aux_steps} aux_loss={last_aux_loss}",
                    flush=True,
                )

        if order_pos >= n_blocks:
            epoch += 1
            order_pos = 0

    # In the control arm, train the same authority head after LM training while
    # the final LM/SELF is frozen. This is the fair post-hoc structural control.
    all_train_tokenized = tokenize_texts(tok, train_sequences)
    all_test_tokenized = tokenize_texts(tok, test_sequences)
    if args.mode == "self_only":
        train_cache = build_rep_cache(model, all_train_tokenized, args.device)
        posthoc_train_authority(
            model, authority, train_rows, train_cache,
            aux_steps=aux_steps,
            batch_size=args.structure_batch,
            device=args.device,
            lr=args.lr,
            seed=args.seed + 9091,
        )
        for p in model.parameters():
            p.requires_grad_(True)

    test_cache = build_rep_cache(model, all_test_tokenized, args.device)
    structural = eval_structure(model, authority, test_sequences, test_cache, args.device)

    elapsed = time.time() - started
    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "authority": authority.state_dict(),
        "tokenizer_kind": kind,
        "model_profile": "5m",
        "variant": args.variant,
        "self_rank": 8,
        "projection_rank": 4,
        "anchor_lr_scale": anchor_lr_scale_for_variant(args.variant),
        "parameter_count": model_params,
        "base_parameter_count": base_params,
        "initial_base_sha256": initial_base_sha,
        "initial_model_sha256": initial_model_sha,
        "stage": "v0.16h-joint-15m",
        "joint_mode": args.mode,
        "steps": step,
        "tokens_seen": seen,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "aux_steps": aux_steps,
        "first_aux_loss": first_aux_loss,
        "last_aux_loss": last_aux_loss,
        "total_elapsed_seconds": elapsed,
        "self_diagnostics": model.self_diagnostics(),
        "structural_eval": structural,
        "authority_parameter_count": authority_params,
    }
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.save)

    report = {
        "experiment": "v0.16h joint LM SELF CGF Authority 15M pilot",
        "mode": args.mode,
        "variant": args.variant,
        "tokens_seen": seen,
        "steps": step,
        "aux_steps": aux_steps,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "first_aux_loss": first_aux_loss,
        "last_aux_loss": last_aux_loss,
        "model_parameters": model_params,
        "authority_parameters": authority_params,
        "initial_base_sha256": initial_base_sha,
        "initial_model_sha256": initial_model_sha,
        "anchor_lr_scale": anchor_lr_scale_for_variant(args.variant),
        "self_diagnostics": model.self_diagnostics(),
        "structural_eval": structural,
        "elapsed_seconds": elapsed,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
