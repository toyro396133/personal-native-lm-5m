from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
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
from train_joint_cgf_15m import (
    fingerprint,
    tokenize_texts,
    encode_text_batch,
    stack,
    build_rep_cache,
)


VARIANT = "self_v1_slow"
ROOT_MODES = {"self", "sham"}
EVENT_KINDS = (
    "goal_pressure",
    "core_pressure",
    "goal_change",
    "core_change",
    "goal_done",
    "focus",
    "tempting_focus",
    "noise",
)


def tensor_sha(t: torch.Tensor) -> str:
    x = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(x.shape)).encode())
    h.update(str(x.dtype).encode())
    h.update(bytes(x.untyped_storage()))
    return h.hexdigest()


def make_optimizer(model, authority, root_proxy, lr):
    anchor_id = id(model.self_anchor)
    main = [p for p in model.parameters() if id(p) != anchor_id]
    main += list(authority.parameters())
    scale = anchor_lr_scale_for_variant(VARIANT)
    return torch.optim.AdamW(
        [
            {"params": main, "lr": lr, "weight_decay": 0.1},
            {"params": [model.self_anchor], "lr": lr * scale, "weight_decay": 0.1},
            {"params": [root_proxy], "lr": lr * scale, "weight_decay": 0.1},
        ]
    )


def optimizer_to_device(opt, device):
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def active_root(model, root_proxy, root_mode):
    return model.self_anchor if root_mode == "self" else root_proxy


def asymmetric_bce(prob, target):
    neg = torch.tensor([3.0, 2.3, 1.0], device=prob.device, dtype=prob.dtype)
    pos = torch.tensor([1.3, 1.5, 1.0], device=prob.device, dtype=prob.dtype)
    prob = prob.clamp(1e-5, 1 - 1e-5)
    return (
        -target * pos * torch.log(prob)
        -(1 - target) * neg * torch.log(1 - prob)
    ).mean()


def build_kind_buckets(rows):
    out = {k: [] for k in EVENT_KINDS}
    for r in rows:
        out.setdefault(r["kind"], []).append(r)
    return out


def sample_structural_rows(buckets, all_rows, rng, batch_size):
    rows = []
    start = rng.randrange(len(EVENT_KINDS))
    for i in range(batch_size):
        kind = EVENT_KINDS[(start + i) % len(EVENT_KINDS)]
        pool = buckets.get(kind) or all_rows
        rows.append(rng.choice(pool))
    return rows


def choose_wrong_rows(rows, all_rows, rng):
    wrong = []
    for r in rows:
        chosen = None
        for _ in range(32):
            cand = rng.choice(all_rows)
            if (
                cand["core"] != r["core"]
                and cand["goal"] != r["goal"]
                and cand["focus"] != r["focus"]
            ):
                chosen = cand
                break
        wrong.append(chosen or rng.choice(all_rows))
    return wrong


def structural_loss(
    model,
    authority,
    root_proxy,
    root_mode,
    tokenized,
    rows,
    wrong_rows,
    device,
):
    keys = (
        "core", "goal", "focus", "event",
        "candidate_core", "candidate_goal", "candidate_focus",
    )
    texts = []
    for r in rows:
        texts.extend(r[k] for k in keys)
    for r in wrong_rows:
        texts.extend((r["core"], r["goal"], r["focus"]))

    rep = encode_text_batch(model, tokenized, texts, device)
    core, goal, focus, event = [
        stack(rep, rows, k) for k in ("core", "goal", "focus", "event")
    ]
    cc, cg, cf = [
        stack(rep, rows, k)
        for k in ("candidate_core", "candidate_goal", "candidate_focus")
    ]
    wc, wg, wf = [
        stack(rep, wrong_rows, k) for k in ("core", "goal", "focus")
    ]

    root_vec = active_root(model, root_proxy, root_mode)
    root = root_vec.unsqueeze(0).expand(len(rows), -1)
    proposal, auth, final = authority(
        root, core, goal, focus, event, cc, cg, cf
    )

    y_final = torch.tensor(
        [r["target"] for r in rows], dtype=torch.float32, device=device
    )
    y_prop = torch.stack([
        proposal_target(r["kind"], r["target"], device) for r in rows
    ])

    loss = (
        0.6 * F.binary_cross_entropy(proposal, y_prop)
        + 1.0 * asymmetric_bce(auth, y_final)
        + 1.6 * asymmetric_bce(final, y_final)
    )

    # Nested counterfactual hierarchy training:
    # wrong CORE invalidates all lower authority;
    # wrong GOAL invalidates GOAL and FOCUS while preserving CORE;
    # wrong FOCUS invalidates only FOCUS.
    _, auth_bad_core, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        authority_hierarchy=(wc, goal, focus),
    )
    _, auth_bad_goal, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        authority_hierarchy=(core, wg, focus),
    )
    _, auth_bad_focus, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        authority_hierarchy=(core, goal, wf),
    )
    _, auth_bad_all, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        authority_hierarchy=(wc, wg, wf),
    )

    z = torch.zeros_like(y_final)
    keep_core = torch.stack((y_final[:, 0], z[:, 1], z[:, 2]), dim=-1)
    keep_core_goal = torch.stack(
        (y_final[:, 0], y_final[:, 1], z[:, 2]), dim=-1
    )
    loss = (
        loss
        + 0.22 * F.binary_cross_entropy(auth_bad_core, z)
        + 0.22 * F.binary_cross_entropy(auth_bad_goal, keep_core)
        + 0.22 * F.binary_cross_entropy(auth_bad_focus, keep_core_goal)
        + 0.14 * F.binary_cross_entropy(auth_bad_all, z)
    )
    return loss


def one(cache, text, device):
    return cache[text].unsqueeze(0).to(device)


@torch.no_grad()
def eval_structure(
    authority,
    root,
    sequences,
    cache,
    device,
    ablation="normal",
):
    authority.eval()

    if ablation == "root_zero":
        eval_root = torch.zeros_like(root)
    elif ablation == "root_shuffle":
        eval_root = torch.roll(root, shifts=1, dims=0)
    elif ablation == "root_negate":
        eval_root = -root
    else:
        eval_root = root

    relation_mask = (1.0, 1.0, 1.0)
    if ablation == "drop_sc":
        relation_mask = (0.0, 1.0, 1.0)
    elif ablation == "drop_cg":
        relation_mask = (1.0, 0.0, 1.0)
    elif ablation == "drop_gf":
        relation_mask = (1.0, 1.0, 0.0)

    total = state_exact = final_exact = whole_exact = 0
    slot_correct = [0, 0, 0]
    first_errors = []
    false = [0, 0, 0]
    false_den = [0, 0, 0]
    true = [0, 0, 0]
    true_den = [0, 0, 0]
    goal_pressure_ok = goal_pressure_n = 0
    core_pressure_ok = core_pressure_n = 0
    after_focus_ok = after_focus_n = 0

    for si, seq in enumerate(sequences):
        pc, pg, pf = seq["initial"]
        tc, tg, tf = seq["initial"]
        first = None
        seq_ok = True
        focus_changes = 0

        other = sequences[(si + 37) % len(sequences)]
        oc, og, of = other["initial"]

        for t, e in enumerate(seq["steps"]):
            override = None
            if ablation == "shuffle_hierarchy":
                override = (
                    one(cache, oc, device),
                    one(cache, og, device),
                    one(cache, of, device),
                )

            _, _, probs = authority(
                eval_root.unsqueeze(0),
                one(cache, pc, device),
                one(cache, pg, device),
                one(cache, pf, device),
                one(cache, e["event"], device),
                one(cache, e["candidate_core"], device),
                one(cache, e["candidate_goal"], device),
                one(cache, e["candidate_focus"], device),
                authority_hierarchy=override,
                relation_mask=relation_mask,
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
            oks = [pc == tc, pg == tg, pf == tf]
            state_exact += int(all(oks))
            for s, ok in enumerate(oks):
                slot_correct[s] += int(ok)
            if not all(oks):
                seq_ok = False
                if first is None:
                    first = t + 1

            if e["kind"] in {"focus", "tempting_focus"}:
                focus_changes += 1
            if e["kind"] in {"goal_change", "goal_done", "core_change"}:
                focus_changes = 0
            if e["kind"] == "goal_pressure":
                goal_pressure_n += 1
                goal_pressure_ok += int(pg == tg)
            if e["kind"] == "core_pressure":
                core_pressure_n += 1
                core_pressure_ok += int(pc == tc)
            if focus_changes >= 3:
                after_focus_n += 1
                after_focus_ok += int(pg == tg)

        final_exact += int((pc, pg, pf) == (tc, tg, tf))
        whole_exact += int(seq_ok)
        first_errors.append(
            first if first is not None else len(seq["steps"]) + 1
        )

    div = lambda a, b: a / b if b else None
    return {
        "state_exact_per_step": div(state_exact, total),
        "core_state_accuracy": div(slot_correct[0], total),
        "goal_state_accuracy": div(slot_correct[1], total),
        "focus_state_accuracy": div(slot_correct[2], total),
        "final_state_exact": div(final_exact, len(sequences)),
        "whole_sequence_exact": div(whole_exact, len(sequences)),
        "mean_steps_until_first_error": statistics.mean(first_errors),
        "core_false_update_rate": div(false[0], false_den[0]),
        "goal_false_update_rate": div(false[1], false_den[1]),
        "focus_false_update_rate": div(false[2], false_den[2]),
        "core_change_recall": div(true[0], true_den[0]),
        "goal_change_recall": div(true[1], true_den[1]),
        "focus_change_recall": div(true[2], true_den[2]),
        "goal_pressure_retention": div(goal_pressure_ok, goal_pressure_n),
        "core_pressure_retention": div(core_pressure_ok, core_pressure_n),
        "goal_retention_after_3plus_focus_changes": div(after_focus_ok, after_focus_n),
    }


@torch.no_grad()
def evaluate_bundle(
    model,
    authority,
    root_proxy,
    root_mode,
    test_sequences,
    test_tokenized,
    device,
):
    was_training = model.training
    model.eval()
    authority.eval()
    cache = build_rep_cache(model, test_tokenized, device)
    root = active_root(model, root_proxy, root_mode).detach().to(device)

    modes = (
        "normal",
        "root_zero",
        "root_shuffle",
        "root_negate",
        "shuffle_hierarchy",
        "drop_sc",
        "drop_cg",
        "drop_gf",
    )
    out = {
        mode: eval_structure(authority, root, test_sequences, cache, device, mode)
        for mode in modes
    }
    base = out["normal"]["state_exact_per_step"]
    out["causal_delta_state_exact"] = {
        mode: out[mode]["state_exact_per_step"] - base
        for mode in modes
        if mode != "normal"
    }
    if was_training:
        model.train()
        authority.train()
    return out


def save_state(
    path,
    cfg,
    model,
    authority,
    root_proxy,
    optimizer,
    root_mode,
    initial_base_sha,
    initial_model_sha,
    base_params,
    model_params,
    authority_params,
    step,
    seen,
    epoch,
    order_pos,
    first_loss,
    last_loss,
    aux_steps,
    first_aux_loss,
    last_aux_loss,
    elapsed,
    milestones,
    seed,
):
    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "authority": authority.state_dict(),
        "root_proxy": root_proxy.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "model_profile": "5m",
        "variant": VARIANT,
        "root_mode": root_mode,
        "self_rank": 8,
        "projection_rank": 4,
        "anchor_lr_scale": anchor_lr_scale_for_variant(VARIANT),
        "parameter_count": model_params,
        "base_parameter_count": base_params,
        "authority_parameter_count": authority_params,
        "control_root_parameter_count": root_proxy.numel(),
        "initial_base_sha256": initial_base_sha,
        "initial_model_sha256": initial_model_sha,
        "stage": "v0.16j-cgf-40m",
        "steps": step,
        "tokens_seen": seen,
        "data_epoch": epoch,
        "data_order_pos": order_pos,
        "seed": seed,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "aux_steps": aux_steps,
        "first_aux_loss": first_aux_loss,
        "last_aux_loss": last_aux_loss,
        "total_elapsed_seconds": elapsed,
        "milestones": milestones,
        "self_diagnostics": model.self_diagnostics(),
        "active_root_norm": float(
            active_root(model, root_proxy, root_mode).detach().norm().cpu()
        ),
        "root_proxy_sha256": tensor_sha(root_proxy),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--root-mode", choices=sorted(ROOT_MODES), required=True)
    ap.add_argument("--target-tokens", type=int, required=True)
    ap.add_argument("--resume")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--structure-every", type=int, default=16)
    ap.add_argument("--structure-batch", type=int, default=8)
    ap.add_argument("--structure-weight", type=float, default=0.18)
    ap.add_argument("--milestone-every", type=int, default=4_000_000)
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
        if ckpt.get("variant") != VARIANT:
            raise ValueError("resume variant mismatch")
        if ckpt.get("root_mode") != args.root_mode:
            raise ValueError("resume root mode mismatch")
        cfg = ModelConfig(**ckpt["config"])
        base, model = make_model(cfg, VARIANT, 8, 4)
        authority = JointCgfAuthority(cfg.d_model)
        root_proxy = nn.Parameter(torch.empty(cfg.d_model))
        model.load_state_dict(ckpt["model"], strict=True)
        authority.load_state_dict(ckpt["authority"], strict=True)
        with torch.no_grad():
            root_proxy.copy_(ckpt["root_proxy"])
        initial_base_sha = ckpt["initial_base_sha256"]
        initial_model_sha = ckpt["initial_model_sha256"]
        base_params = int(ckpt["base_parameter_count"])
        model_params = int(ckpt["parameter_count"])
        authority_params = int(ckpt["authority_parameter_count"])
        step = int(ckpt["steps"])
        seen = int(ckpt["tokens_seen"])
        epoch = int(ckpt.get("data_epoch", 1))
        order_pos = int(ckpt.get("data_order_pos", 0))
        first_loss = ckpt.get("first_loss")
        last_loss = ckpt.get("last_loss")
        aux_steps = int(ckpt.get("aux_steps", 0))
        first_aux_loss = ckpt.get("first_aux_loss")
        last_aux_loss = ckpt.get("last_aux_loss")
        prior_elapsed = float(ckpt.get("total_elapsed_seconds", 0.0))
        milestones = list(ckpt.get("milestones", []))
    else:
        cfg = ModelConfig.hebrew_bpe_5m(tok.vocab_size)
        cfg.max_seq_len = max(cfg.max_seq_len, args.seq_len)
        base, model = make_model(cfg, VARIANT, 8, 4)
        authority = JointCgfAuthority(cfg.d_model)
        root_proxy = nn.Parameter(torch.empty(cfg.d_model))
        nn.init.normal_(root_proxy, mean=0.0, std=0.02)
        initial_base_sha = fingerprint(base)
        initial_model_sha = fingerprint(model)
        base_params = sum(p.numel() for p in base.parameters())
        model_params = sum(p.numel() for p in model.parameters())
        authority_params = sum(p.numel() for p in authority.parameters())
        step = seen = aux_steps = 0
        epoch = 1
        order_pos = 0
        first_loss = last_loss = None
        first_aux_loss = last_aux_loss = None
        prior_elapsed = 0.0
        milestones = []

    if seen >= args.target_tokens:
        raise ValueError(
            f"checkpoint already at {seen:,}; target={args.target_tokens:,}"
        )

    model = model.to(args.device)
    authority = authority.to(args.device)
    root_proxy = nn.Parameter(root_proxy.detach().to(args.device))
    opt = make_optimizer(model, authority, root_proxy, args.lr)
    if args.resume:
        opt.load_state_dict(ckpt["optimizer"])
        optimizer_to_device(opt, args.device)

    train_sequences = make_hard_sequences(
        420, 20, 34, 7101,
        TRAIN_CORES, TRAIN_GOALS, TRAIN_FOCI, TRAIN_TEMPLATES,
    )
    test_sequences = make_hard_sequences(
        220, 38, 56, 11103,
        TEST_CORES, TEST_GOALS, TEST_FOCI, TEST_TEMPLATES,
    )
    train_rows = [e for seq in train_sequences for e in seq["steps"]]
    buckets = build_kind_buckets(train_rows)
    train_tokenized = tokenize_texts(tok, train_sequences)
    test_tokenized = tokenize_texts(tok, test_sequences)

    tokens = load_tokens(args.tokens)
    width = args.seq_len + 1
    usable = tokens.numel() - tokens.numel() % width
    blocks = tokens[:usable].view(-1, width)
    n_blocks = blocks.shape[0]
    if n_blocks < args.batch_size:
        raise RuntimeError("token file too small")
    neutral = PersonalState("neutral").flatten().to(args.device)

    struct_rng = random.Random(args.seed + 9091 + aux_steps)
    start_seen = seen
    start_step = step
    next_mark = (
        milestones[-1]["scheduled_token_mark"] + args.milestone_every
        if milestones
        else args.milestone_every
    )

    print(
        json.dumps(
            {
                "event": "start",
                "experiment": "v0.16j CGF 40M causal hierarchy",
                "root_mode": args.root_mode,
                "resume": bool(args.resume),
                "start_tokens": seen,
                "target_tokens": args.target_tokens,
                "next_milestone": next_mark,
                "variant": VARIANT,
                "initial_model_sha256": initial_model_sha,
                "root_proxy_sha256": tensor_sha(root_proxy),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

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
                rows = sample_structural_rows(
                    buckets, train_rows, struct_rng, args.structure_batch
                )
                wrong = choose_wrong_rows(rows, train_rows, struct_rng)
                aux_loss = structural_loss(
                    model,
                    authority,
                    root_proxy,
                    args.root_mode,
                    train_tokenized,
                    rows,
                    wrong,
                    args.device,
                )
                loss = loss + args.structure_weight * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters())
                + list(authority.parameters())
                + [root_proxy],
                1.0,
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
                elapsed = time.time() - started
                print(
                    f"root={args.root_mode} step={step} tokens={seen:,} "
                    f"lm_loss={last_loss:.4f} aux_steps={aux_steps} "
                    f"aux_loss={last_aux_loss} phase_tok_s="
                    f"{(seen-start_seen)/max(elapsed,1):.1f}",
                    flush=True,
                )

            while seen >= next_mark and next_mark <= 40_000_000:
                bundle = evaluate_bundle(
                    model,
                    authority,
                    root_proxy,
                    args.root_mode,
                    test_sequences,
                    test_tokenized,
                    args.device,
                )
                milestone = {
                    "scheduled_token_mark": next_mark,
                    "tokens_seen": seen,
                    "step": step,
                    "lm_loss": last_loss,
                    "aux_loss": last_aux_loss,
                    "aux_steps": aux_steps,
                    "normal": bundle["normal"],
                    "causal_delta_state_exact": bundle["causal_delta_state_exact"],
                    "ablations": {
                        k: v for k, v in bundle.items()
                        if k not in {"normal", "causal_delta_state_exact"}
                    },
                    "active_root_norm": float(
                        active_root(
                            model, root_proxy, args.root_mode
                        ).detach().norm().cpu()
                    ),
                    "self_anchor_norm": float(
                        model.self_anchor.detach().norm().cpu()
                    ),
                    "root_proxy_norm": float(
                        root_proxy.detach().norm().cpu()
                    ),
                }
                milestones.append(milestone)
                print(
                    json.dumps(
                        {
                            "event": "milestone",
                            "root_mode": args.root_mode,
                            "mark": next_mark,
                            "tokens_seen": seen,
                            "state_exact": bundle["normal"]["state_exact_per_step"],
                            "goal_false_update_rate": bundle["normal"]["goal_false_update_rate"],
                            "goal_change_recall": bundle["normal"]["goal_change_recall"],
                            "causal_delta_state_exact": bundle["causal_delta_state_exact"],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                next_mark += args.milestone_every
                model.train()
                authority.train()

        if order_pos >= n_blocks:
            epoch += 1
            order_pos = 0

    chunk_elapsed = time.time() - started
    total_elapsed = prior_elapsed + chunk_elapsed

    save_state(
        args.save,
        cfg,
        model,
        authority,
        root_proxy,
        opt,
        args.root_mode,
        initial_base_sha,
        initial_model_sha,
        base_params,
        model_params,
        authority_params,
        step,
        seen,
        epoch,
        order_pos,
        first_loss,
        last_loss,
        aux_steps,
        first_aux_loss,
        last_aux_loss,
        total_elapsed,
        milestones,
        args.seed,
    )

    report = {
        "experiment": "v0.16j independent CORE/GOAL/FOCUS 40M",
        "design": {
            "variant": VARIANT,
            "root_mode": args.root_mode,
            "matched_control_root_parameters": int(root_proxy.numel()),
            "five_phases": True,
            "phase_size_tokens": 8_000_000,
            "milestone_every_tokens": args.milestone_every,
            "milestones_total_at_40m": 10,
            "counterfactual_edge_training": True,
            "goal_false_update_asymmetric_penalty": True,
            "causal_ablations": [
                "root_zero",
                "root_shuffle",
                "root_negate",
                "shuffle_hierarchy",
                "drop_sc",
                "drop_cg",
                "drop_gf",
            ],
        },
        "root_mode": args.root_mode,
        "variant": VARIANT,
        "start_tokens": start_seen,
        "target_tokens": args.target_tokens,
        "tokens_seen": seen,
        "chunk_steps": step - start_step,
        "steps": step,
        "aux_steps": aux_steps,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "first_aux_loss": first_aux_loss,
        "last_aux_loss": last_aux_loss,
        "chunk_elapsed_seconds": chunk_elapsed,
        "total_elapsed_seconds": total_elapsed,
        "chunk_tokens_per_second": (seen - start_seen) / max(chunk_elapsed, 1),
        "initial_base_sha256": initial_base_sha,
        "initial_model_sha256": initial_model_sha,
        "self_diagnostics": model.self_diagnostics(),
        "active_root_norm": float(
            active_root(model, root_proxy, args.root_mode).detach().norm().cpu()
        ),
        "root_proxy_sha256": tensor_sha(root_proxy),
        "milestones": milestones,
        "latest_milestone": milestones[-1] if milestones else None,
        "tokenizer_kind": kind,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
