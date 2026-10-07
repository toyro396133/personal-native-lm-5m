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
from joint_cgf_model import proposal_target
from dual_reference_cgf_model import DualReferenceCgfAuthority
from train_joint_cgf_15m import (
    fingerprint,
    tokenize_texts,
    encode_text_batch,
    stack,
    build_rep_cache,
)


VARIANT = "diff_anchor"
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


def optimizer_to_device(opt, device):
    for state in opt.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)


def active_root(model, sham_root, root_mode):
    return model.self_anchor if root_mode == "self" else sham_root


def make_optimizer(model, authority, sham_root, decoy_root, lr):
    anchor_id = id(model.self_anchor)
    main = [p for p in model.parameters() if id(p) != anchor_id]
    main += list(authority.parameters()) + [sham_root, decoy_root]
    scale = anchor_lr_scale_for_variant(VARIANT)
    return torch.optim.AdamW(
        [
            {"params": main, "lr": lr, "weight_decay": 0.1},
            {"params": [model.self_anchor], "lr": lr * scale, "weight_decay": 0.1},
        ]
    )


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
        for _ in range(64):
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
    sham_root,
    decoy_root,
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
    wc, wg, _wf = [
        stack(rep, wrong_rows, k) for k in ("core", "goal", "focus")
    ]

    root_vec = active_root(model, sham_root, root_mode)
    root = root_vec.unsqueeze(0).expand(len(rows), -1)
    decoy = decoy_root.unsqueeze(0).expand(len(rows), -1)

    proposal, auth, final = authority(
        root, core, goal, focus, event, cc, cg, cf
    )
    y = torch.tensor(
        [r["target"] for r in rows], dtype=torch.float32, device=device
    )
    y_prop = torch.stack([
        proposal_target(r["kind"], r["target"], device) for r in rows
    ])

    loss = (
        0.55 * F.binary_cross_entropy(proposal, y_prop)
        + 1.0 * asymmetric_bce(auth, y)
        + 1.55 * asymmetric_bce(final, y)
    )

    # Counterfactuals encode the philosophy directly:
    # - wrong SELF does NOT invalidate CORE, but must invalidate GOAL/FOCUS;
    # - wrong CORE reference does NOT alter CORE authority itself, but invalidates
    #   the SELF↔CORE basis of GOAL and therefore FOCUS;
    # - wrong GOAL reference invalidates only FOCUS.
    _, bad_root, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        reference_root=decoy,
    )
    _, bad_core, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        reference_core=wc,
    )
    _, bad_goal, _ = authority(
        root, core, goal, focus, event, cc, cg, cf,
        reference_goal=wg,
    )

    z = torch.zeros_like(y)
    keep_core = torch.stack((y[:, 0], z[:, 1], z[:, 2]), dim=-1)
    keep_core_goal = torch.stack((y[:, 0], y[:, 1], z[:, 2]), dim=-1)

    loss = (
        loss
        + 0.28 * F.binary_cross_entropy(bad_root, keep_core)
        + 0.28 * F.binary_cross_entropy(bad_core, keep_core)
        + 0.24 * F.binary_cross_entropy(bad_goal, keep_core_goal)
        # Explicit invariance: SELF substitutions must not move CORE authority.
        + 0.18 * F.mse_loss(bad_root[:, 0], auth[:, 0])
    )
    return loss


def one(cache, text, device):
    return cache[text].unsqueeze(0).to(device)


@torch.no_grad()
def eval_structure(
    authority,
    root,
    decoy_root,
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
    elif ablation == "root_decoy":
        eval_root = decoy_root
    else:
        eval_root = root

    mask = (1.0, 1.0, 1.0)
    if ablation == "drop_self_core":
        mask = (0.0, 1.0, 1.0)
    elif ablation == "drop_core_goal":
        mask = (1.0, 0.0, 1.0)
    elif ablation == "drop_goal_focus":
        mask = (1.0, 1.0, 0.0)

    total = state_exact = final_exact = whole_exact = 0
    slot_correct = [0, 0, 0]
    first_errors = []
    false = [0, 0, 0]
    false_den = [0, 0, 0]
    true = [0, 0, 0]
    true_den = [0, 0, 0]

    for si, seq in enumerate(sequences):
        pc, pg, pf = seq["initial"]
        tc, tg, tf = seq["initial"]
        seq_ok = True
        first = None
        other = sequences[(si + 37) % len(sequences)]
        oc, og, _of = other["initial"]

        for t, e in enumerate(seq["steps"]):
            ref_core = None
            ref_goal = None
            if ablation == "shuffle_core_reference":
                ref_core = one(cache, oc, device)
            if ablation == "shuffle_goal_reference":
                ref_goal = one(cache, og, device)

            _, _, probs = authority(
                eval_root.unsqueeze(0),
                one(cache, pc, device),
                one(cache, pg, device),
                one(cache, pf, device),
                one(cache, e["event"], device),
                one(cache, e["candidate_core"], device),
                one(cache, e["candidate_goal"], device),
                one(cache, e["candidate_focus"], device),
                reference_core=ref_core,
                reference_goal=ref_goal,
                relation_mask=mask,
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

        final_exact += int((pc, pg, pf) == (tc, tg, tf))
        whole_exact += int(seq_ok)
        first_errors.append(first if first is not None else len(seq["steps"]) + 1)

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
    }


@torch.no_grad()
def evaluate_bundle(
    model,
    authority,
    sham_root,
    decoy_root,
    root_mode,
    test_sequences,
    test_tokenized,
    device,
):
    was_training = model.training
    model.eval()
    authority.eval()
    cache = build_rep_cache(model, test_tokenized, device)
    root = active_root(model, sham_root, root_mode).detach().to(device)
    decoy = decoy_root.detach().to(device)

    modes = (
        "normal",
        "root_zero",
        "root_shuffle",
        "root_negate",
        "root_decoy",
        "drop_self_core",
        "drop_core_goal",
        "drop_goal_focus",
        "shuffle_core_reference",
        "shuffle_goal_reference",
    )
    out = {
        mode: eval_structure(
            authority, root, decoy, test_sequences, cache, device, mode
        )
        for mode in modes
    }
    base = out["normal"]
    out["causal_delta"] = {
        mode: {
            "state_exact": out[mode]["state_exact_per_step"] - base["state_exact_per_step"],
            "core_accuracy": out[mode]["core_state_accuracy"] - base["core_state_accuracy"],
            "goal_accuracy": out[mode]["goal_state_accuracy"] - base["goal_state_accuracy"],
            "focus_accuracy": out[mode]["focus_state_accuracy"] - base["focus_state_accuracy"],
        }
        for mode in modes if mode != "normal"
    }
    if was_training:
        model.train()
        authority.train()
    return out


def save_checkpoint(
    path,
    cfg,
    model,
    authority,
    sham_root,
    decoy_root,
    optimizer,
    root_mode,
    source_tokens,
    new_seen,
    step,
    epoch,
    order_pos,
    aux_steps,
    milestones,
    seed,
    initial_model_sha,
):
    payload = {
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "authority": authority.state_dict(),
        "sham_root": sham_root.detach().cpu(),
        "decoy_root": decoy_root.detach().cpu(),
        "optimizer": optimizer.state_dict(),
        "variant": VARIANT,
        "root_mode": root_mode,
        "self_rank": 8,
        "projection_rank": 4,
        "anchor_lr_scale": anchor_lr_scale_for_variant(VARIANT),
        "stage": "v0.16k-dual-reference",
        "source_language_tokens": source_tokens,
        "new_language_tokens": new_seen,
        "tokens_seen": source_tokens + new_seen,
        "steps": step,
        "data_epoch": epoch,
        "data_order_pos": order_pos,
        "aux_steps": aux_steps,
        "milestones": milestones,
        "seed": seed,
        "initial_model_sha256": initial_model_sha,
        "self_diagnostics": model.self_diagnostics(),
        "active_root_norm": float(active_root(model, sham_root, root_mode).detach().norm().cpu()),
        "sham_root_sha256": tensor_sha(sham_root),
        "decoy_root_sha256": tensor_sha(decoy_root),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--root-mode", choices=sorted(ROOT_MODES), required=True)
    ap.add_argument("--target-new-tokens", type=int, required=True)
    ap.add_argument("--source-checkpoint")
    ap.add_argument("--resume")
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--lr", type=float, default=2.8e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--structure-every", type=int, default=16)
    ap.add_argument("--structure-batch", type=int, default=8)
    ap.add_argument("--structure-weight", type=float, default=0.18)
    ap.add_argument("--milestone-every", type=int, default=2_000_000)
    ap.add_argument("--save", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if bool(args.source_checkpoint) == bool(args.resume):
        raise SystemExit("provide exactly one of --source-checkpoint or --resume")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    started = time.time()
    tok, kind = load_tokenizer(args.tokenizer)

    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt.get("variant") != VARIANT or ckpt.get("root_mode") != args.root_mode:
            raise ValueError("resume identity mismatch")
        cfg = ModelConfig(**ckpt["config"])
        _base, model = make_model(cfg, VARIANT, 8, 4)
        authority = DualReferenceCgfAuthority(cfg.d_model)
        sham_root = nn.Parameter(torch.empty(cfg.d_model))
        decoy_root = nn.Parameter(torch.empty(cfg.d_model))
        model.load_state_dict(ckpt["model"], strict=True)
        authority.load_state_dict(ckpt["authority"], strict=True)
        with torch.no_grad():
            sham_root.copy_(ckpt["sham_root"])
            decoy_root.copy_(ckpt["decoy_root"])
        source_tokens = int(ckpt["source_language_tokens"])
        new_seen = int(ckpt["new_language_tokens"])
        step = int(ckpt["steps"])
        epoch = int(ckpt.get("data_epoch", 1))
        order_pos = int(ckpt.get("data_order_pos", 0))
        aux_steps = int(ckpt.get("aux_steps", 0))
        milestones = list(ckpt.get("milestones", []))
        initial_model_sha = ckpt["initial_model_sha256"]
    else:
        source = torch.load(args.source_checkpoint, map_location="cpu", weights_only=False)
        if source.get("variant") != VARIANT:
            raise ValueError(f"source must be {VARIANT}, got {source.get('variant')}")
        cfg = ModelConfig(**source["config"])
        _base, model = make_model(cfg, VARIANT, 8, 4)
        model.load_state_dict(source["model"], strict=True)
        source_tokens = int(source["tokens_seen"])
        if source_tokens < 100_000_000:
            raise ValueError(f"source checkpoint is only {source_tokens:,} tokens")
        authority = DualReferenceCgfAuthority(cfg.d_model)
        sham_root = nn.Parameter(torch.empty(cfg.d_model))
        decoy_root = nn.Parameter(torch.empty(cfg.d_model))
        nn.init.normal_(sham_root, mean=0.0, std=0.02)
        nn.init.normal_(decoy_root, mean=0.0, std=0.02)
        new_seen = step = aux_steps = 0
        epoch = 1
        order_pos = 0
        milestones = []
        initial_model_sha = fingerprint(model)

    if new_seen >= args.target_new_tokens:
        raise ValueError("target already reached")

    model = model.to(args.device)
    authority = authority.to(args.device)
    sham_root = nn.Parameter(sham_root.detach().to(args.device))
    decoy_root = nn.Parameter(decoy_root.detach().to(args.device))
    opt = make_optimizer(model, authority, sham_root, decoy_root, args.lr)
    if args.resume:
        opt.load_state_dict(ckpt["optimizer"])
        optimizer_to_device(opt, args.device)

    train_sequences = make_hard_sequences(
        420, 20, 34, 17101,
        TRAIN_CORES, TRAIN_GOALS, TRAIN_FOCI, TRAIN_TEMPLATES,
    )
    test_sequences = make_hard_sequences(
        240, 38, 56, 21103,
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
    struct_rng = random.Random(args.seed + 19091 + aux_steps)

    next_mark = (
        milestones[-1]["scheduled_new_token_mark"] + args.milestone_every
        if milestones else args.milestone_every
    )
    start_new = new_seen

    print(json.dumps({
        "event": "start",
        "experiment": "v0.16k dual-reference SELF+CORE -> GOAL -> FOCUS",
        "root_mode": args.root_mode,
        "source_language_tokens": source_tokens,
        "start_new_tokens": new_seen,
        "target_new_tokens": args.target_new_tokens,
        "topology": "INPUT->CORE; (SELF<->CORE)->GOAL; GOAL->FOCUS",
        "initial_model_sha256": initial_model_sha,
    }, indent=2), flush=True)

    model.train()
    authority.train()
    while new_seen < args.target_new_tokens:
        g = torch.Generator(device="cpu")
        g.manual_seed(args.seed + 1000 * epoch)
        order = torch.randperm(n_blocks, generator=g)

        while order_pos < n_blocks and new_seen < args.target_new_tokens:
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
                    model, authority, sham_root, decoy_root, args.root_mode,
                    train_tokenized, rows, wrong, args.device,
                )
                loss = loss + args.structure_weight * aux_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(authority.parameters())
                + [sham_root, decoy_root], 1.0
            )
            opt.step()

            step += 1
            new_seen += int(y.numel())

            if step % 250 == 0:
                print(
                    f"root={args.root_mode} step={step} "
                    f"new_tokens={new_seen:,} total_tokens={source_tokens+new_seen:,} "
                    f"lm_loss={float(lm_loss.detach()):.4f} aux_steps={aux_steps} "
                    f"aux_loss={None if aux_loss is None else float(aux_loss.detach()):}",
                    flush=True,
                )

            while (
                new_seen >= next_mark
                and next_mark <= args.target_new_tokens
            ):
                bundle = evaluate_bundle(
                    model, authority, sham_root, decoy_root, args.root_mode,
                    test_sequences, test_tokenized, args.device,
                )
                milestone = {
                    "scheduled_new_token_mark": next_mark,
                    "total_language_token_mark": source_tokens + next_mark,
                    "actual_new_tokens": new_seen,
                    "normal": bundle["normal"],
                    "causal_delta": bundle["causal_delta"],
                    "ablations": {
                        k: v for k, v in bundle.items()
                        if k not in {"normal", "causal_delta"}
                    },
                }
                milestones.append(milestone)
                print(json.dumps({
                    "event": "milestone",
                    "root_mode": args.root_mode,
                    "new_mark": next_mark,
                    "total_mark": source_tokens + next_mark,
                    "normal": bundle["normal"],
                    "causal_delta": bundle["causal_delta"],
                }, ensure_ascii=False), flush=True)
                next_mark += args.milestone_every
                model.train()
                authority.train()

        if order_pos >= n_blocks:
            epoch += 1
            order_pos = 0

    save_checkpoint(
        args.save, cfg, model, authority, sham_root, decoy_root, opt,
        args.root_mode, source_tokens, new_seen, step, epoch, order_pos,
        aux_steps, milestones, args.seed, initial_model_sha,
    )

    report = {
        "experiment": "v0.16k dual-reference SELF+CORE -> GOAL -> FOCUS",
        "root_mode": args.root_mode,
        "variant": VARIANT,
        "source_language_tokens": source_tokens,
        "start_new_tokens": start_new,
        "new_language_tokens": new_seen,
        "total_language_tokens": source_tokens + new_seen,
        "initial_model_sha256": initial_model_sha,
        "topology": {
            "core": "anchored to input/content; authority has no SELF input",
            "goal": "depends on SELF<->CORE and CORE<->GOAL relations",
            "focus": "depends on GOAL<->FOCUS relation",
        },
        "counterfactuals": {
            "wrong_self": "preserve CORE target; invalidate GOAL and FOCUS",
            "wrong_core_reference": "preserve CORE target; invalidate GOAL and FOCUS",
            "wrong_goal_reference": "preserve CORE/GOAL targets; invalidate FOCUS",
        },
        "milestone_every_new_tokens": args.milestone_every,
        "milestones": milestones,
        "latest_milestone": milestones[-1] if milestones else None,
        "self_diagnostics": model.self_diagnostics(),
        "sham_root_sha256": tensor_sha(sham_root),
        "decoy_root_sha256": tensor_sha(decoy_root),
        "tokenizer_kind": kind,
        "elapsed_seconds": time.time() - started,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
