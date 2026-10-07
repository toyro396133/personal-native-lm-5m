from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v17 import load_any
from joint_cgf_model import JointCgfAuthority
from personal_state import PersonalState
from train_hebrew import load_tokenizer
from train_joint_cgf_15m import tokenize_texts
from core_goal_focus_hierarchical_lab import (
    make_hard_sequences,
    TEST_CORES, TEST_GOALS, TEST_FOCI, TEST_TEMPLATES,
)


ANCHOR_MODES = ("normal", "zero", "random", "shuffle", "negate")


def make_anchor(model, mode: str, seed: int = 991):
    a = model.self_anchor.detach().clone()
    if mode == "normal":
        return a
    if mode == "zero":
        return torch.zeros_like(a)
    if mode == "negate":
        return -a
    if mode == "shuffle":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        perm = torch.randperm(a.numel(), generator=g).to(a.device)
        return a[perm]
    if mode == "random":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        r = torch.randn(a.numel(), generator=g, dtype=a.dtype).to(a.device)
        return r / r.norm().clamp_min(1e-8) * a.norm().clamp_min(1e-8)
    raise ValueError(mode)


def encode_ids(model, ids, lengths, anchor, adapters_enabled=True):
    base = model.base
    x = base.token_embedding(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)
    x = x + base.position_embedding(pos)[None, :, :]
    for block, adapter in zip(base.blocks, model.self_adapters):
        x = block(x, None)
        if adapters_enabled:
            x = adapter(x, anchor)
    x = base.final_norm(x)

    mask = torch.arange(ids.shape[1], device=ids.device)[None, :] < lengths[:, None]
    w = mask.to(x.dtype).unsqueeze(-1)
    return (x * w).sum(1) / w.sum(1).clamp_min(1.0)


@torch.no_grad()
def build_cache(model, tok_map, device, anchor, adapters_enabled=True, batch_size=64):
    texts = list(tok_map)
    out = {}
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        seqs = [tok_map[t] for t in chunk]
        max_len = max(len(x) for x in seqs)
        ids = torch.zeros((len(seqs), max_len), dtype=torch.long, device=device)
        lengths = torch.tensor([len(x) for x in seqs], dtype=torch.long, device=device)
        for i, seq in enumerate(seqs):
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        reps = encode_ids(model, ids, lengths, anchor, adapters_enabled=adapters_enabled)
        out.update({t: reps[i].detach().cpu() for i, t in enumerate(chunk)})
    return out


def authority_forward(
    authority,
    root,
    core, goal, focus, event,
    cand_core, cand_goal, cand_focus,
    relation_mode="normal",
    wrong_hierarchy=None,
):
    c = authority.content(core)
    g = authority.content(goal)
    f = authority.content(focus)
    e = authority.content(event)
    proposal = torch.sigmoid(
        authority.proposal(torch.cat((c, g, f, e), dim=-1))
    ).clamp(1e-5, 1 - 1e-5)

    if relation_mode == "hierarchy_shuffle":
        if wrong_hierarchy is None:
            raise ValueError("hierarchy_shuffle requires wrong_hierarchy")
        ac, ag, af = wrong_hierarchy
    else:
        ac, ag, af = core, goal, focus

    current_rel = list(authority.hierarchy(root, ac, ag, af))
    candidate_rel = list(authority.hierarchy(root, cand_core, cand_goal, cand_focus))

    zero_edges = {
        "zero_relations": (0, 1, 2),
        "drop_self_core": (0,),
        "drop_core_goal": (1,),
        "drop_goal_focus": (2,),
    }.get(relation_mode, ())
    for edge in zero_edges:
        current_rel[edge] = torch.zeros_like(current_rel[edge])
        candidate_rel[edge] = torch.zeros_like(candidate_rel[edge])

    candidate = (
        authority._candidate(authority.core_candidate, cand_core),
        authority._candidate(authority.goal_candidate, cand_goal),
        authority._candidate(authority.focus_candidate, cand_focus),
    )
    auth = torch.sigmoid(torch.cat((
        authority.auth_core(e, candidate[0], current_rel[0], candidate_rel[0]),
        authority.auth_goal(e, candidate[1], current_rel[1], candidate_rel[1]),
        authority.auth_focus(e, candidate[2], current_rel[2], candidate_rel[2]),
    ), dim=-1)).clamp(1e-5, 1 - 1e-5)

    local = proposal * auth
    p_core = local[:, 0:1]
    p_goal = 1.0 - (1.0 - p_core) * (1.0 - local[:, 1:2])
    p_focus = 1.0 - (1.0 - p_goal) * (1.0 - local[:, 2:3])
    final = torch.cat((p_core, p_goal, p_focus), dim=-1).clamp(1e-5, 1 - 1e-5)
    return proposal, auth, final


@torch.no_grad()
def eval_structure(authority, sequences, cache, root, device, relation_mode="normal"):
    authority.eval()
    total = exact = final_exact = 0
    first_errors = []
    slot_correct = [0, 0, 0]
    false = [0, 0, 0]
    false_den = [0, 0, 0]
    true = [0, 0, 0]
    true_den = [0, 0, 0]

    def one(text):
        return cache[text].unsqueeze(0).to(device)

    for si, seq in enumerate(sequences):
        pc, pg, pf = seq["initial"]
        tc, tg, tf = seq["initial"]
        first = None

        other = sequences[(si + 37) % len(sequences)]
        oc, og, of = other["initial"]
        wrong = (one(oc), one(og), one(of))

        for t, e in enumerate(seq["steps"]):
            _, _, probs = authority_forward(
                authority,
                root.unsqueeze(0),
                one(pc), one(pg), one(pf), one(e["event"]),
                one(e["candidate_core"]), one(e["candidate_goal"]), one(e["candidate_focus"]),
                relation_mode=relation_mode,
                wrong_hierarchy=wrong,
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
            ok = [pc == tc, pg == tg, pf == tf]
            exact += int(all(ok))
            for s, value in enumerate(ok):
                slot_correct[s] += int(value)
            if not all(ok) and first is None:
                first = t + 1

        final_exact += int((pc, pg, pf) == (tc, tg, tf))
        first_errors.append(first if first is not None else len(seq["steps"]) + 1)

    div = lambda a, b: a / b if b else None
    return {
        "state_exact_per_step": div(exact, total),
        "core_state_accuracy": div(slot_correct[0], total),
        "goal_state_accuracy": div(slot_correct[1], total),
        "focus_state_accuracy": div(slot_correct[2], total),
        "final_state_exact": div(final_exact, len(sequences)),
        "mean_steps_until_first_error": statistics.mean(first_errors),
        "core_false_update_rate": div(false[0], false_den[0]),
        "goal_false_update_rate": div(false[1], false_den[1]),
        "focus_false_update_rate": div(false[2], false_den[2]),
        "core_change_recall": div(true[0], true_den[0]),
        "goal_change_recall": div(true[1], true_den[1]),
        "focus_change_recall": div(true[2], true_den[2]),
    }


@torch.no_grad()
def language_eval(model, tok, text, device, anchor, adapters_enabled=True, seq_len=128):
    ids = tok.encode(text, bos=True, eos=True)
    state = PersonalState("neutral").flatten().unsqueeze(0).to(device)
    total_nll = 0.0
    target_tokens = 0
    base = model.base

    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            continue
        xids = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

        x = base.token_embedding(xids)
        pos = torch.arange(xids.shape[1], device=device)
        x = x + base.position_embedding(pos)[None, :, :]
        for block, adapter in zip(base.blocks, model.self_adapters):
            x = block(x, None)
            if adapters_enabled:
                x = adapter(x, anchor)
        x = base.final_norm(x)
        logits = base.lm_head(x)

        total_nll += float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            y.reshape(-1),
            reduction="sum",
        ))
        target_tokens += y.numel()

    chars = len(text)
    nats = total_nll / max(1, chars)
    return {
        "nats_per_char": nats,
        "bits_per_char": nats / math.log(2),
        "evaluated_target_tokens": target_tokens,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--chars", type=int, default=100000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if ckpt.get("variant") != "self_v1_slow" or ckpt.get("joint_mode") != "joint":
        raise SystemExit("v0.16i expects the self_v1_slow + joint v0.16h checkpoint")

    authority = JointCgfAuthority(cfg.d_model).to(args.device)
    authority.load_state_dict(ckpt["authority"], strict=True)
    authority.eval()
    model.eval()

    tok, _ = load_tokenizer(args.tokenizer)
    test_sequences = make_hard_sequences(
        180, 36, 52, 11103,
        TEST_CORES, TEST_GOALS, TEST_FOCI, TEST_TEMPLATES,
    )
    tok_map = tokenize_texts(tok, test_sequences)
    val_text = Path(args.val).read_text(encoding="utf-8")[:args.chars]

    learned = model.self_anchor.detach().clone()
    conditions = {}

    # Whole-SELF interventions: both LM reference and hierarchy root change together.
    for mode in ANCHOR_MODES:
        anchor = make_anchor(model, mode)
        cache = build_cache(model, tok_map, args.device, anchor, adapters_enabled=True)
        conditions[f"whole_self_{mode}"] = {
            "structure": eval_structure(authority, test_sequences, cache, anchor, args.device),
            "language": language_eval(model, tok, val_text, args.device, anchor, adapters_enabled=True),
        }

    # Remove SELF from LM representations but keep learned SELF as authority root.
    cache_no_adapters = build_cache(
        model, tok_map, args.device, learned, adapters_enabled=False
    )
    conditions["lm_self_adapters_off"] = {
        "structure": eval_structure(
            authority, test_sequences, cache_no_adapters, learned, args.device
        ),
        "language": language_eval(
            model, tok, val_text, args.device, learned, adapters_enabled=False
        ),
    }

    # Keep normal LM representations but remove only SELF root from authority.
    normal_cache = build_cache(
        model, tok_map, args.device, learned, adapters_enabled=True
    )
    zero_root = torch.zeros_like(learned)
    conditions["authority_self_root_zero"] = {
        "structure": eval_structure(
            authority, test_sequences, normal_cache, zero_root, args.device
        )
    }

    # Relation-specific causal interventions with normal SELF/representations.
    for relation_mode in (
        "hierarchy_shuffle",
        "zero_relations",
        "drop_self_core",
        "drop_core_goal",
        "drop_goal_focus",
    ):
        conditions[relation_mode] = {
            "structure": eval_structure(
                authority,
                test_sequences,
                normal_cache,
                learned,
                args.device,
                relation_mode=relation_mode,
            )
        }

    normal_struct = conditions["whole_self_normal"]["structure"]["state_exact_per_step"]
    normal_lang = conditions["whole_self_normal"]["language"]["nats_per_char"]
    summary = {}
    for name, result in conditions.items():
        item = {}
        if "structure" in result:
            item["state_exact_delta_vs_normal"] = (
                result["structure"]["state_exact_per_step"] - normal_struct
            )
        if "language" in result:
            item["language_nats_delta_vs_normal"] = (
                result["language"]["nats_per_char"] - normal_lang
            )
        summary[name] = item

    report = {
        "experiment": "v0.16i causal anatomy of joint SELF winner",
        "source": {
            "variant": ckpt.get("variant"),
            "joint_mode": ckpt.get("joint_mode"),
            "tokens_seen": ckpt.get("tokens_seen"),
            "stage": ckpt.get("stage"),
        },
        "normal_reference": {
            "structure_state_exact_per_step": normal_struct,
            "language_nats_per_char": normal_lang,
        },
        "conditions": conditions,
        "deltas": summary,
        "interpretation_guardrails": [
            "Whole-SELF perturbations test runtime dependence, not whether SELF caused the training advantage.",
            "lm_self_adapters_off isolates SELF use inside LM representations while keeping the authority root.",
            "authority_self_root_zero isolates SELF as hierarchy root while keeping normal LM representations.",
            "Relation-edge ablations test causal dependence on the learned hierarchy.",
        ],
    }
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
