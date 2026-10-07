from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer
from core_goal_focus_sequence_lab import (
    TRAIN_CORES, TEST_CORES, TRAIN_GOALS, TEST_GOALS, TRAIN_FOCI, TEST_FOCI,
    TRAIN_TEMPLATES, TEST_TEMPLATES, build_cache,
)
from core_goal_focus_hierarchical_lab import make_hard_sequences, RelationState


MODES = (
    "event_only",
    "event_candidate",
    "neutral_hierarchy",
    "self_hierarchy",
    "capacity_control",
    "self_hierarchy_counterfactual",
)


def sha_model(model):
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(bytes(t.untyped_storage()))
    return h.hexdigest()


class Projector(nn.Module):
    def __init__(self, d_model, hidden=96):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )

    def forward(self, x):
        return self.net(F.layer_norm(x, (x.shape[-1],)))


class FixedAuthorityHead(nn.Module):
    """
    Identical parameter count in every decomposition arm.
    Inputs: event, candidate unary, current structure, candidate structure,
    and current*candidate structure interaction.
    """
    def __init__(self, content_dim=96, rel_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(content_dim + 4 * rel_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, event, candidate, current_struct, candidate_struct):
        return self.net(torch.cat((
            event,
            candidate,
            current_struct,
            candidate_struct,
            current_struct * candidate_struct,
        ), dim=-1))


class DecompositionController(nn.Module):
    def __init__(self, d_model, mode, content_dim=96, rel_dim=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError(mode)
        self.mode = mode
        self.content_dim = content_dim
        self.rel_dim = rel_dim

        self.content = Projector(d_model, content_dim)

        # Proposal path is identical in every arm.
        self.proposal = nn.Sequential(
            nn.Linear(content_dim * 4, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

        # Structural relation machinery exists in every arm, even when masked,
        # so parameter counts are identical across modes.
        self.self_core = RelationState(d_model, content_dim, rel_dim)
        self.core_goal = RelationState(d_model, content_dim, rel_dim)
        self.goal_focus = RelationState(d_model, content_dim, rel_dim)

        # Neutral root isolates hierarchy from the learned SELF anchor.
        self.neutral_root = nn.Parameter(torch.zeros(d_model))

        # Unary candidate/current descriptors used by candidate and capacity controls.
        self.core_unary = nn.Linear(d_model, rel_dim, bias=False)
        self.goal_unary = nn.Linear(d_model, rel_dim, bias=False)
        self.focus_unary = nn.Linear(d_model, rel_dim, bias=False)

        # Capacity-control features: same dimensionality, no parent-child relation.
        self.core_control = nn.Linear(d_model, rel_dim, bias=False)
        self.goal_control = nn.Linear(d_model, rel_dim, bias=False)
        self.focus_control = nn.Linear(d_model, rel_dim, bias=False)

        self.auth_core = FixedAuthorityHead(content_dim, rel_dim)
        self.auth_goal = FixedAuthorityHead(content_dim, rel_dim)
        self.auth_focus = FixedAuthorityHead(content_dim, rel_dim)

    def _norm(self, x):
        return F.layer_norm(x, (x.shape[-1],))

    def unary(self, which, x):
        layer = {"core": self.core_unary, "goal": self.goal_unary, "focus": self.focus_unary}[which]
        return torch.tanh(layer(self._norm(x)))

    def control(self, which, x):
        layer = {"core": self.core_control, "goal": self.goal_control, "focus": self.focus_control}[which]
        return torch.tanh(layer(self._norm(x)))

    def proposal_probs(self, core, goal, focus, event):
        c, g, f, e = map(self.content, (core, goal, focus, event))
        return torch.sigmoid(self.proposal(torch.cat((c, g, f, e), dim=-1))), e

    def hierarchy_relations(self, root, core, goal, focus, use_self):
        core_parent = root if use_self else self.neutral_root.view(1, -1).expand_as(root)
        return (
            self.self_core.initialize(core_parent, core),
            self.core_goal.initialize(core, goal),
            self.goal_focus.initialize(goal, focus),
        )

    def authority_probs(
        self, root,
        core, goal, focus,
        cand_core, cand_goal, cand_focus,
        event_content,
        wrong_hierarchy=None,
    ):
        z = torch.zeros(core.shape[0], self.rel_dim, device=core.device, dtype=core.dtype)

        cand_unary = (
            self.unary("core", cand_core),
            self.unary("goal", cand_goal),
            self.unary("focus", cand_focus),
        )

        if self.mode == "event_only":
            cand = (z, z, z)
            cur_struct = (z, z, z)
            cand_struct = (z, z, z)

        elif self.mode == "event_candidate":
            cand = cand_unary
            cur_struct = (z, z, z)
            cand_struct = (z, z, z)

        elif self.mode in {"neutral_hierarchy", "self_hierarchy", "self_hierarchy_counterfactual"}:
            cand = cand_unary
            use_self = self.mode != "neutral_hierarchy"
            if wrong_hierarchy is None:
                cur_struct = self.hierarchy_relations(root, core, goal, focus, use_self)
            else:
                wc, wg, wf = wrong_hierarchy
                cur_struct = self.hierarchy_relations(root, wc, wg, wf, use_self)
            cand_struct = self.hierarchy_relations(
                root, cand_core, cand_goal, cand_focus, use_self
            )

        elif self.mode == "capacity_control":
            cand = cand_unary
            cur_struct = (
                self.control("core", core),
                self.control("goal", goal),
                self.control("focus", focus),
            )
            cand_struct = (
                self.control("core", cand_core),
                self.control("goal", cand_goal),
                self.control("focus", cand_focus),
            )
        else:
            raise ValueError(self.mode)

        logits = torch.cat((
            self.auth_core(event_content, cand[0], cur_struct[0], cand_struct[0]),
            self.auth_goal(event_content, cand[1], cur_struct[1], cand_struct[1]),
            self.auth_focus(event_content, cand[2], cur_struct[2], cand_struct[2]),
        ), dim=-1)
        return torch.sigmoid(logits).clamp(1e-5, 1 - 1e-5)

    def forward(
        self, root, core, goal, focus, event,
        cand_core, cand_goal, cand_focus,
        wrong_hierarchy=None,
    ):
        proposal, e = self.proposal_probs(core, goal, focus, event)
        authority = self.authority_probs(
            root, core, goal, focus,
            cand_core, cand_goal, cand_focus,
            e, wrong_hierarchy=wrong_hierarchy,
        )
        final = (proposal * authority).clamp(1e-5, 1 - 1e-5)

        # Hierarchical reset semantics are identical in every authority arm.
        p_core = final[:, 0:1]
        p_goal = 1 - (1 - p_core) * (1 - final[:, 1:2])
        p_focus = 1 - (1 - p_goal) * (1 - final[:, 2:3])
        return proposal, authority, torch.cat((p_core, p_goal, p_focus), dim=-1).clamp(1e-5, 1 - 1e-5)


def proposal_target(kind, target, device):
    # Pressure events propose a real high-level change but must fail authority.
    if kind == "goal_pressure":
        x = [0., 1., 1.]
    elif kind == "core_pressure":
        x = [1., 1., 1.]
    else:
        x = list(target)
    return torch.tensor([x], dtype=torch.float32, device=device)


def vec(cache, text, device):
    return cache[text].unsqueeze(0).to(device)


def train_one(ctl, sequences, cache, root, device, epochs, lr, seed):
    opt = torch.optim.AdamW(ctl.parameters(), lr=lr, weight_decay=0.01)
    rng = random.Random(seed)

    for epoch in range(epochs):
        order = list(range(len(sequences)))
        rng.shuffle(order)
        ctl.train()

        for pos, idx in enumerate(order):
            seq = sequences[idx]
            core, goal, focus = seq["initial"]
            cv, gv, fv = vec(cache, core, device), vec(cache, goal, device), vec(cache, focus, device)
            losses = []

            # Same deterministic negative hierarchy for the counterfactual arm.
            other = sequences[order[(pos + 1) % len(order)]]
            oc, og, of = other["initial"]
            wrong = (vec(cache, oc, device), vec(cache, og, device), vec(cache, of, device))

            for e in seq["steps"]:
                ev = vec(cache, e["event"], device)
                ccv = vec(cache, e["candidate_core"], device)
                cgv = vec(cache, e["candidate_goal"], device)
                cfv = vec(cache, e["candidate_focus"], device)

                proposal, authority, final = ctl(
                    root.unsqueeze(0), cv, gv, fv, ev, ccv, cgv, cfv
                )
                y_final = torch.tensor([e["target"]], dtype=torch.float32, device=device)
                y_prop = proposal_target(e["kind"], e["target"], device)
                y_auth = y_final

                # Identical decomposition supervision across all six arms.
                loss = (
                    0.8 * F.binary_cross_entropy(proposal, y_prop)
                    + 1.2 * F.binary_cross_entropy(authority, y_auth)
                    + 1.5 * F.binary_cross_entropy(final, y_final)
                )

                # Only the explicit counterfactual arm gets this extra factor.
                if ctl.mode == "self_hierarchy_counterfactual":
                    _, bad_auth, _ = ctl(
                        root.unsqueeze(0), cv, gv, fv, ev, ccv, cgv, cfv,
                        wrong_hierarchy=wrong,
                    )
                    loss = loss + 0.6 * F.binary_cross_entropy(
                        bad_auth, torch.zeros_like(bad_auth)
                    )

                losses.append(loss)

                # Teacher-forced state for training; evaluation is closed loop.
                cv = vec(cache, e["next_core"], device)
                gv = vec(cache, e["next_goal"], device)
                fv = vec(cache, e["next_focus"], device)

            seq_loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(), 1.0)
            opt.step()

    return ctl


@torch.no_grad()
def evaluate(ctl, sequences, cache, root, device, perturb=None):
    ctl.eval()
    total = state_exact = final_exact = whole_exact = 0
    first_errors = []
    slot_correct = [0, 0, 0]
    false_updates = [0, 0, 0]
    false_den = [0, 0, 0]
    true_updates = [0, 0, 0]
    true_den = [0, 0, 0]

    for si, seq in enumerate(sequences):
        pc, pg, pf = seq["initial"]
        tc, tg, tf = seq["initial"]
        cv, gv, fv = vec(cache, pc, device), vec(cache, pg, device), vec(cache, pf, device)
        seq_ok = True
        first_error = None

        for t, e in enumerate(seq["steps"]):
            ev = vec(cache, e["event"], device)
            ccv = vec(cache, e["candidate_core"], device)
            cgv = vec(cache, e["candidate_goal"], device)
            cfv = vec(cache, e["candidate_focus"], device)

            wrong = None
            if perturb == "shuffle_hierarchy" and ctl.mode in {"neutral_hierarchy", "self_hierarchy", "self_hierarchy_counterfactual"}:
                other = sequences[(si + 37) % len(sequences)]
                oc, og, of = other["initial"]
                wrong = (vec(cache, oc, device), vec(cache, og, device), vec(cache, of, device))

            _, _, probs = ctl(
                root.unsqueeze(0), cv, gv, fv, ev, ccv, cgv, cfv,
                wrong_hierarchy=wrong,
            )
            bits = (probs[0] >= 0.5).float()
            truth = torch.tensor(e["target"], dtype=torch.float32, device=device)

            for s in range(3):
                if truth[s] > 0.5:
                    true_den[s] += 1
                    true_updates[s] += int(bits[s].item())
                else:
                    false_den[s] += 1
                    false_updates[s] += int(bits[s].item())

            pc = e["candidate_core"] if bits[0] else pc
            pg = e["candidate_goal"] if bits[1] else pg
            pf = e["candidate_focus"] if bits[2] else pf
            cv, gv, fv = vec(cache, pc, device), vec(cache, pg, device), vec(cache, pf, device)

            tc, tg, tf = e["next_core"], e["next_goal"], e["next_focus"]
            total += 1
            ok = [pc == tc, pg == tg, pf == tf]
            state_exact += int(all(ok))
            for s, x in enumerate(ok):
                slot_correct[s] += int(x)
            if not all(ok):
                seq_ok = False
                if first_error is None:
                    first_error = t + 1

        final_exact += int((pc, pg, pf) == (tc, tg, tf))
        whole_exact += int(seq_ok)
        first_errors.append(first_error if first_error is not None else len(seq["steps"]) + 1)

    div = lambda a, b: a / b if b else None
    return {
        "state_exact_per_step": div(state_exact, total),
        "core_state_accuracy": div(slot_correct[0], total),
        "goal_state_accuracy": div(slot_correct[1], total),
        "focus_state_accuracy": div(slot_correct[2], total),
        "final_state_exact": div(final_exact, len(sequences)),
        "whole_sequence_exact": div(whole_exact, len(sequences)),
        "mean_steps_until_first_error": statistics.mean(first_errors),
        "core_false_update_rate": div(false_updates[0], false_den[0]),
        "goal_false_update_rate": div(false_updates[1], false_den[1]),
        "focus_false_update_rate": div(false_updates[2], false_den[2]),
        "core_change_recall": div(true_updates[0], true_den[0]),
        "goal_change_recall": div(true_updates[1], true_den[1]),
        "focus_change_recall": div(true_updates[2], true_den[2]),
    }


def aggregate(runs):
    keys = list(runs[0]["normal"].keys())
    out = {}
    for key in keys:
        vals = [r["normal"][key] for r in runs if r["normal"][key] is not None]
        out[key] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--mode", choices=MODES, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-sequences", type=int, default=320)
    ap.add_argument("--test-sequences", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.0012)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if ckpt.get("variant") != "projected_diff":
        raise SystemExit(f"v0.16g expects projected_diff 15M checkpoint, got {ckpt.get('variant')}")

    tok, _ = load_tokenizer(args.tokenizer)
    before = sha_model(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    train = make_hard_sequences(
        args.train_sequences, 18, 30, 7101,
        TRAIN_CORES, TRAIN_GOALS, TRAIN_FOCI, TRAIN_TEMPLATES,
    )
    test = make_hard_sequences(
        args.test_sequences, 36, 52, 11103,
        TEST_CORES, TEST_GOALS, TEST_FOCI, TEST_TEMPLATES,
    )
    cache = build_cache(model, tok, train + test, args.device)

    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("projected_diff checkpoint did not load as SELF variant")
    root = model.self_anchor.detach().clone().to(args.device)

    runs = []
    for seed in (151, 302, 453):
        print(f"[train] mode={args.mode} seed={seed}", flush=True)
        torch.manual_seed(seed)
        ctl = DecompositionController(cfg.d_model, args.mode).to(args.device)
        train_one(ctl, train, cache, root, args.device, args.epochs, args.lr, seed)
        r = {
            "seed": seed,
            "normal": evaluate(ctl, test, cache, root, args.device),
            "parameters": sum(p.numel() for p in ctl.parameters()),
        }
        if args.mode in {"neutral_hierarchy", "self_hierarchy", "self_hierarchy_counterfactual"}:
            r["shuffle_hierarchy"] = evaluate(
                ctl, test, cache, root, args.device, perturb="shuffle_hierarchy"
            )
        runs.append(r)

    after = sha_model(model)
    if before != after:
        raise SystemExit("frozen 15M source model changed")

    result = {
        "experiment": "v0.16g authority decomposition",
        "source_variant": ckpt.get("variant"),
        "source_tokens_seen": ckpt.get("tokens_seen"),
        "mode": args.mode,
        "frozen_model_unchanged": before == after,
        "controller_parameters": runs[0]["parameters"],
        "same_parameterization_across_modes": True,
        "same_train_test_and_seeds_as_v16f": True,
        "aggregate": aggregate(runs),
        "runs": runs,
    }
    if "shuffle_hierarchy" in runs[0]:
        normal = result["aggregate"]["state_exact_per_step"]["mean"]
        vals = [r["shuffle_hierarchy"]["state_exact_per_step"] for r in runs]
        m = statistics.mean(vals)
        result["shuffle_hierarchy"] = {
            "mean": m,
            "delta_vs_normal": m - normal,
        }

    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
