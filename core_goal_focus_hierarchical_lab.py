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
from core_goal_focus_sequence_lab import (
    TRAIN_CORES, TEST_CORES,
    TRAIN_GOALS, TEST_GOALS,
    TRAIN_FOCI, TEST_FOCI,
    TRAIN_TEMPLATES, TEST_TEMPLATES,
    build_cache,
)


def sha_model(model: nn.Module) -> str:
    h = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            t = tensor.detach().cpu().contiguous()
            h.update(name.encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(bytes(t.untyped_storage()))
    return h.hexdigest()


def different(rng, seq, current):
    return rng.choice([x for x in seq if x != current])


def choose_event(rng):
    return rng.choice([
        "noise", "noise",
        "focus", "focus", "focus",
        "tempting_focus", "tempting_focus",
        "goal_pressure", "goal_pressure",
        "goal_change", "goal_done",
        "core_pressure", "core_change",
    ])


def build_hard_sequence(rng, length, cores, goals, foci, templates, seq_id):
    core = rng.choice(cores)
    goal = rng.choice(goals)
    focus = rng.choice(foci)
    initial = (core, goal, focus)
    steps = []

    for t in range(length):
        kind = choose_event(rng)
        next_core, next_goal, next_focus = core, goal, focus
        decoy_core = different(rng, cores, core)
        decoy_goal = different(rng, goals, goal)
        decoy_focus = different(rng, foci, focus)

        # Candidate payload is what the system will actually adopt if it
        # incorrectly accepts a tempting change. This makes false updates
        # persist as real drift in closed-loop evaluation.
        cand_core, cand_goal, cand_focus = core, goal, focus
        target = [0.0, 0.0, 0.0]

        if kind == "noise":
            cand_core, cand_goal, cand_focus = decoy_core, decoy_goal, decoy_focus
        elif kind in {"focus", "tempting_focus"}:
            cand_focus = different(rng, foci, focus)
            next_focus = cand_focus
            target = [0.0, 0.0, 1.0]
        elif kind == "goal_pressure":
            cand_goal, cand_focus = decoy_goal, decoy_focus
        elif kind in {"goal_change", "goal_done"}:
            cand_goal = different(rng, goals, goal)
            cand_focus = different(rng, foci, focus)
            next_goal, next_focus = cand_goal, cand_focus
            target = [0.0, 1.0, 1.0]
        elif kind == "core_pressure":
            cand_core, cand_goal, cand_focus = decoy_core, decoy_goal, decoy_focus
        elif kind == "core_change":
            cand_core = different(rng, cores, core)
            cand_goal = different(rng, goals, goal)
            cand_focus = different(rng, foci, focus)
            next_core, next_goal, next_focus = cand_core, cand_goal, cand_focus
            target = [1.0, 1.0, 1.0]

        event = rng.choice(templates[kind]).format(
            core=cand_core,
            goal=cand_goal,
            focus=cand_focus,
            decoy_core=decoy_core,
            decoy_goal=decoy_goal,
            decoy_focus=decoy_focus,
        )
        steps.append({
            "seq_id": seq_id,
            "step": t,
            "kind": kind,
            "core": core,
            "goal": goal,
            "focus": focus,
            "event": event,
            "candidate_core": cand_core,
            "candidate_goal": cand_goal,
            "candidate_focus": cand_focus,
            "target": target,
            "next_core": next_core,
            "next_goal": next_goal,
            "next_focus": next_focus,
        })
        core, goal, focus = next_core, next_goal, next_focus

    return {"seq_id": seq_id, "initial": initial, "steps": steps}


def make_hard_sequences(n, min_len, max_len, seed, cores, goals, foci, templates):
    rng = random.Random(seed)
    return [
        build_hard_sequence(
            rng, rng.randint(min_len, max_len),
            cores, goals, foci, templates, i
        )
        for i in range(n)
    ]


class ContentProjector(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(F.layer_norm(x, (x.shape[-1],)))


class RelationState(nn.Module):
    """A persistent learned relation object, not a replacement for content."""
    def __init__(self, d_model: int, content_dim: int, relation_dim: int):
        super().__init__()
        self.parent = nn.Linear(d_model, relation_dim, bias=False)
        self.child = nn.Linear(d_model, relation_dim, bias=False)
        self.joint = nn.Sequential(
            nn.Linear(2 * relation_dim, relation_dim),
            nn.Tanh(),
        )
        self.update = nn.GRUCell(
            input_size=content_dim * 3,
            hidden_size=relation_dim,
        )

    def initialize(self, parent, child):
        p = self.parent(F.layer_norm(parent, (parent.shape[-1],)))
        c = self.child(F.layer_norm(child, (child.shape[-1],)))
        return self.joint(torch.cat((p, c), dim=-1))

    def propose_update(self, state, parent_content, child_content, event_content):
        inp = torch.cat((parent_content, child_content, event_content), dim=-1)
        return self.update(inp, state)


class HierarchicalPersistentController(nn.Module):
    def __init__(self, d_model: int, mode: str, content_dim: int = 96, relation_dim: int = 64):
        super().__init__()
        if mode not in {"direct", "relational"}:
            raise ValueError(mode)
        self.mode = mode
        self.content_dim = content_dim
        self.relation_dim = relation_dim

        self.content = ContentProjector(d_model, content_dim)

        self.self_core = RelationState(d_model, content_dim, relation_dim)
        self.core_goal = RelationState(d_model, content_dim, relation_dim)
        self.goal_focus = RelationState(d_model, content_dim, relation_dim)

        # Relation path can only add a residual signal. It cannot erase content.
        self.core_rel = nn.Linear(relation_dim, content_dim, bias=False)
        self.goal_rel = nn.Linear(relation_dim * 2, content_dim, bias=False)
        self.focus_rel = nn.Linear(relation_dim * 3, content_dim, bias=False)
        nn.init.zeros_(self.core_rel.weight)
        nn.init.zeros_(self.goal_rel.weight)
        nn.init.zeros_(self.focus_rel.weight)

        self.core_gate = nn.Parameter(torch.tensor(-3.0))
        self.goal_gate = nn.Parameter(torch.tensor(-3.0))
        self.focus_gate = nn.Parameter(torch.tensor(-3.0))

        # Base content path remains usable even if all relation gates are zero.
        self.core_head = nn.Sequential(
            nn.Linear(content_dim * 3, content_dim),
            nn.GELU(),
            nn.Linear(content_dim, 1),
        )
        self.goal_head = nn.Sequential(
            nn.Linear(content_dim * 4, content_dim),
            nn.GELU(),
            nn.Linear(content_dim, 1),
        )
        self.focus_head = nn.Sequential(
            nn.Linear(content_dim * 5, content_dim),
            nn.GELU(),
            nn.Linear(content_dim, 1),
        )

    def relation_gates(self):
        if self.mode == "direct":
            z = self.core_gate.new_zeros(())
            return z, z, z
        return (
            torch.sigmoid(self.core_gate),
            torch.sigmoid(self.goal_gate),
            torch.sigmoid(self.focus_gate),
        )

    def initialize_relations(self, self_anchor, core, goal, focus):
        r_sc = self.self_core.initialize(self_anchor, core)
        r_cg = self.core_goal.initialize(core, goal)
        r_gf = self.goal_focus.initialize(goal, focus)
        return r_sc, r_cg, r_gf

    def logits(self, core, goal, focus, event, r_sc, r_cg, r_gf, relation_mask=(1.0,1.0,1.0)):
        c = self.content(core)
        g = self.content(goal)
        f = self.content(focus)
        e = self.content(event)

        gc, gg, gf = self.relation_gates()
        m_sc, m_cg, m_gf = relation_mask

        core_extra = gc * self.core_rel(r_sc * m_sc)
        goal_extra = gg * self.goal_rel(torch.cat((r_sc * m_sc, r_cg * m_cg), -1))
        focus_extra = gf * self.focus_rel(
            torch.cat((r_sc * m_sc, r_cg * m_cg, r_gf * m_gf), -1)
        )

        # Content is preserved; relation information is residual.
        core_ctx = c + core_extra
        goal_ctx = g + goal_extra
        focus_ctx = f + focus_extra

        raw_core = self.core_head(torch.cat((core_ctx, e, c * e), -1))
        raw_goal = self.goal_head(torch.cat((core_ctx, goal_ctx, e, g * e), -1))
        raw_focus = self.focus_head(torch.cat((core_ctx, goal_ctx, focus_ctx, e, f * e), -1))

        # Hierarchical logic: higher-level change implies resetting descendants.
        p_core = torch.sigmoid(raw_core)
        p_goal_local = torch.sigmoid(raw_goal)
        p_focus_local = torch.sigmoid(raw_focus)
        p_goal = 1.0 - (1.0 - p_core) * (1.0 - p_goal_local)
        p_focus = 1.0 - (1.0 - p_goal) * (1.0 - p_focus_local)
        probs = torch.cat((p_core, p_goal, p_focus), -1).clamp(1e-5, 1-1e-5)
        return probs, (c, g, f, e)

    def update_relations(
        self, r_sc, r_cg, r_gf,
        self_anchor, core, goal, focus, event,
        next_core, next_goal, next_focus,
        change_bits,
    ):
        c = self.content(core)
        g = self.content(goal)
        f = self.content(focus)
        e = self.content(event)
        nc = self.content(next_core)
        ng = self.content(next_goal)
        nf = self.content(next_focus)

        # Proposed persistent updates. They are applied only at the relevant scope.
        prop_sc = self.self_core.propose_update(r_sc, self.content(self_anchor), nc, e)
        prop_cg = self.core_goal.propose_update(r_cg, nc, ng, e)
        prop_gf = self.goal_focus.propose_update(r_gf, ng, nf, e)

        b_core = change_bits[:, 0:1]
        b_goal = change_bits[:, 1:2]
        b_focus = change_bits[:, 2:3]

        r_sc = r_sc * (1-b_core) + prop_sc * b_core
        r_cg = r_cg * (1-b_goal) + prop_cg * b_goal
        r_gf = r_gf * (1-b_focus) + prop_gf * b_focus
        return r_sc, r_cg, r_gf


def v(cache, texts, device):
    return torch.stack([cache[t] for t in texts]).to(device)


def train_controller(
    ctl, sequences, cache, self_anchor, device, epochs, lr, seed
):
    opt = torch.optim.AdamW(ctl.parameters(), lr=lr, weight_decay=0.01)

    # Sequential teacher-forced training preserves relation-state continuity.
    rng = random.Random(seed)
    for epoch in range(epochs):
        order = list(range(len(sequences)))
        rng.shuffle(order)
        ctl.train()

        for idx in order:
            seq = sequences[idx]
            core, goal, focus = seq["initial"]
            core_v = cache[core].unsqueeze(0).to(device)
            goal_v = cache[goal].unsqueeze(0).to(device)
            focus_v = cache[focus].unsqueeze(0).to(device)
            r_sc, r_cg, r_gf = ctl.initialize_relations(
                self_anchor.unsqueeze(0), core_v, goal_v, focus_v
            )

            losses = []
            for e in seq["steps"]:
                event_v = cache[e["event"]].unsqueeze(0).to(device)
                probs, _ = ctl.logits(
                    core_v, goal_v, focus_v, event_v, r_sc, r_cg, r_gf
                )
                target = torch.tensor([e["target"]], dtype=torch.float32, device=device)

                # Explicit weighting against drift: false high-level updates are costly.
                weights = torch.tensor([[2.5, 1.7, 1.0]], device=device)
                bce = F.binary_cross_entropy(probs, target, reduction="none")
                loss = (bce * weights).mean()
                losses.append(loss)

                next_core_v = cache[e["next_core"]].unsqueeze(0).to(device)
                next_goal_v = cache[e["next_goal"]].unsqueeze(0).to(device)
                next_focus_v = cache[e["next_focus"]].unsqueeze(0).to(device)

                r_sc, r_cg, r_gf = ctl.update_relations(
                    r_sc, r_cg, r_gf,
                    self_anchor.unsqueeze(0),
                    core_v, goal_v, focus_v, event_v,
                    next_core_v, next_goal_v, next_focus_v,
                    target,
                )
                core_v, goal_v, focus_v = next_core_v, next_goal_v, next_focus_v

            seq_loss = torch.stack(losses).mean()
            opt.zero_grad(set_to_none=True)
            seq_loss.backward()
            torch.nn.utils.clip_grad_norm_(ctl.parameters(), 1.0)
            opt.step()

    return ctl


@torch.no_grad()
def eval_closed_loop(
    ctl, sequences, cache, self_anchor, device,
    self_mode="normal",
    relation_mask=(1.0,1.0,1.0),
):
    ctl.eval()

    if self_mode == "zero":
        root = torch.zeros_like(self_anchor)
    elif self_mode == "negate":
        root = -self_anchor
    elif self_mode == "shuffle":
        g = torch.Generator(device="cpu"); g.manual_seed(1771)
        root = self_anchor[
            torch.randperm(self_anchor.numel(), generator=g).to(self_anchor.device)
        ]
    elif self_mode == "random":
        g = torch.Generator(device="cpu"); g.manual_seed(1771)
        root = torch.randn(self_anchor.shape, generator=g).to(device)
        root = root / root.norm().clamp_min(1e-8) * self_anchor.norm()
    else:
        root = self_anchor

    total = 0
    state_exact = 0
    final_exact = 0
    whole_exact = 0
    steps_to_error = []
    slot_correct = [0,0,0]
    false_updates = [0,0,0]
    false_den = [0,0,0]
    true_updates = [0,0,0]
    true_den = [0,0,0]
    goal_pressure_ok = goal_pressure_n = 0
    core_pressure_ok = core_pressure_n = 0
    after_focus_ok = after_focus_n = 0

    for seq in sequences:
        pred_core, pred_goal, pred_focus = seq["initial"]
        true_core, true_goal, true_focus = seq["initial"]

        core_v = cache[pred_core].unsqueeze(0).to(device)
        goal_v = cache[pred_goal].unsqueeze(0).to(device)
        focus_v = cache[pred_focus].unsqueeze(0).to(device)
        r_sc, r_cg, r_gf = ctl.initialize_relations(
            root.unsqueeze(0), core_v, goal_v, focus_v
        )

        seq_ok = True
        first_error = None
        focus_changes_since_goal = 0

        for t,e in enumerate(seq["steps"]):
            event_v = cache[e["event"]].unsqueeze(0).to(device)
            probs, _ = ctl.logits(
                core_v, goal_v, focus_v, event_v,
                r_sc, r_cg, r_gf,
                relation_mask=relation_mask,
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

            cand_core_v = cache[e["candidate_core"]].unsqueeze(0).to(device)
            cand_goal_v = cache[e["candidate_goal"]].unsqueeze(0).to(device)
            cand_focus_v = cache[e["candidate_focus"]].unsqueeze(0).to(device)

            next_pred_core = e["candidate_core"] if bits[0] else pred_core
            next_pred_goal = e["candidate_goal"] if bits[1] else pred_goal
            next_pred_focus = e["candidate_focus"] if bits[2] else pred_focus

            next_core_v = cache[next_pred_core].unsqueeze(0).to(device)
            next_goal_v = cache[next_pred_goal].unsqueeze(0).to(device)
            next_focus_v = cache[next_pred_focus].unsqueeze(0).to(device)

            r_sc, r_cg, r_gf = ctl.update_relations(
                r_sc, r_cg, r_gf,
                root.unsqueeze(0),
                core_v, goal_v, focus_v, event_v,
                next_core_v, next_goal_v, next_focus_v,
                bits.unsqueeze(0),
            )
            pred_core, pred_goal, pred_focus = (
                next_pred_core, next_pred_goal, next_pred_focus
            )
            core_v, goal_v, focus_v = next_core_v, next_goal_v, next_focus_v

            true_core, true_goal, true_focus = (
                e["next_core"], e["next_goal"], e["next_focus"]
            )

            total += 1
            state_bits = [
                pred_core == true_core,
                pred_goal == true_goal,
                pred_focus == true_focus,
            ]
            state_exact += int(all(state_bits))
            for s,ok in enumerate(state_bits):
                slot_correct[s] += int(ok)

            if not all(state_bits):
                seq_ok = False
                if first_error is None:
                    first_error = t + 1

            if e["kind"] in {"focus","tempting_focus"}:
                focus_changes_since_goal += 1
            if e["kind"] in {"goal_change","goal_done","core_change"}:
                focus_changes_since_goal = 0

            if e["kind"] == "goal_pressure":
                goal_pressure_n += 1
                goal_pressure_ok += int(pred_goal == true_goal)
            if e["kind"] == "core_pressure":
                core_pressure_n += 1
                core_pressure_ok += int(pred_core == true_core)
            if focus_changes_since_goal >= 3:
                after_focus_n += 1
                after_focus_ok += int(pred_goal == true_goal)

        final_exact += int(
            (pred_core,pred_goal,pred_focus) == (true_core,true_goal,true_focus)
        )
        whole_exact += int(seq_ok)
        steps_to_error.append(
            first_error if first_error is not None else len(seq["steps"]) + 1
        )

    div = lambda a,b: a/b if b else None
    return {
        "state_exact_per_step": div(state_exact,total),
        "core_state_accuracy": div(slot_correct[0],total),
        "goal_state_accuracy": div(slot_correct[1],total),
        "focus_state_accuracy": div(slot_correct[2],total),
        "final_state_exact": div(final_exact,len(sequences)),
        "whole_sequence_exact": div(whole_exact,len(sequences)),
        "mean_steps_until_first_state_error": statistics.mean(steps_to_error),
        "core_false_update_rate": div(false_updates[0],false_den[0]),
        "goal_false_update_rate": div(false_updates[1],false_den[1]),
        "focus_false_update_rate": div(false_updates[2],false_den[2]),
        "core_change_recall": div(true_updates[0],true_den[0]),
        "goal_change_recall": div(true_updates[1],true_den[1]),
        "focus_change_recall": div(true_updates[2],true_den[2]),
        "goal_pressure_retention": div(goal_pressure_ok,goal_pressure_n),
        "core_pressure_retention": div(core_pressure_ok,core_pressure_n),
        "goal_retention_after_3plus_focus_changes": div(after_focus_ok,after_focus_n),
        "relation_gates": [float(x.detach().cpu()) for x in ctl.relation_gates()],
    }


def aggregate(runs):
    keys = [
        "state_exact_per_step",
        "core_state_accuracy","goal_state_accuracy","focus_state_accuracy",
        "final_state_exact","whole_sequence_exact",
        "mean_steps_until_first_state_error",
        "core_false_update_rate","goal_false_update_rate","focus_false_update_rate",
        "core_change_recall","goal_change_recall","focus_change_recall",
        "goal_pressure_retention","core_pressure_retention",
        "goal_retention_after_3plus_focus_changes",
    ]
    out = {}
    for k in keys:
        vals = [r["normal"][k] for r in runs if r["normal"][k] is not None]
        out[k] = {
            "mean": statistics.mean(vals),
            "stdev": statistics.stdev(vals) if len(vals)>1 else 0.0,
        }

    out["mean_relation_gates"] = [
        statistics.mean([r["normal"]["relation_gates"][i] for r in runs])
        for i in range(3)
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["auto","direct","relational"], default="auto")
    ap.add_argument("--train-sequences", type=int, default=400)
    ap.add_argument("--test-sequences", type=int, default=180)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=0.0010)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt,cfg,model = load_any(args.checkpoint,args.device)
    from train_hebrew import load_tokenizer
    tok,_ = load_tokenizer(args.tokenizer)

    before = sha_model(model)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    train_seq = make_hard_sequences(
        args.train_sequences,18,30,6101,
        TRAIN_CORES,TRAIN_GOALS,TRAIN_FOCI,TRAIN_TEMPLATES
    )
    test_seq = make_hard_sequences(
        args.test_sequences,36,52,10103,
        TEST_CORES,TEST_GOALS,TEST_FOCI,TEST_TEMPLATES
    )
    cache = build_cache(model,tok,train_seq+test_seq,args.device)

    if isinstance(model,SelfVariantPersonalNativeLM):
        self_anchor = model.self_anchor.detach().clone().to(args.device)
        if args.mode == "auto":
            modes = ["direct","relational"]
        else:
            modes = [args.mode]
    else:
        self_anchor = torch.zeros(cfg.d_model,device=args.device)
        if args.mode == "relational":
            raise SystemExit("baseline has no learned SELF anchor; relational mode is invalid")
        modes = ["direct"]

    seeds=[141,282,423]
    all_modes={}
    for mode in modes:
        runs=[]
        for seed in seeds:
            print(f"[train] variant={ckpt.get('variant','baseline')} mode={mode} seed={seed}",flush=True)
            torch.manual_seed(seed)
            ctl=HierarchicalPersistentController(cfg.d_model,mode).to(args.device)
            train_controller(
                ctl,train_seq,cache,self_anchor,args.device,
                args.epochs,args.lr,seed
            )
            r={
                "seed":seed,
                "mode":mode,
                "normal":eval_closed_loop(
                    ctl,test_seq,cache,self_anchor,args.device
                ),
            }
            if mode=="relational":
                for sm in ("zero","random","shuffle","negate"):
                    r[f"self_{sm}"]=eval_closed_loop(
                        ctl,test_seq,cache,self_anchor,args.device,self_mode=sm
                    )
                # Relation-specific causal ablations.
                r["drop_self_core"]=eval_closed_loop(
                    ctl,test_seq,cache,self_anchor,args.device,
                    relation_mask=(0.0,1.0,1.0)
                )
                r["drop_core_goal"]=eval_closed_loop(
                    ctl,test_seq,cache,self_anchor,args.device,
                    relation_mask=(1.0,0.0,1.0)
                )
                r["drop_goal_focus"]=eval_closed_loop(
                    ctl,test_seq,cache,self_anchor,args.device,
                    relation_mask=(1.0,1.0,0.0)
                )
                r["drop_all_relations"]=eval_closed_loop(
                    ctl,test_seq,cache,self_anchor,args.device,
                    relation_mask=(0.0,0.0,0.0)
                )
            runs.append(r)

        entry={"runs":runs,"aggregate":aggregate(runs)}
        if mode=="relational":
            base=entry["aggregate"]["state_exact_per_step"]["mean"]
            causal={}
            for key in (
                "self_zero","self_random","self_shuffle","self_negate",
                "drop_self_core","drop_core_goal","drop_goal_focus","drop_all_relations"
            ):
                vals=[r[key]["state_exact_per_step"] for r in runs]
                mean=statistics.mean(vals)
                causal[key]={
                    "state_exact_per_step_mean":mean,
                    "delta_vs_normal":mean-base,
                }
            entry["causal"]=causal
        all_modes[mode]=entry

    after=sha_model(model)
    if before!=after:
        raise SystemExit("frozen 15M source model changed")

    result={
        "experiment":"v0.16e persistent hierarchical SELF relation lab",
        "source_variant":ckpt.get("variant","baseline"),
        "source_tokens_seen":ckpt.get("tokens_seen"),
        "source_model_frozen":True,
        "frozen_model_unchanged":before==after,
        "design":{
            "content_path_always_preserved":True,
            "relation_path_is_gated_residual":True,
            "relation_gates_initial_logit":-3.0,
            "persistent_relation_states":[
                "SELF->CORE","CORE->GOAL","GOAL->FOCUS"
            ],
            "hierarchical_decision_logic":True,
            "closed_loop_test":True,
            "errors_persist":True,
            "disjoint_train_test_values":True,
            "disjoint_train_test_templates":True,
        },
        "modes":all_modes,
    }
    if "relational" in all_modes and "direct" in all_modes:
        result["relational_minus_direct_state_exact"]=(
            all_modes["relational"]["aggregate"]["state_exact_per_step"]["mean"]
            - all_modes["direct"]["aggregate"]["state_exact_per_step"]["mean"]
        )

    Path(args.out).write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":
    main()
