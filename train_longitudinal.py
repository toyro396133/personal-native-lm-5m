from __future__ import annotations

import argparse
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from longitudinal_data import make_profiles, target_state
from model import PersonalNativeLM
from personal_model import PersonalLearningSystem, PersonalMicroModel
from tokenizer import ByteTokenizer

TOK = ByteTokenizer()

PROMPTS = {
    "core": "למשימה הנוכחית בחר את האפשרות המתאימה. תשובה:",
    "policy": "בחר את פורמט התשובה המועדף. תשובה:",
    "world": "בפרויקט העבודה בחר את הדרך המועדפת. תשובה:",
    "fact": "כמה זה שלוש ועוד שתיים? תשובה:",
}


def answer(profile, kind):
    if kind == "core":
        return "1" if profile.core else "2"
    if kind == "policy":
        return "1" if profile.policy else "2"
    if kind == "world":
        return "1" if profile.world else "2"
    return "5"


def encode_histories(histories, device):
    encoded = [[TOK.encode(e, bos=False, eos=False) for e in h] for h in histories]
    max_e = max(len(h) for h in encoded)
    max_l = max(len(e) for h in encoded for e in h)
    ids, tm, em = [], [], []
    for h in encoded:
        row, row_tm = [], []
        for e in h:
            row.append(e + [TOK.PAD] * (max_l - len(e)))
            row_tm.append([1] * len(e) + [0] * (max_l - len(e)))
        while len(row) < max_e:
            row.append([TOK.PAD] * max_l)
            row_tm.append([0] * max_l)
        ids.append(row)
        tm.append(row_tm)
        em.append([1] * len(h) + [0] * (max_e - len(h)))
    return (
        torch.tensor(ids, dtype=torch.long, device=device),
        torch.tensor(tm, dtype=torch.bool, device=device),
        torch.tensor(em, dtype=torch.bool, device=device),
    )


def make_example(profile, kind):
    q = PROMPTS[kind]
    ans = answer(profile, kind)
    prompt = TOK.encode(q, bos=True, eos=False)
    a = TOK.encode(ans, bos=False, eos=True)
    seq = prompt + a
    x, y = seq[:-1], seq[1:]
    answer_start = len(prompt) - 1
    y = [-100 if i < answer_start else t for i, t in enumerate(y)]
    cond = TOK.encode(q, bos=False, eos=False)
    return x, y, cond, ans


def collate(examples, states, device):
    max_t = max(len(e[0]) for e in examples)
    max_q = max(len(e[2]) for e in examples)
    xs, ys, qs, qm = [], [], [], []
    for x, y, q, _ in examples:
        xs.append(x + [TOK.PAD] * (max_t - len(x)))
        ys.append(y + [-100] * (max_t - len(y)))
        qs.append(q + [TOK.PAD] * (max_q - len(q)))
        qm.append([1] * len(q) + [0] * (max_q - len(q)))
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(qs, dtype=torch.long, device=device),
        torch.tensor(qm, dtype=torch.bool, device=device),
        states,
    )


@torch.no_grad()
def evaluate(lm, personal, profiles, device):
    lm.eval(); personal.eval()
    hids, htm, hem = encode_histories([p.history for p in profiles], device)
    _, states = personal.from_history(hids, htm, hem)
    rows, correct, total = [], 0, 0
    for i, p in enumerate(profiles):
        for kind in ("core", "policy", "world", "fact"):
            q = PROMPTS[kind]
            inp = torch.tensor([TOK.encode(q, bos=True, eos=False)], dtype=torch.long, device=device)
            cond = torch.tensor([TOK.encode(q, bos=False, eos=False)], dtype=torch.long, device=device)
            logits, _ = lm(inp, states[i:i+1], condition_ids=cond)
            candidates = [ord("1"), ord("2"), ord("5")]
            pred = chr(max(candidates, key=lambda c: float(logits[0, -1, c])))
            exp = answer(p, kind)
            ok = pred == exp
            rows.append((p.user_id, kind, exp, pred, ok))
            correct += int(ok); total += 1
    lm.train(); personal.train()
    return correct / total, rows


def train_personal_warmup(personal, profiles, device, steps, lr=2e-3):
    hids, htm, hem = encode_histories([p.history for p in profiles], device)
    target = torch.stack([target_state(p) for p in profiles]).to(device)
    opt = torch.optim.AdamW(personal.parameters(), lr=lr, weight_decay=0.001)
    for step in range(1, steps + 1):
        _, state = personal.from_history(hids, htm, hem)
        loss = (
            F.mse_loss(state[:, :64], target[:, :64])
            + F.mse_loss(state[:, 64:128], target[:, 64:128])
            + F.mse_loss(state[:, 128:192], target[:, 128:192])
            + 0.2 * F.mse_loss(state[:, 224:228], target[:, 224:228])
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(personal.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == steps:
            print(f"long-personal step={step:03d} loss={loss.item():.5f}")


def joint_train(lm, personal, profiles, device, steps, lr):
    opt = torch.optim.AdamW(list(lm.parameters()) + list(personal.parameters()), lr=lr, weight_decay=0.01)
    for step in range(1, steps + 1):
        hids, htm, hem = encode_histories([p.history for p in profiles], device)
        _, states_p = personal.from_history(hids, htm, hem)

        examples, expanded, kinds = [], [], []
        for i, p in enumerate(profiles):
            for kind in ("core", "policy", "world", "fact"):
                examples.append(make_example(p, kind))
                expanded.append(states_p[i])
                kinds.append(kind)
        states = torch.stack(expanded)
        x, y, q, qm, states = collate(examples, states, device)
        _, lm_loss = lm(x, states, y, condition_ids=q, condition_mask=qm)

        q_emb = lm.token_embedding(q)
        _, route_gates, relevance = lm.controller(states, q_emb, qm, return_routing=True)
        route_class = torch.full((len(kinds),), 4, dtype=torch.long, device=device)
        for i, kind in enumerate(kinds):
            if kind == "core": route_class[i] = 0
            elif kind == "policy": route_class[i] = 1
            elif kind == "world": route_class[i] = 2
        route_probs = torch.cat([route_gates, 1.0 - relevance], dim=-1).clamp_min(1e-8)
        route_loss = F.nll_loss(torch.log(route_probs), route_class)

        target = torch.stack([target_state(p) for p in profiles]).to(device)
        state_loss = (
            F.mse_loss(states_p[:, :64], target[:, :64])
            + F.mse_loss(states_p[:, 64:128], target[:, 64:128])
            + F.mse_loss(states_p[:, 128:192], target[:, 128:192])
        )
        loss = lm_loss + 1.5 * state_loss + 1.5 * route_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lm.parameters()) + list(personal.parameters()), 1.0)
        opt.step()
        if step == 1 or step % 10 == 0 or step == steps:
            acc, _ = evaluate(lm, personal, profiles, device)
            print(f"long-joint step={step:03d} loss={loss.item():.4f} lm={lm_loss.item():.4f} state={state_loss.item():.4f} route={route_loss.item():.4f} eval={acc:.3f}")


def freeze_adapt(lm, personal, profile, device, steps=30, lr=0.20):
    for p in lm.parameters(): p.requires_grad_(False)
    for p in personal.parameters(): p.requires_grad_(False)
    lm.eval(); personal.eval()
    lm_before = {k: v.detach().clone() for k, v in lm.state_dict().items()}

    hids, htm, hem = encode_histories([profile.history], device)
    with torch.no_grad():
        z, _ = personal.from_history(hids, htm, hem)
    micro = PersonalMicroModel(z[0]).to(device)

    def pred(kind):
        q = PROMPTS[kind]
        state = micro(personal.state_decoder)
        inp = torch.tensor([TOK.encode(q, bos=True, eos=False)], dtype=torch.long, device=device)
        cond = torch.tensor([TOK.encode(q, bos=False, eos=False)], dtype=torch.long, device=device)
        logits, _ = lm(inp, state, condition_ids=cond)
        candidates = [ord("1"), ord("2"), ord("5")]
        return chr(max(candidates, key=lambda c: float(logits[0, -1, c])))

    before = {k: pred(k) for k in ("core", "policy", "world", "fact")}
    # Flip only the world preference for this user's online adaptation.
    class P: pass
    changed = P(); changed.core = profile.core; changed.policy = profile.policy; changed.world = 1 - profile.world
    support = [make_example(changed, "world")]
    opt = torch.optim.Adam([micro.latent], lr=lr)
    for step in range(1, steps + 1):
        state = micro(personal.state_decoder, batch_size=len(support))
        x, y, q, qm, state = collate(support, state, device)
        _, loss = lm(x, state, y, condition_ids=q, condition_mask=qm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        micro.latent.grad.mul_(PersonalMicroModel.slot_gradient_mask("world", device=micro.latent.device))
        opt.step()
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"long-freeze step={step:03d} loss={loss.item():.4f}")

    after = {k: pred(k) for k in ("core", "policy", "world", "fact")}
    unchanged = all(torch.equal(v, lm.state_dict()[k]) for k, v in lm_before.items())
    desired_world = "1" if (1 - profile.world) else "2"
    expected_core = "1" if profile.core else "2"
    expected_policy = "1" if profile.policy else "2"
    success = (
        after["world"] == desired_world and after["core"] == expected_core and
        after["policy"] == expected_policy and after["fact"] == "5" and unchanged
    )
    print("long-freeze-before", before)
    print("long-freeze-after ", after)
    print(f"long-freeze-lm-unchanged={unchanged} success={success}")
    return success, micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personal-warmup", type=int, default=180)
    ap.add_argument("--joint-steps", type=int, default=40)
    ap.add_argument("--adapt-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base", default="joint-personal-v0.4.pt")
    ap.add_argument("--save", default="longitudinal-v0.5.pt")
    args = ap.parse_args()

    random.seed(41); torch.manual_seed(41)
    profiles = make_profiles()
    cfg = ModelConfig.byte_prototype()
    lm = PersonalNativeLM(cfg).to(args.device)
    personal = PersonalLearningSystem(cfg).to(args.device)

    base = Path(args.base)
    if base.exists():
        ckpt = torch.load(base, map_location=args.device, weights_only=False)
        load = lm.load_state_dict(ckpt["model"], strict=False)
        print(f"loaded_main={base} missing={len(load.missing_keys)} unexpected={len(load.unexpected_keys)}")
        if "personal_learning_system" in ckpt:
            personal.load_state_dict(ckpt["personal_learning_system"], strict=False)
            print("loaded_personal=True")

    print(f"profiles={len(profiles)} history_events={len(profiles[0].history)}")
    before, _ = evaluate(lm, personal, profiles, args.device)
    print(f"long_eval_before={before:.3f}")
    train_personal_warmup(personal, profiles, args.device, args.personal_warmup)
    joint_train(lm, personal, profiles, args.device, args.joint_steps, args.lr)
    after, rows = evaluate(lm, personal, profiles, args.device)
    print(f"long_eval_after={after:.3f}")
    ok, micro = freeze_adapt(lm, personal, profiles[0], args.device, args.adapt_steps)

    torch.save({
        "config": cfg.__dict__,
        "model": lm.state_dict(),
        "personal_learning_system": personal.state_dict(),
        "example_adapted_micro_latent": micro.latent.detach().cpu(),
        "longitudinal_eval_accuracy": after,
        "freeze_adapt_success": ok,
        "stage": "longitudinal-personal-training-v0.5",
    }, args.save)
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
