from __future__ import annotations

import argparse
import copy
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from config import ModelConfig
from model import PersonalNativeLM
from personal_model import PersonalLearningSystem, PersonalMicroModel
from personal_state import PersonalState, StateSlot, CORE_DIM, POLICY_DIM
from tokenizer import ByteTokenizer

TOK = ByteTokenizer()

Q_CORE = "למשימה הזאת בחר אפשרות. תשובה:"
Q_CORE_ALT = "עבור המשימה בחר את האפשרות שלך. תשובה:"
Q_POLICY = "לפורמט התשובה בחר אפשרות. תשובה:"
Q_POLICY_ALT = "בחר את פורמט התשובה המועדף. תשובה:"
Q_FACT = "כמה זה שתיים ועוד שתיים? תשובה:"

H_CORE = {
    0: "באינטראקציה קודמת המשתמש בחר אפשרות 2 למשימה.",
    1: "באינטראקציה קודמת המשתמש בחר אפשרות 1 למשימה.",
}
H_POLICY = {
    0: "באינטראקציה קודמת המשתמש בחר פורמט 2 לתשובה.",
    1: "באינטראקציה קודמת המשתמש בחר פורמט 1 לתשובה.",
}
H_NOISE = [
    "המשתמש פתח את המערכת.",
    "הושלמה אינטראקציה רגילה.",
]


def target_state(a: int, b: int):
    core = [0.0] * CORE_DIM
    policy = [0.0] * POLICY_DIM
    core[0] = 0.6 if a else -0.6
    policy[0] = 0.6 if b else -0.6
    s = PersonalState(
        user_id=f"profile-{a}{b}",
        core=StateSlot(core, confidence=0.8, evidence_count=10),
        policies=StateSlot(policy, confidence=0.8, evidence_count=10),
    )
    return s.flatten()


def history_for(ab):
    a, b = ab
    return [H_NOISE[0], H_CORE[a], H_NOISE[1], H_POLICY[b]]


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


def answer_for(kind, ab):
    a, b = ab
    if kind in ("core", "core_alt"):
        return "1" if a else "2"
    if kind in ("policy", "policy_alt"):
        return "1" if b else "2"
    return "4"


def prompt_for(kind):
    return {
        "core": Q_CORE,
        "core_alt": Q_CORE_ALT,
        "policy": Q_POLICY,
        "policy_alt": Q_POLICY_ALT,
        "fact": Q_FACT,
    }[kind]


def make_lm_example(kind, ab):
    q = prompt_for(kind)
    ans = answer_for(kind, ab)
    prompt = TOK.encode(q, bos=True, eos=False)
    answer = TOK.encode(ans, bos=False, eos=True)
    seq = prompt + answer
    x = seq[:-1]
    y = seq[1:]
    answer_start = len(prompt) - 1
    y = [-100 if i < answer_start else t for i, t in enumerate(y)]
    cond = TOK.encode(q, bos=False, eos=False)
    return x, y, cond, ans


def collate_lm(examples, states, device):
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
def eval_profiles(lm, personal, device, use_history=True):
    lm.eval(); personal.eval()
    rows = []
    correct = 0
    total = 0
    for ab in ((0,0),(0,1),(1,0),(1,1)):
        hids, htm, hem = encode_histories([history_for(ab)], device)
        _, state = personal.from_history(hids, htm, hem)
        for kind in ("core", "core_alt", "policy", "policy_alt", "fact"):
            q = prompt_for(kind)
            inp = torch.tensor([TOK.encode(q, bos=True, eos=False)], dtype=torch.long, device=device)
            cond = torch.tensor([TOK.encode(q, bos=False, eos=False)], dtype=torch.long, device=device)
            logits, _ = lm(inp, state, condition_ids=cond)
            last = logits[0, -1]
            candidates = [ord("1"), ord("2"), ord("4")]
            pred = chr(max(candidates, key=lambda i: float(last[i])))
            expected = answer_for(kind, ab)
            ok = pred == expected
            correct += int(ok); total += 1
            rows.append((ab, kind, expected, pred, ok))
    lm.train(); personal.train()
    return correct / total, rows


def warmup_personal(personal, device, steps=220, lr=3e-3):
    """Teach the personal learner to build canonical state from histories."""
    profiles = [(0,0),(0,1),(1,0),(1,1)]
    hids, htm, hem = encode_histories([history_for(ab) for ab in profiles], device)
    target = torch.stack([target_state(*ab) for ab in profiles]).to(device)
    opt = torch.optim.AdamW(personal.parameters(), lr=lr, weight_decay=0.001)
    for step in range(1, steps + 1):
        _, state = personal.from_history(hids, htm, hem)
        loss = (
            F.mse_loss(state[:, :64], target[:, :64])
            + F.mse_loss(state[:, 64:128], target[:, 64:128])
            + 0.25 * F.mse_loss(state[:, -4:], target[:, -4:])
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(personal.parameters(), 1.0)
        opt.step()
        if step == 1 or step % 50 == 0 or step == steps:
            pairs = [(round(float(state[i,0].detach()), 2), round(float(state[i,64].detach()), 2)) for i in range(4)]
            print(f"personal-warmup step={step:03d} loss={loss.item():.5f} core_policy={pairs}")


def joint_train(lm, personal, device, steps, lr):
    profiles = [(0,0),(0,1),(1,0),(1,1)]
    opt = torch.optim.AdamW(
        list(lm.parameters()) + list(personal.parameters()),
        lr=lr,
        weight_decay=0.01,
    )
    initial_lm = {k: v.detach().clone() for k, v in lm.state_dict().items()}

    for step in range(1, steps + 1):
        # One history per profile, then five request types per profile.
        hids, htm, hem = encode_histories([history_for(ab) for ab in profiles], device)
        _, states4 = personal.from_history(hids, htm, hem)

        examples = []
        kinds = []
        expanded_states = []
        targets = []
        for i, ab in enumerate(profiles):
            for kind in ("core", "core_alt", "policy", "policy_alt", "fact"):
                examples.append(make_lm_example(kind, ab))
                kinds.append(kind)
                expanded_states.append(states4[i])
            targets.append(target_state(*ab))
        states = torch.stack(expanded_states)
        x, y, q, qm, states = collate_lm(examples, states, device)

        _, lm_loss = lm(x, states, y, condition_ids=q, condition_mask=qm)

        # Supervise routing in this synthetic curriculum: core questions may
        # use core, policy questions may use policy, factual questions should
        # ignore personal state entirely.
        q_emb = lm.token_embedding(q)
        _, route_gates, relevance = lm.controller(states, q_emb, qm, return_routing=True)
        # Competitive route target: 0=core, 1=policy, 2=world, 3=routing, 4=none.
        route_class = torch.full((len(kinds),), 4, dtype=torch.long, device=device)
        for ri, kind in enumerate(kinds):
            if kind.startswith("core"):
                route_class[ri] = 0
            elif kind.startswith("policy"):
                route_class[ri] = 1
        none_prob = 1.0 - relevance
        route_probs = torch.cat([route_gates, none_prob], dim=-1).clamp_min(1e-8)
        route_loss = F.nll_loss(torch.log(route_probs), route_class)

        target = torch.stack(targets).to(device)
        # Auxiliary canonical-state loss makes the personal model interpretable
        # instead of letting it invent an arbitrary hidden code.
        state_loss = (
            F.mse_loss(states4[:, :64], target[:, :64])
            + F.mse_loss(states4[:, 64:128], target[:, 64:128])
            + 0.25 * F.mse_loss(states4[:, -4:], target[:, -4:])
        )
        loss = lm_loss + 2.0 * state_loss + 2.0 * route_loss

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(lm.parameters()) + list(personal.parameters()), 1.0)
        opt.step()

        if step == 1 or step % 10 == 0 or step == steps:
            acc, _ = eval_profiles(lm, personal, device)
            print(f"joint step={step:03d} loss={loss.item():.4f} lm={lm_loss.item():.4f} state={state_loss.item():.4f} route={route_loss.item():.4f} eval={acc:.3f}")

    changed = 0
    total_delta = 0.0
    after = lm.state_dict()
    for k in initial_lm:
        d = (after[k] - initial_lm[k]).abs().sum().item()
        total_delta += d
        changed += int(d > 0)
    print(f"joint_lm_changed_tensors={changed}/{len(initial_lm)} total_abs_delta={total_delta:.6f}")


def freeze_adapt_test(lm, personal, device, steps=30, lr=0.25):
    """Freeze the shared system and train only one new user's 116 params."""
    for p in lm.parameters():
        p.requires_grad_(False)
    for p in personal.parameters():
        p.requires_grad_(False)

    lm.eval(); personal.eval()
    lm_before = {k: v.detach().clone() for k, v in lm.state_dict().items()}
    personal_before = {k: v.detach().clone() for k, v in personal.state_dict().items()}

    # New user initially looks like profile (0,0).
    hids, htm, hem = encode_histories([history_for((0,0))], device)
    with torch.no_grad():
        init_z, _ = personal.from_history(hids, htm, hem)
    micro = PersonalMicroModel(init_z[0]).to(device)

    def predict(kind):
        state = micro(personal.state_decoder)
        q = prompt_for(kind)
        inp = torch.tensor([TOK.encode(q, bos=True, eos=False)], dtype=torch.long, device=device)
        cond = torch.tensor([TOK.encode(q, bos=False, eos=False)], dtype=torch.long, device=device)
        logits, _ = lm(inp, state, condition_ids=cond)
        last = logits[0, -1]
        candidates = [ord("1"), ord("2"), ord("4")]
        return chr(max(candidates, key=lambda i: float(last[i])))

    before_core = predict("core_alt")
    before_policy = predict("policy_alt")
    before_fact = predict("fact")

    # The user's core preference changes. We train only this user's micro-model
    # from support interactions; the LM and shared personal system are frozen.
    support = [make_lm_example("core", (1,0)), make_lm_example("core_alt", (1,0))]
    opt = torch.optim.Adam([micro.latent], lr=lr)
    for step in range(1, steps + 1):
        state = micro(personal.state_decoder, batch_size=len(support))
        x, y, q, qm, state = collate_lm(support, state, device)
        _, loss = lm(x, state, y, condition_ids=q, condition_mask=qm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        # This support evidence is explicitly about the core preference. Keep
        # unrelated personal slots stable during online adaptation.
        micro.latent.grad.mul_(PersonalMicroModel.slot_gradient_mask("core", device=micro.latent.device))
        opt.step()
        if step == 1 or step % 10 == 0 or step == steps:
            print(f"freeze-personal step={step:03d} loss={loss.item():.4f}")

    after_core = predict("core_alt")
    after_policy = predict("policy_alt")
    after_fact = predict("fact")

    lm_unchanged = all(torch.equal(v, lm.state_dict()[k]) for k, v in lm_before.items())
    personal_shared_unchanged = all(torch.equal(v, personal.state_dict()[k]) for k, v in personal_before.items())
    print("freeze_before", {"core": before_core, "policy": before_policy, "fact": before_fact})
    print("freeze_after ", {"core": after_core, "policy": after_policy, "fact": after_fact})
    print(f"lm_unchanged={lm_unchanged} shared_personal_unchanged={personal_shared_unchanged}")
    print(f"user_micro_parameters={sum(p.numel() for p in micro.parameters())}")

    success = (
        after_core == "1" and
        after_policy == "2" and
        after_fact == "4" and
        lm_unchanged and
        personal_shared_unchanged
    )
    return success, micro


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personal-warmup", type=int, default=220)
    ap.add_argument("--joint-steps", type=int, default=30)
    ap.add_argument("--adapt-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--base", default="personal-conditioning-v0.3.pt")
    ap.add_argument("--save", default="joint-personal-v0.4.pt")
    args = ap.parse_args()

    random.seed(23)
    torch.manual_seed(23)
    cfg = ModelConfig.byte_prototype()
    lm = PersonalNativeLM(cfg).to(args.device)
    base = Path(args.base)
    ckpt = None
    if base.exists():
        ckpt = torch.load(base, map_location=args.device, weights_only=False)
        load = lm.load_state_dict(ckpt["model"], strict=False)
        print(f"loaded_base={base} missing={len(load.missing_keys)} unexpected={len(load.unexpected_keys)}")
    personal = PersonalLearningSystem(cfg).to(args.device)
    if ckpt is not None and "personal_learning_system" in ckpt:
        personal.load_state_dict(ckpt["personal_learning_system"])
        print("loaded_personal_learning_system=True")

    print(f"main_lm_params={sum(p.numel() for p in lm.parameters()):,}")
    print(f"shared_personal_params={sum(p.numel() for p in personal.parameters()):,}")
    print("per_user_micro_params=116")

    before, _ = eval_profiles(lm, personal, args.device)
    print(f"joint_eval_before_personal_warmup={before:.3f}")
    warmup_personal(personal, args.device, args.personal_warmup)
    warmed, _ = eval_profiles(lm, personal, args.device)
    print(f"joint_eval_after_personal_warmup={warmed:.3f}")
    joint_train(lm, personal, args.device, args.joint_steps, args.lr)
    after, rows = eval_profiles(lm, personal, args.device)
    print(f"joint_eval_after={after:.3f}")

    ok, micro = freeze_adapt_test(lm, personal, args.device, args.adapt_steps)
    print(f"freeze_adapt_success={ok}")

    torch.save({
        "config": cfg.__dict__,
        "model": lm.state_dict(),
        "personal_learning_system": personal.state_dict(),
        "example_adapted_micro_latent": micro.latent.detach().cpu(),
        "joint_eval_accuracy": after,
        "freeze_adapt_success": ok,
        "stage": "joint-main-and-personal-training-v0.4",
    }, args.save)
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
