import argparse
import random
import torch

from config import ModelConfig
from model import PersonalNativeLM
from personal_state import PersonalState, StateSlot, CORE_DIM, POLICY_DIM
from tokenizer import ByteTokenizer

TOK = ByteTokenizer()

Q_CORE = "למשימה הזאת בחר אפשרות. תשובה:"
Q_POLICY = "לפורמט התשובה בחר אפשרות. תשובה:"
Q_FACT = "כמה זה שתיים ועוד שתיים? תשובה:"


def profile(a: int, b: int):
    core = [0.0] * CORE_DIM
    policy = [0.0] * POLICY_DIM
    core[0] = 1.0 if a else -1.0
    policy[0] = 1.0 if b else -1.0
    s = PersonalState(
        user_id=f"u{a}{b}",
        core=StateSlot(core, confidence=1.0, evidence_count=10),
        policies=StateSlot(policy, confidence=1.0, evidence_count=10),
    )
    return s.flatten()

PROFILES = {(a,b): profile(a,b) for a in (0,1) for b in (0,1)}


def make_example(kind=None, ab=None):
    if ab is None:
        ab = random.choice(list(PROFILES))
    a, b = ab
    if kind is None:
        kind = random.choices(["core", "policy", "fact"], weights=[4,4,2])[0]
    if kind == "core":
        q, ans = Q_CORE, ("1" if a else "2")
    elif kind == "policy":
        q, ans = Q_POLICY, ("1" if b else "2")
    else:
        q, ans = Q_FACT, "4"

    prompt = TOK.encode(q, bos=True, eos=False)
    answer = TOK.encode(ans, bos=False, eos=True)
    seq = prompt + answer
    x = seq[:-1]
    y = seq[1:]

    answer_start = len(prompt) - 1
    y = [-100 if i < answer_start else t for i, t in enumerate(y)]
    cond = TOK.encode(q, bos=False, eos=False)
    return x, y, cond, PROFILES[ab], kind, ans


def collate(examples, device):
    max_t = max(len(e[0]) for e in examples)
    max_q = max(len(e[2]) for e in examples)
    xs, ys, qs, qm, states = [], [], [], [], []
    for x, y, q, state, _, _ in examples:
        xs.append(x + [TOK.PAD] * (max_t - len(x)))
        ys.append(y + [-100] * (max_t - len(y)))
        qs.append(q + [TOK.PAD] * (max_q - len(q)))
        qm.append([1] * len(q) + [0] * (max_q - len(q)))
        states.append(state)
    return (
        torch.tensor(xs, dtype=torch.long, device=device),
        torch.tensor(ys, dtype=torch.long, device=device),
        torch.tensor(qs, dtype=torch.long, device=device),
        torch.tensor(qm, dtype=torch.bool, device=device),
        torch.stack(states).to(device),
    )


@torch.no_grad()
def evaluate(model, device):
    model.eval()
    rows = []
    correct = 0
    total = 0
    prompts = {"core": Q_CORE, "policy": Q_POLICY, "fact": Q_FACT}
    for kind in ("core", "policy", "fact"):
        for ab in PROFILES:
            _, _, q, state, _, expected = make_example(kind, ab)
            q_prompt = TOK.encode(prompts[kind], bos=True, eos=False)
            inp = torch.tensor([q_prompt], dtype=torch.long, device=device)
            cond = torch.tensor([q], dtype=torch.long, device=device)
            st = state.unsqueeze(0).to(device)
            logits, _ = model(inp, st, condition_ids=cond)
            last = logits[0, -1]
            candidates = [ord("1"), ord("2"), ord("4")]
            pred_id = max(candidates, key=lambda i: float(last[i]))
            pred = chr(pred_id)
            ok = pred == expected
            correct += int(ok)
            total += 1
            rows.append((kind, ab, expected, pred, ok))
    model.train()
    return correct / total, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=160)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="personal-conditioning-v0.3.pt")
    args = ap.parse_args()

    random.seed(11)
    torch.manual_seed(11)
    cfg = ModelConfig.byte_prototype()
    model = PersonalNativeLM(cfg).to(args.device)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters={params:,} device={args.device}")

    before, _ = evaluate(model, args.device)
    print(f"eval_before={before:.3f}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    for step in range(1, args.steps + 1):
        full = [
            make_example(kind, ab)
            for kind in ("core", "policy", "fact")
            for ab in PROFILES
        ]
        if args.batch_size <= len(full):
            examples = full[:args.batch_size]
        else:
            copies = (args.batch_size + len(full) - 1) // len(full)
            examples = (full * copies)[:args.batch_size]
        random.shuffle(examples)

        x, y, q, qm, s = collate(examples, args.device)
        _, loss = model(x, s, y, condition_ids=q, condition_mask=qm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 20 == 0:
            acc, _ = evaluate(model, args.device)
            print(f"step={step:04d} loss={loss.item():.4f} eval={acc:.3f}")

    acc, rows = evaluate(model, args.device)
    print(f"eval_after={acc:.3f}")
    for r in rows:
        print("case", r)

    torch.save({
        "config": cfg.__dict__,
        "model": model.state_dict(),
        "stage": "contextual-personal-conditioning-bootstrap",
        "eval_accuracy": acc,
    }, args.save)
    print(f"saved={args.save}")


if __name__ == "__main__":
    main()
