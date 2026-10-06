import argparse
import random
import torch
from config import ModelConfig
from model import PersonalNativeLM
from tokenizer import ByteTokenizer
from synthetic_data import sample_example, collate, make_state

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="prototype.pt")
    args = ap.parse_args()

    random.seed(7)
    torch.manual_seed(7)

    cfg = ModelConfig()
    tok = ByteTokenizer()
    model = PersonalNativeLM(cfg).to(args.device)

    print(f"parameters={count_params(model):,}")
    print(f"device={args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    model.train()

    for step in range(1, args.steps + 1):
        batch = [sample_example(tok) for _ in range(args.batch_size)]
        x, y, s = collate(batch, tok.PAD)
        x, y, s = x.to(args.device), y.to(args.device), s.to(args.device)

        _, loss = model(x, s, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1 or step % 10 == 0:
            print(f"step={step:04d} loss={loss.item():.4f}")

    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.save)
    print(f"saved={args.save}")

    # Architectural proof: same token input, different personal state => different logits.
    model.eval()
    prompt = "Choose A or B. Answer: "
    ids = tok.encode(prompt, eos=False)
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    s0 = make_state("u0", 0, world="demo").flatten("demo").unsqueeze(0).to(args.device)
    s1 = make_state("u1", 1, world="demo").flatten("demo").unsqueeze(0).to(args.device)

    with torch.no_grad():
        l0, _ = model(x, s0)
        l1, _ = model(x, s1)
        diff = (l0[:, -1, :] - l1[:, -1, :]).abs().mean().item()
    print(f"same_prompt_personal_state_logit_diff={diff:.6f}")

if __name__ == "__main__":
    main()
