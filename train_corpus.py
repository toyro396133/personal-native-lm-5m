"""
Real-corpus pretraining entry point.

Input: UTF-8 .txt file.
This prototype keeps a byte tokenizer so it runs with no tokenizer dependency.
For serious Hebrew training, swap tokenizer.py for a trained BPE/Unigram model.
"""
import argparse
from pathlib import Path
import torch
from config import ModelConfig
from model import PersonalNativeLM
from tokenizer import ByteTokenizer
from personal_state import PersonalState

def chunks(ids, seq_len):
    for i in range(0, len(ids) - seq_len - 1, seq_len):
        block = ids[i:i + seq_len + 1]
        yield block[:-1], block[1:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("text_file")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="pretrained.pt")
    args = ap.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    tok = ByteTokenizer()
    ids = tok.encode(text)

    cfg = ModelConfig(max_seq_len=max(args.seq_len, 256))
    model = PersonalNativeLM(cfg).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    neutral = PersonalState(user_id="neutral").flatten(None).unsqueeze(0).to(args.device)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for x_ids, y_ids in chunks(ids, args.seq_len):
            x = torch.tensor([x_ids], dtype=torch.long, device=args.device)
            y = torch.tensor([y_ids], dtype=torch.long, device=args.device)
            _, loss = model(x, neutral, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1

            if step % 100 == 0:
                print(f"epoch={epoch+1} step={step} loss={loss.item():.4f}")

    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.save)
    print(f"saved={args.save}")

if __name__ == "__main__":
    main()
