from __future__ import annotations

import argparse
from pathlib import Path
import random
import torch

from config import ModelConfig
from model import PersonalNativeLM
from personal_model import PersonalLearningSystem
from longitudinal_data import make_profiles
from train_hebrew import load_tokenizer
import train_longitudinal as tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--personal-warmup", type=int, default=180)
    ap.add_argument("--joint-steps", type=int, default=35)
    ap.add_argument("--adapt-steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save", default="personal-natural-v0.6.pt")
    args = ap.parse_args()

    random.seed(61)
    torch.manual_seed(61)
    tok, kind = load_tokenizer(args.tokenizer)
    tok.PAD = tok.pad_id
    tl.TOK = tok

    ckpt = torch.load(args.base, map_location=args.device, weights_only=False)
    cfg = ModelConfig(**ckpt["config"])
    if cfg.vocab_size != tok.vocab_size:
        raise ValueError(f"tokenizer/model vocab mismatch: {tok.vocab_size} != {cfg.vocab_size}")

    lm = PersonalNativeLM(cfg).to(args.device)
    load = lm.load_state_dict(ckpt["model"], strict=False)
    print(f"loaded_natural_lm={args.base} tokenizer={kind} missing={len(load.missing_keys)} unexpected={len(load.unexpected_keys)}")
    personal = PersonalLearningSystem(cfg).to(args.device)
    profiles = make_profiles()

    before, _ = tl.evaluate(lm, personal, profiles, args.device)
    print(f"v06_personal_eval_before={before:.3f}")
    tl.train_personal_warmup(personal, profiles, args.device, args.personal_warmup)
    tl.joint_train(lm, personal, profiles, args.device, args.joint_steps, args.lr)
    after, rows = tl.evaluate(lm, personal, profiles, args.device)
    print(f"v06_personal_eval_after={after:.3f}")
    ok, micro = tl.freeze_adapt(lm, personal, profiles[0], args.device, args.adapt_steps)

    torch.save({
        "config": cfg.__dict__,
        "model": lm.state_dict(),
        "personal_learning_system": personal.state_dict(),
        "example_adapted_micro_latent": micro.latent.detach().cpu(),
        "longitudinal_eval_accuracy": after,
        "freeze_adapt_success": ok,
        "stage": "natural-hebrew-plus-personal-v0.6",
        "base_checkpoint": str(Path(args.base)),
        "tokenizer_file": args.tokenizer,
    }, args.save)
    print(f"saved={args.save}")
    if after < 0.95 or not ok:
        raise SystemExit(f"v0.6 personalization gate failed: accuracy={after:.3f} freeze={ok}")

if __name__ == "__main__":
    main()
