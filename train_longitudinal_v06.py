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


def freeze_language_backbone(lm: PersonalNativeLM):
    """Freeze parameters used by ordinary unconditioned language inference.

    Keep trainable only the request-conditioned personal interface: the
    controller and the per-block conditioning delta/gate projections. Since
    Block.forward bypasses those modules when condition_ids is absent, this
    phase cannot change ordinary unconditioned language logits.
    """
    trainable = []
    frozen = []
    for name, p in lm.named_parameters():
        active = name.startswith("controller.") or ".cond_delta." in name or ".cond_gate." in name
        p.requires_grad_(active)
        (trainable if active else frozen).append((name, p.numel()))
    print(f"personal_interface_trainable={sum(n for _,n in trainable):,} frozen_language_params={sum(n for _,n in frozen):,}")
    return [n for n, _ in trainable]


def snapshot_frozen_language(lm):
    return {
        name: p.detach().cpu().clone()
        for name, p in lm.named_parameters()
        if not p.requires_grad
    }


def frozen_language_unchanged(lm, snap):
    current = dict(lm.named_parameters())
    return all(torch.equal(v, current[k].detach().cpu()) for k, v in snap.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--personal-warmup", type=int, default=220)
    ap.add_argument("--joint-steps", type=int, default=180)
    ap.add_argument("--adapt-steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=8e-4)
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

    interface_names = freeze_language_backbone(lm)
    frozen_snapshot = snapshot_frozen_language(lm)
    tl.joint_train(lm, personal, profiles, args.device, args.joint_steps, args.lr)
    backbone_unchanged = frozen_language_unchanged(lm, frozen_snapshot)
    print(f"v06_language_backbone_unchanged={backbone_unchanged}")

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
        "language_backbone_unchanged": backbone_unchanged,
        "personal_interface_trainable_names": interface_names,
        "stage": "natural-hebrew-plus-personal-v0.6",
        "base_checkpoint": str(Path(args.base)),
        "tokenizer_file": args.tokenizer,
    }, args.save)
    print(f"saved={args.save}")
    if after < 0.95 or not ok or not backbone_unchanged:
        raise SystemExit(
            f"v0.6 personalization gate failed: accuracy={after:.3f} "
            f"freeze={ok} backbone_unchanged={backbone_unchanged}"
        )

if __name__ == "__main__":
    main()
