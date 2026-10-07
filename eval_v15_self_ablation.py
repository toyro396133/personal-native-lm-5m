from __future__ import annotations

import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v15 import load_model, char_eval, generate, PROMPTS
from personal_state import PersonalState
from train_hebrew import load_tokenizer


MODES = ("normal", "adapter_off", "anchor_zero", "anchor_random", "anchor_shuffle")


@torch.no_grad()
def set_mode(model, mode: str, original_anchor: torch.Tensor):
    if mode == "normal":
        model.self_anchor.copy_(original_anchor)
        return model

    if mode == "adapter_off":
        # The SELF checkpoint's own trained base weights, with only the
        # entire SELF residual path removed.
        return model.base

    if mode == "anchor_zero":
        model.self_anchor.zero_()
        return model

    if mode == "anchor_random":
        g = torch.Generator(device=original_anchor.device)
        g.manual_seed(20261007)
        r = torch.randn(original_anchor.shape, generator=g, device=original_anchor.device)
        r = r / r.norm().clamp_min(1e-12) * original_anchor.norm()
        model.self_anchor.copy_(r)
        return model

    if mode == "anchor_shuffle":
        g = torch.Generator(device=original_anchor.device)
        g.manual_seed(20261007)
        perm = torch.randperm(original_anchor.numel(), generator=g, device=original_anchor.device)
        model.self_anchor.copy_(original_anchor[perm])
        return model

    raise ValueError(mode)


@torch.no_grad()
def predictive_shift(reference, candidate, tok, text: str, device: str, chars: int = 30000, seq_len: int = 128):
    text = text[:chars]
    ids = tok.encode(text, bos=True, eos=True)
    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(device)
    total_kl = 0.0
    total_l1 = 0.0
    changed_argmax = 0
    tokens = 0

    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            continue
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        rlogits, _ = reference(x, neutral)
        clogits, _ = candidate(x, neutral)

        rp = F.softmax(rlogits, dim=-1)
        rlp = F.log_softmax(rlogits, dim=-1)
        clp = F.log_softmax(clogits, dim=-1)
        total_kl += float((rp * (rlp - clp)).sum(dim=-1).sum())
        total_l1 += float((rp - F.softmax(clogits, dim=-1)).abs().sum(dim=-1).sum())
        changed_argmax += int((rlogits.argmax(-1) != clogits.argmax(-1)).sum())
        tokens += x.numel()

    return {
        "tokens": tokens,
        "mean_kl_from_normal": total_kl / max(tokens, 1),
        "mean_probability_l1": total_l1 / max(tokens, 1),
        "argmax_change_fraction": changed_argmax / max(tokens, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--fineweb-val", required=True)
    ap.add_argument("--external-val", required=True)
    ap.add_argument("--chars", type=int, default=300000)
    ap.add_argument("--shift-chars", type=int, default=30000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_model(args.checkpoint, args.device)
    if ckpt.get("variant") != "self":
        raise ValueError("ablation requires a SELF checkpoint")

    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    fine_text = Path(args.fineweb_val).read_text(encoding="utf-8")
    external_text = Path(args.external_val).read_text(encoding="utf-8")

    original_anchor = model.self_anchor.detach().clone()
    original_diag = deepcopy(model.self_diagnostics())
    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(args.device)

    # Freeze a fully normal reference model for direct predictive-shift tests.
    _, _, reference = load_model(args.checkpoint, args.device)
    reference.eval()

    result = {
        "experiment": "v0.15 SELF causal ablation at 50M",
        "checkpoint": args.checkpoint,
        "tokens_seen": ckpt.get("tokens_seen"),
        "parameters": ckpt.get("parameter_count"),
        "base_parameters": ckpt.get("base_parameter_count"),
        "extra_parameters": ckpt.get("parameter_count", 0) - ckpt.get("base_parameter_count", 0),
        "initial_base_sha256": ckpt.get("initial_base_sha256"),
        "tokenizer_kind": tokenizer_kind,
        "original_self_diagnostics": original_diag,
        "modes": {},
    }

    for mode in MODES:
        # Restore learned anchor before configuring each independent condition.
        with torch.no_grad():
            model.self_anchor.copy_(original_anchor)
        candidate = set_mode(model, mode, original_anchor)
        candidate.eval()

        fine = char_eval(candidate, tok, fine_text[:args.chars], args.device)
        external = char_eval(candidate, tok, external_text[:args.chars], args.device)

        examples = []
        for prompt in PROMPTS:
            examples.append({
                "input": prompt,
                "greedy": generate(candidate, cfg, tok, prompt, neutral, 0.0),
                "sampled": generate(candidate, cfg, tok, prompt, neutral, 0.75),
            })

        if mode == "normal":
            shift = {
                "tokens": 0,
                "mean_kl_from_normal": 0.0,
                "mean_probability_l1": 0.0,
                "argmax_change_fraction": 0.0,
            }
        else:
            shift = predictive_shift(
                reference, candidate, tok, fine_text, args.device,
                chars=args.shift_chars,
            )

        result["modes"][mode] = {
            "fineweb2": fine,
            "external": external,
            "predictive_shift": shift,
            "language_examples": examples,
        }

    normal = result["modes"]["normal"]
    for mode, data in result["modes"].items():
        data["delta_vs_normal"] = {
            "fineweb_nats_per_char": data["fineweb2"]["nats_per_char"] - normal["fineweb2"]["nats_per_char"],
            "fineweb_relative": data["fineweb2"]["nats_per_char"] / normal["fineweb2"]["nats_per_char"] - 1.0,
            "external_nats_per_char": data["external"]["nats_per_char"] - normal["external"]["nats_per_char"],
            "external_relative": data["external"]["nats_per_char"] / normal["external"]["nats_per_char"] - 1.0,
        }

    # Interpretation helpers: not a conclusion, just machine-readable contrasts.
    off = result["modes"]["adapter_off"]["delta_vs_normal"]
    zero = result["modes"]["anchor_zero"]["delta_vs_normal"]
    rand = result["modes"]["anchor_random"]["delta_vs_normal"]
    shuf = result["modes"]["anchor_shuffle"]["delta_vs_normal"]
    result["contrasts"] = {
        "whole_self_path_fineweb_penalty": off["fineweb_nats_per_char"],
        "zero_anchor_fineweb_penalty": zero["fineweb_nats_per_char"],
        "random_anchor_fineweb_penalty": rand["fineweb_nats_per_char"],
        "shuffled_anchor_fineweb_penalty": shuf["fineweb_nats_per_char"],
        "whole_self_path_external_penalty": off["external_nats_per_char"],
        "zero_anchor_external_penalty": zero["external_nats_per_char"],
        "random_anchor_external_penalty": rand["external_nats_per_char"],
        "shuffled_anchor_external_penalty": shuf["external_nats_per_char"],
    }

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
