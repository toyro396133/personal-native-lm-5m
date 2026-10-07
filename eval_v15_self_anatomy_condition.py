from __future__ import annotations

import argparse
import json
import re
import types
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v15 import load_model, char_eval, generate, PROMPTS
from personal_state import PersonalState
from train_hebrew import load_tokenizer


COMPONENTS = ("x", "a", "diff", "prod")


def parse_layers(text: str, n_layers: int):
    if text == "early3":
        return set(range(min(3, n_layers)))
    if text == "late3":
        return set(range(max(0, n_layers - 3), n_layers))
    if text == "odd":
        return {i for i in range(n_layers) if (i + 1) % 2 == 1}
    if text == "even":
        return {i for i in range(n_layers) if (i + 1) % 2 == 0}
    m = re.fullmatch(r"l([1-9][0-9]*)", text)
    if not m:
        raise ValueError(f"bad layer selector: {text}")
    idx = int(m.group(1)) - 1
    if not 0 <= idx < n_layers:
        raise ValueError(f"layer out of range: {text}")
    return {idx}


def same_norm_direction_lerp(source: torch.Tensor, target: torch.Tensor, t: float):
    mixed = (1.0 - t) * source + t * target
    return mixed / mixed.norm().clamp_min(1e-12) * source.norm()


def deterministic_random_like(anchor: torch.Tensor):
    g = torch.Generator(device=anchor.device)
    g.manual_seed(20261007)
    r = torch.randn(anchor.shape, generator=g, device=anchor.device)
    return r / r.norm().clamp_min(1e-12) * anchor.norm()


def deterministic_shuffle(anchor: torch.Tensor):
    g = torch.Generator(device=anchor.device)
    g.manual_seed(20261007)
    perm = torch.randperm(anchor.numel(), generator=g, device=anchor.device)
    s = anchor[perm]
    return s / s.norm().clamp_min(1e-12) * anchor.norm()


def install_adapter_mode(adapter, mode: str):
    if mode == "off":
        def forward_off(self, x, self_anchor):
            return x
        adapter.forward = types.MethodType(forward_off, adapter)
        return

    if mode.startswith("component_off_"):
        component = mode.removeprefix("component_off_")
        keep = {c for c in COMPONENTS if c != component}
    elif mode.startswith("component_only_"):
        component = mode.removeprefix("component_only_")
        keep = {component}
    else:
        raise ValueError(mode)

    if not keep <= set(COMPONENTS):
        raise ValueError(f"bad component mode {mode}")

    def forward_masked(self, x, self_anchor):
        xn = self.norm(x)
        an = F.layer_norm(self_anchor, (self_anchor.shape[-1],))
        a = an.view(1, 1, -1).expand(xn.shape[0], xn.shape[1], -1)
        values = {
            "x": xn,
            "a": a,
            "diff": xn - a,
            "prod": xn * a,
        }
        relative = torch.cat(
            [values[c] if c in keep else torch.zeros_like(values[c]) for c in COMPONENTS],
            dim=-1,
        )
        delta = self.up(F.gelu(self.down(relative)))
        gate = torch.sigmoid(self.gate_logit)
        return x + gate * delta

    adapter.forward = types.MethodType(forward_masked, adapter)


@torch.no_grad()
def apply_condition(model, condition: str, original_anchor: torch.Tensor):
    n_layers = len(model.self_adapters)

    if condition == "normal":
        return
    if condition == "adapter_off":
        for adapter in model.self_adapters:
            install_adapter_mode(adapter, "off")
        return

    if condition.startswith("layer_off_"):
        selector = condition.removeprefix("layer_off_")
        for i in parse_layers(selector, n_layers):
            install_adapter_mode(model.self_adapters[i], "off")
        return

    if condition.startswith("group_off_"):
        selector = condition.removeprefix("group_off_")
        for i in parse_layers(selector, n_layers):
            install_adapter_mode(model.self_adapters[i], "off")
        return

    if condition.startswith("layer_anchor_shuffle_"):
        selector = condition.removeprefix("layer_anchor_shuffle_")
        selected = parse_layers(selector, n_layers)
        shuffled = deterministic_shuffle(original_anchor).detach().clone()
        for i in selected:
            adapter = model.self_adapters[i]
            original_forward = adapter.forward

            def forward_wrong_anchor(self, x, self_anchor, _orig=original_forward, _a=shuffled):
                return _orig(x, _a)
            adapter.forward = types.MethodType(forward_wrong_anchor, adapter)
        return

    m = re.fullmatch(r"layer_(l[1-9][0-9]*)_(component_(?:off|only)_(?:x|a|diff|prod))", condition)
    if m:
        selector, mode = m.groups()
        for i in parse_layers(selector, n_layers):
            install_adapter_mode(model.self_adapters[i], mode)
        return

    if condition.startswith("component_off_") or condition.startswith("component_only_"):
        for adapter in model.self_adapters:
            install_adapter_mode(adapter, condition)
        return

    if condition == "anchor_negate":
        model.self_anchor.copy_(-original_anchor)
        return
    if condition == "anchor_scale_half":
        model.self_anchor.copy_(0.5 * original_anchor)
        return
    if condition == "anchor_scale_double":
        model.self_anchor.copy_(2.0 * original_anchor)
        return
    if condition == "anchor_shift_const":
        model.self_anchor.copy_(original_anchor + torch.ones_like(original_anchor))
        return

    m = re.fullmatch(r"anchor_(random|shuffle)_t(25|50|75|100)", condition)
    if m:
        kind, pct = m.groups()
        t = int(pct) / 100.0
        target = deterministic_random_like(original_anchor) if kind == "random" else deterministic_shuffle(original_anchor)
        model.self_anchor.copy_(same_norm_direction_lerp(original_anchor, target, t))
        return

    raise ValueError(f"unknown condition: {condition}")


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
        cp = F.softmax(clogits, dim=-1)
        total_kl += float((rp * (rlp - clp)).sum(dim=-1).sum())
        total_l1 += float((rp - cp).abs().sum(dim=-1).sum())
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
    ap.add_argument("--condition", required=True)
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
        raise ValueError("requires a SELF checkpoint")
    model.eval()

    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    fine_text = Path(args.fineweb_val).read_text(encoding="utf-8")
    external_text = Path(args.external_val).read_text(encoding="utf-8")

    original_anchor = model.self_anchor.detach().clone()
    apply_condition(model, args.condition, original_anchor)

    _, _, reference = load_model(args.checkpoint, args.device)
    reference.eval()

    fine = char_eval(model, tok, fine_text[:args.chars], args.device)
    external = char_eval(model, tok, external_text[:args.chars], args.device)

    if args.condition == "normal":
        shift = {
            "tokens": 0,
            "mean_kl_from_normal": 0.0,
            "mean_probability_l1": 0.0,
            "argmax_change_fraction": 0.0,
        }
    else:
        shift = predictive_shift(reference, model, tok, fine_text, args.device, chars=args.shift_chars)

    neutral = PersonalState("neutral").flatten().unsqueeze(0).to(args.device)
    examples = []
    for prompt in PROMPTS:
        examples.append({
            "input": prompt,
            "greedy": generate(model, cfg, tok, prompt, neutral, 0.0),
            "sampled": generate(model, cfg, tok, prompt, neutral, 0.75),
        })

    result = {
        "experiment": "v0.15 SELF 50M parallel anatomy",
        "condition": args.condition,
        "tokens_seen": ckpt.get("tokens_seen"),
        "parameters": ckpt.get("parameter_count"),
        "base_parameters": ckpt.get("base_parameter_count"),
        "initial_base_sha256": ckpt.get("initial_base_sha256"),
        "tokenizer_kind": tokenizer_kind,
        "original_self_diagnostics": ckpt.get("self_diagnostics"),
        "fineweb2": fine,
        "external": external,
        "predictive_shift": shift,
        "language_examples": examples,
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "condition": args.condition,
        "fineweb_nats_per_char": fine["nats_per_char"],
        "external_nats_per_char": external["nats_per_char"],
        "predictive_shift": shift,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
