from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v15 import load_model
from personal_state import PersonalState
from train_hebrew import load_tokenizer


COMPONENTS = ("x", "a", "diff", "prod")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--text", required=True)
    ap.add_argument("--chars", type=int, default=50000)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_model(args.checkpoint, args.device)
    if ckpt.get("variant") != "self":
        raise ValueError("requires SELF checkpoint")
    model.eval()
    tok, _ = load_tokenizer(args.tokenizer)
    text = Path(args.text).read_text(encoding="utf-8")[:args.chars]
    ids = tok.encode(text, bos=True, eos=True)

    stats = []
    for li, adapter in enumerate(model.self_adapters):
        d = cfg.d_model
        chunks = adapter.down.weight.detach().split(d, dim=1)
        stats.append({
            "layer": li + 1,
            "tokens": 0,
            "weight_norms": {c: float(w.norm().cpu()) for c, w in zip(COMPONENTS, chunks)},
            "component_sq_sum": {c: 0.0 for c in COMPONENTS},
            "component_abs_sum": {c: 0.0 for c in COMPONENTS},
            "component_cos_sum": {c: 0.0 for c in COMPONENTS},
            "component_elements": {c: 0 for c in COMPONENTS},
            "delta_norm_sum": 0.0,
            "hidden_norm_sum": 0.0,
            "relative_delta_sum": 0.0,
            "x_anchor_cos_sum": 0.0,
            "x_anchor_distance_sum": 0.0,
            "product_norm_sum": 0.0,
        })

    anchor = model.self_anchor
    an = F.layer_norm(anchor, (anchor.shape[-1],))

    for start in range(0, len(ids) - 1, args.seq_len):
        chunk = ids[start:start + args.seq_len]
        if not chunk:
            continue
        input_ids = torch.tensor([chunk], dtype=torch.long, device=args.device)
        T = input_ids.shape[1]
        x = model.base.token_embedding(input_ids)
        pos = torch.arange(T, device=x.device)
        x = x + model.base.position_embedding(pos)[None, :, :]

        for li, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters)):
            x = block(x, None)
            xn = adapter.norm(x)
            a = an.view(1, 1, -1).expand_as(xn)
            values = {
                "x": xn,
                "a": a,
                "diff": xn - a,
                "prod": xn * a,
            }

            d = cfg.d_model
            weights = adapter.down.weight.split(d, dim=1)
            contrib = {
                c: F.linear(values[c], w)
                for c, w in zip(COMPONENTS, weights)
            }
            total = sum(contrib.values())
            raw_delta = adapter.up(F.gelu(total))
            gate = torch.sigmoid(adapter.gate_logit)
            delta = gate * raw_delta

            st = stats[li]
            npos = x.shape[0] * x.shape[1]
            st["tokens"] += npos
            total_norm = total.norm(dim=-1).clamp_min(1e-12)
            for c in COMPONENTS:
                v = contrib[c]
                st["component_sq_sum"][c] += float((v * v).sum().cpu())
                st["component_abs_sum"][c] += float(v.abs().sum().cpu())
                st["component_elements"][c] += v.numel()
                cos = (v * total).sum(dim=-1) / (v.norm(dim=-1).clamp_min(1e-12) * total_norm)
                st["component_cos_sum"][c] += float(cos.sum().cpu())

            hidden_norm = x.norm(dim=-1).clamp_min(1e-12)
            delta_norm = delta.norm(dim=-1)
            st["delta_norm_sum"] += float(delta_norm.sum().cpu())
            st["hidden_norm_sum"] += float(hidden_norm.sum().cpu())
            st["relative_delta_sum"] += float((delta_norm / hidden_norm).sum().cpu())
            st["x_anchor_cos_sum"] += float(F.cosine_similarity(xn, a, dim=-1).sum().cpu())
            st["x_anchor_distance_sum"] += float((xn - a).norm(dim=-1).sum().cpu())
            st["product_norm_sum"] += float((xn * a).norm(dim=-1).sum().cpu())

            x = x + delta

    layers = []
    for st in stats:
        n = max(st["tokens"], 1)
        layer = {
            "layer": st["layer"],
            "tokens": st["tokens"],
            "weight_norms": st["weight_norms"],
            "component_pre_gelu_rms": {},
            "component_pre_gelu_abs_mean": {},
            "component_alignment_with_total": {},
            "mean_hidden_norm": st["hidden_norm_sum"] / n,
            "mean_self_delta_norm": st["delta_norm_sum"] / n,
            "mean_self_delta_relative_to_hidden": st["relative_delta_sum"] / n,
            "mean_x_anchor_cosine": st["x_anchor_cos_sum"] / n,
            "mean_x_anchor_distance": st["x_anchor_distance_sum"] / n,
            "mean_elementwise_product_norm": st["product_norm_sum"] / n,
        }
        for c in COMPONENTS:
            elems = max(st["component_elements"][c], 1)
            layer["component_pre_gelu_rms"][c] = (st["component_sq_sum"][c] / elems) ** 0.5
            layer["component_pre_gelu_abs_mean"][c] = st["component_abs_sum"][c] / elems
            layer["component_alignment_with_total"][c] = st["component_cos_sum"][c] / n
        layers.append(layer)

    result = {
        "experiment": "v0.15 SELF 50M internal contribution profile",
        "tokens_seen": ckpt.get("tokens_seen"),
        "profile_chars": len(text),
        "anchor_norm": float(anchor.norm().cpu()),
        "layers": layers,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
