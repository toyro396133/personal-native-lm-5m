from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_causal_activation_patching_300m import (
    build_centroids,
    encode,
    prompt,
    predict,
    unit,
)
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


def radial2(anchor: torch.Tensor) -> torch.Tensor:
    return -anchor.detach().cpu().float()


def random2(anchor: torch.Tensor, seed: int) -> torch.Tensor:
    a = anchor.detach().cpu().float()
    n = a.norm().clamp_min(1e-12)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    u = torch.randn(a.shape, generator=g)
    u = u / u.norm().clamp_min(1e-12)
    return a + 2.0 * n * u


def init_stream(model: SelfVariantPersonalNativeLM, ids: torch.Tensor):
    x = model.base.token_embedding(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)
    return x + model.base.position_embedding(pos)[None, :, :]


@torch.no_grad()
def forward_capture(
    model: SelfVariantPersonalNativeLM,
    ids: torch.Tensor,
    layer: int,
    replacement: torch.Tensor | None,
):
    x = init_stream(model, ids)
    local = None
    for i, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters), 1):
        x = block(x, None)
        anchor = (
            replacement.to(model.self_anchor)
            if i == layer and replacement is not None
            else model.self_anchor
        )
        x = adapter(x, anchor)
        if i == layer:
            local = F.layer_norm(x.float(), (x.shape[-1],)).mean(dim=1)
    x = model.base.final_norm(x)
    final = x.float().mean(dim=1)
    return local, final


@torch.no_grad()
def encode_condition(model, tok, triples, device, layer, replacement):
    groups = defaultdict(list)
    for c, g, f, text in triples:
        ids = encode(tok, text)
        groups[len(ids)].append((c, g, f, text, ids))

    local = {}
    final = {}
    for length in sorted(groups):
        items = groups[length]
        for start in range(0, len(items), 48):
            chunk = items[start:start + 48]
            ids = torch.tensor([x[4] for x in chunk], dtype=torch.long, device=device)
            l, y = forward_capture(model, ids, layer, replacement)
            l = l.detach().cpu()
            y = y.detach().cpu()
            for row, lr, yr in zip(chunk, l, y):
                text = row[3]
                local[text] = lr
                final[text] = yr
    return {"local": local, "final": final}


def delta_cache(normal: dict[str, torch.Tensor], alt: dict[str, torch.Tensor]):
    out = {}
    norms = []
    for text in normal:
        a = unit(normal[text].unsqueeze(0))[0]
        b = unit(alt[text].unsqueeze(0))[0]
        d = b - a
        norms.append(float(d.norm()))
        out[text] = d
    return out, {
        "mean_effect_norm": sum(norms) / len(norms),
        "median_effect_norm": statistics.median(norms),
        "min_effect_norm": min(norms),
        "max_effect_norm": max(norms),
    }


def stable_split(component: str, c: str, g: str, f: str) -> int:
    if component == "core":
        key = f"{g}|{f}"
    elif component == "goal":
        key = f"{c}|{f}"
    else:
        key = f"{c}|{g}"
    h = hashlib.sha1((component + "|" + key).encode("utf-8")).digest()
    return h[0] & 1


def cross_context_accuracy(delta, triples, component: str):
    idx = {"core": 0, "goal": 1, "focus": 2}[component]
    train = defaultdict(list)
    test = []
    for row in triples:
        c, g, f, text = row
        d = delta[text].float()
        if float(d.norm()) < 1e-9:
            continue
        d = d / d.norm()
        label = row[idx]
        if stable_split(component, c, g, f) == 0:
            train[label].append(d)
        else:
            test.append((label, d))

    labels = sorted(train)
    if any(len(train[x]) == 0 for x in labels) or not test:
        return None
    centroids = {
        label: unit(torch.stack(train[label]).mean(0, keepdim=True))[0]
        for label in labels
    }
    hits = 0
    valid = 0
    for label, d in test:
        if label not in centroids:
            continue
        pred = predict(d, centroids)
        hits += int(pred == label)
        valid += 1
    return {
        "accuracy": hits / max(1, valid),
        "test_n": valid,
        "label_count": len(centroids),
        "chance": 1.0 / max(1, len(centroids)),
    }


def cluster_margin(delta, triples, component: str):
    idx = {"core": 0, "goal": 1, "focus": 2}[component]
    per = defaultdict(list)
    for row in triples:
        d = delta[row[3]].float()
        if float(d.norm()) < 1e-9:
            continue
        per[row[idx]].append(d / d.norm())

    within = []
    centroids = {}
    for label, ds in per.items():
        if len(ds) < 2:
            continue
        centroids[label] = unit(torch.stack(ds).mean(0, keepdim=True))[0]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                within.append(float(ds[i] @ ds[j]))

    between = []
    labels = sorted(centroids)
    for i, a in enumerate(labels):
        for b in labels[i + 1:]:
            between.append(float(centroids[a] @ centroids[b]))

    wi = sum(within) / max(1, len(within))
    be = sum(between) / max(1, len(between))
    return {
        "within_cosine": wi,
        "between_centroid_cosine": be,
        "margin": wi - be,
    }


def field_metrics(delta, triples):
    out = {}
    for component in ("core", "goal", "focus"):
        out[component] = {
            "cross_context": cross_context_accuracy(delta, triples, component),
            "cluster": cluster_margin(delta, triples, component),
        }
    return out


def mean_sd(values):
    m = statistics.mean(values)
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return m, sd


def compare_radial_to_shams(radial_metrics, sham_metrics):
    out = {}
    for component in ("core", "goal", "focus"):
        out[component] = {}
        for metric_name, getter in (
            (
                "cross_context_accuracy",
                lambda x: x[component]["cross_context"]["accuracy"],
            ),
            (
                "cluster_margin",
                lambda x: x[component]["cluster"]["margin"],
            ),
        ):
            radial = getter(radial_metrics)
            shams = [getter(x) for x in sham_metrics]
            m, sd = mean_sd(shams)
            out[component][metric_name] = {
                "radial": radial,
                "random_mean": m,
                "random_sd": sd,
                "radial_minus_random_mean": radial - m,
                "z_vs_random": None if sd < 1e-12 else (radial - m) / sd,
                "radial_rank_among_9_desc": (
                    1 + sum(v > radial for v in shams)
                ),
                "random_values": shams,
            }
    return out


def decode_component_accuracy(reps, triples, clean_centroids):
    hits = {"core": 0, "goal": 0, "focus": 0}
    for c, g, f, text in triples:
        hits["core"] += int(predict(reps[text], clean_centroids["core"]) == c)
        hits["goal"] += int(predict(reps[text], clean_centroids["goal"]) == g)
        hits["focus"] += int(predict(reps[text], clean_centroids["focus"]) == f)
    n = len(triples)
    return {k: v / n for k, v in hits.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--layers", required=True, help="Comma-separated adapter layers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-shams", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("SELF-by-CORE interaction assay requires a SELF variant")
    model.eval()

    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if any(x < 1 or x > cfg.n_layers for x in layers):
        raise ValueError(layers)

    cores = list(TEST_CORES)
    goals = list(TEST_GOALS[:6])
    foci = list(TEST_FOCI[:6])
    triples = [
        (c, g, f, prompt(c, g, f))
        for c in cores for g in goals for f in foci
    ]

    anchor = model.self_anchor.detach().cpu().float()
    radial = radial2(anchor)
    random_replacements = [
        random2(anchor, 750_001 + i * 101)
        for i in range(args.random_shams)
    ]

    report_layers = {}
    for layer in layers:
        print(f"[layer {layer}] normal", flush=True)
        normal = encode_condition(
            model, tok, triples, args.device, layer, replacement=None
        )
        clean_centroids = build_centroids(
            normal["final"],
            triples,
        )

        print(f"[layer {layer}] radial2", flush=True)
        radial_cache = encode_condition(
            model, tok, triples, args.device, layer, replacement=radial
        )

        radial_local_delta, radial_local_effect = delta_cache(
            normal["local"], radial_cache["local"]
        )
        radial_final_delta, radial_final_effect = delta_cache(
            normal["final"], radial_cache["final"]
        )
        radial_local_metrics = field_metrics(radial_local_delta, triples)
        radial_final_metrics = field_metrics(radial_final_delta, triples)

        random_runs = []
        for i, replacement in enumerate(random_replacements, 1):
            print(f"[layer {layer}] random sham {i}/{len(random_replacements)}", flush=True)
            alt = encode_condition(
                model, tok, triples, args.device, layer, replacement=replacement
            )
            local_delta, local_effect = delta_cache(normal["local"], alt["local"])
            final_delta, final_effect = delta_cache(normal["final"], alt["final"])
            random_runs.append({
                "index": i,
                "local_effect": local_effect,
                "final_effect": final_effect,
                "local_metrics": field_metrics(local_delta, triples),
                "final_metrics": field_metrics(final_delta, triples),
                "component_accuracy": decode_component_accuracy(
                    alt["final"], triples, clean_centroids
                ),
            })

        report_layers[str(layer)] = {
            "normal_component_accuracy": decode_component_accuracy(
                normal["final"], triples, clean_centroids
            ),
            "radial_component_accuracy": decode_component_accuracy(
                radial_cache["final"], triples, clean_centroids
            ),
            "radial_local_effect": radial_local_effect,
            "radial_final_effect": radial_final_effect,
            "radial_local_field": radial_local_metrics,
            "radial_final_field": radial_final_metrics,
            "random_shams": random_runs,
            "radial_vs_random_local": compare_radial_to_shams(
                radial_local_metrics,
                [x["local_metrics"] for x in random_runs],
            ),
            "radial_vs_random_final": compare_radial_to_shams(
                radial_final_metrics,
                [x["final_metrics"] for x in random_runs],
            ),
            "random_component_accuracy_mean": {
                comp: statistics.mean(
                    x["component_accuracy"][comp] for x in random_runs
                )
                for comp in ("core", "goal", "focus")
            },
        }

    result = {
        "schema_version": 1,
        "experiment": "300M SELF by CORE interaction field v3",
        "variant": ckpt.get("variant"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "layers": report_layers,
        "random_shams": args.random_shams,
        "primary_question": (
            "Does the causal effect field induced by changing SELF depend "
            "systematically on which CORE is present, beyond generic context "
            "dependence from random same-displacement anchor changes?"
        ),
        "primary_metric": (
            "Cross-context CORE decoding and CORE cluster margin from the "
            "SELF-induced delta field, radial direction versus eight random "
            "same-displacement anchor directions."
        ),
        "guardrails": [
            "All random anchor replacements are exactly two anchor norms away from normal, matching the radial intervention displacement.",
            "CORE decoding from the SELF-effect field trains and tests across disjoint GOAL/FOCUS contexts.",
            "A generic nonlinear context-dependent perturbation is controlled by eight random same-displacement directions.",
            "Interaction evidence does not establish semantic selfhood.",
            "Single-seed checkpoint results require replication.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "variant": result["variant"],
        "tokens_seen": result["tokens_seen"],
        "layers": layers,
        "out": args.out,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
