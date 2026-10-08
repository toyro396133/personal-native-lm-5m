from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v17 import load_any
from train_hebrew import load_tokenizer
from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from model_self_variants import SelfVariantPersonalNativeLM
from eval_self_core_organization import (
    CORE_TEMPLATES,
    centroid_classify,
    core_margin,
    delta_axis_consistency,
    effective_anchor_for_layer,
    encode_exact_length,
    fixed_rep_relative_cache,
    relation_consistency,
)


def norm(x):
    return F.normalize(x.float(), dim=-1)


def focus_relation_consistency(cache, pair_texts, triple_texts):
    per_focus = defaultdict(list)
    for (core, goal, focus), text in triple_texts.items():
        base = cache[pair_texts[(core, goal)]]
        triple = cache[text]
        d = norm(triple.unsqueeze(0))[0] - norm(base.unsqueeze(0))[0]
        if float(d.norm()) > 1e-9:
            per_focus[focus].append(d / d.norm())

    within = []
    centroids = {}
    for focus, ds in per_focus.items():
        if len(ds) < 2:
            continue
        centroids[focus] = norm(torch.stack(ds).mean(0, keepdim=True))[0]
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
        "same_focus_offset_cosine": wi,
        "different_focus_centroid_cosine": be,
        "relation_margin": wi - be,
    }


@torch.no_grad()
def encode_texts_with_anchor(model, tok, texts, device, replacement=None):
    groups = defaultdict(list)
    for text in sorted(set(texts)):
        ids = tok.encode(text, bos=True, eos=True)[:112]
        groups[len(ids)].append((text, ids))

    original = None
    if isinstance(model, SelfVariantPersonalNativeLM) and replacement is not None:
        original = model.self_anchor.detach().clone()
        model.self_anchor.copy_(replacement.to(model.self_anchor))

    caches = defaultdict(dict)
    try:
        for _, items in sorted(groups.items()):
            for start in range(0, len(items), 64):
                chunk = items[start:start + 64]
                ids = torch.tensor([x[1] for x in chunk], dtype=torch.long, device=device)
                by_layer = encode_exact_length(model, ids)
                for layer, reps in by_layer.items():
                    for (text, _), rep in zip(chunk, reps):
                        caches[layer][text] = rep
    finally:
        if original is not None:
            model.self_anchor.copy_(original)
    return dict(caches)


def unit_random_like(anchor, seed):
    a = anchor.detach().cpu().float()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    r = torch.randn(a.shape, generator=g)
    return r / r.norm().clamp_min(1e-12)


def unit_orthogonal_like(anchor, seed):
    a = anchor.detach().cpu().float()
    u = unit_random_like(a, seed)
    u = u - (u @ a) / a.square().sum().clamp_min(1e-12) * a
    return u / u.norm().clamp_min(1e-12)


def unit_shuffle_delta(anchor, seed):
    a = anchor.detach().cpu().float()
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    perm = torch.randperm(a.numel(), generator=g)
    d = a[perm] - a
    return d / d.norm().clamp_min(1e-12)


def intervention_set(anchor):
    a = anchor.detach().cpu().float()
    n = a.norm().clamp_min(1e-12)
    directions = {
        "radial": -a / n,
        "random": unit_random_like(a, 77191),
        "orthogonal": unit_orthogonal_like(a, 77201),
        "shuffle": unit_shuffle_delta(a, 77213),
    }

    out = {"normal": a.clone()}
    for scale, suffix in ((1.0, "1x"), (2.0, "2x")):
        for name, direction in directions.items():
            out[f"{name}_{suffix}"] = a + scale * n * direction
    return out


def intervention_geometry(anchor, replacement):
    a = anchor.detach().cpu().float()
    b = replacement.detach().cpu().float()
    d = b - a
    denom = a.norm() * b.norm()
    return {
        "anchor_norm": float(a.norm()),
        "replacement_norm": float(b.norm()),
        "delta_norm": float(d.norm()),
        "delta_over_anchor_norm": float(d.norm() / a.norm().clamp_min(1e-12)),
        "cosine_to_normal": None if float(denom) < 1e-12 else float((a @ b) / denom),
    }


def same_norm_shams(anchor, count=8):
    a = anchor.detach().cpu().float()
    out = {}
    g = torch.Generator(device="cpu")
    g.manual_seed(88001)
    perm = torch.randperm(a.numel(), generator=g)
    out["shuffle"] = a[perm].clone()
    for i in range(count):
        out[f"random_{i+1}"] = unit_random_like(a, 88100 + i) * a.norm()
    return out


def mean_sd(values):
    mean = sum(values) / max(1, len(values))
    sd = statistics.pstdev(values) if len(values) > 1 else 0.0
    return mean, sd


def learned_reference_specificity(model, layer, normal_cache, core_texts, pair_texts, triple_texts):
    raw = model.self_anchor.detach().cpu()
    candidates = {"learned": effective_anchor_for_layer(model, layer, raw)}
    for name, sham in same_norm_shams(raw).items():
        candidates[name] = effective_anchor_for_layer(model, layer, sham)

    primitive = {}
    for name, anchor in candidates.items():
        rel = fixed_rep_relative_cache(normal_cache, anchor)
        primitive[name] = {
            "core_margin": core_margin(rel, core_texts)["margin"],
            "goal_relation_margin": relation_consistency(rel, core_texts, pair_texts)["relation_margin"],
            "focus_relation_margin": focus_relation_consistency(rel, pair_texts, triple_texts)["relation_margin"],
        }

    shams = [v for k, v in primitive.items() if k != "learned"]
    advantages = {}
    for key in ("core_margin", "goal_relation_margin", "focus_relation_margin"):
        vals = [x[key] for x in shams]
        m, sd = mean_sd(vals)
        learned = primitive["learned"][key]
        advantages[key] = {
            "learned": learned,
            "sham_mean": m,
            "sham_sd": sd,
            "learned_minus_sham_mean": learned - m,
            "z_vs_shams": None if sd < 1e-12 else (learned - m) / sd,
        }
    return {"primitive": primitive, "advantages": advantages}


def core_accuracy(train_by_core, cache, core_texts):
    return centroid_classify(
        train_by_core,
        [(c, cache[core_texts[c][3]]) for c in sorted(core_texts)],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)
    model.eval()

    cores = list(TEST_CORES)
    goals = list(TEST_GOALS[:6])
    foci = list(TEST_FOCI[:6])

    core_texts = {c: [tpl.format(core=c) for tpl in CORE_TEMPLATES] for c in cores}
    pair_texts = {(c, g): f"הליבה היא {c}. היעד הנוכחי הוא {g}." for c in cores for g in goals}
    triple_texts = {
        (c, g, f): f"הליבה היא {c}. היעד הנוכחי הוא {g}. המוקד כעת הוא {f}."
        for c in cores for g in goals[:3] for f in foci[:3]
    }

    all_texts = []
    for xs in core_texts.values():
        all_texts.extend(xs)
    all_texts.extend(pair_texts.values())
    all_texts.extend(triple_texts.values())

    if isinstance(model, SelfVariantPersonalNativeLM):
        raw_anchor = model.self_anchor.detach().cpu()
        replacements = intervention_set(raw_anchor)
    else:
        raw_anchor = None
        replacements = {"normal": None}

    caches = {
        name: encode_texts_with_anchor(model, tok, all_texts, args.device, replacement)
        for name, replacement in replacements.items()
    }
    layers = list(caches["normal"])
    train_cores = cores[: len(cores)//2]
    test_cores = cores[len(cores)//2:]
    self_train_texts = [pair_texts[(c, g)] for c in train_cores for g in goals]
    self_test_texts = [pair_texts[(c, g)] for c in test_cores for g in goals]

    by_layer = {}
    for layer in layers:
        normal = caches["normal"][layer]
        train_by_core = {c: [normal[t] for t in core_texts[c][:3]] for c in cores}

        core_paraphrase = core_accuracy(train_by_core, normal, core_texts)
        core_under_goal = centroid_classify(
            train_by_core,
            [(c, normal[pair_texts[(c, g)]]) for c in cores for g in goals],
        )
        core_under_focus = centroid_classify(
            train_by_core,
            [(c, normal[triple_texts[(c, g, f)]]) for c in cores for g in goals[:3] for f in foci[:3]],
        )

        goal_rel = relation_consistency(normal, core_texts, pair_texts)
        focus_rel = focus_relation_consistency(normal, pair_texts, triple_texts)

        interventions = {}
        if isinstance(model, SelfVariantPersonalNativeLM):
            for name, replacement in replacements.items():
                if name == "normal":
                    continue
                alt = caches[name][layer]
                axis = delta_axis_consistency(
                    normal, alt, self_train_texts, self_test_texts
                )
                interventions[name] = {
                    "geometry": intervention_geometry(raw_anchor, replacement),
                    "core_accuracy": core_accuracy(train_by_core, alt, core_texts),
                    "core_accuracy_delta_from_normal": core_accuracy(train_by_core, alt, core_texts) - core_paraphrase,
                    "axis_consistency": axis["axis_consistency"],
                    "effect_norm": axis["mean_effect_norm"],
                }

        learned_ref = None
        if isinstance(model, SelfVariantPersonalNativeLM):
            learned_ref = learned_reference_specificity(
                model, layer, normal, core_texts, pair_texts, triple_texts
            )

        by_layer[layer] = {
            "core_paraphrase_accuracy": core_paraphrase,
            "core_under_goal_accuracy": core_under_goal,
            "core_under_focus_accuracy": core_under_focus,
            "focus_minus_goal_core_accuracy": core_under_focus - core_under_goal,
            "core_cluster_margin": core_margin(normal, core_texts)["margin"],
            "goal_relation_margin": goal_rel["relation_margin"],
            "focus_relation_margin": focus_rel["relation_margin"],
            "interventions": interventions,
            "learned_reference_specificity": learned_ref,
        }

    report = {
        "schema_version": 1,
        "experiment": "SELF CORE structural checkpoint audit v3",
        "checkpoint": args.checkpoint,
        "variant": ckpt.get("variant", "baseline"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "tokenizer_kind": tokenizer_kind,
        "primary_question": "Does the model organize representations around distinct SELF and CORE references, with selective relational structure, rather than merely improve language loss?",
        "interpretation_guardrails": [
            "Language loss is intentionally absent from the primary audit.",
            "Intervention direction is compared at matched displacement magnitude (1x and 2x anchor norm).",
            "Raw negate-like damage is not interpreted as sign sensitivity without magnitude-matched comparison.",
            "Intervention effect strength and learned-reference specificity are reported separately.",
            "CORE-under-GOAL vs CORE-under-FOCUS is conditional evidence, not direct causal proof of GOAL/FOCUS roles.",
            "Layerwise decodability/geometry is not called causal without activation patching.",
        ],
        "intervention_design": {
            "1x": "replacement is exactly one learned-anchor norm away from normal",
            "2x": "replacement is exactly two learned-anchor norms away from normal",
            "directions": ["radial", "random", "orthogonal", "shuffle-derived"],
            "radial_1x_equivalent": "zero anchor",
            "radial_2x_equivalent": "negated anchor",
        },
        "layers": by_layer,
        "final": by_layer["final"],
        "self_diagnostics": model.self_diagnostics() if isinstance(model, SelfVariantPersonalNativeLM) else None,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "variant": report["variant"],
        "tokens_seen": report["tokens_seen"],
        "final_core_under_goal": report["final"]["core_under_goal_accuracy"],
        "final_core_under_focus": report["final"]["core_under_focus_accuracy"],
        "final_focus_minus_goal": report["final"]["focus_minus_goal_core_accuracy"],
        "has_self": isinstance(model, SelfVariantPersonalNativeLM),
        "out": args.out,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
