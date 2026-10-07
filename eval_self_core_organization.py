from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from eval_v17 import load_any
from train_hebrew import load_tokenizer
from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from model_self_variants import SelfVariantPersonalNativeLM


SELF_MODES = ("normal", "zero", "negate", "shuffle")
CORE_TEMPLATES = (
    "הליבה היא: {core}",
    "העיקרון המרכזי הוא: {core}",
    "גם כאשר הפרטים משתנים, חשוב {core}",
    "נקודת הייחוס המרכזית של המשימה היא {core}",
)


def anchor_variant(anchor: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "normal":
        return anchor.clone()
    if mode == "zero":
        return torch.zeros_like(anchor)
    if mode == "negate":
        return -anchor
    if mode == "shuffle":
        g = torch.Generator(device="cpu")
        g.manual_seed(99173)
        perm = torch.randperm(anchor.numel(), generator=g).to(anchor.device)
        return anchor[perm]
    raise ValueError(mode)


@torch.no_grad()
def encode_exact_length(model, ids: torch.Tensor):
    """Return pooled representations after every transformer block and final norm."""
    out = {}
    if isinstance(model, SelfVariantPersonalNativeLM):
        base = model.base
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = x + base.position_embedding(pos)[None, :, :]
        for i, (block, adapter) in enumerate(zip(base.blocks, model.self_adapters), 1):
            x = block(x, None)
            x = adapter(x, model.self_anchor)
            pooled = F.layer_norm(x, (x.shape[-1],)).mean(dim=1)
            out[f"layer_{i}"] = pooled.detach().cpu()
        x = base.final_norm(x)
    else:
        base = model
        x = base.token_embedding(ids)
        pos = torch.arange(ids.shape[1], device=ids.device)
        x = x + base.position_embedding(pos)[None, :, :]
        for i, block in enumerate(base.blocks, 1):
            x = block(x, None)
            pooled = F.layer_norm(x, (x.shape[-1],)).mean(dim=1)
            out[f"layer_{i}"] = pooled.detach().cpu()
        x = base.final_norm(x)
    out["final"] = x.mean(dim=1).detach().cpu()
    return out


@torch.no_grad()
def encode_texts(model, tok, texts, device, self_mode):
    groups = defaultdict(list)
    for text in sorted(set(texts)):
        ids = tok.encode(text, bos=True, eos=True)[:112]
        groups[len(ids)].append((text, ids))

    original = None
    if isinstance(model, SelfVariantPersonalNativeLM):
        original = model.self_anchor.detach().clone()
        model.self_anchor.copy_(anchor_variant(original, self_mode))

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


def norm(x):
    return F.normalize(x.float(), dim=-1)


def centroid_classify(train_by_label, test_items):
    labels = sorted(train_by_label)
    centroids = torch.stack([
        norm(torch.stack(train_by_label[label]).mean(0, keepdim=True))[0]
        for label in labels
    ])
    correct = 0
    total = 0
    for label, rep in test_items:
        scores = norm(rep.unsqueeze(0))[0] @ centroids.T
        pred = labels[int(scores.argmax())]
        correct += int(pred == label)
        total += 1
    return correct / max(1, total)


def core_margin(cache, core_texts):
    within = []
    between = []
    reps = {
        c: [norm(cache[t].unsqueeze(0))[0] for t in texts]
        for c, texts in core_texts.items()
    }
    cores = sorted(reps)
    for c in cores:
        xs = reps[c]
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                within.append(float(xs[i] @ xs[j]))
    for i, c1 in enumerate(cores):
        for c2 in cores[i + 1:]:
            for x in reps[c1]:
                for y in reps[c2]:
                    between.append(float(x @ y))
    return {
        "within_cosine": sum(within) / max(1, len(within)),
        "between_cosine": sum(between) / max(1, len(between)),
        "margin": (
            sum(within) / max(1, len(within))
            - sum(between) / max(1, len(between))
        ),
    }


def delta_axis_consistency(normal_cache, alt_cache, train_texts, test_texts):
    train_d = []
    mags = []
    for t in train_texts:
        a = norm(normal_cache[t].unsqueeze(0))[0]
        b = norm(alt_cache[t].unsqueeze(0))[0]
        d = a - b
        train_d.append(d)
        mags.append(float(d.norm()))
    axis = torch.stack(train_d).mean(0)
    axis_norm = float(axis.norm())
    if axis_norm < 1e-9:
        return {"axis_consistency": 0.0, "mean_effect_norm": 0.0}
    axis = axis / axis.norm()

    cos = []
    for t in test_texts:
        a = norm(normal_cache[t].unsqueeze(0))[0]
        b = norm(alt_cache[t].unsqueeze(0))[0]
        d = a - b
        mags.append(float(d.norm()))
        if float(d.norm()) < 1e-9:
            cos.append(0.0)
        else:
            cos.append(float((d / d.norm()) @ axis))
    return {
        "axis_consistency": sum(cos) / max(1, len(cos)),
        "mean_effect_norm": sum(mags) / max(1, len(mags)),
    }


def relation_consistency(cache, core_texts, pair_texts):
    # Does adding the same GOAL create a similar displacement from different COREs?
    per_goal = defaultdict(list)
    for (core, goal), text in pair_texts.items():
        base = cache[core_texts[core][0]]
        pair = cache[text]
        d = norm(pair.unsqueeze(0))[0] - norm(base.unsqueeze(0))[0]
        if float(d.norm()) > 1e-9:
            per_goal[goal].append(d / d.norm())

    within = []
    centroids = {}
    for goal, ds in per_goal.items():
        if len(ds) < 2:
            continue
        centroids[goal] = norm(torch.stack(ds).mean(0, keepdim=True))[0]
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                within.append(float(ds[i] @ ds[j]))

    between = []
    goals = sorted(centroids)
    for i, g1 in enumerate(goals):
        for g2 in goals[i + 1:]:
            between.append(float(centroids[g1] @ centroids[g2]))

    return {
        "same_goal_offset_cosine": sum(within) / max(1, len(within)),
        "different_goal_centroid_cosine": sum(between) / max(1, len(between)),
        "relation_margin": (
            sum(within) / max(1, len(within))
            - sum(between) / max(1, len(between))
        ),
    }



def fixed_rep_relative_cache(cache, anchor):
    """Express fixed representations in coordinates relative to a candidate anchor."""
    out = {}
    a = F.layer_norm(anchor.float(), (anchor.numel(),))
    for text, rep in cache.items():
        x = F.layer_norm(rep.float(), (rep.numel(),))
        out[text] = x - a
    return out


def effective_anchor_for_layer(model, layer, base_anchor):
    if not isinstance(model, SelfVariantPersonalNativeLM):
        return None
    if layer == "final":
        adapter = model.self_adapters[-1]
    else:
        idx = int(layer.split("_")[1]) - 1
        adapter = model.self_adapters[idx]
    return adapter._anchor_for_layer(base_anchor).detach().cpu()


def random_like(anchor, seed=44117):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    r = torch.randn(anchor.shape, generator=g)
    return r / r.norm().clamp_min(1e-8) * anchor.detach().cpu().norm()


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

    core_texts = {
        c: [tpl.format(core=c) for tpl in CORE_TEMPLATES]
        for c in cores
    }
    pair_texts = {
        (c, g): f"הליבה היא {c}. היעד הנוכחי הוא {g}."
        for c in cores for g in goals
    }
    triple_texts = {
        (c, g, f): f"הליבה היא {c}. היעד הנוכחי הוא {g}. המוקד כעת הוא {f}."
        for c in cores for g in goals[:3] for f in foci[:3]
    }

    all_texts = []
    for xs in core_texts.values():
        all_texts.extend(xs)
    all_texts.extend(pair_texts.values())
    all_texts.extend(triple_texts.values())

    caches = {
        mode: encode_texts(model, tok, all_texts, args.device, mode)
        for mode in SELF_MODES
    }

    layers = list(caches["normal"])
    by_layer = {}

    train_cores = cores[: len(cores)//2]
    test_cores = cores[len(cores)//2:]
    self_train_texts = [
        pair_texts[(c, g)] for c in train_cores for g in goals
    ]
    self_test_texts = [
        pair_texts[(c, g)] for c in test_cores for g in goals
    ]

    for layer in layers:
        normal = caches["normal"][layer]

        # CORE identity from paraphrases: train on three phrasings, test on the fourth.
        train_by_core = {
            c: [normal[t] for t in core_texts[c][:3]]
            for c in cores
        }
        core_paraphrase = centroid_classify(
            train_by_core,
            [(c, normal[core_texts[c][3]]) for c in cores],
        )

        # Does CORE remain decodable while GOAL/FOCUS vary?
        core_under_goal = centroid_classify(
            train_by_core,
            [(c, normal[pair_texts[(c, g)]]) for c in cores for g in goals],
        )
        core_under_focus = centroid_classify(
            train_by_core,
            [
                (c, normal[triple_texts[(c, g, f)]])
                for c in cores for g in goals[:3] for f in foci[:3]
            ],
        )

        # Does CORE survive direct SELF perturbations?
        core_under_self = {}
        for mode in SELF_MODES[1:]:
            alt = caches[mode][layer]
            core_under_self[mode] = centroid_classify(
                train_by_core,
                [(c, alt[core_texts[c][3]]) for c in cores],
            )

        # Is there a reusable SELF-induced axis across unseen COREs?
        self_axes = {}
        for mode in SELF_MODES[1:]:
            self_axes[mode] = delta_axis_consistency(
                normal,
                caches[mode][layer],
                self_train_texts,
                self_test_texts,
            )

        cm = core_margin(normal, core_texts)
        rel = relation_consistency(normal, core_texts, pair_texts)

        learned_reference = None
        if isinstance(model, SelfVariantPersonalNativeLM):
            raw_anchor = model.self_anchor.detach().cpu()
            shuffled = anchor_variant(raw_anchor, "shuffle").cpu()
            random_anchor = random_like(raw_anchor)

            effective = {
                "learned": effective_anchor_for_layer(model, layer, raw_anchor),
                "shuffle": effective_anchor_for_layer(model, layer, shuffled),
                "random": effective_anchor_for_layer(model, layer, random_anchor),
            }

            relative = {
                k: fixed_rep_relative_cache(normal, a)
                for k, a in effective.items()
            }
            relative_core = {
                k: core_margin(v, core_texts)
                for k, v in relative.items()
            }
            relative_goal = {
                k: relation_consistency(v, core_texts, pair_texts)
                for k, v in relative.items()
            }

            random_core_mean = (
                relative_core["shuffle"]["margin"]
                + relative_core["random"]["margin"]
            ) / 2.0
            random_goal_mean = (
                relative_goal["shuffle"]["relation_margin"]
                + relative_goal["random"]["relation_margin"]
            ) / 2.0

            learned_reference = {
                "relative_core_margin": {
                    k: v["margin"] for k, v in relative_core.items()
                },
                "relative_goal_relation_margin": {
                    k: v["relation_margin"] for k, v in relative_goal.items()
                },
                "learned_core_reference_advantage":
                    relative_core["learned"]["margin"] - random_core_mean,
                "learned_goal_reference_advantage":
                    relative_goal["learned"]["relation_margin"] - random_goal_mean,
            }

        chance_core = 1.0 / len(cores)
        def chance_norm(acc):
            return max(0.0, min(1.0, (acc - chance_core) / (1.0 - chance_core)))

        core_structure = (
            chance_norm(core_paraphrase)
            + chance_norm(core_under_goal)
            + chance_norm(core_under_focus)
            + sum(chance_norm(v) for v in core_under_self.values()) / len(core_under_self)
        ) / 4.0

        mean_axis = sum(v["axis_consistency"] for v in self_axes.values()) / len(self_axes)
        mean_effect = sum(v["mean_effect_norm"] for v in self_axes.values()) / len(self_axes)
        # Descriptive, not a claim of semantic SELF: it rewards a reusable SELF axis
        # while requiring CORE identity to remain stable under SELF perturbation.
        core_invariance = sum(core_under_self.values()) / len(core_under_self)
        two_anchor_factorization = max(0.0, mean_axis) * chance_norm(core_invariance)

        by_layer[layer] = {
            "core_paraphrase_accuracy": core_paraphrase,
            "core_under_goal_accuracy": core_under_goal,
            "core_under_focus_accuracy": core_under_focus,
            "core_under_self_perturbation_accuracy": core_under_self,
            "core_cluster": cm,
            "goal_offset_relative_to_core": rel,
            "learned_self_reference": learned_reference,
            "self_axis": self_axes,
            "mean_self_axis_consistency": mean_axis,
            "mean_self_effect_norm": mean_effect,
            "mean_core_invariance_to_self": core_invariance,
            "core_structure_score": core_structure,
            "two_anchor_factorization_score": two_anchor_factorization,
        }

    best_layer = max(
        by_layer,
        key=lambda k: by_layer[k]["two_anchor_factorization_score"],
    )

    report = {
        "experiment": "SELF/CORE frozen representation organization assay v1",
        "checkpoint": args.checkpoint,
        "variant": ckpt.get("variant", "baseline"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "stage": ckpt.get("stage"),
        "tokenizer_kind": tokenizer_kind,
        "definitions": {
            "core_structure": (
                "CORE identity across paraphrases, changing GOAL/FOCUS, and SELF perturbations"
            ),
            "self_axis_consistency": (
                "whether changing SELF produces a reusable representation-direction "
                "that generalizes to unseen COREs"
            ),
            "two_anchor_factorization": (
                "SELF-axis consistency multiplied by CORE invariance to SELF perturbation; "
                "descriptive geometry score, not proof of semantic selfhood"
            ),
            "learned_self_reference_advantage": (
                "on fixed normal hidden states, asks whether coordinates relative to the "
                "specific learned SELF organize CORE / CORE->GOAL relations better than "
                "equally-sized shuffled or random anchors; this separates learned reference "
                "geometry from effects guaranteed by adapter wiring"
            ),
        },
        "chance_core_accuracy": 1.0 / len(cores),
        "layers": by_layer,
        "final": by_layer["final"],
        "best_factorization_layer": best_layer,
        "best_factorization": by_layer[best_layer]["two_anchor_factorization_score"],
    }
    if isinstance(model, SelfVariantPersonalNativeLM):
        report["self_diagnostics"] = model.self_diagnostics()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
