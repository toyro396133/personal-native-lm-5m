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
from eval_v17 import load_any
from model_self_variants import SelfVariantPersonalNativeLM
from train_hebrew import load_tokenizer


COMPONENTS = ("core", "goal", "focus")


def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.float(), dim=-1)


def prompt(core: str, goal: str, focus: str) -> str:
    return f"הליבה היא {core}. היעד הנוכחי הוא {goal}. המוקד כעת הוא {focus}."


def encode(tok, text: str) -> list[int]:
    return tok.encode(text, bos=True, eos=True)[:112]


def init_stream(model: SelfVariantPersonalNativeLM, ids: torch.Tensor) -> torch.Tensor:
    base = model.base
    x = base.token_embedding(ids)
    pos = torch.arange(ids.shape[1], device=ids.device)
    return x + base.position_embedding(pos)[None, :, :]


def random_self(anchor: torch.Tensor, seed: int, scale: float = 2.0) -> torch.Tensor:
    a = anchor.detach().cpu().float()
    n = a.norm().clamp_min(1e-12)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    d = torch.randn(a.shape, generator=g)
    d = d / d.norm().clamp_min(1e-12)
    return a + scale * n * d


def radial2(anchor: torch.Tensor) -> torch.Tensor:
    return -anchor.detach().cpu().float()


@torch.no_grad()
def forward_capture_layer(
    model: SelfVariantPersonalNativeLM,
    ids: torch.Tensor,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = init_stream(model, ids)
    captured = None
    for i, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters), 1):
        x = block(x, None)
        x = adapter(x, model.self_anchor)
        if i == layer:
            captured = x.clone()
    if captured is None:
        raise RuntimeError(layer)
    final = model.base.final_norm(x).mean(dim=1)
    return captured, final


@torch.no_grad()
def forward_factorial(
    model: SelfVariantPersonalNativeLM,
    ids: torch.Tensor,
    layer: int,
    self_replacement: torch.Tensor | None,
    core_delta: torch.Tensor | None,
) -> torch.Tensor:
    x = init_stream(model, ids)
    for i, (block, adapter) in enumerate(zip(model.base.blocks, model.self_adapters), 1):
        x = block(x, None)
        anchor = model.self_anchor
        if i == layer and self_replacement is not None:
            anchor = self_replacement.to(model.self_anchor)
        x = adapter(x, anchor)
        if i == layer and core_delta is not None:
            x = x + core_delta.to(x).view(1, 1, -1)
    return model.base.final_norm(x).mean(dim=1)


def split_triples(cores, goals, foci):
    train = []
    test = []
    for c in cores:
        for gi, g in enumerate(goals):
            for fi, f in enumerate(foci):
                row = (c, g, f, prompt(c, g, f))
                if (gi + fi) % 2 == 0:
                    train.append(row)
                else:
                    test.append(row)
    return train, test


@torch.no_grad()
def capture_clean(
    model: SelfVariantPersonalNativeLM,
    tok,
    triples,
    layer: int,
    device: str,
):
    out = {}
    for c, g, f, text in triples:
        ids = torch.tensor([encode(tok, text)], dtype=torch.long, device=device)
        stream, final = forward_capture_layer(model, ids, layer)
        out[text] = {
            "layer_pool": stream.mean(dim=1)[0].detach().cpu(),
            "final": final[0].detach().cpu(),
        }
    return out


def build_core_basis(clean, train_triples):
    by_core = defaultdict(list)
    for c, g, f, text in train_triples:
        by_core[c].append(clean[text]["layer_pool"].float())
    labels = sorted(by_core)
    centroids = torch.stack([torch.stack(by_core[c]).mean(0) for c in labels])
    centered = centroids - centroids.mean(0, keepdim=True)
    _, s, vh = torch.linalg.svd(centered, full_matrices=False)
    rank = min(len(labels) - 1, int((s > 1e-8).sum()))
    basis = vh[:rank].T.contiguous()
    return basis, {
        "rank": rank,
        "singular_values": [float(x) for x in s[:rank]],
        "core_labels": labels,
    }


def build_final_centroids(clean, train_triples):
    idx = {"core": 0, "goal": 1, "focus": 2}
    out = {}
    for comp, i in idx.items():
        by = defaultdict(list)
        for row in train_triples:
            by[row[i]].append(unit(clean[row[3]]["final"].unsqueeze(0))[0])
        out[comp] = {
            label: unit(torch.stack(xs).mean(0, keepdim=True))[0]
            for label, xs in by.items()
        }
    return out


def margin(rep: torch.Tensor, centroids, correct: str) -> float:
    r = unit(rep.unsqueeze(0))[0]
    true = float(r @ centroids[correct])
    wrong = max(float(r @ v) for k, v in centroids.items() if k != correct)
    return true - wrong


def pred(rep: torch.Tensor, centroids) -> str:
    labels = sorted(centroids)
    c = torch.stack([centroids[x] for x in labels])
    scores = unit(rep.unsqueeze(0))[0] @ c.T
    return labels[int(scores.argmax())]


def project_core_delta(basis: torch.Tensor, target: torch.Tensor, source: torch.Tensor):
    d = source.float() - target.float()
    return basis @ (basis.T @ d)


def random_core_control(delta: torch.Tensor, basis: torch.Tensor, seed: int):
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    r = torch.randn(delta.shape, generator=g)
    # remove learned CORE-subspace component
    r = r - basis @ (basis.T @ r)
    if float(r.norm()) < 1e-12:
        r = torch.randn(delta.shape, generator=g)
    r = r / r.norm().clamp_min(1e-12) * delta.norm()
    return r


def stats(xs):
    return {
        "mean": statistics.mean(xs),
        "mean_abs": statistics.mean(abs(x) for x in xs),
        "rms": math.sqrt(statistics.mean(x * x for x in xs)),
        "stdev": statistics.pstdev(xs) if len(xs) > 1 else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--self-shams", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    ckpt, cfg, model = load_any(args.checkpoint, args.device)
    if not isinstance(model, SelfVariantPersonalNativeLM):
        raise SystemExit("requires SELF variant")
    model.eval()
    tok, tokenizer_kind = load_tokenizer(args.tokenizer)

    cores = list(TEST_CORES)
    goals = list(TEST_GOALS[:6])
    foci = list(TEST_FOCI[:6])
    train_triples, test_triples = split_triples(cores, goals, foci)
    all_triples = train_triples + test_triples

    clean = capture_clean(model, tok, all_triples, args.layer, args.device)
    basis, basis_meta = build_core_basis(clean, train_triples)
    centroids = build_final_centroids(clean, train_triples)

    anchor = model.self_anchor.detach().cpu().float()
    self_modes = {"normal": None, "radial": radial2(anchor)}
    for i in range(args.self_shams):
        self_modes[f"random_{i+1}"] = random_self(anchor, 740_001 + 131 * i)

    core_labels = sorted(cores)
    sample_cells = []
    for c, g, f, text in test_triples:
        source_core = core_labels[(core_labels.index(c) + 1) % len(core_labels)]
        source_text = prompt(source_core, g, f)
        tpool = clean[text]["layer_pool"]
        spool = clean[source_text]["layer_pool"]
        targeted = project_core_delta(basis, tpool, spool)
        random_c = random_core_control(
            targeted,
            basis,
            int(hashlib.sha1(text.encode("utf-8")).hexdigest()[:8], 16),
        )

        ids = torch.tensor([encode(tok, text)], dtype=torch.long, device=args.device)
        cells = {}
        for sname, srep in self_modes.items():
            for cname, cdelta in (
                ("none", None),
                ("targeted", targeted),
                ("random", random_c),
            ):
                rep = forward_factorial(
                    model,
                    ids,
                    args.layer,
                    srep,
                    cdelta,
                )[0].detach().cpu()
                cells[(sname, cname)] = rep

        labels = {"core": c, "goal": g, "focus": f}
        sample = {
            "target_core": c,
            "source_core": source_core,
            "goal": g,
            "focus": f,
            "core_patch_norm": float(targeted.norm()),
            "core_patch_fraction_of_target_pool": float(
                targeted.norm() / tpool.norm().clamp_min(1e-12)
            ),
            "cells": {},
        }
        for key, rep in cells.items():
            sample["cells"]["|".join(key)] = {
                comp: {
                    "margin": margin(rep, centroids[comp], labels[comp]),
                    "prediction": pred(rep, centroids[comp]),
                }
                for comp in COMPONENTS
            }
            sample["cells"]["|".join(key)]["core_source_prediction"] = (
                pred(rep, centroids["core"]) == source_core
            )
        sample_cells.append(sample)

    def get_margin(sample, s, c, outcome):
        return sample["cells"][f"{s}|{c}"][outcome]["margin"]

    outcomes = {}
    for outcome in COMPONENTS:
        radial_targeted = []
        radial_random_core = []
        random_self_targeted = {f"random_{i+1}": [] for i in range(args.self_shams)}

        for sample in sample_cells:
            base = get_margin(sample, "normal", "none", outcome)
            normal_target = get_margin(sample, "normal", "targeted", outcome)
            normal_random = get_margin(sample, "normal", "random", outcome)
            radial_none = get_margin(sample, "radial", "none", outcome)
            radial_target = get_margin(sample, "radial", "targeted", outcome)
            radial_random = get_margin(sample, "radial", "random", outcome)

            radial_targeted.append(
                radial_target - radial_none - normal_target + base
            )
            radial_random_core.append(
                radial_random - radial_none - normal_random + base
            )

            for sname in random_self_targeted:
                s_none = get_margin(sample, sname, "none", outcome)
                s_target = get_margin(sample, sname, "targeted", outcome)
                random_self_targeted[sname].append(
                    s_target - s_none - normal_target + base
                )

        target_stats = stats(radial_targeted)
        random_self_stats = [stats(v) for v in random_self_targeted.values()]
        random_self_rms = [x["rms"] for x in random_self_stats]
        random_self_mean_abs = [x["mean_abs"] for x in random_self_stats]
        random_core_stats = stats(radial_random_core)

        outcomes[outcome] = {
            "radial_SELF_x_targeted_CORE": target_stats,
            "random_SELF_x_targeted_CORE": {
                "runs": random_self_stats,
                "rms_mean": statistics.mean(random_self_rms),
                "rms_sd": statistics.pstdev(random_self_rms),
                "target_ratio_to_random_self_mean": (
                    target_stats["rms"] / max(statistics.mean(random_self_rms), 1e-12)
                ),
                "target_z_vs_random_self": (
                    None
                    if statistics.pstdev(random_self_rms) < 1e-12
                    else (
                        target_stats["rms"] - statistics.mean(random_self_rms)
                    ) / statistics.pstdev(random_self_rms)
                ),
                "target_rank_among_9_desc": (
                    1 + sum(v > target_stats["rms"] for v in random_self_rms)
                ),
                "mean_abs_mean": statistics.mean(random_self_mean_abs),
            },
            "radial_SELF_x_random_CORE": random_core_stats,
            "target_ratio_to_random_CORE": (
                target_stats["rms"] / max(random_core_stats["rms"], 1e-12)
            ),
        }

    manipulation = {}
    for sname in ("normal", "radial"):
        target_switch = statistics.mean(
            float(sample["cells"][f"{sname}|targeted"]["core_source_prediction"])
            for sample in sample_cells
        )
        random_switch = statistics.mean(
            float(sample["cells"][f"{sname}|random"]["core_source_prediction"])
            for sample in sample_cells
        )
        manipulation[sname] = {
            "targeted_CORE_source_switch_rate": target_switch,
            "random_CORE_source_switch_rate": random_switch,
        }

    report = {
        "schema_version": 1,
        "experiment": "internal SELF x CORE subspace factorial intervention v1",
        "variant": ckpt.get("variant"),
        "tokens_seen": ckpt.get("tokens_seen"),
        "layer": args.layer,
        "tokenizer_kind": tokenizer_kind,
        "train_contexts": len(train_triples),
        "test_contexts": len(test_triples),
        "core_subspace": basis_meta,
        "mean_core_patch_fraction_of_target_pool": statistics.mean(
            x["core_patch_fraction_of_target_pool"] for x in sample_cells
        ),
        "core_manipulation_check": manipulation,
        "interaction_outcomes": outcomes,
        "primary_question": (
            "Does an internal learned CORE-subspace intervention interact with "
            "the learned radial SELF direction on downstream GOAL/FOCUS more "
            "strongly than either random SELF or random non-CORE subspace controls?"
        ),
        "guardrails": [
            "CORE basis is learned only from held-out training context combinations.",
            "Evaluation uses disjoint goal/focus context combinations while retaining every label in both splits.",
            "Only CORE-subspace coordinates are changed; the orthogonal target residual is preserved.",
            "Random CORE controls are orthogonal to the learned CORE subspace and norm-matched per example.",
            "Radial SELF is compared against eight random same-displacement SELF directions.",
            "A strong result requires both SELF-direction specificity and CORE-subspace specificity.",
            "This remains single-seed mechanistic evidence, not semantic selfhood.",
        ],
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "variant": report["variant"],
        "tokens_seen": report["tokens_seen"],
        "layer": args.layer,
        "core_subspace": basis_meta,
        "core_manipulation": manipulation,
        "outcomes": {
            k: {
                "target_rms": v["radial_SELF_x_targeted_CORE"]["rms"],
                "ratio_random_self": v["random_SELF_x_targeted_CORE"]["target_ratio_to_random_self_mean"],
                "ratio_random_core": v["target_ratio_to_random_CORE"],
            }
            for k, v in outcomes.items()
        },
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
