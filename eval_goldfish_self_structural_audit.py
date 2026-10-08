from __future__ import annotations

import argparse
import gc
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from core_goal_focus_sequence_lab import TEST_CORES, TEST_GOALS, TEST_FOCI
from eval_self_core_organization import (
    CORE_TEMPLATES,
    centroid_classify,
    core_margin,
    delta_axis_consistency,
    fixed_rep_relative_cache,
    relation_consistency,
)
from eval_self_core_structural_audit import (
    focus_relation_consistency,
    intervention_geometry,
    intervention_set,
    same_norm_shams,
)
from goldfish_self_variants import GoldfishSelfLM, build_variant, variant_name


def load_archived_model(
    variant_code: str,
    hf_repo: str,
    hf_token: str,
    device: str,
    checkpoint_filename: str | None = None,
):
    checkpoint_path = hf_hub_download(
        repo_id=hf_repo,
        filename=(checkpoint_filename or f"goldfish-124m/{variant_code}/1000k/training_state.pt"),
        token=hf_token,
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if payload.get("variant_code") != variant_code:
        raise RuntimeError(
            f"variant mismatch: expected {variant_code}, got {payload.get('variant_code')}"
        )

    model_id = payload["model_id"]
    revision = payload["model_revision"]
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    base.config.use_cache = False

    rank = int(payload.get("rank") or 8)
    projection_rank = int(payload.get("projection_rank") or 4)
    model = build_variant(base, variant_code, rank=rank, projection_rank=projection_rank)
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device)
    model.eval()

    metadata = {
        k: payload.get(k)
        for k in (
            "schema_version",
            "experiment",
            "variant_code",
            "model_id",
            "model_revision",
            "base_parameter_count",
            "parameter_count",
            "rank",
            "projection_rank",
            "base_lr",
            "adapter_lr",
            "anchor_lr_scale",
            "tokens_seen",
            "steps",
            "seed",
            "data_seed",
            "seq_len",
            "initial_base_sha256",
        )
    }
    del payload
    gc.collect()
    return checkpoint_path, metadata, model


def capture_layers(model, input_ids: torch.Tensor):
    base = model.base if isinstance(model, GoldfishSelfLM) else model
    captured = {}
    handles = []

    def make_block_hook(index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            pooled = F.layer_norm(hidden.float(), (hidden.shape[-1],)).mean(dim=1)
            captured[f"layer_{index}"] = pooled.detach().cpu()
        return hook

    for i, block in enumerate(base.transformer.h, 1):
        handles.append(block.register_forward_hook(make_block_hook(i)))

    def final_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["final"] = hidden.float().mean(dim=1).detach().cpu()

    handles.append(base.transformer.ln_f.register_forward_hook(final_hook))
    try:
        with torch.no_grad():
            model(input_ids=input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    expected = len(base.transformer.h) + 1
    if len(captured) != expected:
        raise RuntimeError(
            f"layer capture mismatch: got {len(captured)}, expected {expected}"
        )
    return captured


@torch.no_grad()
def encode_texts(model, tokenizer, texts, device: str, replacement=None):
    groups = defaultdict(list)
    for text in sorted(set(texts)):
        ids = tokenizer.encode(text, add_special_tokens=False)[:112]
        if tokenizer.bos_token_id is not None:
            ids = [int(tokenizer.bos_token_id)] + ids
        if tokenizer.eos_token_id is not None:
            ids = ids + [int(tokenizer.eos_token_id)]
        ids = ids[:112]
        groups[len(ids)].append((text, ids))

    ctx = (
        model.temporary_anchor(replacement)
        if isinstance(model, GoldfishSelfLM) and replacement is not None
        else _nullcontext()
    )

    caches = defaultdict(dict)
    with ctx:
        for _, items in sorted(groups.items()):
            for start in range(0, len(items), 24):
                chunk = items[start : start + 24]
                ids = torch.tensor(
                    [x[1] for x in chunk],
                    dtype=torch.long,
                    device=device,
                )
                by_layer = capture_layers(model, ids)
                for layer, reps in by_layer.items():
                    for (text, _), rep in zip(chunk, reps):
                        caches[layer][text] = rep
    return dict(caches)


class _nullcontext:
    def __enter__(self):
        return self
    def __exit__(self, *args):
        return False


def effective_anchor_for_layer(model: GoldfishSelfLM, layer: str, raw_anchor):
    if layer == "final":
        idx = model.n_layers - 1
    else:
        idx = int(layer.split("_")[1]) - 1
    return model.effective_anchor(idx, raw_anchor.to(model.self_anchor)).detach().cpu()


def learned_reference_specificity(
    model: GoldfishSelfLM,
    layer: str,
    normal_cache,
    core_texts,
    pair_texts,
    triple_texts,
):
    raw = model.self_anchor.detach().cpu()
    candidates = {
        "learned": effective_anchor_for_layer(model, layer, raw),
    }
    for name, sham in same_norm_shams(raw, count=8).items():
        candidates[name] = effective_anchor_for_layer(model, layer, sham)

    primitive = {}
    for name, anchor in candidates.items():
        rel = fixed_rep_relative_cache(normal_cache, anchor)
        primitive[name] = {
            "core_margin": core_margin(rel, core_texts)["margin"],
            "goal_relation_margin": relation_consistency(
                rel, core_texts, pair_texts
            )["relation_margin"],
            "focus_relation_margin": focus_relation_consistency(
                rel, pair_texts, triple_texts
            )["relation_margin"],
        }

    shams = [v for k, v in primitive.items() if k != "learned"]
    advantages = {}
    for key in ("core_margin", "goal_relation_margin", "focus_relation_margin"):
        vals = [x[key] for x in shams]
        mean = sum(vals) / len(vals)
        variance = sum((x - mean) ** 2 for x in vals) / len(vals)
        sd = variance ** 0.5
        learned = primitive["learned"][key]
        advantages[key] = {
            "learned": learned,
            "sham_mean": mean,
            "sham_sd": sd,
            "learned_minus_sham_mean": learned - mean,
            "z_vs_shams": None if sd < 1e-12 else (learned - mean) / sd,
        }
    return {"primitive": primitive, "advantages": advantages}


def core_accuracy(train_by_core, cache, core_texts):
    return centroid_classify(
        train_by_core,
        [(c, cache[core_texts[c][3]]) for c in sorted(core_texts)],
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant-code", required=True, choices=[f"V{i}" for i in range(1, 8)])
    ap.add_argument("--hf-repo", default="toyro967/personal-native-lm-goldfish-self")
    ap.add_argument("--checkpoint-filename", default=None)
    ap.add_argument("--audit-label", default="1m")
    ap.add_argument("--hf-token", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    checkpoint_path, metadata, model = load_archived_model(
        args.variant_code,
        args.hf_repo,
        args.hf_token,
        args.device,
        args.checkpoint_filename,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        metadata["model_id"],
        revision=metadata["model_revision"],
    )

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

    if isinstance(model, GoldfishSelfLM):
        raw_anchor = model.self_anchor.detach().cpu()
        replacements = intervention_set(raw_anchor)
    else:
        raw_anchor = None
        replacements = {"normal": None}

    caches = {}
    for name, replacement in replacements.items():
        print(f"variant={args.variant_code} encode intervention={name}", flush=True)
        caches[name] = encode_texts(
            model,
            tokenizer,
            all_texts,
            args.device,
            replacement,
        )

    layers = list(caches["normal"])
    train_cores = cores[: len(cores) // 2]
    test_cores = cores[len(cores) // 2 :]
    self_train_texts = [
        pair_texts[(c, g)] for c in train_cores for g in goals
    ]
    self_test_texts = [
        pair_texts[(c, g)] for c in test_cores for g in goals
    ]

    by_layer = {}
    for layer in layers:
        normal = caches["normal"][layer]
        train_by_core = {
            c: [normal[t] for t in core_texts[c][:3]]
            for c in cores
        }

        core_paraphrase = core_accuracy(train_by_core, normal, core_texts)
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

        goal_rel = relation_consistency(normal, core_texts, pair_texts)
        focus_rel = focus_relation_consistency(normal, pair_texts, triple_texts)

        interventions = {}
        if isinstance(model, GoldfishSelfLM):
            for name, replacement in replacements.items():
                if name == "normal":
                    continue
                alt = caches[name][layer]
                axis = delta_axis_consistency(
                    normal,
                    alt,
                    self_train_texts,
                    self_test_texts,
                )
                alt_core = core_accuracy(train_by_core, alt, core_texts)
                interventions[name] = {
                    "geometry": intervention_geometry(raw_anchor, replacement),
                    "core_accuracy": alt_core,
                    "core_accuracy_delta_from_normal": alt_core - core_paraphrase,
                    "axis_consistency": axis["axis_consistency"],
                    "effect_norm": axis["mean_effect_norm"],
                }

        learned_ref = None
        if isinstance(model, GoldfishSelfLM):
            learned_ref = learned_reference_specificity(
                model,
                layer,
                normal,
                core_texts,
                pair_texts,
                triple_texts,
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
        "experiment": "Goldfish 124M SELF CORE structural audit v3",
        "audit_label": args.audit_label,
        "variant_code": args.variant_code,
        "variant_name": variant_name(args.variant_code),
        "archive_checkpoint": str(checkpoint_path),
        "metadata": metadata,
        "primary_question": (
            "Does the transferred SELF mechanism organize a pretrained Hebrew "
            "backbone around distinct SELF and CORE references, beyond merely "
            "becoming functionally active?"
        ),
        "interpretation_guardrails": [
            "Language loss is excluded from the primary structural ranking.",
            "Intervention direction is compared at matched 1x/2x displacement magnitude.",
            "SELF effect strength is separate from learned-reference specificity.",
            "CORE/GOAL/FOCUS geometry is not direct causal proof of semantic roles.",
            "Layerwise geometry is not called causal without activation patching.",
            "This is a single-seed 1M-token transfer canary.",
        ],
        "intervention_design": {
            "directions": ["radial", "random", "orthogonal", "shuffle-derived"],
            "magnitudes": ["1x anchor norm", "2x anchor norm"],
            "radial_1x": "SELF=zero",
            "radial_2x": "SELF=negate",
        },
        "layers": by_layer,
        "final": by_layer["final"],
        "self_diagnostics": (
            model.self_diagnostics() if isinstance(model, GoldfishSelfLM) else None
        ),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    final = report["final"]
    print(json.dumps({
        "variant_code": args.variant_code,
        "variant_name": report["variant_name"],
        "core_under_goal": final["core_under_goal_accuracy"],
        "core_under_focus": final["core_under_focus_accuracy"],
        "goal_relation": final["goal_relation_margin"],
        "focus_relation": final["focus_relation_margin"],
        "out": args.out,
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
