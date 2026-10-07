from __future__ import annotations

from contextlib import contextmanager
import torch
import torch.nn as nn

from model_self_variants import (
    SELF_VARIANTS,
    SelfVariantAdapter,
    anchor_lr_scale_for_variant,
    canonical_architecture,
)

VARIANT_CODE_TO_NAME = {
    "V1": "baseline",
    "V2": "self_v1",
    "V3": "self_v1_slow",
    "V4": "diff_only",
    "V5": "diff_anchor",
    "V6": "projected_diff",
    "V7": "projected_diff_slow",
}
NAME_TO_VARIANT_CODE = {v: k for k, v in VARIANT_CODE_TO_NAME.items()}

def variant_name(code: str) -> str:
    try:
        return VARIANT_CODE_TO_NAME[code]
    except KeyError as exc:
        raise ValueError(f"unknown blind-safe variant code: {code}") from exc

def is_self_variant(code: str) -> bool:
    return variant_name(code) != "baseline"

class GoldfishSelfLM(nn.Module):
    """Inject the existing SELF adapters after every GPT-2 transformer block."""
    def __init__(self, base: nn.Module, variant: str, rank: int = 8, projection_rank: int = 4) -> None:
        super().__init__()
        if variant not in SELF_VARIANTS:
            raise ValueError(variant)
        if not hasattr(base, "transformer") or not hasattr(base.transformer, "h"):
            raise TypeError("GoldfishSelfLM requires a GPT-2 style base model")
        self.base = base
        self.variant = variant
        self.architecture = canonical_architecture(variant)
        self.rank = int(rank)
        self.projection_rank = int(projection_rank)
        self.hidden_size = int(getattr(base.config, "n_embd"))
        self.n_layers = len(base.transformer.h)
        self.self_anchor = nn.Parameter(torch.empty(self.hidden_size))
        nn.init.normal_(self.self_anchor, mean=0.0, std=0.02)
        self.self_adapters = nn.ModuleList([
            SelfVariantAdapter(
                self.hidden_size,
                rank=self.rank,
                architecture=self.architecture,
                projection_rank=self.projection_rank,
            )
            for _ in range(self.n_layers)
        ])
        self._hook_handles = []
        for block, adapter in zip(self.base.transformer.h, self.self_adapters):
            self._hook_handles.append(block.register_forward_hook(self._hook(adapter)))

    def _hook(self, adapter: SelfVariantAdapter):
        def apply_self(_module, _inputs, output):
            if isinstance(output, tuple):
                hidden = adapter(output[0], self.self_anchor)
                return (hidden, *output[1:])
            if torch.is_tensor(output):
                return adapter(output, self.self_anchor)
            raise TypeError(f"unexpected GPT-2 block output type: {type(output)!r}")
        return apply_self

    def forward(self, *args, **kwargs):
        return self.base(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self.base.generate(*args, **kwargs)

    @property
    def config(self):
        return self.base.config

    def self_diagnostics(self) -> dict:
        return {
            "variant": self.variant,
            "architecture": self.architecture,
            "anchor_norm": float(self.self_anchor.detach().norm().cpu()),
            "layers": [adapter.diagnostics() for adapter in self.self_adapters],
        }

    def effective_anchor(self, layer_index: int, anchor: torch.Tensor | None = None) -> torch.Tensor:
        if not 0 <= layer_index < len(self.self_adapters):
            raise IndexError(layer_index)
        source = self.self_anchor if anchor is None else anchor
        return self.self_adapters[layer_index]._anchor_for_layer(source)

    @contextmanager
    def temporary_anchor(self, replacement: torch.Tensor):
        original = self.self_anchor.detach().clone()
        try:
            with torch.no_grad():
                self.self_anchor.copy_(replacement.to(self.self_anchor))
            yield
        finally:
            with torch.no_grad():
                self.self_anchor.copy_(original)

def build_variant(base: nn.Module, variant_code: str, rank: int = 8, projection_rank: int = 4):
    name = variant_name(variant_code)
    if name == "baseline":
        return base
    return GoldfishSelfLM(base, name, rank=rank, projection_rank=projection_rank)

def unwrap_base(model: nn.Module) -> nn.Module:
    return model.base if isinstance(model, GoldfishSelfLM) else model

def anchor_replacement(anchor: torch.Tensor, mode: str, seed: int = 44117) -> torch.Tensor:
    a = anchor.detach()
    if mode == "normal":
        return a.clone()
    if mode == "zero":
        return torch.zeros_like(a)
    if mode == "negate":
        return -a
    if mode == "shuffle":
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        perm = torch.randperm(a.numel(), generator=g).to(a.device)
        return a[perm]
    g = torch.Generator(device="cpu")
    g.manual_seed(seed + (17 if mode == "orthogonal_same_norm" else 0))
    r = torch.randn(a.shape, generator=g, dtype=a.dtype, device="cpu").to(a.device)
    if mode == "orthogonal_same_norm":
        denom = a.dot(a).clamp_min(1e-12)
        r = r - (r.dot(a) / denom) * a
    elif mode != "random_same_norm":
        raise ValueError(mode)
    return r / r.norm().clamp_min(1e-12) * a.norm()

def intervention_geometry(anchor: torch.Tensor, replacement: torch.Tensor) -> dict:
    a = anchor.detach().float()
    b = replacement.detach().float()
    delta = b - a
    denom = a.norm() * b.norm()
    cosine = None if float(denom) < 1e-12 else float((a @ b) / denom)
    return {
        "anchor_norm": float(a.norm()),
        "replacement_norm": float(b.norm()),
        "delta_norm": float(delta.norm()),
        "cosine_to_normal": cosine,
    }
