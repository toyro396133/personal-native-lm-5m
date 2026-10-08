from __future__ import annotations

import math
from pathlib import Path

import torch


def _pack_nibbles(q: torch.Tensor) -> torch.Tensor:
    q = q.to(torch.uint8).flatten()
    if q.numel() % 2:
        q = torch.cat([q, torch.zeros(1, dtype=torch.uint8)])
    return q[0::2] | (q[1::2] << 4)


def _unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    packed = packed.to(torch.uint8).flatten()
    out = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    out[0::2] = packed & 0x0F
    out[1::2] = (packed >> 4) & 0x0F
    return out[:n]


def quantize_tensor_int4(t: torch.Tensor, block_size: int = 256) -> dict:
    x = t.detach().cpu().float().flatten()
    n = x.numel()
    pad = (-n) % block_size
    if pad:
        x = torch.cat([x, torch.zeros(pad)])
    blocks = x.view(-1, block_size)
    scales = blocks.abs().amax(dim=1).clamp_min(1e-12) / 7.0
    q = torch.round(blocks / scales[:, None]).clamp(-7, 7).to(torch.int8)
    # map -7..7 to 1..15; zero becomes 8.
    nib = (q.to(torch.int16) + 8).to(torch.uint8)
    packed = _pack_nibbles(nib)
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "numel": n,
        "block_size": block_size,
        "scales": scales.to(torch.float16),
        "packed": packed,
    }


def dequantize_tensor_int4(payload: dict) -> torch.Tensor:
    n = int(payload["numel"])
    block_size = int(payload["block_size"])
    padded = math.ceil(n / block_size) * block_size
    nib = _unpack_nibbles(payload["packed"], padded)
    q = nib.to(torch.int16) - 8
    q = q.view(-1, block_size).float()
    scales = payload["scales"].float()[:, None]
    x = (q * scales).flatten()[:n].view(tuple(payload["shape"]))
    return x


def save_int4_model_checkpoint(
    path: str,
    model_state: dict[str, torch.Tensor],
    metadata: dict,
    block_size: int = 256,
) -> None:
    encoded = {}
    exact = {}
    for name, tensor in model_state.items():
        if torch.is_floating_point(tensor):
            encoded[name] = quantize_tensor_int4(tensor, block_size=block_size)
        else:
            exact[name] = tensor.detach().cpu()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "format": "blockwise-symmetric-int4-model-only",
            "lossy": True,
            "metadata": metadata,
            "tensors_int4": encoded,
            "tensors_exact": exact,
        },
        path,
    )


def load_int4_model_checkpoint(path: str) -> tuple[dict, dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "blockwise-symmetric-int4-model-only":
        raise ValueError("not a supported Goldfish INT4 checkpoint")
    state = {}
    for name, item in payload["tensors_int4"].items():
        state[name] = dequantize_tensor_int4(item)
    state.update(payload.get("tensors_exact", {}))
    return payload.get("metadata", {}), state


def save_bf16_model_checkpoint(
    path: str,
    model_state: dict[str, torch.Tensor],
    metadata: dict,
) -> None:
    state = {}
    for name, tensor in model_state.items():
        t = tensor.detach().cpu()
        state[name] = t.to(torch.bfloat16) if torch.is_floating_point(t) else t
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "format": "bf16-model-only",
            "lossy": True,
            "metadata": metadata,
            "model_state": state,
        },
        path,
    )
