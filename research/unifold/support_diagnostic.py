#!/usr/bin/env python3
"""UNIFOLD Gate 2a: calibrated few-shot support-use POSITIVE CONTROL.

This is deliberately NOT the M/Q research model. It has a predeclared affine
hypothesis family, and uses support observations to update a fixed-size
posterior M. A positive result proves that the evaluation can detect learning
from demonstrations; it does NOT prove emergent rule discovery or a learned
learning rule Q.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

MODULUS = 5
POSITIONS = 4
MAX_SUPPORT = 12
K_VALUES = (0, 1, 2, 4, 8, 12)
RULES = [(a, b) for a in range(1, MODULUS) for b in range(MODULUS)]


def split_for(a: tuple[int, ...], b: tuple[int, ...]) -> str:
    """Partition complete FOUR-coordinate functions, not individual examples."""
    raw = bytes(a + b)
    bucket = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % 10
    return "train" if bucket < 8 else ("val" if bucket == 8 else "test")


def sample_batch(size: int, rng: random.Random, split: str = "train"):
    if split not in ("train", "val", "test"):
        raise ValueError("bad split")
    inputs, support_outputs, queries, query_outputs, identities = [], [], [], [], []
    while len(inputs) < size:
        a = tuple(rng.randrange(1, MODULUS) for _ in range(POSITIONS))
        b = tuple(rng.randrange(MODULUS) for _ in range(POSITIONS))
        if split_for(a, b) != split:
            continue
        support_input = []
        support_output = []
        for _ in range(MAX_SUPPORT):
            pos, x = rng.randrange(POSITIONS), rng.randrange(MODULUS)
            support_input.append((pos, x))
            support_output.append((a[pos] * x + b[pos]) % MODULUS)
        qpos, qx = rng.randrange(POSITIONS), rng.randrange(MODULUS)
        inputs.append(support_input)
        support_outputs.append(support_output)
        queries.append((qpos, qx))
        query_outputs.append((a[qpos] * qx + b[qpos]) % MODULUS)
        identities.append(tuple(a + b))
    return (
        torch.tensor(inputs, dtype=torch.long),
        torch.tensor(support_outputs, dtype=torch.long),
        torch.tensor(queries, dtype=torch.long),
        torch.tensor(query_outputs, dtype=torch.long),
        identities,
    )


class AffinePosteriorControl(nn.Module):
    """Finite-hypothesis learner: known rules, learned prior and evidence gain.

    The hypothesis bank is intentionally given to the algorithm.
    For each coordinate, evidence rules out inconsistent affine functions.
    No label from a query enters forward(), and no trained weights change
    during evaluation.
    """

    def __init__(self):
        super().__init__()
        self.log_prior = nn.Parameter(torch.zeros(POSITIONS, len(RULES)))
        self.log_evidence_gain = nn.Parameter(torch.tensor(2.0))
        self.register_buffer("a_bank", torch.tensor([r[0] for r in RULES]))
        self.register_buffer("b_bank", torch.tensor([r[1] for r in RULES]))

    def forward(self, support_x, support_y, query_x, k: int):
        if k < 0 or k > support_x.shape[1]:
            raise ValueError("invalid support count")
        if support_x.ndim != 3 or support_x.shape[-1] != 2:
            raise ValueError("input shape must be [batch,support,2]")
        qpos, qv = query_x[:, 0], query_x[:, 1]
        posterior_logits = self.log_prior[qpos].clone()
        if k:
            sx = support_x[:, :k]
            sy = support_y[:, :k]
            pred = (sx[..., 1, None] * self.a_bank + self.b_bank) % MODULUS
            contradictory = ((pred != sy[..., None]) &
                             (sx[..., 0, None] == qpos[:, None, None]))
            evidence_gain = F.softplus(self.log_evidence_gain) + 1.0
            posterior_logits = posterior_logits - evidence_gain * contradictory.float().sum(dim=1)
        weights = posterior_logits.softmax(-1)
        candidate_outputs = (qv[:, None] * self.a_bank + self.b_bank) % MODULUS
        probabilities = torch.zeros(query_x.shape[0], MODULUS,
                                    device=query_x.device).scatter_add(1, candidate_outputs, weights)
        return probabilities.clamp_min(1e-8).log()


def assess(model, sample, mode: str = "real"):
    support_x, support_y, query_x, gold, _ = sample
    if mode == "corrupted":
        # Targets from other independently sampled laws; preserve inputs/count.
        support_y = support_y.roll(1, dims=0)
    elif mode != "real":
        raise ValueError("unknown control")
    data = {}
    with torch.no_grad():
        for k in K_VALUES:
            logits = model(support_x, support_y, query_x, k)
            predicted = logits.argmax(dim=-1)
            data[str(k)] = {
                "accuracy": round(float((predicted == gold).float().mean()), 6),
                "nll": round(float(F.nll_loss(logits, gold)), 6),
                "n": int(gold.shape[0]),
            }
    return data


def train(steps: int = 120, batch_size: int = 128, eval_episodes: int = 4000,
          seed: int = 731, output: Path | None = None):
    if steps < 1 or batch_size < 1 or eval_episodes < 100:
        raise ValueError("invalid experiment size")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = AffinePosteriorControl()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for step in range(steps):
        sx, sy, qx, y, _ = sample_batch(batch_size, rng, "train")
        k = K_VALUES[step % len(K_VALUES)]
        loss = F.nll_loss(model(sx, sy, qx, k), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    eval_sample = sample_batch(eval_episodes, random.Random(seed + 1000003), "test")
    train_sample = sample_batch(256, random.Random(seed + 47), "train")
    train_ids = set(train_sample[-1])
    test_ids = set(eval_sample[-1])
    if train_ids & test_ids:
        raise AssertionError("train/test rule identity overlap")
    result = {
        "status": "POSITIVE_CONTROL_ONLY_NOT_UNIFOLD_METALEARNING",
        "seed": seed,
        "steps": steps,
        "trainable_parameters": sum(p.numel() for p in model.parameters()),
        "eval_episodes": eval_episodes,
        "rule_family": "coordinate-conditional affine (a*x+b) mod 5",
        "train_test_split": "SHA256 of full 4-coordinate rule; no shared full rules",
        "fixed_hypotheses": len(RULES),
        "support_count": list(K_VALUES),
        "real_support": assess(model, eval_sample),
        "corrupted_support": assess(model, eval_sample, "corrupted"),
        "chance_accuracy": 1 / MODULUS,
        "limitations": [
            "Hypothesis forms are hard-coded: positive result is calibration, not emergence.",
            "Train/test share atomic affine primitives, but full 4-coordinate rules are separated.",
            "No changing Q state, no L1/L2 compositional claim.",
            "Support count changes probability of having informative observations.",
        ],
    }
    if output is not None:
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        torch.save({"state_dict": model.state_dict(), "seed": seed}, output / "control.pt")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--eval", type=int, default=4000)
    p.add_argument("--seed", type=int, default=731)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(train(a.steps, a.batch, a.eval, a.seed, a.out), indent=2,
                     ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
