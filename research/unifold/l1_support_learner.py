#!/usr/bin/env python3
"""UNIFOLD Gate 3: trainable set-memory baseline on heldout 2-step L1 laws.

The neural forward pass receives support input/output pairs and query inputs,
never query labels, rule IDs, an oracle, or a catalog of possible programs.
This does not test improving the *learning rule* Q.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from itertools import zip_longest
import json
import math
from pathlib import Path
import random
import torch
from torch import nn
from torch.nn import functional as F

SUPPORT = (0, 1, 2, 4, 8, 12)


def load_episodes(directory: Path):
    by_split = defaultdict(list)
    seen = set()
    with (directory / "tasks.jsonl").open(encoding="utf8") as fpub, \
         (directory / "answers.PRIVATE.jsonl").open(encoding="utf8") as fg:
        for a, b in zip_longest(fpub, fg):
            if a is None or b is None:
                raise ValueError("public/private lengths differ")
            p, g = json.loads(a), json.loads(b)
            if p["id"] != g["id"] or p["id"] in seen:
                raise ValueError("misaligned or duplicate episode")
            seen.add(p["id"])
            if p["level"] != "L1":
                continue
            by_split[p["split"]].append((
                torch.tensor([item["input"] for item in p["support"]], dtype=torch.long),
                torch.tensor([item["output"] for item in p["support"]], dtype=torch.long),
                torch.tensor([item["input"] for item in p["queries"]], dtype=torch.long),
                torch.tensor(g["query_outputs"], dtype=torch.long),
                g["semantic_sha256"],
            ))
    for required in ("train", "val", "test"):
        if not by_split[required]:
            raise ValueError("missing L1 split: " + required)
    semantic = {name: {ep[-1] for ep in by_split[name]}
                for name in ("train", "val", "test")}
    keys = list(semantic)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if semantic[a] & semantic[b]:
                raise ValueError("semantic law leaked between splits")
    return by_split


def batchify(episodes, device="cpu"):
    return tuple(torch.stack([ep[i] for ep in episodes]).to(device) for i in range(4))


class L1MemoryLearner(nn.Module):
    """Learned permutation-invariant memory M with attention to support pairs."""
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.pair = nn.Sequential(nn.Linear(32, width), nn.GELU(),
                                  nn.Linear(width, width))
        self.query = nn.Sequential(nn.Linear(16, width), nn.GELU(),
                                   nn.Linear(width, width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.qproj = nn.Linear(width, width, bias=False)
        self.decoder = nn.Sequential(
            nn.Linear(width * 3 + 16, width * 2), nn.GELU(),
            nn.Linear(width * 2, width), nn.GELU(),
            nn.Linear(width, 16),
        )

    def forward(self, sx, sy, qx, k=12):
        """sx/sy:[B,12,4], qx:[B,16,4] -> logits:[B,16,4,4]."""
        if not 0 <= k <= sx.size(1):
            raise ValueError("invalid support count")
        xhot = F.one_hot(sx, 4).flatten(-2).float()
        yhot = F.one_hot(sy, 4).flatten(-2).float()
        qhot = F.one_hot(qx, 4).flatten(-2).float()
        qfeatures = self.query(qhot)
        if k:
            supports = self.pair(torch.cat((xhot[:, :k], yhot[:, :k]), -1))
            memory = supports.mean(1, keepdim=True).expand(-1, qx.size(1), -1)
            scores = torch.matmul(self.qproj(qfeatures),
                                  self.key(supports).transpose(-1, -2)) / math.sqrt(self.width)
            attended = scores.softmax(-1) @ self.value(supports)
        else:
            memory = torch.zeros_like(qfeatures)
            attended = torch.zeros_like(qfeatures)
        return self.decoder(torch.cat((qfeatures, memory, attended, qhot), -1)).reshape(
            sx.size(0), qx.size(1), 4, 4)


def predict_accuracy(logits, target):
    predicted = logits.argmax(-1)
    token = (predicted == target).float().mean((-1, -2))
    exact = (predicted == target).all(-1).float().mean(-1)
    return token, exact


def evaluate(model, episodes, limit=None, batch=48):
    model.eval()
    data = episodes[:limit] if limit else episodes
    report = {"real": {}, "shuffled": {}}
    with torch.no_grad():
        for mode in report:
            for k in SUPPORT:
                tokens, exacts = [], []
                for offset in range(0, len(data), batch):
                    sub = data[offset:offset + batch]
                    sx, sy, qx, qy = batchify(sub)
                    if mode == "shuffled":
                        sy = sy.roll(1, 0) if len(sub) > 1 else sy.flip(1)
                    logits = model(sx, sy, qx, k)
                    t, e = predict_accuracy(logits, qy)
                    tokens.extend(t.tolist())
                    exacts.extend(e.tolist())
                report[mode][str(k)] = {
                    "token_accuracy": round(sum(tokens) / len(tokens), 6),
                    "exact_accuracy": round(sum(exacts) / len(exacts), 6),
                    "episodes": len(tokens),
                }
    return report


def train(data: Path, output: Path, steps=1200, batch=40, width=128,
          seed=37, eval_limit=400):
    if steps < 1 or batch < 1 or eval_limit < 1:
        raise ValueError("steps/batch/eval_limit must be positive")
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    rng = random.Random(seed)
    episodes = load_episodes(data)
    model = L1MemoryLearner(width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    history = []
    for step in range(steps):
        examples = rng.choices(episodes["train"], k=batch)
        sx, sy, qx, qy = batchify(examples)
        k = SUPPORT[step % len(SUPPORT)]
        model.train()
        prediction = model(sx, sy, qx, k)
        loss = F.cross_entropy(prediction.reshape(-1, 4), qy.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 8) == 0:
            history.append({"step": step + 1, "support": k,
                            "loss": round(float(loss.detach()), 5)})
            print(json.dumps(history[-1]), flush=True)
    validation = evaluate(model, episodes["val"], limit=eval_limit)
    test = evaluate(model, episodes["test"], limit=eval_limit)
    result = {
        "experiment": "UNIFOLD-GATE3-L1-M-ONLY-2026-10-12",
        "status": "MEASURED_NOT_A_METALEARNING_CLAIM",
        "model": "learned neural set memory M, no program list, no Q",
        "parameters": sum(p.numel() for p in model.parameters()),
        "steps": steps, "batch": batch, "width": width, "seed": seed,
        "train_episodes": len(episodes["train"]),
        "val_episodes": len(episodes["val"]),
        "test_episodes": len(episodes["test"]),
        "eval_limit_per_split": eval_limit,
        "train_history": history, "val": validation, "test": test,
        "random_token_baseline": .25,
        "note": "Split by semantic law. Query gold used only in loss/metrics.",
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf8")
    torch.save({"model": model.state_dict(), "width": width, "seed": seed},
               output / "checkpoint.pt")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch", type=int, default=40)
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--seed", type=int, default=37)
    p.add_argument("--eval-limit", type=int, default=400)
    a = p.parse_args()
    result = train(a.data, a.out, a.steps, a.batch, a.width, a.seed, a.eval_limit)
    print(json.dumps({"parameters": result["parameters"], "val": result["val"],
                      "test": result["test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
