#!/usr/bin/env python3
"""UNIFOLD Gate 2b: genuine trainable M-only within-task learner.

Unlike support_diagnostic.py this neural model is NOT given affine operations
or a list of candidate functions. It only receives example pairs, accumulates
an observation matrix M, and learns an amortized prediction from that matrix.
This is a sanity baseline: it does not implement Q or prove meta-plasticity.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

from support_diagnostic import sample_batch, K_VALUES, MODULUS


class MOnlyLearner(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(31, width), nn.ReLU(),
                                 nn.Linear(width, width), nn.ReLU(),
                                 nn.Linear(width, MODULUS))

    def forward(self, sx, sy, qx, k: int):
        if not 0 <= k <= sx.shape[1]:
            raise ValueError('bad support count')
        batch = sx.shape[0]
        coord, x = qx[:, 0], qx[:, 1]
        evidence = torch.zeros(batch, MODULUS * MODULUS, device=sx.device)
        if k:
            # M: observed input/output co-occurrences on the queried coordinate.
            seen = sx[:, :k]
            valid = (seen[:, :, 0] == coord[:, None]).float()
            indices = seen[:, :, 1] * MODULUS + sy[:, :k]
            evidence.scatter_add_(1, indices, valid)
        count = evidence.sum(1, keepdim=True)
        # Fixed-size M; normalized counts preserve signal across support lengths.
        memory = evidence / count.clamp_min(1.)
        features = torch.cat((F.one_hot(x, MODULUS).float(), memory,
                              (count / 12.).clamp(max=1)), dim=1)
        return self.net(features)


def evaluate(model, data):
    sx, sy, qx, gold, _ = data
    model.eval()
    reports = {'real': {}, 'corrupted': {}}
    with torch.no_grad():
        for k in K_VALUES:
            for mode in ('real', 'corrupted'):
                labels = sy if mode == 'real' else sy.roll(1, dims=0)
                logits = model(sx, labels, qx, k)
                reports[mode][str(k)] = {
                    'accuracy': round(float((logits.argmax(-1) == gold).float().mean()), 6),
                    'nll': round(float(F.cross_entropy(logits, gold)), 6),
                }
    return reports


def train(steps=1600, batch=256, eval_episodes=4000, width=128, seed=731,
          output: Path | None = None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    net = MOnlyLearner(width)
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    last = None
    for step in range(steps):
        sx, sy, qx, gold, _ = sample_batch(batch, rng, 'train')
        k = K_VALUES[step % len(K_VALUES)]
        logits = net(sx, sy, qx, k)
        loss = F.cross_entropy(logits, gold)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        last = float(loss.detach())
    val = sample_batch(eval_episodes, random.Random(seed + 1000003), 'test')
    reports = evaluate(net, val)
    metrics = {
        'status': 'NEURAL_M_ONLY_WITHIN_TASK_SANITY_CHECK_NOT_Q_LEARNING',
        'architecture': 'learned MLP from 25-bin episodic observation matrix, no rule bank',
        'seed': seed, 'steps': steps, 'trainable_parameters': sum(p.numel() for p in net.parameters()),
        'train_last_loss': round(last, 6), 'eval_episodes': eval_episodes,
        'real_support': reports['real'], 'corrupted_support': reports['corrupted'],
        'support_values': list(K_VALUES), 'chance': 1/MODULUS,
        'limitations': ['Same atomic affine operation family in train and test; disjoint full rules.',
                        'Fixed engineered observation matrix is an architectural prior.',
                        'No Q state or L2 hierarchical task tested.',
                        'A positive curve is within-task learning, not improved ability to learn new rules.']
    }
    if output:
        output.mkdir(parents=True, exist_ok=True)
        (output/'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True)+'\n')
        torch.save({'state_dict':net.state_dict(), 'width':width, 'seed':seed}, output/'checkpoint.pt')
    return metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--steps', type=int, default=1600)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--eval', type=int, default=4000)
    p.add_argument('--width', type=int, default=128)
    p.add_argument('--seed', type=int, default=731)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(train(args.steps, args.batch, args.eval, args.width, args.seed, args.out), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
