#!/usr/bin/env python3
"""UNIFOLD Gate 5: does A-conditioned Q change the WRITE rule for new examples?

Starts both independently trained arms from the *same* pretrained Gate 3
L1 support-learner checkpoint. M resets at the start of every A/B/C stage.
Only Q persists across stages. Q cannot directly reach the decoder;
it conditions the representation WRITTEN into M from current support pairs.
A useful result would still need controls against Q as content memory.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from l1_support_learner import L1MemoryLearner
from learner import read_dataset, batch_tensors

KS = (0, 1, 2, 4, 8, 12)


class QWriteLearner(nn.Module):
    def __init__(self, width=128, qdim=16):
        super().__init__()
        self.width, self.qdim = width, qdim
        self.base = L1MemoryLearner(width)
        self.controller = nn.GRUCell(width, qdim)
        self.q_scale = nn.Linear(qdim, width, bias=False)
        self.q_shift = nn.Linear(qdim, width, bias=False)
        self.write_gate = nn.Linear(width, width)
        # At Q=0, written memory is EXACTLY the pretrained M-only baseline.
        nn.init.normal_(self.q_scale.weight, std=0.01)
        nn.init.normal_(self.q_shift.weight, std=0.01)
        nn.init.zeros_(self.write_gate.bias)

    def load_gate3(self, path: Path):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        self.base.load_state_dict(checkpoint['model'], strict=True)

    def stage(self, sx, sy, qx, q, k):
        if not 0 <= k <= sx.shape[1]:
            raise ValueError("support outside range")
        n, nqueries = sx.shape[0], qx.shape[1]
        qhot = F.one_hot(qx, 4).flatten(-2).float()
        query_repr = self.base.query(qhot)
        if k:
            xhot = F.one_hot(sx[:, :k], 4).flatten(-2).float()
            yhot = F.one_hot(sy[:, :k], 4).flatten(-2).float()
            raw = self.base.pair(torch.cat((xhot, yhot), -1))
            # Q changes the learned WRITE transform applied to each current
            # support example, before it enters stage-local M and attention.
            gate = self.write_gate(raw).sigmoid()
            scale = torch.tanh(self.q_scale(q)).unsqueeze(1)
            shift = torch.tanh(self.q_shift(q)).unsqueeze(1)
            written = raw + 0.5 * gate * (scale * raw + shift)
            memory = written.mean(1, keepdim=True).expand(-1, nqueries, -1)
            scores = self.base.qproj(query_repr) @ self.base.key(written).transpose(-1, -2)
            scores = scores / (self.width ** 0.5)
            attended = scores.softmax(-1) @ self.base.value(written)
            summary = raw.mean(1)
        else:
            memory = torch.zeros_like(query_repr)
            attended = torch.zeros_like(query_repr)
            summary = torch.zeros(n, self.width, device=sx.device)
        logits = self.base.decoder(torch.cat((query_repr, memory, attended, qhot), -1))
        return logits.reshape(n, nqueries, 4, 4), summary

    def forward(self, sx, sy, qx, mode='adaptive',
                support=(12, 12, 12), intervention='normal'):
        if mode not in ('adaptive', 'fixed', 'frozen'):
            raise ValueError("invalid Q mode")
        if intervention not in ('normal', 'no_A', 'swap_A'):
            raise ValueError("invalid intervention")
        if len(support) != 3 or any(k < 0 or k > sx.shape[2] for k in support):
            raise ValueError("invalid support schedule")
        if intervention == 'swap_A' and sx.shape[0] % 2:
            raise ValueError("paired Q swapping requires even batch")
        q = torch.zeros(sx.shape[0], self.qdim, device=sx.device)
        outputs = []
        for stage_index in range(3):
            # IMPORTANT: stage() builds fresh memory; prior M never crosses B/C.
            logits, summary = self.stage(sx[:, stage_index], sy[:, stage_index],
                                         qx[:, stage_index], q, support[stage_index])
            outputs.append(logits)
            if stage_index >= 2 or mode in ('fixed', 'frozen'):
                continue
            if stage_index == 0 and intervention == 'no_A':
                continue
            if stage_index == 0 and intervention == 'swap_A':
                summary = summary.reshape(-1, 2, self.width).flip(1).reshape(-1, self.width)
            q = self.controller(summary, q)
        return torch.stack(outputs, dim=1)


def evaluate(model, pairs, mode='adaptive', limit=60, batch=12):
    model.eval()
    subset = pairs[:limit]
    report = {}
    with torch.no_grad():
        for k in KS:
            controls = ('normal', 'no_A', 'swap_A') if mode == 'adaptive' else ('normal',)
            for control in controls:
                sums = {f'{arm}_{s}': [0., 0., 0] for arm in ('linked', 'control') for s in ('B','C')}
                for offset in range(0, len(subset), batch):
                    sx, sy, qx, gold = batch_tensors(subset[offset:offset+batch], 'cpu')
                    logits = model(sx, sy, qx, mode=mode,
                                   support=(12, k, k), intervention=control)
                    pred = logits.argmax(-1)
                    token = (pred == gold).float().mean((-1,-2)).reshape(-1,2,3)
                    exact = (pred == gold).all(-1).float().mean(-1).reshape(-1,2,3)
                    for a, arm in enumerate(('linked','control')):
                        for s, stage in ((1,'B'),(2,'C')):
                            key = f'{arm}_{stage}'
                            sums[key][0] += token[:,a,s].sum().item()
                            sums[key][1] += exact[:,a,s].sum().item()
                            sums[key][2] += len(token)
                report[f'{control}_{k}'] = {
                    name: {'token': round(v[0]/v[2],6), 'exact': round(v[1]/v[2],6)}
                    for name, v in sums.items()
                }
    return report


def train(data: Path, pretrained: Path, out: Path, seed=37,
          steps=1200, batch_pairs=8, width=128, eval_pairs=60):
    if min(steps, batch_pairs, eval_pairs) < 1:
        raise ValueError("positive steps, batches and eval count required")
    torch.set_num_threads(1)
    train_pairs = read_dataset(data, 'train')
    val_pairs = read_dataset(data, 'val')[:eval_pairs]
    test_pairs = read_dataset(data, 'test')[:eval_pairs]
    result = {'seed': seed, 'steps': steps, 'width': width, 'eval_pairs': eval_pairs,
              'data_train_pairs': len(train_pairs), 'pretrained_checkpoint': pretrained.name,
              'note': 'write-conditioned Q; fixed-Q is an independently trained model'}
    for mode in ('fixed', 'adaptive'):
        torch.manual_seed(seed)
        randomizer = random.Random(seed)
        model = QWriteLearner(width=width)
        model.load_gate3(pretrained)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0007, weight_decay=0.0001)
        for step in range(steps):
            batch_data = randomizer.choices(train_pairs, k=batch_pairs)
            sx, sy, qx, gold = batch_tensors(batch_data, 'cpu')
            k = KS[step % len(KS)]
            model.train()
            logits = model(sx,sy,qx,mode=mode,support=(12,k,k))
            all_loss = F.cross_entropy(logits.reshape(-1,4), gold.reshape(-1),
                                       reduction='none').reshape(-1,3,16,4).mean((0,2,3))
            loss = all_loss @ torch.tensor([0.10, 0.45, 0.45])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            if step + 1 in (1, steps//2, steps):
                print(json.dumps({'seed':seed,'mode':mode,'step':step+1,
                                  'loss':round(loss.item(),5)}),flush=True)
        params = sum(p.numel() for p in model.parameters())
        result[mode] = {
            'parameters_allocated': params,
            'validation': evaluate(model,val_pairs,mode=mode,limit=eval_pairs),
            'test': evaluate(model,test_pairs,mode=mode,limit=eval_pairs)
        }
        if mode == 'adaptive':
            result['frozen_Q_test'] = evaluate(model,test_pairs,mode='frozen',limit=eval_pairs)
        out.mkdir(parents=True,exist_ok=True)
        torch.save({'model':model.state_dict(),'mode':mode,'seed':seed,'width':width},
                   out/f'{mode}.pt')
    if result['fixed']['parameters_allocated'] != result['adaptive']['parameters_allocated']:
        raise AssertionError('parameter mismatch')
    (out/'metrics.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n',encoding='utf8')
    for stage in ('B','C'):
        for mode in ('fixed','adaptive'):
            for arm in ('linked','control'):
                print(f'{seed} {mode} {arm} {stage} k12 {result[mode]["test"]["normal_12"][f"{arm}_{stage}"]}',flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path,required=True)
    p.add_argument('--pretrained',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=37)
    p.add_argument('--steps',type=int,default=1200)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--eval-pairs',type=int,default=60)
    a=p.parse_args()
    train(a.data,a.pretrained,a.out,a.seed,a.steps,a.batch,a.width,a.eval_pairs)


if __name__ == '__main__':
    main()
