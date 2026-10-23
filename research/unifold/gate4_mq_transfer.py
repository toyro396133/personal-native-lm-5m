#!/usr/bin/env python3
"""UNIFOLD Gate 4: paired A-B-C transfer, adaptive-Q vs fixed-Q controls.

Identical neural architecture and parameter budget in both trained arms.
Q is only permitted to modulate reading current-stage support examples;
there is no Q -> logits path when target-stage support count equals zero.
This is a transfer *probe*, not proof that content improves a learning rule.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import random
import torch
from torch import nn
from torch.nn import functional as F
from learner import read_dataset, batch_tensors

K_VALUES = (0, 1, 2, 4, 8, 12)


class AdaptiveReadout(nn.Module):
    def __init__(self, width=128, qdim=16):
        super().__init__()
        self.width, self.qdim = width, qdim
        self.pair = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, width))
        self.query = nn.Sequential(nn.Linear(16, width), nn.GELU(), nn.Linear(width, width))
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.qproj = nn.Linear(width, width, bias=False)
        self.controller = nn.GRUCell(width, qdim)
        self.gain = nn.Linear(qdim, 2 * width)
        self.decoder = nn.Sequential(
            nn.Linear(width * 3 + 16, width * 2), nn.GELU(),
            nn.Linear(width * 2, width), nn.GELU(), nn.Linear(width, 16),
        )
        nn.init.normal_(self.gain.weight, std=.01)
        nn.init.zeros_(self.gain.bias)

    def encode_pairs(self, sx, sy, k):
        if k < 0 or k > sx.size(1):
            raise ValueError('invalid support size')
        if k == 0:
            return None
        px = F.one_hot(sx[:, :k], 4).flatten(-2).float()
        py = F.one_hot(sy[:, :k], 4).flatten(-2).float()
        return self.pair(torch.cat((px, py), dim=-1))

    def stage(self, sx, sy, qx, q, k):
        batch, nqueries = qx.shape[:2]
        qhot = F.one_hot(qx, 4).flatten(-2).float()
        qfeatures = self.query(qhot)
        supports = self.encode_pairs(sx, sy, k)
        if supports is None:
            memory = torch.zeros_like(qfeatures)
            attended = torch.zeros_like(qfeatures)
        else:
            memory = supports.mean(1, keepdim=True).expand(-1, nqueries, -1)
            scores = self.qproj(qfeatures) @ self.key(supports).transpose(-1,-2) / math.sqrt(self.width)
            attended = scores.softmax(-1) @ self.value(supports)
            gains = 1.0 + .5 * torch.tanh(self.gain(q)).unsqueeze(1)
            memory = memory * gains[..., :self.width]
            attended = attended * gains[..., self.width:]
        return self.decoder(torch.cat((qfeatures, memory, attended, qhot), -1)).reshape(batch, nqueries, 4, 4)

    def forward(self, sx, sy, qx, mode='adaptive', support=(12,12,12), intervention='normal'):
        """Input [2*paired_sequences,3,12,4] -> [2*paired_sequences,3,16,4,4]."""
        if len(support) != 3 or any(k < 0 or k > sx.shape[2] for k in support):
            raise ValueError('bad supports')
        if mode not in ('adaptive','static') or intervention not in ('normal','no_A','swap_A'):
            raise ValueError('bad control mode')
        if intervention == 'swap_A' and sx.shape[0] % 2:
            raise ValueError('swapping Q needs paired sequences')
        q = torch.zeros(sx.shape[0], self.qdim, device=sx.device, dtype=self.gain.weight.dtype)
        result = []
        for stage in range(3):
            k = support[stage]
            result.append(self.stage(sx[:,stage], sy[:,stage], qx[:,stage], q, k))
            features = self.encode_pairs(sx[:,stage], sy[:,stage], k)
            summary = (features.mean(1) if features is not None
                       else torch.zeros(sx.shape[0], self.width, device=sx.device))
            if mode == 'static' or (stage == 0 and intervention == 'no_A'):
                summary = torch.zeros_like(summary)
            if stage == 0 and intervention == 'swap_A':
                summary = summary.reshape(-1, 2, self.width).flip(1).reshape(-1, self.width)
            q = self.controller(summary, q)
        return torch.stack(result, dim=1)


def compare(model, pairs, limit=40, batch=10, mode='adaptive'):
    model.eval()
    subset=pairs[:limit]
    out={}
    with torch.no_grad():
        for k in K_VALUES:
            interventions=('normal','no_A','swap_A') if mode == 'adaptive' else ('normal',)
            for intervention in interventions:
                sums={name:[0.,0.,0] for name in ('linked_B','control_B','linked_C','control_C')}
                for start in range(0,len(subset),batch):
                    sx, sy, qx, qy = batch_tensors(subset[start:start+batch],'cpu')
                    pred=model(sx,sy,qx,mode=mode,support=(12,k,k),intervention=intervention).argmax(-1)
                    token=(pred==qy).float().mean((-1,-2)).reshape(-1,2,3)
                    exact=(pred==qy).all(-1).float().mean(-1).reshape(-1,2,3)
                    for arm_idx,arm in enumerate(('linked','control')):
                        for stage_idx,stage in ((1,'B'),(2,'C')):
                            name=f'{arm}_{stage}'
                            sums[name][0]+=float(token[:,arm_idx,stage_idx].sum())
                            sums[name][1]+=float(exact[:,arm_idx,stage_idx].sum())
                            sums[name][2]+=len(token)
                out[f'{intervention}_{k}']={name:{'token':round(v[0]/v[2],6),
                                               'exact':round(v[1]/v[2],6)}
                                            for name,v in sums.items()}
    return out


def train(data: Path, output: Path, seed=37, steps=900, batch_pairs=8,
          width=128, eval_pairs=40):
    if steps<1 or batch_pairs<1 or eval_pairs<1:
        raise ValueError('invalid training size')
    torch.set_num_threads(1)
    pairs_train=read_dataset(data,'train')
    pairs_val=read_dataset(data,'val')[:eval_pairs]
    pairs_test=read_dataset(data,'test')[:eval_pairs]
    report={'seed':seed, 'steps':steps, 'batch_pairs':batch_pairs, 'width':width,
            'train_pairs':len(pairs_train), 'eval_pairs':len(pairs_val),
            'caveat':'Probing content-conditioned Q. Cannot alone prove improved learning rule.'}
    for mode in ('static','adaptive'):
        # Both models see same random seed, initial weights, and batches.
        rng=random.Random(seed)
        torch.manual_seed(seed)
        model=AdaptiveReadout(width=width)
        params=sum(p.numel() for p in model.parameters())
        optimizer=torch.optim.AdamW(model.parameters(),lr=.0008)
        for step in range(steps):
            items=rng.choices(pairs_train,k=batch_pairs)
            sx,sy,qx,qy=batch_tensors(items,'cpu')
            k=K_VALUES[step % len(K_VALUES)]
            model.train()
            logits=model(sx,sy,qx,mode=mode,support=(12,k,k))
            stage_loss=F.cross_entropy(logits.reshape(-1,4),qy.reshape(-1),reduction='none').reshape(-1,3,16,4).mean((0,2,3))
            loss=stage_loss @ torch.tensor([.10,.45,.45])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            if step+1 in (1, steps//2, steps):
                print(json.dumps({'seed':seed,'mode':mode,'step':step+1,
                                  'loss':round(float(loss.detach()),5)}),flush=True)
        report[mode]={'params':params,
                      'validation':compare(model,pairs_val,limit=eval_pairs,mode=mode),
                      'test':compare(model,pairs_test,limit=eval_pairs,mode=mode)}
        output.mkdir(parents=True,exist_ok=True)
        torch.save({'state_dict':model.state_dict(),'mode':mode,'seed':seed,'width':width},
                   output/f'{mode}.pt')
    (output/'metrics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',encoding='utf8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=37)
    p.add_argument('--steps',type=int,default=900)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--width',type=int,default=128)
    p.add_argument('--eval-pairs',type=int,default=40)
    a=p.parse_args()
    result=train(a.data,a.out,a.seed,a.steps,a.batch,a.width,a.eval_pairs)
    for mode in ('static','adaptive'):
        for stage in ('B','C'):
            for condition in ('linked','control'):
                v=result[mode]['test']['normal_12'][f'{condition}_{stage}']
                print(f'{mode} {condition} {stage} k=12: {v}',flush=True)


if __name__=='__main__':
    main()
