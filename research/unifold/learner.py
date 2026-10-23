"""UNIFOLD early neural M/Q learner. Research prototype, not evidence of meta-learning.

Within an episode theta is fixed. M remembers observed pairs; Q modulates
how quickly M changes. Both arms of each curriculum start from fresh states.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F


class PlasticLearner(nn.Module):
    def __init__(self, width: int = 64, control: int = 16, embedding: int = 8):
        super().__init__()
        self.width, self.control = width, control
        self.symbol = nn.Embedding(4, embedding)
        self.input_encoder = nn.Sequential(nn.Linear(4 * embedding, width), nn.Tanh())
        self.output_encoder = nn.Sequential(nn.Linear(4 * embedding, width), nn.Tanh())
        self.reader = nn.Sequential(nn.Linear(width * 2, width), nn.Tanh(),
                                    nn.Linear(width, 16))
        self.memory_cell = nn.GRUCell(width * 2 + 16, width)
        self.control_cell = nn.GRUCell(width * 2 + 16 + 1, control)
        self.plasticity = nn.Linear(control, width)
        self.new_stage = nn.Linear(width, width)

    def state(self, batch: int, device: torch.device):
        return torch.zeros(batch, self.width, device=device), torch.zeros(batch, self.control, device=device)

    def encode(self, v: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        return encoder(self.symbol(v).flatten(1))

    def predict(self, x: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        return self.reader(torch.cat((self.encode(x, self.input_encoder), m), dim=-1)).reshape(-1, 4, 4)

    def observe(self, x: torch.Tensor, y: torch.Tensor, state, mode: str = 'adaptive'):
        m, q = state
        x_repr = self.encode(x, self.input_encoder)
        y_repr = self.encode(y, self.output_encoder)
        logits = self.reader(torch.cat((x_repr, m), dim=-1))
        error = F.cross_entropy(logits.reshape(-1, 4, 4).transpose(1, 2), y, reduction='none').mean(dim=1, keepdim=True)
        features = torch.cat((x_repr, y_repr, logits.softmax(-1)), dim=-1)
        if mode == 'adaptive':
            q = self.control_cell(torch.cat((features, error), dim=-1), q)
        elif mode != 'frozen':
            raise ValueError(f'Unsupported mode {mode}')
        proposal = self.memory_cell(features, m)
        gain = self.plasticity(q).sigmoid()
        m = m + gain * (proposal - m)
        return m, q

    def forward(self, support_x: torch.Tensor, support_y: torch.Tensor,
                query_x: torch.Tensor, mode: str = 'adaptive', support_sizes=None):
        """Input [batch,stage,examples,4], output [batch,stage,queries,4,4]."""
        m, q = self.state(support_x.shape[0], support_x.device)
        if support_sizes is None:
            support_sizes = (support_x.shape[2],) * 3
        if len(support_sizes) != 3 or any(n < 1 or n > support_x.shape[2] for n in support_sizes):
            raise ValueError("Invalid per-stage support size")
        stage_logits = []
        for stage in range(3):
            # Mark stage transition; preserve prior information in a learned way.
            if stage:
                m = .8 * m + .2 * self.new_stage(m).tanh()
            for i in range(support_sizes[stage]):
                m, q = self.observe(support_x[:, stage, i], support_y[:, stage, i], (m, q), mode)
            qx = query_x[:, stage].reshape(-1, 4)
            qm = m.repeat_interleave(query_x.shape[2], dim=0)
            stage_logits.append(self.predict(qx, qm).reshape(support_x.shape[0], -1, 4, 4))
        return torch.stack(stage_logits, dim=1)


def read_dataset(directory: Path, split: str):
    """The oracle is available to the TRAINER for loss/scoring; never to model.forward."""
    by_pair = defaultdict(dict)
    with (directory / 'sequences.jsonl').open(encoding='utf8') as public, (directory / 'answers.PRIVATE.jsonl').open(encoding='utf8') as private:
        for p, g in zip(public, private):
            p, g = json.loads(p), json.loads(g)
            if p['id'] != g['id']:
                raise ValueError('Mismatched records')
            if p['split'] != split:
                continue
            stage = p['stages']
            item = (
                torch.tensor([[e['input'] for e in s['support']] for s in stage], dtype=torch.long),
                torch.tensor([[e['output'] for e in s['support']] for s in stage], dtype=torch.long),
                torch.tensor([[e['input'] for e in s['queries']] for s in stage], dtype=torch.long),
                torch.tensor([stage_gold['query_outputs'] for stage_gold in g['stages']], dtype=torch.long))
            by_pair[g['pair_id']][g['condition']] = item
    result = []
    for pid in sorted(by_pair):
        arms = by_pair[pid]
        if set(arms) != {'linked', 'control'}:
            raise ValueError(f'Invalid matched pair {pid}')
        # Paired B/C are identical, only A changes (crucial control).
        for field in range(4):
            if not torch.equal(arms['linked'][field][1:], arms['control'][field][1:]):
                raise ValueError(f'Pair not matched: {pid}')
        result.append((arms['linked'], arms['control']))
    if not result:
        raise ValueError(f'No pairs for split {split}')
    return result


def batch_tensors(pairs, device):
    arms = [arm for pair in pairs for arm in pair]
    return [torch.stack([arm[i] for arm in arms]).to(device) for i in range(4)]


def run_one(model, pairs, mode, device, support_sizes=None):
    sx, sy, qx, qy = batch_tensors(pairs, device)
    logits = model(sx, sy, qx, mode, support_sizes)
    loss = F.cross_entropy(logits.reshape(-1, 4), qy.reshape(-1))
    accuracy = (logits.argmax(dim=-1) == qy).float().mean(dim=(-1, -2))
    return loss, accuracy.reshape(-1, 2, 3)


def evaluate(model, pairs, device, batch=8, support_sizes=None):
    model.eval()
    scores = defaultdict(list)
    with torch.no_grad():
        for offset in range(0, len(pairs), batch):
            part = pairs[offset:offset + batch]
            for mode in ('adaptive', 'frozen'):
                _, accuracy = run_one(model, part, mode, device, support_sizes)
                for j, stage in enumerate(('A_L1', 'B_L1', 'C_L2')):
                    for arm_i, arm in enumerate(('linked', 'control')):
                        scores[f'{mode}.{arm}.{stage}'].extend(accuracy[:, arm_i, j].tolist())
    return {k: round(sum(v)/len(v), 6) for k, v in sorted(scores.items())}


def train(data: Path, out: Path, steps: int = 30, batch: int = 4, width: int = 64,
          seed: int = 7, lr: float = 0.001, max_eval_pairs: int = 20):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    device = torch.device('cpu')
    train_pairs = read_dataset(data, 'train')
    val_pairs = read_dataset(data, 'val')[:max_eval_pairs]
    model = PlasticLearner(width=width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    last = None
    for step in range(steps):
        model.train()
        batch_pairs = random.choices(train_pairs, k=batch)
        optimizer.zero_grad(set_to_none=True)
        loss, accuracy = run_one(model, batch_pairs, 'adaptive', device)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        last = {'step': step + 1, 'train_loss': round(float(loss.detach()), 5),
                'train_token_accuracy': round(float(accuracy.detach().mean()), 5)}
        if (step + 1) % max(1, steps // 5) == 0:
            print(json.dumps(last), flush=True)
    evaluation = evaluate(model, val_pairs, device)
    learning_curves = {}
    for stage_index, stage_name in ((1, 'B_L1'), (2, 'C_L2')):
        for k in (1, 2, 4, 8, 12):
            sizes = [12, 12, 12]
            sizes[stage_index] = k
            curve = evaluate(model, val_pairs, device, support_sizes=tuple(sizes))
            for mode in ('adaptive', 'frozen'):
                for arm in ('linked', 'control'):
                    learning_curves.setdefault(f'{mode}.{arm}.{stage_name}', {})[str(k)] = curve[f'{mode}.{arm}.{stage_name}']
    out.mkdir(parents=True, exist_ok=True)
    results = {'status': 'SMOKE_ONLY_NOT_EVIDENCE_OF_LEARNING_TO_LEARN',
               'seed': seed, 'steps': steps, 'batch_pairs': batch, 'width': width,
               'parameter_count': sum(p.numel() for p in model.parameters()),
               'train_pairs': len(train_pairs), 'validation_pairs': len(val_pairs),
               'last_step': last, 'validation_token_accuracy': evaluation, 'learning_curves': learning_curves,
               'chance_token_accuracy': .25,
               'note': 'Accuracy per symbol, not exact four-symbol rule solution or learning-speed AULC.'}
    (out/'metrics.json').write_text(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True)+'\n')
    torch.save({'model': model.state_dict(), 'config': {'width': width, 'control': 16, 'embedding': 8},
                'seed': seed, 'steps': steps}, out/'checkpoint.pt')
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=30)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--eval-pairs', type=int, default=20)
    args = p.parse_args()
    if args.steps < 1 or args.batch < 1 or args.eval_pairs < 1:
        p.error('steps/batch/eval-pairs must be positive')
    print(json.dumps(train(args.data, args.out, args.steps, args.batch, args.width, args.seed, max_eval_pairs=args.eval_pairs), ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
