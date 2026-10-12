#!/usr/bin/env python3
"""UNIFOLD A->B->C sequential-learning dataset (Gate 1, no neural training).

Each PAIR has two sequences with identical B/C support/query inputs, labels,
and timing: in the matched control, A is replaced with an unrelated L1 rule.
Private oracle labels and matching/compositional witnesses are never public.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any

import taskgen as tg

SCHEMA = 'unifold-curriculum-v0.1'
STAGES = ('A', 'B', 'C')
SPLITS = ('train', 'val', 'test')


def candidates(seed: int):
    pool = tg.make_pool(seed)
    full = {k: {sig: prog for prog, sig in rows} for k, rows in pool.items()}
    relations = {}
    transitions = {name: bytes(tg.encode(tg.op(name, x)) for x in tg.DOMAIN)
                   for name in tg.OPS}
    pair_signatures = {(x, y): bytes(transitions[y][z] for z in transitions[x])
                       for x in tg.OPS for y in tg.OPS}
    for split in SPLITS:
        bucket = 'train_l1' if split == 'train' else f'{split}_l1'
        c_bucket = 'train_l1' if split == 'train' else f'{split}_l2'
        cases = []
        seen = set()
        for a_prog, a_sig in pool['train_l1']:
            for b_prog, b_sig in pool[bucket]:
                if a_sig == b_sig:
                    continue
                if split == 'train':
                    constructions = (
                        (a_prog[0], b_prog[1]), (a_prog[1], b_prog[0]),
                        (b_prog[0], a_prog[1]), (b_prog[1], a_prog[0]),
                    )
                else:
                    constructions = (
                        (*a_prog, b_prog[0]), (*a_prog, b_prog[1]),
                        (b_prog[0], *a_prog), (b_prog[1], *a_prog),
                    )
                for witness in constructions:
                    if split == 'train':
                        sig = pair_signatures[witness]
                    elif witness[:2] == a_prog:
                        sig = bytes(transitions[witness[2]][z] for z in a_sig)
                    else:
                        sig = bytes(a_sig[z] for z in transitions[witness[0]])
                    if sig not in full[c_bucket] or sig in (a_sig, b_sig):
                        continue
                    key = (a_sig, b_sig, sig)
                    if key in seen:
                        continue
                    seen.add(key)
                    cases.append((a_prog, a_sig, b_prog, b_sig, witness, sig))
        if not cases:
            raise RuntimeError(f'No compositional curricula for {split}')
        relations[split] = cases
    return pool, relations


def sample_support(sig: bytes, level: str, all_sigs: dict[str, list[bytes]],
                   rng: random.Random, support: int, query: int):
    # Identifiable within the entire declared level, not only own dataset split.
    for _ in range(200):
        selected = rng.sample(range(len(tg.DOMAIN)), support + query)
        s, q = selected[:support], selected[support:]
        if sum(all(other[j] == sig[j] for j in s) for other in all_sigs[level]) == 1:
            return s, q
    raise RuntimeError('Support insufficient to identify rule: increase --support')


def stage_obj(letter: str, sig: bytes, level: str, rng: random.Random,
              all_sigs: dict[str, list[bytes]], support: int, query: int):
    sup, queries = sample_support(sig, level, all_sigs, rng, support, query)
    return ({'stage': letter, 'level': level.upper(),
             'support': [{'input': list(tg.DOMAIN[j]), 'output': list(tg.DOMAIN[sig[j]])} for j in sup],
             'queries': [{'input': list(tg.DOMAIN[j])} for j in queries]},
            {'semantic_sha256': tg.fingerprint(sig),
             'query_outputs': [list(tg.DOMAIN[sig[j]]) for j in queries]})


def compute_counts(total: int):
    if total < 3:
        raise ValueError('Need at least three paired curricula')
    train = max(1, total * 6 // 10)
    val = max(1, total * 2 // 10)
    return {'train': train, 'val': val, 'test': total - train - val}


def generate(output: Path, pairs: int = 1000, seed: int = 42, support: int = 12,
             query: int = 16) -> dict:
    if support < 1 or query < 1 or support + query > len(tg.DOMAIN):
        raise ValueError('Bad example allocation')
    counts = compute_counts(pairs)
    if any(v < 1 for v in counts.values()):
        raise ValueError('Each split needs at least one pair')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Will not overwrite dataset: {output}')
    output.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pool, relations = candidates(seed)
    all_sigs = {level: [sig for key, rows in pool.items() if key.endswith(level)
                        for _, sig in rows] for level in ('l1', 'l2')}
    order = [split for split in SPLITS for _ in range(counts[split])]
    rng.shuffle(order)
    tasks_path = output / 'sequences.jsonl'
    answers_path = output / 'answers.PRIVATE.jsonl'
    used_relations = {split: set() for split in SPLITS}
    observed = Counter()
    with tasks_path.open('w', encoding='utf-8', newline='\n') as tasks, \
         answers_path.open('w', encoding='utf-8', newline='\n') as answers:
        for i, split in enumerate(order):
            # Prefer fresh A/B/C combinations when available; duplicate programs
            # are permitted across pairs after the finite pool is exhausted.
            options = relations[split]
            for _ in range(100):
                candidate = rng.choice(options)
                key = tuple(tg.fingerprint(candidate[j]) for j in (1, 3, 5))
                if key not in used_relations[split]:
                    break
            used_relations[split].add(key)
            a_prog, a_sig, b_prog, b_sig, witness, c_sig = candidate
            # Distractor is semantically distinct and doesn't share primitives.
            unrelated = [(p, s) for p, s in pool['train_l1']
                         if s != a_sig and not (set(p) & (set(a_prog) | set(b_prog)))]
            if not unrelated:
                raise RuntimeError('Unable to construct unrelated A')
            control_prog, control_sig = rng.choice(unrelated)
            c_level = 'l1' if split == 'train' else 'l2'
            # Identical B/C exposures, targets and order for both paired arms.
            b_public, b_gold = stage_obj('B', b_sig, 'l1', rng, all_sigs, support, query)
            c_public, c_gold = stage_obj('C', c_sig, c_level, rng, all_sigs, support, query)
            # Matched example-input indices for A and distractor, but each rule's
            # own outputs; identical counts and computational budget.
            a_public, a_gold = stage_obj('A', a_sig, 'l1', rng, all_sigs, support, query)
            control_public, control_gold = stage_obj('A', control_sig, 'l1', rng, all_sigs, support, query)
            pair_id = f'ufseq-{i:06d}'
            arms = [('linked', a_public, a_gold, a_prog),
                    ('control', control_public, control_gold, control_prog)]
            rng.shuffle(arms)
            for arm_no, (condition, a_pub, a_private, chosen_a_prog) in enumerate(arms):
                uid = f'{pair_id}-{arm_no}'
                public = {'id': uid, 'schema': SCHEMA, 'split': split,
                          'stages': [a_pub, b_public, c_public]}
                private = {'id': uid, 'pair_id': pair_id, 'condition': condition,
                           'stages': [a_private, b_gold, c_gold],
                           'programs': {'A': list(chosen_a_prog), 'B': list(b_prog),
                                        'C': list(witness)},
                           'construct_C_from': 'A+B' if condition == 'linked' else 'control_A_is_unrelated'}
                tasks.write(tg.serialize(public) + '\n')
                answers.write(tg.serialize(private) + '\n')
                observed[f'{split}_{condition}'] += 1
    manifest = {'schema': SCHEMA, 'seed': seed, 'pair_count': pairs,
                'sequence_count': pairs * 2, 'support': support, 'query': query,
                'pair_counts': counts, 'arms': dict(observed),
                'train_levels': ['L1'], 'eval_levels': ['L1', 'L2'],
                'tasks_sha256': tg.file_hash(tasks_path),
                'answers_sha256': tg.file_hash(answers_path),
                'constraint': 'C is composed from A+B in linked arm; B/C matched in control arm',
                'limitations': 'Controls measure transfer, not yet causal Q improvement; function language is closed.'}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False,
                                            sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return manifest


def validate(output: Path) -> dict:
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != SCHEMA:
        raise ValueError('Incompatible curriculum schema')
    for filename, key in (('sequences.jsonl', 'tasks_sha256'),
                          ('answers.PRIVATE.jsonl', 'answers_sha256')):
        if tg.file_hash(output / filename) != manifest[key]:
            raise ValueError('Checksum mismatch: '+filename)
    pool, relations = candidates(manifest['seed'])
    allowed = {key: {tg.fingerprint(sig) for _, sig in rows} for key, rows in pool.items()}
    all_sigs = {level: [sig for key, rows in pool.items() if key.endswith(level)
                        for _, sig in rows] for level in ('l1', 'l2')}
    seen_ids: set[str] = set()
    pairs: dict[str, list[tuple[dict, dict]]] = {}
    per_split: Counter[str] = Counter()
    with (output / 'sequences.jsonl').open(encoding='utf-8') as tasks_handle, \
         (output / 'answers.PRIVATE.jsonl').open(encoding='utf-8') as answers_handle:
        for line_no, lines in enumerate(itertools.zip_longest(
                tasks_handle, answers_handle), 1):
            if None in lines:
                raise ValueError('Public/private count mismatch')
            public, private = (json.loads(text) for text in lines)
            uid = public['id']
            if uid in seen_ids or uid != private['id'] or public['schema'] != SCHEMA:
                raise ValueError('Broken identity at line '+str(line_no))
            seen_ids.add(uid)
            split = public['split']
            if split not in SPLITS:
                raise ValueError('Invalid split')
            if 'programs' in public or 'condition' in public or 'pair_id' in public:
                raise ValueError('Leaked private fields')
            if [s['stage'] for s in public['stages']] != list(STAGES):
                raise ValueError('Bad stage order')
            c_level = 'L1' if split == 'train' else 'L2'
            if [s['level'] for s in public['stages']] != ['L1', 'L1', c_level]:
                raise ValueError('Invalid curriculum complexity')
            for pub_stage, gold_stage, letter in zip(public['stages'], private['stages'], STAGES):
                program = private['programs'][letter]
                sig = bytes(tg.encode(tg.apply_program(program, inp)) for inp in tg.DOMAIN)
                if tg.fingerprint(sig) != gold_stage['semantic_sha256']:
                    raise ValueError('Law mismatch')
                level = pub_stage['level'].lower()
                bucket = ('train_l1' if letter == 'A' else
                          f'{split}_{level}' if split != 'train' else 'train_l1')
                if gold_stage['semantic_sha256'] not in allowed[bucket]:
                    raise ValueError(f'Law belongs to wrong split: {uid}/{letter}')
                if len(program) != (2 if level == 'l1' else 3):
                    raise ValueError('Wrong structural level')
                if len(pub_stage['support']) != manifest['support'] or len(pub_stage['queries']) != manifest['query']:
                    raise ValueError('Wrong support/query size')
                used = set()
                for pair in pub_stage['support']:
                    x, y = tuple(pair['input']), tuple(pair['output'])
                    if x not in tg.DOMAIN or x in used or tg.apply_program(program, x) != y:
                        raise ValueError('Incorrect support')
                    used.add(x)
                for pair, expected in zip(pub_stage['queries'], gold_stage['query_outputs']):
                    x = tuple(pair['input'])
                    if x not in tg.DOMAIN or x in used or tg.apply_program(program, x) != tuple(expected):
                        raise ValueError('Incorrect oracle')
                    used.add(x)
                support_pairs = [(tg.encode(s['input']), tg.encode(s['output'])) for s in pub_stage['support']]
                if sum(all(candidate[x] == y for x, y in support_pairs) for candidate in all_sigs[level]) != 1:
                    raise ValueError('Ambiguous support')
            pair_id = private['pair_id']
            pairs.setdefault(pair_id, []).append((public, private))
            per_split[split] += 1
    if len(seen_ids) != manifest['sequence_count'] or len(pairs) != manifest['pair_count']:
        raise ValueError('Wrong number of sequences or pairs')
    for pair_id, arms in pairs.items():
        if len(arms) != 2 or {gold['condition'] for _, gold in arms} != {'linked', 'control'}:
            raise ValueError('Bad linked/control pairing '+pair_id)
        linked = next((a, b) for a, b in arms if b['condition'] == 'linked')
        control = next((a, b) for a, b in arms if b['condition'] == 'control')
        lp, lg = linked
        cp, cg = control
        if lp['split'] != cp['split']:
            raise ValueError('Cross-split pair')
        if lp['stages'][1:] != cp['stages'][1:] or lg['stages'][1:] != cg['stages'][1:]:
            raise ValueError('Unmatched B/C episodes in paired comparison')
        a_prog, b_prog, c_prog = (tuple(lg['programs'][x]) for x in STAGES)
        if lp['split'] == 'train':
            options = ((a_prog[0], b_prog[1]), (a_prog[1], b_prog[0]),
                       (b_prog[0], a_prog[1]), (b_prog[1], a_prog[0]))
        else:
            options = ((*a_prog, b_prog[0]), (*a_prog, b_prog[1]),
                       (b_prog[0], *a_prog), (b_prog[1], *a_prog))
        if c_prog not in options:
            raise ValueError('C not constructed from linked A and B')
        distractor = tuple(cg['programs']['A'])
        if set(distractor) & (set(a_prog) | set(b_prog)):
            raise ValueError('Distractor overlaps linked primitive rules')
        if cg['stages'][0]['semantic_sha256'] == lg['stages'][0]['semantic_sha256']:
            raise ValueError('Distractor equals A')
        if lp['stages'][0] == cp['stages'][0]:
            raise ValueError('A was not replaced')
    expected = {split: 2 * count for split, count in manifest['pair_counts'].items()}
    if dict(per_split) != expected:
        raise ValueError('Wrong split counts')
    return {'passed': True, 'pair_count': len(pairs), 'sequence_count': len(seen_ids),
            'per_split': dict(per_split), 'matched_BC': True,
            'C_compositional_witness': True, 'unique_support_within_level': True,
            'sha256': manifest['tasks_sha256']}


def main():
    parser = argparse.ArgumentParser(description='UNIFOLD Gate 1 matched curriculum')
    sub = parser.add_subparsers(dest='command', required=True)
    g = sub.add_parser('generate')
    g.add_argument('--output', required=True, type=Path)
    g.add_argument('--pairs', type=int, default=1000)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--support', type=int, default=12)
    g.add_argument('--queries', type=int, default=16)
    v = sub.add_parser('validate')
    v.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = generate(args.output, args.pairs, args.seed, args.support, args.queries) if args.command == 'generate' else validate(args.output)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
