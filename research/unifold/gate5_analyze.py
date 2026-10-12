#!/usr/bin/env python3
"""Summarize Gate 5 Q-write transfer with AULC, full-output accuracy and ablations.

Read only archived metrics; never retrain or consume gold solutions. Differences
of multiple episodes sharing the same law do not constitute independent laws.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import statistics

KS = (0, 1, 2, 4, 8, 12)


def auc(model: dict, level: str, *, split='test', intervention='normal',
        arm='linked', metric='token'):
    curve = [model[split][f'{intervention}_{k}'][f'{arm}_{level}'][metric] for k in KS]
    return sum((y0+y1)*0.5*(k1-k0) for k0,k1,y0,y1 in
               zip(KS[:-1],KS[1:],curve[:-1],curve[1:]))/(KS[-1]-KS[0])


def collect(files):
    if len(files) != 3:
        raise ValueError(f'Expected 3 independent seed files, got {len(files)}')
    models = [json.loads(p.read_text(encoding='utf-8')) for p in files]
    seeds = [v['seed'] for v in models]
    if sorted(seeds) != [37, 38, 39]:
        raise ValueError('Incorrect or repeated seeds')
    for m in models:
        if m['fixed']['parameters_allocated'] != m['adaptive']['parameters_allocated']:
            raise ValueError('Unequal allocated model size')
    out = {'source_run': 38235017905, 'seeds': seeds,
           'allocated_parameters_each': models[0]['fixed']['parameters_allocated'],
           'results': {},
           'scientific_caveat': 'Three seeds and 60 pairs each, repeated semantic laws; only exploratory.'}
    for split in ('validation', 'test'):
        out['results'][split] = {}
        for level in ('B', 'C'):
            rows = []
            for m in models:
                static = auc(m['fixed'],level,split=split)
                adaptive = auc(m['adaptive'],level,split=split)
                unrelated = auc(m['adaptive'],level,split=split,arm='control')
                no_a = auc(m['adaptive'],level,split=split,intervention='no_A')
                swapped = auc(m['adaptive'],level,split=split,intervention='swap_A')
                frozen = (auc(m['frozen_Q_test'],level,split='test')
                          if split=='test' else None)
                s = m['fixed'][split]['normal_12'][f'linked_{level}']
                a = m['adaptive'][split]['normal_12'][f'linked_{level}']
                rows.append({
                    'seed':m['seed'],
                    'fixed_auc': static, 'adaptive_auc': adaptive,
                    'adaptive_minus_fixed_auc_pp':100*(adaptive-static),
                    'linked_minus_unrelated_A_auc_pp':100*(adaptive-unrelated),
                    'normal_minus_no_A_auc_pp':100*(adaptive-no_a),
                    'normal_minus_swap_A_auc_pp':100*(adaptive-swapped),
                    'normal_minus_frozen_A_auc_pp':(100*(adaptive-frozen) if frozen is not None else None),
                    'fixed_token_12':s['token'], 'adaptive_token_12':a['token'],
                    'fixed_exact_12':s['exact'], 'adaptive_exact_12':a['exact'],
                })
            mean = {k:round(statistics.mean(row[k] for row in rows),7)
                    for k in rows[0] if k!='seed' and rows[0][k] is not None}
            out['results'][split][level]={'per_seed':rows,'mean':mean}
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    files=sorted(args.input.rglob('metrics.json'))
    report=collect(files)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+'\n',
                           encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    print('Source run:',report['source_run'],'allocated params:',report['allocated_parameters_each'])
    for split in ('validation','test'):
        for level in ('B','C'):
            m=report['results'][split][level]['mean']
            print(split,level,'AULC fixed=',round(100*m['fixed_auc'],3),
                  'adaptive=',round(100*m['adaptive_auc'],3),
                  'delta pp=',m['adaptive_minus_fixed_auc_pp'],
                  'linked-vs-unrelated A pp=',m['linked_minus_unrelated_A_auc_pp'],
                  'exact12=',round(100*m['adaptive_exact_12'],3),'%')

if __name__=='__main__':
    main()
