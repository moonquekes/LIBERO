#!/usr/bin/env python3
import json, collections, os
D = '/home/x/vla/libero/data/suction_dataset_multi_part_sorting/eval_r2'
for tag in ['A_r13cf', 'B_r13cf', 'A_r13cftd', 'B_r13cftd']:
    p = f'{D}/diag_{tag}.json'
    if not os.path.exists(p):
        print(f'--- {tag}: MISSING ---'); continue
    j = json.load(open(p))
    # 探测结构
    eps = None
    if isinstance(j, list):
        eps = j
    elif isinstance(j, dict):
        for k in ('episodes', 'results', 'records', 'detail', 'items'):
            if isinstance(j.get(k), list):
                eps = j[k]; break
        if eps is None:
            # 打印顶层键帮助定位
            print(f'--- {tag}: dict keys={list(j.keys())} ---'); continue
    by = collections.Counter()
    for e in eps:
        if not isinstance(e, dict):
            continue
        r = e.get('reason') or e.get('label') or e.get('outcome') or e.get('reason_label') or '?'
        instr = (e.get('instruction') or e.get('task') or e.get('bddl') or '')
        shape = 'rect' if 'rectangular' in instr else ('round' if 'round' in instr else ('tri' if 'triangular' in instr else '?'))
        by[(shape, r)] += 1
    print(f'--- {tag} (n={len(eps)}) ---')
    for k, v in sorted(by.items()):
        print(f'  {k}: {v}')
