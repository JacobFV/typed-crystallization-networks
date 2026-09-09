import json, math, os, statistics as st, sys
from collections import defaultdict
OUT = os.path.dirname(os.path.abspath(__file__))


def load(*names):
    rows = []
    for n in names:
        p = os.path.join(OUT, n + '.json')
        if os.path.exists(p):
            rows += json.load(open(p))
    return rows


def med(xs):
    return st.median(xs) if xs else float('nan')


def group(rows, keyfn):
    g = defaultdict(list)
    for r in rows:
        if 'error' in r:
            continue
        g[keyfn(r['config'])].append(r)
    return g


COLS = ['n', 'exact success', 'log2 space', 'cand/node', 'med steps to hit',
        'ever hit', 'zero-loss but wrong', 'med final loss', 'med rows wrong/16',
        'med final H_norm', 'med sec', 'distinct wrong programs']


def table(rows, keyfn, header, order=None):
    g = group(rows, keyfn)
    keys = order or sorted(g)
    lines = ['| ' + ' | '.join(header + COLS) + ' |',
             '|' + '---|' * (len(header) + len(COLS))]
    for k in keys:
        rs = g.get(k, [])
        if not rs:
            continue
        succ = [r for r in rs if r['success']]
        fail = [r for r in rs if not r['success']]
        ever = [r for r in rs if r['first_success_step'] is not None]
        gap = [r for r in fail if r['final_loss'] < .01]
        wrong = {tuple(r['selections']) for r in fail}
        wfn = {r['final_mask'] for r in fail}
        lines.append('| ' + ' | '.join(list(k if isinstance(k, tuple) else (k,)) + [
            str(len(rs)),
            f'{len(succ)}/{len(rs)} ({100*len(succ)/len(rs):.0f}%)',
            f"{rs[0]['log2_space']:.1f}",
            f"{med([med(r['n_candidates']) for r in rs]):.0f}",
            f"{med([r['first_success_step'] for r in succ]):.0f}" if succ else '--',
            f'{100*len(ever)/len(rs):.0f}%',
            f'{len(gap)}/{len(fail)}' if fail else '--',
            f"{med([r['final_loss'] for r in rs]):.4f}",
            f"{med([r['hamming'] for r in rs]):.1f}",
            f"{med([r['final_entropy'] for r in rs]):.3f}",
            f"{med([r['seconds'] for r in rs]):.1f}",
            f'{len(wrong)}/{len(fail)} sel, {len(wfn)}/{len(fail)} fn' if fail else '--',
        ]) + ' |')
    return '\n'.join(lines), g


def entropy_traces(rows, keyfn, keys):
    g = group(rows, keyfn)
    out = []
    for k in keys:
        rs = g.get(k, [])
        if not rs:
            continue
        for label, sel in (('solved', [r for r in rs if r['success']]),
                           ('failed', [r for r in rs if not r['success']])):
            if not sel:
                continue
            steps = [t['step'] for t in sel[0]['trace']]
            hs = [med([r['trace'][i]['norm_entropy'] for r in sel]) for i in range(len(steps))]
            gs = [med([r['trace'][i]['grad_norm'] for r in sel]) for i in range(len(steps))]
            ls = [med([r['trace'][i]['loss'] for r in sel]) for i in range(len(steps))]
            out.append((k, label, len(sel), steps, hs, gs, ls))
    return out
