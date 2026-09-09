import json, math, os, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import load, table, entropy_traces, group, med
OUT = os.path.dirname(os.path.abspath(__file__))

def spark(ys, lo=None, hi=None, width=None):
    chars = ' .:-=+*#%@'
    lo = min(ys) if lo is None else lo; hi = max(ys) if hi is None else hi
    if hi - lo < 1e-9: hi = lo + 1e-9
    return ''.join(chars[min(9, max(0, int(9 * (y - lo) / (hi - lo))))] for y in ys)

def section(title, rows, keyfn, header, order=None):
    t, stats = table(rows, keyfn, header, order)
    return f'### {title}\n\n{t}\n', stats

def main():
    parts = []
    A = load('A_depth_wiring')
    t, _ = table(A, lambda c: (str(c['depth']), c['wiring']), ['depth', 'wiring'],
                 order=[(str(d), w) for d in (1,2,3,4,6,8) for w in ('supplied','free')])
    parts.append('## Axis 1 + 3 - depth x wiring\n\n' + t + '\n')

    B = load('B_table_pool')
    if B:
        t, _ = table(B, lambda c: (c['pool'], c['wiring']), ['pool', 'wiring'],
                     order=[(p, w) for p in ('T2','T4','T8','T16') for w in ('free','supplied')])
        parts.append('## Axis 2a - operator pool (depth 3)\n\n' + t + '\n')
    C = load('C_window')
    if C:
        t, _ = table(C, lambda c: (str(c.get('window')),), ['pred window'],
                     order=[('2',), ('3',), ('4',), ('None',)])
        parts.append('## Axis 2b - predecessor window (depth 4, free wiring)\n\n' + t + '\n')
    D = load('D_supervision')
    if D:
        t, _ = table(D, lambda c: (str(c['depth']), c.get('supervision','sparse')),
                     ['depth', 'supervision'])
        parts.append('## Axis 4 - supervision density (free wiring)\n\n' + t + '\n')
    E = load('E_budget')
    if E:
        t, _ = table(E, lambda c: (str(c['depth']), str(c['steps'])), ['depth', 'steps'])
        parts.append('## Budget control - 2000 steps (free wiring)\n\n' + t + '\n')
    F = load('F_slack')
    if F:
        t, _ = table(F, lambda c: (str(c['n_nodes']),), ['scaffold nodes (depth-2 target)'])
        parts.append('## Scaffold slack (free wiring)\n\n' + t + '\n')
    G = load('G_restarts')
    if G:
        t, _ = table(G, lambda c: (str(c['depth']), 'target#' + str(c['target_index'])),
                     ['depth', 'target'])
        parts.append('## Fixed target, varying init seed\n\n' + t + '\n')
    print('\n'.join(parts))

if __name__ == '__main__':
    main()
