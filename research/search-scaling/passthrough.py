"""Pass-through / degeneracy diagnostic (DLGN depth-pathology comparison).

Reconstructs each run's scaffold deterministically from (config, seed) and reads the
recorded argmax selections, so no retraining is needed.

Gate taxonomy over the 16 two-input tables (index i = 2*a + b):
  wire      : truth_12 (= A) and truth_10 (= B)          <- DLGN "pass-through"
  constant  : truth_0, truth_15
  unary     : truth_3 (~A), truth_5 (~B) - one input ignored but negated
  binary    : the remaining 10 tables - genuinely two-input computation
A node is also non-computing if its two sources are the same port (any table then
collapses to a unary function of that port).
"""
import json, os, statistics as st, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import (POOLS, N_IN, INPUT_MASKS, apply_table, sample_target, build_program)

WIRE = {10, 12}
CONST = {0, 15}
UNARY = {3, 5}

def reconstruct(cfg, seed):
    depth = cfg['depth']; n_nodes = cfg.get('n_nodes', depth)
    tables = POOLS[cfg.get('pool', 'T16')]
    tindex = cfg.get('target_index', 0) + (0 if cfg.get('fixed_target') else 1000 * seed)
    gates, mask, gate_masks, _ = sample_target(
        depth, cfg.get('target_seed', 0), tindex,
        tables=tables if cfg.get('restrict_target', True) else None,
        window=cfg.get('window'), min_relevant=cfg.get('min_relevant', 0))
    program, _ = build_program(n_nodes, tables, cfg['wiring'], gates, cfg.get('window'))
    return program, gates, mask

def live_nodes(program, selections):
    """Nodes in the transitive cone of the single output."""
    chosen = {n.name: n.candidates[selections[i]] for i, n in enumerate(program.nodes)}
    keep = set(); frontier = [program.outputs[0][1]]
    while frontier:
        v = frontier.pop()
        if v in keep or v not in chosen: continue
        keep.add(v); frontier.extend(chosen[v].sources)
    return keep, chosen

def classify(rows):
    out = defaultdict(list)
    for r in rows:
        if 'error' in r: continue
        program, gates, mask = reconstruct(r['config'], r['seed'])
        keep, chosen = live_nodes(program, r['selections'])
        for scope, names in (('all', [n.name for n in program.nodes]), ('live', sorted(keep))):
            wire = const = unary = same = 0
            for name in names:
                c = chosen[name]; t = int(c.operator.name[6:])
                if c.sources[0] == c.sources[1]: same += 1
                if t in WIRE: wire += 1
                elif t in CONST: const += 1
                elif t in UNARY: unary += 1
            n = max(1, len(names))
            noncomp = sum(1 for name in names
                          if int(chosen[name].operator.name[6:]) in WIRE | CONST | UNARY
                          or chosen[name].sources[0] == chosen[name].sources[1])
            out[(r['config'].get('sweep'), r['config']['depth'], r['config']['wiring'],
                 r['config'].get('min_relevant', 0), r['config'].get('supervision', 'sparse'),
                 scope)].append({
                'wire': wire / n, 'const': const / n, 'unary': unary / n,
                'noncomputing': noncomp / n, 'n_nodes': len(names),
                'live_frac': len(keep) / max(1, len(program.nodes)),
                'success': r['success'],
                'target_wire': sum(1 for g in gates if g[2] in WIRE) / len(gates),
                'target_noncomp': sum(1 for g in gates
                                      if g[2] in WIRE | CONST | UNARY or g[0] == g[1]) / len(gates)})
    return out

def report(names, scope='live', only_sweep=None):
    rows = []
    for nm in names:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), nm + '.json')
        if os.path.exists(p): rows += json.load(open(p))
    g = classify(rows)
    keys = sorted(k for k in g if k[5] == scope and (only_sweep is None or k[0] == only_sweep))
    lines = ['| sweep | depth | wiring | min_rel | superv | n runs | live nodes | '
             'wire (pass-through) | constant | unary | non-computing | target circuit wire | '
             'wire on solved | wire on failed |',
             '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
    for k in keys:
        v = g[k]
        s = [x for x in v if x['success']]; f = [x for x in v if not x['success']]
        lines.append('| ' + ' | '.join([
            str(k[0]), str(k[1]), k[2], str(k[3]), k[4], str(len(v)),
            f"{st.mean([x['live_frac'] for x in v]):.2f}",
            f"{100*st.mean([x['wire'] for x in v]):.1f}%",
            f"{100*st.mean([x['const'] for x in v]):.1f}%",
            f"{100*st.mean([x['unary'] for x in v]):.1f}%",
            f"{100*st.mean([x['noncomputing'] for x in v]):.1f}%",
            f"{100*st.mean([x['target_wire'] for x in v]):.1f}%",
            f"{100*st.mean([x['wire'] for x in s]):.1f}%" if s else '--',
            f"{100*st.mean([x['wire'] for x in f]):.1f}%" if f else '--']) + ' |')
    return '\n'.join(lines)

if __name__ == '__main__':
    names = sys.argv[1:] or ['A_depth_wiring', 'H_hard_targets', 'D_supervision', 'I_hard_dense']
    for scope in ('live', 'all'):
        print(f'\n### scope = {scope}\n')
        print(report(names, scope))
