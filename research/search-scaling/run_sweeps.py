import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import sweep

OUT = os.path.dirname(os.path.abspath(__file__))
SEEDS = list(range(12))
S = 500

def jobs(cfgs, seeds=SEEDS):
    return [(c, s) for c in cfgs for s in seeds]

SWEEPS = {
 'A_depth_wiring': jobs([{'depth': d, 'wiring': w, 'pool': 'T16', 'steps': S, 'sweep': 'A'}
                         for d in (1, 2, 3, 4, 6, 8) for w in ('supplied', 'free')]),
 'B_table_pool':   jobs([{'depth': 3, 'wiring': w, 'pool': p, 'steps': S, 'sweep': 'B'}
                         for p in ('T2', 'T4', 'T8', 'T16') for w in ('free', 'supplied')]),
 'C_window':       jobs([{'depth': 4, 'wiring': 'free', 'pool': 'T16', 'window': wd,
                          'steps': S, 'sweep': 'C'} for wd in (2, 3, 4, None)]),
 'D_supervision':  jobs([{'depth': d, 'wiring': 'free', 'pool': 'T16', 'steps': S,
                          'supervision': sup, 'sweep': 'D'}
                         for d in (2, 3, 4, 6) for sup in ('dense',)]),
 'E_budget':       jobs([{'depth': d, 'wiring': 'free', 'pool': 'T16', 'steps': st, 'sweep': 'E'}
                         for d in (3, 4) for st in (2000,)]),
 'G_restarts':    jobs([{'depth': d, 'wiring': 'free', 'pool': 'T16', 'steps': S,
                          'fixed_target': True, 'target_index': ti, 'sweep': 'G'}
                         for d in (2, 3, 4) for ti in (0, 1, 2, 3)], seeds=list(range(8))),
 'H_hard_targets': jobs([{'depth': d, 'wiring': w, 'pool': 'T16', 'steps': S,
                          'min_relevant': 4, 'sweep': 'H'}
                         for d in (3, 4, 6, 8) for w in ('supplied', 'free')], seeds=list(range(8))),
 'I_hard_dense':  jobs([{'depth': d, 'wiring': 'free', 'pool': 'T16', 'steps': S,
                          'min_relevant': 4, 'supervision': 'dense', 'sweep': 'I'}
                         for d in (3, 4, 6)], seeds=list(range(8))),
 'F_slack':        jobs([{'depth': 2, 'n_nodes': n, 'wiring': 'free', 'pool': 'T16',
                          'steps': S, 'sweep': 'F'} for n in (2, 3, 4)]),
}

if __name__ == '__main__':
    which = sys.argv[1:] or list(SWEEPS)
    for name in which:
        t = time.time()
        rows = sweep(SWEEPS[name], workers=int(os.environ.get('TCN_WORKERS','18')), out=os.path.join(OUT, f'{name}.json'))
        ok = sum(1 for r in rows if r.get('success'))
        err = sum(1 for r in rows if 'error' in r)
        print(f'{name}: {len(rows)} runs, {ok} success, {err} errors, {time.time()-t:.0f}s', flush=True)
