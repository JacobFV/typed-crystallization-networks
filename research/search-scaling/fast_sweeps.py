import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastrun

OUT = os.path.dirname(os.path.abspath(__file__))
S = 500
SEEDS = list(range(16))


def jobs(cfgs, seeds=SEEDS):
    return [(c, s) for c in cfgs for s in seeds]


SWEEPS = {
    # Axis 1 x Axis 3 on the raw logic-generator target distribution
    'FA_depth_wiring': jobs([{'depth': d, 'wiring': w, 'steps': S, 'sweep': 'FA'}
                             for d in (1, 2, 3, 4, 6, 8, 12, 16)
                             for w in ('supplied', 'free')]),
    # Axis 1 x Axis 3 on targets that genuinely depend on all four input bits
    'FH_hard': jobs([{'depth': d, 'wiring': w, 'steps': S, 'min_relevant': 4, 'sweep': 'FH'}
                     for d in (3, 4, 6, 8, 12, 16) for w in ('supplied', 'free')]),
    # Axis 2a: operator pool size
    'FB_pool': jobs([{'depth': 4, 'wiring': w, 'pool': p, 'steps': S, 'sweep': 'FB'}
                     for p in ('T2', 'T4', 'T8', 'T16') for w in ('free', 'supplied')]),
    # Axis 2b: predecessor window (how many legal bindings per node)
    'FC_window': jobs([{'depth': d, 'wiring': 'free', 'window': wd, 'steps': S,
                        'min_relevant': 4, 'sweep': 'FC'}
                       for d in (6, 8) for wd in (4, 5, 6, 8, None)]),
    # Axis 4: supervision density (every intermediate gate probed vs output only)
    'FD_supervision': jobs([{'depth': d, 'wiring': w, 'steps': S, 'min_relevant': 4,
                             'supervision': 'dense', 'sweep': 'FD'}
                            for d in (3, 4, 6, 8, 12, 16) for w in ('free',)]),
    # Optimisation budget
    'FE_budget': jobs([{'depth': d, 'wiring': 'free', 'steps': st, 'min_relevant': 4,
                        'sweep': 'FE'} for d in (4, 6, 8) for st in (250, 2000, 8000)]),
    # Fixed target, varying initialisation: combinatorial vs dynamical
    'FG_restarts': jobs([{'depth': d, 'wiring': 'free', 'steps': S, 'min_relevant': 4,
                          'fixed_target': True, 'target_index': ti, 'sweep': 'FG'}
                         for d in (4, 6, 8) for ti in (0, 1, 2, 3)],
                        seeds=list(range(24))),
    # Scaffold slack: more nodes than the target needs
    'FS_slack': jobs([{'depth': 4, 'n_nodes': n, 'wiring': 'free', 'steps': S,
                       'min_relevant': 4, 'sweep': 'FS'} for n in (4, 6, 8, 12)]),
    # Annealing / rounding ablations
    'FT_anneal': jobs([{'depth': d, 'wiring': 'free', 'steps': S, 'min_relevant': 4,
                        'tau1': t1, 'anneal': an, 'sweep': 'FT'}
                       for d in (4, 6, 8) for t1, an in ((1., False), (.1, True),
                                                          (.02, True), (.005, True))]),
}

if __name__ == '__main__':
    which = sys.argv[1:] or list(SWEEPS)
    workers = int(os.environ.get('TCN_WORKERS', '8'))
    for name in which:
        t = time.time()
        rows = fastrun.sweep(SWEEPS[name], workers=workers,
                             out=os.path.join(OUT, f'{name}.json'))
        ok = sum(1 for r in rows if r.get('success'))
        err = sum(1 for r in rows if 'error' in r)
        print(f'{name}: {len(rows)} runs, {ok} success, {err} errors, '
              f'{time.time()-t:.0f}s', flush=True)
