"""Addendum A2: the uniform-random-selection null for Q2.

Q2's falsification criterion is "the gradient path conforms at the same rate on
the min-prefix scaffold and on the counting-only scaffold". That comparison needs
a number for what chance alone achieves, otherwise "same rate" is rhetoric. This
draws uniformly from the same discrete spaces and reports the fraction of draws
that conform on the 24 training episodes, by exact execution through
`tcn.search.evaluate` -- the identical conformance test the gradient arms use.

Post-audit stream, pinned by `prepare.py`. Seed fixed at 0.
"""
from __future__ import annotations
import sys, os, json, time, random, argparse, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, prepare
from run_stage_b import build_module, examples, accuracy, baselines
from dyck_scaffold import stage_b_dyck
from tcn.search import evaluate, space_size, candidate_counts

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
lc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--draws', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--positions', type=int, default=22)
    a = ap.parse_args()
    s = prepare.load()
    train, unseen = s['train'], s['heldout_unseen_lengths']
    ex = examples(train)
    rows = []
    for tag in ('dyck', 'counting'):
        module, registry, _ = build_module()
        if tag == 'dyck':
            prog, signals = stage_b_dyck(module, registry, positions=a.positions)
        else:
            prog, signals = lc.stage_b(module, registry, positions=a.positions)
        names = [n.name for n in prog.nodes]; counts = candidate_counts(prog)
        rnd = random.Random(a.seed)
        t0 = time.perf_counter(); conf = 0; usable = 0; accs = []
        for _ in range(a.draws):
            sel = {n: rnd.randrange(c) for n, c in zip(names, counts)}
            err = evaluate(prog, sel, ex, signals, registry, tolerance=1e-6)
            if err is None:
                continue
            usable += 1
            if err <= 1e-6:
                conf += 1
                accs.append(accuracy(prog, sel, registry, unseen)[0])
        rows.append({'scaffold': tag, 'space_size': space_size(prog), 'draws': a.draws,
                     'usable_draws': usable, 'conforming_draws': conf,
                     'conforming_rate_over_draws': conf / a.draws,
                     'conforming_rate_over_usable': conf / usable if usable else None,
                     'conforming_heldout_unseen': accs,
                     'seconds': time.perf_counter() - t0})
        print(rows[-1], flush=True)
    rep = {'stream': common.STREAM_POST_AUDIT, 'seed': a.seed,
           'baselines_heldout_unseen': baselines(unseen), 'rows': rows}
    out = os.path.join(HERE, 'out', 'q2_null.json')
    json.dump(rep, open(out, 'w'), indent=1)
    print('wrote', out)


if __name__ == '__main__':
    main()
