"""Measured evaluation rate for the Dyck scaffold, so the run is bounded by a
number rather than by a guess.

Samples the enumeration space uniformly, times `tcn.search.evaluate` on exactly
the objects `enumerate_fit` uses, and projects the total.
"""
from __future__ import annotations
import sys, os, json, time, random, itertools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import splits as splits_mod
from run_stage_b import build_module, examples
from dyck_scaffold import stage_b_dyck
from tcn.search import evaluate, space_size, candidate_counts

N = 600

if __name__ == '__main__':
    module, registry, frozen = build_module()
    program, signals = stage_b_dyck(module, registry, positions=22)
    total = space_size(program)
    counts = candidate_counts(program)
    names = [n.name for n in program.nodes]
    s = splits_mod.build()
    tr = examples(s['train'])
    rnd = random.Random(11)
    t0 = time.perf_counter()
    for _ in range(N):
        combo = [rnd.randrange(c) for c in counts]
        evaluate(program, dict(zip(names, combo)), tr, signals, registry, tolerance=1e-6)
    dt = time.perf_counter() - t0
    rate = dt / N
    out = {'space_size': total, 'sampled': N, 'seconds_per_program': rate,
           'projected_total_seconds': rate * total,
           'projected_total_hours': rate * total / 3600,
           'note': 'sampled uniformly at random from the same combination space '
                   'enumerate_fit walks; measured under the load present at the time'}
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, 'dyck_rate.json'), 'w'), indent=1)
