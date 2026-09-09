"""Verify the bound's headline against the real program, not the simulator.

`bound.py`'s sweep says the best member of the stage-B family reaches 0.7031 on
the held-out lengths. That number comes from a Python re-implementation of the
scaffold's semantics. It is re-derived here by running the actual hardened
`scaffolds.stage_b` program with that selection over the same episodes, because
the project has caught eight confident wrong conclusions by checking a headline
against raw data rather than a summary.
"""
from __future__ import annotations
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import splits as S, common
from run_stage_b import build_module, accuracy, baselines, scaffolds

STEPS = (-2, -1, 0, 1, 2)
SUB = list(range(0, 121))
RULES = [(op, v) for v in (-2, -1, 0, 1, 2) for op in ('eq', 'ge', 'le')]

CHECK = [
    {'c': 110, 'plus': -2, 'minus': 2, 'op': 'le', 'v': -2},   # best-in-family on held-out
    {'c': 110, 'plus': -2, 'minus': 1, 'op': 'le', 'v': -2},   # first best-train member in order
    {'c': 101, 'plus': 1, 'minus': -1, 'op': 'eq', 'v': 0},    # the honest counting program
]


def main():
    s = S.build()
    module, registry, frozen = build_module()
    out = {'stream': common.STREAM_POST_AUDIT}
    for positions in (16, 22):
        p, _ = scaffolds.stage_b(module, registry, positions=positions)
        rows = []
        for params in CHECK:
            sel = {n.name: 0 for n in p.nodes}
            sel['symbols'] = SUB.index(params['c'])
            sel['plus'] = STEPS.index(params['plus'])
            sel['minus'] = STEPS.index(params['minus'])
            sel['answer'] = RULES.index((params['op'], params['v']))
            r = {'params': params}
            for name, eps in s.items():
                acc, per = accuracy(p, sel, registry, eps)
                r[name] = {'accuracy': acc, 'majority_constant': baselines(eps)['majority_constant'],
                           'per_length': per}
            rows.append(r)
        out[f'positions_{positions}'] = rows
    json.dump(out, open(os.path.join(HERE, 'verify_best_member.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
