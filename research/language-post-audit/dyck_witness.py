"""Existence by witness: an explicit member of the Dyck family, scored on every split.

The full 680,625-program enumeration over this scaffold was measured at 4.18 h
(`dyck_rate.json`) and was stopped unexhausted, so it carries no certificate.
Existence, however, does not need one -- it needs a witness. This exhibits the
member, states where it sits in the enumeration order, confirms it fits all 24
training episodes at max error 0.0, and reports its held-out accuracy beside the
same baselines every other number in this track is reported against.

The witness is not "the answer read off the data": it is the textbook Dyck
reduction -- step +1 on '(' and -1 on ')' over exactly the string's own symbols,
accept iff the total is 0 and the running minimum never goes below 0 -- and both
readouts are members of the same `{eq, ge, le} x {-2..2}` grid stage B searches.
"""
from __future__ import annotations
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, splits as splits_mod
from run_stage_b import build_module, examples, accuracy, baselines
from dyck_scaffold import stage_b_dyck
from tcn.search import evaluate, space_size, candidate_counts

STEPS = (-2, -1, 0, 1, 2)
SUB = list(range(0, 121))
RULES = [(op, v) for v in (-2, -1, 0, 1, 2) for op in ('eq', 'ge', 'le')]

WITNESS = {'c': 101, 'plus': 1, 'minus': -1,
           'total_ok': ('eq', 0), 'min_ok': ('ge', 0)}


def main():
    module, registry, frozen = build_module()
    program, signals = stage_b_dyck(module, registry, positions=22)
    names = [n.name for n in program.nodes]
    counts = candidate_counts(program)
    sel = {n.name: 0 for n in program.nodes}
    sel['symbols'] = SUB.index(WITNESS['c'])
    sel['plus'] = STEPS.index(WITNESS['plus'])
    sel['minus'] = STEPS.index(WITNESS['minus'])
    sel['total_ok'] = RULES.index(WITNESS['total_ok'])
    sel['min_ok'] = RULES.index(WITNESS['min_ok'])

    # where this member sits in enumerate_fit's mixed-radix enumeration order
    index = 0
    for name, c in zip(names, counts):
        index = index * c + sel[name]

    s = splits_mod.build()
    tr = examples(s['train'])
    err = evaluate(program, sel, tr, signals, registry, tolerance=1e-6)

    chosen = program.harden(sel).pruned()
    out = {'stream': common.STREAM_POST_AUDIT, 'positions': 22,
           'scaffold': 'stage_b_dyck', 'space_size': space_size(program),
           'witness': {k: (list(v) if isinstance(v, tuple) else v) for k, v in WITNESS.items()},
           'enumeration_index': index,
           'fraction_through_enumeration': index / space_size(program),
           'train_max_error': err,
           'conforms_on_train': err is not None and err <= 1e-6,
           'live_nodes': len(chosen.nodes),
           'description_bits': chosen.description_bits(registry),
           'execution_cost': chosen.execution_cost(registry),
           'module_cost': frozen.execution_cost(registry)}
    for name, eps in s.items():
        acc, per_len = accuracy(program, sel, registry, eps)
        out[name] = {'accuracy': acc, 'per_length': per_len,
                     'lengths': sorted({e['length'] for e in eps}),
                     'depths': sorted({e['depth'] for e in eps}), **baselines(eps)}
        print(name, round(acc, 6), 'majority', round(out[name]['majority_constant'], 4),
              per_len, flush=True)
    unseen = s['heldout_unseen_lengths']
    t0 = time.perf_counter(); n = min(50, len(unseen))
    for e in unseen[:n]:
        chosen.run({'text': e['text']}, registry=registry)
    out['batch_one_latency_ms'] = (time.perf_counter() - t0) / n * 1e3
    # The pruned program dict is ~1 MB and an equivalent, search-certified copy is
    # already in `dyck_p22_c95-110.json`; only its digest is kept here.
    out['program_digest'] = chosen.digest
    out['program_note'] = ('full program dict omitted to keep the artifact small; '
                           'the search-selected equivalent is in dyck_p22_c95-110.json')
    json.dump(out, open(os.path.join(HERE, 'dyck_witness.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
