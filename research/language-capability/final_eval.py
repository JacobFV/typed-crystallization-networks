"""The exported program, priced and measured on every split.

Reports the complete inference path -- batch-one latency of the exact exported
program, its description bits and execution cost -- alongside accuracy, per
AGENTS.md's requirement to price the whole path rather than a proxy.
"""
from __future__ import annotations
import sys, os, json, time, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, scaffolds
from run_stage_b import build_module, examples, accuracy, baselines
from tcn.types import Value, BOOL

def main():
    sel = json.load(open(os.path.join(HERE, 'stage_b.json')))['enumeration']['selections']
    module, registry, frozen = build_module()
    program, signals = scaffolds.stage_b(module, registry)
    exported = program.harden(sel).pruned()
    pool = common.dataset(900, seed0=0, split='train')
    train = [e for e in pool if e['length'] in (2, 4, 6)][:24]
    seen = [e for e in pool if e['length'] in (2, 4, 6)][24:144]
    test_pool = common.dataset(1500, seed0=100000, split='test')
    unseen = [e for e in test_pool if e['length'] not in (2, 4, 6)]
    report = {'selection': sel,
              'description_bits': exported.description_bits(registry),
              'execution_cost': exported.execution_cost(registry),
              'live_nodes': len(exported.nodes), 'scaffold_nodes': len(program.nodes)}
    for name, eps in (('train', train), ('heldout_seen_lengths', seen), ('heldout_unseen_lengths', unseen)):
        acc, per_len = accuracy(exported, {n.name: 0 for n in exported.nodes}, registry, eps)
        report[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps),
                        'lengths': sorted({e['length'] for e in eps}),
                        'depths': sorted({e['depth'] for e in eps})}
    # batch-one latency of the complete exact inference path
    t0 = time.perf_counter(); n = 50
    for e in unseen[:n]: exported.run({'text': e['text']}, registry=registry)
    report['batch_one_latency_ms'] = (time.perf_counter() - t0) / n * 1e3
    # what the stage-A module does at positions it was never supervised on
    out_of_string = [0, 0]
    for e in unseen[:200]:
        for p in range(len(e['string']), 16):
            o, _ = frozen.run({'text': e['text'], 'pos': Value.of(scaffolds.POS, p)}, registry=registry)
            out_of_string[bool(o['open'].decoded)] += 1
    report['module_outside_the_string'] = {'reported_close_or_absent': out_of_string[0],
                                           'reported_open': out_of_string[1]}
    # undecomposed search, projected
    flat = 41 * 256 * 121 * 5 * 5 * 15
    report['undecomposed_space'] = {'programs': flat, 'seconds_per_program_measured': 0.0092,
                                    'projected_days': flat * 0.0092 / 86400}
    json.dump(report, open(os.path.join(HERE, 'final_eval.json'), 'w'), indent=1)
    print(json.dumps(report, indent=1))

if __name__ == '__main__':
    main()
