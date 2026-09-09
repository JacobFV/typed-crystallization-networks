"""The retest's arms re-run with the description-cost term on and off.

Seed variation is real here only because `armlib.run` adds N(0, 0.5) noise to the
choice logits before training: `SoftProgram` zero-initializes every logit, so
`torch.manual_seed` alone does not vary synthesis. This is the same
initialization the retest used, so the `mdl_weight = 0` column is comparable to
its numbers.

A run stops at the first conformant checkpoint by default, which is the retest's
criterion and measures where the search *lands*. `nostop` runs the whole budget
instead and reports where it *ends*, which is the only way a preference term
applied after conformance can be seen at all.

usage: experiment.py <scaffold> <arm> <seeds> [nostop] <weight> [<weight> ...]
"""
import json, sys, statistics
import torch

sys.path.insert(0, '../recursive-abstraction-retest')
import common
import armlib
from tcn.operators import Registry


def main():
    scaffold, arm, seeds = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rest = sys.argv[4:]
    stop = True
    if rest and rest[0] == 'nostop': stop, rest = False, rest[1:]
    weights = [float(w) for w in rest]
    r = Registry()
    module = None
    if arm == 'B': module = r.register_module(common.minimal_module(r, 'maj'))
    elif arm == 'C': module = r.register_module(common.minimal_module(r, 'distractor'))
    program = (common.wide_scaffold(r, module) if scaffold == 'wide' else common.tight_scaffold(r, module))
    examples = common.composite_examples()
    out = {'scaffold': scaffold, 'arm': arm, 'seeds': seeds, 'weights': weights,
           'module': module, 'candidates': sum(len(n.candidates) for n in program.nodes), 'runs': []}
    for w in weights:
        for seed in range(seeds):
            rep = armlib.run(program, examples, common.COMP_SIGNALS, r, seed, mdl_weight=w, stop_on_success=stop)
            out['runs'].append(rep)
            f = rep['found']
            print(f"{scaffold} {arm} w={w:g} seed={seed} conformant={rep['conformant']} "
                  f"step={rep['first_conformant_step']} "
                  f"live={f['live_nodes'] if f else '-'} calls={f['module_calls'] if f else '-'} "
                  f"bits={f['description_bits'] if f else '-'} "
                  f"| final live={rep['final']['live_nodes']} calls={rep['final']['module_calls']} "
                  f"bits={rep['final']['description_bits']} {rep['seconds_per_step']:.2f}s/step", flush=True)
    name = f"runs_{scaffold}_{arm}{'' if stop else '_nostop'}.json"
    json.dump(out, open(name, 'w'), indent=1)
    for w in weights:
        got = [x for x in out['runs'] if x['mdl_weight'] == w and x['found']]
        n = len([x for x in out['runs'] if x['mdl_weight'] == w])
        print(f"== w={w:g}: {len(got)}/{n} conformant"
              + (f", median live={statistics.median(x['found']['live_nodes'] for x in got)}"
                 f", median calls={statistics.median(x['found']['module_calls'] for x in got)}"
                 f", median bits={statistics.median(x['found']['description_bits'] for x in got)}"
                 f", median step={statistics.median(x['first_conformant_step'] for x in got)}" if got else ""),
              flush=True)


if __name__ == '__main__':
    main()
