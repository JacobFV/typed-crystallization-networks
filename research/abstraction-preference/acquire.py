"""R3 on learned modules: what registration charges before and after pruning.

The retest measured that the learned MAJ3 keeps a dead gate in 5 of 8 seeds and
that registering the raw export charges those gates transitively at every call
site forever. Registration now prunes, so this re-measures the same eight seeds
against the shipped path: the raw export goes straight to `register_module`, with
no experiment-side pruning at all.
"""
import json, sys
from contextlib import contextmanager

sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.graph import Program
from tcn.operators import Registry


@contextmanager
def pruning(enabled):
    original = Program.pruned
    if not enabled: Program.pruned = lambda self: self
    try: yield
    finally: Program.pruned = original


rows = []
for seed in range(8):
    r = Registry()
    prog, signals = common.sub_scaffold(r)
    ex = common.sub_examples(common.maj)
    model, opt, rep, exported = common.search(prog, ex, signals, r, seed * 131, steps=800, eval_every=5)
    if not rep['final_conformant']:
        rows.append({'seed': seed, 'acquired': False}); continue
    common.crystallize(model, opt, ex, signals, r, rounds=6, retrain_steps=5)
    raw = model.export()
    if not common.conformant(raw, ex, signals, r):
        rows.append({'seed': seed, 'acquired': False}); continue
    with pruning(False):
        r_off = Registry(); name_off = r_off.register_module(raw)
    r_on = Registry(); name_on = r_on.register_module(raw)
    same = common.verify_module(r_on.modules[name_on], common.maj, r_on)
    rows.append({'seed': seed, 'acquired': True, 'steps': rep['steps_run'],
                 'scaffold_nodes': len(prog.nodes),
                 'registered_nodes_unpruned': len(r_off.modules[name_off].nodes),
                 'registered_nodes_pruned': len(r_on.modules[name_on].nodes),
                 'call_cost_unpruned': r_off.resolve(name_off, tuple(t for _, t in raw.inputs)).cost,
                 'call_cost_pruned': r_on.resolve(name_on, tuple(t for _, t in raw.inputs)).cost,
                 'definition_bits_unpruned': r_off.modules[name_off].description_bits(),
                 'definition_bits_pruned': r_on.modules[name_on].description_bits(),
                 'pruned_module_still_exact': same})
    print(rows[-1], flush=True)

ok = [x for x in rows if x.get('acquired')]
summary = {
    'acquired': len(ok), 'of': len(rows),
    'seeds_with_dead_gates': sum(1 for x in ok if x['registered_nodes_pruned'] < x['registered_nodes_unpruned']),
    'median_nodes_unpruned': sorted(x['registered_nodes_unpruned'] for x in ok)[len(ok) // 2] if ok else None,
    'median_nodes_pruned': sorted(x['registered_nodes_pruned'] for x in ok)[len(ok) // 2] if ok else None,
    'all_pruned_modules_exact': all(x['pruned_module_still_exact'] for x in ok),
}
print(summary)
json.dump({'rows': rows, 'summary': summary}, open('acquire.json', 'w'), indent=1)
