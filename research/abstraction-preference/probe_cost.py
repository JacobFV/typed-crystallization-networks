"""How much does the description-cost term itself cost, and what does it read?

Two things the measurement plan depends on: the per-step overhead of the term on
the scaffolds the arms use, and the values it puts on the two routes -- which is
the direct answer to whether the objective can now see the size difference that
`complexity()` could not.
"""
import json, sys, time
import torch

sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.operators import Registry
from tcn.learning import SoftProgram, tensor
from tcn.graph import Program, Node, Candidate
from tcn.types import BOOL

out = {}
r = Registry()
body = common.minimal_module(r, 'maj')
name = r.register_module(body)
out['module'] = {'digest': name, 'nodes': len(r.modules[name].nodes),
                 'definition_bits': r.modules[name].description_bits(),
                 'execution_cost': r.modules[name].execution_cost(r)}

ex = common.composite_examples()

# ---- what the two routes measure, on the wide scaffold both routes fit -------
wide = common.wide_scaffold(r, name)
FLAT = (('and', 'w0', 'a', 'b'), ('or', 'w1', 'a', 'b'), ('or', 'w2', 'c', 'w0'),
        ('and', 'w3', 'w1', 'w2'), ('and', 'w4', 'd', 'e'), ('or', 'w5', 'd', 'e'),
        ('or', 'w6', 'f', 'w4'), ('and', 'w7', 'w5', 'w6'), ('xor', 'y', 'w3', 'w7'))


def pick(program, wanted):
    """Selection indices realising a named program inside the scaffold."""
    sel = {}
    for n in program.nodes:
        want = next((w for w in wanted if w[1] == n.name), None)
        for i, c in enumerate(n.candidates):
            if want is not None and c.operator.name == want[0] and c.sources == tuple(want[2:]):
                sel[n.name] = i; break
        else:
            sel[n.name] = 0            # the rest are dead scaffold, whatever they hold
    return sel


MODULE = ((name, 'w0', 'a', 'b', 'c'), (name, 'w1', 'd', 'e', 'f'), ('xor', 'y', 'w0', 'w1'))
REDUNDANT = ((name, 'w0', 'a', 'b', 'c'), (name, 'w1', 'd', 'e', 'f'), (name, 'w4', 'e', 'd', 'f'),
             ('or', 'w5', 'w1', 'w4'), ('xor', 'y', 'w0', 'w5'))

rows = {}
for label, wanted in (('9-gate flat', FLAT), ('2 calls + xor', MODULE), ('3 calls + or + xor', REDUNDANT)):
    sel = pick(wide, wanted)
    hard = wide.harden(sel)
    m = SoftProgram(wide, r)
    with torch.no_grad():
        for n, p in zip(wide.nodes, m.choices): p[sel[n.name]] += 1000.
    pruned = hard.pruned()
    rows[label] = {
        'conformant': common.conformant(hard, ex, common.COMP_SIGNALS, r),
        'complexity': round(float(m.complexity().detach()), 2),
        'description_cost': round(float(m.description_cost().detach()), 1),
        'pruned_description_bits': pruned.description_bits(r),
        'unpruned_description_bits': hard.description_bits(r),
        'pruned_execution_cost': pruned.execution_cost(r),
        'live_nodes': len(pruned.nodes),
    }
out['wide_routes'] = rows

# ---- per-step overhead of the term -----------------------------------------
timing = {}
for label, scaffold in (('wide arm A', common.wide_scaffold(r)), ('wide arm B', common.wide_scaffold(r, name)),
                        ('tight arm A', common.tight_scaffold(r)), ('tight arm B', common.tight_scaffold(r, name))):
    m = SoftProgram(scaffold, r)
    inputs = {k: torch.stack([tensor(e['inputs'][k]) for e in ex]) for k, _ in scaffold.inputs}
    targets = {s.target: torch.stack([tensor(e['targets'][s.target]) for e in ex]) for s in common.COMP_SIGNALS}
    torch.set_num_threads(1)
    t0 = time.perf_counter(); m.description_cost(); build = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(20): m.description_cost()
    term = (time.perf_counter() - t0) / 20

    def step():
        _, _, trace = m(inputs, return_trace=True)
        loss = m.probe_loss(trace, targets, common.COMP_SIGNALS)
        loss.backward()
    t0 = time.perf_counter(); step(); base = time.perf_counter() - t0
    timing[label] = {'table_build_seconds': round(build, 3), 'term_seconds': round(term, 5),
                     'task_step_seconds': round(base, 3), 'overhead_fraction': round(term / base, 4)}
out['per_step'] = timing

json.dump(out, open('probe_cost.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
