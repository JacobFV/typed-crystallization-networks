"""Cost-ranked enumeration: what the discrete reference answers now.

Two questions. On the two-candidate scaffold where a module call and the
primitive `and` compute the same function, does ranking still depend on
declaration order? And on the tight arm-B scaffold -- 2 709 504 programs, 144 of
them conforming -- does the cheapest conforming program differ from the first?
"""
import json, sys, time
sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size, program_cost
from tcn.types import BOOL, Value

out = {}

# ---- declaration order versus cost -----------------------------------------
r = Registry()
body = Program((('a', BOOL), ('b', BOOL)),
               (Node('g', BOOL, (Candidate(r.resolve('and', (BOOL, BOOL)), ('a', 'b')),), 'core', 1, 0),),
               (('out', 'g'),)).validate(r)
name = r.register_module(body)
mop = r.resolve(name, (BOOL, BOOL)); prim = r.resolve('and', (BOOL, BOOL))
ex = [{'inputs': {'a': Value.of(BOOL, a), 'b': Value.of(BOOL, b)}, 'targets': {'out': Value.of(BOOL, a and b)}}
      for a in (False, True) for b in (False, True)]
sig = (Signal('y', 'out', ('core',), BOOL, 'bce'),)
rows = {}
for label, order in (('primitive declared first', (Candidate(prim, ('a', 'b')), Candidate(mop, ('a', 'b')))),
                     ('module declared first', (Candidate(mop, ('a', 'b')), Candidate(prim, ('a', 'b'))))):
    p = Program((('a', BOOL), ('b', BOOL)), (Node('y', BOOL, order, 'core', 1),), (('out', 'y'),)).validate(r)
    entry = {}
    for rank in ('order', 'description', 'cost'):
        res = enumerate_fit(p, ex, sig, registry=r, rank=rank)
        entry[rank] = {'chosen': p.nodes[0].candidates[res.selections['y']].operator.name[:20],
                       'description_bits': res.description_bits, 'execution_cost': res.execution_cost,
                       'conforming': res.conforming}
    rows[label] = entry
out['equivalent_candidates'] = rows

# ---- the tight arm-B scaffold ----------------------------------------------
r2 = Registry()
module = r2.register_module(common.minimal_module(r2, 'maj'))
tight = common.tight_scaffold(r2, module)
examples = common.composite_examples()
out['tight_arm_B'] = {'space_size': space_size(tight)}


def render(program, selections):
    p = program.harden(selections).pruned()
    return [f"{n.name} = {n.candidates[n.selected].operator.name[:14]}({', '.join(n.candidates[n.selected].sources)})"
            for n in p.nodes]


t0 = time.perf_counter()
first = enumerate_fit(tight, examples, common.COMP_SIGNALS, registry=r2, max_programs=1 << 24, stop_at_first=True)
out['tight_arm_B']['order'] = {'seconds': round(time.perf_counter() - t0, 1), 'solved': first.solved,
                               'evaluated': first.evaluated, 'program': render(tight, first.selections),
                               'description_bits': first.description_bits, 'execution_cost': first.execution_cost}
print('order:', json.dumps(out['tight_arm_B']['order'], indent=1), flush=True)

for rank in ('description', 'cost'):
    t0 = time.perf_counter()
    res = enumerate_fit(tight, examples, common.COMP_SIGNALS, registry=r2, max_programs=1 << 24, rank=rank)
    out['tight_arm_B'][rank] = {'seconds': round(time.perf_counter() - t0, 1), 'solved': res.solved,
                                'exhausted': res.exhausted, 'conforming': res.conforming,
                                'evaluated': res.evaluated, 'program': render(tight, res.selections),
                                'description_bits': res.description_bits, 'execution_cost': res.execution_cost}
    print(rank + ':', json.dumps(out['tight_arm_B'][rank], indent=1), flush=True)

json.dump(out, open('enumerate_rank.json', 'w'), indent=1)
