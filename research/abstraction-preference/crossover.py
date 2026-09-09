"""R3: what pruning changes -- the crossover of FINDINGS section 10, and module size.

FINDINGS section 10 records a description-size crossover against inlining of
"4 call sites -> 2" after F1. That number was measured with a hand-built minimal
body. A *learned* module keeps dead scaffold gates, and before R3 those were
registered and charged transitively at every call site. This measures the
crossover for both, with pruning on and off, on track 5's own construction.

Pruning is disabled by patching `Program.pruned` to the identity, so both columns
run through the same registration path.
"""
import json, sys
from contextlib import contextmanager

sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.types import BOOL


@contextmanager
def pruning(enabled):
    original = Program.pruned
    if not enabled: Program.pruned = lambda self: self
    try: yield
    finally: Program.pruned = original


def maj_body(r, dead_gates):
    """The minimal MAJ3 circuit, optionally with the dead gate a learned one keeps."""
    spec = list(common.MINIMAL_BODIES['maj'])
    nodes, depth = [], {'a': 0, 'b': 0, 'c': 0}
    for name, out, x, y in spec:
        d = max(depth[x], depth[y]) + 1; depth[out] = d
        nodes.append(Node(out, BOOL, (Candidate(r.resolve(name, (BOOL, BOOL)), (x, y)),), 'core', d, 0))
    for k in range(dead_gates):
        nodes.append(Node(f'dead{k}', BOOL, (Candidate(r.resolve('xor', (BOOL, BOOL)), ('a', 'b')),), 'core', 1, 0))
    return Program((('a', BOOL), ('b', BOOL), ('c', BOOL)), tuple(nodes), (('out', 'g3'),)).validate(r)


def module_form(r, name, n):
    op = r.resolve(name, (BOOL, BOOL, BOOL))
    nodes = [Node(f'call{i}', BOOL, (Candidate(op, ('a', 'b', 'c')),), depth=1, selected=0) for i in range(n)]
    return Program((('a', BOOL), ('b', BOOL), ('c', BOOL)), tuple(nodes),
                   tuple((f'o{i}', f'call{i}') for i in range(n))).validate(r)


def flat_form(r, body, n):
    """N inlined copies of whatever the module body actually is."""
    nodes = []
    for i in range(n):
        for node in body.nodes:
            c = node.candidates[node.selected or 0]
            src = tuple(s if s in ('a', 'b', 'c') else f'{s}_{i}' for s in c.sources)
            nodes.append(Node(f'{node.name}_{i}', node.output, (Candidate(c.operator, src),),
                              'core', node.depth, 0))
    return Program((('a', BOOL), ('b', BOOL), ('c', BOOL)), tuple(nodes),
                   tuple((f'o{i}', f'g3_{i}') for i in range(n))).validate(r)


def sweep(dead_gates, prune):
    r = Registry()
    with pruning(prune):
        body = maj_body(r, dead_gates)
        name = r.register_module(body)
    kept = r.modules[name]
    rows = []
    for n in range(1, 13):
        mf, ff = module_form(r, name, n), flat_form(r, kept, n)
        rows.append(dict(call_sites=n, module_bits=mf.description_bits(r), flat_bits=ff.description_bits(r),
                         module_cost=mf.execution_cost(r), flat_cost=ff.execution_cost(r)))
    return dict(dead_gates_registered=dead_gates, pruning=prune,
                module_nodes=len(kept.nodes), definition_bits=kept.description_bits(),
                execution_cost=kept.execution_cost(r),
                ratio_at_8=round(rows[7]['module_bits'] / rows[7]['flat_bits'], 3),
                description_bits_crossover=next((x['call_sites'] for x in rows if x['module_bits'] < x['flat_bits']), None),
                rows=rows)


out = {'minimal body, pruning off': sweep(0, False),
       'minimal body, pruning on': sweep(0, True),
       'learned body with 1 dead gate, pruning off': sweep(1, False),
       'learned body with 1 dead gate, pruning on': sweep(1, True)}
for k, v in out.items():
    print(f"{k:<44} nodes={v['module_nodes']} defbits={v['definition_bits']:>6} cost={v['execution_cost']} "
          f"crossover={v['description_bits_crossover']} ratio@8={v['ratio_at_8']}")
json.dump(out, open('crossover.json', 'w'), indent=1)
