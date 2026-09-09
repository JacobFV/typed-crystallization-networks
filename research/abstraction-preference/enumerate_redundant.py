"""Where ranking changes the answer: a space holding both routes to the target.

On the tight arm-B scaffold all 144 conforming programs are argument-symmetry
variants of one another, so ranking cannot change what enumeration returns. This
scaffold is built so that it can: it admits both the intended `MAJ3 ⊕ MAJ3` and
the redundant three-call program the gradient search actually accepted in the
re-test (wide, seed 1), and the redundant one comes **first** in enumeration
order. 1 024 programs, exhaustible in a second.

The candidate lists are hand-built rather than enumerated from a pool only
because the pool version of this scaffold has 2.7·10¹¹ programs. Every candidate
still comes from `registry.resolve`, and both programs are conformant on all 64
rows.
"""
import itertools, json, sys
sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size
from tcn.types import BOOL

r = Registry()
name = r.register_module(common.minimal_module(r, 'maj'))
M = r.resolve(name, (BOOL, BOOL, BOOL))
def g(op, *s): return Candidate(r.resolve(op, tuple(BOOL for _ in s)), s)

nodes = (
    Node('n1', BOOL, (Candidate(M, ('a', 'b', 'c')), Candidate(M, ('b', 'c', 'a')), g('and', 'a', 'b'), g('xor', 'a', 'b')), 'core', 1),
    Node('n2', BOOL, (Candidate(M, ('d', 'e', 'f')), Candidate(M, ('e', 'd', 'f')), g('or', 'd', 'e'), g('xor', 'd', 'e')), 'core', 1),
    Node('n3', BOOL, (Candidate(M, ('d', 'e', 'f')), Candidate(M, ('f', 'e', 'd')), g('and', 'd', 'e'), g('xor', 'd', 'f')), 'core', 1),
    Node('m', BOOL, (g('or', 'n2', 'n3'), g('and', 'n2', 'n3'), g('xor', 'n2', 'n3'), g('identity', 'n2')), 'core', 2),
    Node('y', BOOL, (g('xor', 'n1', 'm'), g('xor', 'n1', 'n2'), g('xor', 'n1', 'n3'), g('and', 'n1', 'm')), 'core', 3),
)
program = Program(tuple((k, BOOL) for k in common.INPUTS), nodes, (('out', 'y'),)).validate(r)
examples = common.composite_examples()


def render(selections):
    p = program.harden(selections).pruned()
    return [f"{n.name} = {n.candidates[n.selected].operator.name[:14]}({', '.join(n.candidates[n.selected].sources)})"
            for n in p.nodes]


out = {'space_size': space_size(program)}
for rank in ('order', 'description', 'cost'):
    res = enumerate_fit(program, examples, common.COMP_SIGNALS, registry=r, rank=rank)
    out[rank] = {'conforming': res.conforming, 'exhausted': res.exhausted, 'seconds': round(res.seconds, 1),
                 'program': render(res.selections), 'live_nodes': len(render(res.selections)),
                 'description_bits': res.description_bits, 'execution_cost': res.execution_cost}
    print(rank, json.dumps(out[rank], indent=1), flush=True)
json.dump(out, open('enumerate_redundant.json', 'w'), indent=1)
