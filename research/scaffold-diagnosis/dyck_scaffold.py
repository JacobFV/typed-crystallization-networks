"""A stage-B scaffold that can express the Dyck reduction, for the bounding question.

This exists to separate two things the headline result must not conflate:

  * the *stage-B scaffold* of `research/language-capability/scaffolds.py` cannot
    express balancedness on the post-audit stream, because its accumulator is a
    plain sum -- a bracket **count** -- and the hardened draw severed counting
    from balancedness (`bound.py`);
  * the *typed algebra* can express it, using operators already in
    `tcn/operators.py` (`min`, `and`) and no new domain operator.

The only structural addition over `scaffolds.stage_b` is a second accumulator:
the running minimum of the prefix sum. Balancedness is exactly
`total == 0 and min_prefix >= 0`, so the answer is a conjunction of two readouts
of the two accumulators. Both readouts are **searched** over the same
`{eq, ge, le} x {-2..2}` grid stage B uses; nothing is wired by hand.

No file under `tcn/` or `generators/` is touched.
"""
from __future__ import annotations
import sys, os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from tcn.types import Type, Value, BOOL, integer, product
from tcn.graph import Program, Node, Candidate, Signal

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
lc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lc)

CAPACITY, POS, CNT, TEXT = lc.CAPACITY, lc.POS, lc.CNT, lc.TEXT
cand = lc.cand


def stage_b_dyck(module_name, registry, positions=22, sub_range=range(0, 121),
                 step_values=(-2, -1, 0, 1, 2), rule_consts=(-2, -1, 0, 1, 2)):
    r = registry
    consts = [(f'i{i}', Value.of(POS, i)) for i in range(positions)] + \
             [(f's{c}', Value.of(POS, c)) for c in sub_range] + \
             [(f'v{v}', Value.of(CNT, v)) for v in sorted(set(step_values) | set(rule_consts) | {0})]
    nodes = [
        Node('length', POS, (cand(r, 'project', ('text',), (TEXT,), POS, index=0),), depth=1),
        Node('symbols', POS, tuple(cand(r, 'sub', ('length', f's{c}'), (POS, POS)) for c in sub_range), depth=2),
        Node('plus', CNT, tuple(cand(r, 'identity', (f'v{v}',), (CNT,)) for v in step_values), depth=1),
        Node('minus', CNT, tuple(cand(r, 'identity', (f'v{v}',), (CNT,)) for v in step_values), depth=1),
    ]
    for i in range(positions):
        nodes.append(Node(f'open{i}', BOOL, (cand(r, module_name, ('text', f'i{i}'), (TEXT, POS)),), depth=1))
        nodes.append(Node(f'in{i}', BOOL, (cand(r, 'lt', (f'i{i}', 'symbols'), (POS, POS)),), depth=3))
        nodes.append(Node(f'step{i}', CNT, (cand(r, 'mux', (f'open{i}', 'plus', 'minus'), (BOOL, CNT, CNT)),), depth=2))
        nodes.append(Node(f'm{i}', CNT, (cand(r, 'mux', (f'in{i}', f'step{i}', 'v0'), (BOOL, CNT, CNT)),), depth=4))
    # prefix sum, and the running minimum of it (the empty prefix, 0, is included)
    prev = 'm0'
    nodes.append(Node('lo0', CNT, (cand(r, 'min', ('m0', 'v0'), (CNT, CNT)),), depth=5))
    prevlo = 'lo0'
    for i in range(1, positions):
        nodes.append(Node(f'acc{i}', CNT, (cand(r, 'add', (prev, f'm{i}'), (CNT, CNT)),), depth=4 + i))
        prev = f'acc{i}'
        nodes.append(Node(f'lo{i}', CNT, (cand(r, 'min', (prevlo, prev), (CNT, CNT)),), depth=5 + i))
        prevlo = f'lo{i}'
    rules_total = [cand(r, op, (prev, f'v{c}'), (CNT, CNT)) for c in rule_consts for op in ('eq', 'ge', 'le')]
    rules_min = [cand(r, op, (prevlo, f'v{c}'), (CNT, CNT)) for c in rule_consts for op in ('eq', 'ge', 'le')]
    nodes.append(Node('total_ok', BOOL, tuple(rules_total), depth=5 + positions))
    nodes.append(Node('min_ok', BOOL, tuple(rules_min), depth=6 + positions))
    nodes.append(Node('answer', BOOL, (cand(r, 'and', ('total_ok', 'min_ok'), (BOOL, BOOL)),),
                      depth=7 + positions))
    p = Program(inputs=(('text', TEXT),), nodes=tuple(nodes),
                outputs=(('answer', 'answer'),), constants=tuple(consts))
    return p.validate(r), [Signal('answer', 'answer', ('core',), BOOL, loss='bce')]
