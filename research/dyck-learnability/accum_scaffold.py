"""The stage-B scaffold with the second accumulator's fold operator left as a hole.

This is `research/language-post-audit/dyck_scaffold.py:stage_b_dyck` with exactly
one string parameterised: the operator name folded along the prefix sums. With
`fold='min'` it builds the identical program section 45 measured -- asserted, not
assumed, by `q3_operators.py`, which compares the two programs' digests.

The point of the parameterisation is Q3c. "Add a second accumulator beside the
running sum" is a scaffold *template* with one hole; the hole ranges over the
operators core already declares with a `CNT x CNT -> CNT` signature. If sweeping
that hole recovers the operator that works, then the scaffold change was
mechanically generable from core's own operator inventory and did not need a
human who already knew the answer. If it does not, scaffold design is a human
input at the operator level too.

No file under `tcn/` or `generators/` is touched, and no new operator is added:
every fold candidate is already in `tcn.operators.BINARY`.
"""
from __future__ import annotations
import sys, os, importlib.util
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
from tcn.types import Value, BOOL
from tcn.graph import Program, Node, Signal

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
lc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lc)

CAPACITY, POS, CNT, TEXT = lc.CAPACITY, lc.POS, lc.CNT, lc.TEXT
cand = lc.cand


def stage_b_accum(module_name, registry, positions=22, sub_range=range(0, 121),
                  step_values=(-2, -1, 0, 1, 2), rule_consts=(-2, -1, 0, 1, 2),
                  fold='min'):
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
    prev = 'm0'
    nodes.append(Node('lo0', CNT, (cand(r, fold, ('m0', 'v0'), (CNT, CNT)),), depth=5))
    prevlo = 'lo0'
    for i in range(1, positions):
        nodes.append(Node(f'acc{i}', CNT, (cand(r, 'add', (prev, f'm{i}'), (CNT, CNT)),), depth=4 + i))
        prev = f'acc{i}'
        nodes.append(Node(f'lo{i}', CNT, (cand(r, fold, (prevlo, prev), (CNT, CNT)),), depth=5 + i))
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


def core_folds(registry):
    """Every operator core already declares with a CNT x CNT -> CNT signature.

    Read off `tcn.operators.BINARY` by asking the registry to resolve each name
    at that signature: no list is hand-curated, and nothing is added to core.
    """
    from tcn.operators import BINARY
    out = []
    for name in sorted(BINARY):
        try:
            registry.resolve(name, (CNT, CNT), CNT, None)
        except Exception:
            continue
        out.append(name)
    return out
