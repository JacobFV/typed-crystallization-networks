"""Typed scaffolds for the grammaticality task. No domain operator is added."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tcn.types import Type, Value, BOOL, integer, product
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.generation import BYTE

CAPACITY = 128
POS = integer(32, signed=False, bounds=(0, CAPACITY))
CNT = integer(16)
BYTES = product(*(BYTE for _ in range(CAPACITY)))
TEXT = product(POS, BYTES)

def cand(r, name, sources, types, output=None, **params):
    return Candidate(r.resolve(name, types, output, params or None), tuple(sources))

def stage_a(base_range=range(0, 41), byte_values=range(0, 256), registry=None):
    """Lexical perception: is the symbol at position `pos` an opening bracket?

    Searched: the byte value that denotes an opening bracket (over the whole
    0-255 alphabet) and the base address of the symbol field. The address is
    *computed* -- base + pos -- not chosen per position, so one module serves
    every position (ARCHITECTURE section 4; the positional-reuse idiom).
    """
    r = registry or Registry()
    consts = [(f'p{k}', Value.of(POS, k)) for k in base_range] + \
             [(f'b{v}', Value.of(BYTE, v)) for v in byte_values]
    nodes = (
        Node('bytes', BYTES, (cand(r, 'project', ('text',), (TEXT,), BYTES, index=1),), depth=1),
        Node('base', POS, tuple(cand(r, 'identity', (f'p{k}',), (POS,)) for k in base_range), depth=1),
        Node('addr', POS, (cand(r, 'add', ('base', 'pos'), (POS, POS)),), depth=2),
        Node('byte', BYTE, (cand(r, 'index', ('bytes', 'addr'), (BYTES, POS)),), depth=3),
        Node('open', BOOL, tuple(cand(r, 'eq', ('byte', f'b{v}'), (BYTE, BYTE)) for v in byte_values), depth=4),
    )
    p = Program(inputs=(('text', TEXT), ('pos', POS)), nodes=nodes,
                outputs=(('open', 'open'),), constants=tuple(consts))
    return p.validate(r), r, [Signal('open', 'open', ('core',), BOOL, loss='bce')]

def stage_b(module_name, registry, positions=16, sub_range=range(0, 121),
            step_values=(-2, -1, 0, 1, 2), rule_consts=(-2, -1, 0, 1, 2)):
    """Grammaticality from the recovered symbol sequence.

    Searched: the constant relating prompt length to symbol count (which is what
    makes the program length-general), the two step values the symbol/step map
    emits, and the rule that reads the accumulated count. The frozen stage-A
    module is called once per position; nothing about the positions is learned.
    """
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
    for i in range(1, positions):
        nodes.append(Node(f'acc{i}', CNT, (cand(r, 'add', (prev, f'm{i}'), (CNT, CNT)),), depth=4 + i))
        prev = f'acc{i}'
    rules = [cand(r, op, (prev, f'v{c}'), (CNT, CNT)) for c in rule_consts for op in ('eq', 'ge', 'le')]
    nodes.append(Node('answer', BOOL, tuple(rules), depth=4 + positions))
    p = Program(inputs=(('text', TEXT),), nodes=tuple(nodes),
                outputs=(('answer', 'answer'),), constants=tuple(consts))
    return p.validate(r), [Signal('answer', 'answer', ('core',), BOOL, loss='bce')]
