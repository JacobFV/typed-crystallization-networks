"""Program-length metrics that connect a search preference to executed bytecodes.

Four numbers per program, deliberately kept distinct (ARCHITECTURE section 8.1):

  nodes / operators   the pruned hardened program's size -- what a scaffold fixes
  description_bits    `Program.description_bits`, ARCHITECTURE size 3
  execution_cost      `Program.execution_cost`, ARCHITECTURE cost 1
  bytecodes           CPython instructions the *compiled* program executes on one
                      real input, via `sys.monitoring` -- the only one of the four
                      that is a measurement of work rather than a model of it

`research/compiled-runtime/RESULTS.md` section 7 decomposed the visual artifact's
27.6x residue as 11.1x bytecodes times 2.25x per-bytecode cost.  A cost term that
moves `description_bits` without moving `bytecodes` has not attacked that 11.1x,
and this module exists so that claim can be checked rather than asserted.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = pathlib.Path(__file__).resolve().parent / 'out'

_M = sys.monitoring
_TOOL = 2


def _count(fn):
    n = 0

    def on_instr(code, offset):
        nonlocal n
        n += 1
    _M.use_tool_id(_TOOL, 'opcount')
    try:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, on_instr)
        _M.set_events(_TOOL, _M.events.INSTRUCTION)
        fn()
        _M.set_events(_TOOL, 0)
    finally:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, None)
        _M.free_tool_id(_TOOL)
    return n


_BASE = _count(lambda: None)


def count_opcodes(fn):
    """Bytecodes executed by one call, harness wrapper subtracted."""
    return max(0, _count(fn) - _BASE)


def static(program, registry):
    """Size of the pruned program, in the three units the record already uses."""
    p = program.pruned()
    ops = set()
    for n in p.nodes:
        for c in n.candidates:
            ops.add(c.operator.name)
    return {'nodes': len(p.nodes),
            'operators': len(ops),
            'description_bits': p.description_bits(registry),
            'execution_cost': p.execution_cost(registry)}


def compiled(program, registry, native_cases, validate=False):
    """Compile once, then count bytecodes and record the outputs for assertion.

    `validate=False` prices the program alone; the external boundary is measured
    separately in `boundary.py`, because they are different questions.
    """
    from tcn.compile import compile_program
    res = compile_program(program, registry)
    mod = res.module()
    outs, counts = [], []
    for c in native_cases:
        outs.append(mod.run(c, validate=validate)[0])
        counts.append(count_opcodes(lambda c=c: mod.run(c, validate=validate)))
    return {'bytecodes': counts, 'bytecodes_total': sum(counts),
            'source_bytes': len(res.source), 'stats': dict(res.stats)}, outs, mod


def dump(name, obj):
    OUT.mkdir(exist_ok=True)
    path = OUT / (name + '.json')
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    print('wrote', path)
    return path
