"""Where the remaining arm-C-versus-arm-D gap goes.

Once the typed value layer is out of the way, the only honest way to ask whether
the residue is *representation* or *work* is to count the work.  CPython can be
asked exactly: `sys.settrace` with `f_trace_opcodes` counts every bytecode a call
executes.  That is a load-independent unit finer than an operator application and
it does not depend on this host.

If arm C executes k times as many bytecodes as arm D and is k times slower, the
gap is the program doing more work -- an algorithmic property of what the search
found -- and no change of representation or of target language removes it.  If
arm C executes the same bytecodes and is still slower, the residue is per-value
overhead and native compilation is the lever.

Instruction monitoring costs about a microsecond per opcode, so arms A and B are counted
only where their bytecode count is small enough to trace; for the parse their
element-operation counts from `bench.py` are the comparable figure.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fixtures
from harness import cached_interpreter, dump, timed
from tcn.compile import compile_program
from tcn.types import Value

TRACE_A = {'mixed', 'language'}          # the parse is ~5e8 opcodes; not traceable


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


def main(name):
    fx = fixtures.FIXTURES[name]()
    p, r = fx['program'], fx['registry']
    types = dict(p.inputs)
    c0 = fx['cases'][0]
    ins = {k: Value.of(types[k], c0[k]) for k in types}
    native = {k: c0[k] for k in types}
    res = compile_program(p, r)
    mod = res.module()

    reps = {'mixed': 201, 'language': 21, 'visual': 3}[name]
    out = {'artifact': name, 'opcodes': {}, 'median_ms': {}, 'ns_per_opcode': {}}

    def record(arm, fn, trace=True):
        t, _ = timed(fn, reps)
        out['median_ms'][arm] = t['median_ms']
        if trace:
            n = count_opcodes(fn)
            out['opcodes'][arm] = n
            out['ns_per_opcode'][arm] = t['median_ms'] * 1e6 / max(1, n)

    record('A', lambda: p.run(ins, registry=r), name in TRACE_A)
    with cached_interpreter():
        ins_b = {k: Value.of(types[k], c0[k]) for k in types}
        record('B', lambda: p.run(ins_b, registry=r), name in TRACE_A)
    record('C', lambda: mod.run(native, validate=False))
    record('D', lambda: fx['reference'](c0))

    out['ratios'] = {}
    for a in ('A', 'B', 'C'):
        if a in out['opcodes']:
            out['ratios'][f'{a}_over_D_opcodes'] = out['opcodes'][a] / out['opcodes']['D']
        out['ratios'][f'{a}_over_D_time'] = out['median_ms'][a] / out['median_ms']['D']
    print(name)
    for k in ('opcodes', 'median_ms', 'ns_per_opcode', 'ratios'):
        print(' ', k, out[k])
    dump(f'attribute_{name}', out)


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'visual')
