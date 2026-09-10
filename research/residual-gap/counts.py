"""Load-independent counters and the profile diff.

PREREGISTRATION section 6, candidates 1, 3, 4 and 5, and section 7's rule that
primitive operations, executed bytecodes and wall clock are three different
measures that must be reported separately.

`research/lazy-guard/cost.py` supplies the `sys.monitoring` bytecode meter and
its `count_opcodes` total; this file extends the same INSTRUCTION event with a
bucket key so the total can be split **per code object** and **per opcode**,
which is the profile diff the brief asks for.  cProfile is not used: the
functions under test are 3-17 us and the profiler's per-call overhead is larger
than the effects being attributed.
"""
from __future__ import annotations

import dis
import json
import pathlib
import statistics
import sys
import tracemalloc

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "lazy-latency"),
           str(ROOT / "research" / "lazy-guard"),
           str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import arms                                                  # noqa: E402
import cost as lazy_cost                                     # noqa: E402
import gate as GATE                                          # noqa: E402
import rungs as RUNGS                                        # noqa: E402

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

_M = sys.monitoring
_TOOL = 3
_OPMAP: dict = {}


def _opname(code, offset):
    key = id(code)
    m = _OPMAP.get(key)
    if m is None:
        m = {i.offset: i.opname for i in dis.get_instructions(code, adaptive=False)}
        _OPMAP[key] = m
    return m.get(offset, "?")


def profile_bytecodes(fn):
    """Executed bytecodes bucketed by (code object, opcode).

    Returns {"total", "by_code": {name: n}, "by_op": {opname: n},
             "by_code_op": {"name/OP": n}, "py_calls": n}.
    """
    by_code, by_op, by_code_op = {}, {}, {}
    calls = 0

    def on_instr(code, offset):
        nm = code.co_qualname
        by_code[nm] = by_code.get(nm, 0) + 1
        op = _opname(code, offset)
        by_op[op] = by_op.get(op, 0) + 1
        k = nm + "/" + op
        by_code_op[k] = by_code_op.get(k, 0) + 1

    def on_start(code, offset):
        nonlocal calls
        calls += 1

    _M.use_tool_id(_TOOL, "residualgap")
    try:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, on_instr)
        _M.register_callback(_TOOL, _M.events.PY_START, on_start)
        _M.set_events(_TOOL, _M.events.INSTRUCTION | _M.events.PY_START)
        fn()
        _M.set_events(_TOOL, 0)
    finally:
        _M.register_callback(_TOOL, _M.events.INSTRUCTION, None)
        _M.register_callback(_TOOL, _M.events.PY_START, None)
        _M.free_tool_id(_TOOL)
    total = sum(by_code.values())
    return {"total": total, "by_code": by_code, "by_op": by_op,
            "by_code_op": by_code_op, "py_calls": calls}


def _agg(dicts):
    out = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return out


def sweep_profile(call, recs):
    """The profile of one full sweep of `recs`, per record."""
    rows = [profile_bytecodes(lambda r=r: call(r)) for r in recs]
    n = len(rows)
    return {"records": n,
            "bytecodes_per_record": sum(r["total"] for r in rows) / n,
            "py_calls_per_record": sum(r["py_calls"] for r in rows) / n,
            "by_code_per_record": {k: v / n for k, v in
                                   sorted(_agg(r["by_code"] for r in rows).items(),
                                          key=lambda kv: -kv[1])},
            "by_op_per_record": {k: v / n for k, v in
                                 sorted(_agg(r["by_op"] for r in rows).items(),
                                        key=lambda kv: -kv[1])},
            "worst_bytecodes": max(r["total"] for r in rows),
            "best_bytecodes": min(r["total"] for r in rows)}


# ---------------------------------------------------------------------------
# candidate 1 -- allocation
# ---------------------------------------------------------------------------
ALLOC_OPS = ("BUILD_TUPLE", "BUILD_LIST", "BUILD_SET", "BUILD_MAP",
             "BUILD_STRING", "LIST_APPEND", "SET_ADD")


def peak_alloc_per_record(call, recs):
    """Peak traced allocation over one record, in bytes, median over `recs`."""
    peaks = []
    for r in recs:
        tracemalloc.start()
        call(r)
        _, pk = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(pk)
    return {"median_bytes": statistics.median(peaks),
            "max_bytes": max(peaks), "records": len(peaks)}


# ---------------------------------------------------------------------------
# candidate 3 -- executed loop iterations, and primitives per output element
# ---------------------------------------------------------------------------
def loop_iterations(g, recs):
    """`w + h`, the number of scan steps the algorithm must take.

    Arm B breaks its two `for` loops at `i = w` and `i = h`; arm C's two
    `while` loops evaluate their condition `w` and `h` times.  If the two agree
    on every record, both are run scans and the residual cannot be a difference
    of algorithm.
    """
    return [arms.stage3_iterations(r) for r in recs]


def subscript_counts(prof):
    """Executed subscript-like opcodes -- the elementary reads of the raster."""
    ops = prof["by_op_per_record"]
    return sum(v for k, v in ops.items()
               if k.startswith("BINARY_SUBSCR") or k in ("BINARY_OP_SUBSCR_TUPLE_INT",))


def main():
    grep, g, R = GATE.check()
    okey = g["program"].outputs[0][0]
    deploy, uniform = g["deploy"], g["uniform"]
    worst = [g["worst"]]

    entries = {}
    for name in RUNGS.LADDER:
        entries["bare:" + name] = R[name]["module"].run
    entries["env:A"] = g["arms"]["A"]["call"]
    entries["env:B"] = g["arms"]["B"]["call"]
    entries["env:C"] = g["arms"]["C"]["call"]

    rep = {"protocol": "research/residual-gap/PREREGISTRATION.md section 6",
           "bytecode_meter": "sys.monitoring INSTRUCTION + PY_START, extending "
                             "research/lazy-guard/cost.py",
           "gate_passed": grep["ladder_gate_passed"],
           "profiles": {}, "alloc": {}, "totals_via_cost_py": {}}

    for dname, recs in (("deploy", deploy), ("worst", worst)):
        rep["profiles"][dname] = {}
        for k, call in entries.items():
            if k == "env:A" and dname == "worst":
                pass
            print("   profiling %s on %s" % (k, dname), flush=True)
            rep["profiles"][dname][k] = sweep_profile(call, recs)

    # cross-check the totals against research/lazy-guard/cost.py's own meter
    for k, call in entries.items():
        rep["totals_via_cost_py"][k] = statistics.mean(
            lazy_cost.count_opcodes(lambda r=r: call(r)) for r in deploy)

    for k, call in entries.items():
        rep["alloc"][k] = peak_alloc_per_record(call, deploy)

    it_d = loop_iterations(g, deploy)
    it_u = loop_iterations(g, uniform)
    rep["loop_iterations"] = {
        "deploy": {"mean": statistics.mean(it_d), "min": min(it_d),
                   "max": max(it_d), "n": len(it_d)},
        "uniform": {"mean": statistics.mean(it_u), "min": min(it_u),
                    "max": max(it_u), "n": len(it_u)},
        "worst": arms.stage3_iterations(g["worst"]),
        "note": "w + h; identical for every rung by construction of the gate -- "
                "each rung returns the same w and h on every record"}

    (OUT / "counts.json").write_text(json.dumps(rep, indent=1, default=str))
    print("->", OUT / "counts.json")

    print("\ndeployment, bytecodes per record / python calls per record")
    for k in entries:
        p = rep["profiles"]["deploy"][k]
        print("   %-10s %10.1f  %8.2f calls  peak_alloc %6.0f B"
              % (k, p["bytecodes_per_record"], p["py_calls_per_record"],
                 rep["alloc"][k]["median_bytes"]))
    return rep


if __name__ == "__main__":
    main()
