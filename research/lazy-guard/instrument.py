"""Demonstrate the cost instrumentation before any arm is run.

This is the precondition of DESIGN sec 0.3.  It shows, on both miniatures:

  1. `Program.execution_cost` is a single input-independent scalar;
  2. the executed primitive count of the specification is likewise constant
     across the whole declared domain (worst == expected == best), which is what
     makes early exit invisible to the objective;
  3. the new instrument separates worst from expected, and reports executed
     bytecodes alongside, so a program whose expected cost differs from its
     worst case is now measurable.

Point 2 is the demonstration that the instrumentation is *needed*, and point 3
that it *works*: an input-dependent reference (a plain hand-written early-exit
function, used here only as an instrument test, never as a synthesis candidate)
does show a worst/expected split under the same meter.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import cost
import spec
from tcn.compile import compile_program
from tcn.types import Value

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def main():
    report = {}
    for name, (build, dom) in spec.FIXTURES.items():
        fx = build()
        p, r = fx["program"], fx["registry"]
        port, t = fx["port"], fx["input_type"]

        n_cases = len(list(dom(fx)))
        # 1. certify the native evaluator against the typed interpreter, exhaustively
        checked = cost.certify_native(fx, dom(fx))

        # 2. the specification's cost under the old scalar and the new meter
        meter = cost.Meter()
        ev = cost.NativeEval(r, meter)
        prof, _ = cost.profile(lambda c: ev.run(p, {port: c[port]})[0], dom(fx), meter)

        res = compile_program(p, r)
        mod = res.module()

        entry = {
            "nodes": len(p.nodes),
            "modules": len(r.modules),
            "module_leaf_cost": {k: cost.module_leaf_cost(r, k) for k in r.modules},
            "domain_size": n_cases,
            "native_evaluator_certified_on": checked,
            "execution_cost_scalar": p.execution_cost(r),
            "spec_worst_ops": prof["worst_ops"],
            "spec_best_ops": prof["best_ops"],
            "spec_expected_ops": prof["expected_ops"],
            "spec_ops_histogram": prof["histogram"],
            "spec_by_operator_expected": prof["by_operator_expected"],
            "compile_stats": {k: v for k, v in res.stats.items() if isinstance(v, int)},
        }

        # 3. executed bytecodes of the tcn-compiled specification (arm C)
        cases = list(dom(fx))
        probe = [cases[0], cases[len(cases) // 3], cases[-1]]
        entry["spec_compiled_bytecodes"] = [
            cost.count_opcodes(lambda c=c: mod.run({port: c[port]}, validate=False))
            for c in probe]
        report[name] = entry
        print(name, json.dumps(entry, indent=2, default=str)[:1400])

    (OUT / "instrumentation.json").write_text(json.dumps(report, indent=2, default=str))
    print("wrote", OUT / "instrumentation.json")


if __name__ == "__main__":
    main()
