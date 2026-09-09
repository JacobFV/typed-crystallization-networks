"""Batch-one exact latency and cost of the flat vs abstracted programs.

ARCHITECTURE.md sec. 4 says a module's latency still counts toward complexity.
`tcn.runtime.benchmark` reports batch-one p50/p95 for the exact (non-torch)
execution path, plus the repo's own description-bit and operator-cost figures.

NOTE: wall-clock numbers here were taken on a heavily loaded machine; treat the
ratio between arms as the signal, not the absolute milliseconds.
"""
from __future__ import annotations

import json
import sys

from tcn.runtime import benchmark

import experiment as E1
import experiment2 as E2
import solvability as S


def main():
    S.e1_solutions()
    S.e2_solutions()
    e1_inputs = [e["inputs"] for e in E1.composite_examples()]
    e2_inputs = [e["inputs"] for e in E2.composite_examples()]
    out = {}
    for key, rows in (("e1_armA_flat", e1_inputs), ("e1_armB_module", e1_inputs),
                      ("e2_armA_flat", e2_inputs), ("e2_armB_module", e2_inputs)):
        if key not in S.PROGRAMS:
            continue
        prog, reg = S.PROGRAMS[key]
        out[key] = benchmark(prog, rows, reg, repetitions=200)
    for pair in (("e1_armA_flat", "e1_armB_module"), ("e2_armA_flat", "e2_armB_module")):
        if pair[0] in out and pair[1] in out:
            out[f"{pair[1]}_vs_flat"] = dict(
                p50_ratio=out[pair[1]]["p50_ms"] / out[pair[0]]["p50_ms"],
                bits_ratio=out[pair[1]]["description_bits"] / out[pair[0]]["description_bits"],
                cost_ratio=out[pair[1]]["estimated_operator_cost"] / out[pair[0]]["estimated_operator_cost"],
            )
    print(json.dumps(out, indent=2))
    with open(sys.argv[1] if len(sys.argv) > 1 else "latency.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
