"""Where the exact apply time goes at a realistic observation width.

`pair` attaches the whole observation to every position, so the record set holds
N * W values. This measures how much of the cost is that replication rather than
the N module runs that do the work.
"""
from __future__ import annotations

import cProfile
import pathlib
import pstats
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from measure_scaling import bytes_type, scaffold, synthetic
from shared import shared_map_program
from tcn.operators import Registry
from tcn.types import Value

RES = 16

if __name__ == "__main__":
    positions = tuple(3 * i for i in range(RES * RES))
    width = 3 * RES * RES
    r = Registry()
    obs = Value.of(bytes_type(width), synthetic(width, positions, None))
    caller = shared_map_program(r, width, positions, [r.register_module(scaffold(r, width, False))],
                                element=bytes_type(1).items[0])
    caller.run({"observation": obs}, registry=r)
    p = cProfile.Profile()
    p.enable()
    caller.run({"observation": obs}, registry=r)
    p.disable()
    print(f"width={width} positions={len(positions)} record set = {len(positions)*width} values")
    pstats.Stats(p).sort_stats("cumulative").print_stats(14)
