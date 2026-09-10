"""PREREGISTRATION §1: which parameter the parse's *types* actually depend on.

The claim under test is a code reading, so it is measured: build S2' at several
(resolution, span) pairs and print the port types and the node count.  If `span`
appears in no type, only the node count moves with it -- and then `span` is a
schema parameter, not a width.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rung3_root as RR                     # noqa: E402
import rung3_widgets as R                   # noqa: E402
import schema as S                          # noqa: E402
from tcn.operators import Registry          # noqa: E402

S0_VECTOR = {"rg": 7, "same": 2}            # §33's S0 selection, reused frozen


def build(width, span):
    registry = Registry()
    p0 = S.RIGHT.build_s0(registry, width, width)
    module = S.module_of(p0, S.full(p0, S0_VECTOR), registry)
    p1 = RR.corner_scaffold_masked(registry, width, width, module, R.offset_pool(width))
    p2 = RR.rect_scaffold_clamped(registry, width, width, module, R.offset_pool(width),
                                  span=span)
    return registry, p0, p1, p2


def components(t):
    """Total leaf count of a (possibly nested) product type."""
    return sum(components(i) for i in t.items) if t.items else 1


def describe(program):
    return {"nodes": len(program.nodes),
            "input_leaf_components": [(name, components(t)) for name, t in program.inputs],
            "input_top_level_items": [(name, len(t.items)) for name, t in program.inputs],
            "free_nodes": S.free_nodes(program)}


def main():
    rows = []
    for width, span in ((16, 16), (24, 24), (32, 30), (32, 32), (32, 48), (40, 40), (48, 48)):
        registry, p0, p1, p2 = build(width, span)
        row = {"resolution": width, "span": span,
               "s0": describe(p0), "s1": describe(p1), "s2": describe(p2),
               "s2_digest": S.freeze(p2, S.full(p2, {"step_w": 2, "step_h": 3}),
                                     registry).digest}
        rows.append(row)
        print(f"res {width:2d} span {span:2d}: s2 nodes {row['s2']['nodes']:4d}  "
              f"s2 input leaves {row['s2']['input_leaf_components']}")
    (HERE / "out" / "types_probe.json").write_text(json.dumps({"rows": rows}, indent=1))


if __name__ == "__main__":
    main()
