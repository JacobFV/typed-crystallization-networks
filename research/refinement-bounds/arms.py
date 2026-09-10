"""The four arms of PREREGISTRATION section 2.1, built from one place.

    A0   `IDX` (no refinement bound), clamp as `min(pos + d, last)`   -- section 59's artifact
    A1   `IDX`,  clamp as `pos + min(d, last - pos)`                  -- isolates the rewrite
    B    `ADDR = bounds(0, 3071)`, reformulated clamp                 -- isolates the bound
    N    A0 again, compiled a second time                             -- the null control

Every arm freezes the **same selections** found by `research/visual-ladder`'s
exhaustive sweeps (`out/rung3.json`); nothing is searched or re-selected here.

`A0` is built through the untouched default path so that it is the artifact
`research/emitter-guards/measure.py` timed, byte for byte.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "compiled-runtime")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ARMS = ("A0", "A1", "B", "N")
SEEDS = tuple(range(200, 224))


def _track():
    import fixtures
    fixtures._use("visual-ladder")
    return fixtures


def harden_all(program, choices):
    """Freeze every node: the searched ones at `choices`, the rest at their only
    candidate.  Equivalent to `harden(chosen)` for a scaffold whose non-choice
    nodes have exactly one candidate, and robust to a node being renamed."""
    sel = {n.name: 0 for n in program.nodes}
    sel.update(choices)
    return program.harden(sel)


def build(arm, seeds=(200, 201, 202)):
    """The frozen `visual` program for one arm, with its cases and reference."""
    if arm not in ARMS:
        raise ValueError("unknown arm: %s" % arm)
    fixtures = _track()
    if arm in ("A0", "N"):
        f = fixtures.visual(seeds=tuple(seeds))
        f["arm"] = arm
        return f

    from common import FLAT, IDX, Registry, address_type, bytes_type, episode, load
    import rung3_widgets as R

    found = load("rung3")
    registry = Registry()
    probe = episode(0, "train", **FLAT)
    W, H = probe["width"], probe["height"]
    A = address_type(W, H) if arm == "B" else IDX
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H, addr=A)
    m0 = registry.register_module(harden_all(p0, found["s0"]["chosen"]))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets, addr=A)
    m1 = registry.register_module(harden_all(p1, {k: v for k, v in found["s1"]["chosen"].items()
                                                  if k in ("back_a", "back_b", "corner")}))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets, addr=A, reformulated_clamp=True)
    m2 = registry.register_module(harden_all(p2, {k: v for k, v in found["s2"]["chosen"].items()
                                                  if k in ("step_w", "step_h")}))

    eps = [episode(s, "test", **FLAT) for s in seeds]
    ep = eps[0]
    BT = bytes_type(ep["width"], ep["height"])
    program = R.assembly(registry, BT, R.interior_positions(ep), m1, m2, addr=A)
    cases = [{"observation": tuple(e["pixels"]), "_w": e["width"], "_h": e["height"]}
             for e in eps]
    base = fixtures.visual(seeds=tuple(seeds))
    return {"name": "visual", "arm": arm, "program": program, "registry": registry,
            "cases": cases, "reference": base["reference"], "tolerance": 0.0,
            "what": base["what"] + "  [arm %s]" % arm}


def compiled(arm, seeds=(200, 201, 202), inline_bounded=None):
    """`(fixture, CompileResult, module)` for one arm."""
    from tcn.compile import compile_program
    f = build(arm, seeds)
    kw = {} if inline_bounded is None else {"inline_bounded": inline_bounded}
    res = compile_program(f["program"], f["registry"], **kw)
    return f, res, res.module("arm_%s" % arm)


def source_sha(res):
    return hashlib.sha256(res.source.encode()).hexdigest()
