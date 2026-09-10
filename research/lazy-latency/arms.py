"""The three arms, the three distributions, and nothing else.

Arm A is `tcn/compile.py`'s output on the frozen specification.  Arm B is the
committed `research/lazy-guard/out/stage*_algorithm.py`, loaded verbatim -- not
one character of it is edited here.  Arm C exists only for stage 3, and is the
extent loops of `research/compiled-runtime/fixtures.py :: visual().reference`
restricted to a single interior position, because that is the only hand-written
reference the repository contains for any of these three subroutines.

Nothing under `tcn/`, `generators/` or `research/lazy-guard/` is modified.
"""
from __future__ import annotations

import itertools
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LAZY = ROOT / "research" / "lazy-guard"
for _p in (str(ROOT), str(LAZY), str(ROOT / "research" / "visual-ladder"),
           str(LAZY / "out")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

W = H = 32          # the shipped 32x32 parse; asserted against the episode below
SUBSAMPLE = 65536   # AMENDMENT 1, see RESULTS.md: the miniatures' uniform sweep
                    # is the FULL exhausted domain, not the pre-registered
                    # stride-16 subsample.  `full[::16]` fixes the last input
                    # digit at 0, so position 3 can never satisfy the predicate
                    # (measured per-position hit rates 1/16, 1/16, 1/16, 0), and
                    # it therefore under-reports arm B's expected work.  The full
                    # domain removes the estimator entirely and costs ~0.2 s per
                    # sweep, so the concession the subsample bought was not
                    # needed.  This makes B look WORSE, not better.
HIT_SWEEP_SEED = 20260909


# ---------------------------------------------------------------------------
# arm C -- the hand-written reference, for stage 3 only
# ---------------------------------------------------------------------------
def s2_reference(rec):
    """The S2 subroutine written directly, as `fixtures.py` writes the parse.

    `fixtures.py :: visual().reference` computes, for an interior position that
    has already passed the corner test:

        w = 1
        while x + w < W and px[i+3w : i+3w+3] == col: w += 1
        h = 1
        while y + h < H and px[i+3Wh : i+3Wh+3] == col: h += 1

    and packs the pixel's own RGB and its left neighbour's.  That is exactly
    this function; the only change is that it takes one position instead of
    looping over the raster, because the S2 module is a per-position subroutine.
    The `while` loop is the reference's own, character for character.
    """
    pos, px = rec
    t = pos // 3
    x = t % W
    y = t // W
    i = pos
    col = (px[i], px[i + 1], px[i + 2])
    w = 1
    while x + w < W and (px[i + 3 * w], px[i + 3 * w + 1], px[i + 3 * w + 2]) == col:
        w += 1
    h = 1
    while y + h < H and (px[i + 3 * W * h], px[i + 3 * W * h + 1], px[i + 3 * W * h + 2]) == col:
        h += 1
    return (x, y, w, h,
            col[0] | col[1] << 8 | col[2] << 16,
            px[i - 3] | px[i - 2] << 8 | px[i - 1] << 16)


REFERENCE_SOURCE = """\
def run(rec):
    W = H = 32
    pos, px = rec
    t = pos // 3
    x = t % W
    y = t // W
    i = pos
    col = (px[i], px[i + 1], px[i + 2])
    w = 1
    while x + w < W and (px[i + 3 * w], px[i + 3 * w + 1], px[i + 3 * w + 2]) == col:
        w += 1
    h = 1
    while y + h < H and (px[i + 3 * W * h], px[i + 3 * W * h + 1], px[i + 3 * W * h + 2]) == col:
        h += 1
    return (x, y, w, h,
            col[0] | col[1] << 8 | col[2] << 16,
            px[i - 3] | px[i - 2] << 8 | px[i - 1] << 16)
"""


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def compiled_arm(program, registry, name):
    """Arm A: compile the frozen specification and return (module, source)."""
    from tcn.compile import compile_program
    res = compile_program(program, registry)
    src = res.source
    (OUT / ("%s_compiled.py" % name)).write_text(src)
    return res.module("tcn_compiled_%s" % name), src


def resynth_arm(stage):
    """Arm B: the committed lazy-guard algorithm, imported verbatim."""
    import importlib
    mod = importlib.import_module("stage%d_algorithm" % stage)
    src = (LAZY / "out" / ("stage%d_algorithm.py" % stage)).read_text()
    return mod, src


# ---------------------------------------------------------------------------
# stage 3 -- the shipped visual parser's S2 subroutine
# ---------------------------------------------------------------------------
def stage3_fixture():
    import stage3 as S3
    hx = S3.build(span=32, seeds=S3.HELDOUT_SEEDS, split="test")
    assert hx["W"] == W and hx["H"] == H, (hx["W"], hx["H"])
    uniform = [r["rec"] for r in hx["records"]]
    deploy = [r["rec"] for r in S3.corner_records(hx)]
    return {"stage": 3, "name": "stage3", "port": "rec", "fixture": hx,
            "program": hx["program"], "registry": hx["registry"],
            "input_type": hx["input_type"],
            "uniform": uniform, "deploy": deploy,
            "deploy_what": "corner positions -- what the assembly's `filter` passes",
            "uniform_what": "every interior position of held-out seeds 200/201/202"}


def stage3_extent(rec):
    """The achieved extent `max(w, h)` -- the natural way to describe a widget."""
    o = s2_reference(rec)
    return max(o[2], o[3])


def stage3_iterations(rec):
    """`w + h` -- arm B's executed loop iterations, and so its work parameter.

    AMENDMENT 2, see RESULTS.md.  Arm B runs one loop per direction, so its cost
    is linear in `w + h`, not in `max(w, h)`; a record with `max(w, h) = 30` may
    do anything from 31 to 61 iterations.  Binning by `max(w, h)` therefore
    smears arm B's cost within a bin and hides the crossover.  `w + h` is the
    same quantity the pre-registered worst case is chosen by.
    """
    o = s2_reference(rec)
    return o[2] + o[3]


# ---------------------------------------------------------------------------
# stages 1 and 2 -- the miniatures
# ---------------------------------------------------------------------------
def stage1_fixture():
    import spec
    fx = spec.sparse_guard()
    full = [c["x"] for c in spec.sparse_guard_domain(fx)]
    return {"stage": 1, "name": "stage1", "port": "x", "fixture": fx,
            "program": fx["program"], "registry": fx["registry"],
            "input_type": fx["input_type"], "full": full,
            "uniform": full,
            "deploy": hit_rate_inputs(fx, 1.0 / 48.0, len(full)),
            "deploy_what": "hit rate 1/48 -- the parse's own 20-of-961 sparsity",
            "uniform_what": "the exhausted 16^4 domain, all 65,536 inputs",
            "hit": fx["hit"], "n": fx["n"],
            "carrier_values": fx["carrier_values"]}


def hit_rate_inputs(fx, p, count, seed=HIT_SWEEP_SEED):
    """`count` stage-1 inputs whose per-position predicate hit rate is `p`."""
    rng = random.Random(seed + int(round(p * 1e6)))
    hit = fx["hit"]
    miss = [v for v in fx["carrier_values"] if v != hit]
    n = fx["n"]
    return [tuple(hit if rng.random() < p else rng.choice(miss) for _ in range(n))
            for _ in range(count)]


def stage2_fixture():
    import spec
    fx = spec.find_first()
    full = [c["y"] for c in spec.find_first_domain(fx)]
    return {"stage": 2, "name": "stage2", "port": "y", "fixture": fx,
            "program": fx["program"], "registry": fx["registry"],
            "input_type": fx["input_type"], "full": full,
            "uniform": full,
            "deploy": None, "deploy_what": None,
            "uniform_what": "the exhausted 4^8 domain, all 65,536 inputs",
            "n": fx["n"], "carrier_values": fx["carrier_values"]}


# ---------------------------------------------------------------------------
# worst case
# ---------------------------------------------------------------------------
def worst_case(g, bmod):
    """The record on which arm B does the most work.  Chosen by measurement."""
    if g["stage"] == 3:
        # arm B runs one loop per direction, w iterations and h iterations, so
        # its executed loop iterations are w + h.  `max(w, h)` is NOT that, and
        # picking it would understate the worst case.
        best, rec = -1, None
        for r in g["uniform"]:
            o = s2_reference(r)
            e = o[2] + o[3]
            if e > best:
                best, rec = e, r
        return rec, {"criterion": "max arm-B loop iterations, w + h",
                     "value": best,
                     "extents": list(s2_reference(rec)[2:4]),
                     "ceiling": 2 * 31}
    if g["stage"] == 1:
        # every position satisfies the predicate: every guard is taken and every
        # expensive module runs, plus four branch tests paid for nothing
        rec = tuple([g["hit"]] * g["n"])
        return rec, {"criterion": "every position hits", "value": g["n"]}
    # stage 2: the input on which the scan runs to the end
    for r in g["full"]:
        if bmod.run(r) == g["n"]:
            return r, {"criterion": "no position satisfies the predicate -- "
                                    "the scan runs all %d iterations" % g["n"],
                       "value": g["n"]}
    raise AssertionError("no worst-case input found for stage 2")


FIXTURES = {1: stage1_fixture, 2: stage2_fixture, 3: stage3_fixture}


def build(stage):
    g = FIXTURES[stage]()
    amod, asrc = compiled_arm(g["program"], g["registry"], g["name"])
    bmod, bsrc = resynth_arm(stage)
    port = g["port"]

    okey = g["program"].outputs[0][0]

    arms = {
        # `call` is the timed entry, envelope-matched to arm A's only entry
        # point: a dict in, `({outputs}, {state})` out.  `bare` is the plain
        # function, which arm A does not have.  `value` extracts the one output
        # for the equivalence assertions.
        "A": {"what": "compiled specification (tcn/compile.py)",
              "call": lambda rec, _m=amod, _p=port: _m.run({_p: rec}, validate=False),
              "bare": None,
              "value": lambda rec, _m=amod, _p=port, _k=okey:
                  _m.run({_p: rec}, validate=False)[0][_k],
              "source": asrc,
              "source_path": str(OUT / ("%s_compiled.py" % g["name"]))},
        "B": {"what": "resynthesized standalone (research/lazy-guard)",
              "call": lambda rec, _m=bmod, _k=okey: ({_k: _m.run(rec)}, {}),
              "bare": lambda rec, _m=bmod: _m.run(rec),
              "value": lambda rec, _m=bmod: _m.run(rec),
              "source": bsrc,
              "source_path": str(LAZY / "out" / ("stage%d_algorithm.py" % stage))},
    }
    if stage == 3:
        arms["C"] = {"what": "hand-written reference (research/compiled-runtime/fixtures.py)",
                     "call": lambda rec, _k=okey: ({_k: s2_reference(rec)}, {}),
                     "bare": s2_reference,
                     "value": s2_reference,
                     "source": REFERENCE_SOURCE,
                     "source_path": str(OUT / "stage3_reference.py")}
        (OUT / "stage3_reference.py").write_text(REFERENCE_SOURCE)

    g["arms"] = arms
    g["a_module"], g["b_module"] = amod, bmod
    g["worst"], g["worst_note"] = worst_case(g, bmod)
    return g


def interpreter_output(g, rec):
    """The shipped typed interpreter's answer -- the equivalence oracle."""
    from tcn.types import Value
    out, _ = g["program"].run({g["port"]: Value(g["input_type"], rec)},
                              registry=g["registry"])
    return out[g["program"].outputs[0][0]].decoded
