"""Assemble the arms: which library the evaluation task inherits, and nothing else.

Everything except the library contents is held fixed -- same scaffold builder,
same examples, same signals, same enumeration settings.  Modules mined by this
track are loaded from a real `tcn.library.Library` on disk under
`policy="strict"`; the two hand-authored modules are built here because §44's
`later.hand_authored` fixes the hole names `(a, b, c)` at arity 3 and this
family's window is 4-ary.

Arm names follow §52 exactly so the two tables line up row for row.
"""
from __future__ import annotations

from pathlib import Path

import _paths  # noqa: F401

from tcn.graph import Program, Node, Candidate
from tcn.library import Library
from tcn.operators import Registry
from tcn.types import BOOL

import family

HERE = Path(__file__).resolve().parent
LIB = HERE / "library_C-trace"

HOLES = ("w", "x", "y", "z")


def lib_for(band):
    return HERE / f"library_{band}"


def use_band(band):
    """Point `build` at one band's library.  Both bands are declared primary,
    so each has its own on-disk library and the arms are run twice."""
    global LIB
    LIB = lib_for(band)
    return LIB


def hand_authored(r, kind):
    """`W4` or `X4` as a 4-input `Program`, in the same 3-gate minimal form."""
    depth = {h: 0 for h in HOLES}
    nodes = []
    for name, out, p, q in family.WINDOW_BODIES[kind]:
        op = r.resolve(name, (BOOL, BOOL))
        d = max(depth[p], depth[q]) + 1
        depth[out] = d
        nodes.append(Node(out, BOOL, (Candidate(op, (p, q)),), "core", d, 0))
    return Program(tuple((h, BOOL) for h in HOLES), tuple(nodes),
                   (("out", nodes[-1].name),)).validate(r)


DESCRIPTION = {
    "arm1_none": "no library; the flat space",
    "arm2_syntactic": "rank-1 of the rule under digest identity",
    "arm2s_semantic": "rank-1 of the rule under (arity, truth table) identity",
    "arm3_authored": "the hand-authored W4 -- the ceiling",
    "arm4_wrong_authored": "the hand-authored X4, same size and arity",
    "arm4b_wrong_mined": "the syntactic rule's runner-up, same machinery",
    "arm4s_runnerup": "the top arity-4 non-window *pooled* class, same corpus",
    "arm4s_matched": "the top arity-4 non-window pooled class of equal node count",
    "arm4s_offfamily": "the rank-1 arity-4 pooled class mined from the off-family F''",
}

# arm -> ("hand", kind) | ("lib", label) | None
L1_ARMS = {
    "arm1_none": None,
    "arm2_syntactic": ("lib", "syn_rank1"),
    "arm2s_semantic": ("lib", "sem_rank1"),
    "arm3_authored": ("hand", "w4"),
    "arm4_wrong_authored": ("hand", "x4"),
    "arm4b_wrong_mined": ("lib", "syn_runnerup"),
    "arm4s_runnerup": ("lib", "sem_runnerup"),
    "arm4s_matched": ("lib", "sem_matched"),
    "arm4s_offfamily": ("lib", "sem_off"),
}


def build(spec, policy="strict"):
    """(registry, module operator name or None) for one library spec."""
    r = Registry()
    if spec is None:
        return r, None
    kind, name = spec
    if kind == "hand":
        return r, r.register_module(hand_authored(r, name))
    _, aliases = Library(LIB).load([name], registry=r, policy=policy)
    return r, aliases[name]


def runnable_l1_arms():
    """The L1' arms whose library actually exists on disk.

    A selection rule that resolves to *no candidate* -- §52 hit this with
    `arm4s_matched` -- is a measurement, not a failure, and is reported as
    such rather than silently dropped.
    """
    out = []
    for arm, spec in L1_ARMS.items():
        if spec is None or spec[0] == "hand":
            out.append(arm)
            continue
        try:
            build(spec)
        except (KeyError, FileNotFoundError):
            continue
        out.append(arm)
    return out
