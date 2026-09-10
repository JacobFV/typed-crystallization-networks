"""The motif as a schema: one piece of code, three instantiated artifacts.

This is §53's form (`research/depth-encoding/scaffold.py`) applied to §58's
motif. A schema is a **function** from type parameters to a fully typed,
`Program.validate`-checked scaffold. It is not a program, and nothing here is
serialisable as one — which is exactly why §55 stores a schema *reference*.

Two parameters, not one
-----------------------
`width` is the buffer's field count. `address` is the declared carrier of the
index. They are independent: the language and computer buffers declare
`u32` refined to `(0, N)` where `N` is the width, but the visual buffer's
addresses are a plain `u16` with 3,072 fields. `research/cross-domain/RESULTS.md`
§10 named this precisely — "a class identity that abstracts over *product width
and integer carrier together*" — so the schema takes both.

Nothing is relaxed
------------------
Every instantiation constructs the artifact's declared types (`role="byte"`
elements, the exact `bounds` where the artifact declares them), resolves every
operator through `Registry.resolve` and passes `Program.validate(registry)`. No
`TypeError` is caught, no bound is widened, no width is erased, and `tcn/` is not
touched. `run_step2.py` asserts each of those as a gate rather than asserting it
in prose.

The shape of the choice is width-invariant — and for `M2` it is empty
--------------------------------------------------------------------
`M3` has exactly one free node, the address combiner, with 5 candidates at every
width; an integer selection vector found at one width therefore names a program
at every other, which is §53's observation and what `ClassStore.instantiate`
checks before it rebuilds. `M2` has **zero** free nodes at every width, because
`index` is the only operator that reads a product at a computed offset and `eq`
is the only comparison the algebra admits on two `role="byte"` values. That is a
measurement, reported in `RESULTS.md`, not a design choice: the one motif that
spans all three real domains is too small to carry a selection vector.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from dataclasses import replace                                   # noqa: E402
from tcn.generation import BYTE                                   # noqa: E402
from tcn.graph import Candidate, Node, Program                    # noqa: E402
from tcn.operators import Registry                                # noqa: E402
from tcn.types import integer, product                            # noqa: E402

# Candidate pools. Fixed here, before any arm, and identical at every width.
COMBINE = ("add", "sub", "mul", "min", "max")
COMPARE = ("eq", "lt", "le", "gt", "ge")


def compare_pool(element):
    """The comparisons the algebra admits on `element` -- read, never guessed.

    `tcn/types.py:27` declares `byte` an UNCOMMITTED role, so `Type.numeric` is
    false for it and `tcn/operators.py:67` admits **only** `eq` on two bytes.
    That is the type system's decision, not this file's, and it is measured here
    rather than discovered by catching a `TypeError`: the compare node of the
    motif carries no choice at any width, in any of the three domains.
    """
    return COMPARE if element.numeric else ("eq",)

# The three declared settings, read off the artifacts in step 1 and written out
# so the schema constructs them rather than copying a `Type` object.
SETTINGS = {
    "language": {"width": 128, "address": ("u32", 32, False, (0, 128))},
    "visual": {"width": 3072, "address": ("u16", 16, False, None)},
    "computer": {"width": 4096, "address": ("u32", 32, False, (0, 4096))},
}


def address_type(spec):
    _, bits, signed, bounds = spec
    return integer(bits, signed=signed, bounds=bounds) if bounds is not None \
        else integer(bits, signed=signed)


def buffer_type(width, element=BYTE):
    return product(*(element for _ in range(width)))


def _node(name, output, cands, depth):
    return Node(name, output, tuple(cands), "core", depth)


def free_nodes(program):
    """The nodes that actually carry a choice, measured from the built program."""
    return tuple(n.name for n in program.nodes if len(n.candidates) > 1)


def m2_schema(width, address, element=BYTE, registry=None):
    """`<cmp>(index(x0, x1), x2)` — the 2-node motif, one free node.

    Holes, in the order `frag.canonicalize` assigns them to the artifacts:
    x0 buffer, x1 address, x2 the compared byte.
    """
    r = registry or Registry()
    bt, at = buffer_type(width, element), address
    inputs = (("x0", bt), ("x1", at), ("x2", element))
    idx = r.resolve("index", (bt, at))
    n0 = _node("n0", idx.output, [Candidate(idx, ("x0", "x1"))], 1)
    cmps = [r.resolve(c, (idx.output, element)) for c in compare_pool(element)]
    n1 = _node("n1", cmps[0].output, [Candidate(o, ("n0", "x2")) for o in cmps], 2)
    program = Program(inputs, (n0, n1), (("out", "n1"),)).validate(r)
    return free_nodes(program), program, r


def m3_schema(width, address, element=BYTE, registry=None):
    """`<cmp>(index(x2, <comb>(x0, x1)), x3)` — §58's 3-node motif, two free nodes.

    Holes: x0 base, x1 offset, x2 buffer, x3 the compared byte — the order
    `frag.canonicalize` assigns in both `V_same` and `L_stage_a`.
    """
    r = registry or Registry()
    bt, at = buffer_type(width, element), address
    inputs = (("x0", at), ("x1", at), ("x2", bt), ("x3", element))
    combs = [r.resolve(c, (at, at)) for c in COMBINE]
    n0 = _node("n0", combs[0].output, [Candidate(o, ("x0", "x1")) for o in combs], 1)
    idx = r.resolve("index", (bt, combs[0].output))
    n1 = _node("n1", idx.output, [Candidate(idx, ("x2", "n0"))], 2)
    cmps = [r.resolve(c, (idx.output, element)) for c in compare_pool(element)]
    n2 = _node("n2", cmps[0].output, [Candidate(o, ("n1", "x3")) for o in cmps], 3)
    program = Program(inputs, (n0, n1, n2), (("out", "n2"),)).validate(r)
    return free_nodes(program), program, r


# ------------------------------------------------------------------- arm F
def m3_wrong_hardcoded(width, address, element=BYTE, registry=None):
    """F-a. The buffer width is hard-coded to the first certified width.

    Bit-identical to `m3_schema` at width 128 and structurally impossible
    anywhere else: at any other width the caller's buffer no longer has the type
    this schema declares. This is the failure mode a single-width certificate
    cannot see.
    """
    return m3_schema(128, address, element, registry)


def m3_wrong_reordered(width, address, element=BYTE, registry=None):
    """F-b. §53 arm F's exact shape: a width-dependent candidate ordering.

    The address pool is rotated by `(width - 128) % len(COMBINE)`, so at width 128
    this schema is **bit-identical** to `m3_schema` and earns the identical
    certificate, while at width 3,072 the *same* stored selection index 0 names
    `max` instead of `add`. The schema is otherwise type-correct and
    `validate`-clean at every width — which is the point: nothing about a single
    width's `unique` certificate distinguishes it from `m3_schema`.
    """
    r = registry or Registry()
    bt, at = buffer_type(width, element), address
    inputs = (("x0", at), ("x1", at), ("x2", bt), ("x3", element))
    k = (width - 128) % len(COMBINE)
    pool = COMBINE[k:] + COMBINE[:k]
    combs = [r.resolve(c, (at, at)) for c in pool]
    n0 = _node("n0", combs[0].output, [Candidate(o, ("x0", "x1")) for o in combs], 1)
    idx = r.resolve("index", (bt, combs[0].output))
    n1 = _node("n1", idx.output, [Candidate(idx, ("x2", "n0"))], 2)
    cmps = [r.resolve(c, (idx.output, element)) for c in compare_pool(element)]
    n2 = _node("n2", cmps[0].output, [Candidate(o, ("n1", "x3")) for o in cmps], 3)
    program = Program(inputs, (n0, n1, n2), (("out", "n2"),)).validate(r)
    return free_nodes(program), program, r


SCHEMAS = {"M2": m2_schema, "M3": m3_schema,
           "M3_wrong_hardcoded": m3_wrong_hardcoded,
           "M3_wrong_reordered": m3_wrong_reordered}


# ------------------------------------------------------------- instantiation
def freeze(program, selections, registry):
    """Select, prune, and normalise the one field that is provenance, not graph.

    `Program.harden` increments `version`, a monotone provenance counter that is
    not part of the graph; `frag.canonicalize` builds a fresh program, so an
    artifact fragment always carries `version = 1`. Normalising `version` is the
    **only** normalisation applied, and `run_step2.py` reports the field-by-field
    comparison of `to_dict()` so the reader can see that nothing else was touched.
    """
    exact = program.harden(selections).pruned()
    exact = replace(exact, version=1)
    return exact.validate(registry)


def instantiate(name, width, address, selections, element=BYTE):
    """Build the schema at (width, address) and apply the selection vector."""
    free, program, r = SCHEMAS[name](width, address, element)
    full = {n.name: 0 for n in program.nodes}
    full.update(selections)
    return free, freeze(program, full, r), r
