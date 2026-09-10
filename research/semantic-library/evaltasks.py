"""The evaluation tasks and their scaffolds.

`L1` is §44's later task and §44's two scaffolds, imported from
`research/earned-abstraction/later.py` and not restated, so every number here is
directly comparable to §44's and §46's.

The **compact scaffold** is this track's addition and it exists for the held-out
protocol: two nodes over the four Boolean inputs, built by the same candidate
construction as `later.py`.  Every held-out task has certified minimum flat
length 5 in this basis, so no flat program fits two nodes and the no-library arm
must exhaust at zero -- the same structural property that makes §44's tight
scaffold a fair test.
"""
from __future__ import annotations

import itertools

import _paths  # noqa: F401

from tcn.types import BOOL, Value
from tcn.graph import Program, Node, Signal

import later
from corpus import CORPUS_INPUTS, EARLIER_TASKS

import family

COMPACT_INPUTS = CORPUS_INPUTS  # ("a", "b", "c", "d")
SIGNALS = (Signal("y", "out", ("core",), BOOL, "bce"),)


def compact_scaffold(r, module_name=None):
    """Two nodes: `n1` reads the inputs, `y` reads the inputs and `n1`."""
    inputs = tuple((k, BOOL) for k in COMPACT_INPUTS)
    base = {k: BOOL for k in COMPACT_INPUTS}
    n1 = Node("n1", BOOL,
              tuple(later.bool_candidates(r, base)
                    + later.module_candidates(r, module_name, COMPACT_INPUTS)),
              "core", 1)
    top = dict(base, n1=BOOL)
    pool = COMPACT_INPUTS + ("n1",)
    y = Node("y", BOOL,
             tuple(later.bool_candidates(r, top)
                   + later.module_candidates(r, module_name, pool)),
             "core", 2)
    return Program(inputs, (n1, y), (("out", "y"),)).validate(r)


def examples_for(fn):
    rows = []
    for bits in itertools.product((False, True), repeat=len(COMPACT_INPUTS)):
        rows.append({"inputs": {k: Value.of(BOOL, v) for k, v in zip(COMPACT_INPUTS, bits)},
                     "targets": {"out": Value.of(BOOL, bool(fn(*bits)))}})
    return rows


def _parity(a, b, c, d):
    """`a xor b xor c xor d`, as pre-registered.

    The first version of this function was written `bool(a) != bool(b) !=
    bool(c) != bool(d)`, which Python evaluates as the *chained comparison*
    `(a != b) and (b != c) and (c != d)` -- the alternating-sequence predicate,
    not parity.  The defect was caught by its own class balance (14/16 instead
    of 8/16) in the first held-out run and is recorded in `RESULTS.md`; the
    affected rows were re-run.
    """
    return bool(a) ^ bool(b) ^ bool(c) ^ bool(d)


def _d134_xor_d(a, b, c, d):
    return family.d134(a, b, c) != bool(d)


# The five leave-one-out held-out tasks, plus the two declared controls.
HELD_OUT = dict(EARLIER_TASKS)
LOO_TASKS = ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
             "t4_maj_acd_xor_b", "t5_maj_abd_or_c")
CONTROL_TASKS = {"H_par": _parity, "H_d134": _d134_xor_d}
COMPACT_TASKS = dict(HELD_OUT, **CONTROL_TASKS)


def constant_baseline(fn):
    rows = [bool(fn(*b)) for b in itertools.product((False, True), repeat=4)]
    ones = sum(rows)
    return max(ones, len(rows) - ones) / len(rows)
