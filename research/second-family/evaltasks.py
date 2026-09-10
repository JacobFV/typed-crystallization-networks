"""The evaluation scaffold, the held-out tasks and the two off-family controls.

This module *replaces* `research/semantic-library/evaltasks.py` (same module
name, this directory first on `sys.path`) so that §54's `pool.py`,
`context.py` and `run_heldout.py` machinery can be imported unchanged.  It is
the same construction at five inputs instead of four:

*   `compact_scaffold` is two nodes -- `n1` reads the five inputs, `y` reads
    the five inputs and `n1` -- built by `later.bool_candidates` and
    `later.module_candidates`, imported from §44 and not restated.
*   A two-node program in this basis reaches at most **four** distinct inputs
    (`y` reads two ports, one of which may be `n1`, which reads two more), and
    every task of the family depends essentially on all five, so the
    no-library arm cannot solve any of them.  That is the same structural
    property that makes §52's compact scaffold a fair test, and it is verified
    by exhaustion in every arm rather than assumed.
*   With a 4-ary module the space is 710 x 1416 = 1,005,360 programs, small
    enough to exhaust in every cell, so every number carries a certificate.

The two controls are §52's `H_par` / `H_d134` pattern:

    H_par5   a xor b xor c xor d xor e   -- shares nothing with the family
    H_x4     X4(a,b,c,d) xor e           -- the off-family window in a family
                                            shape; the *wrong* module solves
                                            it and the right one must not,
                                            which is what makes the control live
"""
from __future__ import annotations

import itertools

import _paths  # noqa: F401

from tcn.types import BOOL, Value
from tcn.graph import Program, Node, Signal

import later

import family

COMPACT_INPUTS = family.INPUTS            # ("a", "b", "c", "d", "e")
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
    look = COMPACT_INPUTS + ("n1",)
    y = Node("y", BOOL,
             tuple(later.bool_candidates(r, top)
                   + later.module_candidates(r, module_name, look)),
             "core", 2)
    return Program(inputs, (n1, y), (("out", "y"),)).validate(r)


def examples_for(fn):
    rows = []
    for bits in itertools.product((False, True), repeat=len(COMPACT_INPUTS)):
        rows.append({"inputs": {k: Value.of(BOOL, v) for k, v in zip(COMPACT_INPUTS, bits)},
                     "targets": {"out": Value.of(BOOL, bool(fn(*bits)))}})
    return rows


def _par5(a, b, c, d, e):
    """Five-input parity.  Written with `^` deliberately: §52 disclosed that the
    `!=`-chained form is Python's chained comparison, not parity."""
    return bool(a) ^ bool(b) ^ bool(c) ^ bool(d) ^ bool(e)


def _x4_xor_e(a, b, c, d, e):
    return family.x4(a, b, c, d) != bool(e)


def _later(a, b, c, d, e):
    """`L1_w4_bdae_xor_c` -- the seventh in-family task, in no mining corpus.

    Its pairing `{bd|ae}` is used by none of the six corpus tasks, so an arm
    that solves it has transferred the window and not memorised a pairing.
    """
    return family.w4(b, d, a, e) != bool(c)


LATER_TASK = "L1_w4_bdae_xor_c"
LOO_TASKS = family.LOO_TASKS
CONTROL_TASKS = {"H_par5": _par5, "H_x4": _x4_xor_e}
COMPACT_TASKS = dict(dict(family.W4_TASKS), **CONTROL_TASKS)
COMPACT_TASKS[LATER_TASK] = _later
OFF_TASKS = dict(family.X4_TASKS)
ALL_TASKS = dict(COMPACT_TASKS, **OFF_TASKS)


def constant_baseline(fn):
    rows = [bool(fn(*b)) for b in itertools.product((False, True), repeat=len(COMPACT_INPUTS))]
    ones = sum(rows)
    return max(ones, len(rows) - ones) / len(rows)


def random_baseline():
    return 0.5
