"""The second task family, its off-family twin, and the corpora mined from them.

This module *replaces* `research/reuse-ranking/family.py` (same module name,
this directory first on `sys.path`) and keeps its interface exactly:
`FAMILIES`, `corpus_for(family, variant, exclude)`.  Everything downstream --
`pool.build`, `context.build_context`, `objectives.rank` -- is imported from
§54 unchanged and does not know the family changed.

THE FAMILY
----------
The shared abstraction is

    W4(w, x, y, z) = (w XOR x) AND (y XOR z)          "both pairs differ"

over **five** Boolean inputs `a..e`, against §44/§52's `MAJ3(a,b,c)` over four.
It differs from the majority window in every way that could plausibly break
the method:

*   **arity 4, not 3.**  Module call sites cost `ports ** 4`, the pooling key
    is a 16-row table rather than an 8-row one, and arity 4 is exactly
    `mine.MAX_HOLES`, so the class sits at the rule's declared ceiling.
*   **a different algebraic character.**  `MAJ3` is symmetric, monotone and a
    threshold function.  `W4` is a *conjunction of two parities*: not monotone
    (flipping `w` can turn the output off or on), not a threshold function, and
    not symmetric -- it is invariant only under `w<->x`, `y<->z` and the pair
    swap, an 8-element subgroup of S4.
*   **a different size and a different corpus geometry.**  `MAJ3` needs 4 gates
    in this basis; `W4` needs 3.  The family's tasks have certified minimum
    length **4**, against the majority family's 5, so the corpus programs are
    shorter and the fragment lattice around the window is different.
*   **five inputs, not four**, so nothing about the evaluation scaffold, the
    space sizes or the truth-table widths is shared with §52.

The eight-element invariance group is the property that makes `W4` a *fair*
test rather than a rigged one: it is small enough that hole ordering is not
free (`MAJ3` is invariant under all 6 orderings of its holes, so §52's pooling
key could never split it), and large enough that the pairing structure a task
uses is not recoverable from the window alone.

THE TASKS
---------
Six corpus tasks, each `W4` over an ordered 4-subset of `a..e` combined with
the remaining input by one 2-input gate, and each using a *different* pairing:

    u1  W4(a,b,c,d) XOR e        pairs {ab|cd}
    u2  W4(a,c,b,d) AND e        pairs {ac|bd}
    u3  W4(b,c,d,e) OR  a        pairs {bc|de}
    u4  W4(a,c,d,e) XOR b        pairs {ac|de}
    u5  W4(a,b,d,e) OR  c        pairs {ab|de}
    u6  W4(a,b,c,e) AND d        pairs {ab|ce}

`u1..u5` are the leave-one-out held-out tasks; `u6` is in every corpus, the
same protocol §52 used with `t6`.

Every task depends essentially on all five inputs, and a *pruned* straight-line
program of 3 gates over five inputs can depend on at most four of them (the
root reads two ports, each of which is a gate reading two ports), so the
minimum length is at least 4 by construction and is certified at exactly 4 by
`minimal.scaffold_min`.  The same argument makes the two-node evaluation
scaffold unable to express any of them without a module: it reaches at most
four distinct inputs.

THE OFF-FAMILY TWIN
-------------------
`F''` is the same six shapes with `W4` replaced by

    X4(w, x, y, z) = (w AND x) XOR (y OR z)

-- same arity, same 3-gate size, a different table -- exactly the role §52's
`D134` played for `MAJ3`.  It supplies the off-family mining corpus and the
off-family control tasks.

CORPUS CONSTRUCTION
-------------------
§46's machinery, unchanged: `enumerate_programs.exhaustive` for **every**
pruned program at the certified minimum length (`C-minall`) or at minimum + 1
(`C-plus1`), the same per-task cap of 32 with the same deterministic
sha256-of-canonical-key subsample, and every retained program rebuilt as a
`tcn.graph.Program` and executed against its full 32-row truth table before it
enters the corpus.  Only `N` changes from 4 to 5.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import time
from pathlib import Path

import _paths  # noqa: F401

from tcn.operators import Registry

import enumerate_programs as E
import minimal

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

INPUTS = ("a", "b", "c", "d", "e")
N = len(INPUTS)
CAP = 32


# ---------------------------------------------------------------- the windows

def w4(w, x, y, z):
    """(w xor x) and (y xor z) -- the shared abstraction of the second family."""
    return (bool(w) != bool(x)) and (bool(y) != bool(z))


def x4(w, x, y, z):
    """(w and x) xor (y or z) -- the off-family window, same arity, same size."""
    return (bool(w) and bool(x)) != (bool(y) or bool(z))


WINDOW_BODIES = {
    # W4 = and(xor(w,x), xor(y,z)) -- 3 gates, verified minimal by `min_gates`.
    "w4": (("xor", "g0", "w", "x"), ("xor", "g1", "y", "z"),
           ("and", "g2", "g0", "g1")),
    # X4 = xor(and(w,x), or(y,z)) -- 3 gates, same arity, same node count.
    "x4": (("and", "g0", "w", "x"), ("or", "g1", "y", "z"),
           ("xor", "g2", "g0", "g1")),
}


def _shapes(w):
    return (
        ("u1_w_abcd_xor_e", lambda a, b, c, d, e: w(a, b, c, d) != bool(e)),
        ("u2_w_acbd_and_e", lambda a, b, c, d, e: w(a, c, b, d) and bool(e)),
        ("u3_w_bcde_or_a", lambda a, b, c, d, e: w(b, c, d, e) or bool(a)),
        ("u4_w_acde_xor_b", lambda a, b, c, d, e: w(a, c, d, e) != bool(b)),
        ("u5_w_abde_or_c", lambda a, b, c, d, e: w(a, b, d, e) or bool(c)),
        ("u6_w_abce_and_d", lambda a, b, c, d, e: w(a, b, c, e) and bool(d)),
    )


W4_TASKS = _shapes(w4)
X4_TASKS = tuple((n.replace("u", "v", 1), f) for n, f in _shapes(x4))

FAMILIES = {"w4": W4_TASKS, "x4": X4_TASKS}
WINDOW_FN = {"w4": w4, "x4": x4}
LOO_TASKS = tuple(n for n, _ in W4_TASKS)[:5]


def window_table(family_name):
    """The window's 16-row table, in hole order -- the pooling key's second half."""
    fn = WINDOW_FN[family_name]
    return tuple(bool(fn(*b)) for b in itertools.product((False, True), repeat=4))


# ------------------------------------------------------------------ the corpus

def _hash(gates):
    """§46's deterministic, enumeration-order-independent subsample key, at N=5."""
    base = [E.var_table(N, j) for j in range(N)]
    return hashlib.sha256(repr(sorted(E.key_of(gates, base, N))).encode()).hexdigest()


def subsample(gate_lists, cap=CAP):
    if len(gate_lists) <= cap:
        return list(gate_lists)
    return sorted(gate_lists, key=_hash)[:cap]


def to_tcn(gates, registry=None):
    return minimal.to_program(N, [tuple(g) for g in gates], INPUTS, registry or Registry())


def bands_path(family_name):
    return OUT / f"bands_{family_name}.json"


def build_bands(family_name, want=("min", "plus1"), cap=CAP):
    """Every pruned program at the certified minimum length, and at minimum + 1."""
    tasks = []
    t_all = time.perf_counter()
    for name, fn in FAMILIES[family_name]:
        tt = E.table_of(N, fn)
        k, _gates, mstats = minimal.scaffold_min(N, tt, max_len=5)
        assert k is not None, name
        bands = {}
        for label in want:
            length = k if label == "min" else k + 1
            t0 = time.perf_counter()
            sols, exhausted, expanded = E.exhaustive(N, tt, length)
            wall = time.perf_counter() - t0
            capped = subsample(sols, cap)
            r = Registry()
            ok = all(minimal.verify(to_tcn(g, r), fn, INPUTS, r) for g in capped)
            bands[label] = {"length": length, "method": "exhaustive",
                            "exhausted": bool(exhausted), "found": len(sols),
                            "uncapped_found": len(sols), "capped": len(capped),
                            "dfs_expanded": expanded, "wall_seconds": wall,
                            "verified_in_tcn": bool(ok), "programs": capped}
            print(f"  {name} {label} len={length} found={len(sols)} "
                  f"capped={len(capped)} dfs={expanded} {wall:.1f}s verified={ok}",
                  flush=True)
        tasks.append({"task": name, "min_gates": k, "min_certificate": mstats,
                      "bands": bands})
    payload = {"cap": cap, "inputs": list(INPUTS), "family": family_name,
               "window": family_name.upper(), "tasks": tasks,
               "wall_seconds": time.perf_counter() - t_all}
    OUT.mkdir(exist_ok=True)
    bands_path(family_name).write_text(json.dumps(payload, indent=2, sort_keys=True))
    return payload


def load_bands(family_name):
    return json.loads(bands_path(family_name).read_text())


def corpus_for(family_name, variant="C-minall", exclude=()):
    """(corpus, task_of) with `exclude`d tasks removed **entirely**, as §52."""
    data = load_bands(family_name)
    want = {"C-minall": ("min",), "C-plus1": ("plus1",),
            "C-trace": ("min", "plus1")}[variant]
    corpus, task_of = {}, {}
    for t in data["tasks"]:
        for label in want:
            for i, gates in enumerate(t["bands"][label]["programs"]):
                key = f"{t['task']}#{label}{i:03d}"
                corpus[key] = to_tcn(gates)
                task_of[key] = t["task"]
    drop = set(exclude)
    corpus = {e: p for e, p in corpus.items() if task_of[e] not in drop}
    task_of = {e: t for e, t in task_of.items() if t not in drop}
    return corpus, task_of
