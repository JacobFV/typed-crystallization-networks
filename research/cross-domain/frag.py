"""R1/R2 of `research/earned-abstraction/mine.py`, enumerated so it scales.

`mine.py.fragments` takes every subset of a root's ancestor set, which is
`O(|ancestors| choose k)`.  On `V_rect` (586 nodes, ancestor sets in the
hundreds) that is ~10**9 subsets per root and cannot be run.  The set it defines
is nevertheless small, because condition (a) -- every node of S reaches the root
inside S -- makes S a connected sub-DAG.  This module enumerates exactly that
set by growth instead of by subset filtering.

**Correctness argument.**  Growth adds a node `u` that is a source of some
`v` in S.  `v` reaches the root inside S, so `u` reaches the root inside
`S | {u}`: every set produced satisfies (a).  Conversely, take any S satisfying
(a) and remove the node `u != root` whose distance to the root inside S is
maximal; no other node's only in-S path to the root can pass through `u`, since
that node would have strictly greater distance.  So `S \\ {u}` satisfies (a) and
by induction every valid S is reachable.  The two families coincide.

**Correctness check, pre-registered.**  `check_equivalence` compares this
enumerator against `mine.py.fragments` on every program small enough to run
both, and the inventory is void unless the fragment sets are bit-identical.

Conditions (c) single-exit, the hole cap, and canonicalisation are `mine.py`'s
own code, imported rather than re-implemented.
"""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "earned-abstraction"))

import mine  # noqa: E402  the published rule
from tcn.graph import Program, Node, Candidate  # noqa: E402

MAX_NODES = mine.MAX_NODES
MAX_HOLES = mine.MAX_HOLES


def _selected(node):
    return node.candidates[node.selected if node.selected is not None else 0]


def _skeleton(program):
    nodes = {n.name: n for n in program.nodes}
    order = [n.name for n in program.nodes]
    srcs = {k: tuple(_selected(n).sources) for k, n in nodes.items()}
    inner = {k: tuple(s for s in srcs[k] if s in nodes) for k in order}
    readers = {k: set() for k in order}
    for k in order:
        for s in inner[k]:
            readers[s].add(k)
    exits = {v for _, v in program.outputs} | {u for _, _, u in program.state}
    return nodes, order, inner, readers, exits


def connected_sets(root, inner, max_nodes):
    """Every S with root in S, |S| <= max_nodes, every member reaching root in S."""
    start = frozenset((root,))
    seen = {start}
    out = [start]
    frontier = [start]
    while frontier:
        nxt = []
        for S in frontier:
            if len(S) >= max_nodes:
                continue
            cands = {s for k in S for s in inner[k] if s not in S}
            for u in cands:
                T = S | {u}
                if T not in seen:
                    seen.add(T)
                    out.append(T)
                    nxt.append(T)
        frontier = nxt
    return out


def canonicalize(program, S, root, registry=None):
    """`mine.canonicalize`, with the registry made a parameter.

    `mine.canonicalize` validates against a fresh `Registry()`, so any fragment
    containing a frozen `module:` call fails to re-resolve and is silently
    dropped.  That is the published behaviour and `registry=None` reproduces it
    bit-for-bit.  The inventory also runs with the artifact's own registry,
    because dropping every module-calling fragment would delete exactly the
    compositional fragments a cross-domain claim would live in.  Both counts are
    reported.
    """
    if registry is None:
        return mine.canonicalize(program, S, root)
    nodes = {n.name: n for n in program.nodes}
    order = [n.name for n in program.nodes if n.name in S]
    holes = []
    for k in order:
        for s in _selected(nodes[k]).sources:
            if s not in S and s not in holes:
                holes.append(s)
    rename = {k: f"n{i}" for i, k in enumerate(order)}
    hole_name = {h: f"x{i}" for i, h in enumerate(holes)}
    depth = {hole_name[h]: 0 for h in holes}
    new_nodes = []
    for k in order:
        cand = _selected(nodes[k])
        sources = tuple(rename[s] if s in S else hole_name[s] for s in cand.sources)
        d = max(depth[s] for s in sources) + 1
        depth[rename[k]] = d
        new_nodes.append(Node(rename[k], nodes[k].output,
                              (Candidate(cand.operator, sources),), "core", d, 0))
    types = program.port_types()
    inputs = tuple((hole_name[h], types[h]) for h in holes)
    try:
        canon = Program(inputs, tuple(new_nodes), (("out", rename[root]),)).validate(registry)
    except (TypeError, ValueError):
        return None, ()
    return canon, tuple(holes)


def fragments(program, max_nodes=MAX_NODES, max_holes=MAX_HOLES, registry=None):
    """Same tuples as `mine.fragments`: (root, frozenset(S), canonical, holes)."""
    nodes, order, inner, readers, exits = _skeleton(program)
    out = []
    for root in order:
        for S in connected_sets(root, inner, max_nodes):
            if any(readers[k] - S or k in exits for k in S if k != root):
                continue
            canon, holes = canonicalize(program, set(S), root, registry)
            if canon is None or not 1 <= len(holes) <= max_holes:
                continue
            out.append((root, S, canon, holes))
    return out


def _key(rows):
    return sorted((r, tuple(sorted(S)), c.digest, h) for r, S, c, h in rows)


def check_equivalence(program, max_nodes=MAX_NODES, max_holes=MAX_HOLES):
    """Bit-identical against the published rule.  Returns (ok, n_mine, n_here)."""
    a = _key(mine.fragments(program, max_nodes, max_holes))
    b = _key(fragments(program, max_nodes, max_holes))
    return a == b, len(a), len(b)
