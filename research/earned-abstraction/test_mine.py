"""Machinery checks for the selection rule, independent of any search run.

These are not the experiment. They check that fragment enumeration, canonical
identity, rewriting and the MDL score behave as `mine.py` documents, on
programs built by hand so the expected answer is known.
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate

import mine
from corpus import examples_for, maj


def build(gates, inputs=("a", "b", "c", "d"), out=None):
    r = Registry()
    depth = {k: 0 for k in inputs}
    nodes = []
    for op_name, name, *args in gates:
        op = r.resolve(op_name, tuple(BOOL for _ in args))
        d = max(depth[a] for a in args) + 1
        depth[name] = d
        nodes.append(Node(name, BOOL, (Candidate(op, tuple(args)),), "core", d, 0))
    return Program(tuple((k, BOOL) for k in inputs), tuple(nodes),
                   (("out", out or nodes[-1].name),)).validate(r)


MAJ_ABC = [("and", "g0", "a", "b"), ("or", "g1", "a", "b"),
           ("or", "g2", "c", "g0"), ("and", "g3", "g1", "g2")]


def test_fragments_and_canonical_identity():
    # the same 4-gate MAJ3 body over two different input triples must land on
    # one canonical digest -- ports are holes, not names.
    p1 = build(MAJ_ABC + [("xor", "g4", "g3", "d")])
    p2 = build([("and", "h0", "b", "c"), ("or", "h1", "b", "c"),
                ("or", "h2", "d", "h0"), ("and", "h3", "h1", "h2"),
                ("or", "h4", "h3", "a")])
    d1 = {c.digest for _, _, c, _ in mine.fragments(p1)}
    d2 = {c.digest for _, _, c, _ in mine.fragments(p2)}
    shared = d1 & d2
    assert shared, "no shared abstraction found between two MAJ3 programs"
    four = [c for _, _, c, _ in mine.fragments(p1) if c.digest in shared and len(c.nodes) == 4]
    assert four, "the 4-node MAJ3 body is not among the shared abstractions"
    assert mine.truth_table(four[0]) == tuple(
        (bits, maj(*bits)) for bits in itertools.product((False, True), repeat=3))
    print("ok: canonical identity, shared digests =", len(shared))


def test_single_exit_is_enforced():
    # g0 is read by g2 and by the output combiner, so {g0,g2} is not single-exit
    p = build(MAJ_ABC + [("xor", "g4", "g3", "g0")])
    for root, S, canon, holes in mine.fragments(p):
        for k in S:
            if k == root:
                continue
            readers = {n.name for n in p.nodes
                       if k in n.candidates[n.selected or 0].sources}
            assert readers <= S, f"fragment {sorted(S)} leaks {k}"
    print("ok: every enumerated fragment is single-exit")


def test_rewrite_preserves_semantics_and_scores():
    corpus = {
        "x1": build(MAJ_ABC + [("xor", "g4", "g3", "d")]),
        "x2": build(MAJ_ABC + [("and", "g4", "g3", "d")]),
        "x3": build([("and", "h0", "b", "c"), ("or", "h1", "b", "c"),
                     ("or", "h2", "d", "h0"), ("and", "h3", "h1", "h2"),
                     ("or", "h4", "h3", "a")]),
    }
    fns = {"x1": lambda a, b, c, d: maj(a, b, c) != d,
           "x2": lambda a, b, c, d: maj(a, b, c) and d,
           "x3": lambda a, b, c, d: maj(b, c, d) or a}
    ex = {k: examples_for(v) for k, v in fns.items()}
    # the hand-built programs really do compute those functions
    for k, p in corpus.items():
        for row in ex[k]:
            out, _ = p.run(row["inputs"], registry=Registry())
            assert next(iter(out.values())).decoded == row["targets"]["out"].decoded, k
    props = mine.propose(corpus, ex)
    assert props, "no eligible abstraction"
    top = props[0]
    print("ok: top proposal", top.nodes, "nodes,", top.holes, "holes, saving",
          top.saving_bits, "bits, tasks", top.tasks)
    assert top.saving_bits > 0
    assert mine.truth_table(top.canonical) == tuple(
        (bits, maj(*bits)) for bits in itertools.product((False, True), repeat=3)), \
        "expected MAJ3 to be the top abstraction on a corpus built from MAJ3"


if __name__ == "__main__":
    test_fragments_and_canonical_identity()
    test_single_exit_is_enforced()
    test_rewrite_preserves_semantics_and_scores()
