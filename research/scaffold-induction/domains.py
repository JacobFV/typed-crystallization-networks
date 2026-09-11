"""The domains: a base scaffold, a train/held-out episode split, and signals.

Every domain is one of the shipped generators or one of the tasks this
repository already has a recorded result for.  Nothing under `tcn/` or
`generators/` is modified; the generators are imported and run.

    bool   `maj(a,b,c) xor maj(d,e,f)` over six BOOL inputs -- section 44/46's
           task, complete 64-row truth table, split by row.
    arith  `generators/arithmetic` -- two int16 operands and a one-hot goal
           selecting add / sub / max.
    lang   `generators/language`, the bracket grammaticality family of section
           45, at a reduced width so a whole family is exhaustible generically.
    rel    `generators/relations` -- membership in the two-step reachability
           relation of a small directed graph.

Each `build()` returns a dict with `registry, program, signals, train,
admission, final, splits, sites` and is deterministic.

**Three disjoint splits (amendment A13).**  `train` is what a repaired scaffold
must fit to be a candidate; `admission` decides both that a defect is admissible
and that a candidate is a repair; `final` is touched by nothing until the arms
are scored.  A counterexample drawn from `admission` is training information the
moment it is used, so a two-way split would leave nothing genuinely blind to
report a generalization number on.
"""
from __future__ import annotations

import itertools
import sys

import kit  # noqa: F401  (puts ROOT on sys.path)
from tcn.generation import Host, SCALAR
from tcn.graph import Candidate, Node, Program, Signal, legal_candidates
from tcn.operators import Registry
from tcn.types import BOOL, Value, integer, product, setof

sys.path.insert(0, str(kit.ROOT / "research" / "earned-abstraction"))

I16 = integer(16)


def _episodes_from(name, seeds, split, configuration, target_key="target"):
    rows = []
    for s in seeds:
        host = Host.create(name, seed=s, split=split, configuration=configuration)
        rec = host.records[-1]
        view = rec.actor_view()
        rows.append({"inputs": dict(view.observations),
                     "targets": {target_key: rec.probes[target_key]}})
    return rows


# ------------------------------------------------------------------- bool

def _maj_module(r):
    import later
    prog = later.hand_authored(r, "maj")
    return r.register_module(prog)


def build_bool(n_train=8):                  # budget fixed by A8 (re-run under A13)
    r = Registry()
    mn = _maj_module(r)
    import later
    inputs = tuple((k, BOOL) for k in later.INPUTS)
    op3 = r.resolve(mn, (BOOL, BOOL, BOOL))
    # MAJ3 is symmetric, so the ordered triples collapse to the 20 combinations;
    # this is a scaffold-design choice, disclosed, and it is what keeps the base
    # space exhaustible.  Every edit re-derives its pool from the registry and is
    # free to re-introduce the orderings.
    triples = tuple(itertools.combinations(later.INPUTS, 3))
    mods = tuple(Candidate(op3, t) for t in triples)
    top = {"n1": BOOL, "n2": BOOL}
    y = legal_candidates(r, ("and", "or", "xor", "nand", "nor", "xnor", "eq",
                             "not", "identity"), top, BOOL, arities=(1, 2))
    nodes = (Node("n1", BOOL, mods, "core", 1),
             Node("n2", BOOL, mods, "core", 1),
             Node("y", BOOL, tuple(y), "core", 2))
    prog = Program(inputs, nodes, (("out", "y"),)).validate(r)
    rows = later.later_examples()
    # A13: three disjoint splits over the complete 64-row table.  `train` is an
    # even stride; the complement is dealt alternately into `admission` and
    # `final`, so neither is a contiguous or otherwise special region.
    stride = 64 // n_train
    picked = sorted(range(0, 64, stride))
    rest = [i for i in range(64) if i not in set(picked)]
    train = [rows[i] for i in picked]
    admission = [rows[i] for i in rest[0::2]]
    final = [rows[i] for i in rest[1::2]]
    splits = {"train_rows": picked, "admission_rows": rest[0::2],
              "final_rows": rest[1::2]}
    sig = [Signal("y", "out", ("core",), BOOL, "bce")]
    return {"name": "bool", "registry": r, "program": prog, "signals": sig,
            "train": train, "admission": admission, "final": final,
            "splits": splits, "sites": ["n1", "n2", "y"],
            "note": "section 44/46 task; complete 64-row truth table, three "
                    "disjoint splits over row indices"}


# ------------------------------------------------------------------ arith

def build_arith(n_train=16, n_eval=32):  # budget fixed by A8
    r = Registry()
    goal_t = product(SCALAR, SCALAR, SCALAR)
    inputs = (("x", I16), ("y", I16), ("goal", goal_t))
    consts = (("half", Value.of(SCALAR, 0.5)),)
    pool = {"x": I16, "y": I16}
    ops = ("add", "sub", "min", "max", "neg", "identity")
    vc = legal_candidates(r, ops, pool, I16, arities=(1, 2))
    nodes = [
        Node("g0", SCALAR, (Candidate(r.resolve("project", (goal_t,), SCALAR,
                                                {"index": 0}), ("goal",)),), "core", 1),
        Node("g1", SCALAR, (Candidate(r.resolve("project", (goal_t,), SCALAR,
                                                {"index": 1}), ("goal",)),), "core", 1),
        Node("v0", I16, tuple(vc), "core", 1),
        Node("v1", I16, tuple(vc), "core", 1),
        Node("v2", I16, tuple(vc), "core", 1),
        Node("is_add", BOOL, (Candidate(r.resolve("gt", (SCALAR, SCALAR)),
                                        ("g0", "half")),), "core", 2),
        Node("is_sub", BOOL, (Candidate(r.resolve("gt", (SCALAR, SCALAR)),
                                        ("g1", "half")),), "core", 2),
        Node("inner", I16, (Candidate(r.resolve("mux", (BOOL, I16, I16)),
                                      ("is_sub", "v1", "v2")),), "core", 3),
        Node("answer", I16, (Candidate(r.resolve("mux", (BOOL, I16, I16)),
                                       ("is_add", "v0", "inner")),), "core", 4),
    ]
    prog = Program(inputs, tuple(nodes), (("out", "answer"),), consts).validate(r)
    # A13: three disjoint seed ranges, recorded.
    tr = range(0, n_train)
    ad = range(1000, 1000 + n_eval)
    fi = range(5000, 5000 + n_eval)
    train = _episodes_from("arithmetic", tr, "train", {})
    admission = _episodes_from("arithmetic", ad, "test", {})
    final = _episodes_from("arithmetic", fi, "test", {})
    splits = {"train_seeds": [tr.start, tr.stop], "admission_seeds": [ad.start, ad.stop],
              "final_seeds": [fi.start, fi.stop], "split_names": ["train", "test", "test"]}
    sig = [Signal("answer", "target", ("core",), I16, "mse")]
    return {"name": "arith", "registry": r, "program": prog, "signals": sig,
            "train": train, "admission": admission, "final": final, "splits": splits,
            "sites": ["v0", "v1", "v2", "inner", "answer"],
            "note": "generators/arithmetic, objective drawn per episode"}


# -------------------------------------------------------------------- rel

E3 = integer(3, signed=False, role="category")
EDGE = product(E3, E3)
EDGES = setof(EDGE, 64)


def _pair_module(r, pp):
    """(p) -> (p[0][0], p[1][1]): the composition of two chained edges."""
    nodes = (
        Node("l", EDGE, (Candidate(r.resolve("project", (pp,), EDGE, {"index": 0}),
                                   ("p",)),), "core", 1, 0),
        Node("rr", EDGE, (Candidate(r.resolve("project", (pp,), EDGE, {"index": 1}),
                                    ("p",)),), "core", 1, 0),
        Node("a", E3, (Candidate(r.resolve("project", (EDGE,), E3, {"index": 0}),
                                 ("l",)),), "core", 2, 0),
        Node("d", E3, (Candidate(r.resolve("project", (EDGE,), E3, {"index": 1}),
                                 ("rr",)),), "core", 2, 0),
        Node("out", EDGE, (Candidate(r.resolve("tuple", (E3, E3), EDGE), ("a", "d")),),
             "core", 3, 0),
    )
    prog = Program((("p", pp),), nodes, (("edge", "out"),)).validate(r)
    return r.register_module(prog)


BOOLCOMB = ("and", "or", "xor", "nand", "nor", "xnor", "eq", "not", "identity")


def build_rel(n_train=384, n_eval=96):   # budget fixed by A8
    """Reachability over a 3-entity digraph: one, two and three edge steps.

    Three steps is what the closure needs at three entities -- a simple path
    between distinct vertices is at most two edges and a cycle at most three --
    so the base scaffold can express the task exactly.  That is checked, not
    assumed: `run_domain.py` records the base's certificate on all episodes.
    """
    r = Registry()
    inputs = (("edges", EDGES), ("query", EDGE))

    def chain(src_t, src_name, other_t, other_name, depth, name):
        out_t = setof(product(src_t.items[0], other_t.items[0]),
                      src_t.capacity * other_t.capacity)
        cands = [Candidate(r.resolve("join", (src_t, other_t), out_t,
                                     {"left": lf, "right": rg}),
                           (src_name, other_name))
                 for lf in (0, 1) for rg in (0, 1)]
        return Node(name, out_t, tuple(cands), "core", depth), out_t

    c2, c2t = chain(EDGES, "edges", EDGES, "edges", 1, "chain2")
    map2 = r.resolve("map", (c2t,), None, {"module": _pair_module(r, c2t.items[0])})
    s2t = map2.output
    c3, c3t = chain(s2t, "step2", EDGES, "edges", 3, "chain3")
    map3 = r.resolve("map", (c3t,), None, {"module": _pair_module(r, c3t.items[0])})
    s3t = map3.output
    nodes = (
        c2,
        Node("step2", s2t, (Candidate(map2, ("chain2",)),), "core", 2),
        c3,
        Node("step3", s3t, (Candidate(map3, ("chain3",)),), "core", 4),
        Node("m1", BOOL, (Candidate(r.resolve("member", (EDGES, EDGE)),
                                    ("edges", "query")),), "core", 1),
        Node("m2", BOOL, (Candidate(r.resolve("member", (s2t, EDGE)),
                                    ("step2", "query")),), "core", 3),
        Node("m3", BOOL, (Candidate(r.resolve("member", (s3t, EDGE)),
                                    ("step3", "query")),), "core", 5),
        Node("o1", BOOL, tuple(legal_candidates(r, BOOLCOMB, {"m1": BOOL, "m2": BOOL},
                                                BOOL, arities=(1, 2))), "core", 4),
        Node("ans", BOOL, tuple(legal_candidates(r, BOOLCOMB, {"o1": BOOL, "m3": BOOL},
                                                 BOOL, arities=(1, 2))), "core", 6),
    )
    prog = Program(inputs, nodes, (("out", "ans"),)).validate(r)
    cfg = {"entities": 3}
    tr = range(0, n_train)
    ad = range(1000, 1000 + n_eval)
    fi = range(5000, 5000 + n_eval)
    train = _episodes_from("relations", tr, "train", cfg)
    admission = _episodes_from("relations", ad, "test", cfg)
    final = _episodes_from("relations", fi, "test", cfg)
    splits = {"train_seeds": [tr.start, tr.stop], "admission_seeds": [ad.start, ad.stop],
              "final_seeds": [fi.start, fi.stop], "split_names": ["train", "test", "test"]}
    sig = [Signal("ans", "target", ("core",), BOOL, "bce")]
    return {"name": "rel", "registry": r, "program": prog, "signals": sig,
            "train": train, "admission": admission, "final": final, "splits": splits,
            "sites": ["chain2", "chain3", "o1", "ans"],
            "note": "generators/relations at 3 entities, one/two/three edge steps"}


# ------------------------------------------------------------------- lang

def _lang_rows(eps, positions):
    del positions
    return [{"inputs": {"text": e["text"]},
             "targets": {"answer": Value.of(BOOL, bool(e["label"]))}} for e in eps]


def build_lang(positions=12, stream="none"):  # dropped, A4
    """Bracket grammaticality, section 45's family, at a reduced width."""
    sys.path.insert(0, str(kit.ROOT / "research" / "scaffold-diagnosis"))
    sys.path.insert(0, str(kit.ROOT / "research" / "language-capability"))
    import scaffolds2
    import splits as splits_mod
    from run_stage_b import build_module
    module, registry, _frozen = build_module()
    prog, sig = scaffolds2.stage_b_gen(module, registry, positions=positions,
                                       sub_range=range(0, positions + 1),
                                       acc_fold="add", second_fold="min")
    lens = ((4, 6), (8, 10, 12)) if stream == "none" else ((10, 12), (14, 16))
    s = splits_mod.build(hardening=stream, n_train=32,
                         train_lengths=lens[0], heldout_lengths=lens[1])
    train = _lang_rows(s["train"], positions)
    heldout = _lang_rows(s["heldout_unseen_lengths"][:32], positions)
    _ = stream
    return {"name": "lang", "registry": registry, "program": prog, "signals": sig,
            "train": train, "heldout": heldout,
            "sites": [nd.name for nd in prog.nodes if len(nd.candidates) > 1],
            "note": f"section 45's family at positions={positions}, "
                    f"stream={stream!r}"}


BUILDERS = {"bool": build_bool, "arith": build_arith, "rel": build_rel,
            "lang": build_lang}
