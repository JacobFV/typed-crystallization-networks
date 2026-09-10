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

Each `build()` returns a dict with `registry, program, signals, train, heldout,
sites` and is deterministic.
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


def build_bool():
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
    train = [rows[i] for i in range(0, 64, 2)][:24]
    heldout = [rows[i] for i in range(1, 64, 2)]
    sig = [Signal("y", "out", ("core",), BOOL, "bce")]
    return {"name": "bool", "registry": r, "program": prog, "signals": sig,
            "train": train, "heldout": heldout,
            "sites": ["n1", "n2", "y"],
            "note": "section 44/46 task; 64-row truth table split by row parity"}


# ------------------------------------------------------------------ arith

def build_arith():
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
    train = _episodes_from("arithmetic", range(0, 24), "train", {})
    heldout = _episodes_from("arithmetic", range(1000, 1032), "test", {})
    sig = [Signal("answer", "target", ("core",), I16, "mse")]
    return {"name": "arith", "registry": r, "program": prog, "signals": sig,
            "train": train, "heldout": heldout,
            "sites": ["v0", "v1", "v2", "inner", "answer"],
            "note": "generators/arithmetic, objective drawn per episode"}


# -------------------------------------------------------------------- rel

E3 = integer(3, signed=False, role="category")
EDGE = product(E3, E3)
EDGES = setof(EDGE, 64)


def _pair_module(r):
    """(p) -> (p[0][0], p[1][1]): the composition of two chained edges."""
    pp = product(EDGE, EDGE)
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


def build_rel():
    r = Registry()
    mod = _pair_module(r)
    inputs = (("edges", EDGES), ("query", EDGE))
    joined = setof(product(EDGE, EDGE), 64 * 64)
    join_cands = [Candidate(r.resolve("join", (EDGES, EDGES), joined,
                                      {"left": lft, "right": rgt}), ("edges", "edges"))
                  for lft in (0, 1) for rgt in (0, 1)]
    two = r.resolve("map", (joined,), None, {"module": mod})
    step2_t = two.output
    nodes = (
        Node("chain", joined, tuple(join_cands), "core", 1),
        Node("step2", step2_t, (Candidate(two, ("chain",)),), "core", 2),
        Node("m1", BOOL, (Candidate(r.resolve("member", (EDGES, EDGE)),
                                    ("edges", "query")),), "core", 1),
        Node("m2", BOOL, (Candidate(r.resolve("member", (step2_t, EDGE)),
                                    ("step2", "query")),), "core", 3),
        Node("ans", BOOL, tuple(legal_candidates(
            r, ("and", "or", "xor", "nand", "nor", "xnor", "eq", "not", "identity"),
            {"m1": BOOL, "m2": BOOL}, BOOL, arities=(1, 2))), "core", 4),
    )
    prog = Program(inputs, nodes, (("out", "ans"),)).validate(r)
    cfg = {"entities": 4}
    train = _episodes_from("relations", range(0, 24), "train", cfg)
    heldout = _episodes_from("relations", range(1000, 1064), "test", cfg)
    sig = [Signal("ans", "target", ("core",), BOOL, "bce")]
    return {"name": "rel", "registry": r, "program": prog, "signals": sig,
            "train": train, "heldout": heldout,
            "sites": ["chain", "ans"],
            "note": "generators/relations, two-step reachability"}


# ------------------------------------------------------------------- lang

def _lang_rows(eps, positions):
    del positions
    return [{"inputs": {"text": e["text"]},
             "targets": {"answer": Value.of(BOOL, bool(e["label"]))}} for e in eps]


def build_lang(positions=22):
    """Bracket grammaticality, section 45's family, at a reduced width."""
    sys.path.insert(0, str(kit.ROOT / "research" / "scaffold-diagnosis"))
    sys.path.insert(0, str(kit.ROOT / "research" / "language-capability"))
    import scaffolds2
    import splits as splits_mod
    from run_stage_b import build_module
    module, registry, _frozen = build_module()
    prog, sig = scaffolds2.stage_b_gen(module, registry, positions=positions,
                                       sub_range=range(0, 17),
                                       acc_fold="add", second_fold="min")
    s = splits_mod.build(hardening="context_free_language", n_train=24,
                         train_lengths=(10, 12), heldout_lengths=(14, 16))
    train = _lang_rows(s["train"], positions)
    heldout = _lang_rows(s["heldout_unseen_lengths"][:32], positions)
    return {"name": "lang", "registry": registry, "program": prog, "signals": sig,
            "train": train, "heldout": heldout,
            "sites": [nd.name for nd in prog.nodes if len(nd.candidates) > 1],
            "note": "section 45's family at positions=8"}


BUILDERS = {"bool": build_bool, "arith": build_arith, "rel": build_rel,
            "lang": build_lang}
