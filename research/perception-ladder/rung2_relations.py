"""Rung 2 -- `relations`: reachability from the raw edge set.

Two separate findings live here.

(A) The `closure` latent is NOT probe-able.  `join`/`pair` are the only
    set-composition operators and they multiply capacity (64 -> 4096); no
    conversion narrows a set back, and `union`/`intersection` require identical
    types.  So no program can produce a value of type `set[tuple[E,E], 64]`
    equal to the transitive closure, and `Program.validate_signals` rejects a
    probe on anything else.  `closure_wall()` demonstrates this.

(B) Every set operator declares `gradient="none"`, so `relaxed()` falls through
    to `exact_tensor`, which detaches.  A set-valued node passes NO gradient to
    its predecessors; only its own choice logits learn.  `boundary_program()`
    measures this.

The runnable task is the `target` probe: is the query pair in the closure?
The scaffold below computes it from `edges` and `query` alone.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from tcn.generation import Host, Value, BOOL
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.types import integer, product, setof

E = integer(3, signed=False, role="category")
EDGE = product(E, E)
EDGES = setof(EDGE, 64)
ENTITIES = 4
AND, OR, XOR, NOR = 8, 14, 6, 1
POOL4 = (AND, OR, XOR, NOR)
POOL16 = tuple(range(16))


def examples(seeds, entities=ENTITIES, dense=False):
    out = []
    for seed in seeds:
        h = Host.create("relations", seed=seed, configuration={"entities": entities})
        rec = h.records[-1]; view = rec.actor_view()
        edges = set(map(tuple, h.state["edges"])); q = tuple(h.state["query"])
        ex = {"inputs": {"edges": view.observations["edges"], "query": view.observations["query"],
                         **{f"m{k}": Value.of(E, k) for k in range(entities)}},
              "targets": {"target": rec.probes["target"]}}
        if dense:
            # Derived targets.  These are functions of the OBSERVATION (`edges`,
            # `query`), not of privileged state; they are experimenter-supplied
            # dense supervision, not something the generator emits.
            direct = q in edges
            acc = direct
            for k in range(entities):
                p = ((q[0], k) in edges) and ((k, q[1]) in edges)
                ex["targets"][f"P{k}"] = Value.of(BOOL, bool(p))
                acc = acc or p
                ex["targets"][f"O{k}"] = Value.of(BOOL, bool(acc))
        out.append(ex)
    return out


def reachability_program(r, entities=ENTITIES, pool=POOL16, free_p=True, free_o=True,
                         free_mediators=False):
    """`direct or OR_k (edge(q0,m_k) and edge(m_k,q1))` -- the <=2-step relation."""
    # `relaxed("tuple", ...)` in tcn/learning.py concatenates without broadcasting,
    # so an unbatched constant cannot be paired with a batched value (see RESULTS.md, P1).  The mediator identifiers are therefore supplied as constant-valued
    # *inputs* -- the same value in every example, carrying no episode information.
    consts = ()
    nodes = [
        Node("q0", E, (Candidate(r.resolve("project", (EDGE,), E, {"index": 0}), ("query",)),), "obs", 1),
        Node("q1", E, (Candidate(r.resolve("project", (EDGE,), E, {"index": 1}), ("query",)),), "obs", 1),
        Node("D", BOOL, (Candidate(r.resolve("member", (EDGES, EDGE)), ("edges", "query")),), "sets", 1),
    ]
    for k in range(entities):
        med = range(entities) if free_mediators else (k,)
        nodes += [
            Node(f"l{k}", EDGE, tuple(Candidate(r.resolve("tuple", (E, E)), ("q0", f"m{j}")) for j in med), "sets", 2),
            Node(f"rr{k}", EDGE, tuple(Candidate(r.resolve("tuple", (E, E)), (f"m{j}", "q1")) for j in med), "sets", 2),
            Node(f"L{k}", BOOL, (Candidate(r.resolve("member", (EDGES, EDGE)), ("edges", f"l{k}")),), "sets", 3),
            Node(f"R{k}", BOOL, (Candidate(r.resolve("member", (EDGES, EDGE)), ("edges", f"rr{k}")),), "sets", 3),
        ]
    pp = pool if free_p else (AND,)
    po = pool if free_o else (OR,)
    for k in range(entities):
        nodes.append(Node(f"P{k}", BOOL, tuple(Candidate(r.resolve(f"truth_{t}", (BOOL, BOOL)), (f"L{k}", f"R{k}"))
                                               for t in pp), "logic", 4))
    prev = "D"
    for k in range(entities):
        nodes.append(Node(f"O{k}", BOOL, tuple(Candidate(r.resolve(f"truth_{t}", (BOOL, BOOL)), (prev, f"P{k}"))
                                               for t in po), "logic", 5 + k))
        prev = f"O{k}"
    ports = (("edges", EDGES), ("query", EDGE)) + tuple((f"m{k}", E) for k in range(entities))
    return Program(ports, tuple(nodes), (("target", prev),), consts)


def signals(entities=ENTITIES, dense=False):
    last = f"O{entities - 1}"
    out = [Signal(last, "target", ("logic",), BOOL, "bce")]
    if dense:
        for k in range(entities):
            out.append(Signal(f"P{k}", f"P{k}", ("logic",), BOOL, "bce", weight=1.))
            if f"O{k}" != last:
                out.append(Signal(f"O{k}", f"O{k}", ("logic",), BOOL, "bce", weight=1.))
    return tuple(out)


def closure_wall():
    """Show that no node can carry the `closure` latent's type."""
    r = Registry(); notes = {}
    j = r.resolve("join", (EDGES, EDGES), None, {"left": 1, "right": 0})
    notes["join_output"] = {"capacity": j.output.capacity, "width": j.output.width, "gradient": j.gradient}
    for name, args in (("union", (EDGES, j.output)), ("intersection", (EDGES, j.output))):
        try: r.resolve(name, args); notes[name] = "legal"
        except Exception as exc: notes[name] = f"{type(exc).__name__}: {exc}"
    for conv in ("encode", "decode", "quantize", "dequantize", "pack", "unpack"):
        try: r.resolve(conv, (j.output,), EDGES); notes[conv] = "legal"
        except Exception as exc: notes[conv] = f"{type(exc).__name__}: {exc}"
    # and the probe itself
    p = reachability_program(Registry())
    try:
        p.validate_signals((Signal("O3", "closure", ("logic",), EDGES, "mse"),))
        notes["probe_closure_on_bool_node"] = "legal"
    except Exception as exc:
        notes["probe_closure_on_bool_node"] = f"{type(exc).__name__}: {exc}"
    notes["set_operator_gradients"] = {
        n: r.resolve(n, (EDGES, EDGE) if n in {"member", "insert", "remove"} else (EDGES, EDGES),
                     None, {"left": 0, "right": 0} if n == "join" else None).gradient
        for n in ("member", "insert", "remove", "union", "intersection", "pair", "join")}
    return notes
