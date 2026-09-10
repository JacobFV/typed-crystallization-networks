"""Step 1: where does a width actually enter, and where is it compared?

Every site below is cited `file:line` against the working tree and, where it is
load-bearing, demonstrated executably: a depth-1-typed artifact is handed a
depth-2 observation and the exact exception is recorded verbatim.

    .venv/bin/python research/depth-encoding/audit.py

Nothing in `tcn/` or `generators/` is modified. Line numbers are resolved at run
time by matching the recorded source text, so a citation cannot silently drift.
"""
from __future__ import annotations

import json
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.generation import Host
from tcn.graph import Program, Node, Candidate, legal_candidates, operator_parameters
from tcn.library import Library
from tcn.operators import Registry
from tcn.types import (BOOL, Type, Value, floating, integer, product, setof,
                       encode, validate_raw)

OUT = Path(__file__).parent / "out"
F = floating()
IDX = integer(8, signed=False)

# The exact source text of each site, so the citation is resolved by matching
# rather than by a hand-copied line number that can drift.
SITES = [
    # (file, unique source fragment, what it does, class)
    ("tcn/types.py", "items: tuple[Type, ...] = ()",
     "`Type` is a frozen dataclass, so `==` is structural over every field; "
     "`items` carries a tuple's arity and `capacity` a set's bound. This is the "
     "single root of every comparison below.", "definition"),
    ("tcn/types.py", "capacity: int = 0",
     "A set's declared bound is a field of its type, so two capacities are two "
     "types.", "definition"),
    ("tcn/types.py", "return self.capacity*(1+self.items[0].width)",
     "`Type.width` — the flat relaxation width, derived from arity/capacity.",
     "derivation"),
    ("tcn/types.py", 'raise ValueError("tuple arity mismatch")',
     "`encode` refuses a value whose arity disagrees with the declared tuple.",
     "value equality"),
    ("tcn/types.py", 'raise TypeError("invalid tuple")',
     "`validate_raw` — every `Value` construction re-checks arity.",
     "value equality"),
    ("tcn/types.py", 'raise OverflowError("set capacity exceeded")',
     "`encode` refuses a set larger than the declared capacity.",
     "value equality"),
    ("tcn/graph.py", 'raise TypeError("candidate input mismatch")',
     "`Program.validate` — a candidate's declared operator inputs must equal the "
     "types of the ports it reads. This is where a depth-1 candidate spliced "
     "into a depth-2 scaffold dies.", "type equality"),
    ("tcn/graph.py", 'raise TypeError("mixed output types")',
     "`Program.validate` — every candidate at a node must agree with the node's "
     "output type, so a width-carrying output pins the node.", "type equality"),
    ("tcn/graph.py", 'raise TypeError("operator contract drift")',
     "`Program.validate` re-resolves each operator from its own dict and demands "
     "the identical `Operator`; widths inside `inputs`/`output` are part of that.",
     "type equality"),
    ("tcn/graph.py", 'raise TypeError("state update mismatch")',
     "`Program.validate` — a recurrence carrier's type is fixed, width included.",
     "type equality"),
    ("tcn/graph.py", 'raise TypeError("probe signature mismatch")',
     "`validate_signals` — a supervision signal's type must equal the port's.",
     "type equality"),
    ("tcn/graph.py", 'raise TypeError("input representation mismatch")',
     "`Program.execute` — **the runtime gate**. A frozen program handed a "
     "`program` observation of another depth fails here, before any node runs.",
     "type equality"),
    ("tcn/graph.py", 'raise TypeError("state type mismatch")',
     "`Program.execute` — the same for a supplied state carrier.",
     "type equality"),
    ("tcn/graph.py", 'return [{"index":i} for i in range(len(types[0].items))]',
     "`operator_parameters` — the `project` parameter family is derived from the "
     "tuple's arity, so the *search space itself* is width-derived.",
     "derivation"),
    ("tcn/operators.py", "require(ts == tuple(t for _,t in m.inputs))",
     "`Registry.resolve`, module branch — **§30's error**. A hardened module's "
     "input type names the observation, so it cannot be resolved at another "
     "width.", "type equality"),
    ("tcn/operators.py", 'require(len(ts)==1 and ts[0].kind=="tuple" and 0<=p.get("index",-1)<len(ts[0].items))',
     "`project` — the index bound is the arity, so a `project index=3` operator "
     "is illegal on a 3-field tuple.", "type equality"),
    ("tcn/operators.py", 'require(bool(ts[0].items) and len(set(ts[0].items))==1)',
     "`index` — dynamic tuple indexing needs a uniform tuple; its *input* type "
     "still carries the arity even though its output does not.", "type equality"),
    ("tcn/operators.py", 'require(len(ts)==2 and ts[0].kind=="set" and ts[0].items[0]==ts[1])',
     "`member`/`insert`/`remove` — element type equality, capacity carried in "
     "`ts[0]`.", "type equality"),
    ("tcn/operators.py", 'require(len(ts)==2 and ts[0]==ts[1] and ts[0].kind=="set")',
     "`union`/`intersection` — full set-type equality, **capacity included**.",
     "type equality"),
    ("tcn/operators.py", "inferred=setof(product(ts[0].items[0],ts[1].items[0]), ts[0].capacity*ts[1].capacity)",
     "`pair`/`join` — the output capacity is the product of the inputs', so a "
     "capacity change propagates downstream as a type change.", "derivation"),
    ("tcn/operators.py", "require(tuple(t for _,t in m.inputs)==ts[0].items)",
     "`map`/`filter` — the mapped module's input type must equal the set's "
     "element type.", "type equality"),
    ("tcn/operators.py", 'inferred=ts[0] if name=="filter" else setof(outs[0],ts[0].capacity)',
     "`map` — **the declared capacity is copied into the output type**, so a "
     "`map` at capacity 8 and one at capacity 16 are different operators with "
     "different output types.", "derivation"),
    ("tcn/operators.py", "cost=max(1,ts[0].capacity)*m.execution_cost(self)",
     "`map`/`filter` — declared capacity is also the charged execution cost, so "
     "it is not a free parameter to raise.", "derivation"),
    ("tcn/operators.py", 'require(output is None or output==inferred, "output signature mismatch")',
     "`Registry.resolve` — the final gate; a requested output type must equal "
     "the inferred one, widths included.", "type equality"),
    ("tcn/operators.py", 'raise TypeError("input type mismatch")',
     "`Registry.exact` — re-checked at every single exact execution.",
     "type equality"),
    ("tcn/learning.py", 'raise TypeError("input width mismatch")',
     "`SoftProgram` — the relaxed path checks the flat width, the numeric shadow "
     "of the same fact.", "type equality"),
    ("tcn/compile.py", "\"    if len(x) != %d: raise ValueError('tuple arity mismatch')\" % len(t.items)",
     "`compile.py` bakes the arity into the generated boundary check, so a "
     "compiled artifact is width-specific in emitted source too.", "derivation"),
    ("tcn/compile.py", "\"if len(%s) > %d: raise OverflowError('set capacity exceeded')\" % (dst, out.capacity)",
     "the compiled backend likewise bakes the declared set capacity.",
     "derivation"),
    ("tcn/library.py", 'inputs=tuple((k, t.to_dict()) for k, t in stored.inputs),',
     "`Library.publish` — a stored entry records the module's input **type "
     "dicts**, which contain `items`/`capacity`; two widths are two entries with "
     "two digests.", "storage"),
    ("generators/logic/generator.py", "'program':vector_value([x for g in s['gates'] for x in g])",
     "**where the width is created**: `program` is a tuple of `3 × depth` "
     "scalars, so depth moves the type.", "origin"),
    ("generators/logic/generator.py", "if s.get('gate_capacity') is not None: obs['gates']=gate_set_value(s['gates'],s['gate_capacity'])",
     "§15's second typed view: one `set` type at a declared capacity, emitted "
     "only when configured. Depth moves cardinality, not width.", "origin"),
]


def cite(rel, fragment):
    text = (ROOT / rel).read_text().splitlines()
    hits = [i + 1 for i, line in enumerate(text) if fragment in line]
    if not hits:
        raise AssertionError(f"{rel}: no match for {fragment!r}")
    return {"file": rel, "line": hits[0], "lines": hits,
            "cite": f"{rel}:{','.join(map(str, hits))}",
            "source": text[hits[0] - 1].strip()}


def caught(fn):
    """Run `fn`, returning the exception type, message, and raising file:line."""
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the exception *is* the datum
        frame = traceback.extract_tb(exc.__traceback__)[-1]
        rel = Path(frame.filename)
        try:
            rel = rel.relative_to(ROOT)
        except ValueError:
            pass
        return {"raised": True, "type": type(exc).__name__, "message": str(exc),
                "at": f"{rel}:{frame.lineno}"}
    return {"raised": False}


# ---------------------------------------------------------------------------
# executable demonstrations
# ---------------------------------------------------------------------------
BASE = {"inputs": 4, "nondegenerate": True, "min_relevant_inputs": 2}


def observation(depth):
    host = Host.create("logic", seed=0, index=0, split="test",
                       configuration=dict(BASE) | {"depth": depth, "horizon": 4,
                                                   "gate_capacity": 8})
    return host.view().observations


def tiny_program(ptype):
    """The smallest honest artifact typed for one `program` width: read field 0."""
    r = Registry()
    op = r.resolve("project", (ptype,), parameters={"index": 0})
    nodes = (Node("f0", op.output, (Candidate(op, ("program",)),), "core", 1, 0),)
    return Program((("program", ptype),), nodes, (("out", "f0"),),
                   input_depths=(("program", 0),)).validate(r), r


def demonstrations():
    d = {}
    o1, o2 = observation(1), observation(2)
    p1, p2 = o1["program"].type, o2["program"].type
    d["observation_types"] = {
        "depth_1_program_fields": len(p1.items), "depth_1_flat_width": p1.width,
        "depth_2_program_fields": len(p2.items), "depth_2_flat_width": p2.width,
        "types_equal": p1 == p2,
        "gates_type_equal": o1["gates"].type == o2["gates"].type,
        "gates_capacity": o1["gates"].type.capacity,
        "gates_flat_width": o1["gates"].type.width,
        "gates_cardinality_d1": len(o1["gates"].raw),
        "gates_cardinality_d2": len(o2["gates"].raw),
    }

    prog1, r1 = tiny_program(p1)

    # 1. the runtime gate: a depth-1 artifact handed a depth-2 observation.
    d["execute_wrong_width"] = caught(
        lambda: prog1.execute({"program": o2["program"]}, registry=r1))
    d["execute_right_width"] = caught(
        lambda: prog1.execute({"program": o1["program"]}, registry=r1))

    # 2. validate: splice a depth-1 candidate into a depth-2-typed port.
    def splice():
        op = r1.resolve("project", (p1,), parameters={"index": 0})
        nodes = (Node("f0", op.output, (Candidate(op, ("program",)),), "core", 1, 0),)
        Program((("program", p2),), nodes, (("out", "f0"),),
                input_depths=(("program", 0),)).validate(r1)
    d["validate_spliced_candidate"] = caught(splice)

    # 3. §30 on the width axis: a hardened module resolved at another width.
    r = Registry()
    name1 = r.register_module(tiny_program(p1)[0].harden({"f0": 0}))
    name2 = r.register_module(tiny_program(p2)[0].harden({"f0": 0}))
    d["module_digests_differ"] = {"depth_1": name1, "depth_2": name2,
                                 "equal": name1 == name2}
    d["module_resolved_at_other_width"] = caught(lambda: r.resolve(name1, (p2,)))
    d["module_resolved_at_own_width"] = caught(lambda: r.resolve(name1, (p1,)))

    # 4. project's index bound is the arity.
    d["project_index_past_arity"] = caught(
        lambda: r.resolve("project", (p1,), parameters={"index": 3}))
    d["project_parameter_family_size"] = {
        "depth_1": len(operator_parameters(r, "project", [p1])),
        "depth_2": len(operator_parameters(r, "project", [p2])),
    }

    # 5. map/filter: element type equality, and capacity copied into the output.
    elem = product(IDX, IDX)
    body = Program((("x", elem),),
                   (Node("y", IDX, (Candidate(r.resolve("project", (elem,), parameters={"index": 0}),
                                              ("x",)),), "core", 1, 0),),
                   (("y", "y"),), input_depths=(("x", 0),)).validate(r)
    m = r.register_module(body)
    map8 = r.resolve("map", (setof(elem, 8),), None, {"module": m})
    map16 = r.resolve("map", (setof(elem, 16),), None, {"module": m})
    d["map_capacity_in_output_type"] = {
        "capacity_8_output": map8.output.to_dict(),
        "capacity_16_output": map16.output.to_dict(),
        "outputs_equal": map8.output == map16.output,
        "cost_8": map8.cost, "cost_16": map16.cost,
    }
    d["map_wrong_element_width"] = caught(
        lambda: r.resolve("map", (setof(product(IDX, IDX, IDX), 8),), None, {"module": m}))
    d["union_across_capacities"] = caught(
        lambda: r.resolve("union", (setof(elem, 8), setof(elem, 16))))

    # 6. Registry.exact re-checks at every execution.
    op = r.resolve("project", (p1,), parameters={"index": 0})
    d["exact_wrong_width"] = caught(lambda: r.exact(op, [o2["program"]]))

    # 7. Value construction: arity and capacity.
    d["encode_wrong_arity"] = caught(lambda: encode(p1, tuple(range(len(p2.items)))))
    d["validate_raw_wrong_arity"] = caught(
        lambda: validate_raw(p1, tuple(0.0 for _ in p2.items)))
    d["encode_over_capacity"] = caught(
        lambda: Value.of(setof(IDX, 2), (1, 2, 3)))

    # 8. the soft/relaxed path.
    from tcn.learning import SoftProgram
    soft = SoftProgram(prog1, r1)
    d["soft_wrong_width"] = caught(lambda: soft.forward({"program": o2["program"]}))

    # 9. the library: a stored module is a stored width.
    with tempfile.TemporaryDirectory() as tmp:
        lib = Library(Path(tmp))
        e1 = lib.publish("reader", tiny_program(p1)[0].harden({"f0": 0}), Registry())
        e2 = lib.publish("reader", tiny_program(p2)[0].harden({"f0": 0}), Registry())
        loaded = Registry()
        loaded, _aliases = lib.load(["reader@1"])
        d["library"] = {
            "entry_1_digest": e1.digest, "entry_2_digest": e2.digest,
            "same_name_two_versions": (e1.version, e2.version),
            "stored_input_types_equal": e1.inputs == e2.inputs,
            "loaded_at_other_width": caught(lambda: loaded.resolve(e1.operator, (p2,))),
            "loaded_at_own_width": caught(lambda: loaded.resolve(e1.operator, (p1,))),
        }

    # 10. the compiled backend.
    from tcn.compile import compile_program
    src = compile_program(tiny_program(p1)[0].harden({"f0": 0}), Registry())
    text = src if isinstance(src, str) else getattr(src, "source", str(src))
    d["compiled_bakes_arity"] = {
        "mentions_depth_1_arity": ("!= %d" % len(p1.items)) in text
                                  or ("len(x) != %d" % len(p1.items)) in text,
        "arity": len(p1.items),
    }
    return d


def main():
    OUT.mkdir(exist_ok=True)
    report = {
        "sites": [dict(cite(f, frag), what=what, klass=k) for f, frag, what, k in SITES],
        "demonstrations": demonstrations(),
    }
    counts = {}
    for s in report["sites"]:
        counts[s["klass"]] = counts.get(s["klass"], 0) + 1
    report["site_counts"] = counts
    (OUT / "audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
