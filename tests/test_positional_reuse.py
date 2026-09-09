"""Positional reuse: one crystallized module applied across many positions.

`map` and `filter` need a `set`, and every perceptual observation is a `tuple`,
so the question is whether a wide tuple can be turned into an iterable of
positions without naming every position. It can, with operators already in the
registry:

    held    = insert(empty_set_constant, wide_tuple)   set[Wide], capacity 1
    records = pair(position_constant_set, held)        set[(Index, Wide)]
    mapped  = map(records; module = m)                 set[(Index, Out)]

`pair` is the cartesian product, so `records` holds one record per position, each
carrying the whole observation, and the module reads its own position out of the
record. Three caller nodes for any number of positions.

These tests pin that behaviour, the cost accounting that makes it worth doing,
and the type legality that keeps it honest.
"""
import pytest

from tcn.graph import Candidate, Node, Program, legal_candidates, operator_parameters
from tcn.operators import Registry
from tcn.scaffold import positional_scaffold
from tcn.types import BOOL, Value, floating, integer, product, setof

F = floating()
IDX = integer(16, signed=False)


def wide(width):
    return product(*(F for _ in range(width)))


def doubling_module(registry, width):
    """(Index, Wide) -> (Index, 2 * Wide[Index])."""
    W = wide(width)
    REC = product(IDX, W)
    two = Value.of(F, 2.)
    types = {"rec": REC, "two": F}
    depths = {"rec": 0, "two": -1}
    nodes = []

    def add(name, op, sources, params=None):
        o = registry.resolve(op, tuple(types[s] for s in sources), None, params)
        depth = max(depths[s] for s in sources) + 1
        nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth, 0))
        types[name] = o.output
        depths[name] = depth
        return name

    add("pos", "project", ["rec"], {"index": 0})
    add("obs", "project", ["rec"], {"index": 1})
    add("x", "index", ["obs", "pos"])
    add("y", "mul", ["x", "two"])
    add("record", "tuple", ["pos", "y"])
    return Program((("rec", REC),), tuple(nodes), (("out", "record"),), (("two", two),),
                   input_depths=(("rec", 0),)).validate(registry)


def shared_caller(registry, width, n, module_name):
    W = wide(width)
    positions = Value.of(setof(IDX, n), tuple(range(n)))
    empty = Value.of(setof(W, 1), ())
    types = {"observation": W, "positions": positions.type, "empty": empty.type}
    depths = {"observation": 0, "positions": -1, "empty": -1}
    nodes = []

    def add(name, op, sources, params=None):
        o = registry.resolve(op, tuple(types[s] for s in sources), None, params)
        depth = max(depths[s] for s in sources) + 1
        nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth, 0))
        types[name] = o.output
        depths[name] = depth
        return name

    add("held", "insert", ["empty", "observation"])
    add("records", "pair", ["positions", "held"])
    add("mapped", "map", ["records"], {"module": module_name})
    return Program((("observation", W),), tuple(nodes), (("y", "mapped"),),
                   (("positions", positions), ("empty", empty)),
                   input_depths=(("observation", 0),)).validate(registry)


def per_position_caller(registry, width, n, module_name):
    W = wide(width)
    consts = tuple((f"p{i}", Value.of(IDX, i)) for i in range(n))
    types = {"observation": W} | {k: v.type for k, v in consts}
    depths = {"observation": 0} | {k: -1 for k, _ in consts}
    nodes = []

    def add(name, op, sources, params=None):
        o = registry.resolve(op, tuple(types[s] for s in sources), None, params)
        depth = max(depths[s] for s in sources) + 1
        nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth, 0))
        types[name] = o.output
        depths[name] = depth
        return name

    calls = [add(f"call{i}", module_name, [add(f"rec{i}", "tuple", [f"p{i}", "observation"])])
             for i in range(n)]
    add("all", "tuple", calls)
    return Program((("observation", W),), tuple(nodes), (("y", "all"),), consts,
                   input_depths=(("observation", 0),)).validate(registry)


def test_one_module_covers_every_position_with_three_nodes():
    r = Registry()
    width = 5
    name = r.register_module(doubling_module(r, width))
    xs = tuple(float(i) + .5 for i in range(width))
    for n in (1, 3, width):
        caller = shared_caller(r, width, n, name)
        assert len(caller.nodes) == 3, "caller size must not depend on the position count"
        out, _ = caller.run({"observation": Value.of(wide(width), xs)}, registry=r)
        assert out["y"].decoded == frozenset((i, 2 * xs[i]) for i in range(n))


def test_shared_and_per_position_agree_and_sharing_is_cheaper_to_describe():
    r = Registry()
    width = n = 8
    name = r.register_module(doubling_module(r, width))
    xs = tuple(float(i) for i in range(width))
    x = {"observation": Value.of(wide(width), xs)}
    shared = shared_caller(r, width, n, name)
    per = per_position_caller(r, width, n, name)
    assert shared.run(x, registry=r)[0]["y"].decoded == frozenset(per.run(x, registry=r)[0]["y"].decoded)
    # The module definition is charged once in both, so the difference is the
    # caller: three nodes against two per position.
    assert len(per.nodes) == 2 * n + 1
    assert shared.description_bits(r) < per.description_bits(r)
    # Execution is charged per use either way; sharing does not make calls free.
    body = doubling_module(r, width).execution_cost(r)
    assert shared.execution_cost(r) == n * body + 2          # insert and pair
    assert per.execution_cost(r) == n * (body + 1) + 1       # a record and a tuple per call


def test_mapped_module_must_match_the_element_type():
    r = Registry()
    name = r.register_module(doubling_module(r, 4))
    wrong = setof(product(IDX, wide(3)), 4)          # module expects a width-4 observation
    with pytest.raises(TypeError):
        r.resolve("map", (wrong,), None, {"module": name})
    with pytest.raises(TypeError):
        r.resolve("map", (setof(IDX, 4),), None, {"module": name})
    with pytest.raises(TypeError):
        r.resolve("map", (product(IDX, IDX),), None, {"module": name})


def test_indexed_record_set_is_a_lossless_sequence():
    """Positions are unique, so the set determines the sequence it stands for."""
    t = setof(product(IDX, BOOL), 6)
    seq = (True, False, False, True, True, False)
    v = Value.of(t, tuple((i, seq[i]) for i in range(len(seq))))
    assert tuple(b for _, b in sorted(v.decoded)) == seq
    # Flattening sorts elements, and the unique index decides that order, so a
    # prediction and a target over the same positions stay aligned field for field.
    other = Value.of(t, tuple((i, not seq[i]) for i in range(len(seq))))
    assert [a for a, _ in sorted(v.decoded)] == [a for a, _ in sorted(other.decoded)]


def test_enumeration_can_propose_parametric_operators():
    r = Registry()
    name = r.register_module(doubling_module(r, 3))
    ports = {"rec": product(IDX, wide(3))}
    got = legal_candidates(r, ["project"], ports, None, arities=(1,))
    assert sorted(dict(c.operator.parameters)["index"] for c in got) == [0, 1]
    records = setof(product(IDX, wide(3)), 3)
    got = legal_candidates(r, ["map"], {"records": records}, None, arities=(1,))
    assert [dict(c.operator.parameters)["module"] for c in got] == [name]
    # An unregistered module can never be proposed.
    assert operator_parameters(Registry(), "map", (records,)) == []
    assert legal_candidates(Registry(), ["map"], {"records": records}, None, arities=(1,)) == ()


def negating_module(registry, width):
    """A second module with the same interface, so a node can choose between them."""
    W = wide(width)
    REC = product(IDX, W)
    types = {"rec": REC}
    depths = {"rec": 0}
    nodes = []

    def add(name, op, sources, params=None):
        o = registry.resolve(op, tuple(types[s] for s in sources), None, params)
        depth = max(depths[s] for s in sources) + 1
        nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth, 0))
        types[name] = o.output
        depths[name] = depth
        return name

    add("pos", "project", ["rec"], {"index": 0})
    add("obs", "project", ["rec"], {"index": 1})
    add("x", "index", ["obs", "pos"])
    add("y", "neg", ["x"])
    add("record", "tuple", ["pos", "y"])
    return Program((("rec", REC),), tuple(nodes), (("out", "record"),),
                   input_depths=(("rec", 0),)).validate(registry)


def test_positional_scaffold_matches_the_hand_wired_pattern():
    r = Registry()
    width = 6
    name = r.register_module(doubling_module(r, width))
    xs = tuple(float(i) - 2 for i in range(width))
    x = {"observation": Value.of(wide(width), xs)}
    for n in (1, 4, width):
        built = positional_scaffold(r, wide(width), range(n), [name], index=IDX)
        assert len(built.nodes) == 3
        assert built.run(x, registry=r)[0]["mapped"].decoded == \
            shared_caller(r, width, n, name).run(x, registry=r)[0]["y"].decoded


def test_positional_scaffold_offers_a_choice_of_shared_modules():
    r = Registry()
    width = 4
    a = r.register_module(doubling_module(r, width))
    b = r.register_module(negating_module(r, width))
    prog = positional_scaffold(r, wide(width), range(width), [a, b], index=IDX)
    node = next(n for n in prog.nodes if n.name == "mapped")
    assert len(node.candidates) == 2 and node.selected is None
    xs = tuple(float(i) + 1 for i in range(width))
    x = {"observation": Value.of(wide(width), xs)}
    assert prog.harden({"mapped": 0}).run(x, registry=r)[0]["mapped"].decoded == \
        frozenset((i, 2 * xs[i]) for i in range(width))
    assert prog.harden({"mapped": 1}).run(x, registry=r)[0]["mapped"].decoded == \
        frozenset((i, -xs[i]) for i in range(width))


def test_positional_scaffold_refuses_illegal_configurations():
    r = Registry()
    name = r.register_module(doubling_module(r, 4))
    with pytest.raises(TypeError):
        positional_scaffold(r, F, range(4), [name], index=IDX)             # not a tuple
    with pytest.raises(ValueError):
        positional_scaffold(r, wide(4), (), [name], index=IDX)             # no positions
    with pytest.raises(ValueError):
        positional_scaffold(r, wide(4), range(4), [], index=IDX)           # no module
    with pytest.raises(TypeError):
        positional_scaffold(r, wide(5), range(4), [name], index=IDX)       # width mismatch
    with pytest.raises(TypeError):
        positional_scaffold(r, wide(4), range(4), [name],
                            index=integer(16, signed=False, role="category"))
    # Modules that disagree on their interface cannot share one node.
    REC = product(IDX, wide(4))
    project0 = r.resolve("project", (REC,), None, {"index": 0})
    position_only = r.register_module(
        Program((("rec", REC),), (Node("only", IDX, (Candidate(project0, ("rec",)),), "core", 1, 0),),
                (("out", "only"),), input_depths=(("rec", 0),)).validate(r))
    with pytest.raises(TypeError, match="share an output interface"):
        positional_scaffold(r, wide(4), range(4), [name, position_only], index=IDX)


def test_enumeration_budget_still_bounds_parametric_families():
    r = Registry()
    ports = {"rec": product(IDX, wide(64))}
    with pytest.raises(ValueError):
        legal_candidates(r, ["project"], ports, None, arities=(1,), limit=1)
    # A caller may narrow a parametric family without changing the operator.
    got = legal_candidates(r, ["project"], ports, None, arities=(1,),
                           parameters={"project": [{"index": 1}]})
    assert len(got) == 1 and got[0].operator.output == wide(64)
