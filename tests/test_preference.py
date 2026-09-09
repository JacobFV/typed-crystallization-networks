"""Preferring the cheaper program: description cost, dead-node pruning, ranked enumeration.

Reuse is a capability until something in the objective prefers it. These cover
the four pieces that make preference expressible: row deduplication in exact
execution (which must stay bit-identical), a differentiable description-length
term (which must equal the real quantity where the two are comparable), pruning
(which must preserve semantics), and cost-ranked enumeration.
"""
import itertools
import json
import math
import pytest
import torch

from tcn.graph import Program, Node, Candidate, Signal
from tcn.learning import SoftProgram, exact_tensor, tensor
from tcn.operators import Registry
from tcn.runtime import save_program, load_program
from tcn.search import enumerate_fit, program_cost
from tcn.synthesis import fit
from tcn.types import BOOL, Value, floating, integer


def _and_module(r):
    body = Program((("a", BOOL), ("b", BOOL)),
                   (Node("g", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0),),
                   (("out", "g"),)).validate(r)
    return r.register_module(body)


def _maj_module(r):
    """MAJ3 = and(or(a,b), or(c, and(a,b))), the minimal 4-gate circuit."""
    spec = (("and", "g0", "a", "b"), ("or", "g1", "a", "b"), ("or", "g2", "c", "g0"), ("and", "g3", "g1", "g2"))
    depth = {"a": 0, "b": 0, "c": 0}; nodes = []
    for name, out, x, y in spec:
        d = max(depth[x], depth[y]) + 1; depth[out] = d
        nodes.append(Node(out, BOOL, (Candidate(r.resolve(name, (BOOL, BOOL)), (x, y)),), "core", d, 0))
    body = Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes), (("out", "g3"),)).validate(r)
    return r.register_module(body), body


# --------------------------------------------------------------------------
# R1: exact_tensor evaluates each distinct row once
# --------------------------------------------------------------------------
def _reference(registry, op, xs):
    """The undeduplicated computation, row by row, as the baseline for identity."""
    shape = torch.broadcast_shapes(*(x.shape[:-1] for x in xs))
    batch = math.prod(shape) if shape else 1
    flat = [x.detach().expand(*shape, x.shape[-1]).reshape(batch, -1).cpu().tolist() for x in xs]
    vals = [registry.exact(op, [Value.unflat(t, x[i]) for t, x in zip(op.inputs, flat)]).flat() for i in range(batch)]
    return torch.tensor(vals, dtype=torch.float32).reshape(*shape, op.output.width)


def test_row_deduplication_is_bit_identical_on_the_boolean_path():
    r = Registry(); name, _ = _maj_module(r)
    op = r.resolve(name, (BOOL, BOOL, BOOL))
    rows = list(itertools.product((0., 1.), repeat=6))          # every argument row repeats 8x
    xs = [torch.tensor([[row[i]] for row in rows]) for i in range(3)]
    got = exact_tensor(r, op, xs)
    assert torch.equal(got, _reference(r, op, xs))
    assert len({tuple(int(x[i, 0]) for x in xs) for i in range(len(rows))}) == 8


def test_row_deduplication_is_bit_identical_on_the_float_path():
    r = Registry(); F = floating()
    op = r.resolve("atan2", (F, F))
    torch.manual_seed(0)
    xs = [torch.randn(64, 1), torch.randn(64, 1).abs() + .5]     # 64 distinct rows: the worst case
    got = exact_tensor(r, op, xs)
    assert torch.equal(got, _reference(r, op, xs))


def test_signed_zero_does_not_collide_in_the_row_cache():
    """`-0.0 == 0.0` but `atan2` distinguishes them, so such a call skips the cache."""
    r = Registry(); F = floating()
    op = r.resolve("atan2", (F, F))
    xs = [torch.tensor([[0.], [0.]]), torch.tensor([[0.], [-0.]])]
    got = exact_tensor(r, op, xs)
    assert torch.equal(got, _reference(r, op, xs))
    assert float(got[0, 0]) == 0. and abs(float(got[1, 0]) - math.pi) < 1e-6


# --------------------------------------------------------------------------
# R2: the description-cost term
# --------------------------------------------------------------------------
def _wide_bool_scaffold(r, module_name=None, width=4):
    ports = {"a": BOOL, "b": BOOL, "c": BOOL}
    nodes = []
    for i in range(width):
        cands = [Candidate(r.resolve(op, (BOOL, BOOL)), (p, q))
                 for op in ("and", "or", "xor") for p, q in itertools.product(ports, repeat=2)]
        if module_name is not None:
            mop = r.resolve(module_name, tuple(BOOL for _ in range(3)))
            cands += [Candidate(mop, t) for t in itertools.product(("a", "b", "c"), repeat=3)]
        nodes.append(Node(f"n{i}", BOOL, tuple(cands), "core", i + 1))
        ports = dict(ports, **{f"n{i}": BOOL})
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes), (("out", f"n{width-1}"),)).validate(r)


def test_description_cost_equals_the_real_quantity_at_every_one_hot_point():
    r = Registry(); name, _ = _maj_module(r)
    for program in (_wide_bool_scaffold(r), _wide_bool_scaffold(r, name)):
        for seed in range(4):
            torch.manual_seed(seed)
            m = SoftProgram(program, r)
            with torch.no_grad():
                for p in m.choices: p.add_(torch.randn_like(p) * 400.)
            truth = m.export().pruned().description_bits(r)
            assert abs(float(m.description_cost().detach()) - truth) < 1., (seed, truth)


def test_description_cost_charges_a_module_definition_once_however_many_call_sites():
    """ARCHITECTURE section 4: sharing counts a definition once plus its call sites."""
    r = Registry(); name, body = _maj_module(r)
    mop = r.resolve(name, (BOOL, BOOL, BOOL))
    call = Candidate(mop, ("a", "b", "c"))
    sizes = {}
    for calls in (1, 2, 3):
        nodes = [Node(f"n{i}", BOOL, (call,), "core", 1, 0) for i in range(calls)]
        top = nodes[0].name
        for i, n in enumerate(nodes[1:], start=1):
            nodes.append(Node(f"x{i}", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), (top, n.name)),), "core", i + 1, 0))
            top = f"x{i}"
        p = Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), tuple(nodes), (("out", top),)).validate(r)
        sizes[calls] = SoftProgram(p, r).description_cost().item()
    definition = float(body.description_bits())
    # a second call site adds its own node, never a second copy of the definition
    assert sizes[2] - sizes[1] < definition and sizes[3] - sizes[2] < definition
    assert abs((sizes[3] - sizes[2]) - (sizes[2] - sizes[1])) < 400.


def test_description_cost_is_differentiable_and_falls_when_a_node_dies():
    r = Registry()
    program = _wide_bool_scaffold(r, width=3)
    m = SoftProgram(program, r)
    cost = m.description_cost()
    cost.backward()
    assert any(p.grad is not None and torch.any(p.grad != 0) for p in m.choices)
    with torch.no_grad():                        # make the output ignore every earlier node
        idx = next(i for i, c in enumerate(program.nodes[-1].candidates) if c.sources == ("a", "b"))
        m.choices[-1][idx] += 100.
    assert float(m.description_cost().detach()) < float(cost.detach())


def test_complexity_cannot_distinguish_what_description_cost_can():
    """A module call is at exact execution-cost parity with its inlined body."""
    r = Registry(); name = _and_module(r)
    mop = r.resolve(name, (BOOL, BOOL)); prim = r.resolve("and", (BOOL, BOOL))
    assert mop.cost == prim.cost
    costs, bits = [], []
    for op in (prim, mop):
        p = Program((("a", BOOL), ("b", BOOL)),
                    (Node("y", BOOL, (Candidate(op, ("a", "b")),), "core", 1, 0),), (("out", "y"),)).validate(r)
        m = SoftProgram(p, r)
        costs.append(float(m.complexity())); bits.append(float(m.description_cost()))
    assert costs[0] == costs[1]
    assert bits[0] < bits[1]


def test_fit_defaults_the_description_term_off_and_accepts_a_weight():
    from examples.mixed import problem
    p, signals, examples = problem()
    _, off = fit(p, examples, signals, steps=20, polish=0, freeze=False)
    assert off['mdl_weight'] == 0.
    _, on = fit(p, examples, signals, steps=20, polish=0, freeze=False, mdl_weight=1e-5)
    assert on['mdl_weight'] == 1e-5
    assert on['description_bits'] > 0 and on['pruned_description_bits'] > 0
    with pytest.raises(ValueError):
        fit(p, examples, signals, steps=1, mdl_weight=-1.)


# --------------------------------------------------------------------------
# R3: pruning dead nodes
# --------------------------------------------------------------------------
def test_pruning_preserves_semantics_and_drops_only_dead_nodes():
    r = Registry()
    dead = Node("dead", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0)
    live = Node("live", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0)
    p = Program((("a", BOOL), ("b", BOOL)), (dead, live), (("out", "live"),)).validate(r)
    q = p.pruned().validate(r)
    assert [n.name for n in q.nodes] == ["live"]
    for a, b in itertools.product((False, True), repeat=2):
        row = {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)}
        assert p.run(row, registry=r)[0]["out"].flat() == q.run(row, registry=r)[0]["out"].flat()
    assert q.description_bits(r) < p.description_bits(r)
    assert q.execution_cost(r) < p.execution_cost(r)


def test_pruning_a_soft_scaffold_keeps_every_node_any_candidate_can_reach():
    r = Registry()
    program = _wide_bool_scaffold(r, width=3)
    assert len(program.pruned().nodes) == len(program.nodes)


def test_registration_prunes_a_module_so_dead_gates_are_not_charged_at_call_sites():
    r = Registry()
    gates = (Node("g0", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0),
             Node("junk", BOOL, (Candidate(r.resolve("or", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0))
    body = Program((("a", BOOL), ("b", BOOL)), gates, (("out", "g0"),)).validate(r)
    name = r.register_module(body)
    kept = r.modules[name]
    assert [n.name for n in kept.nodes] == ["g0"]
    assert r.resolve(name, (BOOL, BOOL)).cost == 1.0          # not 2.0
    for a, b in itertools.product((False, True), repeat=2):
        row = {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)}
        assert kept.run(row, registry=r)[0]["out"].decoded == body.run(row, registry=r)[0]["out"].decoded


def test_save_program_exports_the_program_not_the_scaffold(tmp_path):
    r = Registry()
    dead = Node("dead", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0)
    live = Node("live", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0)
    p = Program((("a", BOOL), ("b", BOOL)), (dead, live), (("out", "live"),)).validate(r)
    path = save_program(p, tmp_path / "p.json", r)
    back, _ = load_program(path)
    assert [n.name for n in back.nodes] == ["live"]
    assert back.digest == p.pruned().digest


# --------------------------------------------------------------------------
# ranked enumeration
# --------------------------------------------------------------------------
def _equivalent_choice_problem(module_first):
    r = Registry(); name = _and_module(r)
    mop = r.resolve(name, (BOOL, BOOL)); prim = r.resolve("and", (BOOL, BOOL))
    order = (Candidate(mop, ("a", "b")), Candidate(prim, ("a", "b")))
    if not module_first: order = order[::-1]
    p = Program((("a", BOOL), ("b", BOOL)), (Node("y", BOOL, order, "core", 1),), (("out", "y"),)).validate(r)
    ex = [{"inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)}, "targets": {"out": Value.of(BOOL, a and b)}}
          for a in (False, True) for b in (False, True)]
    return p, ex, (Signal("y", "out", ("core",), BOOL, "bce"),), r


def test_enumeration_ranked_by_description_ignores_declaration_order():
    for module_first in (True, False):
        p, ex, sig, r = _equivalent_choice_problem(module_first)
        by_order = enumerate_fit(p, ex, sig, registry=r)
        by_bits = enumerate_fit(p, ex, sig, registry=r, rank='description')
        assert by_order.conforming == 2 and by_bits.conforming == 2
        first = p.nodes[0].candidates[by_order.selections['y']].operator.name
        best = p.nodes[0].candidates[by_bits.selections['y']].operator.name
        assert first.startswith('module:') is module_first    # order decides the unranked answer
        assert best == 'and'                                  # bits decide the ranked one
        assert by_bits.description_bits < by_order.description_bits or not module_first
        assert by_bits.ranked_by == 'description'


def test_ranking_requires_the_full_conforming_set():
    p, ex, sig, r = _equivalent_choice_problem(True)
    with pytest.raises(ValueError):
        enumerate_fit(p, ex, sig, registry=r, rank='description', stop_at_first=True)
    with pytest.raises(ValueError):
        enumerate_fit(p, ex, sig, registry=r, rank='smallest')


def test_reported_cost_is_measured_after_pruning():
    p, ex, sig, r = _equivalent_choice_problem(False)
    res = enumerate_fit(p, ex, sig, registry=r, rank='cost')
    bits, cost = program_cost(p, res.selections, r)
    assert res.description_bits == bits and res.execution_cost == cost
    assert cost == 1.0
