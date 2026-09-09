"""Which fixed choices are gradient boundaries, and which temperature does what.

Two defects are pinned here, both of which silently produced wrong measurements
before they were found (FINDINGS sections 16, 18 and 19):

D1  a node carrying `Node.selected` was evaluated as `exact_tensor(...).detach()`,
    so declaring a fixed operator -- the natural way to write plumbing whose
    choice is already decided -- severed the gradient to everything upstream.
D2  one per-node temperature drove both the candidate softmax and the operator
    relaxation, so no surrogate could be widened without flattening that node's
    choice distribution.
D3  `relaxed`'s `tuple` branch was a bare `torch.cat`, which does not broadcast,
    so packing a batched intermediate alongside an unbatched trainable constant
    raised where every arithmetic operator on the same pair succeeds.
"""
import math
import pytest
import torch

from tcn.crystallize import Crystallizer
from tcn.graph import Candidate, Node, Program
from tcn.learning import SoftProgram, carrier_temperature, relaxed
from tcn.operators import Registry
from tcn.types import BOOL, Value, fixed, floating, integer, product, setof

r = Registry()
F = floating()
BYTE = integer(8, signed=False, role="byte")
UBYTE = integer(8, signed=False)


def pipe_program(selected, extra=False):
    """`k -> identity -> mul(., x)`, with `k` trainable and the pipe node's choice fixed."""
    ops = [r.resolve("identity", (F,))] + ([r.resolve("neg", (F,))] if extra else [])
    nodes = (Node("pipe", F, tuple(Candidate(o, ("k",)) for o in ops), "core", 1, selected),
             Node("y", F, (Candidate(r.resolve("mul", (F, F)), ("pipe", "x")),), "core", 2))
    return Program((("x", F),), nodes, (("out", "y"),),
                   (("k", Value.of(F, 3.0)),), trainable_constants=("k",))


def constant_gradient(model):
    out, _ = model({"x": torch.tensor([1.0])})
    k = model.constants["k"]
    if not (k.requires_grad and out["out"].requires_grad):
        return None, out["out"]
    return torch.autograd.grad(out["out"].sum(), k, allow_unused=True)[0], out["out"]


# --------------------------------------------------------------------- D1

def test_a_declared_selection_does_not_sever_the_upstream_gradient():
    """The exact reproduction: `tensor([1.])` unselected, and `None` before the fix."""
    loose, _ = constant_gradient(SoftProgram(pipe_program(None), r))
    fixed_choice, _ = constant_gradient(SoftProgram(pipe_program(0), r))
    assert loose is not None and loose.tolist() == [1.0]
    assert fixed_choice is not None, "a declared selection severed the upstream gradient"
    assert torch.equal(loose, fixed_choice)


def test_a_declared_selection_behaves_exactly_like_the_same_node_written_alone():
    """`selected=k` declares a choice; it must not change the node's semantics.

    A node with two candidates pinned to the second must forward, differentiate,
    select and export exactly as the one-candidate node holding that operator.
    """
    def two(selected):
        nodes = (Node("y", F, (Candidate(r.resolve("add", (F, F)), ("k", "x")),
                               Candidate(r.resolve("mul", (F, F)), ("k", "x"))), "core", 1, selected),)
        return Program((("x", F),), nodes, (("out", "y"),),
                       (("k", Value.of(F, 3.0)),), trainable_constants=("k",))

    def one():
        nodes = (Node("y", F, (Candidate(r.resolve("mul", (F, F)), ("k", "x")),), "core", 1),)
        return Program((("x", F),), nodes, (("out", "y"),),
                       (("k", Value.of(F, 3.0)),), trainable_constants=("k",))

    pinned = SoftProgram(two(1), r)
    plain = SoftProgram(one(), r)
    xs = {"x": torch.tensor([2.0])}
    a, _ = pinned(xs)
    b, _ = plain(xs)
    assert a["out"].tolist() == b["out"].tolist() == [6.0]
    ga = torch.autograd.grad(a["out"].sum(), pinned.constants["k"])[0]
    gb = torch.autograd.grad(b["out"].sum(), plain.constants["k"])[0]
    assert torch.equal(ga, gb)
    # The declared index survives selection and export -- the choice really is fixed.
    assert pinned.selections()["y"] == 1
    assert pinned.export().nodes[0].candidates[0].operator.name == "mul"
    assert torch.equal(pinned.distributions()[0], torch.tensor([0.0, 1.0]))
    assert float(pinned.entropy()) == 0.0


def test_a_frozen_module_call_still_stops_gradients():
    """ARCHITECTURE section 4: a crystallized module has no internal gradients.

    The boundary belongs to the operator contract (`gradient="none"`), not to the
    node's selection state, so it holds whether or not the call site is declared.
    """
    registry = Registry()
    body = Program((("a", BOOL), ("b", BOOL)),
                   (Node("z", BOOL, (Candidate(registry.resolve("xor", (BOOL, BOOL)), ("a", "b")),),
                         selected=0),), (("out", "z"),))
    name = registry.register_module(body)
    op = registry.resolve(name, (BOOL, BOOL))
    for selected in (None, 0):
        call = Program(body.inputs, (Node("call", BOOL, (Candidate(op, ("a", "b")),), selected=selected),),
                       (("result", "call"),))
        m = SoftProgram(call, registry)
        a = torch.tensor([1.0], requires_grad=True)
        out, _ = m({"a": a, "b": torch.tensor([0.0])})
        # No path from the module's output back to its input, at either selection
        # state. (Unselected, the output still carries a path to this node's own
        # choice logit -- the architecture decision, which is not internal to the
        # module -- so `requires_grad` alone is not the contract.)
        g = (torch.autograd.grad(out["result"].sum(), a, allow_unused=True)[0]
             if out["result"].requires_grad else None)
        assert g is None
    assert not out["result"].requires_grad   # declared: nothing trainable is left


def test_crystallizing_a_node_is_a_gradient_boundary_and_retires_the_declaration():
    """The third case: `freeze()` commits to the exported discrete program.

    Freezing is an irreversible commitment, so the node executes exactly and
    detached -- which is what the scheduler's connectivity guard is there to check
    has not disconnected the interior.
    """
    m = SoftProgram(pipe_program(0), r)
    assert m.pinned == {"pipe": 0} and m.frozen == {"pipe": 0}
    m.freeze("pipe", 0)
    assert "pipe" not in m.pinned and m.frozen == {"pipe": 0}
    g, out = constant_gradient(m)
    assert g is None and out.tolist() == [3.0]


def test_a_rolled_back_freeze_restores_the_declared_selection():
    m = SoftProgram(pipe_program(None), r)
    opt = torch.optim.Adam(m.parameters(), lr=.1)

    def loss():
        out, _ = m({"x": torch.tensor([1.0])})
        return (out["out"] - 99.0).square().mean()

    scheduler = Crystallizer(m, opt, tolerance=-1.)   # never accept
    event = scheduler.try_freeze("pipe", loss, retrain_steps=0)
    assert not event.accepted
    assert "pipe" not in m.frozen and "pipe" not in m.pinned
    g, _ = constant_gradient(m)
    assert g is not None


# --------------------------------------------------------------------- D2

def choice_program():
    return Program((("a", UBYTE), ("b", UBYTE)),
                   (Node("z", BOOL, tuple(Candidate(r.resolve(n, (UBYTE, UBYTE)), ("a", "b"))
                                          for n in ("eq", "lt", "le")), "core", 1),),
                   (("out", "z"),))


def test_the_choice_temperature_and_the_surrogate_temperature_are_independent():
    """Widening a surrogate must not flatten that node's candidate distribution.

    This coupling is why section 16's derived fix could not be applied: scaling
    `eq` also flattened a 256-way choice softmax, and every seed collapsed.
    """
    m = SoftProgram(choice_program(), r)
    with torch.no_grad():
        m.choices[0][0] = 1.0
    base = m.distributions()[0].detach().clone()
    xs = {"a": torch.tensor([0.0]), "b": torch.tensor([40.0])}
    narrow, _ = m(xs)

    m.surrogate_scale["z"] = 64.
    wide, _ = m(xs)
    assert torch.equal(m.distributions()[0], base)          # the choice is untouched
    assert float(m.entropy().detach()) == pytest.approx(float(-(base * base.log()).sum()))
    assert not torch.allclose(wide["out"], narrow["out"])   # the relaxation moved

    m.surrogate_scale["z"] = 1.
    m.temperatures["z"] = 64.
    flat, _ = m(xs)
    assert not torch.equal(m.distributions()[0], base)      # ...and this one moves the choice
    # The shared term still reaches the relaxation, which is what keeps the
    # crystallizer's single anneal schedule sharpening both.
    assert not torch.allclose(flat["out"], narrow["out"])


def test_the_carrier_scaling_is_opt_in_and_names_the_nodes_it_reaches():
    m = SoftProgram(choice_program(), r)
    assert m.carrier_scaled is False
    xs = {"a": torch.tensor([0.0]), "b": torch.tensor([25.6])}
    shipped, _ = m(xs)
    assert m.scale_surrogates() == ("z",)
    assert m.carrier_scaled is True
    scaled, _ = m(xs)
    assert not torch.allclose(shipped["out"], scaled["out"])


def test_eq_scaled_to_the_carrier_is_alive_past_the_float32_dead_point():
    """`eq` at tau=1 is exactly 0.0 from |a-b| >= 11 on a byte; the carrier revives it."""
    op = r.resolve("eq", (BYTE, BYTE))
    assert carrier_temperature(BYTE) == 256.

    def reading(delta, scaled):
        a = torch.tensor([0.0], requires_grad=True)
        y = relaxed(r, op, [a, torch.tensor([float(delta)])], 1., scaled)
        g = abs(float(torch.autograd.grad(y.sum(), a)[0]))
        return float(y.detach()), g

    assert reading(11, False) == (0.0, 0.0)
    assert reading(25.6, False) == (0.0, 0.0)
    v11, g11 = reading(11, True)
    v25, g25 = reading(25.6, True)
    assert v11 > .6 and g11 > 1e-2
    assert v25 == pytest.approx(7.7e-2, abs=5e-3)      # FINDINGS section 16's figure
    assert g25 > 1e-2


def test_the_carrier_scaling_cannot_move_which_candidate_is_closest_to_equal():
    """Why `eq` is the one operator a type-derived temperature is safe for.

    `exp(-||a-b||**2/tau)` is strictly decreasing in `||a-b||` at every positive
    temperature, so the candidate ranking is the distance ranking whatever the
    scale. Checked directly across the whole byte alphabet.
    """
    op = r.resolve("eq", (BYTE, BYTE))
    a = torch.arange(256.).reshape(256, 1)
    b = torch.full_like(a, 40.)
    plain = relaxed(r, op, [a, b], 1.).flatten()
    scaled = relaxed(r, op, [a, b], 1., True).flatten()
    assert int(plain.argmax()) == int(scaled.argmax()) == 40
    # The scaled surrogate ranks the whole alphabet by distance...
    assert torch.equal(torch.argsort(scaled, descending=True, stable=True),
                       torch.argsort((a.flatten() - 40).abs(), stable=True))
    # ...and agrees with the unscaled one everywhere the unscaled one is still
    # nonzero at all. Past |d| = 10 it underflows to exactly 0.0 and holds no
    # ordering to agree with, which is the defect rather than a disagreement.
    live = (plain > 0)
    assert torch.equal(torch.argsort(plain[live], descending=True, stable=True),
                       torch.argsort(scaled[live], descending=True, stable=True))
    assert int(live.sum()) == 21 and not plain[(a.flatten() - 40).abs() >= 11].any()


def test_the_ordering_comparisons_are_deliberately_not_scaled():
    """`lt`/`le`/`gt`/`ge` keep the shipped scale, and this records why.

    Their surrogate is dead past |d| >= 17, but scaling it to the carrier moves
    the loss minimum onto a wrong threshold: a live gradient pointed at the wrong
    answer is worse than a dead one. The temperature that works is the task's
    decision margin, which is not a property of the declared type, so no constant
    is derived and `SoftProgram.surrogates` is where a caller supplies one.
    """
    for name in ("lt", "le", "gt", "ge"):
        op = r.resolve(name, (UBYTE, UBYTE))
        xs = [torch.tensor([0.]), torch.tensor([48.])]
        assert torch.equal(relaxed(r, op, xs, 1.), relaxed(r, op, xs, 1., True))

    op = r.resolve("le", (UBYTE, UBYTE))
    threshold = 128
    g = torch.Generator().manual_seed(5)
    far = torch.randint(0, 256, (94, 1), generator=g).float()
    far = torch.where((far - threshold).abs() < 60, (far + 60) % 256, far)
    x = torch.cat([torch.tensor([[128.0], [129.0]]), far])
    target = (x <= threshold).float()

    def argmin(tau):
        def loss(c):
            p = relaxed(r, op, [x, torch.full_like(x, float(c))], tau).clamp(1e-6, 1 - 1e-6)
            return float(-(target * p.log() + (1 - target) * (1 - p).log()).mean())
        return min(range(256), key=loss)

    assert argmin(1.) == threshold                          # dead gradient, correct optimum
    assert argmin(carrier_temperature(UBYTE)) != threshold   # live gradient, wrong optimum


def test_index_keeps_the_shipped_scale_because_its_temperature_is_a_mixture_weight():
    """`index`'s temperature changes the value returned, so it certainly moves the optimum.

    Its recorded defect is also the other sign -- the kernel is too wide at
    tau=1, not too narrow -- and the fix for it is annealing the surrogate down,
    which the choice/surrogate split now allows on its own.
    """
    tup = product(*([BYTE] * 12))
    op = r.resolve("index", (tup, integer(32, signed=False)))
    values = torch.arange(12.0).reshape(1, 12)
    address = torch.tensor([6.0])
    assert torch.equal(relaxed(r, op, [values, address], 1.),
                       relaxed(r, op, [values, address], 1., True))
    wide = relaxed(r, op, [values, address], 1.)
    sharp = relaxed(r, op, [values, address], .1)
    assert abs(float(sharp) - 6.0) < abs(float(wide) - 6.0)


def test_carrier_temperature_is_read_off_the_declaration():
    assert carrier_temperature(BOOL) == 1.
    assert carrier_temperature(BYTE) == 256.
    assert carrier_temperature(integer(16)) == 65536.
    # A floating encoding declares no value span: its exponent range is not a
    # distance, so the width is not used and the relaxation is unchanged.
    assert carrier_temperature(floating()) == 1.
    assert carrier_temperature(fixed(16, 256)) == 1.
    # `eq` sums the squared difference over the whole width, so a composite takes
    # its widest leaf.
    assert carrier_temperature(product(BOOL, BYTE)) == 256.
    assert carrier_temperature(setof(BOOL, 4)) == 1.


def test_the_node_surrogate_scale_reaches_relaxed():
    """`SoftProgram` must pass `temperatures * surrogate_scale` and its carrier flag."""
    m = SoftProgram(choice_program(), r)
    m.scale_surrogates()
    m.surrogate_scale["z"] = 4.
    xs = {"a": torch.tensor([0.0]), "b": torch.tensor([25.6])}
    out, _ = m(xs)
    ops = [c.operator for c in m.program.nodes[0].candidates]
    direct = sum(relaxed(r, op, [xs["a"], xs["b"]], 4., True) for op in ops) / 3
    assert torch.allclose(out["out"], direct)


# --------------------------------------------------------------------- D3

def test_tuple_packs_a_batched_value_with_an_unbatched_constant():
    """The exact reproduction: an `(8,1)` intermediate packed with a `(1,)` constant.

    `torch.cat` does not broadcast, so this raised
    "Tensors must have same number of dimensions: got 2 and 1" while `add` on the
    identical pair broadcast to `(8,1)` -- which made a trainable constant
    unusable in any tuple a scaffold packs.
    """
    batched = torch.arange(8.).reshape(8, 1)
    constant = torch.tensor([3.0], requires_grad=True)
    assert relaxed(r, r.resolve("add", (F, F)), [batched, constant]).shape == (8, 1)
    op = r.resolve("tuple", (F, F))
    y = relaxed(r, op, [batched, constant])
    assert y.shape == (8, 2)
    assert y[:, 0].tolist() == batched.flatten().tolist()
    assert y[:, 1].tolist() == [3.0] * 8
    # ...and the constant is still on the gradient path through the pack.
    g = torch.autograd.grad(y.sum(), constant)[0]
    assert g.tolist() == [8.0]


def test_tuple_packing_survives_a_whole_program_with_a_batched_input():
    p = Program((("x", F),),
                (Node("packed", product(F, F),
                      (Candidate(r.resolve("tuple", (F, F)), ("x", "k")),), "core", 1),),
                (("out", "packed"),), (("k", Value.of(F, 2.0)),), trainable_constants=("k",))
    m = SoftProgram(p, r)
    out, _ = m({"x": torch.arange(8.).reshape(8, 1)})
    assert out["out"].shape == (8, 2)
    assert torch.autograd.grad(out["out"].sum(), m.constants["k"])[0].tolist() == [8.0]
