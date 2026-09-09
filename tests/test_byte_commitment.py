"""The byte/magnitude commitment: ARCHITECTURE sections 1 and 2.

`role="byte"` is an *uncommitted* carrier, not a third kind of number. These
checks pin down the three things the amendment claims: the uncommitted and
nominal classes stay exactly as restricted as they were, `interpret` is the one
explicit exit and runs in one direction only, and its gradient follows from the
relaxation each committed class already declares rather than from convenience.
"""
import pytest
import torch

from tcn.graph import Candidate, Node, Program
from tcn.learning import SoftProgram, relaxed, tensor
from tcn.operators import Registry
from tcn.types import BOOL, Value, floating, integer, product

r = Registry()
BYTE = integer(8, signed=False, role="byte")
MAG8 = integer(8, signed=False, role="intensity")
NOM8 = integer(8, signed=False, role="category")
FMAG = floating(32, role="intensity")
IDX = integer(16, signed=False)


def test_uncommitted_and_nominal_roles_stay_restricted():
    """Adding a commitment must not loosen what an uncommitted carrier allows."""
    assert not BYTE.numeric and not NOM8.numeric and MAG8.numeric
    for name, types in [("add", (BYTE, BYTE)), ("sub", (BYTE, BYTE)), ("mul", (BYTE, BYTE)),
                        ("lt", (BYTE, BYTE)), ("le", (BYTE, BYTE)), ("neg", (BYTE,)),
                        ("sum", (product(BYTE, BYTE),)), ("mean", (product(BYTE, BYTE),)),
                        ("add", (NOM8, NOM8)), ("lt", (NOM8, NOM8)),
                        ("sum", (product(NOM8, NOM8),))]:
        with pytest.raises(TypeError):
            r.resolve(name, types)
    assert r.resolve("eq", (BYTE, BYTE)).output == BOOL
    assert r.resolve("eq", (NOM8, NOM8)).output == BOOL


def test_interpret_commits_in_one_direction_only():
    assert r.resolve("interpret", (BYTE,), MAG8).gradient == "exact"
    assert r.resolve("interpret", (BYTE,), NOM8).gradient == "none"
    for source, target in [(NOM8, MAG8), (MAG8, NOM8), (BYTE, BYTE), (MAG8, MAG8),
                           (MAG8, BYTE), (NOM8, BYTE),
                           (BYTE, integer(8, signed=False))]:
        with pytest.raises(TypeError):
            r.resolve("interpret", (source,), target)


def test_interpret_preserves_the_carrier():
    """A commitment changes the declared meaning and nothing else."""
    for target in [integer(16, signed=False, role="intensity"),
                   floating(32, role="intensity"),
                   integer(8, signed=True, role="intensity"),
                   integer(8, signed=False, overflow="wrap", role="intensity"),
                   integer(8, signed=False, role="intensity", unit="cd"),
                   integer(8, signed=False, role="intensity", frame="camera"),
                   integer(8, signed=False, role="intensity", bounds=(0., 100.))]:
        with pytest.raises(TypeError):
            r.resolve("interpret", (BYTE,), target)
    with pytest.raises(TypeError):
        r.resolve("interpret", (BYTE,))          # the output must be explicit


def test_interpret_is_exact_and_bit_preserving():
    for x in (0, 1, 127, 128, 255):
        v = Value.of(BYTE, x)
        for target in (MAG8, NOM8):
            got = r.exact(r.resolve("interpret", (BYTE,), target), [v])
            assert got.type == target and got.raw == v.raw and got.decoded == x
    # a magnitude keeps the single-scalar lift; a nominal ID takes the categorical one
    assert Value.of(BYTE, 200).flat() == [200.]
    assert Value.of(MAG8, 200).flat() == [200.]
    assert len(Value.of(NOM8, 200).flat()) == 8


def test_magnitude_commitment_is_differentiable_and_nominal_is_a_boundary():
    a = torch.tensor([[7.]], requires_grad=True)
    y = relaxed(r, r.resolve("interpret", (BYTE,), MAG8), [a])
    assert y.shape == a.shape
    y.sum().backward()
    assert a.grad.item() == 1.
    b = torch.tensor([[7.]], requires_grad=True)
    z = relaxed(r, r.resolve("interpret", (BYTE,), NOM8), [b])
    assert z.shape[-1] == 8 and not z.requires_grad


def _convolution_program():
    """`w0*g(pos) + w1*g(pos+1)` over six raw bytes: the weighted-sum shape."""
    obs = product(*(BYTE for _ in range(6)))
    consts = (("off0", Value.of(IDX, 0)), ("off1", Value.of(IDX, 1)),
              ("w0", Value.of(FMAG, -1.)), ("w1", Value.of(FMAG, 1.)))
    types = {"pos": IDX, "obs": obs} | {k: v.type for k, v in consts}
    nodes = []

    def add(name, op, sources, out=None, params=None, depth=1):
        o = r.resolve(op, tuple(types[s] for s in sources), out, params)
        nodes.append(Node(name, o.output, (Candidate(o, tuple(sources)),), "core", depth))
        types[name] = o.output
        return name

    terms = []
    for i in (0, 1):
        add(f"a{i}", "add", ["pos", f"off{i}"], depth=1)
        add(f"b{i}", "index", ["obs", f"a{i}"], depth=2)
        add(f"m{i}", "interpret", [f"b{i}"], out=MAG8, depth=3)
        add(f"x{i}", "decode", [f"m{i}"], out=FMAG, depth=4)
        terms.append(add(f"t{i}", "mul", [f"x{i}", f"w{i}"], depth=5))
    add("packed", "tuple", terms, depth=6)
    add("conv", "sum", ["packed"], depth=7)
    return obs, Program((("pos", IDX), ("obs", obs)), tuple(nodes), (("y", "conv"),), consts,
                        input_depths=(("pos", 0), ("obs", 0)),
                        trainable_constants=("w0", "w1")).validate(r)


def test_weighted_sum_over_raw_bytes_is_expressible_and_exact():
    obs, prog = _convolution_program()
    pixels = (10, 40, 90, 160, 200, 250)
    out, _, _ = prog.execute({"pos": Value.of(IDX, 1), "obs": Value.of(obs, pixels)},
                             registry=r, selections={n.name: 0 for n in prog.nodes})
    assert out["y"].decoded == pytest.approx(pixels[2] - pixels[1])


def test_gradient_reaches_kernel_weights_through_the_committed_bytes():
    obs, prog = _convolution_program()
    pixels = (10, 40, 90, 160, 200, 250)
    model = SoftProgram(prog, r)
    # `index`'s relaxation is a softmax gather; sharpen it so the relaxed value is
    # the addressed byte. Each of these nodes has one candidate, so the shared
    # temperature cannot flatten any choice distribution here.
    for n in prog.nodes:
        if n.candidates[0].operator.name == "index":
            model.temperatures[n.name] = .02
    inputs = {"pos": tensor(Value.of(IDX, 1)).unsqueeze(0),
              "obs": tensor(Value.of(obs, pixels)).unsqueeze(0)}
    got, _ = model(inputs)
    assert got["y"].item() == pytest.approx(pixels[2] - pixels[1], abs=1e-3)
    got["y"].sum().backward()
    assert model.constants["w0"].grad.item() == pytest.approx(pixels[1], abs=1e-3)
    assert model.constants["w1"].grad.item() == pytest.approx(pixels[2], abs=1e-3)


def test_committed_program_serializes_and_crystallizes():
    from dataclasses import replace
    obs, prog = _convolution_program()
    assert Program.from_dict(prog.to_dict(), r).digest == prog.digest
    frozen = replace(prog.harden({n.name: 0 for n in prog.nodes}), trainable_constants=())
    name = r.register_module(frozen)
    call = r.resolve(name, (IDX, obs))
    assert call.gradient == "none"
    got = r.exact(call, [Value.of(IDX, 1), Value.of(obs, (10, 40, 90, 160, 200, 250))])
    assert got.decoded == pytest.approx(50.)
