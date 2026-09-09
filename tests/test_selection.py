"""Backend selection, surrogate liveness, and the separated surrogate temperature."""
import math

import pytest
import torch

from examples.mixed import problem
from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, carrier_temperature, relaxed
from tcn.operators import Registry
from tcn.scaffold import F
from tcn.search import space_size
from tcn.select import (DEAD_GRADIENT, Decision, enumeration_cost, hybrid_fit,
                        liveness, select_backend)
from tcn.synthesis import fit
from tcn.types import BOOL, Value, integer, product

BYTE = integer(8, signed=False, role="byte")


def byte_program(registry, addresses=4, pool=(24, 30)):
    """Two byte probes compared against constants, then a conjunction."""
    BT = product(*(BYTE for _ in range(addresses)))
    consts = tuple((f"k{v}", Value.of(BYTE, v)) for v in pool)
    nodes = (
        Node("a0", BYTE, tuple(Candidate(registry.resolve("project", (BT,), BYTE, {"index": i}), ("raw",))
                               for i in range(addresses)), "address", 1),
        Node("e0", BOOL, tuple(Candidate(registry.resolve("eq", (BYTE, BYTE)), ("a0", f"k{v}")) for v in pool),
             "compare", 2),
        Node("y", BOOL, (Candidate(registry.resolve("truth_12", (BOOL, BOOL)), ("e0", "e0")),), "logic", 3))
    p = Program((("raw", BT),), nodes, (("out", "y"),), consts).validate(registry)
    return p, (Signal("y", "out", ("logic",), BOOL, "bce"),)


def byte_examples(n=8, addresses=4, target_index=1, target_value=24):
    rows = []
    for i in range(n):
        raw = tuple((target_value if j == target_index and i % 2 == 0 else (40 + 17 * j + 11 * i) % 256)
                    for j in range(addresses))
        rows.append({"inputs": {"raw": Value.of(product(*(BYTE for _ in range(addresses))), raw)},
                     "targets": {"out": Value.of(BOOL, raw[target_index] == target_value)}})
    return rows


# --- the surrogate liveness check -------------------------------------------

def test_eq_surrogate_is_exactly_zero_past_the_measured_spread():
    r = Registry()
    op = r.resolve("eq", (BYTE, BYTE))
    b = torch.tensor([[0.]])
    a = torch.tensor([[11.]], requires_grad=True)
    y = relaxed(r, op, [a, b])
    assert float(y.detach()) == 0.
    assert float(torch.autograd.grad(y.sum(), [a])[0]) == 0.
    a8 = torch.tensor([[8.]], requires_grad=True)
    assert float(relaxed(r, op, [a8, b]).detach()) > 0.


def test_carrier_temperature_is_read_off_the_declared_type():
    assert carrier_temperature(BYTE) == 256.
    assert carrier_temperature(BOOL) == 1.
    assert carrier_temperature(integer(16, signed=False)) == 65536.
    assert carrier_temperature(product(BOOL, BYTE)) == 256.


def test_carrier_scaling_revives_the_dead_surrogate():
    r = Registry()
    a = torch.tensor([[25.6]], requires_grad=True)
    b = torch.tensor([[0.]])
    op = r.resolve("eq", (BYTE, BYTE))
    assert float(relaxed(r, op, [a, b], 1., False).detach()) == 0.
    assert float(relaxed(r, op, [a, b], 1., True).detach()) > 1e-3


def test_liveness_reports_a_candidate_dead_on_every_example():
    r = Registry()
    p, signals = byte_program(r)
    rows = byte_examples()
    shipped = liveness(p, rows, signals, r, carrier_scaled=False)
    scaled = liveness(p, rows, signals, r, carrier_scaled=True)
    assert shipped.reachable_fraction < 1.
    assert "e0" in shipped.dead_nodes
    assert scaled.reachable_fraction > shipped.reachable_fraction
    assert "e0" not in scaled.dead_nodes
    dead = [c for c in shipped.candidates if c.dead and c.operator == "eq"]
    assert dead and all(c.live_rows == 0 for c in dead)


def test_liveness_flags_a_choice_behind_a_gradient_boundary():
    r = Registry()
    U16 = integer(16, signed=False)
    BT = product(BYTE, BYTE, BYTE, BYTE)
    PAIR = product(BYTE, BYTE)
    nodes = (
        Node("byte", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": i}), ("raw",))
                                 for i in range(4)), "address", 1),
        Node("tail", BYTE, (Candidate(r.resolve("project", (BT,), BYTE, {"index": 3}), ("raw",)),), "address", 1),
        Node("pair", PAIR, (Candidate(r.resolve("tuple", (BYTE, BYTE)), ("byte", "tail")),), "convert", 2),
        Node("packed", U16, (Candidate(r.resolve("pack", (PAIR,), U16), ("pair",)),), "convert", 3),
        Node("scale", U16, tuple(Candidate(r.resolve(n, (U16, U16)), ("packed", "one")) for n in ("add", "sub")),
             "algebra", 4))
    p = Program((("raw", BT),), nodes, (("out", "scale"),), (("one", Value.of(U16, 1)),)).validate(r)
    signals = (Signal("scale", "out", ("algebra",), U16),)
    rows = [{"inputs": {"raw": Value.of(BT, tuple((7 * i + 3 * j) % 256 for j in range(4)))},
             "targets": {"out": Value.of(U16, ((7 * i + 6) % 256) * 256 + (7 * i + 9) % 256 + 1)}}
            for i in range(8)]
    lv = liveness(p, rows, signals, r)
    assert "byte" in lv.dead_nodes                      # cut off by pack
    assert "scale" not in lv.dead_nodes                 # downstream of it and still differentiable
    assert "packed" in lv.boundary_nodes


def test_liveness_does_not_call_a_zero_init_cancellation_dead():
    """A uniform truth-table mixture is the constant 0.5 on every row, so a
    balanced target cancels the choice gradient exactly at the shipped zero
    initialization. That is a property of the initialization -- `examples/joint.py`
    carries a residual initialization to break it -- not of the relaxation, so
    liveness is read at a perturbed initialization as well."""
    r = Registry()
    srcs = (("i0", "i1"), ("i2", "i3"))
    nodes = tuple(Node(f"g{k}", BOOL,
                       tuple(Candidate(r.resolve(f"truth_{i}", (BOOL, BOOL)), s) for i in range(16)), "logic", d)
                  for k, (s, d) in enumerate(((srcs[0], 1), (srcs[1], 1), (("g0", "g1"), 2))))
    p = Program(tuple((f"i{i}", BOOL) for i in range(4)), nodes, (("y", "g2"),)).validate(r)
    signals = (Signal("g2", "y", ("logic",), BOOL, "bce"),)
    bits = [(a, b, c, d) for a in (0, 1) for b in (0, 1) for c in (0, 1) for d in (0, 1)]
    ref = lambda a, b, c, d: (a != b) and (c or d)
    rows = [{"inputs": {f"i{j}": Value.of(BOOL, bool(v)) for j, v in enumerate(x)},
             "targets": {"y": Value.of(BOOL, ref(*map(bool, x)) if i >= 2 else not ref(*map(bool, x)))}}
            for i, x in enumerate(bits)]
    assert set(liveness(p, rows, signals, r, inits=(0.,)).dead_nodes) == {"g0", "g1", "g2"}
    assert liveness(p, rows, signals, r).dead_nodes == ()


# --- the separated surrogate temperature ------------------------------------

def test_surrogate_scale_defaults_to_identity_on_a_shipped_fixture():
    p, signals, examples = problem()
    m = SoftProgram(p)
    assert set(m.surrogate_scale.values()) == {1.}
    assert m.carrier_scaled is False
    m2 = SoftProgram(p)
    inputs = {k: torch.stack([torch.tensor(e["inputs"][k].flat()) for e in examples]) for k, _ in p.inputs}
    a, _, _ = m(inputs, return_trace=True)
    b, _, _ = m2(inputs, return_trace=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_scale_surrogates_names_only_the_eq_nodes_and_leaves_choices_alone():
    r = Registry()
    p, signals = byte_program(r)
    m = SoftProgram(p, r)
    before = [q.detach().clone() for q in m.distributions()]
    touched = m.scale_surrogates()
    after = m.distributions()
    assert touched == ("e0",)
    assert m.carrier_scaled is True
    assert all(torch.equal(x, y) for x, y in zip(before, after))


def test_shipped_fixture_is_unchanged_end_to_end():
    p, signals, examples = problem()
    _, report = fit(p, examples, signals, steps=300, tolerance=.005)
    assert report["exact_max_error"] == 0.
    assert report["fully_frozen"]
    assert report["selection"] is None


# --- the rule ---------------------------------------------------------------

def test_small_exact_space_selects_enumeration_and_certifies():
    p, signals, examples = problem()
    d = select_backend(p, examples, signals, tolerance=.005)
    assert d.mode == "enumerate"
    assert d.certified and d.feasible
    assert d.projected_enumeration_seconds < 30.


def test_trainable_constants_select_the_hybrid():
    r = Registry()
    def c(names, types, sources, output=None):
        return tuple(Candidate(r.resolve(n, types, output), sources) for n in names)
    nodes = (Node("logic", BOOL, c([f"truth_{i}" for i in range(16)], (BOOL, BOOL), ("a", "b")), "logic", 1),
             Node("conv", F, c(["encode"], (BOOL,), ("logic",), F), "encoding", 2),
             Node("out", F, c(["mul", "add"], (F, F), ("conv", "k")), "algebra", 3))
    p = Program((("a", BOOL), ("b", BOOL)), nodes, (("y", "out"),), (("k", Value.of(F, 1.)),),
                trainable_constants=("k",)).validate(r)
    signals = (Signal("out", "y", ("algebra",), F),)
    rows = [{"inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)},
             "targets": {"y": Value.of(F, float(a != b) * 1.7)}} for a in (False, True) for b in (False, True)]
    d = select_backend(p, rows, signals, r)
    assert d.mode == "hybrid"
    assert d.trainable_constants == ("k",)
    result, constants = hybrid_fit(p, rows, signals, r, tolerance=1e-3)
    assert result.solved
    assert abs(float(constants["k"]) - 1.7) < 1e-3


def test_environment_coupling_flips_the_choice_to_relaxation():
    p, signals, examples = problem()
    cheap = select_backend(p, examples, signals, tolerance=.005, rollout_cost=0)
    coupled = select_backend(p, examples, signals, tolerance=.005, rollout_cost=64)
    assert cheap.mode == "enumerate"
    assert coupled.mode == "relax"
    assert coupled.environment_sweep == space_size(p) * 64
    assert not coupled.certified


def test_a_large_space_selects_relaxation():
    p, signals, examples = problem()
    d = select_backend(p, examples, signals, tolerance=.005, enumeration_seconds=1e-9)
    assert d.mode == "relax"
    assert not d.certified


def test_a_dead_relaxation_in_a_huge_space_is_reported_infeasible():
    r = Registry()
    U16 = integer(16, signed=False)
    BT = product(*(BYTE for _ in range(6)))
    PAIR = product(BYTE, BYTE)
    nodes = (
        Node("byte", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": i}), ("raw",))
                                 for i in range(6)), "address", 1),
        Node("tail", BYTE, tuple(Candidate(r.resolve("project", (BT,), BYTE, {"index": i}), ("raw",))
                                 for i in range(6)), "address", 1),
        Node("pair", PAIR, (Candidate(r.resolve("tuple", (BYTE, BYTE)), ("byte", "tail")),), "convert", 2),
        Node("packed", U16, (Candidate(r.resolve("pack", (PAIR,), U16), ("pair",)),), "convert", 3))
    p = Program((("raw", BT),), nodes, (("out", "packed"),)).validate(r)
    signals = (Signal("packed", "out", ("convert",), U16),)
    rows = [{"inputs": {"raw": Value.of(BT, tuple((5 * i + 3 * j) % 256 for j in range(6)))},
             "targets": {"out": Value.of(U16, ((5 * i + 6) % 256) * 256 + (5 * i + 15) % 256)}}
            for i in range(6)]
    d = select_backend(p, rows, signals, r, enumeration_seconds=1e-9)
    assert d.mode == "enumerate"
    assert not d.feasible
    assert "byte" in d.dead_nodes


def test_enumeration_cost_prices_the_real_sweep():
    p, signals, examples = problem()
    cost = enumeration_cost(p, examples, signals, tolerance=.005, probes=8)
    assert cost["space_size"] == space_size(p)
    assert cost["probes"] == 8
    assert cost["seconds_per_program"] > 0.
    assert math.isclose(cost["projected_seconds"], cost["space_size"] * cost["seconds_per_program"])


# --- the wiring -------------------------------------------------------------

def test_auto_mode_returns_the_same_report_shape_as_the_gradient_path():
    p, signals, examples = problem()
    _, relaxed_report = fit(p, examples, signals, steps=60, tolerance=.005)
    model, auto = fit(p, examples, signals, steps=60, tolerance=.005, mode="auto")
    assert auto["selection"]["mode"] == "enumerate"
    assert set(relaxed_report) <= set(auto)
    assert auto["exact_conformance"] and auto["exact_max_error"] == 0.
    assert auto["fully_frozen"]
    assert model.export().is_frozen


def test_forced_modes_and_unknown_modes():
    p, signals, examples = problem()
    _, report = fit(p, examples, signals, tolerance=.005, mode="enumerate")
    assert report["selection"]["mode"] == "enumerate"
    assert report["exact_conformance"]
    with pytest.raises(ValueError):
        fit(p, examples, signals, mode="nonsense")


def test_validation_split_is_scored_by_the_discrete_backend():
    """A conforming-on-training program that is wrong on held-out data must not
    be returned when a validation split is supplied."""
    r = Registry()
    node = Node("g", BOOL, tuple(Candidate(r.resolve(f"truth_{i}", (BOOL, BOOL)), ("a", "b")) for i in range(16)),
                "logic", 1)
    p = Program((("a", BOOL), ("b", BOOL)), (node,), (("y", "g"),)).validate(r)
    signals = (Signal("g", "y", ("logic",), BOOL, "bce"),)
    row = lambda a, b, y: {"inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)},
                           "targets": {"y": Value.of(BOOL, y)}}
    train = [row(False, False, False), row(False, True, True)]        # under-determined: 4 tables conform
    validation = [row(True, False, True), row(True, True, False)]     # pins it to xor
    loose = select_backend(p, train, signals, r)
    assert loose.mode == "enumerate"
    _, without = fit(p, train, signals, tolerance=1e-3, mode="auto")
    _, with_split = fit(p, train, signals, tolerance=1e-3, mode="auto", validation=validation)
    assert without["discrete_result"]["conforming"] == 4
    assert with_split["discrete_result"]["conforming"] == 1
    assert with_split["discrete_result"]["unique"]
