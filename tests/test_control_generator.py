"""Contract tests for `generators/control`: MuJoCo behind the generator interface.

The load-bearing test is `test_snapshot_round_trip_continues_bit_identically`:
`ARCHITECTURE.md` section 6 requires complete state restoration, and for an
external simulator that is a claim about the simulator's own state vector, not
about the record format. It is checked by continuing both the original and the
restored episode and comparing the raw integration state element by element.
"""
import copy
import math
import random

import pytest

from tcn.generation import Action, Host, TimedValue
from tcn.operators import Registry
from tcn.policy import sample_exact
from tcn.types import BOOL, Value, integer, product
from generators.control import physics
from generators.control.generator import (
    DISTURBANCE, LAT_RATE, OBS_ANGLE, OBS_LENGTH, OBS_RATE, Implementation, schema_for, torque_type, wrap,
)

TASKS = ("pendulum", "reacher")


def command(task, value):
    """A torque action for `task`, built through the declared action schema."""
    t = schema_for(task, physics.TASKS[task]["torque_limit"])["torque"]["value"]
    return Action("torque", arguments=(("value", Value.of(t, value)),))


def sample(task, value=None):
    if task == "pendulum":
        return command(task, 1.0 if value is None else value)
    return command(task, (0.4, -0.3) if value is None else value)


# --- the shared generator contract ------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_common_contract_replay_restore_and_visibility(task, tmp_path):
    host = Host.create("control", seed=7, configuration={"task": task, "horizon": 6})
    host.step((sample(task),), dt=0.05)
    assert host.replay().digest == host.digest
    restored = Host.restore(host.snapshot())
    assert restored.digest == host.digest
    host.step(dt=0.05)
    restored.step(dt=0.05)
    assert host.digest == restored.digest
    path = tmp_path / (task + ".json.gz")
    host.save(path)
    assert Host.load(path).digest == host.digest
    view = host.view()
    assert not hasattr(view, "probes")
    assert not hasattr(view, "latent_states")
    assert not hasattr(view, "metadata")
    before = copy.deepcopy(host.state)
    view.objective["modified"] = True
    assert host.state == before


@pytest.mark.parametrize("task", TASKS)
def test_snapshot_round_trip_continues_bit_identically(task):
    """Complete state restoration, checked on the simulator state itself."""
    host = Host.create("control", seed=3, configuration={"task": task, "horizon": 40})
    for _ in range(4):
        host.step((sample(task),), dt=0.05)
    restored = Host.restore(host.snapshot())
    assert restored.state["integration"] == host.state["integration"]
    for step in range(10):
        value = 0.5 if task == "pendulum" else (0.5, 0.2)
        host.step((command(task, value),), dt=0.02 + 0.01 * step)
        restored.step((command(task, value),), dt=0.02 + 0.01 * step)
        assert host.state["integration"] == restored.state["integration"]
    assert host.digest == restored.digest


def test_recorded_exogenous_input_is_replayable():
    """A seeded disturbance is state, recorded in the transition, and reproduced."""
    host = Host.create("control", seed=9, configuration={"horizon": 8, "disturbance": 0.5})
    host.step((sample("pendulum"),), dt=0.05)
    exogenous = host.records[-1].transition["exogenous"]
    assert exogenous.type == DISTURBANCE
    assert exogenous.decoded != 0.0
    assert host.state["exogenous"][0] != 0.0
    assert host.replay().digest == host.digest
    assert Host.restore(host.snapshot()).state["exogenous"] == host.state["exogenous"]


def test_named_streams_keep_the_start_pose_independent_of_the_goal():
    base = Host.create("control", seed=4, configuration={"task": "reacher", "horizon": 4})
    moved = Host.create("control", seed=4, configuration={"task": "reacher", "horizon": 4},
                        objective={"target": [0.1, 0.1]})
    assert moved.state["target"] != base.state["target"]
    assert moved.state["integration"] == base.state["integration"]


def test_address_determines_the_episode():
    a = Host.create("control", seed=11, configuration={"horizon": 4})
    b = Host.create("control", seed=11, configuration={"horizon": 4})
    c = Host.create("control", seed=12, configuration={"horizon": 4})
    assert a.digest == b.digest
    assert a.state["integration"] != c.state["integration"]


# --- typed actions -----------------------------------------------------------

def test_action_schema_rejects_wrong_types_and_conflicting_commands():
    host = Host.create("control", configuration={"horizon": 4})
    with pytest.raises(TypeError):
        host.step((Action("torque", arguments=(("value", Value.of(integer(), 1)),)),))
    with pytest.raises(ValueError):
        host.step((Action("thrust"),))
    with pytest.raises(ValueError):
        host.step((sample("pendulum"), sample("pendulum")))


def test_actuator_limit_lives_in_the_type():
    """An out-of-range torque is a type error, not a silent clip."""
    t = torque_type(2.0)
    assert t.bounds == (-2.0, 2.0)
    with pytest.raises(ValueError):
        Value.of(t, 2.5)
    assert Value.of(t, 2.0).decoded == 2.0


def test_typed_policy_emits_in_range_torques_with_no_domain_action_head():
    """`tcn/policy.py` reads the bound off the type; nothing here is control-specific."""
    t = torque_type(2.0)
    rng = random.Random(0)
    for _ in range(50):
        value = sample_exact(t, [rng.gauss(0, 3), rng.gauss(0, 1)], rng)
        assert -2.0 <= value.decoded <= 2.0
    two = schema_for("reacher", 1.0)["torque"]["value"]
    value = sample_exact(two, [0.5, 0.0, -0.5, 0.0], rng)
    assert all(-1.0 <= x <= 1.0 for x in value.decoded)


def test_configuration_pins_the_task_and_the_actuator_limit():
    host = Host.create("control", configuration={"task": "reacher", "horizon": 4})
    assert host.generator.action_schema["torque"]["value"].kind == "tuple"
    assert Host.restore(host.snapshot()).generator.action_schema == host.generator.action_schema
    with pytest.raises(ValueError):
        Host.create("control", configuration={"task": "humanoid"})
    with pytest.raises(ValueError):
        Host.create("control", configuration={"horizon": 0})


# --- typed observations ------------------------------------------------------

@pytest.mark.parametrize("task", TASKS)
def test_observations_are_numeric_and_never_byte_roled(task):
    host = Host.create("control", seed=5, configuration={"task": task, "horizon": 4})
    def leaves(t):
        return [t] if t.kind in {"bool", "int"} else [x for item in t.items for x in leaves(item)]
    for name, value in host.records[-1].observations.items():
        for leaf in leaves(value.value.type):
            assert leaf.role == "", f"{name} leaked a role-tagged carrier"
            assert leaf.numeric, f"{name} is not numeric, so arithmetic on it is illegal"
            assert leaf.encoding.kind == "float"


def test_declared_encodings_admit_the_arithmetic_the_task_needs():
    """Measured against the registry, not asserted in prose."""
    registry = Registry()
    for name in ("sin", "cos", "exp", "neg", "abs"):
        assert registry.resolve(name, (OBS_ANGLE,)).output == OBS_ANGLE
    for name in ("add", "sub", "mul", "atan2", "min", "max"):
        assert registry.resolve(name, (OBS_ANGLE, OBS_ANGLE)).output == OBS_ANGLE
    # A dimensioned rate correctly refuses analytic operations...
    for name in ("sin", "cos", "exp", "log"):
        with pytest.raises(TypeError):
            registry.resolve(name, (OBS_RATE,))
    # ...and cannot be mixed with a dimensionless angle without an explicit step.
    with pytest.raises(TypeError):
        registry.resolve("add", (OBS_RATE, OBS_ANGLE))
    # The one legal route out of a unit is division by the same unit.
    assert registry.resolve("div", (OBS_RATE, OBS_RATE)).output == OBS_ANGLE
    assert registry.resolve("mul", (OBS_RATE, OBS_RATE)).output.unit == "(rad/s)^2"
    # No conversion operator may change a unit or a frame.
    with pytest.raises(TypeError):
        registry.resolve("encode", (OBS_RATE,), OBS_ANGLE)
    with pytest.raises(TypeError):
        registry.resolve("encode", (OBS_LENGTH,), OBS_ANGLE)


def test_a_byte_roled_observation_would_lose_the_arithmetic(monkeypatch):
    """The trap this generator avoids, stated as an executable contrast."""
    registry = Registry()
    pixel = integer(8, signed=False, role="byte")
    assert not pixel.numeric
    for name in ("add", "mul", "sin", "lt"):
        with pytest.raises(TypeError):
            registry.resolve(name, (pixel,) * (1 if name == "sin" else 2))
    assert registry.resolve("eq", (pixel, pixel)).output == BOOL


@pytest.mark.parametrize("task", TASKS)
def test_privileged_channels_carry_more_precision_than_observations(task):
    host = Host.create("control", seed=6, configuration={"task": task, "horizon": 6})
    host.step((sample(task),), dt=0.05)
    record = host.records[-1]
    assert record.latent_states["qvel"].type.bits == 64
    for value in record.observations.values():
        def leaves(t):
            return [t] if t.kind in {"bool", "int"} else [x for item in t.items for x in leaves(item)]
        assert all(leaf.bits == 32 for leaf in leaves(value.value.type))
    exact = record.latent_states["qvel"].decoded
    exact = exact if task == "pendulum" else exact[0]
    observed = record.observations["rates"].value.decoded
    observed = observed if task == "pendulum" else observed[0]
    assert exact == pytest.approx(observed, rel=1e-6)


@pytest.mark.parametrize("task", TASKS)
def test_privileged_truth_is_not_in_the_actor_view(task):
    host = Host.create("control", seed=8, configuration={"task": task, "horizon": 4},
                       objective={"target": [0.12, 0.0]} if task == "reacher" else {})
    record = host.records[-1]
    visible = set(host.view().observations)
    assert visible.isdisjoint(set(record.latent_states) | set(record.probes))
    assert "qpos" not in visible and "qvel" not in visible
    if task == "reacher":
        # The absolute goal is a probe; only the relative vector is observable.
        assert "target" in record.probes and "target" not in visible
        assert "to_target" in visible


@pytest.mark.parametrize("task", TASKS)
def test_reward_components_are_separate_typed_scalars(task):
    host = Host.create("control", seed=2, configuration={"task": task, "horizon": 4})
    rewards = host.step((sample(task),), dt=0.05).reward_components
    expected = {"upright", "rate", "effort"} if task == "pendulum" else {"proximity", "rate", "effort"}
    assert set(rewards) == expected
    assert all(v.type.encoding.kind == "float" for v in rewards.values())


# --- the mechanism itself ----------------------------------------------------

def test_the_pendulum_is_a_real_swing_up_problem():
    """An energy-pumping controller reaches upright; doing nothing does not."""
    def run(active):
        host = Host.create("control", seed=1, configuration={
            "task": "pendulum", "horizon": 160, "start": {"qpos": [0.0], "qvel": [0.0]}})
        best = -1.0
        while not host.records[-1].done:
            rate = host.records[-1].latent_states["qvel"].decoded
            action = (command("pendulum", 2.0 if rate >= 0 else -2.0),) if active else ()
            best = max(best, host.step(action, dt=0.05).reward_components["upright"].decoded)
        return best
    assert run(True) > 0.99
    assert run(False) < -0.99


def test_the_logical_clock_drives_the_physical_one():
    """`data.time` follows the host's `dt`; no host clock is consulted."""
    host = Host.create("control", seed=1, configuration={"horizon": 8})
    total = 0.0
    for dt in (0.01, 0.05, 0.02):
        host.step(dt=dt)
        total += dt
        assert host.state["time"] == pytest.approx(total, abs=1e-12)
        assert host.records[-1].latent_states["physical_time"].decoded == pytest.approx(total, abs=1e-9)


def test_substep_count_is_a_function_of_dt_alone():
    host = Host.create("control", seed=1, configuration={"horizon": 8})
    assert host.step(dt=0.05).transition["substeps"].decoded == 10
    assert host.step(dt=0.002).transition["substeps"].decoded == 1


def test_angles_are_wrapped_for_observation_but_not_in_state():
    assert wrap(3 * math.pi) == pytest.approx(-math.pi)
    assert wrap(0.25) == pytest.approx(0.25)
    host = Host.create("control", seed=1, configuration={
        "task": "pendulum", "horizon": 60, "start": {"qpos": [0.0], "qvel": [12.0]}})
    for _ in range(20):
        host.step(dt=0.05)
        angle = host.records[-1].observations["angles"].value.decoded
        assert -math.pi - 1e-5 <= angle <= math.pi + 1e-5
    assert abs(host.state["integration"][1]) > math.pi
