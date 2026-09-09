"""Continuous control as an ordinary generator: MuJoCo behind the same contract.

This is the first generator in the repository whose mechanism is an external
simulator driven for its own sake rather than as scene physics. It exists to
test one question: can an external environment satisfy
`ARCHITECTURE.md` section 6 -- pinned versions, explicit random streams, a
logical clock, complete state restoration and recorded exogenous inputs -- with
typed observations and the observation/latent/probe split, rather than a bare
`step()` returning an array.

Three deliberate representation decisions, all of which have measurable
consequences in `tcn/operators.py` and are recorded in
`research/external-environments/RESULTS.md`:

* Observations use a **floating** encoding (`floating(32)`), never `role="byte"`.
  `Type.numeric` excludes `category`, `symbol` and `byte` roles, so a byte-roled
  channel admits only `eq`, `pack` and `index`. A joint angle needs `sin`,
  `cos`, `atan2`, `add` and `mul`, so it must not be a byte.
* Angles are declared **dimensionless** (`unit=""`). Radians are dimensionless,
  and the registry requires analytic arguments to be dimensionless, so tagging
  an angle `unit="rad"` would make `sin` and `cos` on it type-illegal.
* Rates and lengths **do** carry units, which is the type system working as
  intended: `sin(velocity)` is dimensional nonsense and is rejected. The cost is
  that no conversion operator can change a unit, so the only legal route from a
  united channel into the dimensionless algebra is division by a same-united
  value (a trainable constant, typically).

Privileged channels are carried at `floating(64)`, the simulator's own
precision; observations are the deliberately lossy `floating(32)` projection.
The exact generalized state is therefore not recoverable from the observation
stream, which is what section 6 asks of an agent-visible channel.
"""
from __future__ import annotations
import math
from tcn.generation import Generator, Value, integer, floating
from tcn.types import product
from . import physics

# --- Agent-visible representations ------------------------------------------
# Radians are dimensionless: `unit=""` is the physically correct declaration and
# it is also what keeps `sin`/`cos`/`atan2` legal on this type.
OBS_ANGLE = floating(32, frame="joint")
OBS_RATE = floating(32, unit="rad/s", frame="joint")
OBS_LENGTH = floating(32, unit="m", frame="world")

# --- Privileged representations, at the simulator's own precision ------------
LAT_ANGLE = floating(64, frame="joint")
LAT_RATE = floating(64, unit="rad/s", frame="joint")
LAT_LENGTH = floating(64, unit="m", frame="world")
LAT_ENERGY = floating(64, unit="J")
LAT_TIME = floating(64, unit="s")
RATIO = floating(64)

COUNT = integer(32, signed=False)


def torque_type(limit):
    """Actuator command type. `bounds` is the actuator limit, in the type.

    `tcn/policy.py:numeric_bounds` reads exactly this field, so a typed policy
    emits in-range torques with no environment-specific action head, and a
    caller that supplies an out-of-range torque gets a `ValueError` from
    `Value.of` rather than a silently clipped command. The MJCF `ctrlrange` is
    deliberately wider than the declared bound: the contract is the type.
    """
    limit = float(limit)
    if not 0 < limit <= 1e3 or float(f"{limit:.7g}") != limit:
        raise ValueError("torque limit must be positive and exactly representable")
    return floating(32, unit="N*m", frame="joint", bounds=(-limit, limit))


# The disturbance is an exogenous input, not a command, so it is not bounded by
# the actuator limit and is recorded in the transition rather than the actions.
DISTURBANCE = floating(32, unit="N*m", frame="joint")


def schema_for(task, limit):
    t = torque_type(limit)
    n = physics.degrees_of_freedom(task)
    return {"wait": {}, "torque": {"value": t if n == 1 else product(*(t for _ in range(n)))}}


def wrap(angle):
    """Principal value in [-pi, pi). A joint angle winds; its observation must not."""
    return (float(angle) + math.pi) % (2 * math.pi) - math.pi


class Implementation(Generator):
    version = "1"
    task = "pendulum"
    torque_limit = physics.TASKS["pendulum"]["torque_limit"]
    action_schema = schema_for("pendulum", physics.TASKS["pendulum"]["torque_limit"])

    def configure(self, configuration):
        """Pin the task and the actuator limit before any episode is built."""
        self.task = str(configuration.get("task", "pendulum"))
        spec = physics.task_spec(self.task)
        self.torque_limit = float(configuration.get("torque_limit", spec["torque_limit"]))
        self.action_schema = schema_for(self.task, self.torque_limit)

    # -- lifecycle ----------------------------------------------------------
    def initialize(self, address, configuration):
        spec = physics.task_spec(self.task)
        n = physics.degrees_of_freedom(self.task)
        # Two named streams. `layout` draws the goal, `start` draws the initial
        # configuration. Keeping them separate is what makes the initial pose
        # independent of whether the objective supplied a target: a single
        # stream would consume a different number of draws and shift the pose.
        layout = address.rng("layout")
        start = address.rng("start")
        objective = dict(configuration.get("objective") or {})
        radius = layout.uniform(0.06, 0.19)
        bearing = layout.uniform(-math.pi, math.pi)
        target = [radius * math.cos(bearing), radius * math.sin(bearing)]
        if objective.get("target") is not None:
            target = [float(x) for x in objective["target"]]
            if len(target) != 2:
                raise ValueError("objective target must be a planar coordinate")
        spread = float(configuration.get("start_spread", math.pi if self.task == "pendulum" else 1.0))
        qpos = [wrap(start.uniform(-spread, spread)) for _ in range(n)]
        qvel = [start.uniform(-0.5, 0.5) for _ in range(n)]
        if configuration.get("start") is not None:
            given = configuration["start"]
            qpos = [float(x) for x in given.get("qpos", qpos)]
            qvel = [float(x) for x in given.get("qvel", qvel)]
            if len(qpos) != n or len(qvel) != n:
                raise ValueError("explicit start must have one entry per joint")
        disturbance = float(configuration.get("disturbance", 0.0))
        if not 0 <= disturbance <= 10:
            raise ValueError("disturbance scale must be in [0, 10]")
        horizon = int(configuration.get("horizon", 64))
        if not 1 <= horizon <= 100000:
            raise ValueError("horizon must be 1..100000")
        return {
            "time": 0.0,
            "tick": 0,
            "task": self.task,
            "torque_limit": self.torque_limit,
            "integration": physics.initialize(self.task, qpos, qvel),
            "target": target,
            "disturbance_scale": disturbance,
            "applied": [0.0] * n,
            "exogenous": [0.0] * n,
            "substeps": 0,
            "horizon": horizon,
            "objective": objective,
            "done": False,
        }

    def advance(self, state, actions, dt, rng):
        if state["task"] != self.task or state["torque_limit"] != self.torque_limit:
            raise ValueError("configuration does not match the episode being advanced")
        n = physics.degrees_of_freedom(self.task)
        command = [0.0] * n
        for a in actions:
            if a.verb == "torque":
                value = a.arg("value")
                command = [float(value)] if n == 1 else [float(x) for x in value]
        # Exogenous input, drawn from the host's per-tick stream and recorded in
        # state, so a replay reproduces it without re-deriving it from a clock.
        scale = state["disturbance_scale"]
        exogenous = [rng.gauss(0.0, scale) if scale else 0.0 for _ in range(n)]
        exogenous = [max(-10.0, min(10.0, x)) for x in exogenous]
        applied = [c + e for c, e in zip(command, exogenous)]
        before = physics.readout(state["task"], state["integration"])
        state["integration"], substeps = physics.advance(state["task"], state["integration"], dt, applied)
        after = physics.readout(state["task"], state["integration"])
        state["applied"] = applied
        state["exogenous"] = exogenous
        state["substeps"] = substeps
        state["done"] = state["tick"] + 1 >= state["horizon"]
        transition = {
            "substeps": Value.of(COUNT, substeps),
            "exogenous": Value.of(DISTURBANCE if n == 1 else product(*(DISTURBANCE for _ in range(n))),
                                  exogenous[0] if n == 1 else tuple(exogenous)),
            "displacement": Value.of(LAT_ANGLE if n == 1 else product(*(LAT_ANGLE for _ in range(n))),
                                     (after["qpos"][0] - before["qpos"][0]) if n == 1
                                     else tuple(b - a for a, b in zip(before["qpos"], after["qpos"]))),
        }
        return state, transition, self.rewards(state, after, applied)

    # -- readouts -----------------------------------------------------------
    def rewards(self, state, view, applied):
        effort = -0.001 * sum(x * x for x in applied)
        if state["task"] == "pendulum":
            spec = physics.task_spec("pendulum")
            upright = (view["site"][2] - spec["pivot"][2]) / spec["length"]
            return {"upright": max(-1.0, min(1.0, upright)),
                    "rate": -0.01 * view["qvel"][0] ** 2,
                    "effort": effort}
        dx = view["site"][0] - state["target"][0]
        dy = view["site"][1] - state["target"][1]
        return {"proximity": -math.hypot(dx, dy),
                "rate": -0.01 * sum(v * v for v in view["qvel"]),
                "effort": effort}

    def observe(self, state):
        task = state["task"]
        n = physics.degrees_of_freedom(task)
        view = physics.readout(task, state["integration"])
        angles = [wrap(q) for q in view["qpos"]]
        pair = product(OBS_ANGLE, OBS_ANGLE)
        orientation_type = product(*(pair for _ in range(n)))
        observations = {
            "angles": Value.of(OBS_ANGLE if n == 1 else product(*(OBS_ANGLE for _ in range(n))),
                               angles[0] if n == 1 else tuple(angles)),
            "orientation": Value.of(pair if n == 1 else orientation_type,
                                    (math.cos(angles[0]), math.sin(angles[0])) if n == 1
                                    else tuple((math.cos(q), math.sin(q)) for q in angles)),
            "rates": Value.of(OBS_RATE if n == 1 else product(*(OBS_RATE for _ in range(n))),
                              view["qvel"][0] if n == 1 else tuple(view["qvel"])),
        }
        if task == "reacher":
            observations["to_target"] = Value.of(
                product(OBS_LENGTH, OBS_LENGTH),
                (state["target"][0] - view["site"][0], state["target"][1] - view["site"][1]))
        latents = {
            "qpos": Value.of(LAT_ANGLE if n == 1 else product(*(LAT_ANGLE for _ in range(n))),
                             view["qpos"][0] if n == 1 else tuple(view["qpos"])),
            "qvel": Value.of(LAT_RATE if n == 1 else product(*(LAT_RATE for _ in range(n))),
                             view["qvel"][0] if n == 1 else tuple(view["qvel"])),
            "physical_time": Value.of(LAT_TIME, view["physical_time"]),
        }
        probes = {
            "site": Value.of(product(LAT_LENGTH, LAT_LENGTH, LAT_LENGTH), tuple(view["site"])),
            "kinetic_energy": Value.of(LAT_ENERGY, view["kinetic"]),
            "potential_energy": Value.of(LAT_ENERGY, view["potential"]),
        }
        if task == "pendulum":
            spec = physics.task_spec("pendulum")
            probes["upright"] = Value.of(RATIO, max(-1.0, min(1.0, (view["site"][2] - spec["pivot"][2]) / spec["length"])))
        else:
            probes["target"] = Value.of(product(LAT_LENGTH, LAT_LENGTH), tuple(state["target"]))
            probes["distance"] = Value.of(LAT_LENGTH, math.hypot(view["site"][0] - state["target"][0],
                                                                 view["site"][1] - state["target"][1]))
        return observations, latents, probes, {"agent_0": tuple(self.action_schema)}
