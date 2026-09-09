"""Exactly restorable MuJoCo integration state for low-dimensional control tasks.

Owned locally by this generator. `generators/world_3d/physics.py` rebuilds a
model from a procedural object list every call because its scene is data; here
the two scenes are fixed, so the compiled model is cached by its XML text and
only `MjData` is created per call. Nothing else differs: the exchanged state is
the same `mjSTATE_INTEGRATION` vector, which is the complete integration state
(time, `qpos`, `qvel`, `act`, warm-start acceleration) and therefore restores a
simulation exactly rather than approximately.

The physical clock advanced here is `data.time`, driven only by the logical `dt`
the host supplies. No host clock is read anywhere in this module.
"""
from __future__ import annotations
import math
from functools import lru_cache
import mujoco
import numpy as np

# A single hinge, hanging at qpos=0, upright at qpos=+-pi. Under-actuated:
# `torque_limit` below the gravitational torque makes swing-up a real problem
# rather than a set-point regulation.
PENDULUM_XML = """
<mujoco model="tcn_control_pendulum">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast"><flag energy="enable"/></option>
  <worldbody>
    <body name="pivot" pos="0 0 0.6">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom name="pole" type="capsule" fromto="0 0 0 0 0 -0.5" size="0.02" mass="1"/>
      <site name="tip" pos="0 0 -0.5" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="hinge" joint="hinge" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""

# Two hinges in the plane, no gravity: the reaching target is a coordinate in
# generator state, not a body, so the compiled model does not vary per episode.
REACHER_XML = """
<mujoco model="tcn_control_reacher">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 0" integrator="implicitfast"><flag energy="enable"/></option>
  <worldbody>
    <body name="link0" pos="0 0 0.01">
      <joint name="shoulder" type="hinge" axis="0 0 1" damping="0.1"/>
      <geom name="g0" type="capsule" fromto="0 0 0 0.1 0 0" size="0.01" mass="0.05"/>
      <body name="link1" pos="0.1 0 0">
        <joint name="elbow" type="hinge" axis="0 0 1" limited="true" range="-3.0 3.0" damping="0.1"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.11 0 0" size="0.01" mass="0.05"/>
        <site name="fingertip" pos="0.11 0 0" size="0.005"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="shoulder" joint="shoulder" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
    <motor name="elbow" joint="elbow" gear="1" ctrllimited="true" ctrlrange="-100 100"/>
  </actuator>
</mujoco>
"""

TASKS = {
    "pendulum": {
        "xml": PENDULUM_XML,
        "joints": ("hinge",),
        "site": "tip",
        "pivot": (0.0, 0.0, 0.6),
        "length": 0.5,
        "torque_limit": 2.0,
        "max_substep": 0.005,
    },
    "reacher": {
        "xml": REACHER_XML,
        "joints": ("shoulder", "elbow"),
        "site": "fingertip",
        "pivot": (0.0, 0.0, 0.01),
        "length": 0.21,
        "torque_limit": 1.0,
        "max_substep": 0.005,
    },
}

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


@lru_cache(maxsize=4)
def _compiled(xml: str):
    """One compiled model per distinct XML text. Compilation is a pure function of it."""
    return mujoco.MjModel.from_xml_string(xml)


def task_spec(task: str) -> dict:
    if task not in TASKS:
        raise ValueError(f"unknown control task {task!r}; expected one of {sorted(TASKS)}")
    return TASKS[task]


def degrees_of_freedom(task: str) -> int:
    return len(task_spec(task)["joints"])


def _load(task: str, saved=None):
    """A model plus a `MjData` positioned at `saved`, with derived fields computed."""
    model = _compiled(task_spec(task)["xml"])
    data = mujoco.MjData(model)
    if saved is not None:
        vector = np.asarray(saved, dtype=np.float64)
        if vector.size != mujoco.mj_stateSize(model, STATE_SPEC):
            raise ValueError("restored integration state has the wrong width")
        mujoco.mj_setState(model, data, vector, STATE_SPEC)
    mujoco.mj_forward(model, data)
    return model, data


def snapshot(model, data) -> list[float]:
    out = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, out, STATE_SPEC)
    return [float(x) for x in out]


def initialize(task: str, qpos, qvel) -> list[float]:
    """Integration state for an explicit initial configuration."""
    model, data = _load(task)
    data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    data.time = 0.0
    mujoco.mj_forward(model, data)
    return snapshot(model, data)


def advance(task: str, saved, dt: float, ctrl) -> tuple[list[float], int]:
    """Integrate `dt` of logical time from `saved` under a constant `ctrl`.

    The substep count is a deterministic function of `dt` alone, so a replayed
    episode takes exactly the same integration steps as the recorded one.
    """
    spec = task_spec(task)
    model, data = _load(task, saved)
    substeps = max(1, math.ceil(dt / spec["max_substep"]))
    model.opt.timestep = dt / substeps
    data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
    for _ in range(substeps):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)
    state = snapshot(model, data)
    if not all(math.isfinite(x) for x in state):
        raise ValueError("control integration diverged to a non-finite state")
    return state, substeps


def readout(task: str, saved) -> dict:
    """Exact derived quantities of a stored state, at full double precision."""
    spec = task_spec(task)
    model, data = _load(task, saved)
    site = data.site_xpos[model.site(spec["site"]).id]
    return {
        "qpos": [float(x) for x in data.qpos],
        "qvel": [float(x) for x in data.qvel],
        "site": [float(x) for x in site],
        "kinetic": float(data.energy[1]),
        "potential": float(data.energy[0]),
        "physical_time": float(data.time),
    }
