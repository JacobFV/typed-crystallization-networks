"""Verify the replay contract on the MuJoCo `control` generator, and price it.

Every claim in section 3 of RESULTS.md is produced here and written to
`out/replay_check.json`. The cross-process check is the load-bearing one: an
episode saved to disk, reloaded in a *fresh interpreter*, continued, and
compared bit for bit against the continuation taken in this process.

    .venv/bin/python research/external-environments/replay_check.py
"""
from __future__ import annotations
import json
import pathlib
import random
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.generation import Action, Host, stable_seed
from tcn.types import Value, product
from generators.control.generator import Implementation, torque_type

OUT = pathlib.Path(__file__).parent / "out"
CONFIG = {"task": "pendulum", "horizon": 40, "disturbance": 0.3}


def torque(value, task="pendulum"):
    t = torque_type(2.0 if task == "pendulum" else 1.0)
    if task == "pendulum":
        return Action("torque", arguments=(("value", Value.of(t, value)),))
    return Action("torque", arguments=(("value", Value.of(product(t, t), value)),))


def rollout(host, steps, task="pendulum"):
    """A deterministic energy-pumping controller, so the episode is not a stub."""
    for _ in range(steps):
        if host.records[-1].done:
            break
        rate = host.records[-1].latent_states["qvel"].decoded
        rate = rate if task == "pendulum" else rate[0]
        limit = 2.0 if task == "pendulum" else 1.0
        command = limit if rate >= 0 else -limit
        host.step((torque(command if task == "pendulum" else (command, 0.0), task),), dt=0.05)
    return host


def main():
    report = {}

    # 1. Replay from recorded inputs, which Host.replay already asserts.
    host = rollout(Host.create("control", seed=11, configuration=CONFIG), 12)
    report["replay_digest_matches"] = host.replay().digest == host.digest

    # 2. Snapshot, restore, continue: exact equality of the simulator state.
    restored = Host.restore(host.snapshot())
    report["restore_digest_matches"] = restored.digest == host.digest
    for _ in range(8):
        host.step((torque(0.4),), dt=0.05)
        restored.step((torque(0.4),), dt=0.05)
    report["continuation_digest_matches"] = host.digest == restored.digest
    report["continuation_state_bit_identical"] = host.state["integration"] == restored.state["integration"]
    report["max_state_difference_after_8_steps"] = max(
        abs(a - b) for a, b in zip(host.state["integration"], restored.state["integration"]))

    # 3. Cross-process: a fresh interpreter must reach the same continuation.
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "episode.json.gz"
    source = rollout(Host.create("control", seed=11, configuration=CONFIG), 12)
    source.save(path)
    for _ in range(8):
        source.step((torque(0.4),), dt=0.05)
    child = subprocess.run(
        [sys.executable, str(pathlib.Path(__file__).parent / "continue_episode.py"), str(path)],
        capture_output=True, text=True, cwd=ROOT, check=True)
    remote = json.loads(child.stdout)
    report["cross_process_digest_matches"] = remote["digest"] == source.digest
    report["cross_process_state_bit_identical"] = remote["integration"] == source.state["integration"]

    # 4. Two hosts at one address are identical; a different seed is not.
    a = Host.create("control", seed=11, configuration=CONFIG)
    b = Host.create("control", seed=11, configuration=CONFIG)
    c = Host.create("control", seed=12, configuration=CONFIG)
    report["same_address_identical"] = a.digest == b.digest
    report["different_seed_differs"] = a.digest != c.digest

    # 5. Named streams: the initial pose must not move when the goal does.
    base = Host.create("control", seed=4, configuration={"task": "reacher", "horizon": 8})
    with_goal = Host.create("control", seed=4, configuration={"task": "reacher", "horizon": 8},
                            objective={"target": [0.1, 0.1]})
    report["pose_invariant_to_supplied_goal"] = base.state["integration"] == with_goal.state["integration"]
    report["goal_actually_changed"] = base.state["target"] != with_goal.state["target"]
    # The counterfactual: one shared stream, goal drawn first, would move the pose.
    def single_stream(draw_goal):
        rng = random.Random(stable_seed({"generator": "control", "seed": 4, "index": 0, "split": "train"}, "shared", 0))
        if draw_goal:
            rng.uniform(0.06, 0.19), rng.uniform(-3.14159, 3.14159)
        return [rng.uniform(-1.0, 1.0) for _ in range(2)]
    report["single_stream_pose_would_move"] = single_stream(True) != single_stream(False)

    # 6. What the float32 observation costs against the float64 privileged state.
    record = host.records[-1]
    exact = record.latent_states["qvel"].decoded
    observed = record.observations["rates"].value.decoded
    report["qvel_float64_latent"] = exact
    report["qvel_float32_observation"] = observed
    report["observation_quantization_error"] = abs(exact - observed)
    report["observation_relative_error"] = abs(exact - observed) / max(1e-12, abs(exact))

    # 7. Visibility separation at the actor boundary.
    view = host.view()
    report["actor_sees"] = sorted(view.observations)
    report["actor_cannot_see_latents"] = not hasattr(view, "latent_states")
    report["actor_cannot_see_probes"] = not hasattr(view, "probes")
    report["latent_channels"] = sorted(record.latent_states)
    report["probe_channels"] = sorted(record.probes)

    # 8. Cost: wall clock per step, both tasks.
    for task in ("pendulum", "reacher"):
        h = Host.create("control", seed=1, configuration={"task": task, "horizon": 200})
        start = time.perf_counter()
        rollout(h, 100, task)
        report[f"seconds_per_step_{task}"] = (time.perf_counter() - start) / 100

    # 9. The task is learnable-shaped: the scripted controller improves return.
    swing = rollout(Host.create("control", seed=2, configuration={"task": "pendulum", "horizon": 200,
                                                                 "start": {"qpos": [0.0], "qvel": [0.0]}}), 120)
    uprights = [r.reward_components["upright"].decoded for r in swing.records[1:]]
    report["pendulum_upright_first_10"] = sum(uprights[:10]) / 10
    report["pendulum_upright_last_10"] = sum(uprights[-10:]) / 10
    report["pendulum_upright_max"] = max(uprights)

    # 9b. The range a fixed-point observation encoding would have to cover.
    peak = 0.0
    probe = Host.create("control", seed=2, configuration={"task": "pendulum", "horizon": 200,
                                                          "start": {"qpos": [0.0], "qvel": [0.0]}})
    while not probe.records[-1].done:
        rate = probe.records[-1].latent_states["qvel"].decoded
        probe.step((torque(2.0 if rate >= 0 else -2.0),), dt=0.05)
        peak = max(peak, abs(probe.records[-1].latent_states["qvel"].decoded))
    report["peak_abs_qvel_during_swing_up"] = peak
    report["fixed16_scale4096_range"] = 2 ** 15 / 4096
    report["fixed16_would_overflow"] = peak > 2 ** 15 / 4096

    # 10. Where that 2 ms goes: raw integration against the whole typed step.
    from generators.control import physics
    h = Host.create("control", seed=1, configuration={"task": "pendulum", "horizon": 500})
    saved = h.state["integration"]
    start = time.perf_counter()
    for _ in range(200):
        saved, _ = physics.advance("pendulum", saved, 0.05, [0.5])
    report["seconds_per_raw_integration"] = (time.perf_counter() - start) / 200
    start = time.perf_counter()
    for _ in range(200):
        physics.readout("pendulum", saved)
    report["seconds_per_readout"] = (time.perf_counter() - start) / 200
    start = time.perf_counter()
    for _ in range(200):
        h.step((torque(0.5),), dt=0.05)
    report["seconds_per_host_step"] = (time.perf_counter() - start) / 200
    report["contract_overhead_fraction"] = 1 - report["seconds_per_raw_integration"] / report["seconds_per_host_step"]

    (OUT / "replay_check.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    for key, value in report.items():
        print(f"{key}: {value}")
    failures = [k for k, v in report.items() if isinstance(v, bool) and not v]
    if failures:
        raise SystemExit("FAILED: " + ", ".join(failures))
    print("\nall boolean checks passed")


if __name__ == "__main__":
    main()
