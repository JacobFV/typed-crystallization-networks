"""Task 2: the flagship joint task of `examples/joint.py`.

The scaffold has 13 nodes.  Exactly two of them carry operator choice:

    relation       16 candidates : truth_i(bit_0, bit_1)
    goal_relation  16 candidates : truth_j(relation, goal)

    |discrete space| = 16 * 16 = 256

Every other node is single-candidate; the five constants (w0, w1, bias0, bias1,
baseline) are trainable floats.  Two enumeration arms are reported, and the
difference between them is the whole point:

  (a) `probe`  -- the SUPERVISED sub-problem.  Ask which (i, j) make the
      program's own probe outputs (`z`, `world`) equal the generator's probe
      labels (`target`, `gate`) on all 8 reachable (bit_0, bit_1, goal) rows.
      No environment interaction at all.  This is squarely inside enumeration's
      reach and is a truth-table conformance question.

  (b) `env`    -- the RL problem as posed.  Harden each of the 256 assignments,
      run the frozen agent in the live generator, score actual return.  This is
      the only arm that is a like-for-like substitute for what JointTrainer
      does, and it is more expensive because the environment must be stepped.
      It is still not the *same* problem: enumeration here inherits the
      declared constant initialization rather than learning it.  See
      `constants_probe()` for exactly how much that initialization gives away.
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from examples.joint import trainer
from tcn.agent import Agent
from tcn.generation import Host
from tcn.types import BOOL, Value, product
from tcn.scaffold import F

from common import Budget, Result, Timer, write, machine
from instrument import Counter, autograd_counting, instrument

CHOICE_NODES = ("relation", "goal_relation")
HORIZON = 4
EVAL_EPISODES = 16


def truth(k, a, b):
    return bool((k >> (2 * int(a) + int(b))) & 1)


# ---------------------------------------------------------------- arm (a)
def enumerate_probe(stop_at_first=True):
    """Pure truth-table conformance over the 8 reachable (bit_0, bit_1, goal) rows.

    Requirement recovered directly from `examples/joint.py` and the generator:
      world = encode(relation(bit_0, bit_1))  must equal probe `gate`  = values[-1]
      z     = encode(goal_relation(relation, goal)) must equal probe `target`
            = values[-1] != invert
    With {'depth':1,'table':6,'fixed_inputs':True} the generator's values[-1] is
    xor(bit_0, bit_1).
    """
    rows = [(a, b, g) for a in (False, True) for b in (False, True) for g in (False, True)]
    gate = {(a, b): (a != b) for a in (False, True) for b in (False, True)}
    solutions = []
    tried = 0
    checks = 0
    with Timer() as t:
        for i, j in itertools.product(range(16), repeat=2):
            tried += 1
            ok = True
            for a, b, g in rows:
                checks += 1
                rel = truth(i, a, b)
                z = truth(j, rel, g)
                if rel != gate[(a, b)] or z != (gate[(a, b)] != g):
                    ok = False
                    break
            if ok:
                solutions.append({"relation": i, "goal_relation": j})
                if stop_at_first:
                    break
    return Result(
        "exhaustive_enumeration_probe" + ("" if stop_at_first else "_full"),
        "joint",
        bool(solutions),
        t.seconds,
        Budget(programs=tried, op_applications=checks * 2),
        {"space_size": 256, "rows": len(rows), "solutions": solutions},
    )


def enumerate_probe_tcn_runtime(stop_at_first=False):
    """The same 256-way sweep, but every candidate executed through the repo's
    own exact runtime (`Program.execute`) instead of a direct truth-table
    evaluator, so the difference is purely the runtime's per-call overhead."""
    t0 = trainer(episodes=1)
    model, registry = t0.model, t0.model.registry
    rows = [(a, b, g) for a in (False, True) for b in (False, True) for g in (False, True)]
    solutions = []
    tried = 0
    executions = 0
    with Timer() as t:
        for i, j in itertools.product(range(16), repeat=2):
            tried += 1
            program = _hardened(model, i, j)
            ok = True
            for a, b, g in rows:
                bits = Value.of(product(BOOL, BOOL, BOOL, BOOL), (a, b, False, False))
                inputs = {
                    "bits": bits,
                    "goal": Value.of(BOOL, g),
                    "action": Value.of(product(F, F), (0.0, 0.0)),
                    "dt": Value.of(F, 1.0),
                }
                _, _, trace = program.execute(inputs, registry=registry)
                executions += 1
                gate = (a != b)
                if bool(round(trace["world"].decoded)) != gate or bool(round(trace["z"].decoded)) != (gate != g):
                    ok = False
                    break
            if ok:
                solutions.append({"relation": i, "goal_relation": j})
                if stop_at_first:
                    break
    return Result(
        "exhaustive_enumeration_probe_tcn_runtime" + ("" if stop_at_first else "_full"),
        "joint",
        bool(solutions),
        t.seconds,
        Budget(programs=tried, op_applications=executions * len(model.program.nodes)),
        {"space_size": 256, "solutions": solutions, "program_executions": executions},
    )


# ---------------------------------------------------------------- arm (b)
def _hardened(model, i, j):
    model.frozen = dict(model.frozen)
    model.frozen["relation"] = i
    model.frozen["goal_relation"] = j
    return model.export()


def _score(program, registry, config, episodes=EVAL_EPISODES, base=30000):
    agent = Agent(program, registry, config)
    total = 0.0
    steps = 0
    for k in range(episodes):
        host = Host.create(
            config.generator,
            seed=config.seed,
            index=base + k,
            split="test",
            configuration=config.generator_config | {"horizon": config.horizon},
            objective=config.objectives[k % len(config.objectives)],
        )
        agent.rollout(host, deterministic=True)
        total += sum(sum(v.decoded for v in r.reward_components.values()) for r in host.records)
        steps += len(host.records) - 1
    return total / episodes, steps


HELD_OUT = 64


def _verify(program, registry, config, episodes=HELD_OUT):
    """Held-out behavioural check of a selected program, disjoint episode indices."""
    score, _ = _score(program, registry, config, episodes, base=70000)
    return score >= config.horizon


def enumerate_env(episodes=EVAL_EPISODES, stop_at_first=True, order=None, label=None):
    """Enumerate discrete assignments, scoring each by ACTUAL return in the live generator.

    `stop_at_first` accepts the first assignment that scores the maximum return
    on its `episodes` selection episodes -- which is what a search would
    actually do, and which can accept a lucky impostor.  Success is therefore
    judged on a disjoint held-out set of `HELD_OUT` episodes.
    """
    t0 = trainer(episodes=1)
    model, config, registry = t0.model, t0.config, t0.model.registry
    combos = order or list(itertools.product(range(16), repeat=2))
    scores = {}
    chosen = None
    env_steps = 0
    tried = 0
    with Timer() as t:
        for i, j in combos:
            tried += 1
            program = _hardened(model, i, j)
            score, steps = _score(program, registry, config, episodes)
            env_steps += steps
            scores[(i, j)] = score
            if chosen is None and score >= config.horizon:
                chosen = (i, j)
                if stop_at_first:
                    break
    if chosen is None and scores:
        chosen = max(scores, key=lambda k: scores[k])
    solved = bool(chosen) and _verify(_hardened(model, *chosen), registry, config)
    perfect = [list(k) for k, v in scores.items() if v >= config.horizon]
    return Result(
        label or ("exhaustive_enumeration_env" + ("" if stop_at_first else "_full")),
        "joint",
        solved,
        t.seconds,
        Budget(
            programs=tried * episodes,
            op_applications=tried * episodes * config.horizon * 2 * len(model.program.nodes),
            env_steps=env_steps,
        ),
        {
            "space_size": 256,
            "combinations_tried": tried,
            "episodes_per_candidate": episodes,
            "chosen": {"relation": chosen[0], "goal_relation": chosen[1]},
            "chosen_is_reference": list(chosen) == [6, 6],
            "chosen_selection_return": scores[chosen],
            "perfect_scoring_candidates": len(perfect),
            "perfect_examples": perfect[:8],
            "maximum_return": config.horizon,
            "held_out_episodes": HELD_OUT,
        },
    )


def selection_noise_sweep(episode_counts=(2, 4, 8, 16, 32, 64)):
    """How many of the 256 assignments look perfect at E selection episodes?"""
    t0 = trainer(episodes=1)
    model, config, registry = t0.model, t0.config, t0.model.registry
    combos = list(itertools.product(range(16), repeat=2))
    out = []
    for e in episode_counts:
        with Timer() as t:
            perfect = []
            for i, j in combos:
                score, _ = _score(_hardened(model, i, j), registry, config, e)
                if score >= config.horizon:
                    perfect.append([i, j])
        out.append({
            "selection_episodes": e,
            "perfect_scoring": len(perfect),
            "reference_included": [6, 6] in perfect,
            "first_perfect": perfect[0] if perfect else None,
            "first_perfect_is_reference": bool(perfect) and perfect[0] == [6, 6],
            "seconds_full_sweep": t.seconds,
        })
        print("  selection_episodes", e, "->", out[-1]["perfect_scoring"], "perfect", f"{t.seconds:.2f}s", flush=True)
    return out


def random_search_env(seed, episodes=EVAL_EPISODES, cap=256, label="random_search_env"):
    rng = random.Random(seed)
    order = [(rng.randrange(16), rng.randrange(16)) for _ in range(cap)]
    r = enumerate_env(episodes=episodes, stop_at_first=True, order=order, label=label)
    r.detail["seed"] = seed
    r.detail["draws_allowed"] = cap
    return r


def random_search_env_matched(seed, env_step_budget=704, episodes=EVAL_EPISODES):
    """Random search given exactly the environment interaction the gradient run uses.

    JointTrainer spends `episodes * horizon` environment steps training plus
    `16 * horizon` evaluating; at horizon 4 and 160 training episodes that is
    704 steps.  Scoring one candidate on 16 episodes costs 64 steps, so this
    arm gets 11 draws.
    """
    cap = max(1, env_step_budget // (episodes * HORIZON))
    r = random_search_env(seed, episodes=episodes, cap=cap, label="random_search_env_matched")
    r.detail["env_step_budget"] = env_step_budget
    return r


def random_search_probe(seed, cap=256):
    """Random search on the supervised sub-problem -- the probe conformance test."""
    rng = random.Random(seed)
    rows = [(a, b, g) for a in (False, True) for b in (False, True) for g in (False, True)]
    gate = {(a, b): (a != b) for a in (False, True) for b in (False, True)}
    found = None
    checks = 0
    with Timer() as t:
        for k in range(1, cap + 1):
            i, j = rng.randrange(16), rng.randrange(16)
            ok = True
            for a, b, g in rows:
                checks += 1
                rel = truth(i, a, b)
                if rel != gate[(a, b)] or truth(j, rel, g) != (gate[(a, b)] != g):
                    ok = False
                    break
            if ok:
                found = (i, j)
                break
    return Result(
        "random_search_probe",
        "joint",
        found is not None,
        t.seconds,
        Budget(programs=k, op_applications=checks * 2),
        {"seed": seed, "space_size": 256, "draws_allowed": cap,
         "solution": found and {"relation": found[0], "goal_relation": found[1]}},
    )


# ------------------------------------------------------- the constants question
def constants_probe():
    """How much does the declared constant initialization already give away?

    logit0 = z*w0 + bias0, logit1 = z*w1 + bias1, with the values declared in
    examples/joint.py.  The greedy action is argmax(logit0, logit1); action 1 is
    `answer True`.  If that mapping is already correct at initialization, the
    RL problem's *continuous* part is solved before training starts and the
    only thing left to learn is the 256-way discrete choice.
    """
    w0, w1, b0, b1 = -2.0, 2.0, 1.0, -1.0
    table = {}
    for z in (0.0, 1.0):
        logits = (z * w0 + b0, z * w1 + b1)
        table[z] = {"logits": logits, "greedy_action": int(logits[1] > logits[0]), "correct": int(logits[1] > logits[0]) == int(z)}
    return {"constants": {"w0": w0, "w1": w1, "bias0": b0, "bias1": b1}, "table": {str(k): v for k, v in table.items()},
            "policy_correct_at_initialization": all(v["correct"] for v in table.values())}


# ---------------------------------------------------------------- gradient arm
def gradient(seed, episodes=160):
    torch.set_num_threads(1)
    counter = Counter()
    with Timer() as t, autograd_counting(counter):
        tr = trainer(episodes=episodes, seed=seed)
        instrument(tr.model, counter)
        ops = sum(len(n.candidates) for n in tr.model.program.nodes)
        history = tr.run()
        evaluations = [tr.episode(10000 + i, train=False, split="test")[0] for i in range(EVAL_EPISODES)]
    mean_return = sum(x["return"] for x in evaluations) / len(evaluations)
    selections = tr.model.selections()
    forward_ops = counter.forward_examples * ops
    backward_ops = 2 * ops * (counter.backward + counter.autograd_grad)
    return Result(
        "gradient_tcn",
        "joint",
        mean_return >= tr.config.horizon,
        t.seconds,
        Budget(
            programs=0,
            op_applications=forward_ops + backward_ops,
            env_steps=(episodes + EVAL_EPISODES) * tr.config.horizon,
            gradient_steps=counter.backward,
        ),
        {
            "seed": seed,
            "episodes": episodes,
            "evaluation_mean_return": mean_return,
            "maximum_return": tr.config.horizon,
            "selections": {k: int(v) for k, v in selections.items() if k in CHOICE_NODES},
            "correct_selection": selections["relation"] == 6 and selections["goal_relation"] == 6,
            "forward_passes": counter.forward,
            "backward_passes": counter.backward,
            "final_prediction_loss": sum(x["prediction_loss"] for x in history[-8:]) / 8,
        },
    )


def main():
    rows = []
    enumerate_probe()  # warm up imports
    rows.append(enumerate_probe(True).row())
    rows.append(enumerate_probe(False).row())
    for s in range(20):
        rows.append(random_search_probe(s).row())
    rows.append(enumerate_env(stop_at_first=True).row())
    rows.append(enumerate_env(stop_at_first=False).row())
    for s in range(20):
        print(f"random env seed {s} ...", flush=True)
        rows.append(random_search_env(s).row())
    for s in range(20):
        rows.append(random_search_env_matched(s).row())
    print("selection-noise sweep ...", flush=True)
    noise = selection_noise_sweep()
    for s in range(5):
        print(f"gradient seed {s} ...", flush=True)
        rows.append(gradient(s).row())
    payload = {"machine": machine(), "constants_probe": constants_probe(),
               "selection_noise_sweep": noise, "results": rows}
    print(write("joint.json", payload))
    from statistics import median
    groups = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    for m, rs in groups.items():
        print(
            f"{m:>34}  n={len(rs):>2}  solved={sum(r['solved'] for r in rs)}/{len(rs)}  "
            f"median={median(r['seconds'] for r in rs)*1000:10.2f} ms  "
            f"env_steps(med)={median(r['budget']['env_steps'] for r in rs):>7.0f}  "
            f"ops(med)={median(r['budget']['op_applications'] for r in rs):>10.0f}"
        )


if __name__ == "__main__":
    main()
