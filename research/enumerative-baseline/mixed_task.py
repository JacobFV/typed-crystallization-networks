"""Task 1: the mixed synthesis task of `examples/mixed.py`.

Candidate space searched by `tcn/learning.py` for this program:

    logic       16 candidates (truth_0 .. truth_15 over (a, b))
    conversion   1 candidate  (encode : bool -> float)
    algebra      3 candidates (add | sub | mul over (conversion, x))
    analytic     2 candidates (sin | identity over (algebra))

    |space| = 16 * 1 * 3 * 2 = 96

`examples/mixed.py` declares NO trainable constants, so the discrete space above
is the entire search space and exhaustive enumeration is exact and complete.
The hybrid enumeration+least-squares arm required by the brief is in
`mixed_constants.py`, on a variant of the same program that does carry trainable
constants.
"""
from __future__ import annotations

import itertools
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from examples.mixed import problem
from tcn.operators import Registry
from tcn.synthesis import fit

from common import Budget, Result, Timer, write, machine
from instrument import Counter, autograd_counting, instrument

TOLERANCE = 0.005  # the tolerance tcn/cli.py:mixed uses


def space(program):
    return [tuple(range(len(n.candidates))) for n in program.nodes]


def combinations(program):
    return list(itertools.product(*space(program)))


def conforms(program, registry, examples, signals, selections, counter=None):
    """Exact conformance of one fully-discrete assignment, no gradients."""
    for ex in examples:
        _, _, trace = program.execute(ex["inputs"], registry=registry, selections=selections)
        if counter is not None:
            counter.exact_executions += 1
        for s in signals:
            a = trace[s.source].flat()
            b = ex["targets"][s.target].flat()
            if max(abs(x - y) for x, y in zip(a, b)) > TOLERANCE:
                return False
    return True


def as_selection(program, combo):
    return {n.name: i for n, i in zip(program.nodes, combo)}


def enumerate_all(stop_at_first=True):
    program, signals, examples = problem()
    registry = Registry()
    program.validate(registry)
    combos = combinations(program)
    nodes = len(program.nodes)
    counter = Counter()
    solutions = []
    with Timer() as t:
        for k, combo in enumerate(combos, 1):
            if conforms(program, registry, examples, signals, as_selection(program, combo), counter):
                solutions.append(combo)
                if stop_at_first:
                    break
    budget = Budget(programs=k, op_applications=counter.exact_executions * nodes)
    return Result(
        "exhaustive_enumeration" + ("" if stop_at_first else "_full"),
        "mixed",
        bool(solutions),
        t.seconds,
        budget,
        {
            "space_size": len(combos),
            "combinations_tried": k,
            "solutions": [as_selection(program, c) for c in solutions],
            "examples": len(examples),
            "nodes": nodes,
        },
    )


def random_search(seed, cap=None):
    program, signals, examples = problem()
    registry = Registry()
    program.validate(registry)
    combos = combinations(program)
    cap = cap or 10 * len(combos)
    rng = random.Random(seed)
    counter = Counter()
    nodes = len(program.nodes)
    found = None
    with Timer() as t:
        for k in range(1, cap + 1):
            combo = tuple(rng.randrange(len(n.candidates)) for n in program.nodes)
            if conforms(program, registry, examples, signals, as_selection(program, combo), counter):
                found = combo
                break
    return Result(
        "random_search",
        "mixed",
        found is not None,
        t.seconds,
        Budget(programs=k, op_applications=counter.exact_executions * nodes),
        {"seed": seed, "space_size": len(combos), "solution": found and as_selection(program, found)},
    )


def gradient(seed, steps=300):
    """The repo's own configuration: tcn/cli.py:mixed -> fit(..., steps, tolerance=.005)."""
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    program, signals, examples = problem()
    counter = Counter()
    ops = sum(len(n.candidates) for n in program.nodes)
    n_examples = len(examples)
    # instrument the SoftProgram fit() builds, by patching the class constructor
    import tcn.synthesis as synthesis

    original = synthesis.SoftProgram
    holder = {}

    def make(*a, **k):
        m = original(*a, **k)
        instrument(m, counter)
        holder["model"] = m
        return m

    synthesis.SoftProgram = make
    try:
        with autograd_counting(counter), Timer() as t:
            model, report = fit(program, examples, signals, steps=steps, tolerance=TOLERANCE)
    finally:
        synthesis.SoftProgram = original
    forward_ops = counter.forward_examples * ops
    backward_ops = 2 * ops * n_examples * (counter.backward + counter.autograd_grad)
    return Result(
        "gradient_tcn",
        "mixed",
        bool(report["exact_conformance"]),
        t.seconds,
        Budget(
            programs=counter.exact_executions,
            op_applications=forward_ops + backward_ops,
            gradient_steps=counter.backward,
        ),
        {
            "seed": seed,
            "steps": steps,
            "forward_passes": counter.forward,
            "backward_passes": counter.backward,
            "autograd_grad_calls": counter.autograd_grad,
            "exact_executions": counter.exact_executions,
            "fully_frozen": bool(report["fully_frozen"]),
            "loss": report["loss"],
            "selections": {k: int(v) for k, v in model.selections().items()},
            "freeze_events": len(report["freeze_events"]),
            "accepted_freezes": sum(1 for e in report["freeze_events"] if e["accepted"]),
        },
    )


def main():
    seeds = list(range(10))
    enumerate_all(True)  # warm imports/caches so the timed calls are comparable
    enumerate_all(False)
    rows = [enumerate_all(True).row(), enumerate_all(False).row()]
    for s in seeds:
        rows.append(random_search(s).row())
    for s in seeds:
        print(f"gradient seed {s} ...", flush=True)
        rows.append(gradient(s).row())
    payload = {"machine": machine(), "results": rows}
    print(write("mixed.json", payload))
    for r in rows:
        print(
            f"{r['method']:>26}  solved={r['solved']!s:>5}  "
            f"{r['seconds']*1000:9.2f} ms  programs={r['budget']['programs']:>7}  "
            f"ops={r['budget']['op_applications']:>10}"
        )


if __name__ == "__main__":
    main()
