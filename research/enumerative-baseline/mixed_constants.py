"""Task 4: the hybrid arm -- discrete enumeration + continuous constant fitting.

`examples/mixed.py` declares no trainable constants, so it is a pure discrete
problem (see `mixed_task.py`).  To answer the brief's question about mixed
discrete/continuous synthesis we need a program that actually has a trainable
constant, so this module builds the smallest honest variant of the same
program, using only operators from `tcn/operators.py`:

    logic      : truth_0..truth_15 over (a, b)          -> bool   [16 candidates]
    conversion : encode                                 -> float  [ 1 candidate]
    scale      : mul(conversion, k)                     -> float  [ 1 candidate]
                 with k a TRAINABLE CONSTANT
    algebra    : add | sub | mul over (scale, x)        -> float  [ 3 candidates]
    analytic   : sin | identity over (algebra)          -> float  [ 2 candidates]

    |discrete| = 16 * 3 * 2 = 96,  plus one continuous parameter k.

Target: answer = sin(K*[a xor b] + x) with K = 1.7.  Unlike `examples/mixed.py`
this variant is supervised ONLY at the output -- no per-node probes -- because
dense probes would hand the constant to both methods directly and make the
comparison vacuous.

Two search methods over that space:
  * hybrid    : enumerate the 96 discrete structures; inside each, fit k by a
                DERIVATIVE-FREE 1-D search (coarse grid then golden section) on
                the exact forward map.  No gradients anywhere.
  * gradient  : the TCN path, SoftProgram + Adam over choice logits and k
                jointly, then `Crystallizer`, via `tcn.synthesis.fit`.
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tcn.types import BOOL, Value, floating
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.synthesis import fit

from common import Budget, Result, Timer, write, machine
from instrument import Counter, autograd_counting, instrument

F = floating()
K_TRUE = 1.7
TOLERANCE = 1e-3


def problem(k_init=0.5):
    r = Registry()

    def cands(names, types, sources, output=None):
        return tuple(Candidate(r.resolve(n, types, output), sources) for n in names)

    nodes = (
        Node("logic", BOOL, cands([f"truth_{i}" for i in range(16)], (BOOL, BOOL), ("a", "b")), "logic", 1),
        Node("conversion", F, cands(["encode"], (BOOL,), ("logic",), F), "encoding", 2),
        Node("scale", F, cands(["mul"], (F, F), ("conversion", "k")), "algebra", 3),
        Node("algebra", F, cands(["add", "sub", "mul"], (F, F), ("scale", "x")), "algebra", 4),
        Node("analytic", F, cands(["sin", "identity"], (F,), ("algebra",)), "readout", 5),
    )
    constants = (("k", Value.of(F, k_init)),)
    p = Program((("a", BOOL), ("b", BOOL), ("x", F)), nodes, (("answer", "analytic"),),
                constants, input_depths=(("x", 3),), trainable_constants=("k",))
    signals = (Signal("analytic", "answer", ("readout",), F),)
    examples = []
    for a in (False, True):
        for b in (False, True):
            for x in (-0.7, -0.2, 0.3, 0.8, 1.4, -1.1):
                z = float(a != b)
                examples.append({
                    "inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "x": Value.of(F, x)},
                    "targets": {"answer": Value.of(F, math.sin(K_TRUE * z + x))},
                })
    return p.validate(r), signals, examples, r


# ------------------------------------------------------------------- hybrid arm
TABLES = list(range(16))
ALG = ["add", "sub", "mul"]
ANA = ["sin", "identity"]


def _forward(table, alg, ana, k, a, b, x):
    z = 1.0 if ((table >> (2 * int(a) + int(b))) & 1) else 0.0
    s = z * k
    u = {"add": s + x, "sub": s - x, "mul": s * x}[alg]
    return math.sin(u) if ana == "sin" else u


def _sse(table, alg, ana, k, examples, counter):
    total = 0.0
    for ex in examples:
        counter[0] += 1
        a = ex["inputs"]["a"].decoded
        b = ex["inputs"]["b"].decoded
        x = ex["inputs"]["x"].decoded
        t = ex["targets"]["answer"].decoded
        d = _forward(table, alg, ana, k, a, b, x) - t
        total += d * d
    return total


def fit_constant(table, alg, ana, examples, counter, lo=-4.0, hi=4.0, grid=161, refine=40):
    """Derivative-free 1-D fit: coarse grid, then golden-section on the best cell."""
    step = (hi - lo) / (grid - 1)
    best_k, best = lo, _sse(table, alg, ana, lo, examples, counter)
    for i in range(1, grid):
        k = lo + i * step
        v = _sse(table, alg, ana, k, examples, counter)
        if v < best:
            best, best_k = v, k
    a, b = best_k - step, best_k + step
    phi = (math.sqrt(5) - 1) / 2
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc = _sse(table, alg, ana, c, examples, counter)
    fd = _sse(table, alg, ana, d, examples, counter)
    for _ in range(refine):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a)
            fc = _sse(table, alg, ana, c, examples, counter)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a)
            fd = _sse(table, alg, ana, d, examples, counter)
    k = (a + b) / 2
    return k, _sse(table, alg, ana, k, examples, counter)


def hybrid(stop_at_first=True):
    _, _, examples, _ = problem()
    counter = [0]
    combos = list(itertools.product(TABLES, ALG, ANA))
    best = None
    tried = 0
    with Timer() as t:
        for table, alg, ana in combos:
            tried += 1
            k, sse = fit_constant(table, alg, ana, examples, counter)
            rmse = math.sqrt(sse / len(examples))
            if best is None or sse < best[0]:
                best = (sse, table, alg, ana, k)
            if stop_at_first and rmse <= TOLERANCE:
                break
    sse, table, alg, ana, k = best
    max_err = max(abs(_forward(table, alg, ana, k, ex["inputs"]["a"].decoded, ex["inputs"]["b"].decoded,
                               ex["inputs"]["x"].decoded) - ex["targets"]["answer"].decoded)
                  for ex in examples)
    return Result(
        "hybrid_enumeration_plus_1d_fit" + ("" if stop_at_first else "_full"),
        "mixed_constants",
        max_err <= TOLERANCE,
        t.seconds,
        Budget(programs=counter[0] // len(examples), op_applications=counter[0] * 5),
        {
            "space_size": len(combos),
            "structures_tried": tried,
            "solution": {"table": table, "algebra": alg, "analytic": ana, "k": k},
            "k_true": K_TRUE,
            "k_error": abs(k - K_TRUE),
            "max_abs_error": max_err,
            "example_evaluations": counter[0],
        },
    )


# ----------------------------------------------------------------- gradient arm
def gradient(seed, steps=600):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    program, signals, examples, registry = problem()
    counter = Counter()
    ops = sum(len(n.candidates) for n in program.nodes)
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
            model, report = fit(program, examples, signals, steps=steps, tolerance=TOLERANCE, registry=registry)
    finally:
        synthesis.SoftProgram = original
    k = float(model.constants["k"].detach()) if "k" in model.constants else None
    forward_ops = counter.forward_examples * ops
    backward_ops = 2 * ops * len(examples) * (counter.backward + counter.autograd_grad)
    return Result(
        "gradient_tcn",
        "mixed_constants",
        bool(report["exact_conformance"]),
        t.seconds,
        Budget(programs=counter.exact_executions, op_applications=forward_ops + backward_ops,
               gradient_steps=counter.backward),
        {
            "seed": seed,
            "steps": steps,
            "k_learned": k,
            "k_error": None if k is None else abs(k - K_TRUE),
            "loss": report["loss"],
            "fully_frozen": bool(report["fully_frozen"]),
            "selections": {kk: int(v) for kk, v in model.selections().items()},
            "forward_passes": counter.forward,
            "backward_passes": counter.backward,
        },
    )


def fairness_sweep():
    """Give the gradient side more budget before concluding anything about it.

    `SoftProgram` initialises choice logits to zeros and the constant to its
    declared value, and nothing in `synthesis.fit` is stochastic, so the seed
    does not change the outcome here -- only steps and learning rate do.
    """
    rows = []
    for steps in (600, 2000, 6000):
        for lr in (0.05, 0.01, 0.005):
            torch.manual_seed(0)
            torch.set_num_threads(1)
            program, signals, examples, registry = problem()
            with Timer() as t:
                model, report = fit(program, examples, signals, steps=steps, lr=lr,
                                    tolerance=TOLERANCE, registry=registry)
            k = float(model.constants["k"].detach()) if "k" in model.constants else None
            rows.append({
                "steps": steps, "lr": lr, "seconds": t.seconds,
                "exact_conformance": bool(report["exact_conformance"]),
                "fully_frozen": bool(report["fully_frozen"]),
                "k_learned": k, "k_error": None if k is None else abs(k - K_TRUE),
                "loss": report["loss"],
                "selections": {kk: int(v) for kk, v in model.selections().items()},
            })
            print(f"  steps={steps:>5} lr={lr:<6} conform={rows[-1]['exact_conformance']} "
                  f"k={k:.6f} err={rows[-1]['k_error']:.2e} {t.seconds:.1f}s", flush=True)
    return rows


def main():
    rows = []
    hybrid()  # warm up
    rows.append(hybrid(True).row())
    rows.append(hybrid(False).row())
    for s in range(10):
        print(f"gradient seed {s} ...", flush=True)
        rows.append(gradient(s).row())
    print("gradient budget/lr fairness sweep ...", flush=True)
    fair = fairness_sweep()
    print(write("mixed_constants.json", {"machine": machine(), "results": rows,
                                         "gradient_fairness_sweep": fair}))
    from statistics import median
    groups = {}
    for r in rows:
        groups.setdefault(r["method"], []).append(r)
    for m, rs in groups.items():
        print(f"{m:>36}  n={len(rs)}  solved={sum(r['solved'] for r in rs)}/{len(rs)}  "
              f"median={median(r['seconds'] for r in rs)*1000:9.2f} ms  "
              f"ops(med)={median(r['budget']['op_applications'] for r in rs):>10.0f}")


if __name__ == "__main__":
    main()
