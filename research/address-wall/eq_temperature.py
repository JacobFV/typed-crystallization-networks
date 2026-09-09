"""The remedy the autopsy points at: widen the surrogate the address feeds.

`ladder_autopsy.py` shows the failing arm's address nodes are not flat and not
near chance -- they pick the reference 0/10 with a gradient 13x LARGER than the
colour nodes', while `eq`'s surrogate at the address mixture reads 3.2e-31.  The
mixture lands where the downstream operator is saturated, so the only way the
loss can fall is by moving the mixture's VALUE toward the constant, and the
gradient therefore ranks addresses by how close their value is to the constant
rather than by whether they are the right address.

That is a testable claim with an obvious remedy, so both are measured here on
the real `geometry` bytes, with the colour constants pinned so that the address
choice is the only free choice in the program:

    a0 = project(bytes, i)      12 candidates   <- the only free nodes
    a1 = project(bytes, j)      12 candidates
    h0 = eq(a0, 24)             1 candidate
    h1 = eq(a1, 30)             1 candidate
    y  = truth_8(h0, h1)        1 candidate     (AND)

144 programs, one Boolean of supervision, no other choice to hide behind.

Pinning the colour nodes to one candidate each matters for a second reason:
`SoftProgram` keeps ONE temperature per node and divides BOTH the candidate
softmax AND the operator's own relaxation by it, so on a node with a real choice
you cannot widen `eq`'s surrogate without flattening that node's choice
distribution at the same time.  With one candidate the choice softmax is
constant, and the temperature means only what we want it to mean.

Run:  .venv/bin/python research/address-wall/eq_temperature.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))

import torch

from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size
from tcn.types import BOOL, Value

from instrument import BUDGET, dump

TAUS = (1., 10., 100., 1000., 1e4, 1e5, 1e6)
SEEDS = range(12)
R = 2
K0, K1 = 6, 7          # the centre pixel's red and green bytes at R=2
BG = (24, 30, 43)


def build(registry, R):
    from rung3_geometry import image_type
    from tcn.generation import BYTE
    IT = image_type(R)
    BT = IT.items[3]
    n = 3 * R * R
    proj = lambda i: Candidate(registry.resolve("project", (BT,), BYTE, {"index": i}), ("bytes",))
    eq = lambda src, kname: Candidate(registry.resolve("eq", (BYTE, BYTE)), (src, kname))
    nodes = (
        Node("bytes", BT, (Candidate(registry.resolve("project", (IT,), BT, {"index": 3}), ("pixels",)),), "obs", 1),
        Node("a0", BYTE, tuple(proj(i) for i in range(n)), "address", 2),
        Node("a1", BYTE, tuple(proj(i) for i in range(n)), "address", 2),
        Node("h0", BOOL, (eq("a0", "k0"),), "compare", 3),
        Node("h1", BOOL, (eq("a1", "k1"),), "compare", 3),
        Node("y", BOOL, (Candidate(registry.resolve("truth_8", (BOOL, BOOL)), ("h0", "h1")),), "logic", 4),
    )
    p = Program((("pixels", IT),), nodes, (("centre", "y"),),
                (("k0", Value.of(BYTE, BG[0])), ("k1", Value.of(BYTE, BG[1]))))
    return p.validate(registry), (Signal("y", "centre", ("logic",), BOOL, "bce"),)


def data(n_train=24, n_test=24):
    from rung3_geometry import examples as gex
    return gex(range(0, n_train), R), gex(range(100, 100 + n_test), R)


def tensors(program, examples, signals):
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in examples]) for s in signals}
    return inputs, targets


def set_tau(m, tau):
    m.temperatures["h0"] = tau
    m.temperatures["h1"] = tau


def pointing_at(tau, program, registry, inputs, targets, signals, seeds=8, noise=.5):
    hits = scored = dead = 0
    picks = []
    grads = []
    for seed in range(seeds):
        m = SoftProgram(program, registry)
        set_tau(m, tau)
        g = torch.Generator().manual_seed(seed + 999)
        with torch.no_grad():
            for p in m.choices:
                if p.requires_grad:
                    p.add_(torch.randn(p.shape, generator=g) * noise)
        _, _, trace = m(inputs, return_trace=True)
        BUDGET.add(len(program.nodes))
        m.probe_loss(trace, targets, signals).backward()
        for node, p, ref in ((program.nodes[1], m.choices[1], K0), (program.nodes[2], m.choices[2], K1)):
            if p.grad is None or float(p.grad.abs().sum()) == 0.:
                dead += 1
                continue
            scored += 1
            picks.append(int(p.grad.argmin()))
            grads.append(float(p.grad.abs().sum()))
            hits += int(p.grad.argmin()) == ref
    return {"tau": tau, "scored": scored, "dead": dead,
            "picks_reference": hits / max(1, scored), "chance": 1 / 12,
            "picks": picks,
            "median_grad_abs_sum": float(torch.tensor(grads).median()) if grads else 0.}


def synthesise(tau, program, registry, inputs, targets, signals, seed,
               steps=800, lr=.05, noise=.5, anneal_to=None):
    m = SoftProgram(program, registry)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * noise)
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    for step in range(steps):
        t = tau if anneal_to is None else tau * (anneal_to / tau) ** (step / max(1, steps - 1))
        set_tau(m, t)
        opt.zero_grad()
        _, _, trace = m(inputs, return_trace=True)
        BUDGET.add(len(program.nodes))
        m.probe_loss(trace, targets, signals).backward()
        opt.step()
    sel = m.selections()
    return {"seed": seed, "a0": sel["a0"], "a1": sel["a1"],
            "hit": (sel["a0"], sel["a1"]) in {(K0, K1), (K1, K0)} and sel["a0"] != sel["a1"]}


def exact_error(program, selections, examples, signals, registry):
    worst = 0.
    for e in examples:
        try:
            _, _, tr = program.execute(e["inputs"], registry=registry, selections=selections)
        except Exception:
            return float("inf")
        for s in signals:
            a = tr[s.source].flat(); b = e["targets"][s.target].flat()
            worst = max(worst, max(abs(x - y) for x, y in zip(a, b)))
    return worst


def main():
    t0 = time.perf_counter()
    train, test = data()
    r = Registry()
    program, signals = build(r, R)
    inputs, targets = tensors(program, train, signals)
    out = {"space": space_size(program), "chance_two_addresses": 1 / 144,
           "reference": {"a0": K0, "a1": K1}}
    print(f"space = {out['space']} programs", flush=True)

    res = enumerate_fit(program, train, signals, registry=r, tolerance=1e-3)
    out["enumeration"] = res.to_dict()
    print(f"enumeration: solved={res.solved} unique={res.unique} "
          f"{res.seconds:.2f}s sel={res.selections}", flush=True)

    print("== does the address gradient point at the reference, as a function of the "
          "eq surrogate temperature? ==", flush=True)
    out["pointing"] = [pointing_at(tau, program, r, inputs, targets, signals) for tau in TAUS]
    for row in out["pointing"]:
        print(f"  tau={row['tau']:>9.0f}  scored={row['scored']:2d} dead={row['dead']:2d} "
              f"picks-ref={row['picks_reference']:.3f} (chance {row['chance']:.3f})  "
              f"|grad|={row['median_grad_abs_sum']:.4g}  picks={row['picks']}", flush=True)

    print("== full synthesis on the same 144-program space ==", flush=True)
    arms = [("fixed_tau_1", 1., None), ("fixed_tau_100", 100., None),
            ("fixed_tau_1e4", 1e4, None), ("fixed_tau_1e5", 1e5, None),
            ("anneal_1e5_to_1", 1e5, 1.), ("anneal_1e4_to_1", 1e4, 1.),
            ("anneal_1e6_to_1", 1e6, 1.)]
    table = []
    for name, tau, to in arms:
        rows = [synthesise(tau, program, r, inputs, targets, signals, s, anneal_to=to)
                for s in SEEDS]
        hits = sum(row["hit"] for row in rows)
        held = []
        for row in rows:
            if row["hit"]:
                held.append(exact_error(program, {"bytes": 0, "a0": row["a0"], "a1": row["a1"],
                                                  "h0": 0, "h1": 0, "y": 0},
                                        test, signals, r))
        table.append({"arm": name, "tau": tau, "anneal_to": to, "hits": hits,
                      "n": len(rows), "held_out_max_error": max(held) if held else None,
                      "rows": rows})
        print(f"  {name:18s} {hits:2d}/{len(rows)}  held-out max error "
              f"{max(held) if held else None}", flush=True)
    out["synthesis"] = table

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("eq_temperature", out)


if __name__ == "__main__":
    main()
