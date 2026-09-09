"""Every remedy on the real task, on the same 144-program space.

The benchmark is `eq_temperature.build`: two free address nodes over the raw
`geometry` bytes, colour constants pinned, one Boolean of supervision.  It is
the perception ladder's failing arm with everything but the address choice
removed, and enumeration certifies its solution unique in 0.01 s.

Every arm reports node evaluations as well as successes, because the honest
comparison for a remedy that costs 144 forward passes is enumeration, which
costs 144 forward passes and returns a uniqueness certificate.

Run:  .venv/bin/python research/address-wall/real_remedies.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))

import torch

from tcn.learning import SoftProgram
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size

from eq_temperature import K0, K1, R, build, data, exact_error, tensors
from instrument import BUDGET, dump

SEEDS = range(12)
STEPS = 800
FREE = ("a0", "a1")


def hit(sel):
    return sel["a0"] == K0 and sel["a1"] == K1


def evaluate_held_out(program, registry, sel, test, signals):
    return exact_error(program, {"bytes": 0, "a0": sel["a0"], "a1": sel["a1"],
                                 "h0": 0, "h1": 0, "y": 0}, test, signals, registry)


def gradient_arm(program, registry, inputs, targets, signals, seed, remedy, tau,
                 steps=STEPS, lr=.05, noise=.5):
    m = SoftProgram(program, registry)
    m.temperatures["h0"] = m.temperatures["h1"] = tau
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * noise)
    idx = {n.name: i for i, n in enumerate(program.nodes)}
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    ev0 = BUDGET.node_evals
    for step in range(steps):
        frac = step / max(1, steps - 1)
        if remedy == "choice_anneal":
            for name in FREE:
                m.temperatures[name] = 8.0 * (0.25 / 8.0) ** frac
        if remedy == "straight_through":
            m.trials = {name: int(m.choices[idx[name]].argmax()) for name in FREE}
        elif remedy == "sampled":
            m.trials = {}
            for name in FREE:
                p = torch.softmax(m.choices[idx[name]] / m.temperatures[name], 0)
                m.trials[name] = int(torch.multinomial(p, 1, generator=g))
        opt.zero_grad()
        _, _, trace = m(inputs, return_trace=True)
        BUDGET.add(len(program.nodes))
        m.probe_loss(trace, targets, signals).backward()
        opt.step()
    m.trials = {}
    sel = m.selections()
    return sel, BUDGET.node_evals - ev0


def perturbation_arm(program, registry, inputs, targets, signals, seed, tau,
                     passes=4, restarts=1):
    """Score addresses by hard substitution, coordinate-descent to a fixpoint."""
    m = SoftProgram(program, registry)
    m.temperatures["h0"] = m.temperatures["h1"] = tau
    n = len(program.nodes[1].candidates)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    best = None
    ev0 = BUDGET.node_evals
    for _ in range(restarts):
        cur = {name: int(torch.randint(0, n, (1,), generator=g)) for name in FREE}
        for _ in range(passes):
            changed = False
            for name in FREE:
                scores = []
                for c in range(n):
                    m.trials = dict(cur); m.trials[name] = c
                    with torch.no_grad():
                        _, _, trace = m(inputs, return_trace=True)
                        BUDGET.add(len(program.nodes))
                        scores.append(float(m.probe_loss(trace, targets, signals)))
                pick = int(min(range(n), key=lambda c: scores[c]))
                changed |= pick != cur[name]
                cur[name] = pick
            if not changed:
                break
        m.trials = dict(cur)
        with torch.no_grad():
            _, _, trace = m(inputs, return_trace=True)
            BUDGET.add(len(program.nodes))
            L = float(m.probe_loss(trace, targets, signals))
        if best is None or L < best[0]:
            best = (L, dict(cur))
    m.trials = {}
    return {"bytes": 0, **best[1], "h0": 0, "h1": 0, "y": 0}, BUDGET.node_evals - ev0


def main():
    t0 = time.perf_counter()
    train, test = data()
    r = Registry()
    program, signals = build(r, R)
    inputs, targets = tensors(program, train, signals)
    out = {"space": space_size(program), "reference": {"a0": K0, "a1": K1},
           "chance": 1 / space_size(program), "train_episodes": len(train),
           "test_episodes": len(test)}

    ev0 = BUDGET.node_evals
    res = enumerate_fit(program, train, signals, registry=r, tolerance=1e-3)
    out["enumeration"] = {**res.to_dict(),
                          "node_evals": res.evaluated * len(program.nodes) * len(train)}
    print(f"enumeration: solved={res.solved} unique={res.unique} {res.seconds:.3f}s "
          f"{out['enumeration']['node_evals']} node evals sel={res.selections}", flush=True)

    arms = [("baseline", "plain", 1.), ("eq_tau_100", "plain", 100.),
            ("choice_anneal", "choice_anneal", 1.),
            ("choice_anneal+tau100", "choice_anneal", 100.),
            ("straight_through", "straight_through", 1.),
            ("straight_through+tau100", "straight_through", 100.),
            ("sampled", "sampled", 1.), ("sampled+tau100", "sampled", 100.)]
    table = []
    for name, remedy, tau in arms:
        rows = []
        for seed in SEEDS:
            sel, evals = gradient_arm(program, r, inputs, targets, signals, seed, remedy, tau)
            rows.append({"seed": seed, "a0": sel["a0"], "a1": sel["a1"], "hit": hit(sel),
                         "node_evals": evals})
        hits = sum(x["hit"] for x in rows)
        table.append({"arm": name, "tau": tau, "hits": hits, "n": len(rows),
                      "mean_node_evals": sum(x["node_evals"] for x in rows) / len(rows),
                      "rows": rows})
        print(f"  {name:24s} {hits:2d}/{len(rows)}  "
              f"{table[-1]['mean_node_evals']:8.0f} node evals", flush=True)

    for name, tau, restarts in (("perturbation", 1., 1), ("perturbation+tau100", 100., 1),
                                ("perturbation_4restarts", 1., 4)):
        rows = []
        for seed in SEEDS:
            sel, evals = perturbation_arm(program, r, inputs, targets, signals, seed, tau,
                                          restarts=restarts)
            held = evaluate_held_out(program, r, sel, test, signals)
            rows.append({"seed": seed, "a0": sel["a0"], "a1": sel["a1"], "hit": hit(sel),
                         "held_out_max_error": held, "node_evals": evals})
        hits = sum(x["hit"] for x in rows)
        table.append({"arm": name, "tau": tau, "hits": hits, "n": len(rows),
                      "mean_node_evals": sum(x["node_evals"] for x in rows) / len(rows),
                      "rows": rows})
        print(f"  {name:24s} {hits:2d}/{len(rows)}  "
              f"{table[-1]['mean_node_evals']:8.0f} node evals", flush=True)
    out["arms"] = table

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("real_remedies", out)


if __name__ == "__main__":
    main()
