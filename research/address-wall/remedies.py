"""Task 3 -- the obvious remedies, and what each one actually buys.

Five benchmarks, chosen because the landscape measurement showed each of them
failing, plus one that reproduces the perception ladder's own shape:

  T1  discrete address, direct supervision, byte-scale values      (16 programs)
  T2  discrete address, `eq` downstream, unequal address means     (16 programs)
  T3  `index` operator, direct supervision, byte-scale values
  T4  `index` operator, direct supervision, uncorrelated values
  T5  TWO discrete addresses, each behind `eq`, AND-folded to one
      output bit -- the shape of `rung3:centre_free_address`      (256 programs)

Remedies, each measured against the same benchmark and the same data seeds,
with node evaluations reported alongside the outcome:

  baseline           Adam on the choice logits / on the address constant
  logit_noise_*      initialisation noise scale (the only thing that varies a
                     `SoftProgram` at all, since it zero-initialises logits)
  choice_anneal      temperature on the choice softmax, high -> low
  index_anneal       temperature on the `index` kernel, wide -> narrow
  index_wide         a fixed wide `index` kernel, no annealing
  index_init_centre  start the address at the middle of the tuple
  straight_through   hard address forward, soft address backward (`trials`)
  sampled            a fresh address sampled every step, straight-through
  perturbation       score every address by hard substitution, take the argmin
  centred_values     diagnostic: the same task on per-address centred values

Run:  .venv/bin/python research/address-wall/remedies.py
"""
from __future__ import annotations

import math
import time

import torch

from tcn.graph import Candidate, Node, Program, Signal
from tcn.learning import SoftProgram
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size
from tcn.types import BOOL, Value, product

from instrument import (BUDGET, EQ_REGIMES, REGIMES, V, batch, discrete_program,
                        dump, eq_arrays, gp_arrays, index_program, loss_at,
                        make_examples, perturbation_pick)

N = 16
M = 64
SEEDS = range(12)


# --------------------------------------------------------------------------
# benchmarks
# --------------------------------------------------------------------------

def bench_discrete(regime, downstream, k, seed):
    if downstream == "direct":
        corr, spread, scale, offset = REGIMES[regime]
        arrays = gp_arrays(N, M, corr, spread, seed=seed * 97 + 11, scale=scale, offset=offset)
        c = 0.0
    else:
        arrays = eq_arrays(regime, N, M, seed=seed * 97 + 11)
        col = arrays[:, k].tolist()
        c = min(sorted(set(col)), key=lambda v: abs(sum(1 for x in col if x == v) / len(col) - .5))
    ex = make_examples(arrays, k, downstream, c)
    r = Registry()
    program, signals = discrete_program(r, N, downstream, c)
    return program, signals, ex, r, {"addr": k}, arrays


def bench_index(regime, seed):
    corr, spread, scale, offset = REGIMES[regime]
    arrays = gp_arrays(N, M, corr, spread, seed=seed * 97 + 11, scale=scale, offset=offset)
    ex = make_examples(arrays, 11, "direct")
    r = Registry()
    program, signals = index_program(r, N, "direct")
    return program, signals, ex, r, {"b": 11}, arrays


def conjunction_program(registry, n, c1, c2):
    """Two free addresses, each compared to a constant, AND-folded to one bit.

    This is `rung3:centre_free_address` reduced to its skeleton: the supervision
    is a single Boolean, so the two address choices are coupled and cannot be
    solved one at a time.
    """
    arr_t = product(*([V] * n))
    proj = lambda i: Candidate(registry.resolve("project", (arr_t,), V, {"index": i}), ("arr",))
    eq = registry.resolve("eq", (V, V), BOOL)
    nodes = (
        Node("a1", V, tuple(proj(i) for i in range(n)), depth=1),
        Node("a2", V, tuple(proj(i) for i in range(n)), depth=1),
        Node("h1", BOOL, (Candidate(eq, ("a1", "c1")),), depth=2),
        Node("h2", BOOL, (Candidate(eq, ("a2", "c2")),), depth=2),
        Node("y", BOOL, (Candidate(registry.resolve("and", (BOOL, BOOL), BOOL), ("h1", "h2")),), depth=3),
    )
    p = Program(inputs=(("arr", arr_t),), nodes=nodes, outputs=(("out", "y"),),
                constants=(("c1", Value.of(V, float(c1))), ("c2", Value.of(V, float(c2)))),
                input_depths=(("arr", 0),))
    return p.validate(registry), (Signal("y", "out", ("core",), BOOL, "mse"),)


def bench_conjunction(regime, seed, k1=11, k2=4):
    arrays = eq_arrays(regime, N, M, seed=seed * 97 + 11)
    pick = lambda k: min(sorted(set(arrays[:, k].tolist())),
                         key=lambda v: abs(sum(1 for x in arrays[:, k].tolist() if x == v) / M - .5))
    c1, c2 = pick(k1), pick(k2)
    r = Registry()
    program, signals = conjunction_program(r, N, c1, c2)
    ex = []
    for row in arrays.tolist():
        ex.append({"inputs": {"arr": Value.of(product(*([V] * N)), tuple(float(v) for v in row))},
                   "targets": {"out": Value.of(BOOL, bool(row[k1] == c1 and row[k2] == c2))}})
    return program, signals, ex, r, {"a1": k1, "a2": k2}, arrays


BENCHES = {
    "T1_discrete_direct_bytelike": lambda s: bench_discrete("bytelike", "direct", 11, s),
    "T2_discrete_eq_meanspread": lambda s: bench_discrete("iid_mean_spread", "eq", 11, s),
    "T3_index_direct_bytelike": lambda s: bench_index("bytelike", s),
    "T4_index_direct_iid": lambda s: bench_index("iid_centred", s),
    "T5_conjunction_two_addresses": lambda s: bench_conjunction("iid_mean_spread", s),
}
DISCRETE = {"T1_discrete_direct_bytelike", "T2_discrete_eq_meanspread",
            "T5_conjunction_two_addresses"}


# --------------------------------------------------------------------------
# one run under one remedy
# --------------------------------------------------------------------------

def run(bench, seed, remedy, steps=600, lr=.05):
    program, signals, ex, r, reference, arrays = BENCHES[bench](seed)
    if remedy == "centred_values":
        # Only defined where the supervision is the addressed value itself.  With
        # an `eq` downstream the target is computed against a raw value and the
        # constant is a raw value, so removing a per-address mean would require
        # already knowing the address; there is no honest centred variant of T2
        # or T5 and this returns "not applicable" rather than a different task.
        if bench not in ("T1_discrete_direct_bytelike", "T3_index_direct_bytelike",
                         "T4_index_direct_iid"):
            return {"selection": None, "reference": reference, "hit": False,
                    "node_evals": 0, "not_applicable": True}
        centred = arrays - arrays.mean(0, keepdim=True)
        ex = make_examples(centred, 11, "direct")
    inputs, targets = batch(program, ex, signals)
    m = SoftProgram(program, r)
    free = [n.name for n in program.nodes if len(n.candidates) > 1] or ["addr"]

    noise = {"logit_noise_0": 0., "logit_noise_0.5": .5, "logit_noise_2": 2.}.get(remedy, .5)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            if p.requires_grad:
                p.add_(torch.randn(p.shape, generator=g) * noise)

    if bench not in DISCRETE:
        b = m.constants["b"]
        start = (N - 1) / 2 if remedy == "index_init_centre" else 0.
        if remedy == "index_init_random":
            start = float(torch.randint(0, N, (1,), generator=g))
        b.data = torch.tensor([float(start)])
        if remedy == "index_wide":
            m.temperatures["addr"] = (N / 2.) ** 2

    if remedy == "perturbation":
        # score every candidate by hard substitution; no gradient at all
        if bench in DISCRETE:
            sel = {}
            for name in free:
                pick, _ = perturbation_pick(m, inputs, targets, signals, name)
                sel[name] = pick
                m.trials[name] = pick
            m.trials = {}
            evals = len(free) * N
            return outcome(sel, reference, evals)
        best, bl = None, float("inf")
        with torch.no_grad():
            for i in range(N):
                m.constants["b"].copy_(torch.tensor([float(i)]))
                m.trials = {"addr": 0}
                L = float(loss_at(m, inputs, targets, signals))
                if L < bl:
                    bl, best = L, i
        m.trials = {}
        return outcome({"b": best}, reference, N)

    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr if bench in DISCRETE else .1)
    ev0 = BUDGET.node_evals
    for step in range(steps):
        frac = step / max(1, steps - 1)
        if remedy == "index_anneal":
            m.temperatures["addr"] = (N / 2.) ** 2 * (0.5 / (N / 2.) ** 2) ** frac
        if remedy == "choice_anneal":
            for n in program.nodes:
                if n.name in free:
                    m.temperatures[n.name] = 8.0 * (0.25 / 8.0) ** frac
        if remedy == "straight_through":
            if bench in DISCRETE:
                m.trials = {n: int(m.choices[i].argmax())
                            for i, n in enumerate([x.name for x in program.nodes]) if n in free}
            else:
                m.trials = {"addr": 0}
        elif remedy == "sampled":
            if bench in DISCRETE:
                m.trials = {}
                for i, n in enumerate([x.name for x in program.nodes]):
                    if n in free:
                        p = torch.softmax(m.choices[i] / m.temperatures[n], 0)
                        m.trials[n] = int(torch.multinomial(p, 1, generator=g))
            else:
                m.trials = {"addr": 0}
        opt.zero_grad()
        if remedy == "sampled" and bench not in DISCRETE:
            with torch.no_grad():
                keep = m.constants["b"].detach().clone()
                m.constants["b"].add_(torch.randn(1, generator=g) * 1.5).clamp_(0, N - 1)
        loss = loss_at(m, inputs, targets, signals)
        loss.backward()
        if remedy == "sampled" and bench not in DISCRETE:
            with torch.no_grad():
                m.constants["b"].copy_(keep)
        opt.step()
        if bench not in DISCRETE:
            with torch.no_grad():
                m.constants["b"].clamp_(0, N - 1)
    m.trials = {}
    if bench in DISCRETE:
        sel = {n.name: int(m.choices[i].argmax()) for i, n in enumerate(program.nodes)
               if n.name in free}
    else:
        sel = {"b": int(round(float(m.constants["b"].detach())))}
    return outcome(sel, reference, BUDGET.node_evals - ev0)


def outcome(sel, reference, evals):
    hit = all(sel.get(k) == v for k, v in reference.items())
    return {"selection": sel, "reference": reference, "hit": bool(hit), "node_evals": evals}


REMEDIES_DISCRETE = ["logit_noise_0", "logit_noise_0.5", "logit_noise_2", "choice_anneal",
                     "straight_through", "sampled", "perturbation", "centred_values"]
REMEDIES_INDEX = ["baseline", "index_init_centre", "index_init_random", "index_wide",
                  "index_anneal", "straight_through", "sampled", "perturbation",
                  "centred_values"]


def main():
    t0 = time.perf_counter()
    out = {"config": {"N": N, "M": M, "seeds": list(SEEDS), "chance_one_address": 1 / N,
                      "chance_two_addresses": 1 / N ** 2}}

    # a discrete reference for each benchmark, as every synthesis claim requires
    print("== enumeration reference ==", flush=True)
    refs = {}
    for name, build in BENCHES.items():
        if name not in DISCRETE:
            refs[name] = {"note": "continuous address; enumeration does not apply"}
            continue
        program, signals, ex, r, reference, _ = build(0)
        res = enumerate_fit(program, ex, signals, registry=r, tolerance=1e-3)
        refs[name] = {"space": space_size(program), **res.to_dict()}
        print(f"  {name:32s} space={space_size(program):6d} solved={res.solved} "
              f"unique={res.unique} {res.seconds:.2f}s sel={res.selections}", flush=True)
    out["enumeration"] = refs

    print("== remedies ==", flush=True)
    table = []
    for bench in BENCHES:
        remedies = REMEDIES_DISCRETE if bench in DISCRETE else REMEDIES_INDEX
        for remedy in remedies:
            rows = []
            for seed in SEEDS:
                try:
                    rows.append(run(bench, seed, remedy))
                except Exception as exc:
                    rows.append({"hit": False, "node_evals": 0, "error": repr(exc)[:160]})
            hits = sum(r["hit"] for r in rows)
            evals = sum(r["node_evals"] for r in rows) / len(rows)
            na = all(r.get("not_applicable") for r in rows)
            table.append({"bench": bench, "remedy": remedy, "hits": hits, "n": len(rows),
                          "not_applicable": na,
                          "mean_node_evals": round(evals, 1), "rows": rows})
            print(f"  {bench:32s} {remedy:18s} {"n/a" if na else str(hits)+"/"+str(len(rows)):8s}"
                  f"{evals:9.0f} node evals", flush=True)
    out["remedies"] = table

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("remedies", out)
    print("budget", out["budget"], flush=True)


if __name__ == "__main__":
    main()
