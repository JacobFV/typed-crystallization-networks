"""Task 2 -- test the stated explanation.

The explanation on record is: "a mixture of candidate operators is a blend of
functions at a valid input, while a mixture of candidate addresses is a blend of
unrelated values and denotes nothing."

Read as a prediction about data, it says two things:

  E1  the failure gets worse as the addressed values become less correlated;
  E2  it largely vanishes when neighbouring addresses hold similar values.

Both are tested here with everything else held fixed:

  Part 1  a smoothly varying array against the SAME array with its positions
          randomly permuted.  Identical value multiset, identical per-address
          marginals, identical second-moment spectrum; only the arrangement
          differs.  This is the cleanest possible form of E1/E2.
  Part 2  a two-factor sweep over neighbour correlation length and per-address
          mean spread.  If E1/E2 are right, only the first factor should move
          the outcome.
  Part 3  the same sweep for the `index` operator, whose relaxation is local
          rather than global.

Run:  .venv/bin/python research/address-wall/explanation.py
"""
from __future__ import annotations

import time

import torch

from tcn.learning import SoftProgram
from tcn.operators import Registry

from instrument import (BUDGET, batch, discrete_program, dump, gp_arrays,
                        gradient_pick, index_program, loss_at, make_examples)

N = 16
M = 64
K = 11
SEEDS = range(12)


# --------------------------------------------------------------------------

def discrete_probe(arrays, k, seed, steps=400, lr=.05, noise=.5):
    """Gradient pick at the uniform mixture, and the outcome of a full run."""
    ex = make_examples(arrays, k, "direct")
    r = Registry()
    program, signals = discrete_program(r, N, "direct")
    inputs, targets = batch(program, ex, signals)

    m = SoftProgram(program, r)
    pick, grad = gradient_pick(m, inputs, targets, signals)

    m2 = SoftProgram(program, r)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m2.choices:
            p.add_(torch.randn(p.shape, generator=g) * noise)
    opt = torch.optim.Adam(m2.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_at(m2, inputs, targets, signals).backward()
        opt.step()
    return {"gradient_pick": pick, "gradient_hit": pick == k,
            "final_pick": int(m2.choices[0].argmax()),
            "final_hit": int(m2.choices[0].argmax()) == k}


def index_probe(arrays, k, seed, steps=600, lr=.1, temperature=1.0, stride=1):
    """Adam on the continuous address from every integer start."""
    ex = make_examples(arrays, k, "direct")
    r = Registry()
    program, signals = index_program(r, N, "direct")
    inputs, targets = batch(program, ex, signals)
    m = SoftProgram(program, r)
    m.temperatures["addr"] = temperature
    b = m.constants["b"]
    hits = 0
    starts = [x for x in range(0, N, stride) if x != k]
    for x in starts:
        b.data = torch.tensor([float(x)])
        opt = torch.optim.Adam([b], lr=lr)
        for _ in range(steps):
            opt.zero_grad()
            loss_at(m, inputs, targets, signals).backward()
            opt.step()
            with torch.no_grad():
                b.clamp_(0, N - 1)
        hits += abs(float(b.detach()) - k) < 0.5
    return {"starts": len(starts), "reached": hits / len(starts)}


def describe(arrays, k):
    x = arrays.double()
    mu = x.mean(0)
    cov = torch.cov(x.T)
    sd = cov.diagonal().sqrt()
    corr = cov / (sd[:, None] * sd[None, :] + 1e-12)
    neigh = float(torch.tensor([corr[i, i + 1] for i in range(N - 1)]).mean())
    return {"mean_spread_over_sd": float(mu.std() / sd.mean()),
            "neighbour_correlation": neigh,
            "mean_abs_offdiag_correlation": float(
                (corr - torch.eye(N, dtype=torch.float64)).abs().sum() / (N * (N - 1)))}


# --------------------------------------------------------------------------
# Part 1: smooth vs the same array permuted
# --------------------------------------------------------------------------

def permutation_pair():
    rows = []
    for seed in SEEDS:
        base = gp_arrays(N, M, corr_length=3.0, mean_spread=0.0, seed=seed * 97 + 11)
        g = torch.Generator().manual_seed(seed * 13 + 7)
        perm = torch.randperm(N, generator=g)
        permuted = base[:, perm]
        k_perm = int((perm == K).nonzero()[0])   # same physical value, new address
        for label, arr, k in (("smooth", base, K), ("permuted", permuted, k_perm)):
            d = discrete_probe(arr, k, seed)
            i = index_probe(arr, k, seed)
            rows.append({"arrangement": label, "seed": seed, "k": k,
                         **describe(arr, k), **d,
                         "index_reached": i["reached"]})
    return rows


# --------------------------------------------------------------------------
# Part 2/3: two-factor sweep
# --------------------------------------------------------------------------

CORRS = (0.0, 0.5, 1.0, 2.0, 4.0, 8.0)
SPREADS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)


def sweep(mechanism):
    rows = []
    for corr in CORRS:
        for spread in SPREADS:
            hits = finals = reached = 0.0
            n = 0
            for seed in SEEDS:
                arr = gp_arrays(N, M, corr, spread, seed=seed * 97 + 11)
                if mechanism == "discrete":
                    d = discrete_probe(arr, K, seed)
                    hits += d["gradient_hit"]; finals += d["final_hit"]
                else:
                    reached += index_probe(arr, K, seed, steps=300, stride=2)["reached"]
                n += 1
            rows.append({"mechanism": mechanism, "corr_length": corr,
                         "mean_spread": spread, "n": n,
                         "gradient_hit_rate": hits / n,
                         "final_hit_rate": finals / n,
                         "index_reached": reached / n})
            print(f"  {mechanism:8s} corr={corr:4.1f} spread={spread:4.2f} "
                  f"grad={hits/n:.3f} final={finals/n:.3f} index={reached/n:.3f}", flush=True)
    return rows


def main():
    t0 = time.perf_counter()
    out = {"config": {"N": N, "M": M, "K": K, "seeds": list(SEEDS),
                      "chance": 1 / N, "corrs": CORRS, "spreads": SPREADS}}

    print("== Part 1: a smooth array against the same array, positions permuted ==", flush=True)
    rows = permutation_pair()
    out["permutation_pair"] = rows
    for label in ("smooth", "permuted"):
        sub = [r for r in rows if r["arrangement"] == label]
        print(f"  {label:9s} neighbour-corr {sum(r['neighbour_correlation'] for r in sub)/len(sub):+.3f}"
              f"  gradient->reference {sum(r['gradient_hit'] for r in sub)}/{len(sub)}"
              f"  optimised->reference {sum(r['final_hit'] for r in sub)}/{len(sub)}"
              f"  index descent reaches {sum(r['index_reached'] for r in sub)/len(sub):.3f}",
              flush=True)

    print("== Part 2: discrete route, correlation x mean spread ==", flush=True)
    out["sweep_discrete"] = sweep("discrete")
    print("== Part 3: index route, correlation x mean spread ==", flush=True)
    out["sweep_index"] = sweep("index")

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("explanation", out)
    print("budget", out["budget"], flush=True)


if __name__ == "__main__":
    main()
