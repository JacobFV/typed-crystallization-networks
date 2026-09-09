"""Does the instrument's diagnosis apply to the task that actually failed?

The perception ladder's failing arm is `rung3:centre_free_address`, whose free
nodes choose which raw `geometry` byte to read.  This script takes those exact
bytes, measures the two data statistics the instrument identified, and then runs
the instrument's own single-address probe on them -- once as they are, and once
with the per-address mean removed.

If removing the per-address mean restores the pick, the mechanism behind the
address wall on real pixels is the same one the synthetic sweep isolated.

Run:  .venv/bin/python research/address-wall/real_data.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception-ladder"))

import torch

from tcn.learning import SoftProgram
from tcn.operators import Registry

from instrument import BUDGET, batch, discrete_program, dump, loss_at, make_examples

M = 64


def geometry_bytes(R, n_episodes=M, seed0=0):
    from rung3_geometry import examples as gex
    ex = gex(range(seed0, seed0 + n_episodes), R)
    rows = []
    for e in ex:
        rows.append([float(v) for v in e["inputs"]["pixels"].decoded[3]])
    return torch.tensor(rows, dtype=torch.float64)


def describe(x, k):
    mu = x.mean(0)
    cov = torch.cov(x.T)
    sd = cov.diagonal().clamp_min(0).sqrt()
    n = x.shape[1]
    corr = cov / (sd[:, None] * sd[None, :] + 1e-12)
    neigh = float(torch.tensor([corr[i, i + 3] for i in range(n - 3)]).mean())  # same channel, next pixel
    return {"addresses": n,
            "value_range": [float(x.min()), float(x.max())],
            "grand_mean": float(mu.mean()),
            "per_address_mean_sd": float(mu.std()),
            "mean_within_address_sd": float(sd.mean()),
            "mean_spread_over_sd": float(mu.std() / max(1e-9, sd.mean())),
            "same_channel_neighbour_correlation": neigh}


def decompose(x, k):
    """Split the logit gradient at the uniform mixture into its two terms.

    With DIRECT supervision the relaxed loss is `p'Cp - 2 (Ce_k)'p + C_kk`, so
    the candidate the first Adam step raises is `argmax_i [C_ik - mean_j C_ij]`.
    Writing `C = mu mu' + Sigma` splits that score into a MEAN term
    `mu_i (mu_k - mean_j mu_j)` and a COVARIANCE term
    `Sigma_ik - mean_j Sigma_ij`.  Whichever term decides the argmax is the
    mechanism.
    """
    n = x.shape[1]
    mu = x.mean(0)
    C = (x.T @ x) / x.shape[0]
    S = C - mu[:, None] * mu[None, :]
    full = C[:, k] - C.mean(1)
    mean_only = mu * (mu[k] - mu.mean())
    cov_only = S[:, k] - S.mean(1)
    return {"pick_full": int(full.argmax()), "pick_mean_term": int(mean_only.argmax()),
            "pick_cov_term": int(cov_only.argmax()),
            "mean_term_magnitude": float(mean_only.abs().max()),
            "cov_term_magnitude": float(cov_only.abs().max()),
            "argmax_decided_by": ("mean" if int(full.argmax()) == int(mean_only.argmax())
                                  else ("covariance" if int(full.argmax()) == int(cov_only.argmax())
                                        else "neither alone"))}


def probe(x, k, seed, steps=800, lr=.05, noise=.5):
    n = x.shape[1]
    ex = make_examples(x, k, "direct")
    r = Registry()
    program, signals = discrete_program(r, n, "direct")
    inputs, targets = batch(program, ex, signals)

    m = SoftProgram(program, r)
    m.choices[0].grad = None
    loss_at(m, inputs, targets, signals).backward()
    pick = int(m.choices[0].grad.argmin())

    m2 = SoftProgram(program, r)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        m2.choices[0].add_(torch.randn(n, generator=g) * noise)
    opt = torch.optim.Adam(m2.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss_at(m2, inputs, targets, signals).backward()
        opt.step()
    return {"gradient_pick": pick, "gradient_hit": pick == k,
            "final_pick": int(m2.choices[0].argmax()),
            "final_hit": int(m2.choices[0].argmax()) == k}


def main():
    t0 = time.perf_counter()
    out = {}
    for R in (2, 4):
        x = geometry_bytes(R)
        n = x.shape[1]
        addrs = [n // 2, n // 2 + 1, 3, n - 2]     # a spread of reference addresses
        centred = x - x.mean(0, keepdim=True)
        block = {"stats_raw": describe(x, addrs[0]),
                 "stats_centred": describe(centred, addrs[0]),
                 "addresses_probed": addrs, "chance": 1 / n, "rows": []}
        for k in addrs:
            block["rows"].append({"k": k, "variant": "raw",
                                  **decompose(x, k),
                                  **probe(x, k, seed=k)})
            block["rows"].append({"k": k, "variant": "per_address_centred",
                                  **decompose(centred, k),
                                  **probe(centred, k, seed=k)})
        out[f"geometry_R{R}"] = block
        print(f"== geometry R={R}: {n} byte addresses ==", flush=True)
        print("   raw     ", block["stats_raw"], flush=True)
        print("   centred ", block["stats_centred"], flush=True)
        for row in block["rows"]:
            print(f"   k={row['k']:3d} {row['variant']:20s} grad->{row['gradient_pick']:3d} "
                  f"hit={row['gradient_hit']!s:5s} final->{row['final_pick']:3d} "
                  f"hit={row['final_hit']!s:5s} argmax decided by {row['argmax_decided_by']}",
                  flush=True)
        for v in ("raw", "per_address_centred"):
            sub = [r for r in block["rows"] if r["variant"] == v]
            print(f"   {v:20s} gradient {sum(r['gradient_hit'] for r in sub)}/{len(sub)}"
                  f"  optimised {sum(r['final_hit'] for r in sub)}/{len(sub)}", flush=True)

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("real_data", out)


if __name__ == "__main__":
    main()
