"""Task 1 -- characterise the failure precisely.

Maps the relaxed loss as a function of the address parameters over the whole
space, for both mechanisms, and distinguishes three possibilities that "worse
than chance" conflates:

  FLAT       the reference is the optimum but the gradient carries no signal;
  MISLEADING the gradient carries signal and it points somewhere else;
  DISPLACED  the reference is not the optimum of the relaxed loss at all.

Run:  .venv/bin/python research/address-wall/landscape.py
"""
from __future__ import annotations

import itertools
import time

import torch

from tcn.learning import SoftProgram
from tcn.operators import Registry

from instrument import (BUDGET, REGIMES, batch, discrete_program, dump,
                        eq_arrays, gp_arrays, gradient_pick, index_program,
                        loss_at, make_examples, quadratic_form, EQ_REGIMES)

N = 16          # addresses
M = 64          # examples
K = 11          # the reference address
SEEDS = range(8)


def direct_arrays(regime, seed):
    corr, spread, scale, offset = REGIMES[regime]
    return gp_arrays(N, M, corr, spread, seed=seed * 97 + 11, scale=scale, offset=offset)


def eq_constant_for(arrays, k):
    col = arrays[:, k].tolist()
    vals = sorted(set(col))
    return min(vals, key=lambda v: abs(sum(1 for x in col if x == v) / len(col) - 0.5))


def problem(regime, seed, downstream, k=K):
    if downstream == "direct":
        arrays = direct_arrays(regime, seed)
        c = 0.0
    else:
        arrays = eq_arrays(regime, N, M, seed=seed * 97 + 11)
        c = eq_constant_for(arrays, k)
    ex = make_examples(arrays, k, downstream, c)
    base_rate = (sum(float(e["targets"]["y"].raw) for e in ex) / len(ex)
                 if downstream == "eq" else None)
    return arrays, c, ex, base_rate


# --------------------------------------------------------------------------
# discrete route
# --------------------------------------------------------------------------

def discrete_row(regime, seed, downstream):
    arrays, c, ex, base_rate = problem(regime, seed, downstream)
    r = Registry()
    program, signals = discrete_program(r, N, downstream, c)
    inputs, targets = batch(program, ex, signals)
    m = SoftProgram(program, r)

    def loss_with(logits):
        with torch.no_grad():
            m.choices[0].copy_(logits)
            return float(loss_at(m, inputs, targets, signals))

    onehot = lambda i: torch.full((N,), -40.).index_fill_(0, torch.tensor([i]), 40.)
    vertex = [loss_with(onehot(i)) for i in range(N)]
    ref_rank = sorted(range(N), key=lambda i: vertex[i]).index(K)
    uniform_loss = loss_with(torch.zeros(N))

    # the gradient at the uniform mixture -- the quantity `pointing.py` measures
    m.choices[0].data = torch.zeros(N)
    pick, grad = gradient_pick(m, inputs, targets, signals)

    # random mixtures: does any interior point of the simplex beat the reference?
    g = torch.Generator().manual_seed(seed * 31 + 5)
    draws, better, best_mix = 400, 0, float("inf")
    dir_ = torch.distributions.Dirichlet(torch.ones(N))
    for _ in range(draws):
        w = dir_.sample()
        L = loss_with(w.clamp_min(1e-12).log())
        best_mix = min(best_mix, L)
        better += L < vertex[K] - 1e-9

    final, traj = optimise_discrete(program, r, inputs, targets, signals, seed)

    row = {
        "regime": regime, "seed": seed, "downstream": downstream,
        "base_rate": base_rate,
        "vertex_loss_reference": vertex[K],
        "vertex_loss_best": min(vertex),
        "vertex_argmin": int(min(range(N), key=lambda i: vertex[i])),
        "reference_rank_among_vertices": ref_rank,
        "uniform_loss": uniform_loss,
        "best_random_mixture_loss": best_mix,
        "mixtures_better_than_reference": better / draws,
        "gradient_pick": pick,
        "gradient_pick_is_reference": pick == K,
        "gradient_spread": None if grad is None else float(grad.std()),
        "gradient_abs_sum": None if grad is None else float(grad.abs().sum()),
        "optimised_pick": final,
        "optimised_pick_is_reference": final == K,
        "trajectory": traj,
    }
    if downstream == "direct":
        C, Ck, Ckk = quadratic_form(arrays, K)
        closed = Ck - C.mean(dim=1)
        row["closed_form_pick"] = int(closed.argmax())
        row["closed_form_matches_autograd"] = int(closed.argmax()) == pick
        row["second_moment_min_eig"] = float(torch.linalg.eigvalsh(C).min())
        row["mean_term"] = float((arrays.mean(0) * (arrays.mean(0)[K] - arrays.mean(0).mean())).abs().mean())
        row["cov_term"] = float(torch.cov(arrays.T)[K].abs().mean())
    return row


def optimise_discrete(program, registry, inputs, targets, signals, seed, steps=400, lr=.05,
                      noise=.5):
    m = SoftProgram(program, registry)
    g = torch.Generator().manual_seed(seed * 7 + 3)
    with torch.no_grad():
        for p in m.choices:
            p.add_(torch.randn(p.shape, generator=g) * noise)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    traj = []
    for step in range(steps):
        opt.zero_grad()
        loss = loss_at(m, inputs, targets, signals)
        loss.backward()
        opt.step()
        if step % 100 == 0:
            traj.append({"step": step, "loss": round(float(loss.detach()), 6),
                         "argmax": int(m.choices[0].argmax())})
    return int(m.choices[0].argmax()), traj


# --------------------------------------------------------------------------
# exhaustive simplex map, four addresses
# --------------------------------------------------------------------------

def simplex_map(regime, seed, downstream, n=4, k=2, steps=25):
    if downstream == "direct":
        corr, spread, scale, offset = REGIMES[regime]
        arrays = gp_arrays(n, M, corr, spread, seed=seed * 97 + 11, scale=scale, offset=offset)
        c = 0.0
    else:
        arrays = eq_arrays(regime, n, M, seed=seed * 97 + 11)
        c = eq_constant_for(arrays, k)
    ex = make_examples(arrays, k, downstream, c)
    r = Registry()
    program, signals = discrete_program(r, n, downstream, c)
    inputs, targets = batch(program, ex, signals)
    m = SoftProgram(program, r)

    best = (float("inf"), None)
    ref_loss = None
    total = 0
    for combo in itertools.product(range(steps + 1), repeat=n - 1):
        if sum(combo) > steps:
            continue
        w = torch.tensor([*combo, steps - sum(combo)], dtype=torch.float32) / steps
        with torch.no_grad():
            m.choices[0].copy_(w.clamp_min(1e-12).log())
            L = float(loss_at(m, inputs, targets, signals))
        total += 1
        if L < best[0]:
            best = (L, [round(v, 3) for v in w.tolist()])
        if float(w[k]) > 0.999:
            ref_loss = L
    return {"regime": regime, "seed": seed, "downstream": downstream, "n": n,
            "grid_points": total, "grid_step": 1 / steps,
            "reference_vertex_loss": ref_loss,
            "global_min_loss": best[0], "global_min_weights": best[1],
            "global_min_is_reference_vertex": bool(best[1] and best[1][k] > 0.999),
            "reference_excess": (ref_loss - best[0]) if ref_loss is not None else None}


# --------------------------------------------------------------------------
# index route: the whole parameter space is one dimension, so the map is exact
# --------------------------------------------------------------------------

def index_map(regime, seed, downstream, k=K, temperature=1.0, eq_temperature=1.0,
              step=0.005, descend=True):
    """Every value of the one continuous address parameter, on a 0.005 grid.

    Two controls are carried, because a naive "does the gradient point at the
    reference" question on a bounded interval is confounded: the softmax kernel
    is truncated at both ends of the tuple, which concentrates it and raises the
    loss there, so *any* target produces a gradient pushing the address inward.
    Every configuration is therefore run at two reference addresses on opposite
    sides of the midpoint, and the plateau statistics are reported so that a
    basin can be told apart from the boundary ramp.
    """
    arrays, c, ex, base_rate = problem(regime, seed, downstream, k)
    r = Registry()
    program, signals = index_program(r, N, downstream, c)
    inputs, targets = batch(program, ex, signals)
    m = SoftProgram(program, r)
    m.temperatures["addr"] = temperature
    if "hit" in m.temperatures:
        m.temperatures["hit"] = eq_temperature
    b = m.constants["b"]

    bs = torch.arange(0, N - 1 + step / 2, step)
    with torch.no_grad():
        losses = []
        for x in bs.tolist():
            b.copy_(torch.tensor([x]))
            losses.append(float(loss_at(m, inputs, targets, signals)))
    lt = torch.tensor(losses)
    minima = [i for i in range(1, len(losses) - 1)
              if losses[i] <= losses[i - 1] and losses[i] < losses[i + 1]]
    at_k = int(round(k / step))
    gmin = int(lt.argmin())
    near = min((abs(i - at_k) for i in minima), default=None)

    # basin: the widest interval around the reference on which the exhaustive
    # map's own slope points at the reference
    d = lt[1:] - lt[:-1]
    lo = at_k
    while lo > 1 and d[lo - 1] < 0:
        lo -= 1
    hi = at_k
    while hi < len(d) - 1 and d[hi] > 0:
        hi += 1
    basin = (hi - lo) * step

    # plateau: everything more than 1.5 addresses away from the reference and
    # away from both ends, so the truncation ramp is excluded
    far = [v for i, v in enumerate(losses)
           if abs(i - at_k) > 1.5 / step and 1.5 / step < i < len(losses) - 1.5 / step]
    plateau = torch.tensor(far) if far else torch.tensor([float("nan")])

    starts = [x for x in torch.arange(0, N - 1 + 1e-9, 1.0).tolist() if abs(x - k) > 1e-9]
    toward = 0
    for x in starts:
        b.data = torch.tensor([float(x)])
        b.grad = None
        loss = loss_at(m, inputs, targets, signals)
        loss.backward()
        gv = 0.0 if b.grad is None else float(b.grad)
        toward += gv != 0.0 and ((x < k and gv < 0) or (x > k and gv > 0))

    reached = None
    if descend:
        hits = 0
        for x in starts:
            b.data = torch.tensor([float(x)])
            opt = torch.optim.Adam([b], lr=.1)
            for _ in range(400):
                opt.zero_grad()
                loss = loss_at(m, inputs, targets, signals)
                loss.backward()
                opt.step()
                with torch.no_grad():
                    b.clamp_(0, N - 1)
            hits += abs(float(b.detach()) - k) < 0.5
        reached = hits / len(starts)
    b.data = torch.tensor([0.])

    return {"regime": regime, "seed": seed, "downstream": downstream, "k": k,
            "temperature": temperature, "eq_temperature": eq_temperature,
            "base_rate": base_rate,
            "grid_points": len(losses),
            "loss_at_reference": losses[at_k],
            "global_min_b": round(float(bs[gmin]), 3), "global_min_loss": losses[gmin],
            "reference_is_global_min": abs(float(bs[gmin]) - k) < 0.5,
            "local_minima": len(minima),
            "nearest_local_min_distance": None if near is None else round(near * step, 3),
            "local_minima_positions": [round(float(bs[i]), 2) for i in minima][:40],
            "basin_width": round(basin, 3),
            "plateau_median": float(plateau.median()), "plateau_std": float(plateau.std()),
            "depth_in_plateau_sds": (float(plateau.median()) - losses[at_k]) / max(1e-12, float(plateau.std())),
            "loss_range": [min(losses), max(losses)],
            "starts": len(starts),
            "gradient_points_toward_reference": toward / len(starts),
            "descent_reaches_reference": reached,
            "curve_every_0.1": [round(v, 6) for v in losses[::20]]}


# --------------------------------------------------------------------------

def main():
    t0 = time.perf_counter()
    out = {"config": {"N": N, "M": M, "K": K, "seeds": list(SEEDS)}}

    print("== A. discrete route (project candidates), direct supervision ==", flush=True)
    out["discrete_direct"] = [discrete_row(rg, s, "direct") for rg in REGIMES for s in SEEDS]
    for rg in REGIMES:
        rows = [r for r in out["discrete_direct"] if r["regime"] == rg]
        print(f"  {rg:20s} grad->ref {sum(r['gradient_pick_is_reference'] for r in rows)}/{len(rows)}"
              f"  opt->ref {sum(r['optimised_pick_is_reference'] for r in rows)}/{len(rows)}"
              f"  ref vertex rank {[r['reference_rank_among_vertices'] for r in rows]}"
              f"  interior better {max(r['mixtures_better_than_reference'] for r in rows):.3f}",
              flush=True)

    print("== B. discrete route, eq downstream ==", flush=True)
    out["discrete_eq"] = [discrete_row(rg, s, "eq") for rg in EQ_REGIMES for s in SEEDS]
    for rg in EQ_REGIMES:
        rows = [r for r in out["discrete_eq"] if r["regime"] == rg]
        print(f"  {rg:20s} grad->ref {sum(r['gradient_pick_is_reference'] for r in rows)}/{len(rows)}"
              f"  opt->ref {sum(r['optimised_pick_is_reference'] for r in rows)}/{len(rows)}"
              f"  ref vertex rank {[r['reference_rank_among_vertices'] for r in rows]}"
              f"  interior better {max(r['mixtures_better_than_reference'] for r in rows):.3f}",
              flush=True)

    print("== C. exhaustive simplex map, 4 addresses ==", flush=True)
    rows = []
    for rg in ("iid_centred", "smooth_centred", "bytelike"):
        for s in range(3):
            rows.append(simplex_map(rg, s, "direct"))
    for rg in EQ_REGIMES:
        for s in range(3):
            rows.append(simplex_map(rg, s, "eq"))
    out["simplex_map"] = rows
    for row in rows:
        print(f"  {row['regime']:20s} {row['downstream']:6s} seed={row['seed']} "
              f"pts={row['grid_points']} ref={row['reference_vertex_loss']:.5f} "
              f"min={row['global_min_loss']:.5f} at {row['global_min_weights']} "
              f"ref_is_min={row['global_min_is_reference_vertex']}", flush=True)

    print("== D. index route, exhaustive 1-D map (two reference addresses) ==", flush=True)
    imaps = []
    for kk in (11, 4):
        for rg in REGIMES:
            for s in range(4):
                imaps.append(index_map(rg, s, "direct", k=kk))
        for rg in EQ_REGIMES:
            for s in range(4):
                imaps.append(index_map(rg, s, "eq", k=kk))
    out["index_map"] = imaps
    for rg in REGIMES:
        for ds in ("direct", "eq"):
            rows = [r for r in imaps if r["regime"] == rg and r["downstream"] == ds]
            if not rows:
                continue
            print(f"  {rg:20s} {ds:6s} gmin=ref {sum(r['reference_is_global_min'] for r in rows)}/{len(rows)}"
                  f"  #minima {[r['local_minima'] for r in rows]}"
                  f"  nearest-min {[r['nearest_local_min_distance'] for r in rows]}"
                  f"  basin {sum(r['basin_width'] for r in rows)/len(rows):.2f}"
                  f"  depth/sd {sum(r['depth_in_plateau_sds'] for r in rows)/len(rows):.2f}"
                  f"  grad-toward {sum(r['gradient_points_toward_reference'] for r in rows)/len(rows):.3f}"
                  f"  descent-reaches {sum(r['descent_reaches_reference'] for r in rows)/len(rows):.3f}",
                  flush=True)

    BUDGET.stop()
    out["budget"] = BUDGET.to_dict()
    out["wall_seconds"] = round(time.perf_counter() - t0, 2)
    dump("landscape", out)
    print("budget", out["budget"], flush=True)


if __name__ == "__main__":
    main()
