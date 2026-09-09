"""Task 3: the scaling dial -- `generators/logic` `depth`.

Task family (identical to the one `research/search-scaling/harness.py` uses, so
the two tracks' numbers are directly comparable; that file is not imported, this
module is self-contained):

  * A real `logic` episode with configuration {'depth': d} supplies the target:
    4 input bits, `d` randomly wired two-input gates with random 16-way truth
    tables, output = last gate.  The target is its 16-row truth table.
  * The TCN scaffold is `d` Boolean nodes; node k's candidate set is EXACTLY
    what `tcn/graph.py:legal_candidates` produces for a bool-output node over
    the bounded predecessor pool {b0..b3, g0..g(k-1)}:

        |K_k| = 16 tables  x  (4 + k)^2 ordered source pairs

    so the space the four methods search is the same object:

        |space(d)| = prod_{k=0}^{d-1} 16 (4+k)^2

        d=1  2.56e2      d=2  1.02e5      d=3  5.90e7
        d=4  4.63e10     d=5  4.74e13     d=6  6.13e16

Four methods over that space:
  1. exhaustive enumeration (prefix-shared nested loops over bitmask semantics)
  2. random search (uniform over the same product space)
  3. CDCL SAT (self-contained; see sat.py) on the standard circuit-synthesis
     encoding of "choose one table and two sources per node so that node d-1
     realises the target truth table"
  4. the gradient path: SoftProgram + annealed softmax + entropy pressure,
     using the configuration `research/search-scaling/harness.py` settled on
     (lr .05, 500 steps, tau 1.0 -> 0.1 geometric anneal, init_scale .1,
     entropy_weight .001 ramped) -- i.e. the gradient side's best known config
     on this exact family, not a strawman.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.learning import SoftProgram, tensor
from tcn.generation import Host

from common import Budget, Result, Timer, write, machine
from sat import Solver, exactly_one

N_IN = 4
N_ROWS = 1 << N_IN
FULL = (1 << N_ROWS) - 1
INPUT_MASKS = tuple(sum(1 << m for m in range(N_ROWS) if (m >> j) & 1) for j in range(N_IN))


def apply_table(am, bm, table):
    na, nb = FULL ^ am, FULL ^ bm
    parts = (na & nb, na & bm, am & nb, am & bm)
    out = 0
    for i, p in enumerate(parts):
        if (table >> i) & 1:
            out |= p
    return out & FULL


def target_from_generator(depth, seed, index):
    """Draw a real `logic` episode and read its circuit + truth table."""
    host = Host.create("logic", seed=seed, index=index, configuration={"depth": depth})
    gates = [tuple(g) for g in host.state["gates"]]
    vals = list(INPUT_MASKS)
    for a, b, t in gates:
        vals.append(apply_table(vals[a], vals[b], t))
    return gates, vals[-1]


def relevant_inputs(mask):
    """How many of the four input bits the target function actually depends on."""
    n = 0
    for j in range(N_IN):
        for m in range(N_ROWS):
            if ((mask >> m) & 1) != ((mask >> (m ^ (1 << j))) & 1):
                n += 1
                break
    return n


def hard_target_from_generator(depth, seed, index, tries=20000):
    """Same generator, rejection-sampled to targets that depend on all four inputs.

    Random `logic` circuits collapse aggressively: most depth>=3 draws realise a
    function of one or two inputs.  This filter keeps the target inside the
    generator's own distribution while removing the degenerate tail.  It is the
    only difficulty knob available -- see RESULTS.md sec. 5 for the measurement
    that random depth>=4 circuits essentially never need more than three gates.
    """
    for k in range(tries):
        gates, mask = target_from_generator(depth, seed, index + 7919 * k)
        if relevant_inputs(mask) == N_IN:
            return gates, mask
    raise RuntimeError("no target with four relevant inputs")


def space_size(depth):
    return math.prod(16 * (N_IN + k) ** 2 for k in range(depth))


# --------------------------------------------------------------- 1. enumeration
def enumerate_exhaustive(depth, target, deadline=None, stop_at_first=True):
    """Complete search over the product space, prefix-shared.

    Node k ranges over (table, src_a, src_b) with sources drawn from the
    4 + k available ports, exactly the scaffold's candidate set.
    """
    tried = 0
    t0 = time.perf_counter()
    solution = None
    stack_vals = [list(INPUT_MASKS)]

    def rec(k, vals):
        nonlocal tried, solution
        if solution is not None and stop_at_first:
            return True
        n = len(vals)
        last = k == depth - 1
        for a in range(n):
            am = vals[a]
            for b in range(n):
                bm = vals[b]
                na, nb = FULL ^ am, FULL ^ bm
                parts = (na & nb, na & bm, am & nb, am & bm)
                for tb in range(16):
                    out = 0
                    if tb & 1:
                        out |= parts[0]
                    if tb & 2:
                        out |= parts[1]
                    if tb & 4:
                        out |= parts[2]
                    if tb & 8:
                        out |= parts[3]
                    if last:
                        tried += 1
                        if out == target:
                            solution = list(path) + [(tb, a, b)]
                            if stop_at_first:
                                return True
                    else:
                        path.append((tb, a, b))
                        vals.append(out)
                        hit = rec(k + 1, vals)
                        vals.pop()
                        path.pop()
                        if hit and stop_at_first:
                            return True
                        if deadline is not None and time.perf_counter() > deadline:
                            raise TimeoutError
        return False

    path = []
    timed_out = False
    with Timer() as t:
        try:
            rec(0, list(INPUT_MASKS))
        except TimeoutError:
            timed_out = True
    return {
        "solution": solution,
        "leaf_programs": tried,
        "seconds": t.seconds,
        "timed_out": timed_out,
    }


# ------------------------------------------------------------- 2. random search
def random_search(depth, target, draws, seed=0):
    rng = random.Random(seed)
    t0 = time.perf_counter()
    for tried in range(1, draws + 1):
        vals = list(INPUT_MASKS)
        for _ in range(depth):
            n = len(vals)
            vals.append(apply_table(vals[rng.randrange(n)], vals[rng.randrange(n)], rng.randrange(16)))
        if vals[-1] == target:
            return {"solved": True, "draws": tried, "seconds": time.perf_counter() - t0}
    return {"solved": False, "draws": draws, "seconds": time.perf_counter() - t0}


# ---------------------------------------------------------------------- 3. SAT
def sat_encode(depth, target):
    """Standard exact-circuit-synthesis CNF over the scaffold's candidate space.

    Variables per node k:
      t[k][m]    m in 0..3   the truth-table bit for input pattern (a,b)=(m>>1,m&1)
      sa[k][p], sb[k][p]     one-hot selectors over the 4+k available ports
      v[k][r]    r in 0..15  the node's value on input row r
    Semantics clause, for every row r, every port pair (p, q) and every
    (alpha, beta) in {0,1}^2:
      sa[k][p] & sb[k][q] & (val(p,r)=alpha) & (val(q,r)=beta) -> (v[k][r] <-> t[k][2a+b])
    The (table, source pair) product is exactly the node's candidate set; the
    factored encoding is the standard one and represents the identical space.
    """
    s = Solver()
    t = [[s.new_var() for _ in range(4)] for _ in range(depth)]
    sa, sb, v = [], [], []
    for k in range(depth):
        ports = N_IN + k
        sa.append([s.new_var() for _ in range(ports)])
        sb.append([s.new_var() for _ in range(ports)])
        v.append([s.new_var() for _ in range(N_ROWS)])
        exactly_one(s, sa[k])
        exactly_one(s, sb[k])

    def port_lit(k, p, r):
        """Literal that is TRUE iff port p of node k has value 1 on row r.
        Returns True/False for constant input ports."""
        if p < N_IN:
            return bool((INPUT_MASKS[p] >> r) & 1)
        return v[p - N_IN][r]

    for k in range(depth):
        ports = N_IN + k
        for r in range(N_ROWS):
            for p in range(ports):
                pa = port_lit(k, p, r)
                for q in range(ports):
                    qb = port_lit(k, q, r)
                    for alpha in (0, 1):
                        if isinstance(pa, bool) and pa != bool(alpha):
                            continue
                        for beta in (0, 1):
                            if isinstance(qb, bool) and qb != bool(beta):
                                continue
                            guard = [-sa[k][p], -sb[k][q]]
                            if not isinstance(pa, bool):
                                guard.append(-pa if alpha else pa)
                            if not isinstance(qb, bool):
                                guard.append(-qb if beta else qb)
                            tb = t[k][2 * alpha + beta]
                            # v <-> tb, under the guard
                            s.add_clause(guard + [-v[k][r], tb])
                            s.add_clause(guard + [v[k][r], -tb])
    for r in range(N_ROWS):
        lit = v[depth - 1][r]
        s.add_clause([lit if (target >> r) & 1 else -lit])
    return s, t, sa, sb, v


def sat_solve(depth, target, max_conflicts=None, max_seconds=120.0):
    with Timer() as build:
        s, t, sa, sb, v = sat_encode(depth, target)
    with Timer() as solve:
        model = s.solve(max_conflicts=max_conflicts, max_seconds=max_seconds)
    program = None
    if isinstance(model, dict):
        program = []
        for k in range(depth):
            table = sum((1 << m) for m in range(4) if model.get(t[k][m], False))
            a = next(i for i, x in enumerate(sa[k]) if model.get(x, False))
            b = next(i for i, x in enumerate(sb[k]) if model.get(x, False))
            program.append((table, a, b))
    return {
        "solved": program is not None,
        "unknown": model == "unknown",
        "program": program,
        "build_seconds": build.seconds,
        "solve_seconds": solve.seconds,
        "seconds": build.seconds + solve.seconds,
        "variables": s.nvars,
        "clauses": len(s.clauses) - s.learned,
        "decisions": s.decisions,
        "conflicts": s.conflicts,
        "propagations": s.propagations,
    }


def verify(depth, program, target):
    vals = list(INPUT_MASKS)
    for tb, a, b in program:
        vals.append(apply_table(vals[a], vals[b], tb))
    return vals[-1] == target


# ----------------------------------------------------------------- 4. gradient
def build_program(depth):
    r = Registry()
    inputs = tuple((f"b{j}", BOOL) for j in range(N_IN))
    names = [f"b{j}" for j in range(N_IN)]
    nodes = []
    for k in range(depth):
        pairs = list(itertools.product(names, repeat=2))
        cands = tuple(Candidate(r.resolve(f"truth_{tb}", (BOOL, BOOL)), p) for p in pairs for tb in range(16))
        nodes.append(Node(f"g{k}", BOOL, cands, "core", k + 1))
        names.append(f"g{k}")
    return Program(inputs, tuple(nodes), (("y", f"g{depth-1}"),)).validate(r), r


def selection_mask(program, selections):
    vals = {f"b{j}": INPUT_MASKS[j] for j in range(N_IN)}
    for n in program.nodes:
        c = n.candidates[selections[n.name]]
        vals[n.name] = apply_table(vals[c.sources[0]], vals[c.sources[1]], int(c.operator.name[6:]))
    return vals[program.outputs[0][1]]


def gradient(depth, target, seed, steps=500, lr=0.05, tau0=1.0, tau1=0.1,
             init_scale=0.1, entropy_weight=0.001):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    program, registry = build_program(depth)
    exs = []
    for m in range(N_ROWS):
        exs.append({
            "inputs": {f"b{j}": Value.of(BOOL, bool((m >> j) & 1)) for j in range(N_IN)},
            "targets": {"y": Value.of(BOOL, bool((target >> m) & 1))},
        })
    signals = (Signal(f"g{depth-1}", "y", ("core",), BOOL, "bce"),)
    model = SoftProgram(program, registry)
    with torch.no_grad():
        for p in model.choices:
            p.normal_(0.0, init_scale)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in exs]) for k, _ in program.inputs}
    targets = {"y": torch.stack([tensor(e["targets"]["y"]) for e in exs])}
    sizes = [len(n.candidates) for n in program.nodes]
    ops_per_forward = sum(sizes) * N_ROWS
    first = None
    with Timer() as t:
        for step in range(steps):
            tau = tau0 * (tau1 / tau0) ** (step / max(1, steps - 1))
            for n in program.nodes:
                model.temperatures[n.name] = tau
            opt.zero_grad()
            _, _, tr = model(inputs, return_trace=True)
            loss = model.probe_loss(tr, targets, signals)
            (loss + entropy_weight * (step / max(1, steps)) * model.entropy()).backward()
            opt.step()
            if step % 5 == 0 or step == steps - 1:
                if selection_mask(program, model.selections()) == target and first is None:
                    first = step
    got = selection_mask(program, model.selections())
    return {
        "solved": got == target,
        "final_mask": got,
        "first_success_step": first,
        "seconds": t.seconds,
        "steps": steps,
        "op_applications": 3 * steps * ops_per_forward,
        "candidates_per_node": sizes,
        "log2_space": sum(math.log2(s) for s in sizes),
        "hamming": bin((got ^ target) & FULL).count("1"),
    }


# ---------------------------------------------------------------------- driver
def solution_density(depth, target, draws=200000, seed=0):
    """Fraction of the discrete space that solves the target (Monte Carlo)."""
    rng = random.Random(seed)
    hits = 0
    for _ in range(draws):
        vals = list(INPUT_MASKS)
        for _ in range(depth):
            n = len(vals)
            vals.append(apply_table(vals[rng.randrange(n)], vals[rng.randrange(n)], rng.randrange(16)))
        if vals[-1] == target:
            hits += 1
    return hits / draws


def sweep(depths=(1, 2, 3, 4, 5, 6), targets_per_depth=2, gradient_seeds=3,
          gradient_max_depth=4,
          enum_budget_seconds=60.0, sat_conflicts=200000, sat_seconds=60.0, source="generator",
          gradient_steps=500, density_draws=100000):
    """source='generator' draws targets from generators/logic at that depth;
    source='uniform' draws uniform 4-input Boolean functions -- outside the
    generator's own distribution, used to probe the hard end of the space."""
    rows = []
    for d in depths:
        size = space_size(d)
        for ti in range(targets_per_depth):
            if source == "generator":
                gates, target = target_from_generator(d, 0, 1000 * ti + d)
            elif source == "hard":
                gates, target = hard_target_from_generator(d, 0, 1000 * ti + d)
            else:
                gates, target = None, random.Random(9871 * ti + d).randrange(1 << N_ROWS)
            rec = {"depth": d, "target_index": ti, "target_mask": target,
                   "target_source": source,
                   "space_size": size, "log2_space": math.log2(size),
                   "generator_gates": gates}
            # 1. exhaustive enumeration, first solution
            e = enumerate_exhaustive(d, target, deadline=time.perf_counter() + enum_budget_seconds)
            rec["enumeration"] = {
                "solved": e["solution"] is not None,
                "seconds": e["seconds"],
                "leaf_programs": e["leaf_programs"],
                "timed_out": e["timed_out"],
                "verified": bool(e["solution"]) and verify(d, e["solution"], target),
            }
            # 2. random search at the gradient run's own evaluation budget.
            #    One gradient step evaluates sum_k |K_k| candidate operators on
            #    16 rows; a backward pass is charged 2x.  One random draw
            #    evaluates d operators on 16 rows.  Matched draws therefore =
            #    3 * steps * sum_k |K_k| / d.
            sizes = [16 * (N_IN + k) ** 2 for k in range(d)]
            draws = max(1, int(3 * gradient_steps * sum(sizes) / d))
            rs = random_search(d, target, draws, seed=ti)
            rs["draws_allowed"] = draws
            rec["random"] = rs
            rec["solution_density"] = solution_density(d, target, draws=density_draws, seed=ti)
            # 3. SAT
            rec["sat"] = sat_solve(d, target, max_conflicts=sat_conflicts, max_seconds=sat_seconds)
            if rec["sat"]["program"]:
                rec["sat"]["verified"] = verify(d, rec["sat"]["program"], target)
            # 4. gradient
            gs = ([gradient(d, target, seed=s, steps=gradient_steps) for s in range(gradient_seeds)]
                  if d <= gradient_max_depth else [])
            if not gs:
                rec["gradient"] = {"skipped": True, "reason": f"depth > gradient_max_depth={gradient_max_depth}",
                                   "success_rate": None, "seconds_median": None}
                rows.append(rec)
                print(f"d={d} t={ti} space=2^{math.log2(size):.1f} "
                      f"enum={'OK' if rec['enumeration']['solved'] else ('TIMEOUT' if rec['enumeration']['timed_out'] else 'none')} "
                      f"{rec['enumeration']['seconds']:.3f}s | "
                      f"sat={'OK' if rec['sat']['solved'] else ('unknown' if rec['sat']['unknown'] else 'UNSAT')} "
                      f"{rec['sat']['seconds']:.3f}s ({rec['sat']['conflicts']} confl) | "
                      f"rand={'OK' if rs['solved'] else 'miss'} {rs['seconds']:.2f}s | "
                      f"dens={rec['solution_density']:.1e} | grad=skipped", flush=True)
                continue
            rec["gradient"] = {
                "success_rate": sum(g["solved"] for g in gs) / len(gs),
                "seconds_median": sorted(g["seconds"] for g in gs)[len(gs) // 2],
                "seconds_total_all_seeds": sum(g["seconds"] for g in gs),
                "op_applications_per_seed": gs[0]["op_applications"],
                "candidates_per_node": gs[0]["candidates_per_node"],
                "runs": gs,
            }
            rows.append(rec)
            print(f"d={d} t={ti} space=2^{math.log2(size):.1f} "
                  f"enum={'OK' if rec['enumeration']['solved'] else ('TIMEOUT' if rec['enumeration']['timed_out'] else 'none')} "
                  f"{rec['enumeration']['seconds']:.3f}s | "
                  f"sat={'OK' if rec['sat']['solved'] else ('unknown' if rec['sat']['unknown'] else 'UNSAT')} "
                  f"{rec['sat']['seconds']:.3f}s ({rec['sat']['conflicts']} confl) | "
                  f"rand={'OK' if rs['solved'] else 'miss'} {rs['seconds']:.2f}s | "
                  f"dens={rec['solution_density']:.1e} | "
                  f"grad={rec['gradient']['success_rate']:.2f} "
                  f"{rec['gradient']['seconds_median']:.2f}s", flush=True)
    return rows


def main():
    source = "generator"
    args = sys.argv[1:]
    if args and args[0] in ("generator", "hard", "uniform"):
        source = args.pop(0)
    depths = [int(x) for x in args] or [1, 2, 3, 4, 5, 6]
    rows = sweep(depths=depths, source=source)
    print(write(f"scaling_{source}_d{'_'.join(map(str, depths))}.json",
                {"machine": machine(), "source": source, "rows": rows}))


if __name__ == "__main__":
    main()
