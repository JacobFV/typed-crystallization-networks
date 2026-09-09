"""Track 5: does crystallized-module reuse measurably help composite learning?

Target (compositional, expressible by this repo, adder-shaped):

  sub-function  HA(a,b)      -> (a xor b, a and b)          [the 2-bit primitive]
  composite     FA(a,b,cin)  -> (sum, carry)                [uses HA twice + one or]

  FA is the canonical two-half-adder decomposition:
      (s1, c1) = HA(a, b)
      (sum,c2) = HA(s1, cin)
      carry    = c1 or c2
  Flat, with two-input gates only, it needs 5 gates:
      t = a xor b ; sum = t xor cin ; carry = (a and b) or (t and cin)

Why this target:
  * The sub-function recurs *more than once* inside the composite, which is the
    thing the recursive-abstraction claim is about.
  * HA has two outputs, so `module:<digest>` is a genuine single typed candidate
    (a one-output module would resolve to product(BOOL), see accounting_check.py
    M1), and it is NOT already a primitive operator -- unlike `xor`, every
    2-input Boolean function of which is available as `truth_k`.
  * 3 Boolean inputs => the complete 8-row truth table is the training set AND
    the exact-conformance set, so "success" is exact program equivalence, not a
    held-out estimate.

Two arms share ONE scaffold with identical node set, identical depths and an
identical Boolean operator vocabulary. The only difference: the two
tuple-producing "call slots" offer `module:HA` as an extra candidate in arm B.
Arm A's call slots can only build `tuple(p,q)` (an inert pass-through pair), so
arm A retains exactly the same expressive power via its Boolean nodes.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tcn.types import BOOL, Value, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.learning import SoftProgram, tensor
from tcn.crystallize import Crystallizer

PAIR = product(BOOL, BOOL)
BOOL_OPS = ("and", "or", "xor")          # binary; functionally complete with not
UNARY_OPS = ("not", "identity")


# --------------------------------------------------------------------------
# candidate enumeration (uses registry.resolve for every contract; no tcn edits)
# --------------------------------------------------------------------------
def bool_candidates(r, ports, extra=()):
    """All legal BOOL-valued applications over `ports` (name -> Type)."""
    out = list(extra)
    names = [k for k, t in ports.items() if t == BOOL]
    for op in BOOL_OPS:
        o = r.resolve(op, (BOOL, BOOL))
        for p, q in itertools.product(names, repeat=2):
            out.append(Candidate(o, (p, q)))
    for op in UNARY_OPS:
        o = r.resolve(op, (BOOL,))
        for p in names:
            out.append(Candidate(o, (p,)))
    return tuple(out)


def projections(r, source, source_type):
    return [
        Candidate(r.resolve("project", (source_type,), source_type.items[i], {"index": i}), (source,))
        for i in range(len(source_type.items))
    ]


def pair_candidates(r, ports, module_name=None):
    """PAIR-valued candidates for a call slot: tuple(p,q), plus module(p,q) in arm B."""
    names = [k for k, t in ports.items() if t == BOOL]
    tup = r.resolve("tuple", (BOOL, BOOL))
    out = [Candidate(tup, (p, q)) for p, q in itertools.product(names, repeat=2)]
    if module_name is not None:
        mop = r.resolve(module_name, (BOOL, BOOL))
        assert mop.output == PAIR, mop.output
        out += [Candidate(mop, (p, q)) for p, q in itertools.product(names, repeat=2)]
    return tuple(out)


# --------------------------------------------------------------------------
# stage 1: learn the half adder on its own simpler task
# --------------------------------------------------------------------------
def half_adder_scaffold(r):
    ports = {"a": BOOL, "b": BOOL}
    nodes = [Node("h", BOOL, bool_candidates(r, ports), "core", 1)]
    ports2 = dict(ports, h=BOOL)
    nodes.append(Node("s", BOOL, bool_candidates(r, ports2), "core", 2))
    nodes.append(Node("c", BOOL, bool_candidates(r, ports2), "core", 2))
    return Program((("a", BOOL), ("b", BOOL)), tuple(nodes), (("sum", "s"), ("carry", "c"))).validate(r)


def half_adder_examples():
    rows = []
    for a in (False, True):
        for b in (False, True):
            rows.append(
                {
                    "inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)},
                    "targets": {"sum": Value.of(BOOL, a != b), "carry": Value.of(BOOL, a and b)},
                }
            )
    return rows


HA_SIGNALS = (
    Signal("s", "sum", ("core",), BOOL, "bce"),
    Signal("c", "carry", ("core",), BOOL, "bce"),
)


# --------------------------------------------------------------------------
# composite scaffold, shared by both arms
# --------------------------------------------------------------------------
def composite_scaffold(r, module_name=None):
    inputs = (("a", BOOL), ("b", BOOL), ("cin", BOOL))
    ports = {"a": BOOL, "b": BOOL, "cin": BOOL}
    nodes = []

    nodes.append(Node("call1", PAIR, pair_candidates(r, ports, module_name), "core", 1))
    proj1 = projections(r, "call1", PAIR)
    nodes.append(Node("p1s", BOOL, bool_candidates(r, ports, proj1), "core", 2))
    nodes.append(Node("p1c", BOOL, bool_candidates(r, ports, proj1), "core", 2))

    ports2 = dict(ports, p1s=BOOL, p1c=BOOL)
    nodes.append(Node("call2", PAIR, pair_candidates(r, ports2, module_name), "core", 3))
    proj2 = projections(r, "call2", PAIR)
    nodes.append(Node("p2s", BOOL, bool_candidates(r, ports2, proj2), "core", 4))
    nodes.append(Node("p2c", BOOL, bool_candidates(r, ports2, proj2), "core", 4))

    ports3 = dict(ports2, p2s=BOOL, p2c=BOOL)
    nodes.append(Node("w1", BOOL, bool_candidates(r, ports3), "core", 5))
    ports4 = dict(ports3, w1=BOOL)
    nodes.append(Node("w2", BOOL, bool_candidates(r, ports4), "core", 6))
    ports5 = dict(ports4, w2=BOOL)
    nodes.append(Node("sum_out", BOOL, bool_candidates(r, ports5), "core", 7))
    nodes.append(Node("carry_out", BOOL, bool_candidates(r, ports5), "core", 7))

    return Program(inputs, tuple(nodes), (("sum", "sum_out"), ("carry", "carry_out"))).validate(r)


def composite_examples():
    rows = []
    for a in (False, True):
        for b in (False, True):
            for c in (False, True):
                total = int(a) + int(b) + int(c)
                rows.append(
                    {
                        "inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "cin": Value.of(BOOL, c)},
                        "targets": {"sum": Value.of(BOOL, bool(total % 2)), "carry": Value.of(BOOL, total >= 2)},
                    }
                )
    return rows


FA_SIGNALS = (
    Signal("sum_out", "sum", ("core",), BOOL, "bce"),
    Signal("carry_out", "carry", ("core",), BOOL, "bce"),
)


# --------------------------------------------------------------------------
# shared training loop (mirrors tcn.synthesis.fit's objective and schedule,
# but records step-resolved convergence and separates search from freezing)
# --------------------------------------------------------------------------
def conformant(exported, examples, signals, registry):
    for ex in examples:
        try:
            _, _, trace = exported.execute(ex["inputs"], registry=registry)
        except (ValueError, TypeError, KeyError, IndexError, OverflowError):
            return False
        for s in signals:
            if trace[s.source].flat() != ex["targets"][s.target].flat():
                return False
    return True


def search(program, examples, signals, registry, seed, steps=400, lr=0.05, eval_every=10, init_noise=0.5):
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    model = SoftProgram(program, registry)
    with torch.no_grad():
        for p in model.choices:
            p.add_(torch.randn_like(p) * init_noise)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}

    def loss_fn():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    first_conformant = None
    curve = []
    t0 = time.perf_counter()
    for step in range(steps):
        opt.zero_grad()
        loss = loss_fn() + 0.001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward()
            opt.step()
        if step % eval_every == 0 or step == steps - 1:
            ok = conformant(model.export(), examples, signals, registry)
            curve.append({"step": step, "loss": float(loss.detach()), "conformant": ok})
            if ok and first_conformant is None:
                first_conformant = step
                break
    wall = time.perf_counter() - t0
    final_loss = float(loss_fn().detach())
    exported = model.export()
    return model, opt, dict(
        first_conformant_step=first_conformant,
        steps_run=curve[-1]["step"] + 1,
        final_loss=final_loss,
        final_conformant=conformant(exported, examples, signals, registry),
        wall_seconds=wall,
        seconds_per_step=wall / max(1, curve[-1]["step"] + 1),
        curve=curve,
    ), exported


def crystallize(model, opt, examples, signals, registry, rounds=8, retrain_steps=5):
    inputs = {k: torch.stack([tensor(ex["inputs"][k]) for ex in examples]) for k, _ in model.program.inputs}
    targets = {s.target: torch.stack([tensor(ex["targets"][s.target]) for ex in examples]) for s in signals}

    def loss_fn():
        _, _, trace = model(inputs, return_trace=True)
        return model.probe_loss(trace, targets, signals)

    sched = Crystallizer(model, opt, tolerance=0.005, entropy_limit=0.9)
    t0 = time.perf_counter()
    sched.run(loss_fn, rounds=rounds, retrain_steps=retrain_steps,
              conformance=lambda ex: conformant(ex, examples, signals, registry))
    return dict(
        events=[asdict(e) for e in sched.events],
        accepted=sum(1 for e in sched.events if e.accepted),
        rejected_disconnected=sum(1 for e in sched.events if e.reason == "disconnected remaining region"),
        fully_frozen=len(model.frozen) == len(model.program.nodes),
        wall_seconds=time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------
def build_module(seed, steps=200):
    """Arm B stage 1: learn + crystallize the half adder, then register it."""
    r = Registry()
    prog = half_adder_scaffold(r)
    ex = half_adder_examples()
    model, opt, rep, exported = search(prog, ex, HA_SIGNALS, r, seed, steps=steps, eval_every=5)
    if not rep["final_conformant"]:
        return None, r, rep
    cry = crystallize(model, opt, ex, HA_SIGNALS, r, rounds=6, retrain_steps=5)
    module = model.export()
    if not conformant(module, ex, HA_SIGNALS, r):
        return None, r, rep
    rep["crystallization"] = cry
    rep["module_digest"] = module.digest
    rep["module_description_bits"] = module.description_bits()
    rep["module_execution_cost"] = module.execution_cost(r)
    return module, r, rep


def run_arm(arm, seed, steps, do_crystallize=True, module=None, module_registry=None):
    r = Registry()
    module_name = None
    if arm == "B":
        module_name = r.register_module(module)
    prog = composite_scaffold(r, module_name)
    ex = composite_examples()
    n_cands = sum(len(n.candidates) for n in prog.nodes)
    model, opt, rep, exported = search(prog, ex, FA_SIGNALS, r, seed, steps=steps)
    rep.update(
        arm=arm,
        seed=seed,
        scaffold_candidates=n_cands,
        scaffold_nodes=len(prog.nodes),
        final_description_bits=exported.description_bits(r),
        final_description_bits_no_registry=exported.description_bits(),
        final_execution_cost=exported.execution_cost(r),
    )
    if do_crystallize:
        rep["crystallization"] = crystallize(model, opt, ex, FA_SIGNALS, r, rounds=8, retrain_steps=5)
        final = model.export()
        rep["post_freeze_conformant"] = conformant(final, ex, FA_SIGNALS, r)
        rep["post_freeze_description_bits"] = final.description_bits(r)
        rep["post_freeze_execution_cost"] = final.execution_cost(r)
    rep["selections"] = {
        n.name: {
            "operator": n.candidates[i].operator.name,
            "sources": list(n.candidates[i].sources),
        }
        for n, i in ((n, model.selections()[n.name]) for n in prog.nodes)
    }
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--no-crystallize", action="store_true")
    ap.add_argument("--out", default="results.json")
    args = ap.parse_args()

    out = {"config": vars(args), "module_stage": [], "runs": []}

    # Arm B needs a module. Learn one per seed so module acquisition cost is
    # measured per seed and no single lucky module is shared across all runs.
    modules = {}
    for seed in range(args.seeds):
        module, mreg, mrep = build_module(1000 + seed)
        mrep["seed"] = seed
        out["module_stage"].append({k: v for k, v in mrep.items() if k != "curve"})
        modules[seed] = module
        print(f"[module] seed={seed} ok={module is not None} steps={mrep['steps_run']} "
              f"digest={mrep.get('module_digest')}", flush=True)

    for seed in range(args.seeds):
        for arm in ("A", "B"):
            if arm == "B" and modules[seed] is None:
                continue
            t = time.perf_counter()
            rep = run_arm(arm, seed, args.steps, not args.no_crystallize, modules[seed])
            rep["total_wall_seconds"] = time.perf_counter() - t
            out["runs"].append(rep)
            Path(args.out).write_text(json.dumps(out, indent=2, default=str))
            print(f"[{arm}] seed={seed} conf={rep['final_conformant']} "
                  f"first_step={rep['first_conformant_step']} loss={rep['final_loss']:.4f} "
                  f"bits={rep['final_description_bits']} cost={rep['final_execution_cost']} "
                  f"t={rep['total_wall_seconds']:.1f}s", flush=True)

    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    summarize(out)


def summarize(out):
    for arm in ("A", "B"):
        runs = [r for r in out["runs"] if r["arm"] == arm]
        if not runs:
            continue
        succ = [r for r in runs if r["final_conformant"]]
        steps = [r["first_conformant_step"] for r in succ if r["first_conformant_step"] is not None]
        print(f"\narm {arm}: n={len(runs)} success={len(succ)}/{len(runs)}")
        if steps:
            print(f"  steps to conformance: median={statistics.median(steps)} mean={statistics.mean(steps):.1f} "
                  f"min={min(steps)} max={max(steps)}")
        print(f"  final loss median={statistics.median([r['final_loss'] for r in runs]):.4f}")
        print(f"  description bits median={statistics.median([r['final_description_bits'] for r in runs])}")
        print(f"  execution cost median={statistics.median([r['final_execution_cost'] for r in runs])}")
        print(f"  candidates in scaffold={runs[0]['scaffold_candidates']}")
        print(f"  sec/step median={statistics.median([r['seconds_per_step'] for r in runs]):.4f}")


if __name__ == "__main__":
    main()
