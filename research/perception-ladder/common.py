"""Shared helpers for the perception ladder.

Rules honoured here, from AGENTS.md:
  * only `record.observations` ever reaches a program input;
  * `record.latent_states` and `record.probes` are used exclusively to build
    `targets`, i.e. as supervision;
  * nothing under `tcn/` or `generators/` is modified.
"""
from __future__ import annotations
import json, math, statistics, time
from dataclasses import dataclass

import torch

from tcn.generation import Host, SCALAR, Value
from tcn.graph import Program, Node, Candidate, Signal
from tcn.operators import Registry
from tcn.search import enumerate_fit, space_size
from tcn.synthesis import fit

F = SCALAR


def episode_observation(generator, seed, ticks, configuration=None, split="train", index=0):
    """Step one episode `ticks` times and return (observations, latents, probes)."""
    h = Host.create(generator, seed=seed, index=index, split=split, configuration=configuration)
    for _ in range(ticks):
        h.step()
    rec = h.records[-1]
    view = rec.actor_view()
    return view.observations, rec.latent_states, rec.probes


def cand(r, names, types, sources, output=None, parameters=None):
    return tuple(Candidate(r.resolve(n, types, output, parameters), sources) for n in names)


def run_seeds(build_problem, seeds, steps=400, lr=.05, polish=200, tolerance=1e-3,
              freeze=False, label="", verbose=True):
    """Run `tcn.synthesis.fit` once per seed; report exact conformance + held-out error."""
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        program, signals, train, test, registry = build_problem(seed)
        t0 = time.perf_counter()
        try:
            model, report = fit(program, train, signals, steps=steps, lr=lr,
                                freeze=freeze, registry=registry, tolerance=tolerance,
                                polish=polish)
            exact = model.export()
            held = held_out_error(exact, test, signals, registry)
            rows.append({"seed": seed, "ok": bool(report["exact_conformance"]),
                         "train_err": report["exact_max_error"], "test_err": held,
                         "relaxed_loss": report["relaxed_loss"],
                         "seconds": time.perf_counter() - t0,
                         "selections": model.selections()})
        except Exception as exc:  # a candidate mixture can leave a declared domain
            rows.append({"seed": seed, "ok": False, "train_err": float("inf"),
                         "test_err": float("inf"), "relaxed_loss": float("nan"),
                         "seconds": time.perf_counter() - t0, "error": repr(exc)[:200]})
        if verbose:
            print(f"  [{label}] seed {seed}: ok={rows[-1]['ok']} "
                  f"train={rows[-1]['train_err']:.3g} test={rows[-1]['test_err']:.3g} "
                  f"({rows[-1]['seconds']:.1f}s)", flush=True)
    return rows


def held_out_error(exact, examples, signals, registry):
    worst = 0.
    for ex in examples:
        try:
            _, _, trace = exact.execute(ex["inputs"], registry=registry)
        except Exception:
            return float("inf")
        for s in signals:
            a = trace[s.source].flat(); b = ex["targets"][s.target].flat()
            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
    return worst


def summarise(rows):
    ok = [r for r in rows if r["ok"]]
    gen = [r for r in rows if r["test_err"] <= 1e-3]
    return {"n": len(rows), "train_success": len(ok), "rate": len(ok) / max(1, len(rows)),
            "generalises": len(gen),
            "median_test_err": statistics.median([r["test_err"] for r in rows]) if rows else float("nan"),
            "median_seconds": statistics.median([r["seconds"] for r in rows]) if rows else float("nan")}


def enumerate_reference(program, examples, signals, registry, tolerance=1e-3, max_programs=1 << 22):
    n = space_size(program)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, signals, registry=registry,
                        tolerance=tolerance, max_programs=max_programs)
    d = res.to_dict(); d["space_size"] = n; d["wall"] = time.perf_counter() - t0
    return d


def random_reference(program, examples, signals, registry, draws, seed=0, tolerance=1e-3):
    """Uniform random discrete programs from the same space -- the track-8 control."""
    import random
    from tcn.search import evaluate
    rng = random.Random(seed)
    names = [n.name for n in program.nodes]
    counts = [len(n.candidates) for n in program.nodes]
    t0 = time.perf_counter(); hits = 0
    for _ in range(draws):
        sel = {k: rng.randrange(c) for k, c in zip(names, counts)}
        err = evaluate(program, sel, examples, signals, registry, tolerance)
        if err is not None and err <= tolerance:
            hits += 1
    return {"draws": draws, "hits": hits, "density": hits / draws, "seconds": time.perf_counter() - t0}


def dump(name, obj):
    from pathlib import Path
    p = Path(__file__).resolve().parent / "out" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1, default=str))
    print("wrote", p)


# --------------------------------------------------------------------------
# `tcn.synthesis.fit` initialises every choice logit to zero, so repeated runs
# are bit-identical and a "seed" changes nothing.  Track 1 hit the same wall and
# added initialisation noise.  `local_fit` is a faithful copy of `fit`'s control
# flow with two additions: `init_noise` (per-seed logit jitter) and a returned
# training history that includes the argmax trajectory.  `check_local_fit.py`
# verifies it is bit-identical to `tcn.synthesis.fit` at init_noise=0.
# --------------------------------------------------------------------------
from dataclasses import asdict as _asdict
from tcn.learning import SoftProgram, tensor as _tensor
from tcn.crystallize import Crystallizer


def local_fit(program, examples, signals, steps=300, lr=.05, freeze=False, registry=None,
              tolerance=.001, polish=200, init_noise=0., seed=0, trace_every=0,
              anneal=None):
    if not examples: raise ValueError('training examples required')
    program.validate_signals(signals)
    model = SoftProgram(program, registry)
    if init_noise:
        g = torch.Generator().manual_seed(int(seed) + 12345)
        with torch.no_grad():
            for p in model.choices:
                p.add_(torch.randn(p.shape, generator=g) * init_noise)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    inputs = {k: torch.stack([_tensor(ex['inputs'][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex['targets'][s.target]) for ex in examples]) for s in signals}

    def loss_fn():
        _, _, tr = model(inputs, return_trace=True)
        return model.probe_loss(tr, targets, signals)

    history = []
    torch.set_num_threads(1)
    for step in range(steps):
        if anneal is not None:
            tau = anneal[0] * (anneal[1] / anneal[0]) ** (step / max(1, steps - 1))
            for n in program.nodes: model.temperatures[n.name] = tau
        optimizer.zero_grad()
        loss = loss_fn() + .001 * (step / max(1, steps)) * model.entropy()
        if loss.requires_grad:
            loss.backward(); optimizer.step()
        if trace_every and step % trace_every == 0:
            history.append({'step': step, 'loss': float(loss.detach()),
                            'entropy': float(model.entropy().detach()),
                            'sel': dict(model.selections())})
    if anneal is not None:
        for n in program.nodes: model.temperatures[n.name] = 1.
    if polish and model.constants:
        held = dict(model.selections()); requires = [p.requires_grad for p in model.choices]
        model.trials = dict(held)
        for p in model.choices: p.requires_grad_(False)
        refiner = torch.optim.Adam([p for p in model.constants.values()], lr=lr)
        for _ in range(polish):
            refiner.zero_grad(); refined = loss_fn()
            if refined.requires_grad: refined.backward(); refiner.step()
        model.trials = {}
        for p, flag in zip(model.choices, requires): p.requires_grad_(flag)

    def exact_error(exact):
        worst = 0.
        for ex in examples:
            _, _, tr = exact.execute(ex['inputs'], registry=model.registry)
            for s in signals:
                a = torch.tensor(tr[s.source].flat()); b = torch.tensor(ex['targets'][s.target].flat())
                worst = max(worst, float((a - b).abs().max()))
        return worst

    if freeze:
        scheduler = Crystallizer(model, optimizer, tolerance=tolerance, entropy_limit=.9)
        scheduler.run(loss_fn, rounds=24, retrain_steps=10,
                      conformance=lambda e: exact_error(e) <= tolerance)
    err = exact_error(model.export())
    return model, {'training': history, 'relaxed_loss': float(loss_fn().detach()),
                   'exact_max_error': err, 'exact_conformance': err <= tolerance,
                   'tolerance': tolerance}


def run_seeds_local(build_problem, seeds, steps=400, lr=.05, polish=200, tolerance=1e-3,
                    freeze=False, label="", init_noise=.5, verbose=True, anneal=None):
    rows = []
    for seed in seeds:
        torch.manual_seed(seed)
        program, signals, train, test, registry = build_problem(seed)
        t0 = time.perf_counter()
        try:
            model, report = local_fit(program, train, signals, steps=steps, lr=lr,
                                      freeze=freeze, registry=registry, tolerance=tolerance,
                                      polish=polish, init_noise=init_noise, seed=seed,
                                      anneal=anneal)
            exact = model.export()
            held = held_out_error(exact, test, signals, registry)
            rows.append({"seed": seed, "ok": bool(report["exact_conformance"]),
                         "train_err": report["exact_max_error"], "test_err": held,
                         "relaxed_loss": report["relaxed_loss"],
                         "seconds": time.perf_counter() - t0,
                         "selections": model.selections()})
        except Exception as exc:
            rows.append({"seed": seed, "ok": False, "train_err": float("inf"),
                         "test_err": float("inf"), "relaxed_loss": float("nan"),
                         "seconds": time.perf_counter() - t0, "error": repr(exc)[:160]})
        if verbose:
            print(f"  [{label}] seed {seed}: ok={rows[-1]['ok']} "
                  f"train={rows[-1]['train_err']:.3g} test={rows[-1]['test_err']:.3g} "
                  f"({rows[-1]['seconds']:.1f}s) {rows[-1].get('error','')}", flush=True)
    return rows


def accuracy(program, selections, examples, signals, registry, tolerance=1e-3):
    """Mean per-element agreement of one discrete selection over `examples`."""
    hit = total = 0
    for ex in examples:
        try:
            _, _, tr = program.execute(ex["inputs"], registry=registry, selections=selections)
        except Exception:
            return 0.
        for s in signals:
            a = tr[s.source].flat(); b = ex["targets"][s.target].flat()
            for x, y in zip(a, b):
                total += 1; hit += abs(x - y) <= tolerance
    return hit / max(1, total)
