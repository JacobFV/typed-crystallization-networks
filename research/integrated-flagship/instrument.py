"""Instrumentation of *why* (PREREGISTRATION §9), for each arm's found program.

    python instrument.py --gap 0 --arms "N''" "N'" N P

For each arm's example first solution (`out/arm_gap<g>_<arm>.json`):

* the frozen `tcn` program it names, built from a one-candidate-per-slot
  scaffold (identical semantics; its digest is reported), with the slot nodes
  that survive `Program.pruned()` -- which inherited abstractions sit on the
  execution path;
* live rewards through the integrated generator and the computer kernel on all
  48 episodes (`tcn.agent.Agent`), compared with the evaluator's hits (V3);
* interpreted versus compiled (`tcn.compile.compile_program`) latency on the
  training episodes -- a measurement only;

and `out/library.json`, the evidence-carrying library: per inherited class its
semantic identity, schema reference, implementations, observed domains and
widths, and reuse outcomes including this task's. No scalar score.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np                     # noqa: E402
import cache                           # noqa: E402
import engine                          # noqa: E402
import env                             # noqa: E402
import family                          # noqa: E402
import validate                        # noqa: E402
from family import SLOTS               # noqa: E402

OUT = HERE / "out"
N_TRAIN = 12


def arm_file(gap, arm):
    return OUT / f"arm_gap{gap}_{arm.replace(chr(39), 'p')}.json"


def singleton_pool(pool, sel):
    return {s: [pool[s][sel[s]]] for s in SLOTS}


def native_inputs(record):
    return {"text": (record["text_length"], tuple(record["text_bytes"])),
            "pixels": (record["height"], record["width"], 3, tuple(record["pixels"])),
            "action.0.pos": 128}


def instrument_arm(gap, arm):
    data = json.loads(arm_file(gap, arm).read_text())
    if not data.get("example_first_solution"):
        return {"arm": arm, "solution": None}
    pool_name = data["pool"]
    pool = family.pools(pool_name)
    sel = data["example_first_solution"]
    one = singleton_pool(pool, sel)
    zero = {s: 0 for s in SLOTS}
    prog, reg = family.build(one, head="agent")
    frozen = prog.harden(family.full_selection(prog, zero))
    pruned = frozen.pruned()
    live_nodes = {n.name for n in pruned.nodes}
    d = cache.load(gap)
    records = d["episodes"]
    eps = engine.Episodes(records)
    # The evaluator needs the arm's full pools (X1..X3 share one step pool); the
    # one-candidate pools above are used only to build the frozen tcn program.
    T = engine.Tables(eps, pool)
    _, hits = engine.evaluate_batch(T, np.array([[sel[s] for s in SLOTS]]))
    engine_hits = [bool(h) for h in hits[0]]
    t0 = time.perf_counter()
    live = []
    from tcn.agent import Agent
    for r in records:
        host = env.make_host(r["seed"], r["index"], r["split"],
                             {"gap": gap, "session": f"inst{gap}"})
        try:
            Agent(frozen, reg, validate.agent_config()).rollout(host, deterministic=True)
            live.append(float(host.records[-1].reward_components["goal"].decoded))
        except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError,
                IndexError):
            live.append(None)
    live_s = time.perf_counter() - t0
    agree = [(x or 0.) == float(h) for x, h in zip(live, engine_hits)]
    # latency: interpreted vs compiled, same frozen program, training episodes
    from tcn.compile import compile_program
    compiled = compile_program(frozen, reg)
    mod = compiled.module()
    interp, comp, same = [], [], []
    for r in records[:N_TRAIN]:
        inp = validate.episode_inputs(r)
        inp["action.0.pos"] = env.Value.of(env.CLICK, 128) if hasattr(env, "Value") else None
        from tcn.types import Value
        inp["action.0.pos"] = Value.of(env.CLICK, 128)
        t = time.perf_counter()
        out_i, _ = frozen.run(inp, registry=reg)
        interp.append(time.perf_counter() - t)
        nat = native_inputs(r)
        t = time.perf_counter()
        out_c, _ = mod.run(nat, validate=False)
        comp.append(time.perf_counter() - t)
        same.append(out_i["pos"].decoded == out_c["pos"])
    return {"arm": arm, "gap": gap, "pool": pool_name,
            "selection": sel, "specs": {s: str(pool[s][sel[s]]) for s in SLOTS},
            "frozen_digest": frozen.digest, "nodes": len(frozen.nodes),
            "pruned_nodes": len(pruned.nodes),
            "slots_on_execution_path": sorted(s for s in SLOTS if s in live_nodes),
            "live_rewards": live, "engine_hits": engine_hits,
            "live_train_accuracy": float(np.mean([x or 0. for x in live[:N_TRAIN]])),
            "live_held_accuracy": float(np.mean([x or 0. for x in live[N_TRAIN:]])),
            "live_agrees_with_engine": all(agree), "live_seconds": live_s,
            "interpreted_median_s": statistics.median(interp),
            "compiled_median_s": statistics.median(comp),
            "compiled_speedup": statistics.median(interp) / statistics.median(comp),
            "compiled_matches_interpreter": all(same),
            "compile_stats": {k: v for k, v in compiled.stats.items() if isinstance(v, (int, float))},
            "execution_cost_scalar": frozen.execution_cost(reg),
            "description_bits": frozen.description_bits(reg)}


def library(gaps):
    """The evidence-carrying library: classes, their evidence, this task's outcome."""
    src = json.loads((OUT / "sources.json").read_text())
    outcome = {}
    for gap in gaps:
        for arm in ("N'", "N''", "D'"):
            f = arm_file(gap, arm)
            if f.exists():
                d = json.loads(f.read_text())
                outcome[f"gap{gap}/{arm}"] = {"solution_exists": d["solution_exists"],
                                             "full_K": d["full_space"]["K"],
                                             "expected_programs_log10":
                                             (d["expected_programs"] or {}).get("log10")}
    classes = [
        {"class": "lexical-detector", "level": "parametric structural schema",
         "semantic": "eq(index(buffer, comb(x0, x1)), literal): the byte at a computed address equals a literal",
         "schema": "research/motif-unification/schema.py:m3_schema (§58/§60)",
         "implementations": ["§19 L_stage_a open (db59f9b09bb7)", "§33 V_same cmp (e38275a68420)"],
         "observed": [{"domain": "language", "width": 128}, {"domain": "visual", "width": 3072},
                      {"domain": "computer", "width": 4096, "outcome": "schema re-selected; vector failed (§60 T_C2)"}],
         "used_here_for": ["cpos", "ra"], "this_task": outcome},
        {"class": "colour-equality-3ch", "level": "parametric structural schema",
         "semantic": "three channel equalities at byte strides +1, +2, combined by two truth tables",
         "schema": "research/visual-ladder/rung3_widgets.py:same_scaffold (§33, §64)",
         "certified_vector": {"rg": 7, "same": 2},
         "observed": [{"domain": "visual", "width": w} for w in (16, 24, 32, 40, 48)],
         "used_here_for": ["M"], "this_task": outcome},
        {"class": "raster-step", "level": "parametric structural schema",
         "semantic": "a position moved by one of offset_pool(W) = (6, 3W+3, 3, 3W, 9) bytes",
         "schema": "research/visual-ladder/rung3_widgets.py:offset_pool (§33 S1'/S2', §64)",
         "certified_vector": {"s1": {"back_a": 2, "back_b": 3}, "s2": {"step_w": 2, "step_h": 3}},
         "observed": [{"domain": "visual", "width": w} for w in (16, 24, 32, 40, 48)],
         "used_here_for": ["X1", "X2", "X3"], "this_task": outcome},
        {"class": "specialization-prior", "level": "specialization policy / prior",
         "semantic": "P(candidate survives | op, operand kinds, V, occurs, table, offset class)",
         "fitted_on": {k: len(v) for k, v in src.items() if isinstance(v, list)},
         "sources": [c["source"] for c in src["certificates"]], "this_task": outcome},
    ]
    return {"format": "integrated-flagship/library-1", "classes": classes,
            "note": "Evidence, not a score: no scalar abstraction score is computed (§57)."}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", type=int, default=0)
    ap.add_argument("--arms", nargs="*", default=["N''", "N'", "N", "P"])
    ap.add_argument("--library", action="store_true")
    a = ap.parse_args()
    if a.library:
        (OUT / "library.json").write_text(json.dumps(library((0, 1)), indent=1))
        print("wrote library.json")
    else:
        rows = []
        for arm in a.arms:
            if not arm_file(a.gap, arm).exists():
                continue
            r = instrument_arm(a.gap, arm)
            rows.append(r)
            print(json.dumps({k: v for k, v in r.items() if k not in ("live_rewards", "engine_hits",
                                                                      "compile_stats")},
                             default=str)[:1500], flush=True)
        (OUT / f"instrument_gap{a.gap}.json").write_text(json.dumps(rows, indent=1, default=str))
