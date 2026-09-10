"""Step 2: do selections alone transfer across observation widths?

    .venv/bin/python research/depth-encoding/run.py baselines
    .venv/bin/python research/depth-encoding/run.py fit
    .venv/bin/python research/depth-encoding/run.py exhaust
    .venv/bin/python research/depth-encoding/run.py transfer
    .venv/bin/python research/depth-encoding/run.py record
    .venv/bin/python research/depth-encoding/run.py hardened
    .venv/bin/python research/depth-encoding/run.py crosscheck

Every sweep is scored with `tcn.search.program_return` -- the same scoring
function `tcn.search.enumerate_environment` uses -- and certified with
`tcn.search.certificate_of`. `crosscheck` re-runs one sweep through
`enumerate_environment` itself and asserts the agreement, so nothing here is a
private reimplementation of the shipped enumerator's verdict.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import scaffold as S
from tcn.search import (EnvironmentTask, EpisodeLedger, candidate_counts,
                        certificate_of, enumerate_environment, program_return,
                        trailing_return)
from tcn.types import BOOL, Value
from tcn.generation import Action

OUT = HERE / "out"
BASE = {"inputs": 4, "nondegenerate": True, "min_relevant_inputs": 2}
HORIZON = 4
DEPTHS = (1, 2, 3, 4, 6, 8)
FIT_INDICES = tuple(range(16))
EVAL_INDICES = tuple(range(10000, 10064))
THRESHOLD = 4.0


# ---------------------------------------------------------------------------
def summarize(xs):
    n = len(xs)
    m = sum(xs) / n if n else 0.
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / max(1, n - 1)) if n > 1 else 0.
    return {"mean": m, "sd": sd, "n": n,
            "min": min(xs) if xs else None, "max": max(xs) if xs else None}


def build(kind, depth):
    host = S.make_host(depth, 0, "test", BASE, HORIZON)
    return S.SCAFFOLDS[kind](host, depth)


def task_for(names, depth, indices, split):
    return EnvironmentTask(
        generator="logic", observations=tuple(names), action_templates=S.ACTIONS,
        generator_config=dict(BASE) | {"depth": depth}, objectives=S.OBJECTIVES,
        indices=tuple(indices), horizon=HORIZON, dt=1., seed=0, split=split,
        scored_ticks=None, deterministic=True, agent_seed=0)


def free_nodes(program):
    return [(n.name, len(n.candidates)) for n in program.nodes if len(n.candidates) > 1]


def sweep(program, registry, task):
    """Every discrete program scored by real environment return, with the set kept.

    `enumerate_environment` reports the size of the conforming set but not its
    members, and this track needs the members: a transfer that only works for
    the first member enumeration happens to reach is under-determined (FINDINGS
    section 47). Scoring and exhaustion are `tcn.search`'s own.
    """
    counts = candidate_counts(program)
    names = [n.name for n in program.nodes]
    total = math.prod(counts)
    ledger = EpisodeLedger()
    started = time.perf_counter()
    scores = []
    evaluated = 0
    for combination in itertools.product(*(range(c) for c in counts)):
        selections = dict(zip(names, combination))
        evaluated += 1
        score = program_return(program, registry, task, selections, ledger)
        if score is not None:
            scores.append((selections, score))
    best = max((v for _, v in scores), default=None)
    found = [s for s, v in scores if v >= THRESHOLD]
    exhausted = evaluated >= total
    free = [k for k, _ in free_nodes(program)]
    return {
        "space_size": total, "evaluated": evaluated, "exhausted": exhausted,
        "unusable": evaluated - len(scores),
        "conforming": len(found), "threshold": THRESHOLD,
        "certificate": certificate_of(exhausted, True, len(found)),
        "best_return": best,
        "conforming_selections": [{k: s[k] for k in free} for s in found],
        "all_returns": sorted(v for _, v in scores),
        "distribution": summarize([v for _, v in scores]),
        "seconds": time.perf_counter() - started,
        "episodes": ledger.episodes, "steps": ledger.steps,
        "free_nodes": free_nodes(program),
    }


def score_selection(kind, depth, selection, indices, split):
    """Rebuild the schema at `depth`, apply the vector, score the episodes."""
    names, program, registry = build(kind, depth)
    free = dict(free_nodes(program))
    if set(free) != set(selection):
        raise ValueError(f"selection names {sorted(selection)} != free nodes {sorted(free)}")
    for k, v in selection.items():
        if not 0 <= v < free[k]:
            raise ValueError(f"selection {k}={v} outside {free[k]} candidates at depth {depth}")
    full = {n.name: 0 for n in program.nodes}
    full.update(selection)
    task = task_for(names, depth, indices, split)
    ledger = EpisodeLedger()
    per = []
    from tcn.search import frozen_selection
    from tcn.agent import Agent
    exact = frozen_selection(program, full, registry)
    agent = Agent(exact, registry, S.agent_config(names, HORIZON), seed=0)
    for i in indices:
        host = S.make_host(depth, i, split, BASE, HORIZON)
        agent.rollout(host, deterministic=True)
        per.append(trailing_return(host))
    return per, exact


# ---------------------------------------------------------------------------
def constant_returns(depth, indices, split, answer):
    out = []
    for i in indices:
        host = S.make_host(depth, i, split, BASE, HORIZON)
        for _ in range(HORIZON):
            if host.records[-1].done:
                break
            value = answer() if callable(answer) else answer
            host.step((Action("answer", arguments=(("value", Value.of(BOOL, value)),)),), 1.)
        out.append(trailing_return(host))
    return out


def episode_baselines(depth, indices, split, rng_seed=0):
    rng = random.Random(rng_seed)
    targets = []
    for i in indices:
        host = S.make_host(depth, i, split, BASE, HORIZON)
        targets.append(bool(host.records[0].probes["target"].decoded))
    out = {
        "always_true": summarize(constant_returns(depth, indices, split, True)),
        "always_false": summarize(constant_returns(depth, indices, split, False)),
        "uniform_random": summarize(constant_returns(depth, indices, split,
                                                     lambda: rng.random() < .5)),
        "fraction_target_true": sum(targets) / len(targets),
    }
    out["best_constant"] = max(out["always_true"]["mean"], out["always_false"]["mean"])
    return out


# ---------------------------------------------------------------------------
def condition_baselines():
    report = {"held_out": {}, "fit": {}}
    for d in DEPTHS:
        report["held_out"][str(d)] = episode_baselines(d, EVAL_INDICES, "test")
        report["fit"][str(d)] = episode_baselines(d, FIT_INDICES, "train")
        print(d, json.dumps(report["held_out"][str(d)]), flush=True)
    (OUT / "baselines.json").write_text(json.dumps(report, indent=2))


def condition_fit():
    report = {}
    for fit_depth in (1, 2):
        names, program, registry = build("interpreter", fit_depth)
        task = task_for(names, fit_depth, FIT_INDICES, "train")
        result = sweep(program, registry, task)
        result["fit_depth"] = fit_depth
        result["nodes"] = len(program.nodes)
        result["program_fields"] = len(dict(program.inputs)["program"].items)
        report[str(fit_depth)] = result
        print(f"fit d={fit_depth}", {k: result[k] for k in
                                     ("space_size", "evaluated", "exhausted", "conforming",
                                      "certificate", "best_return", "seconds", "episodes")},
              result["conforming_selections"], flush=True)
    (OUT / "fit.json").write_text(json.dumps(report, indent=2))


def condition_exhaust():
    """Arm E + arm C: the whole 272-program space scored on held-out, per depth."""
    report = {}
    for d in DEPTHS:
        names, program, registry = build("interpreter", d)
        task = task_for(names, d, EVAL_INDICES, "test")
        result = sweep(program, registry, task)
        result["depth"] = d
        result["nodes"] = len(program.nodes)
        report[str(d)] = result
        print(f"exhaust d={d}", {k: result[k] for k in
                                 ("evaluated", "exhausted", "conforming", "certificate",
                                  "best_return", "seconds")},
              "dist", result["distribution"], flush=True)
        (OUT / "exhaust.json").write_text(json.dumps(report, indent=2))


def condition_transfer():
    fit = json.loads((OUT / "fit.json").read_text())
    report = {}
    for fit_depth, result in fit.items():
        rows = []
        for selection in result["conforming_selections"]:
            row = {"selection": selection, "per_depth": {}}
            for d in DEPTHS:
                per, exact = score_selection("interpreter", d, selection, EVAL_INDICES, "test")
                row["per_depth"][str(d)] = summarize(per)
                row["per_depth"][str(d)]["frozen_digest"] = exact.digest
                row["per_depth"][str(d)]["nodes"] = len(exact.nodes)
                print(f"fit d={fit_depth} sel={selection} -> d={d} "
                      f"{row['per_depth'][str(d)]['mean']:.4f}", flush=True)
            rows.append(row)
        report[fit_depth] = rows
    (OUT / "transfer.json").write_text(json.dumps(report, indent=2))


def condition_record():
    """Arm B: the width-free control scaffold, fit at depth 1, applied everywhere."""
    names, program, registry = build("record", 1)
    task = task_for(names, 1, FIT_INDICES, "train")
    fit = sweep(program, registry, task)
    report = {"fit": fit, "transfer": []}
    print("record fit", {k: fit[k] for k in ("space_size", "evaluated", "exhausted",
                                             "conforming", "certificate", "best_return")},
          flush=True)
    # Nothing may conform at threshold 4.0; carry the best-scoring selections.
    best = fit["best_return"]
    counts = candidate_counts(program)
    node_names = [n.name for n in program.nodes]
    free = [k for k, _ in free_nodes(program)]
    carried = fit["conforming_selections"]
    if not carried:
        # re-derive argmax selections from the same sweep semantics
        ledger = EpisodeLedger()
        carried = []
        for combination in itertools.product(*(range(c) for c in counts)):
            selections = dict(zip(node_names, combination))
            score = program_return(program, registry, task, selections, ledger)
            if score is not None and score >= best:
                carried.append({k: selections[k] for k in free})
        report["carried_are_argmax"] = True
        report["argmax_episodes"] = ledger.episodes
    report["carried"] = carried
    for selection in carried[:4]:
        row = {"selection": selection, "per_depth": {}}
        for d in DEPTHS:
            per, exact = score_selection("record", d, selection, EVAL_INDICES, "test")
            row["per_depth"][str(d)] = summarize(per)
            row["per_depth"][str(d)]["frozen_digest"] = exact.digest
            print(f"record sel={selection} -> d={d} "
                  f"{row['per_depth'][str(d)]['mean']:.4f}", flush=True)
        report["transfer"].append(row)
    (OUT / "record.json").write_text(json.dumps(report, indent=2))


def condition_hardened():
    """Arm D: the hardened artifact itself, offered an episode of another depth."""
    import traceback
    from tcn.search import frozen_selection
    fit = json.loads((OUT / "fit.json").read_text())
    selection = fit["1"]["conforming_selections"][0]
    names, program, registry = build("interpreter", 1)
    full = {n.name: 0 for n in program.nodes}
    full.update(selection)
    exact = frozen_selection(program, full, registry)
    report = {"selection": selection, "frozen_digest": exact.digest,
              "hardened_at_depth": 1, "offered": {}}
    for d in DEPTHS:
        host = S.make_host(d, 10000, "test", BASE, HORIZON)
        view = host.view().observations
        inputs = {k: view[k] for k in names}
        inputs["action"] = Value.of(dict(exact.inputs)["action"], (1., 0.))
        inputs["dt"] = Value.of(dict(exact.inputs)["dt"], 1.)
        try:
            exact.execute(inputs, registry=registry)
            report["offered"][str(d)] = {"raised": False}
        except BaseException as exc:  # noqa: BLE001
            frame = traceback.extract_tb(exc.__traceback__)[-1]
            rel = Path(frame.filename)
            try:
                rel = rel.relative_to(HERE.parents[1])
            except ValueError:
                pass
            report["offered"][str(d)] = {"raised": True, "type": type(exc).__name__,
                                         "message": str(exc),
                                         "at": f"{rel}:{frame.lineno}"}
        print(d, report["offered"][str(d)], flush=True)
    (OUT / "hardened.json").write_text(json.dumps(report, indent=2))


def condition_crosscheck():
    """The shipped enumerator, on the same space, must agree with `sweep`."""
    names, program, registry = build("interpreter", 1)
    task = task_for(names, 1, FIT_INDICES, "train")
    started = time.perf_counter()
    result = enumerate_environment(program, task, registry, threshold=THRESHOLD)
    mine = json.loads((OUT / "fit.json").read_text())["1"]
    free = [k for k, _ in free_nodes(program)]
    report = {
        "enumerate_environment": result.to_dict(),
        "seconds": time.perf_counter() - started,
        "agrees": {
            "evaluated": result.evaluated == mine["evaluated"],
            "space_size": result.space_size == mine["space_size"],
            "exhausted": result.exhausted == mine["exhausted"],
            "conforming": result.conforming == mine["conforming"],
            "certificate": result.certificate == mine["certificate"],
            "best_return": result.best_return == mine["best_return"],
            "chosen": ({k: result.selections[k] for k in free}
                       in mine["conforming_selections"]),
        },
    }
    report["all_agree"] = all(report["agrees"].values())
    print(json.dumps(report, indent=2), flush=True)
    (OUT / "crosscheck.json").write_text(json.dumps(report, indent=2))


def condition_wrong_schema():
    """Amendment 1, arm F: the same 272 space, one edge moved.

    `relation`'s lookup candidate reads `gate_0` instead of `gate_{d-1}`. At
    depth 1 those are the same node, so the depth-1 fit is bound to return the
    same vector with the same certificate; the transfer is where the two schemas
    separate. This is the control that shows a passing transfer is a property of
    the schema and not of the mechanism that carries the vector.
    """
    names, program, registry = build("first_gate", 1)
    task = task_for(names, 1, FIT_INDICES, "train")
    fit = sweep(program, registry, task)
    report = {"fit": fit, "transfer": []}
    print("first_gate fit", {k: fit[k] for k in ("space_size", "evaluated", "exhausted",
                                                 "conforming", "certificate", "best_return")},
          fit["conforming_selections"], flush=True)
    for selection in fit["conforming_selections"]:
        row = {"selection": selection, "per_depth": {}}
        for d in DEPTHS:
            per, exact = score_selection("first_gate", d, selection, EVAL_INDICES, "test")
            row["per_depth"][str(d)] = summarize(per)
            row["per_depth"][str(d)]["frozen_digest"] = exact.digest
            print(f"first_gate sel={selection} -> d={d} "
                  f"{row['per_depth'][str(d)]['mean']:.4f}", flush=True)
        report["transfer"].append(row)
    # And the whole space at each depth, so a collapse is not confused with
    # "nothing in this family works at that depth".
    report["exhaust"] = {}
    for d in DEPTHS:
        n, p, r = build("first_gate", d)
        report["exhaust"][str(d)] = {
            k: v for k, v in sweep(p, r, task_for(n, d, EVAL_INDICES, "test")).items()
            if k != "all_returns"}
        print("first_gate exhaust", d, report["exhaust"][str(d)]["conforming"],
              report["exhaust"][str(d)]["best_return"], flush=True)
    (OUT / "wrong_schema.json").write_text(json.dumps(report, indent=2))


def condition_difficulty():
    """Amendment 1, arm G: achieved difficulty, over the held-out episodes."""
    sys.path.insert(0, str(HERE.parents[1] / "generators" / "logic"))
    from generators.logic.generator import relevant_inputs
    report = {}
    for d in DEPTHS:
        circuits, finals, relevant, targets = [], [], [], []
        for i in EVAL_INDICES:
            host = S.make_host(d, i, "test", BASE, HORIZON)
            gates = host.records[0].state["gates"] if hasattr(host.records[0], "state") else None
            view = host.view().observations
            fields = [round(x) for x in view["program"].decoded]
            gates = [tuple(fields[3 * g:3 * g + 3]) for g in range(d)]
            circuits.append(tuple(gates))
            finals.append(gates[-1][2])
            relevant.append(len(relevant_inputs(BASE["inputs"], [list(g) for g in gates])))
            targets.append(bool(host.records[0].probes["target"].decoded))
        majority = max(sum(targets), len(targets) - sum(targets)) / len(targets)
        report[str(d)] = {
            "episodes": len(EVAL_INDICES),
            "distinct_circuits": len(set(circuits)),
            "distinct_final_tables": sorted(set(finals)),
            "relevant_inputs_histogram": {str(k): relevant.count(k)
                                          for k in sorted(set(relevant))},
            "min_relevant_inputs_achieved": min(relevant),
            "fraction_target_true": sum(targets) / len(targets),
            "majority_fraction": majority,
        }
        print(d, json.dumps(report[str(d)]), flush=True)
    (OUT / "difficulty.json").write_text(json.dumps(report, indent=2))


def condition_size():
    """Amendment 1, arm H: what part of the artifact is width-invariant."""
    fit = json.loads((OUT / "fit.json").read_text())
    selection = fit["1"]["conforming_selections"][0]
    report = {"selection": selection, "per_depth": {}}
    free = None
    for d in DEPTHS:
        names, program, registry = build("interpreter", d)
        free = dict(free_nodes(program))
        full = {n.name: 0 for n in program.nodes}
        full.update(selection)
        from tcn.search import frozen_selection
        exact = frozen_selection(program, full, registry)
        pruned = exact.pruned()
        report["per_depth"][str(d)] = {
            "program_fields": len(dict(program.inputs)["program"].items),
            "program_flat_width": dict(program.inputs)["program"].width,
            "scaffold_nodes": len(program.nodes),
            "frozen_nodes": len(exact.nodes),
            "pruned_nodes": len(pruned.nodes),
            "description_bits": exact.description_bits(registry),
            "pruned_description_bits": pruned.description_bits(registry),
            "execution_cost": exact.execution_cost(registry),
            "digest": exact.digest,
        }
        print(d, report["per_depth"][str(d)], flush=True)
    report["selection_vector_bits"] = sum(math.log2(c) for c in free.values())
    report["selection_vector_entries"] = len(free)
    report["distinct_digests"] = len({v["digest"] for v in report["per_depth"].values()})
    print("selection vector:", report["selection_vector_bits"], "bits;",
          report["distinct_digests"], "distinct frozen artifacts", flush=True)
    (OUT / "size.json").write_text(json.dumps(report, indent=2))


CONDITIONS = {"baselines": condition_baselines, "fit": condition_fit,
              "exhaust": condition_exhaust, "transfer": condition_transfer,
              "record": condition_record, "hardened": condition_hardened,
              "crosscheck": condition_crosscheck,
              "wrong_schema": condition_wrong_schema,
              "difficulty": condition_difficulty, "size": condition_size}

if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for name in sys.argv[1:]:
        CONDITIONS[name]()
