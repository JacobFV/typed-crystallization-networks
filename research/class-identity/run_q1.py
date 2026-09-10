"""Q1 arms: does a stored class let an unseen width reuse a learned result?

    .venv/bin/python research/class-identity/run_q1.py construct
    .venv/bin/python research/class-identity/run_q1.py without
    .venv/bin/python research/class-identity/run_q1.py with
    .venv/bin/python research/class-identity/run_q1.py baselines
    .venv/bin/python research/class-identity/run_q1.py difficulty
    .venv/bin/python research/class-identity/run_q1.py crosscheck
    .venv/bin/python research/class-identity/run_q1.py armf

Everything measured here goes through §53's own code: the schema is
`research/depth-encoding/scaffold.py` unchanged, and the sweep, scoring,
certificate and baselines are `research/depth-encoding/run.py`'s, which §53
already cross-checked against `tcn.search.enumerate_environment`. Nothing in this
file re-implements a verdict.

The with-class and without-class arms call the **same** `tcn.search.program_return`
against the **same** `EnvironmentTask`, so the cost comparison is not an artefact
of two different harnesses: one arm calls it 272 times, the other once.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DE = ROOT / "research" / "depth-encoding"
sys.path.insert(0, str(DE))
sys.path.insert(0, str(ROOT))

import scaffold as S                                              # noqa: E402
import run as DEP                                                 # noqa: E402  §53's own arms
import classes as C                                               # noqa: E402

from tcn.generation import source_fingerprint                     # noqa: E402
from tcn.search import (EpisodeLedger, certificate_of,            # noqa: E402
                        enumerate_environment, frozen_selection, program_return)

OUT = HERE / "out"
STORE = OUT / "library"

# Declared in PREREGISTRATION.md §3 before any arm: never touched by any prior run.
HELD_OUT_WIDTHS = (5, 7, 12)
CONSTRUCT_WIDTHS = (1, 2)
CONFIRM_WIDTHS = (3, 4, 6, 8)                                     # §53's, secondary
SCHEMA_QUALIFIED = "research.depth-encoding.scaffold:interpreter_scaffold"
WRONG_QUALIFIED = "research.depth-encoding.scaffold:first_gate_scaffold"


def builder(kind):
    return lambda width: DEP.build(kind, width)


def freeze(program, full, registry):
    return frozen_selection(program, full, registry)


def _class_id(kind, selections):
    return C.schema_class_id(source_fingerprint(),
                             SCHEMA_QUALIFIED if kind == "interpreter" else WRONG_QUALIFIED,
                             selections)


def _free(program):
    return [(n.name, len(n.candidates)) for n in program.nodes if len(n.candidates) > 1]


def _certify(kind, width, indices, split):
    """Enumerate the schema's whole space at one width. This is the expensive path."""
    names, program, registry = DEP.build(kind, width)
    task = DEP.task_for(names, width, indices, split)
    result = DEP.sweep(program, registry, task)
    result["width"] = width
    result["free_nodes"] = _free(program)
    return names, program, registry, task, result


def _record_for(kind, selections, certifications, members, preferred, provenance):
    names, program, _ = DEP.build(kind, CONSTRUCT_WIDTHS[0])
    return C.ClassRecord(
        class_id=_class_id(kind, selections), kind="schema",
        schema={"module": "research/depth-encoding/scaffold.py",
                "qualified_name": SCHEMA_QUALIFIED if kind == "interpreter" else WRONG_QUALIFIED,
                "free_nodes": _free(program),
                "source_fingerprint": source_fingerprint()},
        selections=dict(selections), certified=tuple(certifications),
        members=tuple(members), preferred=dict(preferred), provenance=dict(provenance))


# ---------------------------------------------------------------- arm A1
def arm_construct(kind="interpreter", widths=CONSTRUCT_WIDTHS, tag="construct"):
    """Certify at the construction widths only; publish under R1-R3."""
    report = {"kind": kind, "construct_widths": list(widths), "per_width": {}}
    certs, vectors = [], []
    for w in widths:
        _, program, registry, _, result = _certify(kind, w, DEP.FIT_INDICES, "train")
        sel = result["conforming_selections"][0] if result["conforming_selections"] else None
        digest = None
        if sel is not None:
            full = {n.name: 0 for n in program.nodes}
            full.update(sel)
            digest = frozen_selection(program, full, registry).digest
        certs.append(C.Certification(
            width=w, space_size=result["space_size"], evaluated=result["evaluated"],
            exhausted=result["exhausted"], conforming=result["conforming"],
            certificate=result["certificate"],
            mean_return=result["best_return"] if result["best_return"] is not None else 0.,
            threshold=DEP.THRESHOLD, artifact_digest=digest or ""))
        vectors.append(sel)
        report["per_width"][str(w)] = {
            k: result[k] for k in ("space_size", "evaluated", "exhausted", "conforming",
                                   "certificate", "best_return", "seconds", "episodes",
                                   "steps", "free_nodes")}
        report["per_width"][str(w)]["conforming_selections"] = result["conforming_selections"]
        report["per_width"][str(w)]["artifact_digest"] = digest
        print(tag, kind, "certify width", w, report["per_width"][str(w)]["certificate"],
              report["per_width"][str(w)]["conforming"], sel, flush=True)

    distinct = {json.dumps(v, sort_keys=True) for v in vectors if v is not None}
    report["vectors_agree"] = len(distinct) == 1 and len(vectors) == len(widths)
    selections = vectors[0] if vectors and vectors[0] is not None else {}
    record = _record_for(kind, selections, certs,
                         members=[c.artifact_digest for c in certs if c.artifact_digest],
                         preferred={"description_bits": certs[0].artifact_digest,
                                    "execution_cost": certs[0].artifact_digest},
                         provenance={"track": "class-identity", "from": "FINDINGS 53",
                                     "construct_widths": list(widths)})
    report["class_id"] = record.class_id
    store = C.ClassStore(STORE)
    try:
        store.admit(record, threshold=DEP.THRESHOLD)
        report["admitted"] = True
        report["refusal"] = None
    except C.ClassError as exc:
        report["admitted"] = False
        report["refusal"] = str(exc)
    print(tag, "admitted", report["admitted"], report["refusal"] or "", flush=True)
    (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- arm A2
def arm_without(widths=HELD_OUT_WIDTHS, kind="interpreter", tag="without"):
    """Enumerate all 272 programs at each held-out width, on held-out episodes."""
    report = {}
    for w in widths:
        _, _, _, _, result = _certify(kind, w, DEP.EVAL_INDICES, "test")
        returns = result.pop("all_returns")
        result["second_best_return"] = sorted(set(returns))[-2] if len(set(returns)) > 1 else None
        result["whole_space_mean"] = result["distribution"]["mean"]
        report[str(w)] = result
        print(tag, "width", w, result["certificate"], "conforming", result["conforming"],
              "best", result["best_return"], "episodes", result["episodes"],
              "seconds", round(result["seconds"], 1), flush=True)
        (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- arm A3
def arm_with(widths=HELD_OUT_WIDTHS, kind="interpreter", tag="with",
             class_id=None, store_root=None):
    """Instantiate the stored class at each held-out width; score once."""
    store = C.ClassStore(store_root or STORE)
    class_id = class_id or store.records[0].class_id
    report = {"class_id": class_id, "per_width": {}}
    for w in widths:
        started = time.perf_counter()
        names, exact, registry, record = store.instantiate(class_id, w, builder(kind), freeze)
        instantiate_seconds = time.perf_counter() - started
        # Same scoring call the enumeration arm makes, once instead of 272 times.
        _, program, reg2 = DEP.build(kind, w)
        task = DEP.task_for(names, w, DEP.EVAL_INDICES, "test")
        full = {n.name: 0 for n in program.nodes}
        full.update(record.selections)
        ledger = EpisodeLedger()
        t0 = time.perf_counter()
        mean = program_return(program, reg2, task, full, ledger)
        verify_seconds = time.perf_counter() - t0
        row = {"width": w, "programs_evaluated": 1, "episodes": ledger.episodes,
               "steps": ledger.steps, "mean_return": mean, "threshold": DEP.THRESHOLD,
               "conforms": mean is not None and mean >= DEP.THRESHOLD,
               "artifact_digest": exact.digest, "nodes": len(exact.nodes),
               "certificate": certificate_of(False, True,
                                             1 if mean is not None and mean >= DEP.THRESHOLD else 0),
               "instantiate_seconds": instantiate_seconds,
               "verify_seconds": verify_seconds,
               "seconds": instantiate_seconds + verify_seconds,
               "enumeration_performed": False}
        report["per_width"][str(w)] = row
        print(tag, "width", w, "mean", mean, "digest", exact.digest[:12],
              "episodes", ledger.episodes, "seconds", round(row["seconds"], 2), flush=True)
        (OUT / f"{tag}.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- arm A4/A5
def arm_baselines(widths=HELD_OUT_WIDTHS):
    report = {"held_out": {}, "fit": {}}
    for w in widths:
        report["held_out"][str(w)] = DEP.episode_baselines(w, DEP.EVAL_INDICES, "test")
        report["fit"][str(w)] = DEP.episode_baselines(w, DEP.FIT_INDICES, "train")
        print("baselines", w, json.dumps(report["held_out"][str(w)]), flush=True)
    (OUT / "baselines.json").write_text(json.dumps(report, indent=2))
    return report


def arm_difficulty(widths=HELD_OUT_WIDTHS):
    sys.path.insert(0, str(ROOT / "generators" / "logic"))
    from generators.logic.generator import relevant_inputs
    from tcn.types import BOOL                                     # noqa: F401
    report = {}
    for d in widths:
        circuits, finals, relevant, targets = [], [], [], []
        for i in DEP.EVAL_INDICES:
            host = S.make_host(d, i, "test", DEP.BASE, DEP.HORIZON)
            view = host.view().observations
            fields = [round(x) for x in view["program"].decoded]
            gates = [tuple(fields[3 * g:3 * g + 3]) for g in range(d)]
            circuits.append(tuple(gates))
            finals.append(gates[-1][2])
            relevant.append(len(relevant_inputs(DEP.BASE["inputs"], [list(g) for g in gates])))
            targets.append(bool(host.records[0].probes["target"].decoded))
        report[str(d)] = {
            "episodes": len(DEP.EVAL_INDICES),
            "program_fields": 3 * d,
            "distinct_circuits": len(set(circuits)),
            "distinct_final_tables": sorted(set(finals)),
            "relevant_inputs_histogram": {str(k): relevant.count(k) for k in sorted(set(relevant))},
            "min_relevant_inputs_achieved": min(relevant),
            "fraction_target_true": sum(targets) / len(targets),
            "majority_fraction": max(sum(targets), len(targets) - sum(targets)) / len(targets)}
        print("difficulty", d, json.dumps(report[str(d)]), flush=True)
    (OUT / "difficulty.json").write_text(json.dumps(report, indent=2))
    return report


def arm_crosscheck(width=HELD_OUT_WIDTHS[0]):
    """The shipped enumerator on the same space at a held-out width must agree."""
    names, program, registry = DEP.build("interpreter", width)
    task = DEP.task_for(names, width, DEP.EVAL_INDICES, "test")
    t0 = time.perf_counter()
    result = enumerate_environment(program, task, registry, threshold=DEP.THRESHOLD)
    mine = json.loads((OUT / "without.json").read_text())[str(width)]
    free = [k for k, _ in _free(program)]
    report = {"width": width, "enumerate_environment": result.to_dict(),
              "seconds": time.perf_counter() - t0,
              "agrees": {"evaluated": result.evaluated == mine["evaluated"],
                         "space_size": result.space_size == mine["space_size"],
                         "exhausted": result.exhausted == mine["exhausted"],
                         "conforming": result.conforming == mine["conforming"],
                         "certificate": result.certificate == mine["certificate"],
                         "best_return": result.best_return == mine["best_return"],
                         "chosen": ({k: result.selections[k] for k in free}
                                    in mine["conforming_selections"])}}
    report["all_agree"] = all(report["agrees"].values())
    print(json.dumps(report["agrees"], indent=2), "all_agree", report["all_agree"], flush=True)
    (OUT / "crosscheck.json").write_text(json.dumps(report, indent=2))
    return report


# ---------------------------------------------------------------- arm F
def arm_f():
    """§53's arm F, run against the storage format rather than against a score."""
    report = {}

    # F1 -- wrong schema, certified at ONE width, offered to the format.
    _, program, registry, _, r1 = _certify("first_gate", 1, DEP.FIT_INDICES, "train")
    sel1 = r1["conforming_selections"][0] if r1["conforming_selections"] else {}
    full = {n.name: 0 for n in program.nodes}
    full.update(sel1)
    dig1 = frozen_selection(program, full, registry).digest if sel1 else ""
    c1 = C.Certification(width=1, space_size=r1["space_size"], evaluated=r1["evaluated"],
                         exhausted=r1["exhausted"], conforming=r1["conforming"],
                         certificate=r1["certificate"], mean_return=r1["best_return"] or 0.,
                         threshold=DEP.THRESHOLD, artifact_digest=dig1)
    rec1 = _record_for("first_gate", sel1, [c1], [dig1], {}, {"arm": "F1"})
    store = C.ClassStore(OUT / "library_armf")
    try:
        store.admit(rec1, threshold=DEP.THRESHOLD)
        f1 = {"admitted": True, "refusal": None}
    except C.ClassError as exc:
        f1 = {"admitted": False, "refusal": str(exc), "rule": str(exc).split(":")[0]}
    report["F1"] = {"width_1_sweep": {k: r1[k] for k in
                                      ("space_size", "evaluated", "exhausted", "conforming",
                                       "certificate", "best_return", "episodes", "seconds")},
                    "selections": sel1, "artifact_digest": dig1, **f1}
    print("F1", f1, flush=True)

    # F2 -- wrong schema, certified at TWO widths.
    _, program2, registry2, _, r2 = _certify("first_gate", 2, DEP.FIT_INDICES, "train")
    full2 = {n.name: 0 for n in program2.nodes}
    full2.update(sel1)
    task2 = DEP.task_for(DEP.build("first_gate", 2)[0], 2, DEP.FIT_INDICES, "train")
    ledger2 = EpisodeLedger()
    mean2 = program_return(program2, registry2, task2, full2, ledger2)
    dig2 = frozen_selection(program2, full2, registry2).digest
    conforming2 = 1 if (mean2 is not None and mean2 >= DEP.THRESHOLD) else 0
    c2 = C.Certification(width=2, space_size=r2["space_size"], evaluated=r2["evaluated"],
                         exhausted=r2["exhausted"], conforming=conforming2,
                         certificate=certificate_of(True, True, conforming2),
                         mean_return=mean2 if mean2 is not None else 0.,
                         threshold=DEP.THRESHOLD, artifact_digest=dig2)
    rec2 = _record_for("first_gate", sel1, [c1, c2], [dig1, dig2], {}, {"arm": "F2"})
    try:
        C.ClassStore(OUT / "library_armf2").admit(rec2, threshold=DEP.THRESHOLD)
        f2 = {"admitted": True, "refusal": None}
    except C.ClassError as exc:
        f2 = {"admitted": False, "refusal": str(exc), "rule": str(exc).split(":")[0]}
    report["F2"] = {"width_2_whole_space": {k: r2[k] for k in
                                            ("space_size", "evaluated", "exhausted",
                                             "conforming", "certificate", "best_return",
                                             "episodes", "seconds")},
                    "stored_vector_mean_return_at_width_2": mean2,
                    "artifact_digest": dig2, **f2}
    print("F2", f2, "vector at width 2 scores", mean2, flush=True)

    # F3 -- a deliberately unsound format: one width admitted. What a consumer gets.
    unsound = C.ClassStore(OUT / "library_unsound")
    unsound.records = [rec1]
    unsound._save()
    rows = {}
    for w in HELD_OUT_WIDTHS:
        names, exact, reg, record = unsound.instantiate(rec1.class_id, w,
                                                        builder("first_gate"), freeze)
        _, p, r = DEP.build("first_gate", w)
        task = DEP.task_for(names, w, DEP.EVAL_INDICES, "test")
        f = {n.name: 0 for n in p.nodes}
        f.update(record.selections)
        led = EpisodeLedger()
        mean = program_return(p, r, task, f, led)
        rows[str(w)] = {"mean_return": mean, "threshold": DEP.THRESHOLD,
                        "conforms": mean is not None and mean >= DEP.THRESHOLD,
                        "artifact_digest": exact.digest, "episodes": led.episodes}
        print("F3 width", w, "mean", mean, flush=True)
    report["F3"] = {"note": "R1 bypassed on purpose; this is the unsound format",
                    "per_width": rows}

    # F4 -- the wrong schema's WHOLE space at each held-out width.
    f4 = {}
    for w in HELD_OUT_WIDTHS:
        _, _, _, _, r = _certify("first_gate", w, DEP.EVAL_INDICES, "test")
        r.pop("all_returns")
        f4[str(w)] = r
        print("F4 width", w, r["certificate"], "conforming", r["conforming"],
              "best", r["best_return"], flush=True)
        report["F4"] = f4
        (OUT / "armf.json").write_text(json.dumps(report, indent=2))
    (OUT / "armf.json").write_text(json.dumps(report, indent=2))
    return report


def arm_confirm():
    """Secondary: the same with/without comparison at §53's own widths."""
    without = {}
    for w in CONFIRM_WIDTHS:
        _, _, _, _, r = _certify("interpreter", w, DEP.EVAL_INDICES, "test")
        r.pop("all_returns")
        without[str(w)] = r
        print("confirm-without", w, r["certificate"], r["conforming"], r["best_return"], flush=True)
    with_ = arm_with(CONFIRM_WIDTHS, tag="confirm_with")
    (OUT / "confirm.json").write_text(json.dumps({"without": without, "with": with_}, indent=2))


ARMS = {"construct": arm_construct, "without": arm_without, "with": arm_with,
        "baselines": arm_baselines, "difficulty": arm_difficulty,
        "crosscheck": arm_crosscheck, "armf": arm_f, "confirm": arm_confirm}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        ARMS[name]()
