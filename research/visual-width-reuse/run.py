"""Arms A/B/C/F/N of `PREREGISTRATION.md`, on the shipped visual parser.

Subcommands
-----------
`enumerate  --resolution R`  arms A and B: the whole staged pipeline
                             (S0 -> S1' -> S2') exhausted at one raster
                             resolution, with baselines and the parse.
`construct`                  admit the class from the certified resolutions.
`instantiate --resolution R` arm C: rebuild the schema at a held-out resolution
                             and apply the stored vector; zero programs searched.
`armf       --resolution R`  arm F: the two deliberately wrong schemas.
`refusal`                    F-a: what a resolution-32 artifact does at 40.
`null`                       the §59 timing null control.

Every cost is charged on all four currencies of PREREGISTRATION §2.2, and both
sides of every ratio go through the same code path.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import time
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / 'research' / 'visual-ladder'),
           str(ROOT / 'research' / 'class-identity'), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import FLAT, accuracy, bytes_type, episode, exact_error, sweep   # noqa: E402
import rung3_widgets as R                                                    # noqa: E402
import rung3_root as RR                                                      # noqa: E402
from tcn.operators import Registry                                           # noqa: E402
from tcn.search import evaluate, space_size                                  # noqa: E402
from tcn.types import Value                                                  # noqa: E402
from tcn.generation import source_fingerprint                                # noqa: E402

import schema as S                                                           # noqa: E402
from classes import (Certification, ClassError, ClassRecord, ClassStore,      # noqa: E402
                     schema_class_id)

import torch                                                                 # noqa: E402

# Host-safety rule after the 2026-09-10 unified-memory crash: one thread per
# worker, so a capped scope's CPU quota is not multiplied by torch's pool.
torch.set_num_threads(1)

OUT = HERE / "out"
TOL = 1e-6
CERTIFIED = (24, 32)
HELD_OUT = (16, 40, 48)

# The supervision budget, identical at every resolution so a width comparison is
# not a data comparison.  These are `rung3_root.main`'s defaults.
TRAIN, VALIDATION, HELD = 6, 3, 6
PAIRS, CORNER_PER_IMAGE = 48, 120
PARSE_FIRST = 200
# The whole-space baseline scores *every* program in the space, so it is the one
# measurement whose cost is quadratic in the supervision.  It is capped at a
# fixed prefix of the held-out rows -- identical at every resolution -- and the
# chosen program is re-scored on exactly those rows so the comparison is
# like for like.  `held_accuracy` is always on the full held-out set.
WHOLE_SPACE_ROWS = 150


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=1, sort_keys=True))


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def config(resolution):
    cfg = dict(FLAT)
    cfg["resolution"] = resolution
    return cfg


# ------------------------------------------------------------------ baselines

def best_constant(examples, signals):
    """Component-wise most frequent target: the trivial predictor to beat."""
    columns = None
    rows = 0
    for example in examples:
        flat = []
        for signal in signals:
            flat.extend(example["targets"][signal.target].flat())
        if columns is None:
            columns = [Counter() for _ in flat]
        for i, v in enumerate(flat):
            columns[i][v] += 1
        rows += 1
    if not columns or not rows:
        return {"accuracy": 0., "components": 0, "rows": 0}
    hits = sum(c.most_common(1)[0][1] for c in columns)
    return {"accuracy": hits / (rows * len(columns)), "components": len(columns), "rows": rows}


def whole_space_accuracy(program, examples, signals, registry, cap=4096):
    """Every program in the space, scored on the same held-out rows.

    This is §55's `whole_space_mean` and `second_best`: an accuracy that does not
    beat the mean of the space it was drawn from has not earned the search.
    """
    import itertools
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    total = space_size(program)
    scores = []
    started = time.perf_counter()
    for i, combination in enumerate(itertools.product(*counts)):
        if i >= cap:
            break
        scores.append(accuracy(program, examples, signals, registry,
                               dict(zip(names, combination)), TOL))
    scores.sort(reverse=True)
    return {"space_size": total, "scored": len(scores), "exhaustive": len(scores) == total,
            "mean": statistics.fmean(scores), "best": scores[0],
            "second_best": scores[1] if len(scores) > 1 else None,
            "worst": scores[-1], "seconds": time.perf_counter() - started}


def exhaustive_walk(program, examples, signals, registry, cap=1 << 22):
    """Arm B's currency: every program in the space, one `evaluate` each.

    Deliberately the *same* `tcn.search.evaluate` the with-arm's conformance
    check calls, so the with/without ratio is a ratio of identical work rather
    than of two interpreters.  `node_units` is programs x rows x nodes -- the
    machine-independent count -- and the shipped `enumerate_prefix` search is
    timed separately in `sweep`.
    """
    import itertools
    names = [n.name for n in program.nodes]
    counts = [range(len(n.candidates)) for n in program.nodes]
    total = space_size(program)
    found, evaluated = [], 0
    started = time.perf_counter()
    for combination in itertools.product(*counts):
        if evaluated >= cap:
            break
        evaluated += 1
        selections = dict(zip(names, combination))
        error = evaluate(program, selections, examples, signals, registry, TOL)
        if error is not None and error <= TOL:
            found.append(selections)
    exhausted = evaluated >= total
    return {"space_size": total, "evaluated": evaluated, "exhausted": exhausted,
            "conforming": len(found), "conforming_selections": found,
            "certificate": ("unique" if exhausted and len(found) == 1 else
                            "none exists" if exhausted and not found else
                            "complete" if exhausted else "budget-limited"),
            "rows": len(examples), "nodes": len(program.nodes),
            "node_units": evaluated * len(examples) * len(program.nodes),
            "row_evaluations": evaluated * len(examples),
            "seconds": time.perf_counter() - started}


def conformance_check(program, selections, examples, signals, registry):
    """The with-arm's whole cost: one program, one pass, no certificate."""
    started = time.perf_counter()
    error = evaluate(program, selections, examples, signals, registry, TOL)
    return {"programs_evaluated": 1, "rows": len(examples), "nodes": len(program.nodes),
            "node_units": len(examples) * len(program.nodes),
            "row_evaluations": len(examples),
            "max_error": None if error is None else float(error),
            "conforms": error is not None and error <= TOL,
            "certificate": "none",
            "seconds": time.perf_counter() - started}


# ------------------------------------------------------------------- the data

def datasets(resolution):
    cfg = config(resolution)
    probe = episode(0, "train", **cfg)
    W, H = probe["width"], probe["height"]
    d = {"W": W, "H": H, "configuration": cfg}
    d["s0"] = {
        "train": R.same_examples(range(TRAIN), "train", PAIRS, seed=1, **cfg),
        "validation": R.same_examples(range(50, 50 + VALIDATION), "validation", PAIRS,
                                      seed=2, **cfg),
        "held": R.same_examples(range(100, 100 + HELD), "test", PAIRS, seed=3, **cfg)}
    d["s1"] = {
        "train": RR.corner_examples_all(range(TRAIN), "train", CORNER_PER_IMAGE, seed=4, **cfg),
        "validation": RR.corner_examples_all(range(50, 50 + VALIDATION), "validation",
                                             CORNER_PER_IMAGE, seed=5, **cfg),
        "held": RR.corner_examples_all(range(100, 100 + HELD), "test", CORNER_PER_IMAGE,
                                       seed=6, **cfg)}
    d["s2"] = {
        "train": RR.rect_examples_all(range(TRAIN), "train", **cfg),
        "validation": RR.rect_examples_all(range(50, 50 + VALIDATION), "validation", **cfg),
        "held": RR.rect_examples_all(range(100, 100 + HELD), "test", **cfg)}
    return d


SIGNALS = {"s0": R.same_signals, "s1": R.corner_signals, "s2": R.rect_signals}


def achieved(resolution):
    """Achieved difficulty, never the requested config."""
    cfg = config(resolution)
    counts, mw, mh = [], 0, 0
    for split, seeds in (("train", range(TRAIN)), ("test", range(100, 100 + HELD))):
        for s in seeds:
            ep = episode(s, split, **cfg)
            counts.append(len(ep["probes"]["hierarchy"]))
            for d in ep["probes"]["hierarchy"]:
                mw = max(mw, d["rect"][2])
                mh = max(mh, d["rect"][3])
    return {"episodes": len(counts), "widgets_mean": statistics.fmean(counts),
            "widgets_min": min(counts), "widgets_max": max(counts),
            "max_widget_width": mw, "max_widget_height": mh,
            "span_used": max(config(resolution)["resolution"], 1),
            "span_sufficient": max(mw, mh) <= resolution}


# --------------------------------------------------------------------- parse

def run_parse(registry, corner_module, rect_module, resolution, screens):
    cfg = config(resolution)
    rows = []
    for seed in range(PARSE_FIRST, PARSE_FIRST + screens):
        ep = episode(seed, "test", **cfg)
        BT = bytes_type(ep["width"], ep["height"])
        positions = RR.all_positions(ep)
        parser = R.assembly(registry, BT, positions, corner_module, rect_module)
        started = time.perf_counter()
        got = parser.run({"observation": Value.of(BT, ep["pixels"])}, registry=registry)[0]
        predicted = [list(map(int, r)) for r in sorted(got["mapped"].decoded)]
        rows.append(RR.score_with_root(predicted, ep) |
                    {"seed": seed, "seconds": time.perf_counter() - started,
                     "positions": len(positions)})
    total = {"screens": len(rows),
             "widgets_in_probe": sum(r["widgets_in_probe"] for r in rows),
             "rects_predicted": sum(r["rects_predicted"] for r in rows),
             "screens_rects_exact": sum(r["rects_exact"] for r in rows),
             "roots_predicted": sum(r["roots_predicted"] for r in rows),
             "parent_links_correct": sum(r["parent_links_correct"] for r in rows),
             "parent_links_wrong": sum(r["parent_links_wrong"] for r in rows),
             "corner_key_collisions": sum(r["corner_key_collisions"] for r in rows),
             "trees_exact": sum(r["tree_exact"] for r in rows),
             "seconds": sum(r["seconds"] for r in rows)}
    total["rect_recall"] = total["rects_predicted"] / max(1, total["widgets_in_probe"])
    total["link_accuracy"] = total["parent_links_correct"] / max(1, total["widgets_in_probe"])
    return {"episodes": rows, "totals": total}


# --------------------------------------------------------- arm A / arm B

def arm_enumerate(resolution, screens, schema=S.RIGHT, tag=None, whole_space=True):
    cfg = config(resolution)
    data = datasets(resolution)
    W, H = data["W"], data["H"]
    registry = Registry()
    result = {"arm": "without", "schema": schema.name, "resolution": resolution,
              "width": W, "height": H, "observation_components": 3 * W * H,
              "configuration": cfg, "offset_pool": list(schema.offsets(W)),
              "span": schema.span(W, H), "achieved": achieved(resolution),
              "source_fingerprint": source_fingerprint(), "stages": {}}
    vectors, modules = {}, {}
    started_all = time.perf_counter()

    for stage in S.STAGES:
        signals = SIGNALS[stage]()
        rows = data[stage]
        if stage == "s0":
            program = schema.build_s0(registry, W, H)
        elif stage == "s1":
            program = schema.build_s1(registry, W, H, modules["s0"])
        else:
            program = schema.build_s2(registry, W, H, modules["s0"])
        row = {"stage": stage, "nodes": len(program.nodes), "space_size": space_size(program),
               "free_nodes": S.free_nodes(program),
               "train_rows": len(rows["train"]), "validation_rows": len(rows["validation"]),
               "held_rows": len(rows["held"])}
        row["walk"] = exhaustive_walk(program, rows["train"], signals, registry)
        row["sweep"] = {k: v for k, v in
                        sweep(program, rows["train"], signals, registry, TOL).items()
                        if k in ("certificate", "conforming", "evaluated", "exhausted",
                                 "node_evaluations", "seconds", "wall", "space_size",
                                 "unique", "solved")}
        conforming = row["walk"].pop("conforming_selections")
        row["conforming_free"] = [{k: v for k, v in s.items() if k in row["free_nodes"]}
                                  for s in conforming]
        if stage == "s0":
            row["distinct_functions"] = len({R.induced_same(s, False) for s in conforming})
        survivors = [s for s in conforming
                     if exact_error(program, rows["validation"], signals, registry, s) <= TOL]
        row["validation_survivors"] = len(survivors)
        if not survivors:
            row["FAILED"] = "0 conforming survive validation -- that is the certificate"
            result["stages"][stage] = row
            dump(tag or f"enum_{resolution}", result)
            raise SystemExit(f"{stage} at resolution {resolution}: {row['FAILED']}")
        chosen = survivors[0]
        vectors[stage] = {k: v for k, v in chosen.items() if k in row["free_nodes"]}
        row["chosen_free"] = vectors[stage]
        row["artifact_digest"] = S.freeze(program, chosen, registry).digest
        row["held_accuracy"] = accuracy(program, rows["held"], signals, registry, chosen, TOL)
        row["held_max_error"] = exact_error(program, rows["held"], signals, registry, chosen)
        row["best_constant"] = best_constant(rows["held"], signals)
        if whole_space:
            sub = rows["held"][:WHOLE_SPACE_ROWS]
            row["whole_space_held"] = whole_space_accuracy(program, sub, signals, registry)
            row["whole_space_held"]["rows"] = len(sub)
            row["whole_space_held"]["chosen_accuracy_same_rows"] = accuracy(
                program, sub, signals, registry, chosen, TOL)
            row["whole_space_held"]["best_constant_same_rows"] = \
                best_constant(sub, signals)["accuracy"]
        if stage in ("s1", "s2"):
            row["chosen_offsets"] = [schema.offsets(W)[chosen[k]] for k in
                                     (("back_a", "back_b") if stage == "s1"
                                      else ("step_w", "step_h"))]
        modules[stage] = S.module_of(program, chosen, registry)
        row["module"] = modules[stage]
        result["stages"][stage] = row
        print(f"  res {resolution} {stage}: space {row['space_size']} evaluated "
              f"{row['walk']['evaluated']} exhausted {row['walk']['exhausted']} conforming "
              f"{row['walk']['conforming']} certificate {row['walk']['certificate']} "
              f"held_acc {row['held_accuracy']:.4f} ({row['walk']['seconds']:.1f}s walk / "
              f"{row['sweep']['seconds']:.1f}s prefix)")
        dump(tag or f"enum_{resolution}", result)

    result["selections"] = vectors
    result["pipeline"] = {
        "programs_evaluated": sum(result["stages"][s]["walk"]["evaluated"] for s in S.STAGES),
        "row_evaluations": sum(result["stages"][s]["walk"]["row_evaluations"] for s in S.STAGES),
        "node_units": sum(result["stages"][s]["walk"]["node_units"] for s in S.STAGES),
        "walk_seconds": sum(result["stages"][s]["walk"]["seconds"] for s in S.STAGES),
        "prefix_seconds": sum(result["stages"][s]["sweep"]["seconds"] for s in S.STAGES),
        "prefix_node_evaluations": sum(result["stages"][s]["sweep"]["node_evaluations"]
                                       for s in S.STAGES),
        "staged_space": sum(result["stages"][s]["space_size"] for s in S.STAGES),
        "joint_space": (result["stages"]["s0"]["space_size"] *
                        result["stages"]["s1"]["space_size"] *
                        result["stages"]["s2"]["space_size"]),
        "total_seconds": time.perf_counter() - started_all}
    result["parse"] = run_parse(registry, modules["s1"], modules["s2"], resolution, screens)
    print(f"  res {resolution} parse: {json.dumps(result['parse']['totals'])}")
    dump(tag or f"enum_{resolution}", result)
    return result


# ------------------------------------------------------------------- arm C

def store_root(name="library"):
    return OUT / name


def arm_construct(store_name="library", schema=S.RIGHT, certified=CERTIFIED,
                  tag="construct", threshold_key="best_constant"):
    runs = {w: load(f"enum_{w}") for w in certified}
    vectors = runs[certified[-1]]["selections"]
    disagreements = {s: [runs[w]["selections"][s] for w in certified]
                     for s in S.STAGES
                     if len({json.dumps(runs[w]["selections"][s], sort_keys=True)
                             for w in certified}) > 1}
    store = ClassStore(store_root(store_name))
    out = {"certified_widths": list(certified), "schema": schema.name,
           "vectors": vectors, "vector_disagreements": disagreements,
           "source_fingerprint": source_fingerprint(), "records": {}}
    for stage in S.STAGES:
        free = runs[certified[-1]]["stages"][stage]["free_nodes"]
        record = ClassRecord(
            class_id=schema_class_id(source_fingerprint(), schema.qualified(stage),
                                     vectors[stage]),
            kind="schema",
            schema={"module": "research.visual-width-reuse.schema",
                    "qualified_name": schema.qualified(stage),
                    "free_nodes": sorted(free.items()),
                    "source_fingerprint": source_fingerprint(),
                    "note": schema.note},
            selections=vectors[stage],
            certified=tuple(Certification(
                width=w,
                space_size=runs[w]["stages"][stage]["space_size"],
                evaluated=runs[w]["stages"][stage]["walk"]["evaluated"],
                exhausted=runs[w]["stages"][stage]["walk"]["exhausted"],
                conforming=runs[w]["stages"][stage]["walk"]["conforming"],
                certificate=runs[w]["stages"][stage]["walk"]["certificate"],
                mean_return=runs[w]["stages"][stage]["held_accuracy"],
                threshold=runs[w]["stages"][stage][threshold_key]["accuracy"],
                artifact_digest=runs[w]["stages"][stage]["artifact_digest"])
                for w in certified),
            members=tuple(runs[w]["stages"][stage]["artifact_digest"] for w in certified),
            provenance={"track": "research/visual-width-reuse", "stage": stage,
                        "generator": "gui", "axis": "raster resolution"})
        try:
            store.admit(record)
            out["records"][stage] = {"admitted": True, "class_id": record.class_id,
                                     "widths": list(record.widths),
                                     "selections": record.selections,
                                     "digests": list(record.members)}
        except ClassError as exc:
            out["records"][stage] = {"admitted": False, "rule": str(exc),
                                     "class_id": record.class_id}
        print(f"  {stage}: {out['records'][stage]}")
    out["all_admitted"] = all(v.get("admitted") for v in out["records"].values())
    dump(tag, out)
    return out


def arm_instantiate(resolution, screens, store_name="library", tag=None):
    construct = load("construct")
    store = ClassStore(store_root(store_name))
    data = datasets(resolution)
    W, H = data["W"], data["H"]
    registry = Registry()
    result = {"arm": "with", "resolution": resolution, "width": W, "height": H,
              "observation_components": 3 * W * H, "stages": {},
              "achieved": achieved(resolution)}
    vectors = construct["vectors"]
    modules = {}
    started_all = time.perf_counter()

    for stage in S.STAGES:
        signals = SIGNALS[stage]()
        rows = data[stage]
        t0 = time.perf_counter()
        builder = S.stage_builder(S.RIGHT, stage, vectors)
        names, exact, sub_registry, record = store.instantiate(
            construct["records"][stage]["class_id"], resolution, builder, S.freeze)
        build_seconds = time.perf_counter() - t0
        # Rebuild in *this* registry so the pipeline shares one registry, exactly
        # as the without arm does; the digest is the store's, checked below.
        if stage == "s0":
            program = S.RIGHT.build_s0(registry, W, H)
        elif stage == "s1":
            program = S.RIGHT.build_s1(registry, W, H, modules["s0"])
        else:
            program = S.RIGHT.build_s2(registry, W, H, modules["s0"])
        chosen = S.full(program, vectors[stage])
        row = {"stage": stage, "programs_searched": 0, "nodes": len(program.nodes),
               "space_size": space_size(program), "build_seconds": build_seconds,
               "artifact_digest": S.freeze(program, chosen, registry).digest,
               "store_digest": exact.digest,
               "selections": vectors[stage],
               "certificate": "none (a conformance check, never a uniqueness certificate)"}
        row["check"] = conformance_check(program, chosen, rows["train"], signals, registry)
        row["held_accuracy"] = accuracy(program, rows["held"], signals, registry, chosen, TOL)
        row["held_max_error"] = exact_error(program, rows["held"], signals, registry, chosen)
        row["best_constant"] = best_constant(rows["held"], signals)
        modules[stage] = S.module_of(program, chosen, registry)
        row["module"] = modules[stage]
        result["stages"][stage] = row
        print(f"  res {resolution} {stage}: 0 searched, digest {row['artifact_digest']} "
              f"conforms {row['check']['conforms']} held_acc {row['held_accuracy']:.4f} "
              f"({row['check']['seconds']:.2f}s)")
        dump(tag or f"with_{resolution}", result)

    result["pipeline"] = {
        "programs_evaluated": sum(result["stages"][s]["check"]["programs_evaluated"]
                                  for s in S.STAGES),
        "row_evaluations": sum(result["stages"][s]["check"]["row_evaluations"]
                               for s in S.STAGES),
        "node_units": sum(result["stages"][s]["check"]["node_units"] for s in S.STAGES),
        "check_seconds": sum(result["stages"][s]["check"]["seconds"] for s in S.STAGES),
        "build_seconds": sum(result["stages"][s]["build_seconds"] for s in S.STAGES),
        "total_seconds": time.perf_counter() - started_all}
    result["parse"] = run_parse(registry, modules["s1"], modules["s2"], resolution, screens)
    print(f"  res {resolution} parse: {json.dumps(result['parse']['totals'])}")
    dump(tag or f"with_{resolution}", result)
    return result


# ------------------------------------------------------------------- arm F

def arm_f(resolutions, tag="armf"):
    """§53's arm F on this artifact: a schema that agrees at exactly one width."""
    out = {"reference_resolution": S.REFERENCE_RESOLUTION, "schemas": {}}
    right = {w: load(f"enum_{w}") for w in resolutions if (OUT / f"enum_{w}.json").exists()}

    for wrong in (S.FROZEN_OFFSETS, S.FROZEN_SPAN):
        rec = {"note": wrong.note, "per_resolution": {}}
        for resolution in resolutions:
            cfg = config(resolution)
            data = datasets(resolution)
            W, H = data["W"], data["H"]
            registry = Registry()
            # S0 does not depend on either frozen derivation; take the certified
            # vector so the wrong schema is given every advantage.
            vectors = load("construct")["vectors"]
            p0 = S.RIGHT.build_s0(registry, W, H)
            module = S.module_of(p0, S.full(p0, vectors["s0"]), registry)
            row = {"width": W, "offsets": list(wrong.offsets(W)), "span": wrong.span(W, H)}
            for stage in ("s1", "s2"):
                signals = SIGNALS[stage]()
                program = (wrong.build_s1(registry, W, H, module) if stage == "s1"
                           else wrong.build_s2(registry, W, H, module))
                walk = exhaustive_walk(program, data[stage]["train"], signals, registry)
                walk.pop("conforming_selections")
                stored = S.full(program, vectors[stage])
                entry = {"nodes": len(program.nodes), "walk": walk,
                         "stored_vector_conforms":
                             (lambda e: e is not None and e <= TOL)(
                                 evaluate(program, stored, data[stage]["train"], signals,
                                          registry, TOL)),
                         "stored_vector_held_accuracy":
                             accuracy(program, data[stage]["held"], signals, registry,
                                      stored, TOL),
                         "best_constant": best_constant(data[stage]["held"], signals)["accuracy"],
                         "digest": S.freeze(program, stored, registry).digest}
                if resolution in right and stage in right[resolution].get("stages", {}):
                    entry["right_schema_digest"] = \
                        right[resolution]["stages"][stage]["artifact_digest"]
                    entry["bit_identical_to_right"] = \
                        entry["digest"] == entry["right_schema_digest"]
                row[stage] = entry
                print(f"  {wrong.name} res {resolution} {stage}: conforming "
                      f"{walk['conforming']} certificate {walk['certificate']} "
                      f"stored_conforms {entry['stored_vector_conforms']} "
                      f"held_acc {entry['stored_vector_held_accuracy']:.4f}")
            rec["per_resolution"][str(resolution)] = row
            dump(tag, out | {"schemas": out["schemas"] | {wrong.name: rec}})
        out["schemas"][wrong.name] = rec

    # The format's refusals: R1 at one width, R2 at two.
    out["format"] = {}
    for wrong in (S.FROZEN_OFFSETS, S.FROZEN_SPAN):
        vectors = load("construct")["vectors"]
        for label, widths in (("F1_one_width", (S.REFERENCE_RESOLUTION,)),
                              ("F2_two_widths", (24, S.REFERENCE_RESOLUTION))):
            store = ClassStore(store_root(f"library_unsound_{wrong.name}_{label}"))
            stage = "s2"
            per = out["schemas"][wrong.name]["per_resolution"]
            certified = []
            for w in widths:
                d = per[str(w)][stage]
                certified.append(Certification(
                    width=w, space_size=d["walk"]["space_size"],
                    evaluated=d["walk"]["evaluated"], exhausted=d["walk"]["exhausted"],
                    conforming=d["walk"]["conforming"], certificate=d["walk"]["certificate"],
                    mean_return=d["stored_vector_held_accuracy"],
                    threshold=d["best_constant"], artifact_digest=d["digest"]))
            record = ClassRecord(
                class_id=schema_class_id(source_fingerprint(), wrong.qualified(stage),
                                         vectors[stage]),
                kind="schema",
                schema={"module": "research.visual-width-reuse.schema",
                        "qualified_name": wrong.qualified(stage),
                        "free_nodes": sorted(
                            {"step_w": 5, "step_h": 5}.items()),
                        "source_fingerprint": source_fingerprint(), "note": wrong.note},
                selections=vectors[stage], certified=tuple(certified),
                members=tuple(c.artifact_digest for c in certified),
                provenance={"arm": "F", "schema": wrong.name})
            try:
                store.admit(record)
                out["format"][f"{wrong.name}/{label}"] = {"admitted": True}
            except ClassError as exc:
                out["format"][f"{wrong.name}/{label}"] = {"admitted": False, "rule": str(exc)}
            print(f"  format {wrong.name}/{label}: {out['format'][f'{wrong.name}/{label}']}")
    dump(tag, out)
    return out


# ------------------------------------------------------------ F-a: the refusal

def arm_refusal(built_at=32, offered_at=(16, 40, 48), tag="refusal"):
    """Does a resolution-32 artifact actually refuse a resolution-40 screen?

    PREREGISTRATION F-a: §1.1 is a code reading.  This is the measurement.
    """
    vectors = load("construct")["vectors"]
    registry = Registry()
    W = H = built_at
    p0 = S.RIGHT.build_s0(registry, W, H)
    module = S.module_of(p0, S.full(p0, vectors["s0"]), registry)
    p2 = S.RIGHT.build_s2(registry, W, H, module)
    hard = p2.harden(S.full(p2, vectors["s2"]))
    out = {"built_at": built_at, "digest": S.freeze(p2, S.full(p2, vectors["s2"]),
                                                    registry).digest, "offered": {}}
    for res in (built_at,) + tuple(offered_at):
        rows = RR.rect_examples_all(range(1), "train", **config(res))
        try:
            hard.execute(rows[0]["inputs"], registry=registry)
            out["offered"][str(res)] = {"accepted": True}
        except Exception as exc:                                   # noqa: BLE001
            out["offered"][str(res)] = {"accepted": False, "error": type(exc).__name__,
                                        "message": str(exc)[:300]}
        print(f"  artifact built at {built_at} offered at {res}: {out['offered'][str(res)]}")
    dump(tag, out)
    return out


# ------------------------------------------------------------------ arm N

def arm_null(repeats=7, tag="null"):
    """§59's instrument floor: the same work timed repeatedly, nothing changed."""
    data = datasets(32)
    registry = Registry()
    vectors = load("construct")["vectors"]
    p0 = S.RIGHT.build_s0(registry, 32, 32)
    module = S.module_of(p0, S.full(p0, vectors["s0"]), registry)
    p1 = S.RIGHT.build_s1(registry, 32, 32, module)
    chosen = S.full(p1, vectors["s1"])
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        evaluate(p1, chosen, data["s1"]["train"], R.corner_signals(), registry, TOL)
        times.append(time.perf_counter() - t0)
    out = {"repeats": repeats, "seconds": times, "median": statistics.median(times),
           "min": min(times), "max": max(times),
           "spread_fraction": (max(times) - min(times)) / statistics.median(times)}
    print("  null control:", json.dumps(out))
    dump(tag, out)
    return out


# --------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("enumerate", "construct", "instantiate",
                                        "armf", "refusal", "null"))
    ap.add_argument("--resolution", type=int)
    ap.add_argument("--screens", type=int, default=6)
    ap.add_argument("--tag")
    ap.add_argument("--no-whole-space", action="store_true")
    a = ap.parse_args()
    if a.command == "enumerate":
        arm_enumerate(a.resolution, a.screens, tag=a.tag, whole_space=not a.no_whole_space)
    elif a.command == "construct":
        arm_construct()
    elif a.command == "instantiate":
        arm_instantiate(a.resolution, a.screens, tag=a.tag)
    elif a.command == "armf":
        arm_f((16, 24, 32, 40, 48))
    elif a.command == "refusal":
        arm_refusal()
    elif a.command == "null":
        arm_null()


if __name__ == "__main__":
    main()
