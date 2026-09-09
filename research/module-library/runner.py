"""Curriculum runner for the three-stage chain: publish, inherit, measure.

Every stage runs twice over the identical data, supervision and tolerance --
once with the previous stage's crystallized module available as a candidate
operator and once without -- and the two are reported side by side. The
inheriting arm is what the stage's gates judge and what it publishes.

The runner is passed to `tcn.curriculum.Curriculum.run`, which is the shipped
orchestrator: the artifact flow being exercised here is the shipped one, not a
private path. `Artifacts.modules` carries exactly the library references the
stage declared in `inherits`, so a stage cannot reach a module it did not name.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research" / "module-library"))

import chain as C  # noqa: E402
from common import all_conforming  # noqa: E402
from tcn.library import Library  # noqa: E402
from tcn.operators import Registry  # noqa: E402
from tcn.search import evaluate, program_cost, space_size  # noqa: E402
from tcn.types import Value  # noqa: E402

TOLERANCE = 1e-6


def accuracy(program, examples, signals, registry, selections=None):
    hits = 0
    for ex in examples:
        try:
            _, _, trace = program.execute(ex["inputs"], registry=registry, selections=selections)
        except Exception:
            return 0.
        hits += all(abs(a - b) <= TOLERANCE
                    for s in signals
                    for a, b in zip(trace[s.source].flat(), ex["targets"][s.target].flat()))
    return hits / len(examples)


def majority(examples, target):
    labels = [bool(ex["targets"][target].decoded) for ex in examples]
    return max(sum(labels), len(labels) - sum(labels)) / len(labels)


def density(program, examples, signals, registry, draws=400, seed=0):
    """Chance rate: how often a uniform draw from this space conforms."""
    rng = random.Random(seed)
    counts = [len(n.candidates) for n in program.nodes]
    names = [n.name for n in program.nodes]
    hits = 0
    for _ in range(draws):
        selections = dict(zip(names, (rng.randrange(c) for c in counts)))
        error = evaluate(program, selections, examples, signals, registry, TOLERANCE)
        hits += int(error is not None and error <= TOLERANCE)
    return {"draws": draws, "hits": hits, "density": hits / draws}


def arm(program, train, held, signals, registry, target, label, budget=1 << 24,
        with_density=True, validation=None):
    """One search arm, reported identically whatever it inherited.

    The whole conforming set is collected rather than the first hit, because
    conformance on a supervised sample is not identification: the
    discrete-perception track measured `enumerate_fit`'s lexicographic pick to
    be wrong on fresh episodes at exactly this rung. When `validation` is given,
    the pick is the first conforming program that is *also* exact at every
    position of a validation split -- the documented fix, applied here rather
    than rediscovered.
    """
    total = space_size(program)
    started = time.perf_counter()
    sweep = all_conforming(program, train, signals, registry, tolerance=TOLERANCE, limit=budget)
    conforming = sweep["conforming"]
    survivors = conforming
    if validation is not None:
        survivors = [s for s in conforming
                     if (lambda e: e is not None and e <= TOLERANCE)(
                         evaluate(program, s, validation, signals, registry, TOLERANCE))]
    selections = survivors[0] if survivors else None
    row = {"arm": label, "space_size": total, "evaluated": sweep["evaluated"],
           "exhausted": sweep["exhausted"], "budget": budget,
           "conforming": sweep["count"], "unique": sweep["unique"] if sweep["exhausted"] else None,
           "validated": len(survivors) if validation is not None else None,
           "solved": bool(selections), "selections": selections,
           "nodes": len(program.nodes), "wall_seconds": time.perf_counter() - started}
    row["seconds_per_program"] = row["wall_seconds"] / max(1, row["evaluated"])
    row["projected_exhaustive_seconds"] = row["seconds_per_program"] * total
    if selections:
        row["held_out_max_error"] = C.exact_error(program, held, signals, registry,
                                                  selections=selections)
        row["held_out_accuracy"] = accuracy(program, held, signals, registry, selections)
        bits, cost = program_cost(program, selections, registry)
        row["description_bits"], row["execution_cost"] = bits, cost
    else:
        row["held_out_max_error"] = None
        row["held_out_accuracy"] = 0.
        row["description_bits"] = row["execution_cost"] = None
    row["majority_class_accuracy"] = majority(held, target)
    if with_density:
        row["random_control"] = density(program, train, signals, registry)
    return row


def module_agreement(registry, name, examples, target, port="rec", field=1):
    """Exactness of an *inherited* module on the consuming stage's own data.

    A stage that consumes a module inherits its errors. Stage 2's target is
    exact by construction from the generator's probe, so one wrong position in
    the parent makes the child's target unreachable -- not harder, unreachable.
    This measures that interface rather than assuming it, using the same probe
    the parent was supervised with.
    """
    module = registry.modules[name]
    wrong = 0
    for ex in examples:
        out, _ = module.run({port: ex["inputs"][port]}, registry=registry)
        got = next(iter(out.values())).decoded
        got = got[field] if isinstance(got, tuple) else got
        wrong += int(bool(got) != bool(ex["targets"][target].decoded))
    return {"records": len(examples), "wrong": wrong,
            "accuracy": 1 - wrong / max(1, len(examples)), "exact": wrong == 0}


def inherited_interface(registry, name, resolution, cfg):
    """The inherited module measured at every position of each split."""
    train_seeds, held_seeds = _seeds(cfg)
    rows = {}
    for tag, seeds, split in (("train", train_seeds, "train"),
                              ("validation", tuple(range(300, 300 + cfg.get("validation_seeds", 4))),
                               "validation"),
                              ("test", held_seeds, "test")):
        examples = C.pixel_examples(seeds, resolution, split, None, seed=99)
        rows[tag] = module_agreement(registry, name, examples, "foreground")
    return rows


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------

def _seeds(configuration):
    return (tuple(range(configuration.get("train_seeds", 6))),
            tuple(range(100, 100 + configuration.get("held_seeds", 6))))


def stage_foreground(stage, out, artifacts):
    """Stage 1: foreground/background from raw pixels. Nothing to inherit."""
    cfg = stage.configuration
    R = cfg.get("resolution", 8)
    train_seeds, held_seeds = _seeds(cfg)
    registry = Registry()
    train = C.pixel_examples(train_seeds, R, "train", cfg.get("per_image", 48), seed=1)
    held = C.pixel_examples(held_seeds, R, "test", cfg.get("per_image", 48), seed=2)
    scaffold = C.module_scaffold(registry, R, C.POOL)
    signals = C.fg_signals()
    # Validation is a third seed range in its own split, kept away from both the
    # supervised sample and the held-out test episodes, and read at *every*
    # position rather than the supervised subsample.
    validation = C.pixel_examples(tuple(range(300, 300 + cfg.get("validation_seeds", 2))),
                                  R, "validation", None, seed=7)
    row = arm(scaffold, train, held, signals, registry, "foreground", "stage1",
              validation=validation)
    if not row["solved"]:
        raise ValueError("stage 1 found no program that survives validation")
    module = scaffold.harden(row["selections"])
    every = C.pixel_examples(held_seeds, R, "test", None, seed=12)
    row["accuracy_every_position_held_out"] = accuracy(scaffold, every, signals, registry,
                                                       row["selections"])
    row["validation_records"] = len(validation)
    library = Library(artifacts.library)
    entry = library.publish("perception.foreground", module, registry,
                            fixture=[x["inputs"] for x in train[:8]],
                            provenance={"stage": stage.name, "resolution": R,
                                        "space_size": row["space_size"],
                                        "held_out_max_error": row["held_out_max_error"],
                                        "training_records": len(train)})
    result = {"resolution": R, "train_examples": len(train), "held_examples": len(held),
              "arms": [row], "published": entry.reference,
              "module_digest": entry.digest, "module_nodes": entry.nodes,
              "module_execution_cost": entry.execution_cost,
              "module_description_bits": entry.description_bits,
              "solved": int(row["solved"]), "space_size": row["space_size"],
              "held_out_max_error": row["held_out_max_error"],
              "held_out_accuracy": row["held_out_accuracy"],
              "accuracy_every_position_held_out": row["accuracy_every_position_held_out"],
              "conforming": row["conforming"], "validated": row["validated"],
              "search_seconds": row["wall_seconds"]}
    _dump(out, result)
    return result


def stage_edge(stage, out, artifacts):
    """Stage 2: the two-position boundary relation, with and without stage 1."""
    cfg = stage.configuration
    R = cfg.get("resolution", 8)
    train_seeds, held_seeds = _seeds(cfg)
    library = Library(artifacts.library)
    registry, aliases = library.load(list(artifacts.modules),
                                     policy=cfg.get("source_policy", "strict"))
    fg_name = aliases["perception.foreground"]
    train = C.window_examples(train_seeds, R, "train", cfg.get("per_image", 48), seed=3)
    held = C.window_examples(held_seeds, R, "test", cfg.get("per_image", 48), seed=4)
    signals = C.edge_signals()

    interface = inherited_interface(registry, fg_name, R, cfg)
    validation = C.window_examples(tuple(range(300, 300 + cfg.get("validation_seeds", 2))),
                                   R, "validation", None, seed=7)
    staged = C.staged_scaffold(registry, R, fg_name)
    inherited = arm(staged, train, held, signals, registry, "edge", "stage2:inherits_stage1",
                    validation=validation)
    if not inherited["solved"]:
        raise ValueError("stage 2 found no program that survives validation")

    # The same target with nothing inherited: every channel comparison and every
    # combinator free again, which is stage 1's content re-searched inside
    # stage 2. Budget-capped, with the exhaustive time projected from the
    # measured rate rather than guessed.
    flat_registry = Registry()
    flat = C.flat_scaffold(flat_registry, R, C.POOL)
    flat_row = arm(flat, train, held, signals, flat_registry, "edge", "stage2:flat",
                   budget=cfg.get("flat_budget", 20000), with_density=False,
                   validation=validation)

    module = staged.harden(inherited["selections"])
    entry = library.publish("perception.edge", module, registry,
                            fixture=[x["inputs"] for x in train[:8]],
                            provenance={"stage": stage.name, "resolution": R,
                                        "inherits": list(artifacts.modules),
                                        "space_size": inherited["space_size"],
                                        "held_out_max_error": inherited["held_out_max_error"]})
    result = {"resolution": R, "train_examples": len(train), "held_examples": len(held),
              "inherited_interface": interface,
              "arms": [inherited, flat_row], "published": entry.reference,
              "module_digest": entry.digest, "module_nodes": entry.nodes,
              "module_execution_cost": entry.execution_cost,
              "module_description_bits": entry.description_bits,
              "module_requires": list(entry.requires),
              "inherited_space": inherited["space_size"], "flat_space": flat_row["space_size"],
              "space_ratio": flat_row["space_size"] / inherited["space_size"],
              "inherited_seconds": inherited["wall_seconds"],
              "flat_projected_seconds": flat_row["projected_exhaustive_seconds"],
              "solved": int(inherited["solved"]), "space_size": inherited["space_size"],
              "held_out_max_error": inherited["held_out_max_error"],
              "held_out_accuracy": inherited["held_out_accuracy"],
              "search_seconds": inherited["wall_seconds"]}
    _dump(out, result)
    return result


def stage_region(stage, out, artifacts):
    """Stage 3: a region-level predicate over stage 2's per-position output."""
    cfg = stage.configuration
    R = cfg.get("resolution", 8)
    tile = cfg.get("tile", C.TILE)
    train_seeds, held_seeds = _seeds(cfg)
    library = Library(artifacts.library)
    registry, aliases = library.load(list(artifacts.modules),
                                     policy=cfg.get("source_policy", "strict"))
    fg_name = aliases["perception.foreground"]
    edge_name = aliases["perception.edge"]
    train = C.region_examples(train_seeds, R, "train", tile, seed=5)
    held = C.region_examples(held_seeds, R, "test", tile, seed=6)
    validation = C.region_examples(tuple(range(300, 300 + cfg.get("validation_seeds", 2))),
                                   R, "validation", tile, seed=7)
    signals = C.region_signals()
    keep = registry.register_module(C.keep_flag_module(registry))
    interface = inherited_interface(registry, fg_name, R, cfg)

    # Arm A -- stage 2 inherited. The mapped module is a *choice* between the
    # inherited edge and two same-shaped distractors built on the same stage-1
    # module with a wrong offset and a wrong combinator, so selection is
    # evidence rather than plumbing.
    distractors = [C.wrapper_over_foreground(registry, R, fg_name, C.offsets(R)[1], 6),
                   C.wrapper_over_foreground(registry, R, fg_name, C.EDGE_STEP, 9)]
    wrappers_a = [registry.register_module(C.wrapper_over_edge(registry, R, edge_name))]
    wrappers_a += [registry.register_module(d) for d in distractors]
    scaffold_a = C.region_scaffold(registry, R, wrappers_a, keep, tile)
    row_a = arm(scaffold_a, train, held, signals, registry, "region",
                "stage3:inherits_stage2", validation=validation)
    if not row_a["solved"]:
        raise ValueError("stage 3 found no program that survives validation")

    # Arm B -- only stage 1 inherited, so stage 2's content is back inside
    # stage 3's search: one wrapper per (offset, combinator).
    registry_b, aliases_b = library.load(["perception.foreground"],
                                         policy=cfg.get("source_policy", "strict"))
    keep_b = registry_b.register_module(C.keep_flag_module(registry_b))
    wrappers_b = [registry_b.register_module(
        C.wrapper_over_foreground(registry_b, R, aliases_b["perception.foreground"], step, truth))
        for step in C.offsets(R) for truth in range(16)]
    scaffold_b = C.region_scaffold(registry_b, R, wrappers_b, keep_b, tile)
    row_b = arm(scaffold_b, train, held, signals, registry_b, "region",
                "stage3:inherits_stage1_only", budget=cfg.get("arm_b_budget", 1 << 24),
                validation=validation)

    # Arm C -- nothing inherited. Reported as a construction result, because
    # `map` takes a module: with an empty library there is no legal program that
    # aggregates over a region at all, and the smallest library that admits one
    # is stage 2's own flat space.
    flat_edge_space = space_size(C.flat_scaffold(Registry(), R, C.POOL))
    arm_c = {"arm": "stage3:no_library", "expressible": False,
             "reason": "map requires a registered module; an empty library admits no aggregate "
                       "over positions, so the region predicate is not a program in this algebra",
             "smallest_admitting_library": "one module drawn from stage 2's flat space",
             "wrapper_space": flat_edge_space,
             "space_size": flat_edge_space * len(C.COMPARISONS) * len(C.THRESHOLDS),
             "projected_seconds": row_b["seconds_per_program"] * flat_edge_space
                                  * len(C.COMPARISONS) * len(C.THRESHOLDS)}

    module = scaffold_a.harden(row_a["selections"])
    wrapper_entry = library.publish("perception.tile_call",
                                    registry.modules[wrappers_a[0]], registry,
                                    fixture=_wrapper_fixture(registry, wrappers_a[0], train),
                                    provenance={"stage": stage.name, "role": "map call site"})
    keep_entry = library.publish("perception.keep_flag", registry.modules[keep], registry,
                                 fixture=[{"rec": Value.of(registry.modules[keep].inputs[0][1],
                                                           (0, True))}],
                                 provenance={"stage": stage.name, "role": "filter predicate"})
    entry = library.publish("perception.region", module, registry,
                            fixture=[x["inputs"] for x in train[:8]],
                            provenance={"stage": stage.name, "resolution": R, "tile": tile,
                                        "inherits": list(artifacts.modules),
                                        "space_size": row_a["space_size"],
                                        "held_out_max_error": row_a["held_out_max_error"]})
    result = {"resolution": R, "tile": tile, "train_examples": len(train),
              "inherited_interface": interface,
              "held_examples": len(held), "arms": [row_a, row_b, arm_c],
              "published": [wrapper_entry.reference, keep_entry.reference, entry.reference],
              "module_digest": entry.digest, "module_nodes": entry.nodes,
              "module_execution_cost": entry.execution_cost,
              "module_description_bits": entry.description_bits,
              "module_requires": list(entry.requires),
              "space_ratio_stage1_only": row_b["space_size"] / row_a["space_size"],
              "seconds_ratio_stage1_only": row_b["wall_seconds"] / max(1e-9, row_a["wall_seconds"]),
              "solved": int(row_a["solved"]), "space_size": row_a["space_size"],
              "held_out_max_error": row_a["held_out_max_error"],
              "held_out_accuracy": row_a["held_out_accuracy"],
              "majority_class_accuracy": row_a["majority_class_accuracy"],
              "module_selected": int(bool(row_a["solved"] and row_a["selections"]["mapped"] == 0)),
              "search_seconds": row_a["wall_seconds"]}
    _dump(out, result)
    return result


def _wrapper_fixture(registry, wrapper, examples, limit=4):
    module = registry.modules[wrapper]
    port, kind = module.inputs[0]
    offsets = C.tile_offsets(8)
    out = []
    for i, ex in enumerate(examples[:limit]):
        rec = ex["inputs"]["rec"]
        out.append({port: Value(kind, (offsets[i % len(offsets)], rec.raw))})
    return out


OPERATIONS = {"foreground": stage_foreground, "edge": stage_edge, "region": stage_region}


def _dump(out, result):
    out = pathlib.Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))


def stage_runner(stage, out, artifacts):
    if stage.operation not in OPERATIONS:
        raise ValueError("unknown stage operation " + stage.operation)
    return OPERATIONS[stage.operation](stage, pathlib.Path(out), artifacts)
