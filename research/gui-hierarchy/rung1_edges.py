"""Rung one, and only rung one: widget edges from raw GUI pixels.

The target is the generator's own ownership boundary,
`owner(i) != owner(i + step)`, taken from the `owner` probe and never from the
raster.  The programme is the one the discrete-perception track established:
search a per-position module discretely, then apply it at every position with
three caller nodes.  Addresses are **computed** from the module's own position
argument by `add`, never chosen from a menu over the image, because relaxing an
input address is measured worse than chance in three independent places.

HAND-INITIALISATION, DECLARED.  Everything the scaffold supplies is coarse
structure in the sense of ARCHITECTURE section 4, and every item is ablated in
`main()` with both numbers reported side by side:

  H1  the region, the node depths and the predecessor pool -- one `core` region,
      a fixed DAG.  Not ablated: without a graph there is no scaffold.
  H2  the offset pool the `shifted` node may choose from, hand-restricted to five
      plausible spatial steps.  ABLATED by `--wide-offsets`, which offers every
      byte offset from 1 to 3*width+3 instead.
  H3  the channel correspondence: `cmp_c` compares channel c here against
      channel c there.  ABLATED by the `free` arm, where each comparison also
      may take a constant byte from a pool, so the operand binding is searched.
  H5  single-candidate nodes are written `selected=0`, which `SoftProgram` treats
      as frozen and evaluates detached -- severing the gradient to every choice
      upstream of one.  ABLATED by `relax_single=True`, which leaves them relaxed
      at no change to the discrete space.
  H4  choice-logit initialisation noise for the gradient arm.  `SoftProgram`
      zero-initialises every logit, so a seed alone changes nothing; noise breaks
      the symmetry without supplying an answer.  ABLATED by `init_noise=0`,
      reported as the single outcome it is.

No choice is initialised to, or restricted to, the correct answer.  The correct
offset (3) is one of five candidates in the narrow pool and one of 51 in the wide
one; both combinator nodes offer all 16 two-input truth tables.
"""
from __future__ import annotations

import argparse
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, IDX, Builder, accuracy, all_conforming, bytes_type, dump,
                    edge_positions, enumerate_reference, episode, exact_error, gradient_arm,
                    positional_caller, random_reference, record_type, report, summarise)
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import space_size
from tcn.types import BOOL, Value, product, setof

POOL = (30, 100, 170)          # distractor byte values for the `free` arm

# A screen dense enough that the rung-1 target is not a report on a constant.
# `dial.py` measures that requesting 12 widgets at min_size 6 and R=16 achieves
# only 3 -- the request is not the achievement, which is the F-bench lesson --
# so the achieved count is reported alongside every number here.
SCREEN = {"resolution": 16, "widgets": 12, "nesting": 4, "min_size": 4}


def induced_function(program, selections, offsets):
    """The (offset, operand bindings, 3-input Boolean function) a selection denotes.

    Two conforming programs that denote the same function are not two answers;
    they are one answer written twice, because `rg` and `edge` between them can
    express the same composite in more than one way.  Counting distinct
    *functions* rather than distinct *selections* is what separates "the probe
    underdetermines the target" from "the scaffold is redundant".
    """
    def truth(t, x, y):
        return bool((t >> (2 * int(x) + int(y))) & 1)
    table = tuple(truth(selections["edge"], truth(selections["rg"], a, b), c)
                  for a in (False, True) for b in (False, True) for c in (False, True))
    operands = tuple(selections.get(f"cmp_{c}", 0) for c in "rgb")
    return (offsets[selections["shifted"]], operands, table)


def narrow_offsets(width):
    """Five plausible spatial steps in bytes; 3 (one pixel right) is the target."""
    return (3, 6, 9, 3 * width, 3 * width + 3)


def wide_offsets(width):
    """Every byte offset up to one row and one pixel: no hand restriction at all."""
    return tuple(range(1, 3 * width + 4))


def scaffold(registry, ep, offsets, free=False, relax_single=False):
    """`(position, image bytes) -> (position, edge)`.

    Free choices: the neighbour offset, optionally each channel comparison's
    second operand, and the two Boolean combinators.
    """
    REC = record_type(ep["width"], ep["height"])
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in offsets)
    if free:
        consts += tuple((f"byte_{v}", Value.of(BYTE, v)) for v in POOL)
    b = Builder(registry, (("rec", REC),), consts, relax_single=relax_single)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("shifted", [("add", ("pos", f"off{k}"), None) for k in offsets])
    b.add("here_g", "add", ["pos", "one"])
    b.add("here_b", "add", ["pos", "two"])
    b.add("there_g", "add", ["shifted", "one"])
    b.add("there_b", "add", ["shifted", "two"])
    for tag, base in (("here", "pos"), ("there", "shifted")):
        b.add(f"{tag}_r_v", "index", ["obs", base])
        b.add(f"{tag}_g_v", "index", ["obs", f"{tag}_g"])
        b.add(f"{tag}_b_v", "index", ["obs", f"{tag}_b"])
    for channel in ("r", "g", "b"):
        options = [("eq", (f"here_{channel}_v", f"there_{channel}_v"), None)]
        if free:
            options += [("eq", (f"here_{channel}_v", f"byte_{v}"), None) for v in POOL]
        b.choice(f"cmp_{channel}", options)
    b.choice("rg", [(f"truth_{t}", ("cmp_r", "cmp_g"), None) for t in range(16)])
    b.choice("edge", [(f"truth_{t}", ("rg", "cmp_b"), None) for t in range(16)])
    b.add("record", "tuple", ["pos", "edge"])
    return b.program((("y", "record"),))


def signals():
    return (Signal("edge", "edge", ("core",), BOOL, "bce"),)


def edge_target_set(ep, step_axis="x"):
    owner = ep["probes"]["owner"]
    step = 1 if step_axis == "x" else ep["width"]
    positions = edge_positions(ep)
    return Value.of(setof(product(IDX, BOOL), len(positions)),
                    tuple((3 * i, owner[i] != owner[i + step]) for i in positions))


def examples(seeds, split, per_image=None, seed=0, step_axis="x", **configuration):
    """One supervised example per (episode, position); the dense probe, unaggregated."""
    rows = []
    rng = random.Random(seed)
    for s in seeds:
        ep = episode(s, split, **configuration)
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(bytes_type(ep["width"], ep["height"]), ep["pixels"]).raw
        owner = ep["probes"]["owner"]
        step = 1 if step_axis == "x" else ep["width"]
        positions = edge_positions(ep)
        chosen = positions if per_image is None else [
            positions[i] for i in sorted(rng.sample(range(len(positions)),
                                                    min(per_image, len(positions))))]
        for i in chosen:
            rows.append({"inputs": {"rec": Value(REC, (3 * i, raw))},
                         "targets": {"edge": Value.of(BOOL, owner[i] != owner[i + step])}})
    rng.shuffle(rows)
    return rows


def arm(registry, name, ep, offsets, free, train, validation, held, tolerance, gradient_seeds,
        steps, lr, conform_limit=None, run_gradient=True):
    """One scaffold, measured discretely and by gradient descent on the same space."""
    build = lambda: scaffold(registry, ep, offsets, free)
    program = build()
    out = {"name": name, "free_operands": free, "offsets": list(offsets),
           "space_size": space_size(program), "nodes": len(program.nodes),
           "train_records": len(train), "validation_records": len(validation),
           "held_records": len(held)}
    report(f"[{name}] space / nodes / records",
           f"{out['space_size']} / {out['nodes']} / {len(train)}")

    sig = signals()
    if out["space_size"] <= 20000:
        out["enumerate_fit"] = enumerate_reference(program, train, sig, registry, tolerance)
        report(f"[{name}] enumerate_fit solved/exhausted/unique/evaluated/s",
               f"{out['enumerate_fit']['solved']} / {out['enumerate_fit']['exhausted']} / "
               f"{out['enumerate_fit']['unique']} / {out['enumerate_fit']['evaluated']} / "
               f"{out['enumerate_fit']['seconds']:.1f}")

    survey = all_conforming(program, train, sig, registry, tolerance, limit=conform_limit)
    conforming = survey.pop("conforming")
    out["conforming"] = survey
    report(f"[{name}] conforming / space / exhausted / s",
           f"{survey['count']} / {survey['space_size']} / {survey['exhausted']} / "
           f"{survey['seconds']:.1f}")

    # The lexicographically first conforming program is exactly what
    # `enumerate_fit` returns, since it walks the same `itertools.product` order.
    if conforming:
        functions = {induced_function(program, s, offsets) for s in conforming}
        out["conforming"]["distinct_functions"] = len(functions)
        first = conforming[0]
        out["lexicographic_pick"] = {
            "selections": first,
            "validation_max_error": exact_error(program, validation, sig, registry, first),
            "held_max_error": exact_error(program, held, sig, registry, first),
            "held_accuracy": accuracy(program, held, sig, registry, first)}
        # The measured fix: require exactness at EVERY position of a validation
        # split drawn from different episodes, not a lexicographic tie-break.
        survivors = [s for s in conforming
                     if exact_error(program, validation, sig, registry, s) <= tolerance]
        out["validation_filtered"] = {
            "survivors": len(survivors),
            "unique": len(survivors) == 1,
            "distinct_functions": len({induced_function(program, s, offsets) for s in survivors}),
            "held_max_error": [exact_error(program, held, sig, registry, s) for s in survivors[:8]],
            "held_accuracy": [accuracy(program, held, sig, registry, s) for s in survivors[:8]],
            "selections": survivors[:8]}
        report(f"[{name}] lexicographic pick held-out max error / accuracy",
               f"{out['lexicographic_pick']['held_max_error']} / "
               f"{out['lexicographic_pick']['held_accuracy']:.6f}")
        report(f"[{name}] conforming selections / distinct functions",
               f"{survey['count']} / {out['conforming']['distinct_functions']}")
        report(f"[{name}] validation-filtered survivors / distinct functions / held-out errors",
               f"{len(survivors)} / {out['validation_filtered']['distinct_functions']} / "
               f"{out['validation_filtered']['held_max_error']}")
    out["random_control"] = random_reference(program, train, sig, registry, draws=400,
                                             tolerance=tolerance)
    report(f"[{name}] random-draw solution density",
           f"{out['random_control']['density']:.4f}")

    if run_gradient:
        rows = gradient_arm(build, tuple(range(gradient_seeds)), train, sig, registry,
                            steps=steps, lr=lr, init_noise=.5, label=name, tolerance=tolerance,
                            held=held)
        out["gradient_noise"] = {"init_noise": .5, "rows": rows, "summary": summarise(rows)}
        plain = gradient_arm(build, (0,), train, sig, registry, steps=steps, lr=lr,
                             init_noise=0., label=name + " noise0", tolerance=tolerance,
                             held=held)
        out["gradient_zero_init"] = {"init_noise": 0., "rows": plain, "summary": summarise(plain),
                                     "note": "SoftProgram zero-initialises every choice logit, "
                                             "so this is one outcome, not a seed average"}
        report(f"[{name}] gradient successes (noise .5 / noise 0)",
               f"{out['gradient_noise']['summary']['successes']}/{gradient_seeds} "
               f"(held exact {out['gradient_noise']['summary']['held_exact']}) / "
               f"{out['gradient_zero_init']['summary']['successes']}/1")
    return out, program, conforming


def module_as_candidate(registry, program, good, bad, seeds, tolerance=1e-6, **configuration):
    """Stage C: freeze rung one, register it, and let a second program CHOOSE it.

    A parser the system learns from pixels against probes and then crystallises is
    an earned abstraction under ARCHITECTURE section 4, not the hand-written
    detector AGENTS.md forbids as a model input.  Registering it makes it one
    typed candidate operator: its internals are charged to description size and
    its execution is charged per call.  Here a single `map` node is offered the
    edge module and a same-shaped distractor (the identical program reading the
    row below instead of the pixel to the right), and the choice is *searched*.
    """
    names = [registry.register_module(program.harden(s)) for s in (good, bad)]
    probe = episode(seeds[0], "test", **configuration)
    BT = bytes_type(probe["width"], probe["height"])
    positions = tuple(3 * i for i in edge_positions(probe))
    caller = positional_caller(registry, BT, positions, names)
    label = setof(product(IDX, BOOL), len(positions))
    sig = (Signal("mapped", "edges", ("core",), label, "mse"),)
    rows = []
    for seed in seeds:
        ep = episode(seed, "test", **configuration)
        rows.append({"inputs": {"observation": Value.of(BT, ep["pixels"])},
                     "targets": {"edges": edge_target_set(ep)}})
    started = time.perf_counter()
    survey = all_conforming(caller, rows, sig, registry, tolerance)
    chosen = survey.pop("conforming")
    return {"candidates": len(names), "module_names": names,
            "caller_nodes": len(caller.nodes), "positions": len(positions),
            "episodes": len(rows), "space_size": survey["space_size"],
            "conforming": survey["count"], "unique": survey["unique"],
            "selected_candidate": chosen[0]["mapped"] if chosen else None,
            "selected_is_edge_module": bool(chosen) and chosen[0]["mapped"] == 0,
            "seconds": time.perf_counter() - started,
            "module_execution_cost": registry.modules[names[0]].execution_cost(registry),
            "module_description_bits": registry.modules[names[0]].description_bits(),
            "caller_execution_cost": caller.execution_cost(registry),
            "caller_description_bits": caller.description_bits(registry)}


def apply_everywhere(registry, program, selections, seeds, step_axis="x", **configuration):
    """Stage B: the found module at every position of fresh screens, three caller nodes."""
    module = program.harden(selections)
    name = registry.register_module(module)
    errors, seconds = [], []
    for seed in seeds:
        ep = episode(seed, "test", **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        positions = tuple(3 * i for i in edge_positions(ep))
        caller = positional_caller(registry, BT, positions, [name])
        owner = ep["probes"]["owner"]
        step = 1 if step_axis == "x" else ep["width"]
        target = Value.of(setof(product(IDX, BOOL), len(positions)),
                          tuple((3 * i, owner[i] != owner[i + step]) for i in edge_positions(ep)))
        started = time.perf_counter()
        got = caller.run({"observation": Value.of(BT, ep["pixels"])}, registry=registry)[0]["mapped"]
        seconds.append(time.perf_counter() - started)
        errors.append(max((abs(a - b) for a, b in zip(got.flat(), target.flat())), default=0.))
    return {"episodes": len(seeds), "positions": len(positions), "caller_nodes": len(caller.nodes),
            "max_error": max(errors), "median_seconds": sorted(seconds)[len(seconds) // 2]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--validation", type=int, default=4)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=24)
    ap.add_argument("--apply", type=int, default=3)
    ap.add_argument("--gradient-seeds", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=.15)
    ap.add_argument("--free-limit", type=int, default=None)
    ap.add_argument("--tag", default="rung1")
    args = ap.parse_args()

    registry = Registry()
    tolerance = 1e-6
    configuration = dict(SCREEN)
    probe = episode(0, "train", **configuration)
    offsets = narrow_offsets(probe["width"])
    achieved = [len(episode(s, "train", **configuration)["probes"]["hierarchy"])
                for s in range(args.train)]
    result = {"arguments": vars(args), "configuration": configuration,
              "achieved_widgets_mean": sum(achieved) / len(achieved),
              "achieved_widgets_max": max(achieved),
              "observation_bytes": len(probe["pixels"]),
              "positions_per_screen": len(edge_positions(probe)),
              "narrow_offsets": list(offsets), "wide_offsets": len(wide_offsets(probe["width"]))}

    train = examples(range(args.train), "train", args.per_image, seed=1, **configuration)
    validation = examples(range(50, 50 + args.validation), "validation", args.per_image, seed=2,
                          **configuration)
    held = examples(range(100, 100 + args.held), "test", args.per_image, seed=3, **configuration)
    result["positive_fraction"] = sum(ex["targets"]["edge"].decoded for ex in train) / len(train)
    report("edge records (positive fraction)",
           f"{len(train)} ({result['positive_fraction']:.1%})")

    print("\n--- arm A: narrow offsets, channel correspondence supplied (H2, H3 in place) ---")
    result["narrow"], program, conforming = arm(
        registry, "narrow", probe, offsets, False, train, validation, held, tolerance,
        args.gradient_seeds, args.steps, args.lr)
    dump(args.tag, result)

    if conforming:
        survivors = [s for s in conforming
                     if exact_error(program, validation, signals(), registry, s) <= tolerance]
        result["stage_b"] = apply_everywhere(registry, program, survivors[0],
                                             range(200, 200 + args.apply), **configuration)
        report("stage B: module at every position of fresh screens, max error",
               result["stage_b"]["max_error"])
        dump(args.tag, result)

        distractor = dict(survivors[0])
        distractor["shifted"] = offsets.index(3 * probe["width"])
        result["stage_c"] = module_as_candidate(registry, program, survivors[0], distractor,
                                                tuple(range(200, 200 + args.apply)),
                                                **configuration)
        report("stage C: registered module chosen over distractor / conforming / unique",
               f"{result['stage_c']['selected_is_edge_module']} / "
               f"{result['stage_c']['conforming']} / {result['stage_c']['unique']}")
        dump(args.tag, result)

    print("\n--- ablation H2: the offset pool is not hand-restricted ---")
    result["wide_offsets_ablation"], _, _ = arm(
        registry, "wide-offsets", probe, wide_offsets(probe["width"]), False, train, validation,
        held, tolerance, args.gradient_seeds, args.steps, args.lr)
    dump(args.tag, result)

    print("\n--- ablation H3: the channel correspondence is searched, not supplied ---")
    result["free_operand_ablation"], _, _ = arm(
        registry, "free-operands", probe, offsets, True, train, validation, held, tolerance,
        args.gradient_seeds, args.steps, args.lr, conform_limit=args.free_limit)
    dump(args.tag, result)


if __name__ == "__main__":
    main()
