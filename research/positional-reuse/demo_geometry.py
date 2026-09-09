"""End to end: learn one sub-program, crystallize it, apply it at every pixel.

Task. `generators/geometry` at resolution 8 with six objects and one camera step
(an ordinary generator action, so the episode stays replayable). The agent sees
only `pixels`. The dense probe `object_ids` gives a per-pixel label; the
supervision is the per-pixel predicate `object_ids >= 0` (foreground).
`inspect_geometry2.py` measures that this predicate is exactly determined by the
pixel, so an exact program exists.

Stage A. A module scaffold `(position, pixel bytes) -> (position, foreground)` is
learned by `tcn.synthesis.fit` over (episode, position) pairs -- every pixel of
every training episode is one training example, which is the dense probe doing
the work. `tcn.search.enumerate_fit` runs the same space as the discrete
reference, and reports whether the training set identifies the program uniquely.
The exported program is frozen and registered as a module.

Stage B. `hold / pair / map` applies that one module at all 64 positions of a
held-out episode with three caller nodes, and the result is compared against the
whole dense probe at once.

Stage C. `filter / count` over the same result set, showing that the pattern
composes and that an aggregate readout needs no set-to-tuple conversion.

Stage D. The same module called position by position, for cost comparison.

Byte values are compared with `eq`, never added: `role="byte"` makes
`Type.numeric` false, so the type system already forbids arithmetic on a pixel
byte. Nothing here preprocesses the observation; the byte field is reached with
an ordinary `project` node inside the graph.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from shared import IDX, Builder, per_position_program, shared_map_program
from tcn.generation import SCALAR, Action, Host
from tcn.graph import Candidate, Node, Signal
from tcn.operators import Registry
from tcn.search import enumerate_fit
from tcn.synthesis import fit
from tcn.types import BOOL, Value, integer, product, setof

RES = 8
OBJECTS = 6
APPROACH = (-2.4, -2.4, -4.2)
VEC3 = product(SCALAR, SCALAR, SCALAR)
BYTE = integer(8, signed=False, role="byte")
BYTES = product(*(BYTE for _ in range(RES * RES * 3)))
REC = product(IDX, BYTES)
PIXEL_STARTS = tuple(3 * i for i in range(RES * RES))
LABELLED = product(IDX, BOOL)
RECORD_SET = setof(LABELLED, RES * RES)
COUNT = integer(32, signed=False)
BACKGROUND = (24, 30, 43)

# Byte values the comparison nodes may choose between. 24/30/43 is the renderer's
# background; the rest are distractors, so the choice is a real search.
CANDIDATE_BYTES = (24, 30, 43, 99, 150)
TRUE_PICK = (0, 1, 2)          # indices of 24, 30, 43
TRUE_TRUTH = (8, 7)            # and, then nand -> not(r==24 and g==30 and b==43)

TRAIN_SEEDS = tuple(range(4))
HELD_SEEDS = tuple(range(100, 110))


def episode(seed, split="train"):
    h = Host.create("geometry", seed=seed, split=split,
                    configuration={"resolution": RES, "objects": OBJECTS, "horizon": 4})
    rec = h.step([Action("camera", arguments=(("delta", Value.of(VEC3, APPROACH)),))])
    view = rec.actor_view()
    assert set(view.observations) == {"pixels"}, view.observations
    data = Value.of(BYTES, view.observations["pixels"].decoded[3])
    labels = tuple(lab >= 0 for lab in rec.probes["object_ids"].decoded)
    return data, labels


def pixel_examples(seeds, split="train"):
    """One training example per pixel: the dense probe, unaggregated."""
    out = []
    for seed in seeds:
        data, labels = episode(seed, split)
        for i, start in enumerate(PIXEL_STARTS):
            out.append({"inputs": {"rec": Value.of(REC, (start, data.decoded))},
                        "targets": {"foreground": Value.of(BOOL, labels[i])}})
    return out


def module_scaffold(registry, searched=True):
    """The searched space: which byte each channel is compared to, and how the
    three comparisons combine. 5^3 * 16^2 = 32000 discrete programs."""
    consts = tuple((f"byte_{v}", Value.of(BYTE, v)) for v in CANDIDATE_BYTES)
    consts += (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("g_at", "add", ["pos", "one"])
    b.add("b_at", "add", ["pos", "two"])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    for k, (name, source) in enumerate((("cmp_r", "red"), ("cmp_g", "green"), ("cmp_b", "blue"))):
        picks = CANDIDATE_BYTES if searched else (CANDIDATE_BYTES[TRUE_PICK[k]],)
        cands = [Candidate(registry.resolve("eq", (BYTE, BYTE)), (source, f"byte_{v}")) for v in picks]
        node = Node(name, BOOL, tuple(cands), "core", b.depths[source] + 1, None if searched else 0)
        b.nodes.append(node); b.types[name] = BOOL; b.depths[name] = node.depth
    for k, (name, sources) in enumerate((("rg", ("cmp_r", "cmp_g")), ("foreground", ("rg", "cmp_b")))):
        ks = range(16) if searched else (TRUE_TRUTH[k],)
        cands = [Candidate(registry.resolve(f"truth_{t}", (BOOL, BOOL)), sources) for t in ks]
        depth = max(b.depths[s] for s in sources) + 1
        node = Node(name, BOOL, tuple(cands), "core", depth, None if searched else 0)
        b.nodes.append(node); b.types[name] = BOOL; b.depths[name] = node.depth
    b.add("record", "tuple", ["pos", "foreground"])
    return b.program((("y", "record"),))


def keep_flag_module(registry):
    """`(position, flag) -> flag`, the predicate `filter` needs."""
    b = Builder(registry, (("rec", LABELLED),))
    b.add("flag", "project", ["rec"], params={"index": 1})
    return b.program((("y", "flag"),))


def target_set(labels):
    """The dense probe as a relation of indexed values (ARCHITECTURE section 1).

    A declared representation change on the supervision side only, and lossless:
    positions are unique, so the set determines the sequence.
    """
    return Value.of(RECORD_SET, tuple((start, labels[i]) for i, start in enumerate(PIXEL_STARTS)))


def structural_symbols(program, registry, seen=None):
    """Nodes plus constants, counting each module definition once.

    `description_bits` measures serialized JSON length, which for these programs
    is dominated by the 192-field observation type repeated in every node's
    signature. This counts program structure instead.
    """
    seen = set() if seen is None else seen
    total = len(program.nodes) + len(program.constants)
    for node in program.nodes:
        for c in node.candidates:
            for name in (c.operator.name, dict(c.operator.parameters).get("module", "")):
                if name.startswith("module:") and name not in seen:
                    seen.add(name)
                    total += structural_symbols(registry.modules[name], registry, seen)
    return total


def report(name, value):
    print(f"{name:52s} {value}")


def main():
    torch.manual_seed(0)
    results = {"configuration": {
        "generator": "geometry", "resolution": RES, "objects": OBJECTS,
        "camera_delta": APPROACH, "positions": len(PIXEL_STARTS),
        "observation_width": BYTES.width, "candidate_bytes": list(CANDIDATE_BYTES),
        "train_seeds": list(TRAIN_SEEDS), "held_out_seeds": list(HELD_SEEDS)}}

    train = pixel_examples(TRAIN_SEEDS)
    positives = sum(ex["targets"]["foreground"].decoded for ex in train)
    report("stage A training pixels (foreground fraction)",
           f"{len(train)}  ({positives/len(train):.1%})")
    results["train"] = {"pixels": len(train), "foreground_fraction": positives / len(train)}

    signals = (Signal("foreground", "foreground", ("core",), BOOL, "bce"),)
    r = Registry()
    scaffold = module_scaffold(r)
    space = 1
    for n in scaffold.nodes:
        space *= len(n.candidates)
    report("stage A search space", space)

    # Discrete reference over exactly the same candidate space.
    search = enumerate_fit(scaffold, train, signals, r, tolerance=1e-6, stop_at_first=False)
    report("enumerate_fit  solved / unique / seconds",
           f"{search.solved} / {search.unique} / {search.seconds:.1f}")
    results["enumeration"] = search.to_dict()

    started = time.perf_counter()
    model, info = fit(scaffold, train, signals, steps=400, lr=.15, freeze=True, registry=r,
                      tolerance=1e-6, polish=0)
    gradient_seconds = time.perf_counter() - started
    report("synthesis.fit  exact_max_error / frozen / seconds",
           f"{info['exact_max_error']:.3g} / {info['fully_frozen']} / {gradient_seconds:.1f}")
    results["gradient"] = {k: info[k] for k in
                           ("exact_max_error", "exact_conformance", "fully_frozen", "relaxed_loss")}
    results["gradient"]["seconds"] = gradient_seconds
    results["gradient"]["selections"] = model.selections()
    agree = search.solved and search.selections == model.selections()
    report("gradient and enumeration pick the same program", agree)
    results["agreement_with_enumeration"] = agree

    reference = module_scaffold(r, searched=False)
    learned = model.export() if info["exact_conformance"] else scaffold.harden(search.selections)
    report("module taken from", "gradient" if info["exact_conformance"] else "enumeration")
    results["module_source"] = "gradient" if info["exact_conformance"] else "enumeration"
    def chosen(program):
        return [(n.name, n.candidates[n.selected].operator.name,
                 n.candidates[n.selected].sources) for n in program.nodes]

    results["module_matches_renderer_test"] = chosen(learned) == chosen(reference)
    report("learned module identical to the renderer's own test",
           results["module_matches_renderer_test"])

    name = r.register_module(learned)
    ref_name = r.register_module(reference)

    # Stage B: three caller nodes, all 64 positions, one module.
    caller = shared_map_program(r, BYTES.width, PIXEL_STARTS, [name], element=BYTE)
    ref_caller = shared_map_program(r, BYTES.width, PIXEL_STARTS, [ref_name], element=BYTE)
    per = per_position_program(r, BYTES.width, PIXEL_STARTS, name, element=BYTE)
    report("stage B caller nodes / stage D per-position caller nodes",
           f"{len(caller.nodes)} / {len(per.nodes)}")

    # Stage C: filter then count, composed on top of the same result set.
    keep = r.register_module(keep_flag_module(r))
    b = Builder(r, (("observation", BYTES),),
                (("positions", Value.of(setof(IDX, len(PIXEL_STARTS)), PIXEL_STARTS)),
                 ("empty", Value.of(setof(BYTES, 1), ()))))
    b.add("held", "insert", ["empty", "observation"])
    b.add("records", "pair", ["positions", "held"])
    b.add("mapped", "map", ["records"], params={"module": name})
    b.add("kept", "filter", ["mapped"], params={"module": keep})
    b.add("total", "count", ["kept"])
    counter = b.program((("y", "total"),))

    errors, count_errors, held_fg = [], [], 0
    for seed in HELD_SEEDS:
        data, labels = episode(seed, split="test")
        held_fg += sum(labels)
        target = target_set(labels)
        got = caller.run({"observation": data}, registry=r)[0]["y"]
        errors.append(max((abs(a - b) for a, b in zip(got.flat(), target.flat())), default=0.))
        assert ref_caller.run({"observation": data}, registry=r)[0]["y"].decoded == target.decoded
        total = counter.run({"observation": data}, registry=r)[0]["y"]
        count_errors.append(abs(total.decoded - sum(labels)))
    held_pixels = len(HELD_SEEDS) * len(PIXEL_STARTS)
    report("stage B held-out pixels (foreground fraction)",
           f"{held_pixels}  ({held_fg/held_pixels:.1%})")
    report("stage B held-out max error over the dense probe", max(errors))
    report("stage C held-out max foreground-count error", max(count_errors))
    results["held_out"] = {"pixels": held_pixels, "foreground_fraction": held_fg / held_pixels,
                           "max_error": max(errors), "max_count_error": max(count_errors)}

    data, _ = episode(HELD_SEEDS[0], split="test")
    x = {"observation": data}
    for p in (caller, per, counter):
        p.run(x, registry=r)

    def timed(p, repeats=5):
        start = time.perf_counter()
        for _ in range(repeats):
            p.run(x, registry=r)
        return (time.perf_counter() - start) / repeats

    cost = {}
    for label, prog in (("shared", caller), ("per_position", per)):
        cost[label] = {"caller_nodes": len(prog.nodes),
                       "description_bits": prog.description_bits(r),
                       "structural_symbols": structural_symbols(prog, r),
                       "execution_cost": prog.execution_cost(r),
                       "seconds": timed(prog)}
        c = cost[label]
        report(f"{label:12s} nodes / bits / symbols / cost / ms",
               f"{c['caller_nodes']:4d} / {c['description_bits']:9d} / "
               f"{c['structural_symbols']:5d} / {c['execution_cost']:7.0f} / "
               f"{1e3*c['seconds']:.1f}")
    results["cost"] = cost
    results["module"] = {"nodes": len(learned.nodes),
                         "description_bits": learned.description_bits(),
                         "execution_cost": learned.execution_cost(r), "digest": learned.digest}

    path = pathlib.Path(__file__).parent / "demo_geometry.json"
    path.write_text(json.dumps(results, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
