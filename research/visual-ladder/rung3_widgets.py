"""Rung three: widget-level structure, and the first screenshot-to-hierarchy parse.

Four stages, each searched over an exhaustible space so each carries a
certificate, each staged on the frozen output of the one below:

  S0  `same(a, b, obs) -> bool`     -- rung one with the offset *externalised*.
      `research/gui-hierarchy` learned `owner(i) != owner(i+3)` with the offset
      baked in; the same Boolean function with two free address arguments is
      strictly more reusable and costs the same 256-program space.  Supervised by
      `owner(a) == owner(b)` at pairs drawn from the whole raster, which is
      denser supervision than the neighbour-only rung and is what makes the
      answer unique as a function.
  S1  `corner(rec) -> bool`         -- two calls to the frozen S0 module at two
      searched offsets, combined by a searched truth table.  Supervised by the
      `hierarchy` probe: is this position a widget's top-left corner.
  S2  `rect(rec) -> (x, y, w, h, key, parent_key)` -- at a corner, the widget's
      rectangle by same-colour counting along the row and the column, plus the
      packed colour of the pixel to the left, which `bounds.py` measures to be
      the parent at 214/214.
  S3  the assembly -- `insert`/`pair`/`filter`/`map`: four caller nodes turn the
      whole raster into `set[(x, y, w, h, key, parent_key)]`, which is the
      `hierarchy` probe's relation up to the episode's colour relabelling.

HAND-INITIALISATION, DECLARED.  Ablations are in `main()` and both numbers are
reported side by side; where the space is exhaustible the enumeration certificate
is the stronger statement and it is reported too.

  H1  one `core` region, fixed node depths, a fixed predecessor pool.  Not
      ablated: without a graph there is no scaffold.
  H2  S0's channel correspondence -- `cmp_c` compares channel c of a against
      channel c of b.  ABLATED by `--free`, which lets each comparison take a
      constant byte from a pool instead, so the operand binding is searched.
  H3  S1's offset pool: five plausible spatial steps.  The correct pair (3 and
      3W) is NOT at index 0 of the pool, because a zero-initialised logit and
      `enumerate_fit`'s tie-break both prefer candidate 0.
  H4  S2's step pool: the same five steps, searched independently for the row
      and the column direction.  The module must discover that a row step is 3
      bytes and a column step is 3W.
  H5  S2's counting structure -- one term per candidate offset along the row and
      the column, masked by an in-bounds test, summed.  This is coarse structure
      (H1), and it rests on a rendering fact that `bounds.py` checks rather than
      assumes: a widget's own top row and left column are never covered by a
      child, and no other pixel of that row or column carries its colour, so the
      count is the extent and no sequential run-length accumulator is needed.
  H6  the root widget is excluded: `corner` reads a left and an upper neighbour,
      so position (0,0) is not in the position set.  Reported as a limitation,
      not repaired by a special case.

No choice is initialised to, or restricted to, the correct answer.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BYTE, FLAT, IDX, Builder, Registry, accuracy, all_conforming, bytes_type,
                    colour_at, dump, episode, exact_error, random_reference, record_type,
                    report, sweep)
from tcn.graph import Candidate, Node, Program, Signal
from tcn.search import space_size
from tcn.types import BOOL, Value, integer, product, setof

KEY = integer(24, signed=False)          # a packed RGB triple: the widget's key
POOL = (30, 100, 170)                    # distractor byte constants for the S0 ablation


# --------------------------------------------------------------------------
# S0: same(a, b, obs)
# --------------------------------------------------------------------------
def same_scaffold(registry, width, height, free=False):
    """`(a, b, obs) -> bool`: do two addressed pixels carry the same colour?

    Free choices: the two Boolean combinators, and -- under `free` -- each
    channel comparison's second operand, so the operand binding is searched
    rather than supplied.
    """
    BT = bytes_type(width, height)
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    if free:
        consts += tuple((f"byte_{v}", Value.of(BYTE, v)) for v in POOL)
    b = Builder(registry, (("a", IDX), ("b", IDX), ("obs", BT)), consts)
    for tag in ("a", "b"):
        b.add(f"{tag}_g", "add", [tag, "one"])
        b.add(f"{tag}_b", "add", [tag, "two"])
        b.add(f"{tag}_r_v", "index", ["obs", tag])
        b.add(f"{tag}_g_v", "index", ["obs", f"{tag}_g"])
        b.add(f"{tag}_b_v", "index", ["obs", f"{tag}_b"])
    for channel in ("r", "g", "b"):
        neighbour = ("eq", (f"a_{channel}_v", f"b_{channel}_v"), None, None)
        if free:
            # The correct binding is deliberately not candidate 0.
            options = [("eq", (f"a_{channel}_v", f"byte_{POOL[0]}"), None, None),
                       ("eq", (f"a_{channel}_v", f"byte_{POOL[1]}"), None, None),
                       neighbour,
                       ("eq", (f"a_{channel}_v", f"byte_{POOL[2]}"), None, None)]
        else:
            options = [neighbour]
        b.choice(f"cmp_{channel}", options)
    b.choice("rg", [(f"truth_{t}", ("cmp_r", "cmp_g"), None, None) for t in range(16)])
    b.choice("same", [(f"truth_{t}", ("rg", "cmp_b"), None, None) for t in range(16)])
    return b.program((("y", "same"),))


def same_examples(seeds, split, per_image, seed=0, arbitrary=False, **configuration):
    """Four-neighbour pairs, labelled by the `owner` probe.

    MEASURED CORRECTION to the draft.  The draft drew *arbitrary* address pairs,
    on the argument that a same-colour predicate is a predicate about two
    arbitrary addresses and that denser supervision would break the two-spelling
    tie `research/gui-hierarchy` reported.  Run, that space returns **0
    conforming programs out of 256, exhausted** -- and the reason is the same one
    that broke S2's extent: fill colour is not injective per widget at
    `palette 32` (`bounds.json`, `colour_injective` = 1 of 12 episodes).  Over
    48,000 arbitrary pairs on 12 flat episodes, `owner(a) == owner(b)` disagrees
    with "same colour" on **1.41%** of them, so no colour predicate can reproduce
    the label and the target is not a function of the context at all.  Over
    46,528 four-neighbour pairs on the same episodes the disagreement is
    **0.00%**, which is the same fact `bounds.json`'s `corner_colour` states as
    226/0/0.  `arbitrary=True` reproduces the draft's unsatisfiable variant, and
    it is reported beside the neighbour arm rather than dropped.
    """
    import random
    rng = random.Random(seed)
    rows = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        raw = Value.of(BT, ep["pixels"]).raw
        owner = ep["probes"]["owner"]
        w, h = ep["width"], ep["height"]
        n = w * h
        drawn = 0
        while drawn < per_image:
            i = rng.randrange(n)
            if arbitrary:
                j = i + 1 if rng.random() < .5 and i + 1 < n else rng.randrange(n)
            else:
                x, y = i % w, i // w
                dx, dy = rng.choice(((1, 0), (-1, 0), (0, 1), (0, -1)))
                if not (0 <= x + dx < w and 0 <= y + dy < h):
                    continue
                j = (y + dy) * w + (x + dx)
            drawn += 1
            rows.append({"inputs": {"a": Value.of(IDX, 3 * i), "b": Value.of(IDX, 3 * j),
                                    "obs": Value(BT, raw)},
                         "targets": {"same": Value.of(BOOL, owner[i] == owner[j])}})
    rng.shuffle(rows)
    return rows


def same_signals():
    return (Signal("same", "same", ("core",), BOOL, "bce"),)


def induced_same(selections, free):
    """The Boolean function a selection denotes, so two spellings count once."""
    def truth(t, x, y):
        return bool((t >> (2 * int(x) + int(y))) & 1)
    table = tuple(truth(selections["same"], truth(selections["rg"], a, b), c)
                  for a in (False, True) for b in (False, True) for c in (False, True))
    return (table, tuple(selections.get(f"cmp_{c}", 0) for c in "rgb") if free else ())


# --------------------------------------------------------------------------
# S1: corner(rec)
# --------------------------------------------------------------------------
def offset_pool(width):
    """Five plausible spatial steps in bytes; 3 and 3W are the answers.

    Deliberately not at index 0: a zero-initialised choice logit makes candidate
    0 the argmax and `enumerate_fit`'s tie-break prefers it, so listing an answer
    first would hand it to a run that never moved.
    """
    return (6, 3 * width + 3, 3, 3 * width, 9)


def corner_scaffold(registry, width, height, module, offsets):
    """`rec -> bool`: two frozen `same` calls at searched offsets, combined."""
    BT = bytes_type(width, height)
    consts = tuple((f"off{k}", Value.of(IDX, k)) for k in offsets)
    b = Builder(registry, (("rec", record_type(width, height)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("back_a", [("sub", ("pos", f"off{k}"), None, None) for k in offsets])
    b.choice("back_b", [("sub", ("pos", f"off{k}"), None, None) for k in offsets])
    b.add("same_a", module, ["back_a", "pos", "obs"])
    b.add("same_b", module, ["back_b", "pos", "obs"])
    b.choice("corner", [(f"truth_{t}", ("same_a", "same_b"), None, None) for t in range(16)])
    return b.program((("y", "corner"),))


def interior_positions(ep):
    """Byte addresses of every pixel with a left and an upper neighbour (H6)."""
    w, h = ep["width"], ep["height"]
    return tuple(3 * (y * w + x) for y in range(1, h) for x in range(1, w))


def corner_examples(seeds, split, per_image=None, seed=0, **configuration):
    import random
    rng = random.Random(seed)
    rows = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(BT, ep["pixels"]).raw
        corners = {(d["rect"][0], d["rect"][1]) for d in ep["probes"]["hierarchy"]}
        positions = interior_positions(ep)
        chosen = positions if per_image is None else [
            positions[i] for i in sorted(rng.sample(range(len(positions)),
                                                    min(per_image, len(positions))))]
        for p in chosen:
            i = p // 3
            x, y = i % ep["width"], i // ep["width"]
            rows.append({"inputs": {"rec": Value(REC, (p, raw))},
                         "targets": {"corner": Value.of(BOOL, (x, y) in corners)}})
    rng.shuffle(rows)
    return rows


def corner_signals():
    return (Signal("corner", "corner", ("core",), BOOL, "bce"),)


# --------------------------------------------------------------------------
# S2: rect(rec)
# --------------------------------------------------------------------------
def rect_scaffold(registry, width, height, module, offsets, span=None):
    """`rec -> (x, y, w, h, key, parent_key)` at a widget's top-left corner.

    The extent is a *count*, not a run: `bounds.py` checks that a widget's own
    top row and left column are never covered by a child and that no other pixel
    of that row or column carries its colour, so summing `same(p, p + k*step)`
    over the whole span is the extent and needs no sequential accumulator.  The
    in-bounds mask is what keeps a row count from wrapping into the next row.
    """
    BT = bytes_type(width, height)
    span = span or max(width, height)
    last = 3 * width * height - 3
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)),
              ("three", Value.of(IDX, 3)), ("last", Value.of(IDX, last)),
              ("width", Value.of(IDX, width)), ("height", Value.of(IDX, height)))
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in offsets)
    consts += tuple((f"k{k}", Value.of(IDX, k)) for k in range(1, span))
    b = Builder(registry, (("rec", record_type(width, height)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("linear", "idiv", ["pos", "three"])
    b.add("x", "mod", ["linear", "width"])
    b.add("y", "idiv", ["linear", "width"])
    b.choice("step_w", [("identity", (f"off{k}",), None, None) for k in offsets])
    b.choice("step_h", [("identity", (f"off{k}",), None, None) for k in offsets])
    for tag, step, coordinate, limit in (("w", "step_w", "x", "width"),
                                         ("h", "step_h", "y", "height")):
        terms, run = [], None
        for k in range(1, span):
            b.add(f"{tag}d{k}", "mul", [step, f"k{k}"])
            b.add(f"{tag}a{k}", "add", ["pos", f"{tag}d{k}"])
            b.add(f"{tag}c{k}", "min", [f"{tag}a{k}", "last"])
            b.add(f"{tag}s{k}", module, [f"{tag}c{k}", "pos", "obs"])
            b.add(f"{tag}i{k}", "add", [coordinate, f"k{k}"])
            b.add(f"{tag}m{k}", "lt", [f"{tag}i{k}", limit])
            b.add(f"{tag}t{k}", "and", [f"{tag}s{k}", f"{tag}m{k}"])
            # A prefix conjunction, not a bare mask.  H5 as drafted summed the
            # masked same-colour bits over the whole row, on the claim that no
            # other pixel of that row carries the widget's colour.  That claim is
            # false and `bounds.py` already contained the refutation it did not
            # consult: `colour_injective` is 1 of 12 episodes on the flat screen
            # (192 distinct colours over 226 widgets), so a far-away sibling in
            # the same row is counted and the extent overshoots.  Conjoining each
            # term with every earlier one makes the sum the length of the
            # *contiguous* run, which is the rule `extent_rule` actually checked
            # at 226/226.  Still a fixed-depth feedforward graph; no accumulator
            # and no recurrence.
            run = (f"{tag}t{k}" if run is None
                   else b.add(f"{tag}r{k}", "and", [run, f"{tag}t{k}"]))
            terms.append(b.add(f"{tag}e{k}", "encode", [run], out=IDX))
        b.add(f"{tag}_tuple", "tuple", terms)
        b.add(f"{tag}_count", "sum", [f"{tag}_tuple"])
        b.add(f"{tag}_extent", "add", [f"{tag}_count", "one"])
    b.add("left", "sub", ["pos", "three"])
    for tag, base in (("own", "pos"), ("par", "left")):
        b.add(f"{tag}_g", "add", [base, "one"])
        b.add(f"{tag}_b", "add", [base, "two"])
        b.add(f"{tag}_rv", "index", ["obs", base])
        b.add(f"{tag}_gv", "index", ["obs", f"{tag}_g"])
        b.add(f"{tag}_bv", "index", ["obs", f"{tag}_b"])
        b.add(f"{tag}_tup", "tuple", [f"{tag}_rv", f"{tag}_gv", f"{tag}_bv"])
        b.add(f"{tag}_key", "pack", [f"{tag}_tup"], out=KEY)
    b.add("record", "tuple", ["x", "y", "w_extent", "h_extent", "own_key", "par_key"])
    return b.program((("y", "record"),))


RECT = product(IDX, IDX, IDX, IDX, KEY, KEY)


def rect_target(ep, widget):
    x, y, w, h = widget["rect"]
    own = colour_at(ep, x, y)
    parent = colour_at(ep, x - 1, y)
    return Value.of(RECT, (x, y, w, h, pack_rgb(own), pack_rgb(parent)))


def pack_rgb(colour):
    r, g, blue = colour
    return int(r) | (int(g) << 8) | (int(blue) << 16)


def rect_examples(seeds, split, **configuration):
    """One example per non-root widget, at its own corner."""
    rows = []
    for s in seeds:
        ep = episode(s, split, **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        REC = record_type(ep["width"], ep["height"])
        raw = Value.of(BT, ep["pixels"]).raw
        for d in ep["probes"]["hierarchy"]:
            x, y = d["rect"][0], d["rect"][1]
            if x < 1 or y < 1:
                continue
            rows.append({"inputs": {"rec": Value(REC, (3 * (y * ep["width"] + x), raw))},
                         "targets": {"rect": rect_target(ep, d)}})
    return rows


def rect_signals():
    return (Signal("record", "rect", ("core",), RECT, "mse"),)


# --------------------------------------------------------------------------
# S3: the assembly
# --------------------------------------------------------------------------
def assembly(registry, observation, positions, corner_module, rect_module, port="observation"):
    """Four caller nodes: hold, pair, filter, map.

    `tcn.scaffold.positional_scaffold` supplies three of these; the fourth is the
    `filter` that keeps only the positions the corner module accepts, so the
    expensive rectangle module runs at the corners rather than everywhere.  It is
    an ordinary registered operator over the same set type, not a new primitive.
    """
    positions = tuple(positions)
    holder = setof(observation, 1)
    locations = setof(IDX, len(set(positions)))
    hold = registry.resolve("insert", (holder, observation))
    records = registry.resolve("pair", (locations, hold.output))
    kept = registry.resolve("filter", (records.output,), None, {"module": corner_module})
    mapped = registry.resolve("map", (kept.output,), None, {"module": rect_module})
    nodes = (Node("held", hold.output, (Candidate(hold, ("empty", port)),), "core", 1, 0),
             Node("records", records.output, (Candidate(records, ("positions", "held")),),
                  "core", 2, 0),
             Node("kept", kept.output, (Candidate(kept, ("records",)),), "core", 3, 0),
             Node("mapped", mapped.output, (Candidate(mapped, ("kept",)),), "core", 4, 0))
    constants = (("positions", Value.of(locations, positions)),
                 ("empty", Value.of(holder, ())))
    return Program(((port, observation),), nodes, (("mapped", "mapped"),), constants,
                   input_depths=((port, 0),)).validate(registry)


def score_tree(predicted, ep):
    """Compare the parsed relation with the `hierarchy` probe.

    `predicted` is `set[(x, y, w, h, key, parent_key)]`.  The keys are the
    program's own colour identifiers, not the generator's ids, so the comparison
    is on **rectangles**: a widget is recovered when its rectangle appears, and
    its parent link is recovered when the rectangle its `parent_key` resolves to
    is its true parent's rectangle.  This is the object-identity result restated:
    a permutation of the widget list leaves the raster identical, so no program
    can produce the generator's numbering, and the tree is recovered up to that
    relabelling.  The root is supplied only as "the widget whose rectangle is the
    whole screen", which the observation's own type already declares (H6).
    """
    truth = {d["id"]: d for d in ep["probes"]["hierarchy"]}
    rects = {d["id"]: tuple(d["rect"]) for d in truth.values()}
    non_root = [d for d in truth.values() if d["parent"] != 255]
    by_key = {row[4]: row for row in predicted}
    got_rects = {(row[0], row[1], row[2], row[3]) for row in predicted}
    want_rects = {rects[d["id"]] for d in non_root}
    root_key = pack_rgb(colour_at(ep, 0, 0))
    links_right = links_wrong = 0
    for row in predicted:
        parent_key = row[5]
        parent_rect = ((0, 0, ep["width"], ep["height"]) if parent_key == root_key
                       else tuple(by_key[parent_key][:4]) if parent_key in by_key else None)
        match = [d for d in truth.values() if tuple(d["rect"]) == tuple(row[:4])]
        if len(match) == 1 and parent_rect is not None and \
                parent_rect == tuple(rects[match[0]["parent"]]):
            links_right += 1
        else:
            links_wrong += 1
    keys = [row[4] for row in predicted]
    colliding = len(keys) - len(set(keys))
    return {"key_collisions": colliding,
            "widgets_in_probe": len(truth), "non_root": len(non_root),
            "rects_predicted": len(got_rects), "rects_true": len(want_rects),
            "rects_exact": got_rects == want_rects,
            "rects_missing": sorted(want_rects - got_rects),
            "rects_spurious": sorted(got_rects - want_rects),
            "parent_links_correct": links_right, "parent_links_wrong": links_wrong,
            "tree_exact": got_rects == want_rects and links_wrong == 0}


# --------------------------------------------------------------------------
def stage(name, program, train, validation, held, signals, registry, tolerance=1e-6,
          induced=None, draws=400):
    """One searched stage: exhaust, count conforming and distinct functions, validate."""
    out = {"name": name, "space_size": space_size(program), "nodes": len(program.nodes),
           "train_records": len(train), "validation_records": len(validation),
           "held_records": len(held)}
    out["search"] = sweep(program, train, signals, registry, tolerance)
    survey = all_conforming(program, train, signals, registry, tolerance)
    conforming = survey.pop("conforming")
    out["conforming"] = survey
    if induced is not None and conforming:
        out["conforming"]["distinct_functions"] = len({induced(s) for s in conforming})
    if conforming:
        first = conforming[0]
        out["lexicographic_pick"] = {
            "selections": first,
            "validation_max_error": exact_error(program, validation, signals, registry, first),
            "held_max_error": exact_error(program, held, signals, registry, first),
            "held_accuracy": accuracy(program, held, signals, registry, first)}
        survivors = [s for s in conforming
                     if exact_error(program, validation, signals, registry, s) <= tolerance]
        out["validation_filtered"] = {
            "survivors": len(survivors), "unique": len(survivors) == 1,
            "held_max_error": [exact_error(program, held, signals, registry, s)
                               for s in survivors[:8]],
            "held_accuracy": [accuracy(program, held, signals, registry, s)
                              for s in survivors[:8]],
            "selections": survivors[:8]}
        if induced is not None:
            out["validation_filtered"]["distinct_functions"] = len({induced(s) for s in survivors})
        out["chosen"] = survivors[0] if survivors else first
    out["random_control"] = random_reference(program, train, signals, registry, draws=draws,
                                             tolerance=tolerance)
    report(f"[{name}] space / evaluated / conforming / certificate",
           f"{out['space_size']} / {out['search']['evaluated']} / "
           f"{out['search']['conforming']} / {out['search']['certificate']} "
           f"({out['search']['seconds']:.1f}s)")
    if conforming:
        report(f"[{name}] distinct functions / survivors / held-out max error",
               f"{out['conforming'].get('distinct_functions')} / "
               f"{out['validation_filtered']['survivors']} / "
               f"{out['validation_filtered']['held_max_error'][:4]}")
    report(f"[{name}] random-draw density", f"{out['random_control']['density']:.4f}")
    return out, conforming


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--validation", type=int, default=3)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--pairs", type=int, default=48)
    ap.add_argument("--corner-per-image", type=int, default=120)
    ap.add_argument("--parse", type=int, default=3)
    ap.add_argument("--free", action="store_true", help="run the S0 operand ablation too")
    ap.add_argument("--tag", default="rung3")
    args = ap.parse_args()

    registry = Registry()
    tolerance = 1e-6
    configuration = dict(FLAT)
    probe = episode(0, "train", **configuration)
    W, H = probe["width"], probe["height"]
    achieved = [len(episode(s, "train", **configuration)["probes"]["hierarchy"])
                for s in range(args.train)]
    result = {"arguments": vars(args), "configuration": configuration,
              "achieved_widgets_mean": sum(achieved) / len(achieved),
              "observation_bytes": len(probe["pixels"]),
              "interior_positions": len(interior_positions(probe)),
              "offset_pool": list(offset_pool(W))}

    # ---- S0 -------------------------------------------------------------
    print("\n--- S0: same(a, b, obs), the offset externalised ---")
    tr = same_examples(range(args.train), "train", args.pairs, seed=1, **configuration)
    va = same_examples(range(50, 50 + args.validation), "validation", args.pairs, seed=2,
                       **configuration)
    he = same_examples(range(100, 100 + args.held), "test", args.pairs, seed=3, **configuration)
    result["s0_positive_fraction"] = sum(e["targets"]["same"].decoded for e in tr) / len(tr)
    program = same_scaffold(registry, W, H)
    result["s0"], conforming = stage("S0 same", program, tr, va, he, same_signals(), registry,
                                     tolerance, induced=lambda s: induced_same(s, False))
    if "chosen" not in result["s0"]:
        raise SystemExit("S0 has no conforming program; the stages above it cannot be staged.")
    same_module = registry.register_module(program.harden(result["s0"]["chosen"]))

    print("\n--- S0 negative control: the draft's arbitrary-pair supervision ---")
    atr = same_examples(range(args.train), "train", args.pairs, seed=1, arbitrary=True,
                        **configuration)
    result["s0_arbitrary_positive_fraction"] = (
        sum(e["targets"]["same"].decoded for e in atr) / len(atr))
    result["s0_arbitrary"], _ = stage("S0 arbitrary", same_scaffold(registry, W, H), atr, atr,
                                      atr, same_signals(), registry, tolerance,
                                      induced=lambda s: induced_same(s, False))
    result["s0"]["module"] = same_module
    dump(args.tag, result)

    if args.free:
        print("\n--- ablation H2: S0's operand binding is searched, not supplied ---")
        free_program = same_scaffold(registry, W, H, free=True)
        result["s0_free"], _ = stage("S0 free", free_program, tr, va, he, same_signals(),
                                     registry, tolerance, induced=lambda s: induced_same(s, True))
        dump(args.tag, result)

    # ---- S1 -------------------------------------------------------------
    print("\n--- S1: corner, from two frozen S0 calls at searched offsets ---")
    offsets = offset_pool(W)
    ctr = corner_examples(range(args.train), "train", args.corner_per_image, seed=4,
                          **configuration)
    cva = corner_examples(range(50, 50 + args.validation), "validation", args.corner_per_image,
                          seed=5, **configuration)
    che = corner_examples(range(100, 100 + args.held), "test", args.corner_per_image, seed=6,
                          **configuration)
    result["s1_positive_fraction"] = sum(e["targets"]["corner"].decoded for e in ctr) / len(ctr)
    corner_program = corner_scaffold(registry, W, H, same_module, offsets)
    result["s1"], _ = stage("S1 corner", corner_program, ctr, cva, che, corner_signals(),
                            registry, tolerance)
    corner_module = registry.register_module(corner_program.harden(result["s1"]["chosen"]))
    result["s1"]["module"] = corner_module
    result["s1"]["offsets_chosen"] = [offsets[result["s1"]["chosen"]["back_a"]],
                                      offsets[result["s1"]["chosen"]["back_b"]]]
    report("S1 offsets chosen (bytes)", result["s1"]["offsets_chosen"])
    dump(args.tag, result)

    # ---- S2 -------------------------------------------------------------
    print("\n--- S2: the rectangle, by counting rather than by a run ---")
    rtr = rect_examples(range(args.train), "train", **configuration)
    rva = rect_examples(range(50, 50 + args.validation), "validation", **configuration)
    rhe = rect_examples(range(100, 100 + args.held), "test", **configuration)
    rect_program = rect_scaffold(registry, W, H, same_module, offsets)
    result["s2"], _ = stage("S2 rect", rect_program, rtr, rva, rhe, rect_signals(), registry,
                            tolerance, draws=50)
    rect_module = registry.register_module(rect_program.harden(result["s2"]["chosen"]))
    result["s2"]["module"] = rect_module
    result["s2"]["steps_chosen"] = [offsets[result["s2"]["chosen"]["step_w"]],
                                    offsets[result["s2"]["chosen"]["step_h"]]]
    report("S2 steps chosen (bytes)", result["s2"]["steps_chosen"])
    dump(args.tag, result)

    # ---- S3 -------------------------------------------------------------
    print("\n--- S3: the whole screen parsed, four caller nodes ---")
    rows = []
    for seed in range(200, 200 + args.parse):
        ep = episode(seed, "test", **configuration)
        BT = bytes_type(ep["width"], ep["height"])
        positions = interior_positions(ep)
        parser = assembly(registry, BT, positions, corner_module, rect_module)
        started = time.perf_counter()
        got = parser.run({"observation": Value.of(BT, ep["pixels"])}, registry=registry)[0]
        elapsed = time.perf_counter() - started
        predicted = sorted(got["mapped"].decoded)
        row = score_tree(predicted, ep) | {"seed": seed, "seconds": elapsed,
                                           "positions": len(positions),
                                           "caller_nodes": len(parser.nodes)}
        rows.append(row)
        report(f"parse seed {seed}: rects {row['rects_predicted']}/{row['rects_true']} exact="
               f"{row['rects_exact']} links {row['parent_links_correct']}/"
               f"{row['parent_links_correct'] + row['parent_links_wrong']} tree_exact="
               f"{row['tree_exact']}", f"{elapsed:.1f}s")
    result["parse"] = {"episodes": rows,
                       "all_trees_exact": all(r["tree_exact"] for r in rows),
                       "caller_nodes": rows[0]["caller_nodes"] if rows else None,
                       "corner_module_cost": registry.modules[corner_module].execution_cost(registry),
                       "rect_module_cost": registry.modules[rect_module].execution_cost(registry),
                       "same_module_cost": registry.modules[same_module].execution_cost(registry)}
    dump(args.tag, result)


if __name__ == "__main__":
    main()
