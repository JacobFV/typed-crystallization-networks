"""Rung 4 -- object identity as a two-position SAME-OBJECT relation, learned.

`bounds.py` certifies that the integer label `object_ids` is not a function of
the observation at any context size.  What survives that certificate is the
*partition* the labels induce, and its two-position restriction is

    same(i, j)  =  object_ids[i] == object_ids[j]

which is invariant to exactly the two things the labels are not: the order of
the generator's object list and the colours it re-draws every episode.

`chroma_bound.py` measures the ceiling of the family the algebra can express
for it.  The renderer paints `clip(base_colour * shade)`, so two pixels of one
object are collinear with the origin whatever their facing, and collinearity is
the integer cross product -- reachable once `pack` strips `role="byte"` from a
pixel and `encode` widens it.  The predicate is

    collinear_T(a, b) = |r1*g2 - r2*g1| <= T and |g1*b2 - g2*b1| <= T
                        and |r1*b2 - r2*b1| <= T

with `T` a searched constant and the two combinators searched, exactly as at
rung 3.  The neighbour offset is searched too, as at rung 3.5.

Three arms:
  direct  -- the collinearity module alone;
  staged  -- the rung-3 foreground module frozen, then combined with it;
  flat    -- the same target with nothing frozen and the byte pool free.
"""
from __future__ import annotations

import argparse, json, pathlib, random, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import (BACKGROUND, BYTE, IDX, Builder, accuracy, all_conforming, bytes_type,
                    enumerate_reference, episode, exact_error, gradient_arm, pixel_starts,
                    positional_caller, random_reference, record_type, report, summarise)
from rung3_mask import POOL, module_scaffold, pixel_examples, signals as fg_signals
from common2 import OUT
from tcn.graph import Signal
from tcn.operators import Registry
from tcn.search import evaluate, space_size
from tcn.types import BOOL, Value, integer, product, setof

U8 = integer(8, signed=False)
I32 = integer(32, signed=True)
# Threshold candidates.  192 is the value `chroma_bound.py` measures to separate
# the classes exactly; the rest are distractors spanning three orders of
# magnitude, so the choice is a real search.
THRESHOLDS = (0, 16, 48, 96, 192, 384, 1024, 4096)


def dump(name, obj):
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    print("wrote", p, flush=True)
    return p


def offsets(resolution):
    """right, two right, one row down -- in bytes."""
    return (3, 6, 3 * resolution)


def window_positions(resolution):
    n = resolution * resolution
    return tuple(3 * i for i in range(n - resolution))


def same_labels(probes, resolution, step=1):
    ids = probes["object_ids"]
    n = resolution * resolution
    return {3 * i: (ids[i] == ids[i + step]) for i in range(n - resolution)}


def same_examples(seeds, resolution, split="train", per_image=None, seed=0, objects=6):
    out = []
    REC = record_type(resolution)
    rng = random.Random(seed)
    positions = window_positions(resolution)
    for s in seeds:
        pixels, probes = episode(s, resolution, split=split, objects=objects)
        raw = Value.of(bytes_type(resolution), pixels).raw
        labels = same_labels(probes, resolution)
        chosen = positions if per_image is None else [
            positions[i] for i in sorted(rng.sample(range(len(positions)),
                                                    min(per_image, len(positions))))]
        for start in chosen:
            out.append({"inputs": {"rec": Value(REC, (start, raw))},
                        "targets": {"same": Value.of(BOOL, labels[start])}})
    rng.shuffle(out)
    return out


def same_signals():
    return (Signal("same", "same", ("core",), BOOL, "bce"),)


def _channels(b, tag, base):
    """Read a pixel's three bytes and lift them out of `role="byte"` into arithmetic.

    `pack` on a one-field tuple is the algebra's declared conversion from a
    byte carrier to a plain integer; `encode` then widens it so a product of
    two channels cannot overflow.  Both are registered universal operators and
    neither is domain specific.
    """
    b.add(f"{tag}_g_at", "add", [base, "one"])
    b.add(f"{tag}_b_at", "add", [base, "two"])
    for ch, addr in (("r", base), ("g", f"{tag}_g_at"), ("bl", f"{tag}_b_at")):
        b.add(f"{tag}_{ch}_byte", "index", ["obs", addr])
        b.add(f"{tag}_{ch}_tup", "tuple", [f"{tag}_{ch}_byte"])
        b.add(f"{tag}_{ch}_u8", "pack", [f"{tag}_{ch}_tup"], out=U8)
        b.add(f"{tag}_{ch}", "encode", [f"{tag}_{ch}_u8"], out=I32)


def _cross(b, name, x1, y2, x2, y1):
    b.add(f"{name}_p", "mul", [x1, y2])
    b.add(f"{name}_q", "mul", [x2, y1])
    b.add(f"{name}_d", "sub", [f"{name}_p", f"{name}_q"])
    b.add(f"{name}_a", "abs", [f"{name}_d"])
    b.add(f"{name}_le", "le", [f"{name}_a", "thr"])


def collinear_scaffold(registry, resolution, thresholds=THRESHOLDS, truths=range(16),
                       offs=None):
    """The direct arm.  |space| = |offsets| * |thresholds| * 16 * 16."""
    offs = offs or offsets(resolution)
    REC = record_type(resolution)
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in offs)
    consts += tuple((f"thr_{t}", Value.of(I32, t)) for t in thresholds)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("shifted", [("add", ("pos", f"off{k}"), None) for k in offs])
    # one threshold, chosen once and shared by all three comparisons
    b.choice("thr", [("identity", (f"thr_{t}",), None) for t in thresholds])
    _channels(b, "a", "pos")
    _channels(b, "b", "shifted")
    _cross(b, "rg", "a_r", "b_g", "b_r", "a_g")
    _cross(b, "gb", "a_g", "b_bl", "b_g", "a_bl")
    _cross(b, "rb", "a_r", "b_bl", "b_r", "a_bl")
    b.choice("m1", [(f"truth_{t}", ("rg_le", "gb_le"), None) for t in truths])
    b.choice("same", [(f"truth_{t}", ("m1", "rb_le"), None) for t in truths])
    b.add("record", "tuple", ["pos", "same"])
    return b.program((("y", "record"),))


def staged_scaffold(registry, resolution, module_name, thresholds=THRESHOLDS,
                    truths=range(16), offs=None):
    """The staged arm: the frozen rung-3 foreground module called at both positions,
    combined with the searched collinearity predicate."""
    offs = offs or offsets(resolution)
    REC = record_type(resolution)
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in offs)
    consts += tuple((f"thr_{t}", Value.of(I32, t)) for t in thresholds)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("shifted", [("add", ("pos", f"off{k}"), None) for k in offs])
    b.choice("thr", [("identity", (f"thr_{t}",), None) for t in thresholds])
    b.add("here_rec", "tuple", ["pos", "obs"])
    b.add("here", module_name, ["here_rec"])
    b.add("here_flag", "project", ["here"], params={"index": 1})
    b.add("there_rec", "tuple", ["shifted", "obs"])
    b.add("there", module_name, ["there_rec"])
    b.add("there_flag", "project", ["there"], params={"index": 1})
    b.choice("fg_agree", [(f"truth_{t}", ("here_flag", "there_flag"), None) for t in truths])
    _channels(b, "a", "pos")
    _channels(b, "b", "shifted")
    _cross(b, "rg", "a_r", "b_g", "b_r", "a_g")
    _cross(b, "gb", "a_g", "b_bl", "b_g", "a_bl")
    b.choice("coll", [(f"truth_{t}", ("rg_le", "gb_le"), None) for t in truths])
    b.choice("same", [(f"truth_{t}", ("fg_agree", "coll"), None) for t in truths])
    b.add("record", "tuple", ["pos", "same"])
    return b.program((("y", "record"),))


def flat_scaffold(registry, resolution, thresholds=THRESHOLDS, pool=POOL, truths=range(16)):
    """Nothing frozen: the foreground test's byte constants are free too."""
    offs = offsets(resolution)
    REC = record_type(resolution)
    consts = (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    consts += tuple((f"off{k}", Value.of(IDX, k)) for k in offs)
    consts += tuple((f"thr_{t}", Value.of(I32, t)) for t in thresholds)
    consts += tuple((f"byte_{v}", Value.of(BYTE, v)) for v in pool)
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.choice("shifted", [("add", ("pos", f"off{k}"), None) for k in offs])
    b.choice("thr", [("identity", (f"thr_{t}",), None) for t in thresholds])
    _channels(b, "a", "pos")
    _channels(b, "b", "shifted")
    for tag in ("a", "b"):
        for ch in ("r", "g", "bl"):
            b.choice(f"{tag}_cmp_{ch}", [("eq", (f"{tag}_{ch}_byte", f"byte_{v}"), None)
                                         for v in pool])
        b.choice(f"{tag}_rg", [(f"truth_{t}", (f"{tag}_cmp_r", f"{tag}_cmp_g"), None)
                               for t in truths])
        b.choice(f"{tag}_fg", [(f"truth_{t}", (f"{tag}_rg", f"{tag}_cmp_bl"), None)
                               for t in truths])
    b.choice("fg_agree", [(f"truth_{t}", ("a_fg", "b_fg"), None) for t in truths])
    _cross(b, "rg", "a_r", "b_g", "b_r", "a_g")
    _cross(b, "gb", "a_g", "b_bl", "b_g", "a_bl")
    b.choice("coll", [(f"truth_{t}", ("rg_le", "gb_le"), None) for t in truths])
    b.choice("same", [(f"truth_{t}", ("fg_agree", "coll"), None) for t in truths])
    b.add("record", "tuple", ["pos", "same"])
    return b.program((("y", "record"),))


def select(program, conforming, validation, signals, registry, tolerance=1e-6):
    """The selection rule the previous track had to introduce.

    `enumerate_fit` returns the lexicographically first conforming program, and
    at rung 3 that program was wrong on fresh episodes.  What fixed it was
    requiring exactness at EVERY position of a validation split, so that is the
    rule here, and both counts are reported.
    """
    survivors = [s for s in conforming
                 if exact_error(program, validation, signals, registry, selections=s) <= tolerance]
    return survivors


def choice_gradients(build, examples, signals, registry, init_noise=.5, seed=0):
    """Which choice logits are even reachable by autograd, and how strongly.

    Rung 3.5 found the offset node with `grad = None` because it feeds a frozen
    module.  Here the whole arithmetic path sits behind `pack`, which declares
    `gradient="none"`, so this is the measurement that says which of the two
    backends can settle which choice.
    """
    import torch
    from tcn.learning import SoftProgram, tensor as _tensor
    program = build()
    model = SoftProgram(program, registry)
    g = torch.Generator().manual_seed(seed + 12345)
    with torch.no_grad():
        for p_ in model.choices:
            p_.add_(torch.randn(p_.shape, generator=g) * init_noise)
    inputs = {k: torch.stack([_tensor(ex["inputs"][k]) for ex in examples]) for k, _ in program.inputs}
    targets = {s.target: torch.stack([_tensor(ex["targets"][s.target]) for ex in examples])
               for s in signals}
    _, _, tr = model(inputs, return_trace=True)
    loss = model.probe_loss(tr, targets, signals)
    loss.backward()
    # `SoftProgram.choices` has ONE entry per node, free or not -- zipping it
    # against only the free nodes silently misaligns the report.
    out = {}
    for n, p_ in zip(program.nodes, model.choices):
        if n.selected is not None:
            continue
        out[n.name] = None if p_.grad is None else float(p_.grad.abs().sum())
    return {"loss": float(loss.detach()), "choice_grad_l1": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=8)
    ap.add_argument("--validation", type=int, default=8)
    ap.add_argument("--held", type=int, default=8)
    ap.add_argument("--per-image", type=int, default=0)     # 0 = every position
    ap.add_argument("--gradient-seeds", type=int, default=6)
    ap.add_argument("--objects", type=int, default=6)
    ap.add_argument("--arms", default="direct,staged")
    ap.add_argument("--flat-budget", type=int, default=60000)
    ap.add_argument("--apply", default="8,16,24")
    ap.add_argument("--tag", default="rung4_segment")
    args = ap.parse_args()

    R = args.resolution
    per = args.per_image or None
    tol = 1e-6
    r = Registry()
    train_seeds = tuple(range(args.train))
    val_seeds = tuple(range(50, 50 + args.validation))
    held_seeds = tuple(range(100, 100 + args.held))
    arms = set(args.arms.split(","))
    result = {"arguments": vars(args), "resolution": R, "offsets": list(offsets(R)),
              "thresholds": list(THRESHOLDS),
              "window_positions": len(window_positions(R))}

    train = same_examples(train_seeds, R, "train", per, seed=3, objects=args.objects)
    val = same_examples(val_seeds, R, "train", None, seed=5, objects=args.objects)
    held = same_examples(held_seeds, R, "test", None, seed=4, objects=args.objects)
    sig = same_signals()
    result["train_examples"] = len(train)
    result["validation_examples"] = len(val)
    result["held_examples"] = len(held)
    result["positive_fraction"] = sum(ex["targets"]["same"].decoded for ex in train) / len(train)
    report("same-object records (positive fraction)",
           f"{len(train)} train / {len(val)} val / {len(held)} held "
           f"({result['positive_fraction']:.1%} positive)")

    def run_arm(tag, build, gradient=True, lr=.15, steps=400):
        program = build()
        n = space_size(program)
        report(f"[{tag}] space", n)
        enum = enumerate_reference(program, train, sig, r, tolerance=tol)
        report(f"[{tag}] enumerate_fit solved/exhausted/unique/s",
               f"{enum['solved']}/{enum['exhausted']}/{enum['unique']}/{enum['seconds']:.1f}")
        ident = all_conforming(program, train, sig, r, tolerance=tol)
        survivors = select(program, ident["conforming"], val, sig, r, tolerance=tol)
        report(f"[{tag}] conforming on train / also exact at every validation position",
               f"{ident['count']} / {len(survivors)} of {n}")
        row = {"space_size": n, "enumerate_fit": enum,
               "conforming_on_train": ident["count"],
               "conforming_and_validation_exact": len(survivors),
               "unique_on_train": ident["unique"],
               "unique_after_validation": len(survivors) == 1,
               "enumeration_seconds": ident["seconds"]}
        for label, sel in (("lexicographic", ident["conforming"][0] if ident["conforming"] else None),
                           ("validation_filtered", survivors[0] if survivors else None)):
            if sel is None:
                row[label] = None
                continue
            row[label] = {"selections": sel,
                          "held_max_error": exact_error(program, held, sig, r, selections=sel),
                          "held_accuracy": accuracy(program, held, sig, r, selections=sel)}
            report(f"[{tag}] {label}: held-out max error / accuracy",
                   f"{row[label]['held_max_error']} / {row[label]['held_accuracy']:.6f}")
        row["random_density"] = random_reference(program, train, sig, r, 400, seed=1, tolerance=tol)
        try:
            row["choice_gradients"] = choice_gradients(build, train[:64], sig, r)
            report(f"[{tag}] choice logit |grad|_1",
                   json.dumps(row["choice_gradients"]["choice_grad_l1"]))
        except Exception as exc:
            row["choice_gradients"] = {"error": repr(exc)[:200]}
            report(f"[{tag}] choice gradient probe failed", repr(exc)[:120])
        if gradient:
            rows = gradient_arm(build, tuple(range(args.gradient_seeds)), train, sig, r,
                                steps=steps, lr=lr, init_noise=.5, label=tag,
                                tolerance=tol, held=held)
            row["gradient"] = {"rows": rows, "summary": summarise(rows), "init_noise": .5,
                               "held_exact": sum(1 for x in rows if x.get("held_err") == 0.)}
            report(f"[{tag}] gradient successes / held-out exact",
                   f"{row['gradient']['summary']['successes']}/{len(rows)} / "
                   f"{row['gradient']['held_exact']}")
        return program, row

    if "direct" in arms:
        prog, row = run_arm("direct", lambda: collinear_scaffold(r, R))
        result["direct"] = row
        dump(args.tag, result)
        # positional application is in `apply.py`: a module hardened at one
        # width cannot be registered against another, because its input type
        # names the observation.  The scaffold is rebuilt per width there and
        # the SELECTIONS are reused, which is what makes the space
        # resolution-independent in the first place.

    if "staged" in arms:
        fg_train = pixel_examples(train_seeds, R, "train", None, seed=1, objects=args.objects)
        fg_held = pixel_examples(held_seeds, R, "test", None, seed=2, objects=args.objects)
        fg_scaffold = module_scaffold(r, R, POOL)
        fg_sig = fg_signals()
        from incremental import incremental_conforming
        fg_enum = incremental_conforming(fg_scaffold, fg_train, fg_sig, r, tolerance=tol)
        fg_sel = fg_enum["conforming"][0] if fg_enum["conforming"] else None
        fg_enum.pop("conforming", None)
        fg_enum["accuracy_all_positions_train"] = accuracy(fg_scaffold, fg_train, fg_sig, r,
                                                           selections=fg_sel)
        fg_enum["accuracy_all_positions_held_out"] = accuracy(fg_scaffold, fg_held, fg_sig, r,
                                                              selections=fg_sel)
        result["stage1_foreground"] = fg_enum
        report("stage 1 conforming / accuracy at every position train / held",
               f"{fg_enum['count']} / {fg_enum['accuracy_all_positions_train']:.6f} / "
               f"{fg_enum['accuracy_all_positions_held_out']:.6f}")
        name = r.register_module(fg_scaffold.harden(fg_sel))
        _, row = run_arm("staged", lambda: staged_scaffold(r, R, name))
        result["staged"] = row
        dump(args.tag, result)

    if "flat" in arms:
        flat = flat_scaffold(r, R)
        n = space_size(flat)
        report("[flat] space", n)
        t0 = time.perf_counter()
        partial = enumerate_reference(flat, train, sig, r, tolerance=tol,
                                      max_programs=args.flat_budget)
        rate = partial["evaluated"] / max(1e-9, partial["wall"])
        result["flat"] = {"space_size": n, "enumerate_partial": partial,
                          "programs_per_second": rate,
                          "projected_exhaustive_seconds": n / max(1e-9, rate)}
        report("[flat] evaluated/solved/rate/projected exhaustive s",
               f"{partial['evaluated']}/{partial['solved']}/{rate:.0f}/"
               f"{result['flat']['projected_exhaustive_seconds']:.3g}")
        rows = gradient_arm(lambda: flat_scaffold(r, R), tuple(range(min(3, args.gradient_seeds))),
                            train, sig, r, steps=400, lr=.15, init_noise=.5, label="flat",
                            tolerance=tol, held=held)
        result["flat"]["gradient"] = {"rows": rows, "summary": summarise(rows)}
        dump(args.tag, result)

    dump(args.tag, result)


if __name__ == "__main__":
    main()
