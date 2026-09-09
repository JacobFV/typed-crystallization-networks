"""A Euclidean spatial operator over raw pixels: the convolution the tree could not express.

Three arms over one target family, all on real `geometry` pixels read as
`role="byte"` observations:

* **A. continuous kernel** -- a 3x3 Sobel-x over the green channel with nine
  trainable float weights, learned by gradient descent straight from the bytes.
  This is a weighted sum over a computed neighbourhood: `add(pos, off)` builds
  the stencil, `index` gathers, `interpret` commits the octet to a magnitude,
  `decode` widens it, `mul` scales it and `sum` adds it up.
* **B. discrete kernel** -- the two-tap horizontal derivative with the tap
  *addresses* and the tap *weights* both discrete choices, so the same target
  can be enumerated exhaustively and certified.
* **B'. the `pack` control** -- arm B with the byte leaving `role="byte"`
  through `pack` on a one-field tuple, which is how `research/object-identity`
  crossed the boundary before this branch.  `pack` is `gradient="none"`, so the
  address choice sits behind a hard boundary; this measures the difference.

Then the crystallized kernel is applied at every position with the three-node
`insert`/`pair`/`map` caller, which needs no new operator.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from common import (DERIVATIVE_X, FMAG, FPLAIN, IDX, MAG8, PLAIN8, SOBEL_X, STENCIL,
                    Builder, all_conforming, block_positions, bytes_type, convolve, dump,
                    enumerate_reference,
                    episode, exact_error, exact_mse, examples, local_fit, record_type, report,
                    signals, tap_offset, target_stats)
from tcn.operators import Registry
from tcn.scaffold import positional_scaffold
from tcn.search import space_size
from tcn.types import Value, product, setof


# --------------------------------------------------------------------------
# arm A: a continuous 3x3 kernel over raw bytes
# --------------------------------------------------------------------------
def kernel_scaffold(registry, resolution, taps=STENCIL, channel=1, init=0.):
    """`sum_k w_k * decode(interpret(index(obs, pos + off_k)))`.

    Every `w_k` is a trainable float32 magnitude.  The stencil addresses are
    computed, not chosen, exactly as `research/positional-reuse` requires.
    """
    consts = tuple((f"off_{dy}{dx}", Value.of(IDX, tap_offset(resolution, dy, dx, channel)))
                   for dy, dx in taps)
    consts += tuple((f"w_{dy}{dx}", Value.of(FMAG, init)) for dy, dx in taps)
    b = Builder(registry, (("rec", record_type(resolution)),), consts,
                trainable=tuple(f"w_{dy}{dx}" for dy, dx in taps))
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    terms = []
    for dy, dx in taps:
        t = f"{dy}{dx}"
        b.add(f"addr_{t}", "add", ["pos", f"off_{dy}{dx}"])
        b.add(f"byte_{t}", "index", ["obs", f"addr_{t}"])
        b.add(f"mag_{t}", "interpret", [f"byte_{t}"], out=MAG8)
        b.add(f"x_{t}", "decode", [f"mag_{t}"], out=FMAG)
        terms.append(b.add(f"term_{t}", "mul", [f"x_{t}", f"w_{dy}{dx}"]))
    b.add("packed", "tuple", terms)
    b.add("conv", "sum", ["packed"])
    b.add("record", "tuple", ["pos", "conv"])
    return b.program((("y", "record"),))


def fixed_kernel_scaffold(registry, resolution, weights, taps=STENCIL, channel=1):
    """The same graph with the weights baked as ordinary constants."""
    consts = tuple((f"off_{dy}{dx}", Value.of(IDX, tap_offset(resolution, dy, dx, channel)))
                   for dy, dx in taps)
    consts += tuple((f"w_{dy}{dx}", Value.of(FMAG, float(weights[(dy, dx)]))) for dy, dx in taps)
    b = Builder(registry, (("rec", record_type(resolution)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    terms = []
    for dy, dx in taps:
        t = f"{dy}{dx}"
        b.add(f"addr_{t}", "add", ["pos", f"off_{dy}{dx}"])
        b.add(f"byte_{t}", "index", ["obs", f"addr_{t}"])
        b.add(f"mag_{t}", "interpret", [f"byte_{t}"], out=MAG8)
        b.add(f"x_{t}", "decode", [f"mag_{t}"], out=FMAG)
        terms.append(b.add(f"term_{t}", "mul", [f"x_{t}", f"w_{dy}{dx}"]))
    b.add("packed", "tuple", terms)
    b.add("conv", "sum", ["packed"])
    b.add("record", "tuple", ["pos", "conv"])
    return b.program((("y", "record"),))


# --------------------------------------------------------------------------
# arm B / B': a discrete two-tap kernel, two ways out of `role="byte"`
# --------------------------------------------------------------------------
WEIGHT_POOL = (-1., 0., 1.)


def discrete_scaffold(registry, resolution, exit="interpret", channel=1, pool=WEIGHT_POOL):
    """Two taps; each picks its address from the 3x3 stencil and its weight from `pool`.

    `exit="interpret"` commits the octet to a magnitude (`gradient="exact"`).
    `exit="pack"` wraps it in a one-field tuple and packs it to a plain integer,
    which is the pre-existing loophole and is `gradient="none"`.
    """
    out_scalar = FMAG if exit == "interpret" else FPLAIN
    consts = tuple((f"off_{dy}{dx}", Value.of(IDX, tap_offset(resolution, dy, dx, channel)))
                   for dy, dx in STENCIL)
    consts += tuple((f"wv{i}", Value.of(out_scalar, w)) for i, w in enumerate(pool))
    b = Builder(registry, (("rec", record_type(resolution)),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    terms = []
    for tag in ("a", "b"):
        b.choice(f"addr_{tag}", [("add", ("pos", f"off_{dy}{dx}"), None, None)
                                 for dy, dx in STENCIL])
        b.add(f"byte_{tag}", "index", ["obs", f"addr_{tag}"])
        if exit == "interpret":
            b.add(f"mag_{tag}", "interpret", [f"byte_{tag}"], out=MAG8)
            b.add(f"x_{tag}", "decode", [f"mag_{tag}"], out=FMAG)
        else:
            b.add(f"tup_{tag}", "tuple", [f"byte_{tag}"])
            b.add(f"plain_{tag}", "pack", [f"tup_{tag}"], out=PLAIN8)
            b.add(f"x_{tag}", "decode", [f"plain_{tag}"], out=FPLAIN)
        b.choice(f"w_{tag}", [("identity", (f"wv{i}",), None, None) for i in range(len(pool))])
        terms.append(b.add(f"term_{tag}", "mul", [f"x_{tag}", f"w_{tag}"]))
    b.add("packed", "tuple", terms)
    b.add("conv", "sum", ["packed"])
    b.add("record", "tuple", ["pos", "conv"])
    return b.program((("y", "record"),))


def index_temperatures(model, tau):
    """Sharpen the `index` gather without touching any choice distribution.

    `index`'s relaxation is `softmax(-(addr - k)^2 / tau)`, which at the shipped
    tau=1 keeps only 0.564 of its mass on the addressed element (FINDINGS 16,
    mechanism M3), so the relaxed gather returns a blur of the neighbourhood.
    The D2 coupling -- one temperature for both the surrogate and the candidate
    softmax -- does not bite on these nodes because each has exactly one
    candidate, so a 1-way softmax is temperature-invariant.  Reported, not tuned
    away: `gather_blur` below measures what it costs at each setting.
    """
    for n in model.program.nodes:
        if all(c.operator.name == "index" for c in n.candidates) and len(n.candidates) == 1:
            model.temperatures[n.name] = tau
    return model


def gather_blur(registry, resolution, rows, taus=(1., .25, .05, .01)):
    """Relaxed-vs-exact gather error for the stencil, at several index temperatures."""
    from tcn.learning import SoftProgram, tensor as _tensor
    prog = fixed_kernel_scaffold(registry, resolution, SOBEL_X)
    inputs = {"rec": torch.stack([_tensor(ex["inputs"]["rec"]) for ex in rows])}
    out = {}
    for tau in taus:
        model = index_temperatures(SoftProgram(prog, registry), tau)
        with torch.no_grad():
            _, _, tr = model(inputs, return_trace=True)
        worst = 0.
        for i, ex in enumerate(rows):
            _, _, exact = prog.execute(ex["inputs"], registry=registry,
                                       selections={n.name: 0 for n in prog.nodes})
            for dy, dx in STENCIL:
                got = float(tr[f"byte_{dy}{dx}"][i, 0])
                want = float(exact[f"byte_{dy}{dx}"].decoded)
                worst = max(worst, abs(got - want))
        out[str(tau)] = worst
    return out


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution", type=int, default=8)
    ap.add_argument("--train", type=int, default=6)
    ap.add_argument("--held", type=int, default=6)
    ap.add_argument("--enum-train", type=int, default=3)
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=.25)
    ap.add_argument("--index-tau", type=float, default=.02)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--apply", type=int, default=3)
    ap.add_argument("--apply-resolution", type=int, default=16)
    ap.add_argument("--grad-taus", type=float, nargs="*", default=[1., .25, .05, .02])
    ap.add_argument("--tag", default="convolution")
    args = ap.parse_args()

    R = args.resolution
    r = Registry()
    tol = 1e-6
    sigs = signals()
    train_seeds = tuple(range(args.train))
    held_seeds = tuple(range(100, 100 + args.held))
    result = {"arguments": vars(args)}

    # ---- data ----
    train = examples(train_seeds, R, SOBEL_X, "train")
    held = examples(held_seeds, R, SOBEL_X, "test")
    result["positions_per_image"] = len(block_positions(R))
    result["train_rows"] = len(train)
    result["held_rows"] = len(held)
    result["target_train"] = target_stats(train, sigs)
    result["target_held"] = target_stats(held, sigs)
    report("3x3 blocks per image / train rows / held-out rows",
           f"{len(block_positions(R))} / {len(train)} / {len(held)}")
    report("Sobel-x target range (train)",
           f"[{result['target_train']['min']:.0f}, {result['target_train']['max']:.0f}]")

    # ---- the relaxed gather, measured before it is used ----
    result["gather_blur"] = gather_blur(r, R, train[:8])
    report("relaxed index gather, max byte error by temperature",
           " ".join(f"tau={k}: {v:.3g}" for k, v in result["gather_blur"].items()))

    # ---- arm A: learn the nine weights ----
    prog = kernel_scaffold(r, R)
    result["arm_a"] = {"nodes": len(prog.nodes), "trainable_constants": list(prog.trainable_constants),
                       "space_size": space_size(prog)}
    rows = []
    for seed in range(args.seeds):
        t0 = time.perf_counter()
        model, exported, info = local_fit(
            prog, train, sigs, steps=args.steps, lr=args.lr, registry=r,
            init_noise=0., seed=seed, temperatures={"index": args.index_tau},
            constant_noise=.05)
        learned = {f"{dy}{dx}": float(model.constants[f"w_{dy}{dx}"].detach())
                   for dy, dx in STENCIL}
        rounded = {k: round(v) for k, v in learned.items()}
        rprog = fixed_kernel_scaffold(r, R, {(int(k[0]), int(k[1])): v for k, v in rounded.items()})
        sel = {n.name: 0 for n in exported.nodes}
        rsel = {n.name: 0 for n in rprog.nodes}
        rows.append({
            "seed": seed, "seconds": time.perf_counter() - t0,
            "relaxed_loss": info["relaxed_loss"],
            "learned_weights": learned, "rounded_weights": rounded,
            "rounded_is_reference": rounded == {f"{dy}{dx}": int(w) for (dy, dx), w in SOBEL_X.items()},
            "train_mse": exact_mse(exported, train, sigs, r, sel),
            "held_mse": exact_mse(exported, held, sigs, r, sel),
            "held_max_error": exact_error(exported, held, sigs, r, sel),
            "rounded_train_max_error": exact_error(rprog, train, sigs, r, rsel),
            "rounded_held_max_error": exact_error(rprog, held, sigs, r, rsel),
        })
        report(f"arm A seed {seed}: held mse / held max err / rounded held max err",
               f"{rows[-1]['held_mse']:.4g} / {rows[-1]['held_max_error']:.4g} / "
               f"{rows[-1]['rounded_held_max_error']:.4g}")
    result["arm_a"]["rows"] = rows
    result["arm_a"]["exact_after_rounding"] = sum(x["rounded_held_max_error"] == 0. for x in rows)
    result["arm_a"]["baselines"] = {
        "predict_zero_held_mse": result["target_held"]["mse_about_zero"],
        "predict_train_mean_held_mse": sum(
            (v - result["target_train"]["mean"]) ** 2
            for ex in held for s in sigs for v in ex["targets"][s.target].flat()) / result["target_held"]["n"],
    }
    report("arm A exact after rounding (of seeds)",
           f"{result['arm_a']['exact_after_rounding']}/{len(rows)}")
    report("baseline held mse: predict 0 / predict train mean",
           f"{result['arm_a']['baselines']['predict_zero_held_mse']:.4g} / "
           f"{result['arm_a']['baselines']['predict_train_mean_held_mse']:.4g}")
    dump(args.tag, result)

    # ---- arm B and B': the enumerable two-tap derivative ----
    etrain = examples(tuple(range(args.enum_train)), R, DERIVATIVE_X, "train")
    eheld = examples(held_seeds, R, DERIVATIVE_X, "test")
    result["derivative_target_train"] = target_stats(etrain, sigs)
    for exit_name, out_type in (("interpret", FMAG), ("pack", FPLAIN)):
        sig = signals(out_type)
        d = discrete_scaffold(r, R, exit=exit_name)
        arm = {"space_size": space_size(d), "nodes": len(d.nodes)}
        tr = [{"inputs": ex["inputs"],
               "targets": {"conv": Value.of(out_type, ex["targets"]["conv"].decoded)}}
              for ex in etrain]
        hl = [{"inputs": ex["inputs"],
               "targets": {"conv": Value.of(out_type, ex["targets"]["conv"].decoded)}}
              for ex in eheld]
        enum = enumerate_reference(d, tr, sig, r, tolerance=tol)
        if enum["solved"]:
            enum["held_out_max_error"] = exact_error(d, hl, sig, r, enum["selections"])
        arm["enumerate_fit"] = enum
        arm["all_conforming"] = all_conforming(d, tr, sig, r, tolerance=tol)
        report(f"arm B [{exit_name}] space/solved/exhausted/unique/conforming/seconds",
               f"{arm['space_size']} / {enum['solved']} / {enum['exhausted']} / "
               f"{enum['unique']} / {enum['conforming']} / {enum['seconds']:.1f}")
        # The relaxed arm is swept over the `index` temperature rather than run at
        # one setting, because that single number serves two purposes at one node:
        # it sharpens the gather (FINDINGS 16 mechanism M3) *and* it decides
        # whether a mixed address has any gradient at all.
        arm["gradient"] = {}
        for tau in args.grad_taus:
            grows = []
            for seed in range(args.seeds):
                t0 = time.perf_counter()
                _, exported, info = local_fit(
                    d, tr, sig, steps=args.steps, lr=.15, registry=r, init_noise=.5, seed=seed,
                    temperatures={"index": tau},
                    report_grads=("addr_a", "addr_b", "w_a", "w_b"))
                sel = {n.name: 0 for n in exported.nodes}
                err = exact_error(exported, tr, sig, r, sel)
                grows.append({"seed": seed, "seconds": time.perf_counter() - t0,
                              "exact_max_error": err, "conforming": err <= tol,
                              "choice_gradients": info["choice_gradients"],
                              "held_max_error": exact_error(exported, hl, sig, r, sel),
                              "selections": info["selections"]})
            addr = [g for x in grows for k, g in x["choice_gradients"].items()
                    if k.startswith("addr") and g is not None]
            arm["gradient"][str(tau)] = {
                "rows": grows, "init_noise": .5,
                "successes": sum(x["conforming"] for x in grows),
                "address_logit_gradient_max": max(addr, default=None),
                "address_logit_gradient_min": min(addr, default=None),
                "address_logit_gradient_none": sum(
                    1 for x in grows for k, g in x["choice_gradients"].items()
                    if k.startswith("addr") and g is None)}
            report(f"arm B [{exit_name}] index tau={tau}: successes / addr logit grad [min,max]",
                   f"{arm['gradient'][str(tau)]['successes']}/{len(grows)} / "
                   f"[{arm['gradient'][str(tau)]['address_logit_gradient_min']:.4g}, "
                   f"{arm['gradient'][str(tau)]['address_logit_gradient_max']:.4g}]")
        result[f"arm_b_{exit_name}"] = arm
        dump(args.tag, result)

    # ---- apply the crystallized kernel at every position ----
    ref = fixed_kernel_scaffold(r, R, SOBEL_X)
    frozen = ref.harden({n.name: 0 for n in ref.nodes})
    name = r.register_module(frozen)
    apply_rows = []
    for AR in sorted({R, args.apply_resolution}):
        aref = fixed_kernel_scaffold(r, AR, SOBEL_X)
        aname = r.register_module(aref.harden({n.name: 0 for n in aref.nodes}))
        positions = block_positions(AR)
        caller = positional_scaffold(r, bytes_type(AR), positions, [aname], index=IDX)
        LAB = product(IDX, FMAG)
        errors, seconds = [], []
        for s in range(200, 200 + args.apply):
            pixels = episode(s, AR, split="test")
            labels = convolve(pixels, AR, SOBEL_X)
            t0 = time.perf_counter()
            got = caller.run({"observation": Value.of(bytes_type(AR), pixels)}, registry=r)[0]["mapped"]
            seconds.append(time.perf_counter() - t0)
            target = Value.of(setof(LAB, len(positions)),
                              tuple((p, labels[p]) for p in positions))
            errors.append(max((abs(a - b) for a, b in zip(got.flat(), target.flat())), default=0.))
        apply_rows.append({"resolution": AR, "positions": len(positions),
                           "caller_nodes": len(caller.nodes),
                           "max_error": max(errors),
                           "median_seconds": sorted(seconds)[len(seconds) // 2]})
        report(f"apply at R={AR}: positions / caller nodes / max error",
               f"{len(positions)} / {len(caller.nodes)} / {max(errors)}")
    result["apply"] = apply_rows
    result["module"] = {"name": name,
                        "description_bits": frozen.pruned().description_bits(r),
                        "execution_cost": frozen.pruned().execution_cost(r)}
    dump(args.tag, result)
    print(f"\nwrote {(pathlib.Path(__file__).parent / 'out' / (args.tag + '.json'))}")


if __name__ == "__main__":
    main()
