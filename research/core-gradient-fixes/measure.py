"""Re-measure both defects: severing, coupling, surrogate values AND loss minima.

Every surrogate arm below is named by its **effective** temperature, and reached
through the shipped `relaxed` as `temperature = tau_eff / surrogate_scale(op)`.
So `tau=1` is the pre-fix behaviour for every operator, bit for bit, and the two
arms differ only in the scale rather than in the code path.

A gradient magnitude alone cannot say whether a wider surrogate helps: a
temperature that restores a derivative while moving the loss minimum onto a wrong
answer is worse than the dead surrogate it replaced. So every family below is
reported twice -- the surrogate's value and derivative at the operating distance,
and **where the loss minimum sits** over the candidate set the search chooses
from.

`SoftProgram` zero-initializes every choice logit, so `torch.manual_seed` does not
vary anything here; the surrogate numbers are closed-form single outcomes and the
descent probe sweeps every start rather than sampling seeds.

Run:  .venv/bin/python research/core-gradient-fixes/measure.py
"""
import json
import math
import sys
import torch

from tcn.types import BOOL, Value, integer, floating, fixed, product, setof
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate
from tcn.learning import SoftProgram, relaxed, carrier_temperature

R = Registry()
BYTE = integer(8, signed=False, role="byte")   # the language track's raw text carrier
UBYTE = integer(8, signed=False)               # the same width, numeric, so `lt` is legal
IDX = integer(32, signed=False)


def at_tau(op, xs, tau_eff):
    """The relaxation at an explicit effective temperature.

    The carrier scaling is opt-in, so a plain call already has an effective
    temperature of exactly `tau_eff` for every operator, and `tau_eff = 1.0` is
    the pre-fix behaviour bit for bit.
    """
    return relaxed(R, op, xs, tau_eff)


def bce(pred, target):
    p = pred.clamp(1e-6, 1 - 1e-6)
    return float(-(target * p.log() + (1 - target) * (1 - p).log()).mean())


# --------------------------------------------------------------------------- D1

def severing():
    """D1 -- a trainable constant feeding `mul` through an identity node.

    The exact reproduction from FINDINGS section 18: `tensor([1.])` when the node
    is left unselected, `None` when it carries `selected=0`.
    """
    F = floating()

    def build(selected, candidates=1):
        ops = [R.resolve("identity", (F,))] + ([R.resolve("neg", (F,))] if candidates > 1 else [])
        node = Node("pipe", F, tuple(Candidate(o, ("k",)) for o in ops), "core", 1, selected)
        out = Node("y", F, (Candidate(R.resolve("mul", (F, F)), ("pipe", "x")),), "core", 2)
        return Program((("x", F),), (node, out), (("out", "y"),),
                       (("k", Value.of(F, 3.0)),), trainable_constants=("k",))

    rows = {}
    for label, prog in (("single candidate, unselected", build(None)),
                        ("single candidate, selected=0", build(0)),
                        ("two candidates, selected=0", build(0, 2))):
        m = SoftProgram(prog, R)
        out, _ = m({"x": torch.tensor([1.0])})
        g = torch.autograd.grad(out["out"].sum(), m.constants["k"], allow_unused=True)[0]
        rows[label] = {"constant_gradient": None if g is None else g.tolist(),
                       "value": out["out"].tolist()}
    # The third route: a node committed by the crystallizer, which stays a boundary.
    # `freeze()` also retires the constant, whose only user is now frozen, so the
    # boundary shows up twice -- as a severed path and as a retired parameter.
    m = SoftProgram(build(None), R)
    m.freeze("pipe", 0)
    out, _ = m({"x": torch.tensor([1.0])})
    k = m.constants["k"]
    g = (torch.autograd.grad(out["out"].sum(), k, allow_unused=True)[0]
         if k.requires_grad and out["out"].requires_grad else None)
    rows["crystallized by freeze()"] = {"constant_gradient": None if g is None else g.tolist(),
                                        "constant_requires_grad": bool(k.requires_grad),
                                        "output_requires_grad": bool(out["out"].requires_grad),
                                        "value": out["out"].tolist()}
    return rows


# --------------------------------------------------------------------------- D2

def coupling():
    """D2 -- one temperature served the choice softmax and the surrogate at once."""
    p = Program((("a", UBYTE), ("b", UBYTE)),
                (Node("z", BOOL, tuple(Candidate(R.resolve(n, (UBYTE, UBYTE)), ("a", "b"))
                                       for n in ("eq", "lt", "gt")), "core", 1),),
                (("out", "z"),))
    m = SoftProgram(p, R)
    with torch.no_grad():
        m.choices[0][0] = 1.0
    xs = {"a": torch.tensor([0.0]), "b": torch.tensor([48.0])}

    def readout(choice_tau, surrogate_tau):
        m.temperatures["z"] = choice_tau
        m.surrogate_scale["z"] = surrogate_tau / choice_tau
        for q in m.choices:
            q.grad = None
        out, _ = m(xs)
        out["out"].sum().backward()
        w = torch.softmax(m.choices[0].detach() / choice_tau, 0)
        return {"choice_entropy": float(-(w * w.clamp_min(1e-12).log()).sum()),
                "choice_grad_absmax": float(m.choices[0].grad.abs().max())}

    return {"choice 1, surrogate 1 (shipped)": readout(1.0, 1.0),
            "choice 1, surrogate 8 (now possible)": readout(1.0, 8.0),
            "choice 8, surrogate 8 (the coupling)": readout(8.0, 8.0)}


def eq_surrogate():
    """D2a -- `eq`'s Gaussian surrogate is exactly 0.0 in float32 from |a-b| >= 11."""
    op = R.resolve("eq", (BYTE, BYTE))
    rows = []
    for d in (1, 5, 10, 11, 16, 25.6, 32, 48, 64, 128, 255):
        out = {"delta": d}
        for label, tau in (("before", 1.0), ("after", carrier_temperature(BYTE))):
            a = torch.tensor([0.0], requires_grad=True)
            y = at_tau(op, [a, torch.tensor([float(d)])], tau)
            g = torch.autograd.grad(y.sum(), a, allow_unused=True)[0]
            out[label + "_value"] = float(y)
            out[label + "_grad"] = abs(float(g))
        rows.append(out)
    return {"carrier_temperature": carrier_temperature(BYTE), "opt_in": True, "rows": rows}


def eq_minimum(true_byte=40, n=64, seed=3):
    """Where does `eq`'s loss minimum sit over the 256 constants a search chooses from?

    This is the shape of the language track's stage-A node: `eq(index(bytes, k), c)`
    with `c` searched over the whole alphabet. Scaling a Gaussian kernel cannot
    move its own maximum off `d = 0`, so the correct constant must stay the
    minimizer at every temperature; what changes is whether the objective can be
    told apart from its neighbours at all.
    """
    g = torch.Generator().manual_seed(seed)
    # `eq` sums squared differences over the last axis, so each row must be its
    # own one-wide value rather than one 64-wide vector.
    x = torch.randint(0, 256, (n, 1), generator=g).float()
    x[: n // 4] = float(true_byte)                       # a quarter of the batch matches
    target = (x == true_byte).float()
    op = R.resolve("eq", (BYTE, BYTE))
    out = {}
    for label, tau in (("tau=1 (before)", 1.0), ("tau=12.8 (annealed floor, 0.05*256)", 12.8),
                       ("tau=32", 32.0), ("tau=64", 64.0), ("tau=256 (carrier, shipped)", 256.0),
                       ("tau=65536 (carrier squared)", 65536.0)):
        losses = [bce(at_tau(op, [x, torch.full_like(x, float(c))], tau), target) for c in range(256)]
        best = min(range(256), key=lambda c: losses[c])
        a = torch.tensor([[float(true_byte + 25)]], requires_grad=True)
        y = at_tau(op, [a, torch.tensor([[float(true_byte)]])], tau)
        grad = abs(float(torch.autograd.grad(y.sum(), a, allow_unused=True)[0]))
        out[label] = {"argmin": best, "correct": best == true_byte,
                      "loss_spread": max(losses) - min(losses),
                      "loss_at_correct": losses[true_byte],
                      "loss_at_neighbour": losses[true_byte + 1],
                      "surrogate_grad_at_delta_25": grad}
    return {"true_byte": true_byte, "batch": n, "arms": out}


def compare_surrogate():
    """D2b -- `lt`/`le`/`gt`/`ge` gradients are exactly 0.0 from |a-b| >= 17."""
    out = {}
    for name in ("lt", "le", "gt", "ge"):
        op = R.resolve(name, (UBYTE, UBYTE))
        rows = []
        for d in (1, 8, 16, 17, 24, 48, 96, 200):
            row = {"delta": d}
            for label, tau in (("tau=1 (shipped, kept)", 1.0), ("tau=32 (margin)", 32.0),
                               ("tau=256 (carrier, rejected)", 256.0)):
                a = torch.tensor([0.0], requires_grad=True)
                y = at_tau(op, [a, torch.tensor([float(d)])], tau)
                row[label] = abs(float(torch.autograd.grad(y.sum(), a, allow_unused=True)[0]))
            rows.append(row)
        out[name] = {"carrier_scaling_applies": False,
                     "carrier_temperature": carrier_temperature(UBYTE), "rows": rows}
    return out


def compare_minimum(true_threshold=128, n=96, seed=5, gap=60):
    """Where does `le`'s loss minimum sit over the thresholds a search chooses from?

    The check the coordinating session asked for, run independently here. `le`'s
    surrogate is monotone in `d`, but it enters the loss against a threshold that
    is *itself* the thing being chosen, so the minimizing threshold moves with the
    temperature. Byte-scale operands with a wide spread, exactly the regime where
    `tau = 1` is numerically dead.
    """
    g = torch.Generator().manual_seed(seed)
    # The operating regime the defect was recorded in: most operands sit far past
    # the point where `tau = 1` is numerically dead, with only two adjacent
    # operands straddling the boundary. Those two are what pin the answer to a
    # single correct threshold, and they are exactly what a flattened surrogate
    # stops being able to see.
    far = torch.randint(0, 256, (n - 2, 1), generator=g).float()
    far = torch.where((far - true_threshold).abs() < gap,
                      (far + gap) % 256, far)
    x = torch.cat([torch.tensor([[float(true_threshold)], [float(true_threshold + 1)]]), far])
    target = (x <= true_threshold).float()
    op = R.resolve("le", (UBYTE, UBYTE))
    gaps = (x - true_threshold).abs()
    out = {}
    for label, tau in (("tau=1 (shipped, kept)", 1.0), ("tau=8", 8.0), ("tau=32 (margin)", 32.0),
                       ("tau=256 (carrier, rejected)", 256.0), ("tau=65536", 65536.0)):
        losses = [bce(at_tau(op, [x, torch.full_like(x, float(c))], tau), target) for c in range(256)]
        best = min(range(256), key=lambda c: losses[c])
        # The threshold is exact for any cut between the largest x <= t and the
        # smallest x > t, so "correct" means the hard predicate agrees everywhere.
        correct = bool(((x <= best).float() == target).all())
        out[label] = {"argmin": best, "hard_predicate_correct": correct,
                      "loss_spread": max(losses) - min(losses)}
    return {"true_threshold": true_threshold, "batch": n,
            "median_operand_gap": float(gaps.median()),
            "fraction_past_the_dead_point_17": float((gaps >= 17).float().mean()),
            "arms": out}


def index_kernel(count=12, seed=11):
    """D2c -- `index` kernel mass on the addressed element, minima, and descent.

    Kernel mass is a property of the lattice alone. The landscape and the descent
    are measured on byte-scale values in an **arbitrary arrangement**, which is
    what makes the landscape multi-modal at all: against a monotone ramp the
    squared error is unimodal at every temperature and says nothing.

    The wide arm here is the one this fix does **not** ship, and the sharp arm is
    what the choice/surrogate split newly makes reachable without concentrating
    the node's candidate distribution at the same time.
    """
    tup = product(*([BYTE] * count))
    op = R.resolve("index", (tup, IDX))
    g = torch.Generator().manual_seed(seed)
    values = torch.randint(0, 256, (count,), generator=g).float().reshape(1, count)
    target_index = 7
    target = values[0, target_index]

    def mass(tau):
        w = torch.softmax(-(torch.tensor([count / 2.0]) - torch.arange(count)) ** 2 / tau, dim=-1)
        return float(w.max())

    def loss_at(tau, x):
        return (at_tau(op, [values, x.reshape(1)], tau) - target) ** 2

    def minima(tau):
        xs = torch.arange(0.0, count - 1 + 1e-9, 0.02)
        losses = [float(loss_at(tau, x)) for x in xs]
        return sum(1 for i in range(1, len(losses) - 1)
                   if losses[i] < losses[i - 1] and losses[i] < losses[i + 1])

    def descent(tau):
        hits = 0
        for start in range(count):
            x = torch.tensor([float(start)], requires_grad=True)
            opt = torch.optim.Adam([x], lr=.1)
            for _ in range(400):
                opt.zero_grad(); loss_at(tau, x).sum().backward(); opt.step()
            hits += int(round(float(x))) == target_index
        return hits

    arms = {}
    for label, tau in (("tau=1 (shipped, kept)", 1.0), ("tau=12 (positions, rejected)", float(count)),
                       ("tau=0.1 (annealed, now separable)", .1)):
        arms[label] = {"center_kernel_mass": mass(tau), "local_minima": minima(tau),
                       "descent_hits": descent(tau), "starts": count}
    return {"count": count, "carrier_scaling_applies": False, "target_index": target_index,
            "arms": arms}


def temperatures():
    return {
        "bool": carrier_temperature(BOOL),
        "int[8] role=byte": carrier_temperature(BYTE),
        "int[8] unsigned": carrier_temperature(UBYTE),
        "int[16] signed": carrier_temperature(integer(16)),
        "float32": carrier_temperature(floating()),
        "fixed[16] scale 256": carrier_temperature(fixed(16, 256)),
        "int[8] role=category": carrier_temperature(integer(8, signed=False, role="category")),
        "tuple(bool,byte)": carrier_temperature(product(BOOL, BYTE)),
        "set[bool] capacity 4": carrier_temperature(setof(BOOL, 4)),
    }


# --------------------------------------------------------------------------- D3

def tuple_broadcast():
    """D3 -- `relaxed("tuple")` was a bare `torch.cat`, which does not broadcast."""
    F = floating()
    batched = torch.arange(8.).reshape(8, 1)
    constant = torch.tensor([3.0], requires_grad=True)
    out = {"add_on_the_same_pair": list(relaxed(R, R.resolve("add", (F, F)),
                                                [batched, constant]).shape)}
    try:
        y = torch.cat([batched, constant], dim=-1)
        out["before"] = list(y.shape)
    except RuntimeError as e:
        out["before"] = f"RuntimeError: {e}"
    y = relaxed(R, R.resolve("tuple", (F, F)), [batched, constant])
    out["after"] = list(y.shape)
    out["constant_gradient"] = torch.autograd.grad(y.sum(), constant)[0].tolist()
    return out


# --------------------------------------------------------------------------- context

def candidate_liveness(alphabet=256, examples=12, seed=17):
    """Liveness read **per candidate per example**, not pooled over the batch.

    A batch aggregate calls a node differentiable whenever *any* candidate is
    live on *any* example, which is exactly how a node whose reference candidate
    can never be selected passes for healthy. This is the same shape as the
    language track's stage-A node: 256 `eq(x, c)` candidates against a handful of
    byte examples.
    """
    op = R.resolve("eq", (BYTE, BYTE))
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, 256, (examples, 1), generator=g).float()
    out = {}
    for label, scaled in (("before", False), ("after", True)):
        live = torch.zeros(alphabet, examples)
        for c in range(alphabet):
            a = x.clone().requires_grad_(True)
            y = relaxed(R, op, [a, torch.full_like(x, float(c))], 1., scaled)
            grad = torch.autograd.grad(y.sum(), a)[0].abs()
            live[c] = (grad.flatten() > 0).float()
        per_candidate = live.sum(1)
        out[label] = {
            "candidates": alphabet, "examples": examples,
            "candidates_live_on_no_example": int((per_candidate == 0).sum()),
            "candidates_live_on_every_example": int((per_candidate == examples).sum()),
            "mean_examples_live_per_candidate": float(per_candidate.mean()),
            "pooled_node_looks_differentiable": bool(live.any()),
        }
    out["operand_bytes"] = [int(v) for v in x.flatten()]
    return out


def truth_table_reachability():
    """`truth_0` and `truth_15` are constants: no gradient can ever select them.

    Their relaxation has identically zero derivative in both inputs, so a mixture
    can move toward them only through a choice logit that never receives a signal
    through the value path. Enumeration searches all 16. Reported for the two
    shipped fixtures, whose scaffolds are built from this family.
    """
    from examples.mixed import problem
    from examples.joint import trainer
    reachable = []
    for k in range(16):
        op = R.resolve(f"truth_{k}", (BOOL, BOOL))
        a = torch.tensor([.5], requires_grad=True); b = torch.tensor([.25], requires_grad=True)
        y = relaxed(R, op, [a, b])
        ga, gb = torch.autograd.grad(y.sum(), [a, b], allow_unused=True)
        alive = any(t is not None and bool((t != 0).any()) for t in (ga, gb))
        reachable.append(alive)
    out = {"truth_family": {"reachable": sum(reachable), "total": 16,
                            "dead": [k for k, v in enumerate(reachable) if not v],
                            "fraction": sum(reachable) / 16}}
    for name, program in (("examples/mixed.py", problem()[0]),
                          ("examples/joint.py", trainer(1).model.program)):
        # The reachable share of the SPACE, which is the product over nodes --
        # the quantity enumeration searches all of.
        total = alive = 1
        for n in program.nodes:
            ok = sum(0 if (c.operator.name.startswith("truth_")
                           and not reachable[int(c.operator.name[6:])]) else 1
                     for c in n.candidates)
            total *= len(n.candidates); alive *= ok
        out[name] = {"space_size": total, "reachable_by_relaxation": alive,
                     "fraction": alive / total}
    return out


if __name__ == "__main__":
    torch.set_num_threads(1)
    report = {"severing": severing(), "coupling": coupling(),
              "tuple_broadcast": tuple_broadcast(), "carrier_temperatures": temperatures(),
              "eq_surrogate": eq_surrogate(), "eq_minimum": eq_minimum(),
              "candidate_liveness": candidate_liveness(),
              "compare_surrogate": compare_surrogate(), "compare_minimum": compare_minimum(),
              "index": index_kernel(), "reachability": truth_table_reachability()}
    from pathlib import Path
    Path(__file__).with_name("measure.json").write_text(json.dumps(report, indent=1))
    json.dump(report, sys.stdout, indent=1)
    print()
