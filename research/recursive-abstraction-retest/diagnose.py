"""Why the search does or does not put the module on the output path.

Track 5 measured the outcome (0 of 20 runs) but attributed it only loosely. If
the module is still not selected now that it is cheaper, the accounting is not
the explanation and the search is, so the candidate explanations have to be
separated:

  P1 detachment        a module candidate's output carries no gradient, so a
                       node *feeding* a module gets no signal through it.
  P2 hard thresholding  arguments are thresholded at 0.5 before exact execution,
                       so during soft search a module is a step function of the
                       mixture rather than a relaxation of it -- except when its
                       arguments are the exact inputs, where it is exact.
  P3 dilution          the module contributes many candidates at every node, so
                       softmax mass on the *correct* binding starts tiny.
  P4 credit structure   the composite gives no partial credit: one correct call
                       alone does not reduce the loss, so there is no gradient
                       path that rewards acquiring the first call before the
                       second.
  P5 selection gradient is the correct module binding's logit gradient actually
                       competitive with the best primitive's at initialization?
  P6 basin             if the correct choice is handed to the search, does
                       training keep it or drift away?
"""
from __future__ import annotations

import itertools
import json
import statistics
from pathlib import Path

import torch

from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.learning import SoftProgram, tensor, exact_tensor, relaxed

import common as C

HERE = Path(__file__).parent


def build(scaffold="tight"):
    r = Registry()
    name = r.register_module(C.minimal_module(r, "maj"))
    prog = (C.tight_scaffold if scaffold == "tight" else C.wide_scaffold)(r, name)
    ex = C.composite_examples()
    inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in ex]) for k, _ in prog.inputs}
    targets = {s.target: torch.stack([tensor(e["targets"][s.target]) for e in ex]) for s in C.COMP_SIGNALS}
    return r, prog, ex, inputs, targets


def loss_of(model, inputs, targets):
    _, _, tr = model(inputs, return_trace=True)
    return model.probe_loss(tr, targets, C.COMP_SIGNALS)


def correct_index(node, sources):
    for i, c in enumerate(node.candidates):
        if c.operator.name.startswith("module:") and c.sources == sources:
            return i
    raise LookupError


def main():
    out = {}
    r, prog, ex, inputs, targets = build("tight")
    n1, n2, y = prog.nodes

    # ---- P1 / P2 : the gradient boundary --------------------------------
    mop = next(c.operator for c in n1.candidates if c.operator.name.startswith("module:"))
    xs_exact = [inputs[k].clone().requires_grad_(True) for k in ("a", "b", "c")]
    ym = exact_tensor(r, mop, xs_exact)
    xop = r.resolve("xor", (BOOL, BOOL))
    yx = relaxed(r, xop, xs_exact[:2])
    step = exact_tensor(r, mop, [torch.tensor([[0.49], [0.51]]), torch.tensor([[0.51], [0.51]]),
                                 torch.tensor([[0.51], [0.51]])])
    out["P1_P2_boundary"] = {
        "module_output_requires_grad": bool(ym.requires_grad),
        "primitive_output_requires_grad": bool(yx.requires_grad),
        "module_gradient_declaration": mop.gradient,
        "threshold_step_0.49_vs_0.51": [float(step[0, 0]), float(step[1, 0])],
        "arguments_from_inputs_are_exact": True,
        "note": "bindings drawn from the program inputs receive exact 0/1 arguments, "
                "so a module call over inputs is exact during soft search; only a "
                "module fed by another soft node sees the step function",
    }

    # ---- P3 : dilution ---------------------------------------------------
    dil = {}
    for n in prog.nodes:
        idx = [i for i, c in enumerate(n.candidates) if c.operator.name.startswith("module:")]
        dil[n.name] = {"candidates": len(n.candidates), "module_candidates": len(idx),
                       "uniform_mass_per_correct_binding": round(1 / len(n.candidates), 6)}
    r2 = Registry()
    wide = C.wide_scaffold(r2, r2.register_module(C.minimal_module(r2, "maj")))
    dil["wide_scaffold_totals"] = {
        "candidates": sum(len(n.candidates) for n in wide.nodes),
        "module_candidates": sum(sum(1 for c in n.candidates if c.operator.name.startswith("module:"))
                                 for n in wide.nodes)}
    out["P3_dilution"] = dil

    # ---- P4 : partial credit --------------------------------------------
    # Hold n1 at the correct call and n2 at every candidate in turn; how much of
    # the loss does one correct call buy on its own?
    model = SoftProgram(prog, r)
    torch.manual_seed(0)
    base = float(loss_of(model, inputs, targets).detach())
    i1 = correct_index(n1, ("a", "b", "c"))
    i2 = correct_index(n2, ("d", "e", "f"))
    iy = next(i for i, c in enumerate(y.candidates)
              if c.operator.name == "xor" and c.sources == ("n1", "n2"))

    def loss_with(sel):
        m = SoftProgram(prog, r)
        for name, idx in sel.items():
            m.freeze(name, idx)
        return float(loss_of(m, inputs, targets).detach())

    out["P4_partial_credit"] = {
        "uniform_mixture_loss": round(base, 5),
        "correct_n1_only": round(loss_with({"n1": i1}), 5),
        "correct_n1_and_y": round(loss_with({"n1": i1, "y": iy}), 5),
        "correct_n2_and_y": round(loss_with({"n2": i2, "y": iy}), 5),
        "correct_n1_n2_only": round(loss_with({"n1": i1, "n2": i2}), 5),
        "all_three_correct": round(loss_with({"n1": i1, "n2": i2, "y": iy}), 5),
    }

    # ---- P5 : is the correct binding's logit gradient competitive? -------
    ranks = {}
    for seed in range(8):
        torch.manual_seed(seed)
        m = SoftProgram(prog, r)
        with torch.no_grad():
            for p in m.choices:
                p.add_(torch.randn_like(p) * 0.5)
        loss = loss_of(m, inputs, targets)
        loss.backward()
        for node, idx, logits in ((n1, i1, m.choices[0]), (n2, i2, m.choices[1]), (y, iy, m.choices[2])):
            g = logits.grad.detach()
            # a more negative gradient means gradient descent raises that logit
            order = torch.argsort(g)
            rank = int((order == idx).nonzero()[0, 0]) + 1
            ranks.setdefault(node.name, []).append(
                {"seed": seed, "rank_of_correct": rank, "of": len(g),
                 "grad_correct": round(float(g[idx]), 6),
                 "grad_best": round(float(g.min()), 6),
                 "percentile": round(100 * rank / len(g), 1)})
    out["P5_selection_gradient"] = {
        k: {"median_rank": statistics.median([x["rank_of_correct"] for x in v]),
            "of": v[0]["of"],
            "median_percentile": statistics.median([x["percentile"] for x in v]),
            "per_seed": v}
        for k, v in ranks.items()}

    # ---- P6 : basin -- hand the search the answer and let it train -------
    torch.manual_seed(0)
    m = SoftProgram(prog, r)
    with torch.no_grad():
        for p in m.choices:
            p.add_(torch.randn_like(p) * 0.5)
        m.choices[0][i1] += 3.0
        m.choices[1][i2] += 3.0
        m.choices[2][iy] += 3.0
    opt = torch.optim.Adam(m.parameters(), lr=0.05)
    traj = []
    for step in range(200):
        opt.zero_grad(); loss = loss_of(m, inputs, targets) + 0.001 * (step / 200) * m.entropy()
        loss.backward(); opt.step()
        if step % 20 == 0 or step == 199:
            q = [torch.softmax(p, 0).detach() for p in m.choices]
            traj.append({"step": step, "loss": round(float(loss.detach()), 5),
                         "mass_n1_correct": round(float(q[0][i1]), 4),
                         "mass_n2_correct": round(float(q[1][i2]), 4),
                         "mass_y_correct": round(float(q[2][iy]), 4),
                         "argmax_is_solution": (int(q[0].argmax()) == i1 and int(q[1].argmax()) == i2
                                                and int(q[2].argmax()) == iy)})
    out["P6_basin"] = {"seeded_with_the_answer": traj,
                       "final_conformant": C.conformant(m.export(), ex, C.COMP_SIGNALS, r)}

    (HERE / "diagnose.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
