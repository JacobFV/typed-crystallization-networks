"""What a frozen-module candidate does to learning signal inside a soft graph.

ARCHITECTURE.md sec. 5: "At the frozen callable boundary, gradients stop.
Remaining trainable regions need their own probe/objective or another valid
gradient path; otherwise defer freezing. Outside-in freezing must never
silently disconnect all learning signal to the interior."

Three measurable consequences of the current implementation:

  G1. `relaxed()` sends any operator with gradient=='none' (every module call)
      through `exact_tensor`, which detaches. The call's OUTPUT carries no
      gradient at all -- not even a surrogate.
  G2. A node whose only consumer is a module call receives exactly zero
      gradient. Its choice logits cannot move.
  G3. The module's arguments are hard-thresholded at 0.5 by `Value.unflat`
      before exact execution, so during soft search a module candidate sees a
      Boolean rounding of the mixture, not the mixture.
  G4. The soft-mixture WEIGHT on a module candidate does still get gradient
      (dL/dw_i = z_i . dL/dy), so *selecting* a module is learnable even though
      *training through* one is not.
"""
from __future__ import annotations

import json
import sys

import torch

from tcn.types import BOOL, Value, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate
from tcn.learning import SoftProgram, relaxed


def half_adder(r):
    return Program(
        (("a", BOOL), ("b", BOOL)),
        (
            Node("s", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), selected=0),
            Node("c", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), selected=0),
        ),
        (("sum", "s"), ("carry", "c")),
    ).validate(r)


def main():
    out = {}
    r = Registry()
    ha = half_adder(r)
    name = r.register_module(ha)
    mop = r.resolve(name, (BOOL, BOOL))
    xop = r.resolve("xor", (BOOL, BOOL))

    # G1 -----------------------------------------------------------------
    a = torch.tensor([[0.7], [0.2]], requires_grad=True)
    b = torch.tensor([[0.4], [0.9]], requires_grad=True)
    ym = relaxed(r, mop, [a, b])
    yx = relaxed(r, xop, [a, b])
    out["G1_module_output_detached"] = dict(
        module_gradient_declaration=mop.gradient,
        module_output_requires_grad=bool(ym.requires_grad),
        primitive_output_requires_grad=bool(yx.requires_grad),
    )

    # G3 -----------------------------------------------------------------
    probe = {}
    for va, vb in ((0.49, 0.51), (0.51, 0.51), (0.99, 0.01)):
        y = relaxed(r, mop, [torch.tensor([[va]]), torch.tensor([[vb]])])
        probe[f"{va},{vb}"] = y.flatten().tolist()
    out["G3_arguments_hard_thresholded"] = dict(
        outputs=probe,
        note="Value.unflat maps bool fields with `next(it) >= .5`; the module is a step function of its soft arguments",
    )

    # G2 / G4 -------------------------------------------------------------
    # u is a learned BOOL node; its ONLY consumer is the module call.
    prog = Program(
        (("a", BOOL), ("b", BOOL)),
        (
            Node("u", BOOL, tuple(Candidate(r.resolve(n, (BOOL, BOOL)), ("a", "b"))
                                  for n in ("and", "or", "xor")), "core", 1),
            Node("call", product(BOOL, BOOL),
                 (Candidate(mop, ("u", "b")), Candidate(r.resolve("tuple", (BOOL, BOOL)), ("u", "b"))),
                 "core", 2),
        ),
        (("o", "call"),),
    ).validate(r)
    m = SoftProgram(prog, r)
    inputs = {"a": torch.tensor([[1.0], [0.0], [1.0], [0.0]]),
              "b": torch.tensor([[1.0], [1.0], [0.0], [0.0]])}

    # (i) with both candidates live at the call node
    o, _ = m(inputs)
    loss = o["o"].square().mean()
    g = torch.autograd.grad(loss, list(m.choices), allow_unused=True, retain_graph=True)
    mixed = [None if x is None else float(x.abs().sum()) for x in g]

    # (ii) with the call node frozen onto the MODULE candidate
    m2 = SoftProgram(prog, r)
    m2.freeze("call", 0)
    o2, _ = m2(inputs)
    loss2 = o2["o"].square().mean()
    active = [p for p in m2.parameters() if p.requires_grad]
    g2 = torch.autograd.grad(loss2, active, allow_unused=True) if loss2.requires_grad else None
    out["G2_upstream_starvation"] = dict(
        grad_abs_sum_u_when_mixture=mixed[0],
        grad_abs_sum_call_when_mixture=mixed[1],
        loss_requires_grad_after_freezing_call_to_module=bool(loss2.requires_grad),
        grads_after_freezing=None if g2 is None else [None if x is None else float(x.abs().sum()) for x in g2],
        note=("freezing the only consumer onto a module leaves the upstream node with no gradient path; "
              "Crystallizer.try_freeze detects this as 'disconnected remaining region' and rolls back"),
    )
    out["G4_choice_weight_still_learnable"] = dict(
        grad_on_call_node_logits=mixed[1],
        note="the mixture weight over a frozen candidate receives gradient even though the candidate's output does not",
    )

    print(json.dumps(out, indent=2))
    with open(sys.argv[1] if len(sys.argv) > 1 else "gradient_probe.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
