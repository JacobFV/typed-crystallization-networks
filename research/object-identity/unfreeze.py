"""Where exactly the gradient dies in this scaffold, and which of two causes it is.

The corrected probe says `shifted` (the offset) and `thr` (the threshold) both
have `grad = None`.  There are two candidate explanations and they have
different consequences:

  A. `pack` declares `gradient="none"`, so `relaxed` routes it through
     `exact_tensor`, which detaches.  Everything computed from a pixel's
     arithmetic value is then a constant of the parameters.
  B. `SoftProgram.forward` treats every node with `selected is not None` as
     FROZEN and evaluates it with `exact_tensor(...).detach()`.  `Builder.add`
     (and `tcn.scaffold`) set `selected = 0` on every single-candidate node, so
     every deterministic node in any hand-written scaffold detaches -- whatever
     its operator's declared gradient class.

B is the more consequential one, because it is not a property of the operator
algebra at all.  This separates them: the same program with `selected = None`
restored on the deterministic nodes is the identical search space (a node with
one candidate contributes a factor of 1), so any change in the gradients is B.
"""
from __future__ import annotations
import dataclasses, json, sys, pathlib, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "discrete-perception"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from common import gradient_arm, report, summarise
from rung4_segment import (collinear_scaffold, choice_gradients, dump, same_examples,
                           same_signals)
from tcn.operators import Registry
from tcn.search import space_size


def unfreeze(program):
    """Same program, same space; deterministic nodes no longer marked selected."""
    nodes = tuple(dataclasses.replace(n, selected=None) if len(n.candidates) == 1 else n
                  for n in program.nodes)
    return dataclasses.replace(program, nodes=nodes)


def main():
    R = 8
    r = Registry()
    train = same_examples(range(8), R, "train", None, seed=3)
    held = same_examples(range(100, 108), R, "test", None, seed=4)
    sig = same_signals()
    base = lambda: collinear_scaffold(r, R)
    free = lambda: unfreeze(collinear_scaffold(r, R))
    out = {"space_size_as_built": space_size(base()),
           "space_size_unfrozen": space_size(free())}
    report("space size, as built / with deterministic nodes unfrozen",
           f"{out['space_size_as_built']} / {out['space_size_unfrozen']}")
    out["as_built"] = choice_gradients(base, train[:64], sig, r)
    out["unfrozen"] = choice_gradients(free, train[:64], sig, r)
    report("as built   |grad|_1", json.dumps(out["as_built"]["choice_grad_l1"]))
    report("unfrozen   |grad|_1",
           json.dumps({k: v for k, v in out["unfrozen"]["choice_grad_l1"].items()
                       if k in ("shifted", "thr", "m1", "same")}))
    dump("unfreeze", out)
    rows = gradient_arm(free, tuple(range(4)), train, sig, r, steps=400, lr=.15,
                        init_noise=.5, label="unfrozen", tolerance=1e-6, held=held)
    out["gradient_unfrozen"] = {"rows": rows, "summary": summarise(rows),
                                "held_exact": sum(1 for x in rows if x.get("held_err") == 0.)}
    report("unfrozen gradient successes / held-out exact",
           f"{out['gradient_unfrozen']['summary']['successes']}/{len(rows)} / "
           f"{out['gradient_unfrozen']['held_exact']}")
    dump("unfreeze", out)


if __name__ == "__main__":
    main()
