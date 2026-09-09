"""Does the discrete reference *prefer* the cheaper program, or just find one?

The tight scaffold answers "can enumeration select the module" (yes -- all 144
of its solutions use one). It does not answer "does enumeration select the
module *because* it is cheaper", because there is nothing else to select.

`tcn.search.enumerate_fit` appends every conforming selection and returns
`found[0]`: it ranks by enumeration order, not by description bits or execution
cost. This demonstrates that directly on a scaffold where a module call and a
primitive compute the same function, and where the two differ in size.
"""
import json
from pathlib import Path
from tcn.types import BOOL, Value
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate, Signal
from tcn.search import enumerate_fit, space_size
import common as C

r = Registry()
# a one-gate module whose call is semantically identical to the primitive `and`
body = Program((("a", BOOL), ("b", BOOL)),
               (Node("g", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), "core", 1, 0),),
               (("out", "g"),)).validate(r)
name = r.register_module(body)
mop = r.resolve(name, (BOOL, BOOL))
prim = r.resolve("and", (BOOL, BOOL))

out = {}
for order, label in (((Candidate(prim, ("a", "b")), Candidate(mop, ("a", "b"))), "primitive first"),
                     ((Candidate(mop, ("a", "b")), Candidate(prim, ("a", "b"))), "module first")):
    prog = Program((("a", BOOL), ("b", BOOL)),
                   (Node("y", BOOL, order, "core", 1),), (("out", "y"),)).validate(r)
    ex = [{"inputs": {"a": Value.of(BOOL, a), "b": Value.of(BOOL, b)},
           "targets": {"out": Value.of(BOOL, a and b)}}
          for a in (False, True) for b in (False, True)]
    sig = (Signal("y", "out", ("core",), BOOL, "bce"),)
    res = enumerate_fit(prog, ex, sig, registry=r, tolerance=.001)
    chosen = prog.nodes[0].candidates[res.selections["y"]].operator.name
    hard = prog.harden(res.selections).validate(r)
    out[label] = {
        "solved": res.solved, "unique": res.unique, "space_size": space_size(prog),
        "chosen": chosen[:20],
        "chosen_is_module": chosen.startswith("module:"),
        "description_bits_of_choice": hard.description_bits(r),
        "candidate_costs": [c.operator.cost for c in prog.nodes[0].candidates],
    }
out["conclusion"] = ("enumerate_fit returns found[0]: it ranks conforming programs by "
                     "enumeration order, not by description bits or execution cost, so it "
                     "selects whichever of two equivalent candidates is declared first")
Path("enumerate_preference.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
