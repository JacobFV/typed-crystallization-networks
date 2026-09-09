"""Re-verify track 5's cost claims independently, after F1/F2 were fixed.

Track 5 measured, against the pre-fix tree:
  * a one-output module call resolves to `product(BOOL)`, forcing a `project`
    node at every call site (F1);
  * a module candidate costs 16-98x a primitive candidate inside the soft graph
    because `exact_tensor` re-validated the module body on every batch row (F2);
  * description-size crossover for a 4-gate body at 4 call sites, execution-cost
    crossover never.

Nothing here trusts the reported "after" numbers: every quantity is measured
again from the current tree, with the same definitions track 5 used.

M-checks   the F1 fix (a single-output module is BOOL-typed and needs no
           projection; a two-output module still forms a product).
C-checks   candidate-evaluation cost of a module against a primitive, over the
           batch sizes the experiments actually use.
X-checks   description-bit and execution-cost crossover against inlining, for
           bodies of several sizes and call-site counts.
L-checks   batch-one exact latency of a flat program against its abstracted
           equivalent.
V-check    validation memoization actually holds (the F2 fix), measured as the
           per-row cost of a module call against the number of rows.
"""
from __future__ import annotations

import itertools
import json
import statistics
import time
from pathlib import Path

import torch

from tcn.types import BOOL, Value, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate
from tcn.learning import exact_tensor, relaxed
from tcn.runtime import benchmark

HERE = Path(__file__).parent


# --------------------------------------------------------------------------
# small hand-built frozen programs used as module bodies
# --------------------------------------------------------------------------
def gate(r, name, out, *srcs):
    op = r.resolve(name, tuple(BOOL for _ in srcs))
    return Node(out, BOOL, (Candidate(op, srcs),), "core", 0, 0)


def with_depths(nodes):
    """Assign the minimal legal depth to a straight-line chain of nodes."""
    depth = {}
    out = []
    for n in nodes:
        d = max((depth.get(s, 0) for s in n.candidates[0].sources), default=0) + 1
        depth[n.name] = d
        out.append(Node(n.name, n.output, n.candidates, n.region, d, 0))
    return tuple(out)


def maj3_body(r):
    """MAJ3(a,b,c) = or(and(a,b), and(c, xor(a,b))) -- the 4-gate minimum."""
    ns = with_depths([
        gate(r, "and", "g0", "a", "b"),
        gate(r, "xor", "g1", "a", "b"),
        gate(r, "and", "g2", "c", "g1"),
        gate(r, "or", "g3", "g0", "g2"),
    ])
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), ns, (("out", "g3"),)).validate(r)


def chain_body(r, k):
    """A k-gate single-output body on three inputs (a synthetic size knob)."""
    nodes = [gate(r, "xor", "g0", "a", "b")]
    prev = "g0"
    for i in range(1, k):
        src = ("c" if i % 2 else "a")
        nodes.append(gate(r, ("and", "or", "xor")[i % 3], f"g{i}", prev, src))
        prev = f"g{i}"
    ns = with_depths(nodes)
    return Program((("a", BOOL), ("b", BOOL), ("c", BOOL)), ns, (("out", prev),)).validate(r)


def half_adder_body(r):
    ns = with_depths([gate(r, "xor", "s", "a", "b"), gate(r, "and", "c", "a", "b")])
    return Program((("a", BOOL), ("b", BOOL)), ns, (("sum", "s"), ("carry", "c"))).validate(r)


# --------------------------------------------------------------------------
# M: the F1 fix
# --------------------------------------------------------------------------
def m_checks():
    r = Registry()
    one = r.register_module(maj3_body(r))
    two = r.register_module(half_adder_body(r))
    op1 = r.resolve(one, (BOOL, BOOL, BOOL))
    op2 = r.resolve(two, (BOOL, BOOL))
    # the call actually computes the right thing and returns a plain BOOL value
    vals = {}
    for a, b, c in itertools.product((False, True), repeat=3):
        v = r.exact(op1, [Value.of(BOOL, a), Value.of(BOOL, b), Value.of(BOOL, c)])
        vals[(a, b, c)] = v.decoded
    correct = all(vals[k] == (sum(k) >= 2) for k in vals)
    return {
        "M1_single_output_type": str(op1.output.to_dict()),
        "M1_is_bool": op1.output == BOOL,
        "M1_projection_needed": op1.output != BOOL,
        "M1_exact_correct": correct,
        "M2_two_output_type": str(op2.output.to_dict()),
        "M2_is_product": op2.output == product(BOOL, BOOL),
        "M3_module_cost_equals_body_cost": (op1.cost, maj3_body(r).execution_cost(r)),
        "M4_module_gradient": op1.gradient,
    }


# --------------------------------------------------------------------------
# C: candidate evaluation cost, module vs primitive
# --------------------------------------------------------------------------
def c_checks(batches=(8, 16, 32, 64, 128), reps=30):
    r = Registry()
    name = r.register_module(maj3_body(r))
    mop = r.resolve(name, (BOOL, BOOL, BOOL))
    xop = r.resolve("xor", (BOOL, BOOL))
    rows = []
    for batch in batches:
        xs = [torch.rand(batch, 1) for _ in range(3)]
        # warm up both paths first so the first-call cost is not attributed
        exact_tensor(r, mop, xs)
        relaxed(r, xop, xs[:2])
        tm, tp = [], []
        for _ in range(reps):
            t = time.perf_counter(); exact_tensor(r, mop, xs); tm.append(time.perf_counter() - t)
            t = time.perf_counter(); relaxed(r, xop, xs[:2]); tp.append(time.perf_counter() - t)
        m = statistics.median(tm) * 1e6
        p = statistics.median(tp) * 1e6
        rows.append({"batch": batch, "primitive_us": round(p, 2), "module_us": round(m, 2),
                     "ratio": round(m / p, 1), "module_us_per_row": round(m / batch, 2)})
    return rows


def v_check(batches=(1, 8, 64, 512), reps=20):
    """Per-row cost should be flat in batch if validation is memoized (F2)."""
    r = Registry()
    name = r.register_module(maj3_body(r))
    mop = r.resolve(name, (BOOL, BOOL, BOOL))
    out = []
    for batch in batches:
        xs = [torch.rand(batch, 1) for _ in range(3)]
        exact_tensor(r, mop, xs)
        ts = []
        for _ in range(reps):
            t = time.perf_counter(); exact_tensor(r, mop, xs); ts.append(time.perf_counter() - t)
        out.append({"batch": batch, "us_per_row": round(statistics.median(ts) * 1e6 / batch, 2)})
    return out


# --------------------------------------------------------------------------
# X: crossover against inlining
# --------------------------------------------------------------------------
def abstracted_program(r, name, body_inputs, sites):
    """`sites` independent module calls, combined pairwise by xor."""
    inputs = tuple((f"x{i}", BOOL) for i in range(body_inputs * sites))
    nodes = []
    for s in range(sites):
        srcs = tuple(f"x{body_inputs*s+j}" for j in range(body_inputs))
        op = r.resolve(name, tuple(BOOL for _ in srcs))
        nodes.append(Node(f"call{s}", BOOL, (Candidate(op, srcs),), "core", 0, 0))
    prev = "call0"
    for s in range(1, sites):
        nodes.append(gate(r, "xor", f"comb{s}", prev, f"call{s}"))
        prev = f"comb{s}"
    return Program(inputs, with_depths(nodes), (("out", prev),)).validate(r)


def inlined_program(r, body, sites):
    """The same computation with the body's gates copied at every site."""
    bi = len(body.inputs)
    inputs = tuple((f"x{i}", BOOL) for i in range(bi * sites))
    nodes = []
    for s in range(sites):
        rename = {k: f"x{bi*s+j}" for j, (k, _) in enumerate(body.inputs)}
        for n in body.nodes:
            c = n.candidates[n.selected or 0]
            srcs = tuple(rename.get(x, f"s{s}_{x}") for x in c.sources)
            nodes.append(gate(r, c.operator.name, f"s{s}_{n.name}", *srcs))
            rename.setdefault(n.name, f"s{s}_{n.name}")
        rename[n.name] = f"s{s}_{n.name}"
    outs = [f"s{s}_{body.outputs[0][1]}" for s in range(sites)]
    prev = outs[0]
    for s in range(1, sites):
        nodes.append(gate(r, "xor", f"comb{s}", prev, outs[s]))
        prev = f"comb{s}"
    return Program(inputs, with_depths(nodes), (("out", prev),)).validate(r)


def x_checks(body_sizes=(3, 4, 8), site_counts=(1, 2, 3, 4, 6, 8)):
    out = {}
    for k in body_sizes:
        rows = []
        r0 = Registry()
        body = chain_body(r0, k)
        for sites in site_counts:
            r = Registry()
            name = r.register_module(chain_body(r, k))
            a = abstracted_program(r, name, 3, sites)
            f = inlined_program(r, body, sites)
            rows.append({
                "sites": sites,
                "module_bits": a.description_bits(r), "flat_bits": f.description_bits(r),
                "bits_ratio": round(a.description_bits(r) / f.description_bits(r), 3),
                "module_cost": a.execution_cost(r), "flat_cost": f.execution_cost(r),
                "cost_ratio": round(a.execution_cost(r) / f.execution_cost(r), 3),
            })
        out[f"body_{k}_gates"] = rows
    return out


def equivalence_check():
    """The abstracted and inlined forms must compute the same function."""
    r = Registry()
    body = chain_body(r, 4)
    name = r.register_module(chain_body(r, 4))
    a = abstracted_program(r, name, 3, 2)
    f = inlined_program(r, body, 2)
    same = True
    for bits in itertools.product((False, True), repeat=6):
        ins = {f"x{i}": Value.of(BOOL, v) for i, v in enumerate(bits)}
        oa, _ = a.run(ins, registry=r)
        of, _ = f.run(ins, registry=r)
        same &= oa["out"].decoded == of["out"].decoded
    return same


# --------------------------------------------------------------------------
# L: batch-one exact latency
# --------------------------------------------------------------------------
def l_checks(sites=2, repeats=200):
    r = Registry()
    body = chain_body(r, 4)
    name = r.register_module(chain_body(r, 4))
    a = abstracted_program(r, name, 3, sites)
    f = inlined_program(r, body, sites)
    ins_a = {k: Value.of(BOOL, False) for k, _ in a.inputs}
    ins_f = {k: Value.of(BOOL, False) for k, _ in f.inputs}
    ba = benchmark(a, [ins_a] * repeats, registry=r)
    bf = benchmark(f, [ins_f] * repeats, registry=r)
    return {"abstracted": ba, "flat": bf}


def main():
    out = {
        "M_f1_fix": m_checks(),
        "C_candidate_cost": c_checks(),
        "V_per_row_cost": v_check(),
        "X_crossover": x_checks(),
        "X_equivalence_verified": equivalence_check(),
        "L_latency": l_checks(),
    }
    (HERE / "costs.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
