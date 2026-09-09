"""Empirical check of the ARCHITECTURE.md section-4 accounting claim.

Claim under test (ARCHITECTURE.md sec. 4):

    "Its internal description, precision, latency, and storage still count
     toward complexity/cost; a call is not free. Sharing counts a module
     definition once plus its call sites, with execution cost charged per use."

Three sub-claims:
  A1. description_bits counts the module definition ONCE regardless of the
      number of call sites, and counts each call site.
  A2. execution_cost is charged PER USE (linear in call sites).
  A3. Both are charged TRANSITIVELY through nested modules.

Also probes two mechanical facts that matter for the reuse experiment:
  M1. what output type a single-output module call has;
  M2. how expensive a module candidate is to evaluate during soft search.
"""
from __future__ import annotations
import json
import time
import sys

from tcn.types import BOOL, Value, product
from tcn.operators import Registry
from tcn.graph import Program, Node, Candidate


def half_adder(r: Registry) -> Program:
    """HA(a,b) -> (sum, carry) = (a xor b, a and b): a frozen 2-node program."""
    return Program(
        (("a", BOOL), ("b", BOOL)),
        (
            Node("s", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), selected=0),
            Node("c", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), selected=0),
        ),
        (("sum", "s"), ("carry", "c")),
    ).validate(r)


def unary_not(r: Registry) -> Program:
    return Program(
        (("a", BOOL),),
        (Node("y", BOOL, (Candidate(r.resolve("not", (BOOL,)), ("a",)),), selected=0),),
        (("out", "y"),),
    ).validate(r)


def caller_with_n_sites(r: Registry, module_name: str, n: int) -> Program:
    """A program with n independent call sites of the same frozen module."""
    op = r.resolve(module_name, (BOOL, BOOL))
    nodes = tuple(
        Node(f"call{i}", op.output, (Candidate(op, ("a", "b")),), depth=1, selected=0)
        for i in range(n)
    )
    outs = tuple((f"o{i}", f"call{i}") for i in range(n))
    return Program((("a", BOOL), ("b", BOOL)), nodes, outs).validate(r)


def flat_equivalent(r: Registry, n: int) -> Program:
    """The same n outputs written out inline, without any module."""
    nodes = []
    for i in range(n):
        nodes.append(Node(f"s{i}", BOOL, (Candidate(r.resolve("xor", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0))
        nodes.append(Node(f"c{i}", BOOL, (Candidate(r.resolve("and", (BOOL, BOOL)), ("a", "b")),), depth=1, selected=0))
        nodes.append(Node(f"t{i}", product(BOOL, BOOL), (Candidate(r.resolve("tuple", (BOOL, BOOL)), (f"s{i}", f"c{i}")),), depth=2, selected=0))
    outs = tuple((f"o{i}", f"t{i}") for i in range(n))
    return Program((("a", BOOL), ("b", BOOL)), tuple(nodes), outs).validate(r)


def main():
    out = {}
    r = Registry()
    ha = half_adder(r)
    ha_name = r.register_module(ha)

    # ---- A1 / A2: scaling in the number of call sites -------------------
    rows = []
    for n in (1, 2, 4, 8):
        p = caller_with_n_sites(r, ha_name, n)
        flat = flat_equivalent(r, n)
        rows.append(
            dict(
                call_sites=n,
                bits_with_registry=p.description_bits(r),
                bits_without_registry=p.description_bits(),
                module_definition_bits=ha.description_bits(),
                exec_cost=p.execution_cost(r),
                flat_bits=flat.description_bits(r),
                flat_exec_cost=flat.execution_cost(r),
            )
        )
    out["call_site_scaling"] = rows

    # Derived checks.
    b = {row["call_sites"]: row["bits_with_registry"] for row in rows}
    c = {row["call_sites"]: row["exec_cost"] for row in rows}
    defbits = ha.description_bits()
    # definition counted once: bits(n) - bits(1) must not contain another copy
    # of the definition, i.e. the increment per extra call site must be far
    # smaller than the definition itself and must be constant.
    increments = [b[2] - b[1], (b[4] - b[2]) / 2, (b[8] - b[4]) / 4]
    out["A1_definition_counted_once"] = dict(
        definition_bits=defbits,
        per_call_site_increment=increments,
        increment_is_constant=max(increments) - min(increments) < 1e-9,
        increment_much_smaller_than_definition=max(increments) < defbits,
        bits_without_registry_excludes_definition=all(
            row["bits_with_registry"] - row["bits_without_registry"] == defbits for row in rows
        ),
    )
    out["A2_execution_cost_per_use"] = dict(
        costs=c,
        module_unit_cost=ha.execution_cost(r),
        linear_in_call_sites=all(c[n] == n * c[1] for n in (2, 4, 8)),
    )

    # ---- A3: transitive accounting through nesting -----------------------
    r2 = Registry()
    inner = unary_not(r2)
    inner_name = r2.register_module(inner)
    inner_op = r2.resolve(inner_name, (BOOL,))
    # middle module calls inner twice, then combines
    middle = Program(
        (("a", BOOL), ("b", BOOL)),
        (
            Node("na", inner_op.output, (Candidate(inner_op, ("a",)),), depth=1, selected=0),
            Node("nb", inner_op.output, (Candidate(inner_op, ("b",)),), depth=1, selected=0),
            Node("pa", BOOL, (Candidate(r2.resolve("project", (inner_op.output,), BOOL, {"index": 0}), ("na",)),), depth=2, selected=0),
            Node("pb", BOOL, (Candidate(r2.resolve("project", (inner_op.output,), BOOL, {"index": 0}), ("nb",)),), depth=2, selected=0),
            Node("y", BOOL, (Candidate(r2.resolve("and", (BOOL, BOOL)), ("pa", "pb")),), depth=3, selected=0),
        ),
        (("out", "y"),),
    ).validate(r2)
    middle_name = r2.register_module(middle)
    middle_op = r2.resolve(middle_name, (BOOL, BOOL))
    top = Program(
        (("a", BOOL), ("b", BOOL)),
        (Node("m", middle_op.output, (Candidate(middle_op, ("a", "b")),), depth=1, selected=0),),
        (("out", "m"),),
    ).validate(r2)
    out["A3_transitive"] = dict(
        inner_definition_bits=inner.description_bits(),
        middle_definition_bits=middle.description_bits(),
        middle_bits_with_registry=middle.description_bits(r2),
        top_bits_no_registry=top.description_bits(),
        top_bits_with_registry=top.description_bits(r2),
        top_minus_own=top.description_bits(r2) - top.description_bits(),
        expected_if_transitive=middle.description_bits() + inner.description_bits(),
        transitive_description=(
            top.description_bits(r2) - top.description_bits()
            == middle.description_bits() + inner.description_bits()
        ),
        inner_exec_cost=inner.execution_cost(r2),
        middle_exec_cost=middle.execution_cost(r2),
        top_exec_cost=top.execution_cost(r2),
        transitive_cost=top.execution_cost(r2) >= middle.execution_cost(r2),
    )

    # ---- A4: a module reachable from two levels is still counted once ----
    r5 = Registry()
    inner5 = half_adder(r5)
    inner5_name = r5.register_module(inner5)
    iop = r5.resolve(inner5_name, (BOOL, BOOL))
    wrapper = Program(
        (("a", BOOL), ("b", BOOL)),
        (Node("k", iop.output, (Candidate(iop, ("a", "b")),), depth=1, selected=0),),
        (("o", "k"),),
    ).validate(r5)
    wrapper_name = r5.register_module(wrapper)
    wop = r5.resolve(wrapper_name, (BOOL, BOOL))
    # top calls the wrapper AND the inner module directly
    both = Program(
        (("a", BOOL), ("b", BOOL)),
        (
            Node("u", wop.output, (Candidate(wop, ("a", "b")),), depth=1, selected=0),
            Node("v", iop.output, (Candidate(iop, ("a", "b")),), depth=1, selected=0),
        ),
        (("o1", "u"), ("o2", "v")),
    ).validate(r5)
    out["A4_shared_across_levels"] = dict(
        library_bits_charged=both.description_bits(r5) - both.description_bits(),
        wrapper_definition_bits=wrapper.description_bits(),
        inner_definition_bits=inner5.description_bits(),
        expected_once_each=wrapper.description_bits() + inner5.description_bits(),
        counted_once_each=(
            both.description_bits(r5) - both.description_bits()
            == wrapper.description_bits() + inner5.description_bits()
        ),
    )

    # ---- A5: unselected module candidates also pay their definition ------
    r6 = Registry()
    inner6 = half_adder(r6)
    n6 = r6.register_module(inner6)
    mop6 = r6.resolve(n6, (BOOL, BOOL))
    top6 = r6.resolve("tuple", (BOOL, BOOL))
    soft = Program(
        (("a", BOOL), ("b", BOOL)),
        (Node("n", product(BOOL, BOOL), (Candidate(top6, ("a", "b")), Candidate(mop6, ("a", "b"))), depth=1, selected=0),),
        (("o", "n"),),
    ).validate(r6)
    out["A5_offered_but_unselected"] = dict(
        selected_index=0,
        selected_operator="tuple",
        library_bits_charged=soft.description_bits(r6) - soft.description_bits(),
        module_definition_bits=inner6.description_bits(),
        note="description_bits scans all candidates, so merely offering a module in the search space already charges its definition",
        exec_cost_of_selected_tuple=soft.execution_cost(r6),
    )

    # ---- M1: output type of a single-output module call ------------------
    r3 = Registry()
    nm = unary_not(r3)
    nm_name = r3.register_module(nm)
    nm_op = r3.resolve(nm_name, (BOOL,))
    out["M1_single_output_module_call_type"] = dict(
        module_declared_output=BOOL.to_dict(),
        call_node_output=nm_op.output.to_dict(),
        is_bool=nm_op.output == BOOL,
        note="a call always yields product(*outputs); a 1-output module is not a drop-in BOOL candidate",
    )

    # ---- M2: relative evaluation cost of a module candidate --------------
    import torch
    from tcn.learning import relaxed, exact_tensor

    r4 = Registry()
    ha2 = half_adder(r4)
    ha2_name = r4.register_module(ha2)
    mop = r4.resolve(ha2_name, (BOOL, BOOL))
    xop = r4.resolve("xor", (BOOL, BOOL))
    a = torch.rand(32, 1)
    bt = torch.rand(32, 1)
    reps = 20
    t0 = time.perf_counter()
    for _ in range(reps):
        relaxed(r4, xop, [a, bt])
    t_prim = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        relaxed(r4, mop, [a.round(), bt.round()])
    t_mod = (time.perf_counter() - t0) / reps
    out["M2_candidate_evaluation_cost"] = dict(
        batch=32,
        primitive_xor_seconds=t_prim,
        module_call_seconds=t_mod,
        slowdown=t_mod / t_prim,
        note="module candidates have gradient='none' -> exact_tensor -> per-row Python execution with full Program.validate on every row",
    )

    print(json.dumps(out, indent=2, default=str))
    with open(sys.argv[1] if len(sys.argv) > 1 else "accounting.json", "w") as f:
        json.dump(out, f, indent=2, default=str)


if __name__ == "__main__":
    main()
