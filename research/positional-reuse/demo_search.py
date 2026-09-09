"""Search chooses which crystallized module to apply at every position.

`shared_map_program` in the other scripts names the module by hand. Here the map
node's candidates come from `tcn.graph.legal_candidates` over the registry, so
nothing about positional reuse is wired in advance: the enumerator proposes one
`map` candidate per registered module, and `tcn.search.enumerate_fit` picks the
one that reproduces the whole dense probe of `generators/geometry`.

This is the part `legal_candidates` could not do before. Parameters are part of
an operator's contract, and the enumerator resolved every candidate with empty
parameters, so `project`, `map`, `filter` and `join` were unreachable -- the one
family that expresses recursive abstraction.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from demo_geometry import (BYTE, BYTES, CANDIDATE_BYTES, PIXEL_STARTS, RECORD_SET,
                           TRUE_PICK, TRUE_TRUTH, episode, module_scaffold, target_set)
from shared import IDX
from tcn.graph import Candidate, Node, Program, Signal, legal_candidates
from tcn.operators import Registry
from tcn.search import enumerate_fit
from tcn.types import Value, setof

# One correct sub-program and three plausible wrong ones. `TRUE_PICK` selects the
# renderer's background bytes and `TRUE_TRUTH` the `and` / `nand` combination.
VARIANTS = {
    "renderer_background": (TRUE_PICK, TRUE_TRUTH),
    "wrong_blue_byte": ((0, 1, 3), TRUE_TRUTH),
    "or_instead_of_and": (TRUE_PICK, (14, 7)),
    "unnegated": (TRUE_PICK, (8, 8)),
}


def variant(registry, picks, truths):
    scaffold = module_scaffold(registry, searched=True)
    sel = {}
    for node in scaffold.nodes:
        sel[node.name] = 0
    for name, pick in zip(("cmp_r", "cmp_g", "cmp_b"), picks):
        sel[name] = pick
    for name, t in zip(("rg", "foreground"), truths):
        sel[name] = t
    return scaffold.harden(sel)


def main():
    r = Registry()
    names = {}
    for label, (picks, truths) in VARIANTS.items():
        names[label] = r.register_module(variant(r, picks, truths))
    assert len(set(names.values())) == len(VARIANTS), "variants must be distinct modules"

    positions = Value.of(setof(IDX, len(PIXEL_STARTS)), PIXEL_STARTS)
    empty = Value.of(setof(BYTES, 1), ())
    held = Node("held", r.resolve("insert", (empty.type, BYTES)).output,
                (Candidate(r.resolve("insert", (empty.type, BYTES)), ("empty", "observation")),),
                "core", 1, 0)
    records_op = r.resolve("pair", (positions.type, held.output))
    records = Node("records", records_op.output, (Candidate(records_op, ("positions", "held")),),
                   "core", 2, 0)

    # The whole point: the candidates are enumerated, not named.
    proposed = legal_candidates(r, ["map"], {"records": records.output}, RECORD_SET, arities=(1,))
    modules = [dict(c.operator.parameters)["module"] for c in proposed]
    print(f"legal_candidates proposed {len(proposed)} map candidates over "
          f"{len(r.modules)} registered modules")
    mapped = Node("mapped", RECORD_SET, proposed, "core", 3, None)

    program = Program((("observation", BYTES),), (held, records, mapped), (("y", "mapped"),),
                      (("positions", positions), ("empty", empty)),
                      input_depths=(("observation", 0),)).validate(r)

    examples = []
    for seed in range(3):
        data, labels = episode(seed)
        examples.append({"inputs": {"observation": data},
                         "targets": {"object_ids_foreground": target_set(labels)}})
    signals = (Signal("mapped", "object_ids_foreground", ("core",), RECORD_SET, "mse"),)
    result = enumerate_fit(program, examples, signals, r, tolerance=1e-6)
    picked = modules[result.selections["mapped"]] if result.solved else None
    label = next((k for k, v in names.items() if v == picked), None)
    print(f"solved={result.solved} unique={result.unique} picked={label} "
          f"space={result.space_size} seconds={result.seconds:.2f}")

    held = [(seed, *episode(seed, split="test")) for seed in range(100, 110)]

    def worst(candidate_index, rows):
        chosen = program.harden({"mapped": candidate_index})
        return max(max(abs(a - b) for a, b in
                       zip(chosen.run({"observation": data}, registry=r)[0]["y"].flat(),
                           target_set(labels).flat()))
                   for _, data, labels in rows)

    train_rows = [(s, *episode(s)) for s in range(3)]
    table = {}
    for i, module in enumerate(modules):
        variant_label = next(k for k, v in names.items() if v == module)
        table[variant_label] = {"train_max_error": worst(i, train_rows),
                                "held_out_max_error": worst(i, held)}
        print(f"  {variant_label:22s} train {table[variant_label]['train_max_error']:5.1f}  "
              f"held-out {table[variant_label]['held_out_max_error']:5.1f}")
    held_errors = [table[label]["held_out_max_error"]] if label else []

    out = {"registered_modules": {k: v for k, v in names.items()},
           "proposed_map_candidates": len(proposed),
           "search": result.to_dict(), "picked": label, "variants": table,
           "held_out_pixels": len(held) * len(PIXEL_STARTS),
           "held_out_max_error": max(held_errors) if held_errors else None}
    path = pathlib.Path(__file__).parent / "demo_search.json"
    path.write_text(json.dumps(out, indent=1, default=str))
    print("wrote", path)


if __name__ == "__main__":
    main()
