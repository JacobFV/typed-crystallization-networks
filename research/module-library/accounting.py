"""Cost accounting for the composed chain, against the flat equivalent.

Three questions, all measured on programs that are built and executed here
rather than estimated:

1. **Description size.** The composed program with every module internal charged
   transitively, against the same computation written out flat.
   `Program.description_bits(registry)` is the shipped measure: the serialized
   size of the program, plus each distinct module definition once. Section 4's
   rule is that a definition is counted once and every call site individually,
   so a wider region must grow the flat program and leave the composed one flat.
2. **Execution cost per call site.** `Program.execution_cost` charges a module
   call its whole body, and `map` charges it once per element of the set. A call
   and its inlined body should pay equally; any difference is the plumbing that
   reaches the module's interface, and it is reported separately.
3. **The crossover.** `research/abstraction-preference` measured description
   size crossing over at 2 call sites for a 3-gate body. This re-measures the
   same crossover with the chain's own bodies over a 192-byte observation.

The flat side is produced by a **generic inliner** that walks a frozen module's
selected candidates and re-emits them into the caller, recursing through nested
module calls. It is not a hand-written lookalike: every flat program here is
checked to agree with its composed counterpart on real episodes before its cost
is reported, so the comparison is between two spellings of one function.

Caveat stated up front, because it is load-bearing: `description_bits` is the
serialized length of the program's JSON, not its learned content
(`research/FINDINGS.md` section 3). A call site carries the module operator's
full input and output type dictionaries, and at R=8 the observation type alone
is a 192-field product, so a call site is expensive in this metric for reasons
that have nothing to do with information. Node counts are reported beside every
bit count so the reader can see which effect is which.
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "discrete-perception"))

import chain as C  # noqa: E402
from tcn.library import Library  # noqa: E402
from tcn.types import Value  # noqa: E402

R = 8
LIBRARY = HERE / "library"


# --------------------------------------------------------------------------
# a generic inliner: the flat spelling of any composed program
# --------------------------------------------------------------------------

def module_constants(registry, module, prefix):
    """Every constant the module and its callees need, renamed per call site."""
    out = [(f"{prefix}_{k}", v) for k, v in module.constants]
    for node in module.nodes:
        candidate = node.candidates[node.selected]
        name = candidate.operator.name
        if name.startswith("module:"):
            out += module_constants(registry, registry.modules[name],
                                    f"{prefix}_{node.name}")
    return out


def inline(builder, registry, module, arguments, prefix):
    """Emit the module's selected body into `builder`; return its output names."""
    mapping = dict(zip((k for k, _ in module.inputs), arguments))
    mapping |= {k: f"{prefix}_{k}" for k, _ in module.constants}
    for node in module.nodes:
        candidate = node.candidates[node.selected]
        sources = [mapping[s] for s in candidate.sources]
        name = f"{prefix}_{node.name}"
        if candidate.operator.name.startswith("module:"):
            inner = registry.modules[candidate.operator.name]
            outs = inline(builder, registry, inner, sources, name)
            if len(outs) == 1:
                mapping[node.name] = outs[0]
            else:
                mapping[node.name] = builder.add(name, "tuple", outs)
        else:
            builder.add(name, candidate.operator.name, sources, out=node.output,
                        params=dict(candidate.operator.parameters))
            mapping[node.name] = name
    return [mapping[v] for _, v in module.outputs]


def _fold(builder, flags, tag):
    current = flags[0]
    for i, f in enumerate(flags[1:]):
        current = builder.add(f"{tag}_or{i}", "or", [current, f])
    return current


# --------------------------------------------------------------------------
# the region predicate: composed against flat, at growing region size
# --------------------------------------------------------------------------

def composed_region(library, tile, policy="revalidate"):
    registry, aliases = library.load(["perception.foreground", "perception.edge"], policy=policy)
    edge_name = aliases["perception.edge"]
    keep = registry.register_module(C.keep_flag_module(registry))
    wrapper = registry.register_module(C.wrapper_over_edge(registry, R, edge_name))
    program = C.region_scaffold(registry, R, [wrapper], keep, tile).harden(
        {"mapped": 0, "region": 0}).pruned()
    return registry, program, wrapper, edge_name, aliases["perception.foreground"]


def flat_region(registry, wrapper, tile):
    """The same predicate with every module inlined and no sharing at all."""
    module = registry.modules[wrapper]
    offs = C.tile_offsets(R, tile)
    consts = tuple((f"off{d}", Value.of(C.IDX, d)) for d in offs)
    for d in offs:
        consts += tuple(module_constants(registry, module, f"c{d}"))
    builder = C.Builder(registry, (("rec", C.record_type(R)),), consts)
    flags = []
    for d in offs:
        arg = builder.add(f"arg{d}", "tuple", [f"off{d}", "rec"])
        outs = inline(builder, registry, module, [arg], f"c{d}")
        flags.append(builder.add(f"flag{d}", "project", [outs[0]], params={"index": 1}))
    return builder.program((("y", _fold(builder, flags, "any")),))


def agree(a, b, registry, tile, seeds=(0, 1)):
    for example in C.region_examples(seeds, R, "train", tile, seed=5):
        x = a.run(example["inputs"], registry=registry)[0]["y"].decoded
        y = b.run(example["inputs"], registry=registry)[0]["y"].decoded
        if bool(x) != bool(y):
            raise AssertionError(f"composed and flat disagree at tile {tile}")
    return True


def region_costs(library):
    rows = []
    for tile in (2, 3, 4):
        registry, composed, wrapper, _, _ = composed_region(library, tile)
        flat = flat_region(registry, wrapper, tile)
        agree(composed, flat, registry, tile)
        row = {"tile": f"{tile}x{tile}", "positions": len(C.tile_offsets(R, tile)),
               "composed_nodes": len(composed.nodes),
               "composed_description_bits": composed.description_bits(registry),
               "composed_own_bits": composed.description_bits(None),
               "composed_execution_cost": composed.execution_cost(registry),
               "flat_nodes": len(flat.nodes),
               "flat_description_bits": flat.description_bits(registry),
               "flat_execution_cost": flat.execution_cost(registry),
               "agree_on_episodes": True}
        row["description_ratio"] = row["composed_description_bits"] / row["flat_description_bits"]
        row["execution_ratio"] = row["composed_execution_cost"] / row["flat_execution_cost"]
        rows.append(row)
    return rows


# --------------------------------------------------------------------------
# the crossover: N call sites, abstracted against inlined
# --------------------------------------------------------------------------

def call_sites(library, module_reference, limit=8, policy="revalidate"):
    registry, aliases = library.load([module_reference], policy=policy)
    name = aliases[module_reference]
    module = registry.modules[name]
    rows = []
    positions = C.pixel_starts(R)
    for n in range(1, limit + 1):
        chosen = positions[:n]
        consts = tuple((f"at{p}", Value.of(C.IDX, p)) for p in chosen)
        b = C.Builder(registry, (("obs", C.bytes_type(R)),), consts)
        flags = []
        for p in chosen:
            b.add(f"rec{p}", "tuple", [f"at{p}", "obs"])
            b.add(f"call{p}", name, [f"rec{p}"])
            flags.append(b.add(f"flag{p}", "project", [f"call{p}"], params={"index": 1}))
        abstracted = b.program((("y", _fold(b, flags, "any")),))

        consts2 = consts
        for p in chosen:
            consts2 += tuple(module_constants(registry, module, f"c{p}"))
        b2 = C.Builder(registry, (("obs", C.bytes_type(R)),), consts2)
        flags = []
        for p in chosen:
            arg = b2.add(f"rec{p}", "tuple", [f"at{p}", "obs"])
            outs = inline(b2, registry, module, [arg], f"c{p}")
            flags.append(b2.add(f"flag{p}", "project", [outs[0]], params={"index": 1}))
        inlined = b2.program((("y", _fold(b2, flags, "any")),))

        pixels, _ = C.episode(0, R, split="test")
        observation = {"obs": Value.of(C.bytes_type(R), pixels)}
        assert (abstracted.run(observation, registry=registry)[0]["y"].decoded
                == inlined.run(observation, registry=registry)[0]["y"].decoded)
        rows.append({"call_sites": n,
                     "abstracted_nodes": len(abstracted.nodes),
                     "abstracted_bits": abstracted.description_bits(registry),
                     "abstracted_cost": abstracted.execution_cost(registry),
                     "inlined_nodes": len(inlined.nodes),
                     "inlined_bits": inlined.description_bits(registry),
                     "inlined_cost": inlined.execution_cost(registry)})
        rows[-1]["abstraction_pays"] = rows[-1]["abstracted_bits"] < rows[-1]["inlined_bits"]
    first = next((r["call_sites"] for r in rows if r["abstraction_pays"]), None)
    return {"module": module_reference, "body_nodes": len(module.nodes),
            "crossover_call_sites": first, "rows": rows}


# --------------------------------------------------------------------------
# transitive charging
# --------------------------------------------------------------------------

def transitive(library, policy="revalidate"):
    rows = []
    for name in library.names():
        entry = library.head(name)
        registry, aliases = library.load([name], policy=policy)
        module = registry.modules[aliases[name]]
        row = {"module": entry.reference, "nodes": len(module.nodes),
               "requires": list(entry.requires),
               "own_bits": module.description_bits(None),
               "transitive_bits": module.description_bits(registry),
               "execution_cost": module.execution_cost(registry)}
        row["callee_bits"] = row["transitive_bits"] - row["own_bits"]
        rows.append(row)
    return rows


def main():
    library = Library(LIBRARY)
    result = {"library": str(LIBRARY),
              "transitive_description": transitive(library),
              "region": region_costs(library),
              "crossover_foreground": call_sites(library, "perception.foreground"),
              "crossover_edge": call_sites(library, "perception.edge")}
    (HERE / "out").mkdir(parents=True, exist_ok=True)
    (HERE / "out" / "accounting.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
