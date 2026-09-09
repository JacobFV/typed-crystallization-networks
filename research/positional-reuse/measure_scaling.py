"""How wide an input the positional-reuse pattern actually survives.

Three separate walls, measured separately because they scale differently.

  A. exact apply       the frozen shared program run over one observation:
                       O(N) module runs, each O(1) reads, but every value is
                       rebuilt and revalidated, so it is O(N * W) in practice.
  B. module search     learning the shared sub-program. The scaffold sees one
                       (position, observation) record per example, so it is
                       O(W) per example and independent of N.
  C. soft map choice   choosing between shared modules with the relaxed path.
                       `map`, `pair` and `insert` have no relaxation, so the
                       whole `records` set is materialised as a flat tensor of
                       N * (2 + W) values per batch row. This is the wall.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time
import tracemalloc

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import torch

from shared import IDX, Builder, shared_map_program
from tcn.graph import Candidate, Node, Signal
from tcn.learning import SoftProgram, tensor
from tcn.operators import Registry
from tcn.search import enumerate_fit
from tcn.types import BOOL, Value, integer, product

BYTE = integer(8, signed=False, role="byte")
BACKGROUND = (24, 30, 43)
CANDIDATE_BYTES = (24, 30, 43, 99)


def bytes_type(width):
    return product(*(BYTE for _ in range(width)))


def scaffold(registry, width, searched=True):
    """Same shape as the demonstration's module scaffold, at any width."""
    B = bytes_type(width)
    REC = product(IDX, B)
    consts = tuple((f"byte_{v}", Value.of(BYTE, v)) for v in CANDIDATE_BYTES)
    consts += (("one", Value.of(IDX, 1)), ("two", Value.of(IDX, 2)))
    b = Builder(registry, (("rec", REC),), consts)
    b.add("pos", "project", ["rec"], params={"index": 0})
    b.add("obs", "project", ["rec"], params={"index": 1})
    b.add("g_at", "add", ["pos", "one"])
    b.add("b_at", "add", ["pos", "two"])
    b.add("red", "index", ["obs", "pos"])
    b.add("green", "index", ["obs", "g_at"])
    b.add("blue", "index", ["obs", "b_at"])
    picks = CANDIDATE_BYTES if searched else None
    for name, source, truth in (("cmp_r", "red", 24), ("cmp_g", "green", 30), ("cmp_b", "blue", 43)):
        values = picks if searched else (truth,)
        cands = [Candidate(registry.resolve("eq", (BYTE, BYTE)), (source, f"byte_{v}")) for v in values]
        sel = None if searched else 0
        node = Node(name, BOOL, tuple(cands), "core", b.depths[source] + 1, sel)
        b.nodes.append(node); b.types[name] = BOOL; b.depths[name] = node.depth
    for name, sources, truth in (("rg", ("cmp_r", "cmp_g"), 8), ("foreground", ("rg", "cmp_b"), 7)):
        ks = range(16) if searched else (truth,)
        cands = [Candidate(registry.resolve(f"truth_{k}", (BOOL, BOOL)), sources) for k in ks]
        depth = max(b.depths[s] for s in sources) + 1
        node = Node(name, BOOL, tuple(cands), "core", depth, None if searched else 0)
        b.nodes.append(node); b.types[name] = BOOL; b.depths[name] = node.depth
    b.add("record", "tuple", ["pos", "foreground"])
    return b.program((("y", "record"),))


def synthetic(width, positions, rng):
    """A synthetic raster with the same byte semantics; the generator is exercised
    separately in demo_geometry.py, this is only for scaling."""
    data = []
    for i in range(len(positions)):
        data.extend(BACKGROUND if (i * 7 + 3) % 3 else (200, 40, 90))
    return tuple(data[:width])


def budget(seconds, label, fn):
    start = time.perf_counter()
    value = fn()
    return value, time.perf_counter() - start


def run(res, examples=48, limit=120.):
    positions = tuple(3 * i for i in range(res * res))
    width = 3 * res * res
    B = bytes_type(width)
    REC = product(IDX, B)
    row = {"resolution": res, "width": width, "positions": len(positions)}
    r = Registry()
    data = synthetic(width, positions, None)
    obs = Value.of(B, data)

    # B: module search over the same space, at this width.
    sc = scaffold(r, width, searched=True)
    labels = [(data[p], data[p + 1], data[p + 2]) != BACKGROUND for p in positions]
    exs = [{"inputs": {"rec": Value.of(REC, (positions[i], data))},
            "targets": {"foreground": Value.of(BOOL, labels[i])}}
           for i in range(min(examples, len(positions)))]
    signals = (Signal("foreground", "foreground", ("core",), BOOL, "bce"),)
    start = time.perf_counter()
    search = enumerate_fit(sc, exs, signals, r, tolerance=1e-6, stop_at_first=True)
    row["enumerate_seconds"] = time.perf_counter() - start
    row["enumerate_solved"] = search.solved
    row["space_size"] = search.space_size

    model = SoftProgram(sc, r)
    inputs = {"rec": torch.stack([tensor(ex["inputs"]["rec"]) for ex in exs])}
    start = time.perf_counter()
    model(inputs)
    row["soft_forward_seconds"] = time.perf_counter() - start

    # A: exact apply of the frozen shared program at every position.
    frozen = scaffold(r, width, searched=False)
    name = r.register_module(frozen)
    caller = shared_map_program(r, width, positions, [name], element=BYTE)
    row["caller_nodes"] = len(caller.nodes)
    row["caller_description_bits"] = caller.description_bits(r)
    row["caller_execution_cost"] = caller.execution_cost(r)
    start = time.perf_counter()
    out, _ = caller.run({"observation": obs}, registry=r)
    row["exact_apply_seconds"] = time.perf_counter() - start
    row["exact_apply_correct"] = out["y"].decoded == frozenset(
        (positions[i], labels[i]) for i in range(len(positions)))

    # C: one relaxed forward through the map node, which is where the record set
    # has to be materialised as a flat tensor.
    row["records_flat_values"] = len(positions) * (1 + 1 + width)
    if row["records_flat_values"] <= 4_000_000:
        soft = shared_map_program(r, width, positions, [name], element=BYTE)
        m = SoftProgram(soft, r)
        x = {"observation": tensor(obs).unsqueeze(0)}
        tracemalloc.start()
        start = time.perf_counter()
        try:
            m(x)
            row["soft_map_seconds"] = time.perf_counter() - start
            row["soft_map_peak_mib"] = tracemalloc.get_traced_memory()[1] / 2**20
        except (MemoryError, RuntimeError) as exc:
            row["soft_map_seconds"] = None
            row["soft_map_error"] = repr(exc)[:120]
        tracemalloc.stop()
    else:
        row["soft_map_seconds"] = None
        row["soft_map_error"] = "skipped: flat record set exceeds 4e6 values"
    return row


if __name__ == "__main__":
    rows = []
    for res in (2, 4, 6, 8, 12, 16, 24, 32):
        row = run(res)
        rows.append(row)
        print(f"res={row['resolution']:3d} W={row['width']:5d} N={row['positions']:5d} | "
              f"enum {row['enumerate_seconds']:7.2f}s solved={row['enumerate_solved']} | "
              f"soft-fwd {row['soft_forward_seconds']:6.2f}s | "
              f"apply {row['exact_apply_seconds']:7.2f}s ok={row['exact_apply_correct']} | "
              f"soft-map {row['soft_map_seconds']}")
        sys.stdout.flush()
        if row["exact_apply_seconds"] > 240:
            break
    path = pathlib.Path(__file__).parent / "scaling.json"
    path.write_text(json.dumps(rows, indent=1))
    print("wrote", path)
