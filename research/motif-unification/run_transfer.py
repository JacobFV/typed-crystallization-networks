"""Step 4 — does the unified class help a domain it was not derived from?

The class is derived from **language (128) and visual (3072)** only. The
**computer** domain (4096) contributes nothing to the schema, the selection
vector or the certification. `T_C` is a decision over the real computer track's
terminal observation, on that track's own train/test document split.

Writes `out/transfer.json`.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "class-identity"))
sys.path.insert(0, str(ROOT / "research" / "computer-capability"))

import classes as C                                   # noqa: E402
import episodes as E                                  # noqa: E402
import schema as S                                    # noqa: E402
from tcn.generation import text_value                 # noqa: E402
from tcn.graph import Candidate, Node, Program, Signal  # noqa: E402
from tcn.operators import Registry                    # noqa: E402
from tcn.search import enumerate_fit, space_size      # noqa: E402
from tcn.types import BOOL, Value, floating, integer, product  # noqa: E402

OUT = HERE / "out"
STORE = OUT / "library"

F = floating()
TERMINAL = text_value("", 4096).type
LEN, DATA = TERMINAL.items
BYTE = DATA.items[0]
ACTION = product(F, F, F)

# Constant pools, fixed before any arm. Every arm sees the same pools.
ADDR_VALUES = (0, 1, 2, 3)
BYTE_VALUES = tuple(ord(c) for c in "{\"pa=")        # includes the right literal
WRITE = 2                                            # TEMPLATES index of `write`

CONSTANTS = (("half", Value.of(F, .5)),) + \
            tuple((f"p{v}", Value.of(LEN, v)) for v in ADDR_VALUES) + \
            tuple((f"b{v}", Value.of(BYTE, v)) for v in BYTE_VALUES)


def examples(documents):
    """The real scripted trajectories; target is `should the agent write`."""
    rows = E.computer_records(documents)
    out = []
    for r in rows:
        term = Value.from_dict(r["terminal"])
        prev = Value.of(ACTION, tuple(1. if j == r["previous"] else 0. for j in range(3)))
        label = (r["reference"] == WRITE)
        out.append({"inputs": {"terminal": term, "action": prev},
                    "targets": {"y": Value.of(BOOL, label)},
                    "label": label})
    return out


def _head(r, b):
    b.add("data", [b.op("project", ("terminal",), parameters={"index": 1})])
    b.add("prev", [b.op("project", ("action",), parameters={"index": 2})])
    b.add("wrote", [b.op("gt", ("prev", "half"))])


class _B:
    """Nodes plus a depth ledger; the same idiom as §53's scaffold builder."""

    def __init__(self, registry, inputs, constants):
        self.r = registry
        self.nodes = []
        self.types = dict(inputs) | {k: v.type for k, v in constants}
        self.depth = {k: 0 for k, _ in inputs} | {k: -1 for k, _ in constants}

    def op(self, name, sources, output=None, parameters=None):
        return Candidate(self.r.resolve(name, tuple(self.types[s] for s in sources),
                                        output, parameters), tuple(sources))

    def add(self, name, cands, region="core"):
        out = cands[0].operator.output
        d = max((self.depth[s] for c in cands for s in c.sources), default=0) + 1
        self.nodes.append(Node(name, out, tuple(cands), region, max(d, 1)))
        self.types[name] = out
        self.depth[name] = max(d, 1)
        return name


def build(arm, module=None):
    """One scaffold per arm. The head and the combiner are identical everywhere."""
    r = Registry()
    inputs = (("terminal", TERMINAL), ("action", ACTION))
    b = _B(r, inputs, CONSTANTS)
    _head(r, b)

    if arm == "A1_budget1":
        cands = []
        for i in ADDR_VALUES:
            for j in ADDR_VALUES:
                for o in ("eq", "lt", "le", "gt", "ge"):
                    cands.append(b.op(o, (f"p{i}", f"p{j}")))
        for u in BYTE_VALUES:
            for v in BYTE_VALUES:
                cands.append(b.op("eq", (f"b{u}", f"b{v}")))
        b.add("detect", cands)
    elif arm == "A1_budget2":
        b.add("d0", [b.op("index", ("data", f"p{i}")) for i in ADDR_VALUES])
        b.add("detect", [b.op("eq", ("d0", f"b{v}")) for v in BYTE_VALUES])
    elif arm == "A1_budget3":
        b.add("d0", [b.op(o, (f"p{i}", f"p{j}"))
                     for o in S.COMBINE for i in ADDR_VALUES for j in ADDR_VALUES])
        b.add("d1", [b.op("index", ("data", "d0"))])
        b.add("detect", [b.op("eq", ("d1", f"b{v}")) for v in BYTE_VALUES])
    elif arm in ("A2_class", "A4_hand", "A3_wrong_motif"):
        name = r.register_module(module)
        if arm == "A3_wrong_motif":                       # (buffer, addr, addr)
            b.add("detect", [b.op(name, ("data", f"p{i}", f"p{j}"))
                             for i in ADDR_VALUES for j in ADDR_VALUES])
        else:                                             # (base, offset, buffer, byte)
            b.add("detect", [b.op(name, (f"p{i}", f"p{j}", "data", f"b{v}"))
                             for i in ADDR_VALUES for j in ADDR_VALUES
                             for v in BYTE_VALUES])
    else:
        raise ValueError(arm)

    b.add("answer", [b.op(f"truth_{j}", ("detect", "wrote")) for j in range(16)])
    program = Program(inputs, tuple(b.nodes), (("y", "answer"),), CONSTANTS).validate(r)
    return program, r


# ------------------------------------------------------------------ modules
def class_module():
    """The transferred artifact: the stored class, instantiated at 4,096.

    Nothing about the computer domain took part in deriving this. The class was
    certified at 128 and 3,072; `ClassStore.instantiate` rebuilds the schema at
    4,096, checks the shape of the choice is what the record declares, applies
    the stored vector and returns the frozen program.
    """
    store = C.ClassStore(STORE)
    rec = store.records[0]
    at = S.address_type(S.SETTINGS["computer"]["address"])

    def builder(width):
        return S.SCHEMAS["M3"](width, at)

    names, exact, reg, record = store.instantiate(rec.class_id, 4096, builder, S.freeze)
    return exact, record


def hand_module():
    """A4: the same detector written by hand for the computer domain."""
    r = Registry()
    inputs = (("x0", LEN), ("x1", LEN), ("x2", DATA), ("x3", BYTE))
    b = _B(r, inputs, ())
    b.add("addr", [b.op("add", ("x0", "x1"))])
    b.add("byte", [b.op("index", ("x2", "addr"))])
    b.add("flag", [b.op("eq", ("byte", "x3"))])
    return Program(inputs, tuple(b.nodes), (("out", "flag"),)).validate(r).harden(
        {n.name: 0 for n in b.nodes}), r


def wrong_motif_module():
    """A3: the *other* real motif, `V_same`'s, instantiated at 4,096.

    Three nodes, same size as the class's body, drawn from a real artifact rather
    than invented: "are the bytes at two computed addresses equal". It is a wrong
    module for this task, not a broken one.
    """
    r = Registry()
    inputs = (("x0", DATA), ("x1", LEN), ("x2", LEN))
    b = _B(r, inputs, ())
    b.add("a", [b.op("index", ("x0", "x1"))])
    b.add("c", [b.op("index", ("x0", "x2"))])
    b.add("flag", [b.op("eq", ("a", "c"))])
    return Program(inputs, tuple(b.nodes), (("out", "flag"),)).validate(r).harden(
        {n.name: 0 for n in b.nodes}), r


def _structural(program):
    """`to_dict()` with node names, the output port name and `version` erased.

    Weaker than the digest and used only to say *why* two digests differ; never
    used as an identity anywhere a bit-identity claim is made.
    """
    rename = {n.name: f"m{i}" for i, n in enumerate(program.nodes)}
    d = program.to_dict()
    d["version"] = None
    d["outputs"] = [["out", rename.get(v, v)] for _, v in program.outputs]
    for nd, n in zip(d["nodes"], program.nodes):
        nd["name"] = rename[n.name]
        for c in nd["candidates"]:
            c["sources"] = [rename.get(s, s) for s in c["sources"]]
    return json.dumps(d, sort_keys=True)


def wrong_width_module():
    """A3b: §30's control -- the class instantiated at the *language* width."""
    at = S.address_type(S.SETTINGS["language"]["address"])
    _, exact, _ = S.instantiate("M3", 128, at, {"n0": 0})
    return exact


# ---------------------------------------------------------------------- run
def score(arm, program, registry, train, test):
    signals = (Signal("answer", "y", ("core",), BOOL),)
    t0 = time.perf_counter()
    res = enumerate_fit(program, train, signals, registry, tolerance=.001)
    d = res.to_dict()
    d["arm"] = arm
    d["seconds"] = round(time.perf_counter() - t0, 2)
    d["nodes"] = len(program.nodes)
    d["free_nodes"] = [(n.name, len(n.candidates)) for n in program.nodes
                       if len(n.candidates) > 1]
    d["achieved_difficulty"] = (res.conforming / res.space_size) if res.space_size else None
    d["held_out"] = None
    if res.selections:
        exact = program.harden(res.selections)
        ok = 0
        for ex in test:
            try:
                o, _ = exact.run(ex["inputs"], registry=registry)
                ok += int(next(iter(o.values())).decoded == ex["label"])
            except Exception:
                pass
        d["held_out"] = round(ok / len(test), 6)
    return d


def baselines(train, test):
    p = sum(e["label"] for e in train) / len(train)
    q = sum(e["label"] for e in test) / len(test)
    return {"train_label_rate": round(p, 6), "best_constant_train": round(max(p, 1 - p), 6),
            "test_label_rate": round(q, 6), "best_constant_test": round(max(q, 1 - q), 6),
            "n_train": len(train), "n_test": len(test)}


def main():
    sys.path.insert(0, str(ROOT / "research" / "computer-capability"))
    import task as CT
    train = examples(CT.TRAIN_DOCUMENTS)
    test = examples(CT.TEST_DOCUMENTS)
    report = {"baselines": baselines(train, test), "arms": []}

    cls, record = class_module()
    hand, _ = hand_module()
    wrong, _ = wrong_motif_module()
    report["class_module_digest"] = cls.digest
    report["hand_module_digest"] = hand.pruned().digest
    report["class_equals_hand_authored"] = cls.digest == hand.pruned().digest
    # The digest is over node *names* and over the `version` counter, so a
    # hand-authored program with its own names differs from the instantiated one
    # even when the graphs coincide. Report the structural comparison too, and
    # say which fields it normalises: node names, output name, and `version`.
    report["class_hand_structurally_identical"] = _structural(cls) == _structural(
        hand.pruned())
    report["structural_normalises"] = ["node names", "output port name", "version"]
    report["class_id"] = record.class_id
    report["class_certified_widths"] = list(record.widths)

    for arm, module in (("A1_budget1", None), ("A1_budget2", None), ("A1_budget3", None),
                        ("A2_class", cls), ("A3_wrong_motif", wrong), ("A4_hand", hand)):
        program, reg = build(arm, module)
        d = score(arm, program, reg, train, test)
        # uniform random over the arm's own space, measured not assumed
        report["arms"].append(d)
        print(f"  {arm:14s} space={d['space_size']:8d} evaluated={d['evaluated']:8d} "
              f"exhausted={d['exhausted']} conforming={d['conforming']:6d} "
              f"cert={d['certificate']:9s} held_out={d['held_out']} "
              f"difficulty={d['achieved_difficulty']} {d['seconds']}s")

    # A3b: §30's wrong-width control -- a construction refusal, not a score
    try:
        build("A2_class", wrong_width_module())
        report["A3b_wrong_width"] = {"constructed": True, "error": None}
    except Exception as e:
        report["A3b_wrong_width"] = {"constructed": False,
                                     "error": f"{type(e).__name__}: {e}"}

    (OUT / "transfer.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    rep = main()
    b = rep["baselines"]
    print(f"\nbaselines: train constant {b['best_constant_train']} (n={b['n_train']}), "
          f"test constant {b['best_constant_test']} (n={b['n_test']})")
    print("class module digest", rep["class_module_digest"],
          "== hand-authored:", rep["class_equals_hand_authored"])
    print("A3b wrong width:", rep["A3b_wrong_width"]["error"])
