"""Step 4, second held-out task — one whose address cannot be a constant.

`T_C1` (`run_transfer.py`) turned out to be solvable from a *constant* byte
address, so it cannot separate a transferred address computation from a lookup.
`T_C2` is the task §23 itself says is causal: the document name varies in length,
so **no constant byte address finds the digit**. Reading it needs
`length - k`, which is a computed address.

This is the sharper test of the same question, and it is the one that separates
the two halves of a §55 class: the **schema** (a function from width to a
scaffold) and the **certified selection vector** (`n0 = add`).

Writes `out/transfer2.json`.
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
import schema as S                                    # noqa: E402
import run_transfer as T1                             # noqa: E402
from tcn.graph import Program, Signal                 # noqa: E402
from tcn.operators import Registry                    # noqa: E402
from tcn.search import enumerate_fit                  # noqa: E402
from tcn.types import BOOL, Value                     # noqa: E402

OUT = HERE / "out"

LEN, DATA, BYTE = T1.LEN, T1.DATA, T1.BYTE
ADDR_VALUES = (0, 1, 2, 3)
BYTE_VALUES = tuple(ord(c) for c in "{\"pa=5")        # includes the right literal
CONSTANTS = tuple((f"p{v}", Value.of(LEN, v)) for v in ADDR_VALUES) + \
            tuple((f"b{v}", Value.of(BYTE, v)) for v in BYTE_VALUES)
SOURCES = ("blen",) + tuple(f"p{v}" for v in ADDR_VALUES)

# Documents. The generator is the computer track's own; only the (name, digit)
# pairs are chosen here, half with the target digit so the best constant is 0.5,
# and with name lengths 1..11 so no constant address can find the digit. The test
# names and digits are disjoint from the train names.
TRAIN_DOCS = [("n", 5), ("ab", 3), ("cnt", 5), ("total", 1), ("counter", 5),
              ("q", 2), ("idx", 5), ("value", 8), ("register", 5), ("k", 0),
              ("bb", 5), ("tot", 6), ("delta", 5), ("m", 7), ("count", 5),
              ("xy", 4), ("offset", 5), ("cc", 0), ("increment", 5), ("gap", 1)]
TEST_DOCS = [("z", 5), ("pq", 4), ("sum", 5), ("index", 7), ("accumul", 5),
             ("w", 6), ("len", 5), ("width", 8), ("position", 5), ("j", 2),
             ("ratio", 5), ("hh", 3), ("stride", 5), ("t", 0), ("marker", 5),
             ("nn", 1), ("threshold", 5), ("uv", 8), ("remainder", 5), ("y", 6)]
TARGET_DIGIT = 5


def examples(documents):
    """The tick at which the terminal shows the file's own content."""
    import task as CT
    rows = []
    for name, digit in documents:
        h, acts = CT.scripted_episode(name, digit)
        for r in CT.examples_from(h, acts):
            if r["terminal_text"].startswith("{"):
                continue                                  # the JSON result ticks
            term = Value.from_dict(r["terminal"])
            rows.append({"inputs": {"terminal": term},
                         "targets": {"y": Value.of(BOOL, digit == TARGET_DIGIT)},
                         "label": digit == TARGET_DIGIT,
                         "text": r["terminal_text"]})
    return rows


def build(arm, module=None, modules=None):
    r = Registry()
    inputs = (("terminal", T1.TERMINAL),)
    b = T1._B(r, inputs, CONSTANTS)
    b.add("blen", [b.op("project", ("terminal",), parameters={"index": 0})])
    b.add("data", [b.op("project", ("terminal",), parameters={"index": 1})])

    if arm == "A1_budget2":
        b.add("d0", [b.op("index", ("data", s)) for s in SOURCES])
        b.add("answer", [b.op("eq", ("d0", f"b{v}")) for v in BYTE_VALUES])
    elif arm == "A1_budget3":
        b.add("d0", [b.op(o, (s1, s2)) for o in S.COMBINE
                     for s1 in SOURCES for s2 in SOURCES])
        b.add("d1", [b.op("index", ("data", "d0"))])
        b.add("answer", [b.op("eq", ("d1", f"b{v}")) for v in BYTE_VALUES])
    elif arm in ("A2_class", "A4_hand"):
        name = r.register_module(module)
        b.add("answer", [b.op(name, (s1, s2, "data", f"b{v}"))
                         for s1 in SOURCES for s2 in SOURCES for v in BYTE_VALUES])
    elif arm == "A2b_schema":
        names = [r.register_module(m) for m in modules]
        b.add("answer", [b.op(n, (s1, s2, "data", f"b{v}"))
                         for n in names for s1 in SOURCES for s2 in SOURCES
                         for v in BYTE_VALUES])
    elif arm == "A3_wrong_motif":
        name = r.register_module(module)
        b.add("answer", [b.op(name, ("data", s1, s2))
                         for s1 in SOURCES for s2 in SOURCES])
    else:
        raise ValueError(arm)
    return Program(inputs, tuple(b.nodes), (("y", "answer"),), CONSTANTS).validate(r), r


def score(arm, program, registry, train, test):
    signals = (Signal("answer", "y", ("core",), BOOL),)
    t0 = time.perf_counter()
    res = enumerate_fit(program, train, signals, registry, tolerance=.001)
    d = res.to_dict()
    d["arm"] = arm
    d["seconds"] = round(time.perf_counter() - t0, 2)
    d["free_nodes"] = [(n.name, len(n.candidates)) for n in program.nodes
                       if len(n.candidates) > 1]
    d["achieved_difficulty"] = (res.conforming / res.space_size) if res.space_size else None
    d["held_out"] = None
    d["chosen_operator"] = None
    if res.selections:
        exact = program.harden(res.selections)
        cand = exact.nodes[-1].candidates[0]
        d["chosen_operator"] = f"{cand.operator.name}({', '.join(cand.sources)})"
        ok = 0
        for ex in test:
            try:
                o, _ = exact.run(ex["inputs"], registry=registry)
                ok += int(next(iter(o.values())).decoded == ex["label"])
            except Exception:
                pass
        d["held_out"] = round(ok / len(test), 6)
    return d


def main():
    train, test = examples(TRAIN_DOCS), examples(TEST_DOCS)
    p = sum(e["label"] for e in train) / len(train)
    q = sum(e["label"] for e in test) / len(test)
    report = {"baselines": {"n_train": len(train), "n_test": len(test),
                            "train_label_rate": round(p, 6),
                            "best_constant_train": round(max(p, 1 - p), 6),
                            "test_label_rate": round(q, 6),
                            "best_constant_test": round(max(q, 1 - q), 6)},
              "arms": []}

    cls, record = T1.class_module()
    hand_add, _ = T1.hand_module()
    wrong, _ = T1.wrong_motif_module()
    at = S.address_type(S.SETTINGS["computer"]["address"])
    schema_family = [S.instantiate("M3", 4096, at, {"n0": i})[1] for i in range(5)]
    report["class_module_digest"] = cls.digest
    report["class_id"] = record.class_id
    report["class_certified_widths"] = list(record.widths)
    report["schema_family_digests"] = [m.digest for m in schema_family]
    report["schema_family_operators"] = list(S.COMBINE)

    plan = (("A1_budget2", None, None), ("A1_budget3", None, None),
            ("A2_class", cls, None), ("A2b_schema", None, schema_family),
            ("A3_wrong_motif", wrong, None), ("A4_hand", hand_add, None))
    for arm, module, modules in plan:
        program, reg = build(arm, module, modules)
        d = score(arm, program, reg, train, test)
        report["arms"].append(d)
        print(f"  {arm:16s} space={d['space_size']:6d} evaluated={d['evaluated']:6d} "
              f"exhausted={d['exhausted']} conforming={d['conforming']:5d} "
              f"cert={d['certificate']:9s} held={d['held_out']} "
              f"random={d['achieved_difficulty']} {d['seconds']}s")
        if d["chosen_operator"]:
            print(f"       chosen: {d['chosen_operator']}")

    (OUT / "transfer2.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    rep = main()
    b = rep["baselines"]
    print(f"\nbaselines: best constant train {b['best_constant_train']} (n={b['n_train']}), "
          f"test {b['best_constant_test']} (n={b['n_test']})")
