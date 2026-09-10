"""The earlier domains' instantiations, as labelled rows for the learned prior.

PREREGISTRATION §5. Every candidate of every source slot is labelled by whether
it survives in *some* conforming program of its own exhausted search. Nothing
here knows about the integrated task. The searches are re-run from the tracks'
own scaffolds and compared with their recorded certificates; the visual rows are
read from committed `research/visual-width-reuse/out/enum_32.json`.

Streams, stated: §19 stage A on the language generator with `hardening='none'`
(the pre-audit stream §19 was measured on; §39, §45); §23 on the computer
generator's default shell interface through its own scripted episodes.

    python sources.py        # -> out/sources.json
"""
from __future__ import annotations

import itertools
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import tcn.search  # noqa: E402,F401  (import our tree's tcn before any track file inserts another path)
import generators.computer.generator  # noqa: E402,F401
from tcn.operators import Registry  # noqa: E402
from tcn.search import evaluate, space_size  # noqa: E402
from tcn.generation import read_text  # noqa: E402
from tcn.types import Value  # noqa: E402

OUT = HERE / "out"
LANG = ROOT / "research" / "language-capability"
COMP = ROOT / "research" / "computer-capability"
VIS = ROOT / "research" / "visual-width-reuse" / "out" / "enum_32.json"


def conforming_set(program, examples, signals, registry, tolerance):
    names = [n.name for n in program.nodes]
    counts = [len(n.candidates) for n in program.nodes]
    found = []
    for combo in itertools.product(*(range(c) for c in counts)):
        sel = dict(zip(names, combo))
        err = evaluate(program, sel, examples, signals, registry, tolerance)
        if err is not None and err <= tolerance:
            found.append(sel)
    return found, space_size(program)


def varies(values_by_context):
    """V: the addressed byte takes >= 2 values across episodes for some fixed context."""
    return any(len(set(v)) >= 2 for v in values_by_context.values())


# ------------------------------------------------------------------ §19
def language_rows():
    sys.path.insert(0, str(LANG))
    import common            # noqa: E402
    import scaffolds         # noqa: E402
    import run_stage_a       # noqa: E402
    pool = common.dataset(600, seed0=0, split="train", hardening=common.STREAM_PRE_AUDIT)
    train = [e for e in pool if e["length"] in (2, 4, 6)][:12]
    rows = run_stage_a.examples(train)
    program, registry, signals = scaffolds.stage_a()
    started = time.perf_counter()
    found, size = conforming_set(program, rows, signals, registry, 1e-6)
    bases = [int(c.sources[0][1:]) for c in program.nodes[[n.name for n in program.nodes].index("base")].candidates]
    literal_names = [c.sources[1] for c in program.nodes[[n.name for n in program.nodes].index("open")].candidates]
    literals = [int(s[1:]) for s in literal_names]
    surv_base = {bases[s["base"]] for s in found}
    surv_pair = {(bases[s["base"]], literals[s["open"]]) for s in found}
    # bytes read at base+pos, per fixed pos, across training episodes
    by = {}
    for k in bases:
        ctx = {}
        for e in train:
            raw = e["text"].decoded[1]
            for p in range(len(e["string"])):
                a = k + p
                ctx.setdefault(p, []).append(raw[a] if a < len(raw) else None)
        by[k] = ctx
    addr = [{"source": "s19_stage_a", "slot": "base", "candidate": f"add(p{k}, pos)",
             "op": "add", "kinds": ["constant", "position"], "V": varies(by[k]),
             "survive": k in surv_base} for k in bases]
    lit = []
    for k in sorted(surv_base):
        seen = {b for vals in by[k].values() for b in vals if b is not None}
        for v in literals:
            lit.append({"source": "s19_stage_a", "slot": "open", "address": f"add(p{k}, pos)",
                        "literal": v, "occurs": v in seen, "survive": (k, v) in surv_pair})
    cert = {"source": "s19_stage_a", "stream": "hardening='none'", "space": size,
            "conforming": len(found), "selections": found,
            "certificate": "unique" if len(found) == 1 else "complete",
            "recorded": {"space": 10496, "conforming": 1, "base": 14, "byte": 40},
            "seconds": time.perf_counter() - started}
    return addr, lit, cert


# ------------------------------------------------------------------ §23
def kinds_of(sources, length_ports):
    return sorted("length" if s in length_ports else "constant" for s in sources)


def computer_rows():
    sys.path.insert(0, str(COMP))
    import program as P      # noqa: E402
    import task as T         # noqa: E402
    import search_run as SR  # noqa: E402
    payload = T.collect(T.TRAIN_DOCUMENTS, str(OUT / "sources_computer_train.json"))
    certs, addr, lit = [], [], []
    registry = Registry()
    for name, build, rows_of, sig in (
            ("transform", lambda: P.transform_program(registry, encode=False), SR.transform_examples, P.signals()),
            ("policy", lambda: P.policy_program(registry), SR.policy_examples, P.policy_signals())):
        program = build()
        rows = rows_of(payload)
        started = time.perf_counter()
        found, size = conforming_set(program, rows, sig, registry, .001)
        certs.append({"source": f"s23_{name}", "space": size, "conforming": len(found),
                      "certificate": "unique" if len(found) == 1 else "complete",
                      "recorded": {"transform": 2, "policy": 21}[name],
                      "seconds": time.perf_counter() - started})
        slot = "pos" if name == "transform" else "ppos"
        node = program.nodes[[n.name for n in program.nodes].index(slot)]
        consts = dict(program.constants)
        survivors = {s[slot] for s in found}
        for i, c in enumerate(node.candidates):
            ctx = {0: []}
            for r in rows:
                length, data = r["inputs"]["terminal"].decoded
                vals = [length if s == "length" else consts[s].decoded for s in c.sources]
                if c.operator.name == "identity":
                    a = vals[0]
                elif c.operator.name == "sub":
                    a = vals[0] - vals[1]
                else:
                    a = vals[0] + vals[1]
                ctx[0].append(data[a] if 0 <= a < len(data) else None)
            addr.append({"source": f"s23_{name}", "slot": slot,
                         "candidate": f"{c.operator.name}({', '.join(c.sources)})",
                         "op": c.operator.name, "kinds": kinds_of(c.sources, {"length"}),
                         "V": varies(ctx), "survive": i in survivors, "_bytes": ctx[0]})
        if name == "policy":
            bnode = program.nodes[[n.name for n in program.nodes].index("brand")]
            for i in sorted(survivors):
                seen = {b for b in addr[-len(node.candidates) + i]["_bytes"] if b is not None}
                pairs = {(s["ppos"], s["brand"]) for s in found}
                for j, c in enumerate(bnode.candidates):
                    v = consts[c.sources[1]].decoded
                    lit.append({"source": "s23_policy", "slot": "brand",
                                "address": addr[-len(node.candidates) + i]["candidate"],
                                "literal": v, "occurs": v in seen, "survive": (i, j) in pairs})
    for row in addr:
        row.pop("_bytes", None)
    return addr, lit, certs


# ------------------------------------------------------------------ §33
def offset_class(off, width):
    return {3: "pixel", 3 * width: "row", 6: "2pixel", 3 * width + 3: "row+pixel",
            9: "3pixel"}.get(off, "other")


def visual_rows():
    d = json.loads(VIS.read_text())
    width = d["width"]
    pool = d["offset_pool"]
    st = d["stages"]
    truth, step = [], []
    s0 = st["s0"]["conforming_free"]
    for slot in ("rg", "same"):
        surv = {c[slot] for c in s0}
        truth += [{"source": "s33_s0", "slot": slot, "table": t, "survive": t in surv}
                  for t in range(16)]
    s1 = st["s1"]["conforming_free"]
    surv = {c["corner"] for c in s1}
    truth += [{"source": "s33_s1", "slot": "corner", "table": t, "survive": t in surv}
              for t in range(16)]
    for slot in ("back_a", "back_b"):
        surv = {c[slot] for c in s1}
        step += [{"source": "s33_s1", "slot": slot, "op": "sub", "offset": pool[i],
                  "offset_class": offset_class(pool[i], width), "survive": i in surv}
                 for i in range(len(pool))]
    s2 = st["s2"]["conforming_free"]
    for slot in ("step_w", "step_h"):
        surv = {c[slot] for c in s2}
        step += [{"source": "s33_s2", "slot": slot, "op": "add", "offset": pool[i],
                  "offset_class": offset_class(pool[i], width), "survive": i in surv}
                 for i in range(len(pool))]
    cert = {"source": "s33", "resolution": d["resolution"], "file": str(VIS.relative_to(ROOT)),
            "stages": {k: {"space": st[k]["space_size"], "conforming": st[k]["sweep"]["conforming"],
                           "certificate": st[k]["sweep"]["certificate"]} for k in ("s0", "s1", "s2")}}
    return truth, step, cert


def main():
    OUT.mkdir(exist_ok=True)
    la, ll, lc = language_rows()
    ca, cl, cc = computer_rows()
    vt, vs, vc = visual_rows()
    payload = {"ADDR": la + ca, "LIT": ll + cl, "TRUTH": vt, "STEP": vs, "GROUND": [],
               "certificates": [lc] + cc + [vc]}
    (OUT / "sources.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    for c in payload["certificates"]:
        print({k: v for k, v in c.items() if k != "selections"})
    for kind in ("ADDR", "LIT", "TRUTH", "STEP"):
        rows = payload[kind]
        print(kind, len(rows), "rows,", sum(r["survive"] for r in rows), "survive")


if __name__ == "__main__":
    main()
