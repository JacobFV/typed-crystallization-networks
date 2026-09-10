"""Step 1 — confirm the motif and its instances, and test F1 first and hardest.

Writes `out/step1.json`. Every number in `RESULTS.md` §1-§2 comes from this file.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import random
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import motif                                  # noqa: E402
from tcn.types import Value                   # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)


def binding_kind(program, name):
    if name in dict(program.inputs):
        return "input port"
    if name in dict(program.constants):
        v = dict(program.constants)[name]
        try:
            return f"constant = {v.decoded!r}"
        except Exception:
            return "constant"
    if name in {n.name for n in program.nodes}:
        n = next(n for n in program.nodes if n.name == name)
        c = n.candidates[n.selected if n.selected is not None else 0]
        return f"node = {c.operator.name}({', '.join(c.sources)})"
    return "unknown"


def type_signature(row):
    """(input types, output type) hashed exactly -- §58's S prerequisite."""
    blob = json.dumps({"in": row["input_type_tags"], "out": row["output_type_tag"]},
                      sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# --------------------------------------------------------------- C3 behaviour
def characterise(row, seed=0, limit=None):
    """Exhaustive over the buffer's own index range. States its quantifier.

    For every address `a` in [0, N) -- or the first `limit` of them, recorded --
    four exact checks against a seeded buffer R:

      1. f(R, a, R[a])            is True     (it reads cell a and compares)
      2. f(R, a, R[a]+1 mod 256)  is False    (the comparison is equality)
      3. f(P_a, a, R[a])          is False    (P_a = R with cell a bumped)
      4. f(P_a, a', R[a'])        is True     (a' = a+1 mod N: no other cell matters)

    For M3 the address is supplied as (base, offset) with base + offset = a, and
    a fifth check confirms a different decomposition of the same sum agrees.
    """
    canon, reg = row["_canon"], row["_registry"]
    types = {k: t for k, t in canon.inputs}
    shape = row["shape"]
    if shape == "M2":
        buf_k, addr_k, oth_k = "x0", "x1", "x2"
    elif shape == "M3":
        base_k, off_k, buf_k, oth_k = "x0", "x1", "x2", "x3"
    else:
        addr_k, buf_k, oth_k = "x0", "x1", "x2"
    bt = types[buf_k]
    n = len(bt.items)
    at = types[addr_k if shape != "M3" else base_k]
    ot = types[oth_k]

    rng = random.Random(seed)
    base_buf = tuple(rng.randrange(256) for _ in range(n))
    R = Value.of(bt, base_buf)

    addr_hi = n - 1
    if at.bounds is not None:
        addr_hi = min(addr_hi, int(at.bounds[1]))
    addresses = list(range(addr_hi + 1))
    if limit is not None:
        addresses = addresses[:limit]

    def run(bufval, addrpair, other):
        args = {buf_k: bufval, oth_k: Value.of(ot, other)}
        if shape == "M3":
            args[base_k] = Value.of(at, addrpair[0])
            args[off_k] = Value.of(types[off_k], addrpair[1])
        else:
            args[addr_k] = Value.of(at, addrpair)
        out, _ = canon.run(args, registry=reg)
        return next(iter(out.values())).decoded

    fails = []
    t0 = time.time()
    for a in addresses:
        ap = (a // 2, a - a // 2) if shape == "M3" else a
        v = base_buf[a]
        ap2 = a
        try:
            if run(R, ap, v) is not True:
                fails.append(("c1", a))
            if run(R, ap, (v + 1) % 256) is not False:
                fails.append(("c2", a))
            bumped = base_buf[:a] + ((v + 1) % 256,) + base_buf[a + 1:]
            P = Value.of(bt, bumped)
            if run(P, ap, v) is not False:
                fails.append(("c3", a))
            b = (a + 1) % (addr_hi + 1)
            bp = (b // 2, b - b // 2) if shape == "M3" else b
            if run(P, bp, bumped[b]) is not True:
                fails.append(("c4", a))
            if shape == "M3" and a >= 3:
                alt = (a - 3, 3)
                if run(R, alt, v) is not True:
                    fails.append(("c5", a))
        except Exception as e:                       # never silently dropped
            fails.append((f"error:{type(e).__name__}", a))
    return {"addresses_checked": len(addresses),
            "address_range": [0, addr_hi],
            "buffer_fields": n,
            "exhaustive_over_addresses": limit is None,
            "checks_per_address": 5 if shape == "M3" else 4,
            "failures": fails[:20],
            "n_failures": len(fails),
            "seconds": round(time.time() - t0, 2)}


def main(max_nodes=5, limit=None):
    t0 = time.time()
    arts, rows = motif.find(max_nodes)
    report = {"max_nodes": max_nodes, "occurrences": [], "mine_seconds": None}

    for r in rows:
        prog = r["_program"]
        rec = {k: v for k, v in r.items() if not k.startswith("_")}
        rec["type_signature"] = type_signature(r)
        rec["hole_bindings"] = [{"hole": f"x{i}", "artifact_port": h,
                                 "binding": binding_kind(prog, h)}
                                for i, h in enumerate(r["holes_in_artifact"])]
        report["occurrences"].append(rec)

    # --- the three-instance tables, one per shape
    by_shape = {}
    for rec in report["occurrences"]:
        by_shape.setdefault(rec["shape"], []).append(rec)
    report["by_shape"] = {
        s: {"domains": sorted({x["domain"] for x in v}),
            "n_domains": len({x["domain"] for x in v}),
            "artifacts": sorted({x["artifact"] for x in v}),
            "occurrences": len(v),
            "distinct_digests": sorted({x["digest"] for x in v}),
            "distinct_type_signatures": sorted({x["type_signature"] for x in v}),
            "buffer_widths": sorted({int(x["input_labels"][{"M2": 0, "M3": 2,
                                                            "M3_identity": 1}[s]]
                                         .split("x")[0].lstrip("(")) for x in v}),
            "address_carriers": sorted({x["input_labels"][{"M2": 1, "M3": 0,
                                                           "M3_identity": 0}[s]] for x in v}),
            }
        for s, v in by_shape.items()}

    # --- F1: is the 3-node `add` motif in all three domains?
    m3 = report["by_shape"].get("M3", {"n_domains": 0, "domains": []})
    report["F1"] = {
        "claim_under_test": "the 3-node motif eq(index(buffer, add(base, offset)), literal) "
                            "recurs in all three real artifacts (FINDINGS section 58 prose)",
        "M3_domains": m3["domains"],
        "M3_n_domains": m3["n_domains"],
        "M2_domains": report["by_shape"]["M2"]["domains"],
        "computer_three_node_shape": "M3_identity" if "M3_identity" in report["by_shape"] else None,
        "fires": m3["n_domains"] < 3,
    }
    # the role of the third argument, per domain -- the second half of F1
    report["F1"]["third_argument_role"] = {
        rec["artifact"] + "/" + rec["root"]: rec["hole_bindings"][-1]["binding"]
        for rec in report["occurrences"] if rec["shape"] in ("M2", "M3")}

    # --- C3 behaviour, one representative per (shape, artifact)
    seen = set()
    report["behaviour"] = []
    for r in rows:
        key = (r["shape"], r["artifact"])
        if key in seen:
            continue
        seen.add(key)
        b = characterise(r, limit=limit)
        b.update(shape=r["shape"], artifact=r["artifact"], domain=r["domain"],
                 digest=r["digest"])
        report["behaviour"].append(b)
        print(f"  behaviour {r['artifact']:12s} {r['shape']:12s} "
              f"{b['addresses_checked']} addresses, {b['n_failures']} failures, "
              f"{b['seconds']}s")

    report["mine_seconds"] = round(time.time() - t0, 1)
    (OUT / "step1.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    lim = None
    if "--limit" in sys.argv:
        lim = int(sys.argv[sys.argv.index("--limit") + 1])
    rep = main(limit=lim)
    print()
    for s, v in rep["by_shape"].items():
        print(f"{s:12s} domains={v['domains']} occurrences={v['occurrences']} "
              f"digests={len(v['distinct_digests'])} widths={v['buffer_widths']} "
              f"addr={v['address_carriers']}")
    print("\nF1 fires:", rep["F1"]["fires"], "-- M3 domains:", rep["F1"]["M3_domains"])
    for k, v in rep["F1"]["third_argument_role"].items():
        print(f"   {k:24s} {v}")
