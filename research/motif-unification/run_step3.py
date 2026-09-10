"""Step 3 — certify at two widths, then run §53's arm F against this schema.

Writes `out/step3.json`. The class format is
`research/class-identity/classes.py`, **imported unmodified**; every refusal
below is that file raising `ClassError` and naming its own rule.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "research" / "class-identity"))

import classes as C                                   # noqa: E402
import episodes as E                                  # noqa: E402
import schema as S                                    # noqa: E402
from tcn.generation import source_fingerprint         # noqa: E402
from tcn.graph import Signal                          # noqa: E402
from tcn.search import enumerate_fit                  # noqa: E402
from tcn.types import BOOL, Value                     # noqa: E402

OUT = HERE / "out"
STORE = OUT / "library"

QUALIFIED = "research.motif-unification.schema:m3_schema"
WRONG_A = "research.motif-unification.schema:m3_wrong_hardcoded"
WRONG_B = "research.motif-unification.schema:m3_wrong_reordered"

N_TRAIN_BUFFERS = 12
N_HELD_BUFFERS = 12
PER_BUFFER = 8

DOMAIN_OF_WIDTH = {128: "language", 3072: "visual", 4096: "computer"}


def examples_for(width, seed0, split, seed):
    """Real episodes of the width's own domain, as `enumerate_fit` examples."""
    domain = DOMAIN_OF_WIDTH[width]
    cfg = S.SETTINGS[domain]
    at = S.address_type(cfg["address"])
    element = S.buffer_type(1).items[0]
    bt = S.buffer_type(width)
    if domain == "language":
        bufs = E.language_buffers(N_TRAIN_BUFFERS, seed0, split)
        rows = E.motif_episodes(bufs, E.LANGUAGE_BASE, tuple(range(1, 40)),
                                width, seed=seed, n_per_buffer=PER_BUFFER)
    else:
        bufs = E.visual_buffers(N_TRAIN_BUFFERS, seed0, split)
        rows = E.motif_episodes(bufs, None, E.VISUAL_OFFSETS,
                                width, seed=seed, n_per_buffer=PER_BUFFER)
    out = []
    for r in rows:
        out.append({"inputs": {"x0": Value.of(at, r["base"]),
                               "x1": Value.of(at, r["offset"]),
                               "x2": Value.of(bt, r["buffer"]),
                               "x3": Value.of(element, r["other"])},
                    "targets": {"y": Value.of(BOOL, r["label"])},
                    "label": r["label"]})
    return out


def sweep(schema_name, width, examples):
    """Exhaust the schema's whole space at one width. Certificates, not opinions."""
    domain = DOMAIN_OF_WIDTH[width]
    at = S.address_type(S.SETTINGS[domain]["address"])
    free, program, reg = S.SCHEMAS[schema_name](width, at)
    signals = (Signal("n2", "y", ("core",), BOOL),)
    t0 = time.perf_counter()
    res = enumerate_fit(program, examples, signals, reg, tolerance=.001)
    d = res.to_dict()
    d["seconds"] = round(time.perf_counter() - t0, 2)
    d["free_nodes"] = [(n.name, len(n.candidates)) for n in program.nodes
                       if len(n.candidates) > 1]
    d["width"] = width
    d["domain"] = domain
    d["schema"] = schema_name
    # per-candidate accuracy, so a refusal can be read rather than inferred
    d["per_candidate_accuracy"] = {}
    for i in range(len(program.nodes[0].candidates)):
        name = program.nodes[0].candidates[i].operator.name
        _, exact, r2 = S.instantiate(schema_name, width, at, {"n0": i})
        ok = 0
        for ex in examples:
            try:
                o, _ = exact.run({k: v for k, v in ex["inputs"].items()}, registry=r2)
                ok += int(next(iter(o.values())).decoded == ex["label"])
            except Exception:
                pass
        d["per_candidate_accuracy"][f"{i}:{name}"] = round(ok / len(examples), 6)
    labels = [ex["label"] for ex in examples]
    p = sum(labels) / len(labels)
    d["n_examples"] = len(examples)
    d["label_rate"] = round(p, 6)
    d["best_constant"] = round(max(p, 1 - p), 6)
    accs = list(d["per_candidate_accuracy"].values())
    d["uniform_random_over_space"] = round(sum(accs) / len(accs), 6)
    d["whole_space_mean"] = d["uniform_random_over_space"]
    d["stored_vector_accuracy"] = d["per_candidate_accuracy"]["0:" +
                                  program.nodes[0].candidates[0].operator.name]
    d["artifact_digest"] = S.instantiate(schema_name, width, at, {"n0": 0})[1].digest
    return d


def certification(d):
    return C.Certification(width=d["width"], space_size=d["space_size"],
                           evaluated=d["evaluated"], exhausted=d["exhausted"],
                           conforming=d["conforming"], certificate=d["certificate"],
                           mean_return=d["stored_vector_accuracy"], threshold=1.0,
                           artifact_digest=d["artifact_digest"])


def record_for(qualified, sweeps, selections, free_nodes, provenance):
    return C.ClassRecord(
        class_id=C.schema_class_id(source_fingerprint(), qualified, selections),
        kind="schema",
        schema={"module": "research/motif-unification/schema.py",
                "qualified_name": qualified,
                "free_nodes": tuple(tuple(x) for x in free_nodes),
                "source_fingerprint": source_fingerprint()},
        selections=selections,
        certified=tuple(certification(d) for d in sweeps),
        members=tuple(d["artifact_digest"] for d in sweeps),
        provenance=provenance)


def try_admit(store, record, label):
    try:
        store.admit(record)
        return {"arm": label, "admitted": True, "refusal": None}
    except C.ClassError as e:
        return {"arm": label, "admitted": False, "refusal": str(e)}


def main():
    if STORE.exists():
        shutil.rmtree(STORE)
    report = {"sweeps": [], "admission": [], "arm_F": []}

    # ---- certify the right schema at two widths, on each domain's own episodes
    train = {w: examples_for(w, 0, "train", seed=11 + w) for w in (128, 3072)}
    held = {w: examples_for(w, 500, "test", seed=97 + w) for w in (128, 3072)}
    right = []
    for w in (128, 3072):
        d = sweep("M3", w, train[w])
        h = sweep("M3", w, held[w])
        d["held_out_stored_vector_accuracy"] = h["stored_vector_accuracy"]
        d["held_out_best_constant"] = h["best_constant"]
        d["held_out_conforming"] = h["conforming"]
        d["held_out_certificate"] = h["certificate"]
        d["held_out_n_examples"] = h["n_examples"]
        right.append(d)
        report["sweeps"].append(d)

    store = C.ClassStore(STORE)
    rec = record_for(QUALIFIED, right, {"n0": 0}, right[0]["free_nodes"],
                     {"derived_from": ["language/128", "visual/3072"],
                      "motif": "eq(index(buffer, add(base, offset)), other)",
                      "section": "FINDINGS 58"})
    report["admission"].append(try_admit(store, rec, "M3, certified at 128 and 3072"))
    report["class_id"] = rec.class_id

    # R1: the same record with only one certification
    one = record_for(QUALIFIED, right[:1], {"n0": 0}, right[0]["free_nodes"], {})
    report["admission"].append(try_admit(C.ClassStore(OUT / "scratch1"), one,
                                         "M3, certified at 128 only (R1 probe)"))

    # M2: bit-identical at three widths, but no free node anywhere
    m2free = [(n.name, len(n.candidates))
              for n in S.SCHEMAS["M2"](128, S.address_type(S.SETTINGS["language"]["address"]))[1].nodes
              if len(n.candidates) > 1]
    m2certs = []
    for w in (128, 3072, 4096):
        dom = DOMAIN_OF_WIDTH[w]
        at = S.address_type(S.SETTINGS[dom]["address"])
        m2certs.append({"width": w, "space_size": 1, "evaluated": 1, "exhausted": True,
                        "conforming": 1, "certificate": "unique",
                        "stored_vector_accuracy": 1.0,
                        "artifact_digest": S.instantiate("M2", w, at, {})[1].digest})
    m2rec = record_for("research.motif-unification.schema:m2_schema", m2certs, {},
                       m2free, {"note": "no free node at any width"})
    report["admission"].append(try_admit(C.ClassStore(OUT / "scratch2"), m2rec,
                                         "M2, bit-identical at 128/3072/4096, empty vector"))
    report["M2_free_nodes"] = m2free
    report["M2_digests"] = [c["artifact_digest"] for c in m2certs]

    # ---- arm F
    for name, qual, label in ((("M3_wrong_hardcoded"), WRONG_A, "F-a hard-coded width"),
                              (("M3_wrong_reordered"), WRONG_B, "F-b width-dependent order")):
        rows = []
        for w in (128, 3072):
            try:
                d = sweep(name, w, train[w])
            except Exception as e:
                d = {"width": w, "domain": DOMAIN_OF_WIDTH[w], "schema": name,
                     "build_error": f"{type(e).__name__}: {e}",
                     "space_size": None, "evaluated": 0, "exhausted": False,
                     "conforming": 0, "certificate": "none",
                     "stored_vector_accuracy": 0.0, "artifact_digest": "",
                     "per_candidate_accuracy": {}, "best_constant": None,
                     "free_nodes": []}
            rows.append(d)
        wrec = record_for(qual, rows, {"n0": 0},
                          rows[0].get("free_nodes") or [["n0", 5]], {"arm": label})
        entry = try_admit(C.ClassStore(OUT / f"scratchF_{name}"), wrec, label)
        entry["sweeps"] = rows
        # a single-width record of the same wrong schema, to show the certificate
        # at the agreeing width is indistinguishable from the right one
        single = record_for(qual, rows[:1], {"n0": 0},
                            rows[0].get("free_nodes") or [["n0", 5]], {"arm": label})
        entry["single_width"] = try_admit(C.ClassStore(OUT / f"scratchF1_{name}"),
                                          single, label + ", one width")
        entry["digest_at_128_equals_right_schema"] = (
            rows[0].get("artifact_digest") == right[0]["artifact_digest"])
        entry["certificate_at_128"] = rows[0]["certificate"]
        entry["stored_vector_accuracy"] = [r["stored_vector_accuracy"] for r in rows]
        report["arm_F"].append(entry)

    for p in ("scratch1", "scratch2", "scratchF_M3_wrong_hardcoded",
              "scratchF_M3_wrong_reordered", "scratchF1_M3_wrong_hardcoded",
              "scratchF1_M3_wrong_reordered"):
        shutil.rmtree(OUT / p, ignore_errors=True)

    (OUT / "step3.json").write_text(json.dumps(report, indent=1))
    return report


if __name__ == "__main__":
    rep = main()
    print("CERTIFICATION")
    for d in rep["sweeps"]:
        print(f"  {d['domain']:9s} w={d['width']:5d} space={d['space_size']} "
              f"evaluated={d['evaluated']} exhausted={d['exhausted']} "
              f"conforming={d['conforming']} cert={d['certificate']} "
              f"vector={d['stored_vector_accuracy']} const={d['best_constant']} "
              f"rand={d['uniform_random_over_space']} held={d['held_out_stored_vector_accuracy']}")
        print("      per candidate:", d["per_candidate_accuracy"])
    print("\nADMISSION")
    for a in rep["admission"]:
        print(f"  {'ADMITTED' if a['admitted'] else 'REFUSED '} {a['arm']}")
        if a["refusal"]:
            print(f"      {a['refusal'][:160]}")
    print("\nARM F")
    for a in rep["arm_F"]:
        print(f"  {a['arm']}: bit-identical at 128 = {a['digest_at_128_equals_right_schema']}, "
              f"certificate at 128 = {a['certificate_at_128']}, "
              f"stored vector accuracy {a['stored_vector_accuracy']}")
        print(f"      two widths: {'ADMITTED' if a['admitted'] else 'REFUSED'} "
              f"{(a['refusal'] or '')[:140]}")
        s = a["single_width"]
        print(f"      one width : {'ADMITTED' if s['admitted'] else 'REFUSED'} "
              f"{(s['refusal'] or '')[:140]}")
