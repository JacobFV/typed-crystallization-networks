"""Q2 arms: can §52's pooled abstraction be registered as a class and inherited?

    .venv/bin/python research/class-identity/run_q2.py partition members inherit compat

§52's library at `research/semantic-library/library` is **copied** into
`out/library_q2` and only the copy is written to. `tcn/` is not modified.
"""
from __future__ import annotations

import itertools
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEM = ROOT / "research" / "semantic-library"
PREMIN = ROOT / "research" / "premin-abstraction"
EARNED = ROOT / "research" / "earned-abstraction"
for p in (str(HERE), str(SEM), str(PREMIN), str(EARNED), str(ROOT)):
    sys.path.insert(0, p)

import classes as C                                               # noqa: E402
from tcn.library import Library                                   # noqa: E402
from tcn.operators import Registry                                # noqa: E402
from tcn.types import BOOL, Value                                 # noqa: E402

OUT = HERE / "out"
SRC_LIB = SEM / "library"
COPY_LIB = OUT / "library_q2"


def _copy():
    if COPY_LIB.exists():
        shutil.rmtree(COPY_LIB)
    shutil.copytree(SRC_LIB, COPY_LIB)
    return Library(COPY_LIB)


def _load_program(lib, digest):
    r = Registry()
    entry = lib.by_digest(digest)
    lib._load_module(entry, r, "revalidate", set())
    return r.modules[entry.operator], r


def truth_table(program, registry):
    """Extensional key, computable only over a small finite domain."""
    if len(program.inputs) > 4 or any(t != BOOL for _, t in program.inputs):
        return None
    rows = []
    for bits in itertools.product((False, True), repeat=len(program.inputs)):
        out, _ = program.run({k: Value.of(BOOL, v) for (k, _), v in zip(program.inputs, bits)},
                             registry=registry)
        rows.append(bool(next(iter(out.values())).decoded))
    return tuple(rows)


# ------------------------------------------------------------------- arm B1
def arm_partition():
    """The class partition of §52's shipped library, as it actually stands."""
    lib = _copy()
    digests, rows = [], []
    for e in lib.entries:
        if e.digest not in digests:
            digests.append(e.digest)
    for d in digests:
        program, r = _load_program(lib, d)
        tt = truth_table(program, r)
        cid = C.extensional_class_id(len(program.inputs), tt) if tt else None
        names = sorted({e.name for e in lib.entries if e.digest == d})
        rows.append({"digest": d, "arity": len(program.inputs),
                     "truth_table": [int(v) for v in tt] if tt else None,
                     "class_id": cid, "nodes": len(program.nodes),
                     "execution_cost": program.execution_cost(r),
                     "description_bits": program.description_bits(r),
                     "published_under": names, "entries": len(names)})
    by_class = {}
    for row in rows:
        by_class.setdefault(row["class_id"], []).append(row["digest"])
    report = {"library": str(SRC_LIB.relative_to(ROOT)),
              "manifest_entries": len(lib.entries),
              "distinct_digests": len(digests),
              "distinct_classes": len(by_class),
              "classes_with_multiple_members": {k: v for k, v in by_class.items() if len(v) > 1},
              "digests_per_class": {k: len(v) for k, v in by_class.items()},
              "rows": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
    for row in rows:
        print(" ", row["digest"][:12], "arity", row["arity"], "nodes", row["nodes"],
              row["class_id"], "published under", row["entries"], "names", flush=True)
    (OUT / "q2_partition.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------- arm B2
def arm_members():
    """The pool §52's miner computed and discarded, and its cost spread.

    `mine_semantic.propose` elects `min(reps, key=(len(nodes), digest))` and
    publishes that one circuit. Everything else in the pool is dropped at the
    library boundary. This arm recovers the pool for the rank-1 majority class.
    """
    import family                                                  # noqa: E402
    import mine                                                    # noqa: E402
    import mine_semantic                                           # noqa: E402

    table = json.loads((SEM / "out" / "mined_maj_minall_sem.json").read_text())
    target = next(r for r in table["ranked"] if r["is_window"])
    key = (target["arity"], tuple(bool(v) for v in target["truth_table"]))

    corpus, _task_of = family.corpus_for(table["family"], table["variant"], table["excluded"])
    cache, reps = {}, {}
    for _entry, program in corpus.items():
        for _root, _S, canon, _holes in mine.fragments(program, mine_semantic.MAX_NODES,
                                                       mine_semantic.MAX_HOLES):
            if (len(canon.inputs), mine_semantic.truth_table(canon, cache)) == key:
                reps.setdefault(canon.digest, canon)

    rows = []
    for digest, canon in sorted(reps.items()):
        r = Registry()
        rows.append({"digest": digest, "nodes": len(canon.nodes),
                     "execution_cost": canon.execution_cost(r),
                     "description_bits": canon.description_bits(r),
                     "body": [[n.candidates[0].operator.name, n.name,
                               list(n.candidates[0].sources)] for n in canon.nodes]})
    elected = min(rows, key=lambda x: (x["nodes"], x["digest"]))
    report = {
        "class_id": C.extensional_class_id(*key),
        "recorded_circuits_pooled": target["circuits_pooled"],
        "recovered_members": len(rows),
        "elected_by_miner": elected["digest"],
        "published_by_section_52": target["digest"],
        "node_counts": sorted({x["nodes"] for x in rows}),
        "execution_costs": sorted({x["execution_cost"] for x in rows}),
        "description_bits": sorted({x["description_bits"] for x in rows}),
        "cost_axes_agree_on_preferred": len({
            min(rows, key=lambda x: (x[axis], x["digest"]))["digest"]
            for axis in ("nodes", "execution_cost", "description_bits")}) == 1,
        "members": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "members"}, indent=2), flush=True)
    (OUT / "q2_members.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------- arm B3
def arm_inherit():
    """Two distinct digests, one class: the mined circuit and the hand-authored one."""
    import later                                                   # noqa: E402
    lib = _copy()
    r = Registry()
    authored = later.hand_authored(r, "maj")
    fixture = [{k: Value.of(BOOL, v) for (k, _), v in zip(authored.inputs, bits)}
               for bits in itertools.product((False, True), repeat=len(authored.inputs))]
    entry = lib.publish("hand_maj3", authored, Registry(), fixture=fixture,
                        provenance={"source": "later.hand_authored('maj')",
                                    "arm": "FINDINGS 52 arm3_authored (the ceiling)"})
    mined = "165bc290d9c82b70a8ea3cc2"
    rows = {}
    for label, digest in (("mined_rank1", mined), ("hand_authored", entry.digest)):
        program, reg = _load_program(lib, digest)
        tt = truth_table(program, reg)
        rows[label] = {"digest": digest, "arity": len(program.inputs),
                       "class_id": C.extensional_class_id(len(program.inputs), tt),
                       "nodes": len(program.nodes),
                       "execution_cost": program.execution_cost(reg),
                       "description_bits": program.description_bits(reg),
                       "body": [[n.candidates[0].operator.name, n.name,
                                 list(n.candidates[0].sources)] for n in program.nodes]}
    a, b = rows["mined_rank1"], rows["hand_authored"]
    cid = a["class_id"]
    store = C.ClassStore(COPY_LIB)
    preferred = {axis: min((a, b), key=lambda x: (x[axis], x["digest"]))["digest"]
                 for axis in ("nodes", "execution_cost", "description_bits")}
    record = C.ClassRecord(class_id=cid, kind="extensional",
                           members=(a["digest"], b["digest"]), preferred=preferred,
                           provenance={"identity": "(arity, truth table)",
                                       "from": "FINDINGS 52 arm2s_semantic + arm3_authored",
                                       "measured_capability_l1_conforming": {
                                           a["digest"]: 144, b["digest"]: 144}})
    store.admit(record)
    report = {"rows": rows,
              "digest_separates": a["digest"] != b["digest"],
              "class_unifies": C.same_class(a["class_id"], b["class_id"]),
              "class_id": cid, "preferred": preferred,
              "cost_axes_agree_on_preferred": len(set(preferred.values())) == 1,
              "recorded_section_52_capability": {
                  "arm2s_semantic (165bc290)": {"conforming": 144, "tight": "18/24",
                                                "wide": "8/8", "median_accuracy": 1.0},
                  "arm3_authored (hand)": {"conforming": 144, "tight": "18/24",
                                           "wide": "8/8", "median_accuracy": 1.0}},
              "verify_after_publish": lib.verify()}
    ok = all(row["ok"] for row in report["verify_after_publish"])
    report["library_verify_all_ok"] = ok
    print(json.dumps({k: v for k, v in report.items() if k != "verify_after_publish"},
                     indent=2), flush=True)
    print("library verify all ok:", ok, flush=True)
    (OUT / "q2_inherit.json").write_text(json.dumps(report, indent=2))
    return report


# ------------------------------------------------------------------- arm B4
def _probe(root):
    """Everything `tcn.library` promises, exercised on one library root."""
    lib = Library(root)
    out = {"entries_loaded": len(lib.entries), "names": list(lib.names())}
    out["verify"] = [{k: row.get(k) for k in
                      ("reference", "digest", "loads", "fixture_cases",
                       "fixture_reproduces", "source_current", "stale", "ok", "error",
                       "execution_cost", "description_bits")}
                     for row in lib.verify()]
    out["verify_all_ok"] = all(row["ok"] for row in out["verify"])
    loads, errors = [], []
    for name in lib.names():
        for policy in ("strict", "revalidate"):
            try:
                _, aliases = lib.load([name], registry=Registry(), policy=policy)
                loads.append({"name": name, "policy": policy, "operator": aliases[name]})
            except Exception as exc:                                # reported, never swallowed
                errors.append({"name": name, "policy": policy,
                               "error": f"{type(exc).__name__}: {str(exc)[:80]}"})
    out["loads"] = loads
    out["errors"] = errors
    return lib, out


def arm_compat():
    """Does an unmodified `tcn.library.Library` tolerate an extra manifest field?

    This is the forward-compatibility half of "additive, off by default": a
    manifest written by a future version that records `semantic_id` per entry
    must still load, verify and replay under the version shipped today, with
    `tcn/` receiving no diff.

    The control is the same probe on an **unstamped** copy of the same library.
    §52's modules were built against an earlier `source_fingerprint`, so
    `policy="strict"` is *expected* to raise `SourceRevisionMismatch` on both
    copies -- that is the shipped guard working. The claim is only that the two
    copies behave **identically**, so nothing observed is caused by the field.
    """
    stamped_root, control_root = OUT / "library_compat", OUT / "library_control"
    for root in (stamped_root, control_root):
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(SRC_LIB, root)
    manifest = stamped_root / "manifest.json"
    d = json.loads(manifest.read_text())
    for e in d["entries"]:
        e["semantic_id"] = f"tt/{len(e['inputs'])}/unknown"
        e["class_members"] = [e["digest"]]
    manifest.write_text(json.dumps(d, indent=2, sort_keys=True))

    lib, stamped = _probe(stamped_root)
    _, control = _probe(control_root)
    report = {"entries_stamped": len(d["entries"]),
              "extra_keys": ["semantic_id", "class_members"],
              "stamped": stamped, "control": control}
    report["identical_to_control"] = {
        k: stamped[k] == control[k]
        for k in ("entries_loaded", "names", "verify", "verify_all_ok", "loads", "errors")}
    report["all_identical"] = all(report["identical_to_control"].values())
    # Round-trip: today's `Entry.from_dict` reads named keys, so a save rewrites
    # the manifest from the fields it knows. Whether the extra keys survive is a
    # fact about the format, and it is recorded either way.
    lib._save()
    after = json.loads(manifest.read_text())
    report["extra_keys_survive_save"] = all("semantic_id" in e for e in after["entries"])
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("stamped", "control")}, indent=2), flush=True)
    print("stamped errors:", len(stamped["errors"]), "control errors:", len(control["errors"]),
          flush=True)
    print("example error:", (stamped["errors"] or [{}])[0], flush=True)
    (OUT / "q2_compat.json").write_text(json.dumps(report, indent=2))
    return report


ARMS = {"partition": arm_partition, "members": arm_members,
        "inherit": arm_inherit, "compat": arm_compat}

if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:]:
        ARMS[name]()
