"""Build one domain's corpus and decide every typed edit of every admitted case.

    python run_domain.py <domain> [--shard i --of n] [--max-cases k]

Phases, in order (PREREGISTRATION sections 2.2, 2.3, 3, 5):

  1. the base scaffold, decided on the training episodes and on all episodes;
  2. the defect generator, deterministic over the base's own sites;
  3. **admission**: a defect enters the corpus only with `conforming_on_train ==
     0`, `exhausted`, certificate `complete` -- a proof, not a timeout;
  4. for every admitted case, every typed edit is applied and decided twice:
     on the training episodes (the deployed-protocol label, M2) and on the
     training *and* held-out episodes (the repair label, M1).

Nothing here reads an arm, an ordering or a prior.  Features are computed from
the failed scaffold and the **training** episodes only.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import kit
import domains
import edits as E
from tcn.search import enumerate_prefix, space_size

TOL = 1e-6
MAX_SPACE = 400_000
MAX_PROGRAMS = 1 << 26

SELECTIVE = {"min", "max", "reduce_min", "reduce_max"}
ARITHMETIC = {"add", "sub", "mul", "div", "pow", "mod", "idiv", "shl", "shr",
              "neg", "abs", "exp", "log", "sin", "cos", "sqrt", "atan2",
              "sum", "mean", "count"}
LOGICAL = {"and", "or", "xor", "nand", "nor", "xnor", "not", "mux"}
COMPARISON = {"eq", "lt", "le", "gt", "ge"}
STRUCT = {"tuple", "project", "index", "member", "insert", "remove", "union",
          "intersection", "pair", "join", "map", "filter"}
CONVERT = {"encode", "decode", "quantize", "dequantize", "pack", "unpack", "interpret"}


def op_class(name):
    if name is None:
        return "none"
    if name in SELECTIVE:
        return "selective"
    if name in ARITHMETIC:
        return "arithmetic"
    if name in LOGICAL:
        return "logical"
    if name in COMPARISON:
        return "comparison"
    if name in STRUCT:
        return "structural"
    if name in CONVERT:
        return "conversion"
    if name == "identity":
        return "identity"
    return "module" if name.startswith("module:") else "other"


def decide(program, examples, signals, registry):
    """Existence of a conforming member: a witness, or exhaustion."""
    s = space_size(program)
    if s > MAX_SPACE:
        return {"space": s, "decided": False, "conforming": None,
                "exhausted": False, "certificate": "undecided", "witness": None}
    res = enumerate_prefix(program, examples, signals, registry, tolerance=TOL,
                           max_programs=MAX_PROGRAMS, stop_at_first=True)
    got = res.conforming > 0
    return {"space": s, "decided": True, "conforming": bool(got),
            "exhausted": bool(res.exhausted),
            "certificate": "witness" if got else res.certificate,
            "witness": res.selections if got else None}


def constant_site(program, registry, site, train):
    """Is the site's first candidate constant across the training batch?

    Section 47's probe.  Section 50 showed it does not generalise; it is recorded
    so this track's own measurement of it is on the record.
    """
    nd = next((x for x in program.nodes if x.name == site), None)
    if nd is None:
        return None
    sel = {x.name: 0 for x in program.nodes}
    seen = set()
    for ex in train:
        try:
            _, _, trace = program.execute(ex["inputs"], registry=registry, selections=sel)
        except Exception:                                        # noqa: BLE001
            return None
        seen.add(tuple(trace[site].flat()))
        if len(seen) > 1:
            return False
    return True


def features(program, registry, edit, edited, train, const_cache):
    nd = next((x for x in program.nodes if x.name == edit.site), None)
    base_n = len(nd.candidates) if nd else 0
    new_nd = next((x for x in edited.nodes if x.name == edit.site), None)
    added = (len(new_nd.candidates) - base_n) if new_nd else 0
    if edit.family == "ADD_NODE":
        fresh = [x for x in edited.nodes if x.name.startswith("ind_")]
        added += sum(len(x.candidates) for x in fresh)
    if edit.family == "ADD_PATH":
        fresh = [x for x in edited.nodes if x.name.startswith("path_")]
        added += sum(len(x.candidates) for x in fresh)
    depth_max = max(x.depth for x in program.nodes)
    present = {c.operator.name for x in program.nodes for c in x.candidates}
    arities = sorted({len(c.operator.inputs) for c in (new_nd.candidates if new_nd else ())})
    if edit.site not in const_cache:
        const_cache[edit.site] = constant_site(program, registry, edit.site, train)
    return {
        "family": edit.family,
        "added_candidates": max(0, added),
        "log_added": round(math.log1p(max(0, added)), 4),
        "log_space": round(math.log10(max(1, space_size(edited))), 4),
        "op_class": op_class(edit.operator),
        "novel_operator": bool(edit.operator is not None and edit.operator not in present),
        "site_depth_frac": round((nd.depth / depth_max) if nd and depth_max else 0.0, 4),
        "site_is_output": bool(nd is not None and nd.name == program.outputs[0][1]),
        "site_constant_on_train": const_cache[edit.site],
        "max_arity": max(arities) if arities else 0,
    }


def enumerate_defects(base, registry, sites):
    out = []
    for site in sorted(sites):
        nd = next((x for x in base.nodes if x.name == site), None)
        if nd is None:
            continue
        ops = sorted({c.operator.name for c in nd.candidates})
        ports = sorted({s for c in nd.candidates for s in c.sources})
        for op in ops:
            out.append(("drop_operator", site, op))
        for port in ports:
            out.append(("drop_source", site, port))
        for k in (1, 2, 3, 4, 6, 8):
            out.append(("keep_prefix", site, k))
        out.append(("delete_node", site, None))
    return out


APPLY = {"drop_operator": lambda p, r, s, a: E.drop_operator(p, r, s, a),
         "drop_source": lambda p, r, s, a: E.drop_source(p, r, s, a),
         "keep_prefix": lambda p, r, s, a: E.keep_prefix(p, r, s, a),
         "delete_node": lambda p, r, s, a: E.delete_node(p, r, s)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("domain")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=24)
    a = ap.parse_args()

    kit.check_floor(f"run_domain:{a.domain}:shard{a.shard}")
    t0 = time.perf_counter()
    d = domains.BUILDERS[a.domain]()
    base, r, sig = d["program"], d["registry"], d["signals"]
    train, held = d["train"], d["heldout"]
    allep = train + held

    report = {"domain": a.domain, "note": d["note"],
              "n_train": len(train), "n_heldout": len(held),
              "base": {"nodes": len(base.nodes), "space": space_size(base),
                       "on_train": decide(base, train, sig, r),
                       "on_all": decide(base, allep, sig, r)},
              "max_space": MAX_SPACE, "tolerance": TOL,
              "max_new": E.MAX_NEW, "max_new_node": E.MAX_NEW_NODE,
              "defects_tried": 0, "admitted": 0,
              "rejected_solvable_on_train": 0, "rejected_no_repair": 0,
              "rejected_invalid": 0, "cases": []}

    defects = enumerate_defects(base, r, [nd.name for nd in base.nodes])
    report["defects_tried"] = len(defects)
    mine = [x for i, x in enumerate(defects) if i % a.of == a.shard]

    for kind, site, arg in mine:
        if report["admitted"] >= a.max_cases:
            break
        failed = APPLY[kind](base, r, site, arg)
        if failed is None:
            report["rejected_invalid"] += 1
            continue
        tr = decide(failed, train, sig, r)
        if not tr["decided"] or tr["conforming"] or not tr["exhausted"]:
            report["rejected_solvable_on_train"] += 1
            continue
        case_id = f"{a.domain}:{kind}:{site}:{arg}"
        es = E.enumerate_edits(failed, r)
        const_cache = {}
        rows = []
        for edit, edited in es:
            on_all = decide(edited, allep, sig, r)
            if not on_all["decided"]:
                on_train = {"decided": False, "conforming": None,
                            "certificate": "undecided"}
            elif on_all["conforming"]:
                # a member conforming on every episode conforms on the training
                # ones, so the second decision is skipped, not guessed
                on_train = {"decided": True, "conforming": True,
                            "certificate": "implied by the all-episode witness"}
            else:
                on_train = decide(edited, train, sig, r)
            rows.append({
                "key": edit.key, **edit.to_dict(),
                "space": on_all["space"], "decided": on_all["decided"],
                "repair": on_all["conforming"],
                "certificate_all": on_all["certificate"],
                "train_conforming": on_train["conforming"],
                "certificate_train": on_train["certificate"],
                "witness": on_all["witness"],
                "is_inverse": E.is_inverse((kind, site, arg), edit),
                "truncated": edit.key in E.TRUNCATED,
                "features": features(failed, r, edit, edited, train, const_cache)})
        n_rep = sum(1 for x in rows if x["repair"])
        if n_rep == 0:
            report["rejected_no_repair"] += 1
            report["cases"].append({"case_id": case_id, "admitted": False,
                                    "reason": "no repair in the edit space",
                                    "defect": {"kind": kind, "site": site, "arg": arg},
                                    "failed": tr, "n_edits": len(rows),
                                    "n_repairs": 0,
                                    "n_undecided": sum(1 for x in rows if not x["decided"])})
            continue
        report["admitted"] += 1
        report["cases"].append({
            "case_id": case_id, "admitted": True,
            "defect": {"kind": kind, "site": site, "arg": arg},
            "failed": tr, "n_edits": len(rows), "n_repairs": n_rep,
            "n_undecided": sum(1 for x in rows if not x["decided"]),
            "n_train_conforming": sum(1 for x in rows if x["train_conforming"]),
            "edits": rows})
        print(f"  {case_id}: {len(rows)} edits, {n_rep} repairs, "
              f"{report['cases'][-1]['n_undecided']} undecided", flush=True)

    report["seconds"] = time.perf_counter() - t0
    report["peak_rss_gb"] = round(kit.peak_rss_gb(), 3)
    name = f"cases_{a.domain}" + (f"_s{a.shard}" if a.of > 1 else "")
    print("wrote", kit.dump(name, report))


if __name__ == "__main__":
    main()
