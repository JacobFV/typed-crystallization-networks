"""Re-derive every claim this track has made, from raw JSON under `out/`.

Prints PASS/FAIL per claim, writes `out/verify.json` and `out/headline.json`,
and exits non-zero on any FAIL. Three rules it is built to obey:

1. **Re-derive, never re-read.** A claim is recomputed here from the raw
   fields, without importing `phase0.py`, so a bug in a run script cannot make
   its own claim true.
2. **Provenance first (V9).** Before any figure is read, every artifact under
   `out/` must carry a `stamp.provenance` block that agrees with the inputs,
   parameters and sources on disk *now*. An **absent** block fails exactly as a
   disagreeing one does; otherwise an artifact written by a job that was still
   running when the directory was cleaned silently counts as current, which is
   how a superseded corpus was nearly reported as this project's result.
3. **Promises are checks (V10).** `check_promises.py` fails when a promise in
   `promises.json` whose phase has run has no implementing `verify_*` function
   here. Every function named in the registry is defined in this file.

Functions are named `verify_<id>_<slug>` and the registry maps promise ids onto
them; adding a promise without adding its function fails V10.

    python verify.py
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import stamp

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"
FLAGSHIP = ROOT / "research" / "integrated-flagship"
results = []

# The declared constants of PREREGISTRATION §2.1 and §3.2, retyped here so a
# check does not import the file it is checking.
S0_OFFSETS = (6, 51, 3, 48, 9)
WIDEN_SOURCE = tuple(sorted(set(range(16)) | {1, 2, 3} | set(S0_OFFSETS)))
RELATIONS = ("above", "left_of", "right_of")


def claim(name, ok, detail=""):
    results.append({"pass": bool(ok), "claim": name, "detail": str(detail)})


def J(path):
    p = pathlib.Path(path)
    return json.loads(p.read_text()) if p.exists() else None


# ------------------------------------------------------------------ V9, V10
#: The verifier's own outputs, which this run is about to overwrite. They are
#: stamped like everything else — for the merge gate to check — but scanning
#: last run's copy here would fail whenever `verify.py` itself changed, which
#: is precisely the run that regenerates them. The exemption is narrow, named,
#: and reported as a claim so it cannot become a silent hole.
SELF_WRITTEN = ("verify.json", "headline.json")


def verify_v9_provenance():
    """Every artifact under out/ carries a stamp that agrees with disk now."""
    artifacts = sorted(p for p in OUT.glob("*.json") if p.name not in SELF_WRITTEN)
    claim("V9 provenance: the only artifacts exempt from the scan are this run's own",
          True, ", ".join(SELF_WRITTEN))
    if not artifacts:
        claim("V9 provenance: at least one artifact exists", False, "out/ holds no JSON")
        return
    for path in artifacts:
        try:
            blob = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            claim(f"V9 provenance: {path.name} is JSON", False, exc)
            continue
        try:
            checked = stamp.present(blob)
            claim(f"V9 provenance: {path.name} carries a stamp", True,
                  f"{len(checked)} recorded input/source field(s)")
        except stamp.StampError as exc:
            claim(f"V9 provenance: {path.name} carries a stamp", False, exc)
            continue
        if blob["provenance"].get("kind") == "phase0":
            verify_v9_phase0_inputs(path, blob)


def verify_v9_phase0_inputs(path, blob):
    """phase0's stamp must still describe the episode caches on disk now."""
    inputs, sources = {}, []
    for gap in (0, 1):
        cache = FLAGSHIP / "out" / f"episodes_gap{gap}.json"
        inputs[f"episodes_gap{gap}.file"] = stamp.digest_file(cache)
    for name in ("phase0.py", "stamp.py"):
        sources.append(HERE / name)
    try:
        checked = stamp.require(blob, kind="phase0", inputs=inputs, sources=sources)
        claim(f"V9 provenance: {path.name} inputs and sources unchanged", True,
              f"{len(checked)} field(s) compared")
    except stamp.StampError as exc:
        claim(f"V9 provenance: {path.name} inputs and sources unchanged", False, exc)


def verify_s1_provenance_standard():
    """The owner's permanent standard (2026-09-11), on every artifact.

    base scaffold/program digest, split/corpus digest, mutation-grammar version
    digest, producing commit, pre-registration revision. An explicit absence
    with a reason is acceptable where an omission is not — a missing field
    reads as "not recorded" and would let a later artifact inherit the silence.
    """
    for path in sorted(p for p in OUT.glob("*.json") if p.name not in SELF_WRITTEN):
        try:
            p = json.loads(path.read_text())["provenance"]
        except (json.JSONDecodeError, KeyError, TypeError):
            claim(f"S1 standard: {path.name} has a readable provenance block", False, path.name)
            continue
        for field in stamp.STANDARD:
            claim(f"S1 standard: {path.name} records {field}", field in p,
                  "present" if field in p else "OMITTED")
        g = p.get("grammar_digest") or {}
        claim(f"S1 standard: {path.name} records the mutation-grammar version, "
              f"or an explicit absence with a reason",
              g.get("digest") is not None or bool(g.get("absent_because")),
              g.get("digest") or g.get("absent_because"))
        claim(f"S1 standard: {path.name}'s revision matches this code's",
              p.get("revision") == stamp.REVISION,
              f"{p.get('revision')} vs {stamp.REVISION}")
        claim(f"S1 standard: {path.name} records a producing commit",
              bool((p.get("commit") or {}).get("head")),
              (p.get("commit") or {}).get("head", "")[:12])
        claim(f"S1 standard: {path.name} records the base scaffold digest",
              p.get("base_digest") is not None, (p.get("base") or {}).get("schema"))
        claim(f"S1 standard: {path.name} records the split digests, "
              f"including the blind ones it did not read",
              p.get("splits_digest") is not None and "gap2.final" in (p.get("splits") or {}),
              sorted(p.get("splits") or {}))


def verify_v11_guards_bite():
    """The provenance guards are driven to their failure states and must raise.

    A guard nobody has seen fail is a guard nobody has tested; both incidents
    this machinery exists to prevent were checks that passed on an object they
    were not actually checking.
    """
    run = subprocess.run([sys.executable, str(HERE / "selftest_guards.py")],
                         capture_output=True, text=True, cwd=str(HERE), timeout=300)
    claim("V11 guards: every provenance guard refuses its failure case",
          run.returncode == 0,
          (run.stdout + run.stderr).strip().splitlines()[-1:])


def verify_v10_promises_kept():
    """check_promises.py exits 0: every active promise has an implementing check."""
    run = subprocess.run([sys.executable, str(HERE / "check_promises.py")],
                         capture_output=True, text=True, cwd=str(HERE), timeout=300)
    claim("V10 promises: every promise whose phase has run is implemented",
          run.returncode == 0, (run.stdout + run.stderr).strip().splitlines()[-1:])


# --------------------------------------------------------------------- V2
def verify_v2_expressivity():
    """The per-relation conforming-step sets, re-derived from the raw ranges."""
    d = J(OUT / "phase0.json")
    if d is None:
        claim("V2 expressivity: out/phase0.json exists", False, "missing")
        return
    v2 = d["V2_expressivity"]
    claim("V2: S0's offsets are §2.1's five",
          tuple(v2["s0_offsets"]) == S0_OFFSETS, v2["s0_offsets"])
    for gap in (0, 1):
        per = v2["per_gap"][f"gap{gap}"]
        claim(f"V2: gap {gap} covers all three relations",
              sorted(per) == sorted(RELATIONS), sorted(per))
        for relation in RELATIONS:
            row = per[relation]
            claim(f"V2: gap {gap} {relation} used 16 episodes",
                  row["n_episodes"] == 16, row["n_episodes"])
            # min_required_offset is the minimum over every reported range
            lows = [a for ranges in row["solution_ranges"].values() for a, _ in ranges]
            claim(f"V2: gap {gap} {relation} min-required re-derived from its ranges",
                  min(lows) == row["min_required_offset"],
                  f"{min(lows)} vs {row['min_required_offset']}")
            # s0_expressible is exactly "some S0 offset lies in some range"
            hits = [[spec, o] for spec, ranges in row["solution_ranges"].items()
                    for a, b in ranges for o in S0_OFFSETS if a <= o <= b]
            claim(f"V2: gap {gap} {relation} S0 coverage re-derived",
                  sorted(hits) == sorted(row["s0_solutions"])
                  and bool(hits) == row["s0_expressible"],
                  f"{sorted(hits)} vs {sorted(row['s0_solutions'])}")
    claim("V2: S0 solves every relation at gap 0",
          v2["s0_solves_gap0"] is True
          and all(v2["per_gap"]["gap0"][r]["s0_expressible"] for r in RELATIONS))
    ce = v2["counterexample"]
    claim("V2: the counterexample is `above` at gap 1, unsayable in S0",
          ce["relation"] == "above" and ce["configuration"] == "gap 1"
          and ce["s0_expressible"] is False and ce["holds"] is True)
    claim("V2: min required 55 > max S0 offset 51",
          ce["min_required_offset"] == 55 and ce["max_s0_offset"] == 51
          and ce["min_required_offset"] > ce["max_s0_offset"],
          f"{ce['min_required_offset']} > {ce['max_s0_offset']}")
    claim("V2: max S0 offset equals max of §2.1's pool",
          ce["max_s0_offset"] == max(S0_OFFSETS))


# --------------------------------------------------------------------- V3, F2
def verify_v3_widen_insufficiency():
    d = J(OUT / "phase0.json")
    if d is None:
        claim("V3 widening: out/phase0.json exists", False, "missing")
        return
    v3, v2 = d["V3_widen_insufficiency"], d["V2_expressivity"]
    claim("V3: the declared literal source L is §3.2's",
          tuple(v3["widen_source"]) == WIDEN_SOURCE, v3["widen_source"])
    reach = sorted(set(v3["widen_source"]) | set(S0_OFFSETS))
    claim("V3: reachable offsets re-derived as L union S0",
          list(v3["reachable_offsets"]) == reach)
    claim("V3: max reachable re-derived", v3["max_reachable"] == max(reach),
          f"{v3['max_reachable']} vs {max(reach)}")
    required = set()
    for ranges in v2["per_gap"]["gap1"]["above"]["solution_ranges"].values():
        for a, b in ranges:
            required.update(range(a, b + 1))
    claim("V3: reachable and required offsets are disjoint",
          sorted(set(reach) & required) == [] == list(v3["reachable_and_required"]))
    claim("V3: max(reachable) < min(required), by token equality on both integers",
          v3["max_reachable"] == 51 and v3["min_required_offset"] == 55
          and v3["max_reachable"] < v3["min_required_offset"],
          f"{v3['max_reachable']} < {v3['min_required_offset']}")
    claim("V3: widening-within-L insufficiency holds", v3["holds"] is True)
    # The scope qualifier is load-bearing (owner, 2026-09-11) and must be in
    # the artifact, not only in the prose that quotes it.
    claim("V3: the artifact records the scope qualifier",
          v3.get("scope") == "within S0's declared literal source only", v3.get("scope"))
    claim("V3: the artifact explicitly disclaims the stronger reading",
          "NOT a claim that widening cannot solve the task" in (v3.get("reading") or "")
          and v3.get("does_not_claim") == "that unrestricted widening fails",
          v3.get("does_not_claim"))


def verify_f2_no_design_collapse():
    """F2: WIDEN over L does not repair gap 1, so the design has not collapsed."""
    d = J(OUT / "phase0.json")
    if d is None:
        claim("F2: out/phase0.json exists", False, "missing")
        return
    v3 = d["V3_widen_insufficiency"]
    collapsed = bool(v3["reachable_and_required"]) or \
        v3["max_reachable"] >= v3["min_required_offset"]
    claim("F2: the experiment has NOT collapsed into `restore a candidate`",
          not collapsed,
          "a WIDEN literal reaches the repair" if collapsed else "no literal in L reaches it")


# --------------------------------------------------------------------- V8
def verify_v8_cost_model():
    """The affine fit quoted in §10, re-derived from the pilot's raw rows."""
    d = J(OUT / "phase0.json")
    if d is None or "V8_cost_pilot" not in d:
        claim("V8 cost model: the pilot ran", False, "no V8_cost_pilot block")
        return
    rows = d["V8_cost_pilot"]["rows"]
    claim("V8: at least three points, so the fit is over-determined", len(rows) >= 3,
          len(rows))
    for r in rows:
        claim(f"V8: n={r['n_step_candidates']} candidate count re-derived from offsets",
              r["n_step_candidates"] == r["n_offsets"] * 4,
              f"{r['n_step_candidates']} vs {r['n_offsets']} * 4")
    n = len(rows)
    xs = [r["n_step_candidates"] for r in rows]
    ys = [r["seconds"] for r in rows]
    mx, my = sum(xs) / n, sum(ys) / n
    denom = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom
    intercept = my - slope * mx
    residual = max(abs(y - (intercept + slope * x)) for x, y in zip(xs, ys))
    # Wall clock varies run to run on a shared host; a point estimate from one
    # run is not a reproducible claim and must not be verified as though it
    # were. §10 declares the band, and this checks the band. The *exact*
    # quantities the pilot also produces (the conformer counts) are checked for
    # equality instead, because those are reproducible.
    claim("V8: §10's declared band — intercept in [45, 85] s",
          45.0 <= intercept <= 85.0, f"{intercept:.1f} s")
    claim("V8: §10's declared band — slope in [0.8, 1.4] s per step candidate",
          0.8 <= slope <= 1.4, f"{slope:.4f} s")
    claim("V8: §10's residual bound (< 3 s) holds", residual < 3.0, f"{residual:.2f} s")
    claim("V8: the counter is affine in the step pool, not superlinear "
          "(the 80-candidate point is within 10% of the 20/40 extrapolation)",
          abs(ys[-1] - (intercept + slope * xs[-1])) / ys[-1] < 0.10,
          f"{ys[-1]:.1f} s measured")
    peak = max(r["peak_rss_gb"] for r in rows)
    claim("V8: §10's peak RSS bound (< 0.5 GB) and the floor it implies (< 8 GB, "
          "the declared MemoryMax)", peak < 0.5 and 20 * peak < 8.0,
          f"peak {peak:.3f} GB, floor {20 * peak:.1f} GB")
    # The exact quantity, which must NOT vary between runs.
    counts = {r["n_step_candidates"]: r["conformers_48ep"] for r in rows}
    expected = {20: "1376372736", 40: "2767343616", 80: "11224903680"}
    claim("V8: the pilot's exact 48-episode conformer counts, by token equality",
          all(counts.get(k) == v for k, v in expected.items()), counts)


CHECKS = (verify_v9_provenance, verify_s1_provenance_standard,
          verify_v10_promises_kept, verify_v11_guards_bite,
          verify_v2_expressivity, verify_v3_widen_insufficiency,
          verify_f2_no_design_collapse, verify_v8_cost_model)


def main():
    for check in CHECKS:
        try:
            check()
        except Exception as exc:                               # noqa: BLE001
            claim(f"{check.__name__} raised", False, repr(exc))
    failed = [r for r in results if not r["pass"]]
    for r in results:
        print(f"  {'PASS' if r['pass'] else 'FAIL'}  {r['claim']}"
              + (f"   [{r['detail']}]" if r["detail"] else ""))
    print(f"\n{len(results) - len(failed)} PASS, {len(failed)} FAIL")
    OUT.mkdir(exist_ok=True)
    scanned = {p.name: stamp.digest_file(p) for p in sorted(OUT.glob("*.json"))
               if p.name not in SELF_WRITTEN}
    prov = stamp.make("verify", scanned,
                      {"checks": [c.__name__ for c in CHECKS],
                       "exempt": list(SELF_WRITTEN)},
                      [HERE / "verify.py", HERE / "stamp.py",
                       HERE / "check_promises.py", HERE / "promises.json"])
    stamp.write(OUT / "verify.json",
                {"format": "schema-induction/verify-1",
                 "n_pass": len(results) - len(failed), "n_fail": len(failed),
                 "results": results}, prov)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
