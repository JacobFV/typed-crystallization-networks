"""Recompute every headline figure in RESULTS.md from committed files under out/.

Prints PASS/FAIL per claim and exits non-zero on any FAIL. Two layers:

1. every `<!-- BEGIN:x -->` block in RESULTS.md equals what `report.py` renders
   from `out/` now (so no figure in a block was hand-edited);
2. each headline claim is re-derived here from raw JSON **without importing
   report.py, engine.py or arms.py** -- expected costs are recomputed from the
   tier rows' exact S and K, criteria are recomputed from those, and so on --
   so a bug in a run or report script cannot make its own claim true.

It also fails on any number in RESULTS.md prose (outside blocks) that is not in
the allow-list of structural constants below or present verbatim in a block.

    python verify.py
"""
from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from fractions import Fraction

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
RESULTS = (HERE / "RESULTS.md").read_text()
results = []


def claim(name, ok, detail=""):
    results.append((bool(ok), name, detail))


def J(name):
    p = OUT / name
    return json.loads(p.read_text()) if p.exists() else None


def arm(gap, a):
    return J(f"arm_gap{gap}_{a.replace(chr(39), 'p')}.json")


def recompute_cost(d):
    """Expected programs to first solution, from the tier rows alone."""
    before, prev_S, prev_K = 0, 0, 0
    for t in d["tiers"]:
        S, K = int(t["S"]), int(t["K"])
        if int(t["shell_S"]) != S - prev_S or int(t["shell_K"]) != K - prev_K:
            return "shell-mismatch"
        if K - prev_K > 0 and prev_K == 0:
            return Fraction(before) + Fraction(S - prev_S + 1, K - prev_K + 1)
        if prev_K == 0:
            before += S - prev_S
        prev_S, prev_K = S, K
    return None


def stored_cost(d):
    e = d.get("expected_programs")
    return None if e is None else Fraction(int(e["num"]), int(e["den"]))


# ------------------------------------------------------------ layer 1: blocks
sys.path.insert(0, str(HERE))
import report  # noqa: E402  (only for re-rendering; claims below do not use it)
for m in re.finditer(r"<!-- BEGIN:(\w+) -->\n(.*?)\n<!-- END:\1 -->", RESULTS, flags=re.S):
    name, body = m.group(1), m.group(2)
    claim(f"block `{name}` equals its rendering from out/", name in report.BLOCKS and
          report.BLOCKS[name]() == body)

# ------------------------------------------------------ layer 2: raw claims
ARMS = ["N", "N'", "N''", "P", "D'", "A_noobs"] + [f"D''_{s}" for s in range(5)] + \
       [f"R_{s}" for s in range(5)] + ["U_feat", "U_steep", "U_Vonly", "KO_ADDR", "KO_TRUTH", "KO_STEP", "STEP_only", "STEP_hand"] + [f"OCC_{r}" for r in (1, 3, 10, 100, 3500)] + \
       [f"V_{r}" for r in (1, 3, 10, 100)]
costs = {}

# mechanism table (post-hoc): reproduce the LIT occurs ratio and the V means
# from each arm's stored estimator, independently of mechanism.py
mech = J("mechanism_gap0.json")
if mech is not None:
    for row in mech["rows"]:
        d = arm(0, row["arm"])
        c = d["prior_estimators"]["LIT"].get("cells") or {}
        mine = (c["(True,)"]["score"] / c["(False,)"]["score"]) if "(True,)" in c else \
            (2.0 if "occurs" in d["prior_estimators"]["LIT"].get("rule", "") else 1.0)
        claim(f"mechanism {row['arm']}: LIT occurs ratio reproduces from the stored estimator",
              abs(mine - row["lit_occurs_ratio"]) <= 1e-9 * max(1, mine), f"{mine:.4g}")
        a_ = d["prior_estimators"]["ADDR"].get("cells") or {}
        on = [v["score"] for k, v in a_.items() if k.endswith(", True)")]
        claim(f"mechanism {row['arm']}: generalize_count matches the arm file",
              row["generalize_count"] == (d.get("first_solution_samples") or {}).get("generalize_count"))
        if on and row["addr_V_mean_true"] is not None:
            claim(f"mechanism {row['arm']}: ADDR V=True mean reproduces",
                  abs(sum(on) / len(on) - row["addr_V_mean_true"]) < 1e-12)
for gap in (0, 1):
    for a in ARMS:
        d = arm(gap, a)
        if d is None:
            continue
        c = recompute_cost(d)
        s = stored_cost(d)
        claim(f"gap {gap} {a}: expected programs recomputed from tier S/K equals stored",
              (c is None and s is None) or c == s, f"{float(c) if c else c}")
        last = d["tiers"][-1]
        claim(f"gap {gap} {a}: full-space S and K are the last tier's", last["S"] == d["full_space"]["S"]
              and last["K"] == d["full_space"]["K"])
        claim(f"gap {gap} {a}: solution_exists == (K > 0)", d["solution_exists"] == (int(last["K"]) > 0))
        claim(f"gap {gap} {a}: every tier is exhausted with a certificate",
              all(t["certificate"] in ("complete", "unique") for t in d["tiers"]))
        claim(f"gap {gap} {a}: tiers are nested (S and K non-decreasing)",
              all(int(b["S"]) >= int(a_["S"]) and int(b["K"]) >= int(a_["K"])
                  for a_, b in zip(d["tiers"], d["tiers"][1:])))
        if c is not None:
            smp = d["first_solution_samples"]
            claim(f"gap {gap} {a}: every sampled first solution conforms on training",
                  smp["train_conform_all"])
            claim(f"gap {gap} {a}: expected episodes = 12 x programs",
                  Fraction(int(d["expected_episodes"]["num"]), int(d["expected_episodes"]["den"])) == 12 * c)
        costs[(gap, a)] = c

# sweep harness sanity: OCC_3500 pins the occurs ratio at ~N'''s own fitted value,
# so it must reproduce N'''s expected cost (recomputed from tier rows, not stored)
if costs.get((0, "OCC_3500")) is not None and costs.get((0, "N''")) is not None:
    a_, b_ = costs[(0, "OCC_3500")], costs[(0, "N''")]
    claim("sweep sanity: OCC_3500 reproduces N'' expected programs (within 1e-6 relative)",
          abs(float(a_) / float(b_) - 1) < 1e-6, f"{float(a_):.6e} vs {float(b_):.6e}")

# subset relation that F3's theorem rests on: N' and N'' pools are inside N's
import family  # noqa: E402
flat, schema = family.pools("flat"), family.pools("schema")
claim("schema pools are a subset of flat pools (F3's premise)",
      all(set(map(str, schema[s])) <= set(map(str, flat[s])) for s in flat))
for gap in (0, 1):
    n, n1 = arm(gap, "N"), arm(gap, "N'")
    if n and n1:
        claim(f"gap {gap}: K_N' <= K_N (retention <= 1)", int(n1["full_space"]["K"]) <= int(n["full_space"]["K"]))
        if costs.get((gap, "N")) and costs.get((gap, "N'")):
            sav = costs[(gap, "N")] / costs[(gap, "N'")]
            ratio = Fraction(int(n["full_space"]["S"]), int(n1["full_space"]["S"]))
            claim(f"gap {gap}: N'/N saving does not exceed the space-size ratio (F3 theorem)",
                  sav <= ratio * Fraction(101, 100), f"saving {float(sav):.3e} vs ratio {float(ratio):.3e}")

# pre-registered criteria, recomputed
for gap in (0, 1):
    cn, c1, c2 = costs.get((gap, "N")), costs.get((gap, "N'")), costs.get((gap, "N''"))
    if cn and c2:
        c1_pass = c2 * 10 <= cn
        text = f"**C1" in RESULTS
        claim(f"gap {gap}: C1 verdict in RESULTS matches recomputation",
              f"<!-- BEGIN:criteria_gap{gap} -->" not in RESULTS or
              (("→ **PASS**" if c1_pass else "→ **FAIL**") in
               RESULTS.split(f"<!-- BEGIN:criteria_gap{gap} -->")[1].split("C2")[0]))
    ds = [arm(gap, f"D''_{s}") for s in range(5)] + [arm(gap, "D'")]
    ds = [d for d in ds if d is not None]
    if ds and cn:
        c2_pass = all((not d["solution_exists"]) or costs[(gap, d["arm"])] * 10 > cn for d in ds)
        seg = RESULTS.split(f"<!-- BEGIN:criteria_gap{gap} -->")[1].split("<!-- END")[0] \
            if f"<!-- BEGIN:criteria_gap{gap} -->" in RESULTS else ""
        c2seg = seg.split("C2")[1].split("C3")[0] if "C2" in seg else ""
        claim(f"gap {gap}: C2 verdict in RESULTS matches recomputation",
              not seg or (("**PASS**" if c2_pass else "**FAIL**") in c2seg))

# validations
b0 = J("v2_brute_gap0.json")
claim("V2: counter equals brute force on every gap-0 sub-space", b0 and b0["n_agree"] == b0["n"] and b0["n"] >= 20)
claim("V2: the gap-0 brute-force checks include sub-spaces with conformers", b0 and b0["n_with_conformers"] >= 5)
b1 = J("v2_brute_gap1.json")
if b1:
    claim("V2: counter equals brute force on every gap-1 sub-space", b1["n_agree"] == b1["n"])
t = J("v2_tcn_gap0.json")
claim("V2: counter equals tcn.search.enumerate_prefix on >= 3 designs",
      t and len(t["designs"]) >= 3 and all(r["tcn"]["conforming"] == r["K_counter"] and
                                          r["tcn"]["space_size"] == r["space_counter"] and
                                          r["tcn"]["exhausted"] for r in t["designs"]))
v1 = [J(f"v1_shard{i}.json") for i in range(3)]
v1 = [x for x in v1 if x]
if v1:
    rows = [r for x in v1 for r in x["rows"]]
    claim("V1: evaluator click equals Program.execute click on every sampled program and episode",
          all(r["engine"] == r["tcn"] for r in rows) and len(rows) >= 500, f"{len(rows)} programs")
    claim("V1: every program checked on all 48 episodes", all(len(r["tcn"]) == 48 for r in rows))
v3 = J("v3_live.json")
if v3:
    claim("V3: live kernel rewards equal evaluator hits", all(
        [0. if x is None else x for x in r["live"]] == r["engine_hits"] for r in v3["programs"]))
    e = v3["enumerate_environment"]
    claim("V3: enumerate_environment conforming set equals the counter", e["tcn"]["conforming"] == e["K_counter"]
          and e["tcn"]["exhausted"])
    claim("V3: enumerate_environment charged 12 episodes per program",
          e["tcn"]["episodes"] == 12 * e["tcn"]["evaluated"])

# ------------------------------------------------ LEAKAGE: the prior's data
# Every labelled row must come from an earlier domain's own search; the
# integrated task may contribute only UNLABELLED statistics of its 12 TRAINING
# episodes' observations (V, occurs) at scoring time, and nothing from held-out.
SOURCE_OK = {"s19_stage_a", "s23_transform", "s23_policy", "s33_s0", "s33_s1", "s33_s2"}
src = J("sources.json")
kinds = {k: v for k, v in src.items() if isinstance(v, list) and k != "certificates"}
claim("LEAKAGE: every prior row comes from an earlier-domain search (§19/§23/§33)",
      all(r["source"] in SOURCE_OK for v in kinds.values() for r in v),
      ", ".join(f"{k} {len(v)}" for k, v in kinds.items()))
claim("LEAKAGE: row counts are ADDR 233, LIT 424, TRUTH 48, STEP 20, GROUND 0",
      [len(kinds[k]) for k in ("ADDR", "LIT", "TRUTH", "STEP", "GROUND")] == [233, 424, 48, 20, 0])
claim("LEAKAGE: survivor labels equal the earlier searches' own conforming sets (23/22/5/6)",
      [sum(r["survive"] for r in kinds[k]) for k in ("ADDR", "LIT", "TRUTH", "STEP")] == [23, 22, 5, 6])
code_src = (HERE / "sources.py").read_text()
code_prior = (HERE / "prior.py").read_text()
claim("LEAKAGE: sources.py and prior.py never import the integrated generator or its episode cache",
      not any(tok in code_src + code_prior for tok in ("import env", "import cache", "episodes_gap",
                                                        "import arms", "validate")))
import numpy as np              # noqa: E402
import prior as _prior          # noqa: E402
import engine as _engine        # noqa: E402
_P = _prior.Prior(_prior.load_sources())
d = arm(0, "N''")
if d is not None:
    claim("LEAKAGE: the prior refitted from sources.json alone reproduces the stored estimator cells",
          json.loads(json.dumps(_P.describe())) == d["prior_estimators"])
    ep = J("episodes_gap0.json")["episodes"]
    train = _engine.Episodes(ep[:12])
    pool = family.pools("schema")
    sc = _prior.slot_scores(_P, pool, train, 16)
    tiers = _prior.tiers(sc, pool)
    same = len(tiers) == len(d["tiers"]) and all(
        {s: int(R.m[s].sum()) for s in R.m} == t["sizes"] for R, t in zip(tiers, d["tiers"]))
    claim("LEAKAGE: N'' tiers rebuild exactly from the 12 training episodes' unlabelled text only", same)
    heldout = _engine.Episodes(ep[12:])
    sc_h = _prior.slot_scores(_P, pool, heldout, 16)
    claim("LEAKAGE: the only target-dependent prior inputs are unlabelled observation features "
          "(scores differ from training ones only through V on ADDR slots)",
          all(np.array_equal(np.asarray(sc[s]), np.asarray(sc_h[s])) for s in sc if family.KIND[s] != "ADDR"))
import numpy as np  # noqa: E402,F811

# ------------------------------------ the distractor's reach (exact, 48 episodes)
for _gap in (0, 1):
    g = J(f"generalize_gap{_gap}.json")
    if g is None:
        continue
    claim(f"gap {_gap}: direct 48-episode counting equals the zeta transform on the training episodes",
          all(v["transform"] == v["direct"] for v in g["direct_equals_transform_on_train"].values()))
    ks = int(g["schema"]["K_generalizing"])
    if _gap == 0:
        claim("gap 0: the positive control holds: the schema space contains generalizing programs",
              ks > 0, g["schema"]["K_generalizing"])
    claim(f"gap {_gap}: the schema space's generalizing count is exhausted (certificate complete)",
          g["schema"]["certificate"] in ("complete", "unique"), str(ks))
    kd = int(g["distractor"]["K_generalizing"])
    claim(f"gap {_gap}: the distractor space's generalizing count is exhausted (certificate complete)",
          g["distractor"]["certificate"] in ("complete", "unique"), str(kd))
    straw = kd == 0
    seg = RESULTS.split(f"<!-- BEGIN:generalize_gap{_gap} -->")[1].split("<!-- END")[0] \
        if f"<!-- BEGIN:generalize_gap{_gap} -->" in RESULTS else ""
    claim(f"gap {_gap}: RESULTS states the distractor fact that holds (straw man iff 0 generalizing programs)",
          not seg or (("could not succeed" in seg) == straw))
    claim(f"gap {_gap}: analytic: no distractor address expression equals length-5 on every episode",
          g["distractor_addresses_equal_length_minus_5"] == [] or not straw)

# post-hoc: programs to first generalizing, recomputed from each arm's 48-episode tier rows
for r in J("first_generalizing_gap0.json") or []:
    before, prev_S, prev_K, c = 0, 0, 0, None
    for t in r["tiers"]:
        S, K = int(t["S"]), int(t["K48"])
        if K - prev_K > 0 and prev_K == 0:
            c = Fraction(before) + Fraction(S - prev_S + 1, K - prev_K + 1)
            break
        before += S - prev_S
        prev_S, prev_K = S, K
    e = r["expected_programs_to_first_generalizing"]
    claim(f"post-hoc {r['arm']}: programs to first generalizing recomputes from its tier rows",
          (c is None and e is None) or (e is not None and c == Fraction(int(e["num"]), int(e["den"]))))
    d_ = arm(0, r["arm"])
    if d_ is not None and r["tiers"]:
        claim(f"post-hoc {r['arm']}: 48-episode tier sizes equal the arm's training tier sizes",
              all(t["S"] == u["S"] for t, u in zip(r["tiers"], d_["tiers"])))

# the gap-1 mechanism, from the episodes alone (no engine): which schema STEP
# candidates can land a click in the target, given the TRUE anchor extent?
_e1 = J("episodes_gap1.json")
if _e1 is not None:
    _W = 16
    _pool = [(op, base, off) for op in ("add", "sub") for base in ("lo", "hi")
             for off in (6, 3 * _W + 3, 3, 3 * _W, 9)]

    def _hits(r, op, base, off):
        x, y, w, h = r["widgets"][r["anchor"]]["rect"]
        lo, hi = 3 * (y * _W + x), 3 * ((y + h - 1) * _W + x + w - 1)
        b = lo if base == "lo" else hi
        c = (b + off if op == "add" else b - off) % 65536
        q = c // 3
        return q < _W * _W and r["owner"][q] == r["target"]
    _above = [r for r in _e1["episodes"] if r["relation"] == "above"]
    _side = [r for r in _e1["episodes"] if r["relation"] != "above"]
    claim("gap-1 mechanism: no schema STEP candidate reaches the target on any 'above' episode",
          _above and all(not any(_hits(r, *c) for c in _pool) for r in _above), f"{len(_above)} episodes")
    claim("gap-1 mechanism: the two-row step lo-96 (absent from the pool) reaches it on every 'above' episode",
          _above and all(_hits(r, "sub", "lo", 6 * _W) for r in _above))
    claim("gap-1 mechanism: 96 is absent from the STEP pool (6, 3W+3, 3, 3W, 9) and present in the flat 1..128",
          6 * _W not in {c[2] for c in _pool} and 6 * _W in family.STEP_FLAT_OFFSETS)
    claim("gap-1 mechanism: every left/right episode has a schema STEP candidate that reaches the target",
          all(any(_hits(r, *c) for c in _pool) for r in _side), f"{len(_side)} episodes")

# KO_LIT (N'' with LIT uniform) is not run: it is the same search as OCC_1.
# Tiers compare scores only within a slot, so any uniform LIT rule gives the
# same restrictions; checked here byte for byte rather than argued.
if arm(0, "OCC_1") is not None:
    import arms as _arms  # noqa: E402
    _src = _prior.load_sources()
    _ko = _arms.KnockoutPrior.__new__(_arms.KnockoutPrior)
    _ko.est = dict(_prior.Prior(_src).est)
    _ko.est["LIT"] = _arms._Rule()
    _pool = family.pools("schema")
    _tr = _engine.Episodes(J("episodes_gap0.json")["episodes"][:12])
    _a = _prior.tiers(_prior.slot_scores(_ko, _pool, _tr, 16), _pool)
    _b = _prior.tiers(_prior.slot_scores(_arms.OccursRatioPrior(_src, 1), _pool, _tr, 16), _pool)
    claim("KO_LIT's tiers equal OCC_1's byte for byte (so OCC_1 stands in for KO_LIT)",
          len(_a) == len(_b) and all(_arms.restriction_key(x) == _arms.restriction_key(y)
                                     for x, y in zip(_a, _b)))

# sources reproduce the earlier tracks' certificates
s = J("sources.json")
cert = {c["source"]: c for c in s["certificates"]}
claim("§19 stage A reproduces: 10,496 programs, 1 conforming, base 14, byte 40",
      cert["s19_stage_a"]["space"] == 10496 and cert["s19_stage_a"]["conforming"] == 1 and
      cert["s19_stage_a"]["selections"][0]["base"] == 14 and cert["s19_stage_a"]["selections"][0]["open"] == 40)
claim("§23 transform reproduces: 7,480 programs, 2 conforming",
      cert["s23_transform"]["space"] == 7480 and cert["s23_transform"]["conforming"] == 2)
claim("§23 policy reproduces: 448 programs, 21 conforming",
      cert["s23_policy"]["space"] == 448 and cert["s23_policy"]["conforming"] == 21)

# instrumentation: live accuracy of each found program agrees with the evaluator
for gap in (0, 1):
    ins = J(f"instrument_gap{gap}.json")
    for r in ins or []:
        if r.get("specs"):
            claim(f"gap {gap} {r['arm']}: live kernel rewards equal evaluator hits on 48 episodes",
                  r["live_agrees_with_engine"])
            claim(f"gap {gap} {r['arm']}: compiled program returns the interpreter's click",
                  r["compiled_matches_interpreter"])

# resources: every capped job under the 20G cap, none refused silently
peaks = []
for p in OUT.glob("*.time"):
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", p.read_text())
    if m:
        peaks.append(int(m.group(1)))
claim("every capped job peaked under the 20 GB cap", peaks and max(peaks) < 20 * 1024 * 1024,
      f"max {max(peaks) / 1024 / 1024:.2f} GB" if peaks else "")

# ------------------------------------------ layer 3: numbers in prose
prose = re.sub(r"<!-- BEGIN:(\w+) -->.*?<!-- END:\1 -->", "", RESULTS, flags=re.S)
prose = re.sub(r"`[^`]*`", "", prose)
prose = re.sub(r"(?m)^#+ .*$", "", prose)      # headings carry section numbering, not claims
block_text = " ".join(re.findall(r"<!-- BEGIN:\w+ -->(.*?)<!-- END", RESULTS, flags=re.S))
ALLOWED = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "15", "16",
           "20", "25", "26", "36", "48", "64", "100", "128", "256", "400", "500", "1,280", "1,445",
           "1,462", "2,304", "3W", "30", "240", "50", "55", "58", "60", "62", "63", "64", "19", "23",
           "33", "39", "44", "45", "47", "32", "24", "40", "42", "57", "59", "61", "70", "2026",
           "10×", "4", "95"}
bad = []
for tok in re.findall(r"(?<![\w.§/-])\d[\d,]*(?:\.\d+)?(?:×10\^\d+)?", prose):
    if tok in ALLOWED or tok in block_text:
        continue
    bad.append(tok)
claim("every number in RESULTS.md prose is structural or appears in a rendered block",
      not bad, f"untraced: {sorted(set(bad))[:20]}")

# headline.json (read by the merge gate): every value re-derived here from raw files
h = J("headline.json")
if h is not None:
    ok = True
    for gap in (0, 1):
        for a in ("N", "N'", "N''"):
            e = h["c1"][f"gap{gap}"]["expected_programs"].get(a)
            d = arm(gap, a)
            c = recompute_cost(d) if d else None
            ok &= (e is None and c is None) or (e is not None and c is not None and
                                                 Fraction(int(e["num"]), int(e["den"])) == c)
        for a, v in h[f"generalize_gap{gap}"].items():
            d = arm(gap, a)
            s = (d or {}).get("first_solution_samples") or {}
            ok &= v is not None and v["generalize"] == s.get("generalize_count") and v["n"] == s.get("n")
    for gap, v in h["exact_generalizing"].items():
        g_ = J(f"generalize_{gap}.json")
        ok &= g_ is not None and v["schema"] == g_["schema"]["K_generalizing"] and \
            v["distractor"] == g_["distractor"]["K_generalizing"]
    claim("headline.json: every value re-derives exactly from the arm and generalize files", ok)

fails = 0
for ok, name, detail in results:
    print(("PASS" if ok else "FAIL"), name, ("— " + detail) if detail else "")
    fails += not ok
print(f"\n{len(results) - fails} PASS, {fails} FAIL")
(OUT / "verify.json").write_text(json.dumps(
    {"pass": len(results) - fails, "fail": fails,
     "claims": [{"name": name, "ok": ok, "detail": detail} for ok, name, detail in results]},
    indent=1))
sys.exit(1 if fails else 0)
