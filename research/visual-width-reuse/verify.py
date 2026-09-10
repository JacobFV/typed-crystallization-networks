"""Recompute every headline claim of RESULTS.md from `out/` and print PASS/FAIL.

Exit status is non-zero on any FAIL.  Three kinds of check:

1. **Claims** -- each headline statement, recomputed from raw JSON *without*
   going through `report.py`, so a bug in the renderer cannot certify itself.
   The digest claims go further and rebuild the schema at each held-out
   resolution, freeze the stored vector, and compare the digest to both arms.
2. **Blocks** -- every `<!-- BEGIN:name -->` span of RESULTS.md must equal what
   `report.render()` produces from `out/` right now.
3. **Prose** -- no number outside a generated block may be untraced: it must be a
   section/line/commit reference, appear verbatim in a generated block, or be an
   explicitly cited external figure listed in `CITED` with its source.

Run it inside the capped scope, like everything else:
    research/visual-width-reuse/capped.sh verify verify.py
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(ROOT / "research" / "visual-ladder"),
           str(ROOT / "research" / "class-identity"), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import report                                      # noqa: E402

OUT = HERE / "out"
CERTIFIED, HELD_OUT, ALL, STAGES = report.CERTIFIED, report.HELD_OUT, report.ALL, report.STAGES
RESULTS = HERE / "RESULTS.md"

# Figures RESULTS.md cites from *other* tracks, with where each is recorded.  A
# number here is not produced by this track, so it is traced to its file instead.
CITED = {
    "272": "research/class-identity/out/check.json (a2_programs, episode_ratio)",
    "17": "research/depth-encoding: 17 x 16 candidates (§53/§55)",
    "227": "research/visual-ladder/out/rung3_root.json parse totals",
    "0.248836": "HANDOFF.md shipped fixture", "0.002231": "HANDOFF.md shipped fixture",
    "337": "test suite size, research/MERGE-QUEUE.md", "336": "worktree pass count",
    "30": "§41 span sweep, research/program-length/out/span_sweep_fine.json",
    "29": "§41 span sweep", "4": "§41 span sweep lower end; 4/4 fixture score",
    "121": "host RAM, GB, coordinator's crash report",
    "40": "MemoryMax=40G cap", "800": "CPUQuota=800% cap",
    "0.6": "§59 instrument floor",
}

results = []


def claim(name, ok, detail=""):
    results.append((bool(ok), name, detail))


def J(name, base=OUT):
    return json.loads((base / f"{name}.json").read_text())


# ------------------------------------------------------------------ claims

def check_claims():
    enum = {w: J(f"enum_{w}") for w in ALL}
    with_ = {w: J(f"with_{w}") for w in HELD_OUT}
    construct, armf, refusal = J("construct"), J("armf"), J("refusal")
    root = J("rung3_root", ROOT / "research/visual-ladder/out")
    wid = J("rung3", ROOT / "research/visual-ladder/out")

    # completeness: every enumeration is a finished, exhausted run
    for w in ALL:
        e = enum[w]
        ok = (all(st in e["stages"] for st in STAGES) and "pipeline" in e and "parse" in e
              and all(e["stages"][st]["walk"]["exhausted"]
                      and e["stages"][st]["walk"]["evaluated"] == e["stages"][st]["space_size"]
                      and e["stages"][st]["sweep"]["exhausted"]
                      for st in STAGES if st in e["stages"]))
        claim(f"enum_{w} is complete and every stage exhausted (walk and prefix search)", ok)
    for w in HELD_OUT:
        claim(f"with_{w} is complete", all(st in with_[w]["stages"] for st in STAGES)
              and "parse" in with_[w])

    # certificates, every resolution
    expect = {"s0": ("complete", 2), "s1": ("complete", 2), "s2": ("unique", 1)}
    for w in ALL:
        got = {st: (enum[w]["stages"][st]["walk"]["certificate"],
                    enum[w]["stages"][st]["walk"]["conforming"]) for st in STAGES}
        claim(f"res {w}: certificates S0 complete/2, S1' complete/2, S2' unique/1", got == expect,
              str(got))
        claim(f"res {w}: walk and prefix search agree on the conforming count",
              all(enum[w]["stages"][st]["sweep"]["conforming"] ==
                  enum[w]["stages"][st]["walk"]["conforming"] for st in STAGES))
        claim(f"res {w}: S0's two conforming programs are one function",
              enum[w]["stages"]["s0"]["distinct_functions"] == 1)

    # §33 does not move -- against §33's own files, not this track's constants
    e32 = enum[32]
    t33, t32 = root["parse"]["totals"], e32["parse"]["totals"]
    claim("§33 parse totals reproduce exactly at res 32 (12 screens, seeds 200-211)",
          all(t33[k] == t32[k] for k in ("widgets_in_probe", "rects_predicted",
                                         "screens_rects_exact", "roots_predicted",
                                         "parent_links_correct", "parent_links_wrong",
                                         "trees_exact", "corner_key_collisions")),
          f"{t32['parent_links_correct']}/{t32['widgets_in_probe']} links, "
          f"{t32['trees_exact']}/{t32['screens']} trees")
    claim("§33 S0 module digest reproduces bit-exactly at res 32",
          wid["s0"]["module"] == e32["stages"]["s0"]["module"], wid["s0"]["module"])
    claim("§33 S1'/S2' certificates and space sizes reproduce at res 32",
          root["s1"]["search"]["certificate"] == e32["stages"]["s1"]["walk"]["certificate"]
          and root["s1"]["conforming"]["count"] == e32["stages"]["s1"]["walk"]["conforming"]
          and root["s2"]["search"]["certificate"] == e32["stages"]["s2"]["walk"]["certificate"]
          and root["s2"]["conforming"]["count"] == e32["stages"]["s2"]["walk"]["conforming"]
          and root["s1"]["space_size"] == e32["stages"]["s1"]["space_size"]
          and root["s2"]["space_size"] == e32["stages"]["s2"]["space_size"])
    for st in ("s1", "s2"):
        free = e32["stages"][st]["free_nodes"]
        claim(f"§33 {st} selection vector reproduces at res 32",
              {k: v for k, v in root[st]["chosen"].items() if k in free}
              == e32["stages"][st]["chosen_free"])

    # the certified vector is width-invariant
    claim("the selected vector is identical at all five resolutions and equals the stored one",
          all(enum[w]["stages"][st]["chosen_free"] == construct["vectors"][st]
              for w in ALL for st in STAGES))
    claim("the class is admitted at exactly the two certified widths, vectors agreeing",
          construct["all_admitted"] and not construct["vector_disagreements"]
          and all(construct["records"][st]["widths"] == list(CERTIFIED) for st in STAGES))
    claim("every stage artifact digest differs across the five resolutions",
          all(len({enum[w]["stages"][st]["artifact_digest"] for w in ALL}) == len(ALL)
              for st in STAGES))

    # reproduction, recomputed by rebuilding the schema here
    import schema as S
    from tcn.operators import Registry
    for w in HELD_OUT:
        reg = Registry()
        vectors = construct["vectors"]
        p0 = S.RIGHT.build_s0(reg, w, w)
        m = S.module_of(p0, S.full(p0, vectors["s0"]), reg)
        progs = {"s0": p0, "s1": S.RIGHT.build_s1(reg, w, w, m),
                 "s2": S.RIGHT.build_s2(reg, w, w, m)}
        for st in STAGES:
            d = S.freeze(progs[st], S.full(progs[st], vectors[st]), reg).digest
            claim(f"res {w} {st}: rebuilt digest == with-arm digest == enumerated digest",
                  d == with_[w]["stages"][st]["artifact_digest"]
                  == with_[w]["stages"][st]["store_digest"]
                  == enum[w]["stages"][st]["artifact_digest"], d)
            claim(f"res {w} {st}: with-arm searched 0 programs and its check conforms",
                  with_[w]["stages"][st]["programs_searched"] == 0
                  and with_[w]["stages"][st]["check"]["conforms"])

    # the parse, both arms
    for w in HELD_OUT:
        a = {k: v for k, v in enum[w]["parse"]["totals"].items() if k != "seconds"}
        b = {k: v for k, v in with_[w]["parse"]["totals"].items() if k != "seconds"}
        claim(f"res {w}: the instantiated parse equals the enumerated parse, component for "
              f"component", a == b, f"{b['rects_predicted']}/{b['widgets_in_probe']} rects, "
              f"{b['trees_exact']}/{b['screens']} trees")

    # baselines
    for w in HELD_OUT:
        for st in STAGES:
            sa, sb = enum[w]["stages"][st], with_[w]["stages"][st]
            ws = sa["whole_space_held"]
            claim(f"res {w} {st}: accuracy beats best constant, uniform-random mean and the "
                  f"second-best program",
                  sb["held_accuracy"] > sb["best_constant"]["accuracy"]
                  and ws["exhaustive"]
                  and ws["chosen_accuracy_same_rows"] > ws["mean"]
                  and ws["chosen_accuracy_same_rows"] > ws["second_best"]
                  and ws["chosen_accuracy_same_rows"] > ws["best_constant_same_rows"],
                  f"{sb['held_accuracy']:.4f} vs const {sb['best_constant']['accuracy']:.4f}, "
                  f"mean {ws['mean']:.4f}, 2nd {ws['second_best']:.4f}")

    # the saving and the space size
    for w in HELD_OUT:
        largest = max(enum[w]["stages"][st]["space_size"] for st in STAGES)
        for st in STAGES:
            wk, ck = enum[w]["stages"][st]["walk"], with_[w]["stages"][st]["check"]
            claim(f"res {w} {st}: program and row ratios equal the stage space exactly",
                  wk["evaluated"] / ck["programs_evaluated"] == wk["space_size"]
                  and wk["row_evaluations"] / ck["row_evaluations"] == wk["space_size"])
        pa, pb = enum[w]["pipeline"], with_[w]["pipeline"]
        claim(f"res {w}: pipeline program ratio is 681/3 and below the largest stage space",
              pa["programs_evaluated"] == 681 and pb["programs_evaluated"] == 3
              and pa["programs_evaluated"] / pb["programs_evaluated"] <= largest)
        claim(f"res {w}: pipeline row-evaluation ratio does not exceed the largest stage space",
              pa["row_evaluations"] / pb["row_evaluations"] <= largest,
              f"{pa['row_evaluations'] / pb['row_evaluations']:.1f} vs {largest}")
        claim(f"res {w}: same-path wall-clock ratio does not exceed the largest stage space",
              pa["walk_seconds"] / pb["check_seconds"] <= largest,
              f"{pa['walk_seconds'] / pb['check_seconds']:.1f} vs {largest}")

    # arm F
    fo = armf["schemas"]["frozen_offsets"]["per_resolution"]
    fs = armf["schemas"]["frozen_span"]["per_resolution"]
    claim("arm F: both wrong schemas are bit-identical to the right one at res 32",
          all(armf["schemas"][s]["per_resolution"]["32"][st]["bit_identical_to_right"]
              for s in armf["schemas"] for st in ("s1", "s2")))
    claim("arm F1 (frozen offsets): 0 conforming, exhausted, at 16/24/40/48 for S1' and S2'",
          all(fo[str(w)][st]["walk"]["conforming"] == 0 and fo[str(w)][st]["walk"]["exhausted"]
              for w in (16, 24, 40, 48) for st in ("s1", "s2")))
    claim("arm F2 (frozen span): 0 conforming, exhausted, at 40 and 48 for S2'",
          all(fs[str(w)]["s2"]["walk"]["conforming"] == 0 and fs[str(w)]["s2"]["walk"]["exhausted"]
              for w in (40, 48)))
    for k, v in armf["format"].items():
        claim(f"arm F format: `{k}` is refused", not v["admitted"], v.get("rule", "ADMITTED")[:80])

    # F-a
    claim("F-a: the res-32 artifact accepts a res-32 screen",
          refusal["offered"]["32"]["accepted"])
    for w in HELD_OUT:
        v = refusal["offered"][str(w)]
        claim(f"F-a: the res-32 artifact refuses a res-{w} screen with a TypeError",
              not v["accepted"] and v.get("error") == "TypeError", v.get("message", ""))

    # the one parse miss is a colour collision, not a width failure
    c = J("collision")["rows"][0]
    claim("the res-40 miss is a parent-colour collision within the span",
          not c["left_differs"] and not c["upper_differs"] and c["extent_within_span"])

    # resources
    for p in sorted(OUT.glob("*.time")):
        t = report.timelog(p.stem)
        if p.stem == "verify":
            continue
        claim(f"resource: `{p.stem}` exited 0 with peak RSS under 30 GB",
              t["exit"] == 0 and t["peak_rss_kb"] < 30 * 1048576,
              f"{t['peak_rss_kb'] / 1048576:.2f} GB")


# ------------------------------------------------------------------ blocks

def check_blocks():
    text = RESULTS.read_text()
    present = report.blocks_in(text)
    rendered = report.render()
    for name, body in rendered.items():
        claim(f"block `{name}` in RESULTS.md equals its rendering from out/",
              present.get(name) == body,
              "missing" if name not in present else ("" if present[name] == body else "DRIFT"))
    for name in sorted(set(present) - set(rendered)):
        claim(f"block `{name}` has a renderer", False)
    return text, rendered


def check_prose(text, rendered):
    prose = report.BLOCK.sub("", text)
    prose = re.sub(r"```.*?```", "", prose, flags=re.S)           # code
    prose = re.sub(r"`[^`\n]*`", "", prose)                        # inline code: paths, ids
    prose = re.sub(r"§\s?\d+(\.\d+)?", "", prose)                   # section references
    prose = re.sub(r"\b[0-9a-f]{7,40}\b", "", prose)                # commit SHAs
    prose = re.sub(r"\b(20\d\d-\d\d-\d\d)\b", "", prose)            # dates
    prose = re.sub(r"\[[^\]]*\]\([^)]*\)", "", prose)               # links
    blob = "\n".join(rendered.values())
    bad = []
    for tok in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?(?![\w])", prose):
        bare = tok.replace(",", "")
        if tok in blob or bare in blob or bare in CITED:
            continue
        if bare in {"0", "1", "2", "3"}:          # counting words written as digits
            continue
        bad.append(tok)
    claim("prose outside generated blocks carries no untraced figure", not bad,
          f"untraced: {sorted(set(bad))}" if bad else "")


def main():
    check_claims()
    text, rendered = check_blocks()
    check_prose(text, rendered)
    for ok, name, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))
    fails = sum(not ok for ok, _, _ in results)
    print(f"\n{len(results) - fails} PASS, {fails} FAIL")
    (OUT / "verify.json").write_text(json.dumps(
        {"pass": len(results) - fails, "fail": fails,
         "results": [{"ok": ok, "claim": n, "detail": d} for ok, n, d in results]}, indent=1))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
