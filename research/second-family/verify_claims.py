"""Check the specific numbers `RESULTS.md` asserts against the raw JSON.

Every claim that is not a direct copy of a printed table is re-derived here, so
a surprising headline is verified against raw data rather than against a
summary of it.
"""
from __future__ import annotations

import json
from pathlib import Path

import _paths  # noqa: F401

import family

OUT = Path(__file__).resolve().parent / "out"


def main():
    for fam in ("w4", "x4"):
        d = json.loads((OUT / f"bands_{fam}.json").read_text())
        for label in ("min", "plus1"):
            print(fam, label,
                  [t["bands"][label]["uncapped_found"] for t in d["tasks"]],
                  "capped", [t["bands"][label]["capped"] for t in d["tasks"]],
                  "exhausted", {t["bands"][label]["exhausted"] for t in d["tasks"]},
                  "verified", {t["bands"][label]["verified_in_tcn"] for t in d["tasks"]})
        print(fam, "min_gates", {t["min_gates"] for t in d["tasks"]},
              "exhausted_lengths",
              sorted({tuple(t["min_certificate"]["exhausted_lengths"])
                      for t in d["tasks"]}))

    m = json.loads((OUT / "modules_C-minall.json").read_text())["arm4s_offfamily"]
    print("C-minall off-family module:", m["digest"], "arity", m["arity"],
          "tt", m["truth_table"], "body", m["body"])

    syn = json.loads((OUT / "mined_w4_C-minall_syntactic.json").read_text())
    sem = json.loads((OUT / "mined_w4_C-minall_semantic.json").read_text())
    print("C-minall syntactic top 5:",
          [(r["rank"], r["digest"][:8], r["saving_bits"], r["occurrences"],
            r["is_window"]) for r in syn["ranked"][:5]])
    print("C-minall semantic  top 3:",
          [(r["rank"], r["digest"][:8], r["saving_bits"], r["occurrences"],
            r["circuits_pooled"], r["is_window"]) for r in sem["ranked"][:3]])

    print("window table W4", [int(v) for v in family.window_table("w4")])
    print("window table X4", [int(v) for v in family.window_table("x4")])

    b = json.loads((OUT / "b1_alt.json").read_text())
    print("Amendment 1 helps", b["helps_of_5"], "of 5; cells",
          [(r["task"], r["conforming"], r["exhausted"], r["certificate"])
           for r in b["rows"]])


if __name__ == "__main__":
    main()
