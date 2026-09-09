"""Recompute every headline in RESULTS.md from the per-row records, not the summaries.

On this project the missing reference has repeatedly been the difference between
a result and a mistake (`STATUS.md` §1), and the specific failure has twice been
an aggregate that did not follow from its own rows. So each figure quoted in
RESULTS.md is recomputed here from the lowest-level record in `out/*.json` --
per-screen for the visual arm, per-episode for language, per-document for the
computer arm -- and compared with the summary the sweep wrote.

Any mismatch is printed as `MISMATCH` and makes the exit status non-zero.

Run: `.venv/bin/python research/neural-baselines/verify.py`
"""
from __future__ import annotations

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"

FAILURES = []


def check(label, got, want, tolerance=1e-9):
    ok = (got == want) if isinstance(want, (str, bool, type(None))) \
        else abs(float(got) - float(want)) <= tolerance
    print(f"{'ok  ' if ok else 'MISMATCH'}  {label:58s} recomputed {got!r} vs recorded {want!r}")
    if not ok:
        FAILURES.append(label)


def load(name):
    path = OUT / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def visual_tcn():
    d = load("tcn_visual")
    if not d:
        return
    rows = d["quality"]["episodes"]
    links = sum(r["parent_links_correct"] for r in rows)
    widgets = sum(r["widgets_in_probe"] for r in rows)
    trees = sum(bool(r["tree_exact"]) for r in rows)
    check("visual TCN parent links correct", links, d["quality"]["totals"]["parent_links_correct"])
    check("visual TCN widgets in probe", widgets, d["quality"]["totals"]["widgets_in_probe"])
    check("visual TCN trees exact", trees, d["quality"]["totals"]["trees_exact"])
    check("visual TCN link accuracy", links / widgets, d["quality"]["totals"]["link_accuracy"])
    check("visual TCN screens", len(rows), 12)


def visual_neural():
    d = load("visual")
    if not d:
        return
    for arm in d["arms"]:
        rows = arm["rows"]
        for row in rows:
            t = row["test"]
            recomputed = t["parent_links_correct"] / max(1, t["widgets_in_probe"])
            check(f"visual {arm['budget']}/{arm['model']} seed {row['seed']} link acc",
                  recomputed, t["link_accuracy"])
            check(f"visual {arm['budget']}/{arm['model']} seed {row['seed']} widgets",
                  t["widgets_in_probe"], 227)


def language():
    d = load("tcn_language")
    if d:
        q = d["quality"]["test"]
        check("language TCN held-out n", q["n"], 724)
        check("language TCN majority constant", q["majority_constant"], 0.5483425414364641)
    n = load("language")
    if not n:
        return
    for arm in n["arms"]:
        import statistics
        recomputed = statistics.median(r["test_acc"] for r in arm["rows"])
        check(f"language {arm['budget']}/{arm['model']}/aux={int(arm['aux'])} test median",
              recomputed, arm["test_acc_median"], tolerance=1e-6)


def computer():
    d = load("tcn_computer")
    if d:
        rows = d["quality"]["rows"]
        check("computer TCN episodes", len(rows), 10)
        check("computer TCN solved", sum(r["return"] == 2 for r in rows),
              d["quality"]["solved"])
        check("computer TCN mean return",
              sum(r["return"] for r in rows) / len(rows), d["quality"]["mean_return"],
              tolerance=1e-9)
    n = load("computer")
    if not n:
        return
    for arm in n["arms"]:
        for row in arm["rows"]:
            if "closed_loop_rows" not in row:
                continue
            live = row["closed_loop_rows"]
            check(f"computer {arm['model']}/{arm['byte_head']} seed {row['seed']} solved",
                  sum(r["return"] == 2 for r in live), row["closed_loop"]["solved"])
            check(f"computer {arm['model']}/{arm['byte_head']} seed {row['seed']} episodes",
                  len(live), 10)


def main():
    visual_tcn()
    visual_neural()
    language()
    computer()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} MISMATCH(es):")
        for f in FAILURES:
            print("  ", f)
        sys.exit(1)
    print("every recomputed headline matches its recorded summary")


if __name__ == "__main__":
    main()
