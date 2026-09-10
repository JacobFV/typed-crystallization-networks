"""Everything downstream of the off-family corpus, in one process.

Mines F' under both identity relations, publishes the pooled off-family module
(`arm4s_offfamily`), enumerates that arm on L1, then runs the L1 gradient arms,
the whole held-out protocol and the F3 checks.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import _paths  # noqa: F401

import poolmine
import run_enum

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"
PY = sys.executable


def sh(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([PY, str(HERE / args[0])] + list(args[1:]), check=True)


def main():
    # 1. mine F' under both identities and publish the pooled wrong module
    for identity, label in (("syntactic", "off_minall_syn"), ("semantic", "off_minall_sem")):
        p = poolmine.run(family_name="off", variant="C-minall", identity=identity,
                         label=label)
        top = p["ranked"][0] if p["ranked"] else None
        print(label, "entries", p["entries"], "eligible", p["eligible"],
              "rank1_is_window", p["rank1_is_window"],
              "best_window_rank", p["best_window_rank"],
              "rank1 arity", (top["arity"] if top else None),
              "nodes", (top["nodes"] if top else None), flush=True)

    pub = json.loads((OUT / "published_maj.json").read_text())
    payload = json.loads((OUT / "mined_off_minall_sem.json").read_text())
    for rule, label in (("rank1_arity3", "sem_off"),):
        row = poolmine.pick(payload, rule)
        if row is None:
            print("NO CANDIDATE", label)
            continue
        rec = poolmine.publish(payload, row, label)
        rec["rule"] = rule
        pub[label] = rec
        print(label, rec["operator"], rec["nodes"], "nodes", "rank", rec["rank"],
              "arity", rec["arity"], "off-window", rec["is_window"], flush=True)
    (OUT / "published.json").write_text(json.dumps(pub, indent=2, sort_keys=True))

    # 2. the remaining L1 arm
    run_enum.enumerate_arm("arm4s_offfamily", "tight")

    # 3. L1 gradient, 4. held-out, 5. checks
    sh("run_grad.py", "tight", "24")
    sh("run_grad.py", "wide", "8")
    sh("run_heldout.py", "all")
    sh("run_checks.py")


if __name__ == "__main__":
    main()
