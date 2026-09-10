"""Build F', mine every declared corpus under both identity relations, and
publish the modules the pre-registered selection rules pick.

`python research/semantic-library/run_mine.py [stage]`
    stage `off`    -- build and verify the off-family corpus F'
    stage `mine`   -- mine every corpus (full and leave-one-out)
    stage `publish`-- resolve the selection rules and publish to a real Library
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import _paths  # noqa: F401

import evaltasks
import family
import poolmine

OUT = Path(__file__).resolve().parent / "out"


def stage_off():
    t0 = time.perf_counter()
    payload = family.build_off_bands()
    for t in payload["tasks"]:
        print(t["task"], "min_gates", t["min_gates"],
              {k: (v["found"], v["capped"], v["exhausted"], v["verified_in_tcn"],
                   v["dfs_expanded"]) for k, v in t["bands"].items()}, flush=True)
    print("off-family bands built in %.1fs" % (time.perf_counter() - t0))


def stage_mine():
    jobs = []
    # full-corpus arms
    jobs.append(dict(family_name="maj", variant="C-minall", identity="syntactic",
                     label="maj_minall_syn"))
    jobs.append(dict(family_name="maj", variant="C-minall", identity="semantic",
                     label="maj_minall_sem"))
    jobs.append(dict(family_name="maj", variant="C-trace", identity="semantic",
                     label="maj_trace_sem"))
    jobs.append(dict(family_name="off", variant="C-minall", identity="semantic",
                     label="off_minall_sem"))
    jobs.append(dict(family_name="off", variant="C-minall", identity="syntactic",
                     label="off_minall_syn"))
    # leave-one-out, both identities
    for h in evaltasks.LOO_TASKS:
        short = h.split("_")[0]
        jobs.append(dict(family_name="maj", variant="C-minall", identity="semantic",
                         exclude=(h,), label=f"maj_minall_sem_wo_{short}"))
        jobs.append(dict(family_name="maj", variant="C-minall", identity="syntactic",
                         exclude=(h,), label=f"maj_minall_syn_wo_{short}"))
    summary = {}
    for job in jobs:
        p = poolmine.run(**job)
        summary[p["name"]] = {k: p[k] for k in
                              ("family", "variant", "identity", "excluded", "entries",
                               "eligible", "rank1_is_window", "best_window_rank",
                               "mine_wall_seconds")}
        top = p["ranked"][0] if p["ranked"] else None
        print(p["name"], "entries", p["entries"], "eligible", p["eligible"],
              "rank1_is_window", p["rank1_is_window"],
              "best_window_rank", p["best_window_rank"],
              "rank1", (top["body"] if top else None),
              "arity", (top["arity"] if top else None),
              "%.0fs" % p["mine_wall_seconds"], flush=True)
    (OUT / "mine_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))


def stage_publish():
    published = {}
    plan = [
        ("maj_minall_sem", "rank1", "sem_minall"),
        ("maj_minall_sem", "runnerup", "sem_minall_runnerup"),
        ("maj_minall_sem", "node_matched", "sem_minall_matched"),
        ("maj_trace_sem", "rank1", "sem_trace"),
        ("maj_minall_syn", "rank1", "syn_minall"),
        ("maj_minall_syn", "runnerup", "syn_minall_runnerup"),
        ("off_minall_sem", "rank1_arity3", "sem_off"),
    ]
    for h in evaltasks.LOO_TASKS:
        short = h.split("_")[0]
        plan.append((f"maj_minall_sem_wo_{short}", "rank1", f"sem_wo_{short}"))
        plan.append((f"maj_minall_syn_wo_{short}", "rank1", f"syn_wo_{short}"))
    for name, rule, label in plan:
        path = OUT / f"mined_{name}.json"
        if not path.exists():
            print("missing", name)
            continue
        payload = json.loads(path.read_text())
        row = poolmine.pick(payload, rule)
        if row is None:
            print("no candidate for", label, "under", rule)
            published[label] = None
            continue
        rec = poolmine.publish(payload, row, label)
        rec["rule"] = rule
        published[label] = rec
        print(label, rule, rec["operator"], rec["nodes"], "nodes", "rank", rec["rank"],
              "arity", rec["arity"], "window", rec["is_window"], flush=True)
    (OUT / "published.json").write_text(json.dumps(published, indent=2, sort_keys=True))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    {"off": stage_off, "mine": stage_mine, "publish": stage_publish}[sys.argv[1]]()
