"""Second, independent measurement of the two headline comparisons.

The main runs were taken while the machine was carrying other tracks' jobs.
This re-measures the load-bearing arms back to back in one process, adds the
`Program.execute` variant of the joint probe sweep (so the enumeration figure
can be quoted both with a direct evaluator and through the repo's own exact
runtime), and records the load average at the time.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import write, machine

import joint_task as J
import mixed_task as M


def main():
    rows = []
    load_before = os.getloadavg()

    # warm imports / caches so the timed calls are comparable
    M.enumerate_all(True)
    J.enumerate_probe(True)

    rows.append(M.enumerate_all(True).row())
    rows.append(M.enumerate_all(False).row())
    for s in range(3):
        rows.append(M.gradient(s).row())
        print("mixed gradient", s, "done", flush=True)

    rows.append(J.enumerate_probe(True).row())
    rows.append(J.enumerate_probe(False).row())
    rows.append(J.enumerate_probe_tcn_runtime(False).row())
    print("joint probe arms done", flush=True)
    rows.append(J.enumerate_env(stop_at_first=False).row())
    print("joint env sweep done", flush=True)
    for s in range(3):
        rows.append(J.gradient(s).row())
        print("joint gradient", s, "done", flush=True)

    payload = {"machine": machine(), "loadavg_before": load_before,
               "loadavg_after": os.getloadavg(), "results": rows}
    print(write("headline.json", payload))
    for r in rows:
        print(f"{r['task']:<16} {r['method']:<44} solved={r['solved']!s:<5} "
              f"{r['seconds']*1000:12.3f} ms")


if __name__ == "__main__":
    main()
