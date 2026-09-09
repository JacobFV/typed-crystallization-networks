"""Run several arm enumerations, one process each, and log them separately."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "out"
PY = sys.executable

JOBS = {
    "arm1_none": 24,
    "arm2_earned": 24,
    "arm2p_trace": 24,
    "arm3_authored": 24,
    "arm4_wrong_authored": 24,
    "arm4b_wrong_mined": 24,
    "arm3p_mined_maj3": 24,
    # a 4-ary module makes the call-site candidate block 6**4 wide, so this
    # arm's tight space is 64 155 648 programs -- 24x the others.  It still
    # gets a full sweep, with the cap raised so the certificate survives.
    "arm2p_plus1one": 27,
}


def main():
    arms = sys.argv[1:] or list(JOBS)
    OUT.mkdir(exist_ok=True)
    procs = []
    for arm in arms:
        log = open(OUT / f"enum_{arm}.log", "w")
        p = subprocess.Popen([PY, str(HERE / "run_enum_premin.py"), arm, "tight",
                              str(JOBS[arm])], stdout=log, stderr=subprocess.STDOUT,
                             env=dict(os.environ))
        procs.append((arm, p, log))
        print("started", arm, flush=True)
    t0 = time.perf_counter()
    for arm, p, log in procs:
        p.wait()
        log.close()
        print("done", arm, "rc", p.returncode, "at %.0fs" % (time.perf_counter() - t0),
              flush=True)


if __name__ == "__main__":
    main()
