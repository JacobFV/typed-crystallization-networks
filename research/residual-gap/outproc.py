"""An out-of-process replication of the ladder on a different interpreter.

`research/lazy-latency/RESULTS.md` section 6.1 established the habit: the
headline is re-measured in a fresh `/usr/bin/python3 -I` (3.12.3, no venv, no
repository on the path, no torch, no numpy, no `tcn`) so that a CPython-3.13
specific effect cannot masquerade as an attribution.  The rung whose size most
depends on the interpreter is R5 -- typed-guard elimination -- so it is the one
that most needs this.

The child imports the committed `out/R*.py` sources directly; nothing else.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(HERE), str(ROOT), str(ROOT / "research" / "lazy-latency")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gate as GATE                                          # noqa: E402
import rungs as RUNGS                                        # noqa: E402

OUT = HERE / "out"

CHILD = '''\
import gc, importlib.util, json, statistics, sys, time

here = sys.argv[1]
data = json.load(open(sys.argv[2]))
obs = [tuple(o) for o in data["obs"]]
recs = [(p, obs[i]) for p, i in data["recs"]]
names = sys.argv[3].split(",")
repeats = int(sys.argv[4])

mods = {}
for n in names:
    spec = importlib.util.spec_from_file_location("rung_" + n, here + "/" + n + ".py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    mods[n] = m

ref = [list(mods[names[0]].run(r)) for r in recs]
for n in names:
    got = [list(mods[n].run(r)) for r in recs]
    if got != ref:
        print(json.dumps({"error": "gate failed in child for " + n}))
        raise SystemExit(1)

def sweep(fn):
    gc.collect(); gc.disable()
    try:
        t = time.perf_counter_ns()
        for r in recs:
            fn(r)
        return (time.perf_counter_ns() - t) / 1e3 / len(recs)
    finally:
        gc.enable()

for n in names:
    sweep(mods[n].run)
s = {n: [] for n in names}
for _ in range(repeats):
    for n in names:
        s[n].append(sweep(mods[n].run))
print(json.dumps({"python": sys.version.split()[0], "records": len(recs),
                  "median_us": {n: statistics.median(v) for n, v in s.items()},
                  "leaked": [m for m in ("tcn", "torch", "numpy") if m in sys.modules]}))
'''


def main(repeats=11):
    grep, g, R = GATE.check()
    if not grep["ladder_gate_passed"]:
        raise SystemExit("gate failed")

    obs, index = [], {}
    recs = []
    for pos, ob in g["deploy"]:
        k = id(ob)
        if k not in index:
            index[k] = len(obs)
            obs.append(list(ob))
        recs.append([pos, index[k]])
    dat = OUT / "records_deploy.json"
    dat.write_text(json.dumps({"obs": obs, "recs": recs}))

    child = OUT / "child.py"
    child.write_text(CHILD)
    names = ",".join(RUNGS.LADDER)
    cmd = ["/usr/bin/python3", "-I", str(child), str(OUT), str(dat), names, str(repeats)]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         env={"PATH": "/usr/bin:/bin"})
    if res.returncode != 0:
        raise SystemExit("child failed: %s\n%s" % (res.stdout, res.stderr))
    out = json.loads(res.stdout.strip().splitlines()[-1])

    med = out["median_us"]
    out["ratios"] = {}
    for i in range(len(RUNGS.LADDER) - 1):
        a, b = RUNGS.LADDER[i], RUNGS.LADDER[i + 1]
        out["ratios"]["%s_over_%s" % (a, b)] = med[a] / med[b]
    out["R0_over_R7"] = med["R0"] / med["R7"]
    out["repeats"] = repeats
    out["distribution"] = "deployment (54 corner records)"
    (OUT / "outproc.json").write_text(json.dumps(out, indent=1))
    print("->", OUT / "outproc.json")
    print("python", out["python"], "leaked", out["leaked"])
    for n in RUNGS.LADDER:
        print("   %-3s %8.3f us" % (n, med[n]))
    print("   R0/R7 =", round(out["R0_over_R7"], 3))
    return out


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 11)
