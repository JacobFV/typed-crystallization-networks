"""PREREGISTRATION section 3 -- peak RSS, bytes on disk, cold start, per arm.

Each arm is run in a fresh `/usr/bin/python3 -I` child with `PATH` only, exactly
as `research/lazy-guard/runner.py :: deploy` and FINDINGS section 48 do, so
"stdlib-only" is a property of the process and not an assertion.  The child
reports its own `ru_maxrss` after one full sweep of the uniform distribution.

Two deliberate choices, both of which would otherwise wreck the numbers:

* **cold start excludes the input load.**  It is `import + module compile`
  plus the first call -- the time to be ready and produce one answer.  Reading
  the fixture is identical for every arm and dominates it, so charging it to the
  arm would make all three arms look the same.
* **the raster is shared by reference.**  FINDINGS section 48 measured exactly
  this: "961 records share one immutable 3,072-element observation by reference
  instead of each re-encoding a copy", 48.4 MB -> 0.121 MB.  Storing 2,883
  independent 3,072-tuples would put ~70 MB of fixture into every arm's peak RSS
  and hide the difference between them.
"""
from __future__ import annotations

import json
import pathlib
import statistics
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import arms  # noqa: E402

OUT = arms.OUT
COLD_RUNS = 11

CHILD = '''\
import json, pathlib, resource, sys, time
_T0 = time.perf_counter_ns()
sys.path.insert(0, str(pathlib.Path(sys.argv[1]).resolve().parent))
import %(mod)s as M
_T1 = time.perf_counter_ns()
blob = json.load(open(sys.argv[2]))
if %(is3)s:
    obs = [tuple(o) for o in blob["obs"]]
    cases = [(p, obs[i]) for i, p in blob["recs"]]
else:
    cases = [tuple(c) for c in blob["recs"]]
_T2 = time.perf_counter_ns()
first = %(call_first)s
_T3 = time.perf_counter_ns()
for c in cases:
    %(call_all)s
_T4 = time.perf_counter_ns()
# `ru_maxrss` is inherited across fork from whatever launched this process, so a
# child spawned from a large parent reports the PARENT's peak.  `VmHWM` is the
# high-water mark of this process's own mm, which `execve` replaces.  Both are
# reported and RESULTS.md records the discrepancy.
peak_rusage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
peak = 0.0
for _line in open("/proc/self/status"):
    if _line.startswith("VmHWM:"):
        peak = float(_line.split()[1]) / 1024.0
print(json.dumps({"import_ms": (_T1 - _T0) / 1e6,
                  "first_call_ms": (_T3 - _T2) / 1e6,
                  "cold_start_ms": (_T1 - _T0 + _T3 - _T2) / 1e6,
                  "sweep_ms": (_T4 - _T3) / 1e6,
                  "peak_rss_mb": peak,
                  "peak_rss_rusage_mb": peak_rusage,
                  "records": len(cases),
                  "leaked_modules": sorted(m for m in sys.modules
                                           if m in ("tcn", "torch", "numpy")),
                  "first": repr(first)[:120]}))
'''


def arm_module(g, stage, arm):
    """(module file on disk, module name, call template) for one arm."""
    port = g["port"]
    if arm == "A":
        path = OUT / ("stage%d_compiled.py" % stage)
        path.write_text(g["arms"]["A"]["source"])
        return path, path.stem, "M.run({%r: %%s}, validate=False)" % port
    if arm == "B":
        path = arms.LAZY / "out" / ("stage%d_algorithm.py" % stage)
        return path, path.stem, "M.run(%s)"
    path = OUT / "stage3_reference.py"
    path.write_text(arms.REFERENCE_SOURCE)
    return path, path.stem, "M.run(%s)"


def run_arm(g, stage, arm, inputs_path, cold_runs=COLD_RUNS):
    path, modname, entry = arm_module(g, stage, arm)
    child = OUT / ("child_s%d%s.py" % (stage, arm))
    child.write_text(CHILD % {"mod": modname,
                              "is3": "True" if stage == 3 else "False",
                              "call_first": entry % "cases[0]",
                              "call_all": entry % "c"})
    env = {"PATH": "/usr/bin:/bin"}
    runs = []
    for _ in range(cold_runs):
        t = time.perf_counter_ns()
        proc = subprocess.run(["/usr/bin/python3", "-I", str(child), str(path),
                               str(inputs_path)],
                              capture_output=True, text=True, env=env)
        wall = (time.perf_counter_ns() - t) / 1e6
        if proc.returncode != 0:
            return {"error": proc.stderr[-1500:], "returncode": proc.returncode}
        r = json.loads(proc.stdout)
        r["process_wall_ms"] = wall
        runs.append(r)

    def med(k):
        return statistics.median(r[k] for r in runs)
    return {"bytes_on_disk": path.stat().st_size, "source_path": str(path),
            "import_ms_median": med("import_ms"),
            "first_call_ms_median": med("first_call_ms"),
            "cold_start_ms_median": med("cold_start_ms"),
            "cold_start_ms_min": min(r["cold_start_ms"] for r in runs),
            "sweep_ms_median": med("sweep_ms"),
            "process_wall_ms_median": med("process_wall_ms"),
            "peak_rss_mb_median": med("peak_rss_mb"),
            "peak_rss_mb_max": max(r["peak_rss_mb"] for r in runs),
            "peak_rss_rusage_mb_median": med("peak_rss_rusage_mb"),
            "records": runs[0]["records"], "first": runs[0]["first"],
            "leaked_modules": runs[0]["leaked_modules"], "runs": cold_runs,
            "interpreter": "/usr/bin/python3 -I, env PATH only"}


def write_inputs(g, stage, which="uniform"):
    p = OUT / ("stage%d_%s_inputs.json" % (stage, which))
    recs_in = g[which]
    if stage == 3:
        seen, order = {}, []
        recs = []
        for pos, obs in recs_in:
            k = id(obs)
            if k not in seen:
                seen[k] = len(order)
                order.append(list(obs))
            recs.append([seen[k], pos])
        p.write_text(json.dumps({"obs": order, "recs": recs}))
    else:
        p.write_text(json.dumps({"recs": [list(r) for r in recs_in]}))
    return p


def main(stages=(1, 2, 3)):
    rep = {"protocol": "research/lazy-latency/PREREGISTRATION.md section 3",
           "cold_start_definition": "import + module compile, plus one call; the "
                                    "fixture load is excluded because it is "
                                    "identical for every arm and dominates",
           "rss_note": "the raster is shared by reference across records, as "
                       "FINDINGS section 48 measured the compiled path doing. "
                       "peak_rss_mb is /proc/self/status VmHWM, NOT ru_maxrss: "
                       "ru_maxrss is inherited across fork and reported this "
                       "measurement parent's 375 MB identically for every arm."}
    for stage in stages:
        print("== deployment, stage %d" % stage)
        g = arms.build(stage)
        ip = write_inputs(g, stage)
        rep["stage%d" % stage] = {a: run_arm(g, stage, a, ip) for a in sorted(g["arms"])}
        rep["stage%d" % stage]["_fixture_bytes"] = ip.stat().st_size
        if g.get("deploy"):
            # an out-of-process replication of the headline distribution, in a
            # different interpreter with no gc discipline and no venv
            dp = write_inputs(g, stage, "deploy")
            rep["stage%d_deploy" % stage] = {
                a: run_arm(g, stage, a, dp) for a in sorted(g["arms"])}
        for a, v in sorted(rep["stage%d" % stage].items()):
            if not isinstance(v, dict) or "bytes_on_disk" not in v:
                continue
            print("   %s  %8d B on disk  cold %7.2f ms  peak RSS %7.2f MB  leaked %s"
                  % (a, v.get("bytes_on_disk", -1), v.get("cold_start_ms_median", -1),
                     v.get("peak_rss_mb_median", -1),
                     v.get("leaked_modules", v.get("error"))))
    (OUT / "deployment.json").write_text(json.dumps(rep, indent=1, default=str))
    print("-> %s" % (OUT / "deployment.json"))
    return rep


if __name__ == "__main__":
    main(tuple(int(x) for x in sys.argv[1:]) or (1, 2, 3))
