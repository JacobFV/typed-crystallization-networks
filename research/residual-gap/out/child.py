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
