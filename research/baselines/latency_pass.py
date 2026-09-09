"""One quiet pass measuring batch-one latency for every method, back to back.

All numbers use the tcn/runtime.py:benchmark protocol (perf_counter_ns around a
single call, 100 repetitions, sorted, median / index-95).  Running them in one
process on an otherwise idle machine keeps them comparable.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcn.runtime import load_program  # noqa: E402
from tcn.types import BOOL, Value, floating, product  # noqa: E402
from common import bench  # noqa: E402
import mixed_baseline as MB  # noqa: E402
import joint_baseline as JB  # noqa: E402

F = floating()
OUT = Path("research/baselines/out")


def numpy_call(m, dims):
    ws = [(l.weight.detach().numpy().astype(np.float32), l.bias.detach().numpy().astype(np.float32))
          for l in list(m.body) if isinstance(l, torch.nn.Linear)]
    heads = {k: (getattr(m, k).weight.detach().numpy().astype(np.float32),
                 getattr(m, k).bias.detach().numpy().astype(np.float32)) for k in dims}

    def call(v):
        h = v
        for w, b in ws:
            h = np.tanh(w @ h + b)
        return {k: (w @ h + b) for k, (w, b) in heads.items()}
    return call


def main():
    torch.set_num_threads(1)
    res = {}

    # ---- mixed task -------------------------------------------------------
    p, r = load_program(OUT / "tcn_mixed" / "program.json")
    tcn_inputs = [{"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "x": Value.of(F, x)}
                  for a, b in MB.AB for x in MB.TRAIN_X]
    res["mixed/tcn_frozen_program"] = bench(lambda i: p.run(i, registry=r), tcn_inputs,
                                            scope="exact model, typed input through typed output, batch one")
    res["mixed/tcn_frozen_program"]["description_bits"] = p.description_bits(r)

    cfg = json.loads((OUT / "mixed_baseline.json").read_text())["selected"]
    for mode in ("answer", "signals"):
        c = cfg[mode]["config"]
        m, _, _ = MB.train(c["width"], c["depth"], c["lr"], c["steps"], mode, seed=0)
        m.eval()
        with torch.no_grad():
            ts = [torch.tensor([[float(a), float(b), x]], dtype=torch.float32)
                  for a, b in MB.AB for x in MB.TRAIN_X]
            res[f"mixed/mlp_{mode}_torch"] = bench(lambda t: m(t), ts,
                                                   scope="torch MLP forward, batch one, no_grad")
            res[f"mixed/mlp_{mode}_numpy"] = MB.numpy_export_benchmark(m)

    res["mixed/python_oracle"] = MB.oracle_benchmark()

    # ---- joint task -------------------------------------------------------
    jp_path = OUT / "tcn_joint" / "program.json"
    if jp_path.exists():
        jp, jr = load_program(jp_path)
        host = JB.make_host(10000, "test", JB.OBJECTIVES[0])
        view = host.view()
        jin = [{"bits": view.observations["bits"], "goal": view.observations["goal"],
                "action": Value.of(product(F, F), (1.0, 0.0)), "dt": Value.of(F, 1.0)}]
        res["joint/tcn_frozen_program"] = bench(lambda i: jp.run(i, registry=jr), jin,
                                                scope="exact model, typed input through typed output, batch one")
        res["joint/tcn_frozen_program"]["description_bits"] = jp.description_bits(jr)

    jcfg = json.loads((OUT / "joint_baseline.json").read_text())["selected"]
    host = JB.make_host(10000, "test", JB.OBJECTIVES[0])
    vec = [JB.obs_vector(host.view(), 0)]
    for mode in ("reinforce", "aux"):
        c = jcfg[mode]["config"]
        m, _, _ = JB.train_agent(c["width"], c["depth"], c["lr"], mode, c["episodes"], seed=0)
        m.eval()
        with torch.no_grad():
            res[f"joint/mlp_{mode}_torch"] = bench(lambda t: m(t), vec,
                                                   scope="torch policy+value forward, batch one, no_grad")
            dims = ["policy", "value"] + (["prediction", "probe"] if mode == "aux" else [])
            call = numpy_call(m, dims)
            v = vec[0].numpy()
            res[f"joint/mlp_{mode}_numpy"] = bench(call, [v], scope="numpy-exported MLP, batch one")

    # The budget-matched winner from matched_budget.py: a 153-parameter net.
    import joint_supervised as JS  # noqa: E402
    from matched_budget import joint_matched  # noqa: E402
    for w, d in ((8, 2), (32, 2)):
        m, _, _, _ = joint_matched(w, d, 0.04, 0)
        m.eval()
        with torch.no_grad():
            res[f"joint/mlp_matched_w{w}_d{d}_torch"] = bench(lambda t: m(t), vec,
                scope="torch MLP forward, batch one, no_grad")
            ws = [(l.weight.detach().numpy().astype(np.float32), l.bias.detach().numpy().astype(np.float32))
                  for l in list(m.body) + [m.head] if isinstance(l, torch.nn.Linear)]

            def call(v, ws=ws):
                h = v
                for i, (a, b) in enumerate(ws):
                    h = a @ h + b
                    if i < len(ws) - 1:
                        h = np.tanh(h)
                return h
            res[f"joint/mlp_matched_w{w}_d{d}_numpy"] = bench(call, [vec[0].numpy()],
                scope="numpy-exported MLP, batch one")

    tbl, _, _ = JS.train_table(331)
    counts = tbl["t"]

    def table_call(v):
        k = sum(int(v[i] > 0.5) << i for i in range(5))
        c = counts.get(k)
        return 0 if c is None else int(c[1] > c[0])
    res["joint/lookup_table_32bit"] = bench(table_call, [vec[0].tolist()],
                                            scope="32-entry lookup, batch one")
    res["joint/lookup_table_32bit"]["description_bits"] = 32

    def oracle(v):
        return int((bool(v[0]) != bool(v[1])) != bool(v[4]))
    res["joint/python_oracle"] = bench(oracle, [vec[0].tolist()],
                                       scope="hand-written Python oracle, batch one")

    (OUT / "latency_pass.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()


def warmed():
    """Same protocol, but with 20 discarded warm-up calls first.

    The tcn benchmark does no warm-up, which is fine for a pure-Python program
    but charges torch's lazy first-call initialisation to the median in a
    100-repetition run.  Both sides get the same treatment here.
    """
    import mixed_baseline as MB
    import joint_baseline as JB
    import joint_supervised as JS
    from matched_budget import joint_matched
    torch.set_num_threads(1)
    res = {}

    def w(call, inputs, scope):
        for i in range(20):
            call(inputs[i % len(inputs)])
        return bench(call, inputs, scope=scope)

    p, r = load_program(OUT / "tcn_mixed" / "program.json")
    ti = [{"a": Value.of(BOOL, a), "b": Value.of(BOOL, b), "x": Value.of(F, x)}
          for a, b in MB.AB for x in MB.TRAIN_X]
    res["mixed/tcn_frozen_program"] = w(lambda i: p.run(i, registry=r), ti, "exact model, batch one, warmed")
    cfg = json.loads((OUT / "mixed_baseline.json").read_text())["selected"]
    for mode in ("answer", "signals"):
        c = cfg[mode]["config"]
        m, _, _ = MB.train(c["width"], c["depth"], c["lr"], c["steps"], mode, seed=0)
        m.eval()
        with torch.no_grad():
            ts = [torch.tensor([[float(a), float(b), x]], dtype=torch.float32)
                  for a, b in MB.AB for x in MB.TRAIN_X]
            res[f"mixed/mlp_{mode}_torch"] = w(lambda t: m(t), ts, "torch MLP, batch one, warmed")
    res["mixed/python_oracle"] = w(lambda t: math.sin(float(t[0] != t[1]) + t[2]),
                                   [(a, b, x) for a, b in MB.AB for x in MB.TRAIN_X],
                                   "python oracle, batch one, warmed")

    jp, jr = load_program(OUT / "tcn_joint" / "program.json")
    host = JB.make_host(10000, "test", JB.OBJECTIVES[0])
    view = host.view()
    jin = [{"bits": view.observations["bits"], "goal": view.observations["goal"],
            "action": Value.of(product(F, F), (1.0, 0.0)), "dt": Value.of(F, 1.0)}]
    res["joint/tcn_frozen_program"] = w(lambda i: jp.run(i, registry=jr), jin, "exact model, batch one, warmed")
    vec = [JB.obs_vector(view, 0)]
    for wd, dd in ((8, 2), (32, 2)):
        m, _, _, _ = joint_matched(wd, dd, 0.04, 0)
        m.eval()
        with torch.no_grad():
            res[f"joint/mlp_matched_w{wd}_d{dd}_torch"] = w(lambda t: m(t), vec, "torch MLP, batch one, warmed")
    tbl, _, _ = JS.train_table(331)
    counts = tbl["t"]

    def table_call(v):
        c = counts.get(sum(int(v[i] > 0.5) << i for i in range(5)))
        return 0 if c is None else int(c[1] > c[0])
    res["joint/lookup_table_32bit"] = w(table_call, [vec[0].tolist()], "32-entry lookup, batch one, warmed")
    res["joint/python_oracle"] = w(lambda v: int((bool(v[0]) != bool(v[1])) != bool(v[4])),
                                   [vec[0].tolist()], "python oracle, batch one, warmed")
    (OUT / "latency_warmed.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
