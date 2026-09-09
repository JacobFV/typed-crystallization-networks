"""Every method's batch-one cost, measured back to back in one process.

`research/baselines/RESULTS.md` §5 set the protocol and the caveat, and both are
kept: `common.py:bench` is a line-for-line reimplementation of
`tcn/runtime.py:benchmark` (perf_counter_ns immediately around one batch-one
call, sorted samples, median), with the warm-up made explicit so the cold first
call is reported as its own number instead of being blended into the median.

Two things worth stating before the table is read:

* **Latency, parameter count and serialized size do not depend on the weights'
  values, only on the architecture.** The neural rows are therefore measured on
  the same architectures the sweeps selected, freshly constructed; nothing here
  needs the fitted checkpoints, and nothing here can be improved by having them.
* **The neural rows are torch, and torch dispatch dominates at these sizes.**
  `research/baselines/RESULTS.md` measured the same weights at 10-20x faster once
  re-expressed as plain numpy matmuls. So each neural latency below is an *upper
  bound* on what that model costs deployed -- the conservative direction for a
  comparison whose question is whether the typed program is cheaper.

The host is shared and was under load throughout this track; every row is a
median of 100 warm calls and the p95 is recorded in the JSON.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import torch

import common as harness

torch.set_num_threads(1)


def count_macs(model, call):
    """Multiply-accumulates for one batch-one forward pass.

    The host is shared and every wall clock here is contended, so each latency is
    reported beside a load-independent count -- the discipline
    `research/inference-cost/RESULTS.md` imposes with its operator-application
    column. This is the neural side's equivalent unit.
    """
    total = [0]

    def hook(module, inputs, output):
        if isinstance(module, torch.nn.Linear):
            total[0] += module.in_features * module.out_features * output.numel() // max(
                1, output.shape[-1])
        elif isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            kernel = 1
            for k in module.kernel_size:
                kernel *= k
            positions = output.numel() // max(1, output.shape[1])
            total[0] += module.in_channels * module.out_channels * kernel * positions
        elif isinstance(module, torch.nn.GRU):
            steps = inputs[0].shape[1] if module.batch_first else inputs[0].shape[0]
            total[0] += 3 * steps * module.hidden_size * (module.input_size + module.hidden_size)
        elif isinstance(module, torch.nn.MultiheadAttention):
            length, dim = inputs[0].shape[1], module.embed_dim
            total[0] += 4 * length * dim * dim + 2 * length * length * dim

    handles = [m.register_forward_hook(hook) for m in model.modules()]
    with torch.no_grad():
        call()
    for h in handles:
        h.remove()
    return total[0]


def neural_rows(specs):
    import language_baseline as LB
    import visual_baseline as VB
    import computer_baseline as CB

    rows = []
    for spec in specs:
        arm, name, extra = spec["arm"], spec["model"], spec.get("extra", {})
        if arm == "language":
            model = LB.ZOO[name](extra.get("aux", False))
            tokens = torch.randint(0, 256, (1, LB.CAP))
            length = torch.tensor([[100.0]])
            call = lambda m=model, t=tokens, l=length: m(t, l)
        elif arm == "visual":
            model = VB.ZOO[name]()
            pixels = torch.rand(1, 3, 32, 32)
            call = lambda m=model, p=pixels: m(p)
        else:
            factory = CB.ZOO.get(name) or CB.HINT_ZOO[name]
            model = factory(extra.get("byte_mode", "reg"))
            tokens = torch.randint(0, 256, (1, CB.PREFIX))
            length = torch.tensor([[44.0]])
            previous = torch.tensor([[1.0, 0.0, 0.0]])
            call = lambda m=model, t=tokens, l=length, p=previous: m(t, l, p)
        macs = count_macs(model, call)
        with torch.no_grad():
            cold = harness.cold_call_ms(call)
            warm = harness.bench(call, repetitions=spec.get("repetitions", 100))
        rows.append({"method": "neural", "arm": arm, "model": name, **extra,
                     **harness.torch_size_report(model), "macs_per_inference": macs,
                     "cold_first_call_ms": cold, "warm_p50_ms": warm["p50_ms"],
                     "warm_p95_ms": warm["p95_ms"], "warm_min_ms": warm["min_ms"]})
        harness.report(f"neural {arm}/{name}",
                       f"{rows[-1]['parameters']} params, {macs} MACs, "
                       f"warm {warm['p50_ms']:.4f} ms (min {warm['min_ms']:.4f}), "
                       f"cold {cold:.3f} ms")
    return rows


def neural_process_cold_start(specs):
    """A fresh interpreter that imports torch, builds the model and infers once.

    The comparable TCN row is `research/inference-cost/RESULTS.md`'s `.pyz` cold
    start under `/usr/bin/python3 -I` -- no torch, no numpy, no repository. The
    asymmetry is real and is the point of the row: the typed artifact's runtime
    dependency is the standard library.
    """
    rows = []
    for spec in specs:
        script = f'''
import json, resource, sys, time
t0 = time.perf_counter()
sys.path.insert(0, {str(ROOT)!r})
sys.path.insert(0, {str(HERE)!r})
import torch
torch.set_num_threads(1)
import_ms = (time.perf_counter() - t0) * 1e3
{spec["build"]}
t1 = time.perf_counter()
with torch.no_grad():
    out = call()
infer_ms = (time.perf_counter() - t1) * 1e3
print(json.dumps({{"arm": {spec["arm"]!r}, "model": {spec["model"]!r},
                   "import_torch_ms": import_ms, "first_inference_ms": infer_ms,
                   "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}}))
'''
        result = harness.subprocess_cold_start(script, label=f'{spec["arm"]}_{spec["model"]}')
        rows.append(result)
        harness.report(f"cold process {spec['arm']}/{spec['model']}", json.dumps(result))
    return rows


BUILDS = {
    "language": '''
import language_baseline as LB
model = LB.ZOO[{model!r}]({aux})
tokens = torch.randint(0, 256, (1, LB.CAP)); length = torch.tensor([[100.0]])
call = lambda: model(tokens, length)
''',
    "visual": '''
import visual_baseline as VB
model = VB.ZOO[{model!r}]()
pixels = torch.rand(1, 3, 32, 32)
call = lambda: model(pixels)
''',
    "computer": '''
import computer_baseline as CB
factory = CB.ZOO.get({model!r}) or CB.HINT_ZOO[{model!r}]
model = factory({byte_mode!r})
tokens = torch.randint(0, 256, (1, CB.PREFIX)); length = torch.tensor([[44.0]])
previous = torch.tensor([[1.0, 0.0, 0.0]])
call = lambda: model(tokens, length, previous)
''',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=str(HERE / "latency_spec.json"),
                    help="which architectures to price; written by the sweeps' verdicts")
    ap.add_argument("--tag", default="latency")
    args = ap.parse_args()

    specs = json.loads(pathlib.Path(args.spec).read_text())
    report = {"protocol": "tcn/runtime.py:benchmark, warm-up separated",
              "torch_threads": 1,
              "in_process": neural_rows(specs),
              "cold_process": neural_process_cold_start(
                  [{"arm": s["arm"], "model": s["model"],
                    "build": BUILDS[s["arm"]].format(
                        model=s["model"], aux=s.get("extra", {}).get("aux", False),
                        byte_mode=s.get("extra", {}).get("byte_mode", "reg"))}
                   for s in specs])}
    harness.dump(args.tag, report)
    print(json.dumps(report["in_process"], indent=1))


if __name__ == "__main__":
    main()
