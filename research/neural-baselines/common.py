"""Shared measurement harness for the matched neural baselines.

Every arm reports the same six things for both methods, so the tables in
`RESULTS.md` are comparable cell by cell:

  parameters        trainable scalars (TCN frozen programs have none)
  serialized bytes  what you put on disk to ship it
  batch-one latency warm p50 and cold first-call, on this host, batch size one
  peak RSS          `ru_maxrss` of a fresh process that loads it and runs once
  training budget   examples / rollouts consumed, and optimizer steps
  task quality      on the artifact's own held-out set, beside its own trivial
                    baseline (constant / majority / random)

`bench` is a line-for-line reimplementation of `tcn/runtime.py:benchmark`'s
protocol (perf_counter_ns immediately around one batch-one call, sorted samples,
median), with an explicit warm-up count so the cold and warm numbers are
separable rather than blended -- the fault `research/baselines/RESULTS.md` §5
identified in the unwarmed figures.

Nothing under `tcn/` or `generators/` is imported for measurement purposes other
than read-only, and nothing there is modified.
"""
from __future__ import annotations

import gc
import gzip
import io
import json
import os
import pathlib
import resource
import statistics
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)
VENV_PYTHON = "/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python"
SYSTEM_PYTHON = "/usr/bin/python3"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- timing ----------------------------------------------------------------
def bench(fn, repetitions=100, warmup=20):
    """Warm batch-one p50, in ms. Mirrors `tcn/runtime.py:benchmark`."""
    for _ in range(warmup):
        fn()
    samples = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repetitions):
            start = time.perf_counter_ns()
            fn()
            samples.append((time.perf_counter_ns() - start) / 1e6)
    finally:
        gc.enable()
    samples.sort()
    return {"p50_ms": statistics.median(samples), "min_ms": samples[0],
            "p95_ms": samples[min(len(samples) - 1, int(.95 * len(samples)))],
            "repetitions": repetitions, "warmup": warmup}


def cold_call_ms(fn):
    """The very first call, with nothing warmed. Charged honestly, not hidden."""
    gc.collect()
    start = time.perf_counter_ns()
    fn()
    return (time.perf_counter_ns() - start) / 1e6


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def timed(fn, repeats=3):
    samples = []
    gc.collect()
    gc.disable()
    try:
        for _ in range(repeats):
            t = time.perf_counter_ns()
            value = fn()
            samples.append((time.perf_counter_ns() - t) / 1e6)
    finally:
        gc.enable()
    return statistics.median(samples), min(samples), value


# --- size ------------------------------------------------------------------
def torch_size_report(model):
    """Serialized bytes three ways, so 'size' is not one number's artifact.

    * `torch_save_bytes`  -- `torch.save(state_dict)`, what you actually ship.
    * `float32_bytes`     -- 4 x parameters, the information you need to run it.
    * `gzip_bytes`        -- the float32 payload deflated, for parity with the
                             gzip column TCN artifacts are reported with.
    * `json_bytes`        -- weights as JSON text, the measure `description_bits`
                             uses on the TCN side; included only so the two are
                             on the same footing when that column is quoted.
    """
    import torch
    params = sum(p.numel() for p in model.parameters())
    buf = io.BytesIO()
    torch.save({k: v for k, v in model.state_dict().items()}, buf)
    blob = buf.getvalue()
    flat = b"".join(p.detach().to(torch.float32).numpy().tobytes()
                    for p in model.parameters())
    text = json.dumps([p.detach().reshape(-1).tolist() for p in model.parameters()])
    return {"parameters": params, "torch_save_bytes": len(blob),
            "float32_bytes": len(flat), "float32_bits": 8 * len(flat),
            "gzip_bytes": len(gzip.compress(flat, 9)),
            "gzip_bits": 8 * len(gzip.compress(flat, 9)),
            "json_bytes": len(text), "json_bits": 8 * len(text)}


def program_size_report(program, registry):
    """The TCN side of the size column, reported the same three ways.

    A reimplementation of `research/inference-cost/harness.py:size_report` (that
    module hardcodes the main checkout's paths and writes into that track's
    `out/`, which this track must not touch). `description_bits` is the repo's own
    measure -- `8 * len(json.dumps(to_dict(), sort_keys=True))` -- and is reported
    beside gzip so the reader can see how much of it is punctuation, exactly as
    `research/inference-cost/RESULTS.md` §3 established.
    """
    pruned = program.pruned()
    blob = json.dumps(pruned.to_dict(), sort_keys=True).encode()
    modules = {}

    def walk(p):
        for node in p.nodes:
            for candidate in node.candidates:
                for name in (candidate.operator.name,
                             dict(candidate.operator.parameters).get("module", "")):
                    if isinstance(name, str) and name.startswith("module:") and name not in modules:
                        modules[name] = registry.modules[name]
                        walk(registry.modules[name])

    walk(pruned)
    total = blob + b"".join(json.dumps(v.to_dict(), sort_keys=True).encode()
                            for v in modules.values())
    return {"trainable_parameters": 0,
            "pruned_nodes": len(pruned.nodes), "module_count": len(modules),
            "total_static_nodes": len(pruned.nodes) + sum(len(v.nodes) for v in modules.values()),
            "distinct_operators": len({c.operator.name for p in [pruned, *modules.values()]
                                       for n in p.nodes for c in n.candidates}),
            "description_bits": pruned.description_bits(registry),
            "json_bytes": len(total), "json_bits": 8 * len(total),
            "gzip_bytes": len(gzip.compress(total, 9)),
            "gzip_bits": 8 * len(gzip.compress(total, 9)),
            "estimated_operator_cost": pruned.execution_cost(registry)}


# --- out-of-process cold start --------------------------------------------
def subprocess_cold_start(script_text, python=None, label=""):
    """Run a fresh interpreter that loads the artifact and infers once.

    Returns wall time of the whole process and the peak RSS the child reports
    for itself, so the number is comparable with the `.pyz` rows in
    `research/inference-cost/RESULTS.md`, which are measured the same way.
    """
    python = python or VENV_PYTHON
    path = OUT / f"_cold_{label or 'x'}.py"
    path.write_text(script_text)
    started = time.perf_counter()
    proc = subprocess.run([python, "-I", str(path)], capture_output=True, text=True)
    wall = (time.perf_counter() - started) * 1e3
    if proc.returncode != 0:
        return {"failed": proc.stderr[-800:]}
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    payload["process_wall_ms"] = wall
    return payload


# --- reporting -------------------------------------------------------------
def dump(name, obj):
    path = OUT / f"{name}.json"
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    print(f"-> {path}", flush=True)
    return path


def load(name):
    return json.loads((OUT / f"{name}.json").read_text())


def report(name, value):
    print(f"{name:60s} {value}", flush=True)


def majority_reference(labels):
    """The constant baseline every quality figure has to be read against."""
    labels = list(labels)
    n = max(1, len(labels))
    positives = sum(bool(x) for x in labels)
    return {"n": len(labels), "positive_rate": positives / n,
            "majority_constant": max(positives, len(labels) - positives) / n,
            "random": 0.5}
