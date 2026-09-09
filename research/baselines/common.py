"""Shared measurement helpers for matched-information baselines.

Latency is measured exactly the way `tcn/runtime.py:benchmark` measures it:
`time.perf_counter_ns()` immediately around one batch-one forward call, cycling
over a pre-built list of inputs, 100 repetitions, sorted samples, median for p50
and `samples[int(.95*n)]` for p95. Nothing is warmed up beyond what the TCN
benchmark warms up (i.e. nothing), so both sides pay the same first-call cost.
"""
import json
import statistics
import time

import torch


def bench(call, inputs, repetitions=100, scope=""):
    """Identical protocol to tcn.runtime.benchmark."""
    samples = []
    for i in range(repetitions):
        x = inputs[i % len(inputs)]
        start = time.perf_counter_ns()
        call(x)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return {
        "scope": scope,
        "repetitions": repetitions,
        "p50_ms": statistics.median(samples),
        "p95_ms": samples[min(len(samples) - 1, int(0.95 * repetitions))],
    }


def param_count(module):
    return sum(p.numel() for p in module.parameters())


def description_bits(module):
    """Two honest readings of 'serialized size'.

    - `float32_bits`: the information actually needed to run the net (weights at
      the precision they are used).
    - `json_bits`: the same measure TCN reports, i.e. 8 * len(json.dumps(...)) of
      the artifact the repo would write. TCN's 17,728 bits is an indented-free
      `json.dumps(program.to_dict())`, so we mirror that for the state dict.
    """
    n = param_count(module)
    state = {k: v.detach().flatten().tolist() for k, v in module.state_dict().items()}
    blob = json.dumps(state, sort_keys=True)
    return {"params": n, "float32_bits": 32 * n, "json_bits": 8 * len(blob.encode())}


def set_threads():
    torch.set_num_threads(1)
