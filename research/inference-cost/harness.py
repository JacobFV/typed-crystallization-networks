"""Shared measurement helpers. Nothing under tcn/ or generators/ is modified.

Two instruments the wall clock cannot replace, because the host is shared:

* `ops` counts every application of `Registry.exact` -- one typed operator
  evaluation -- by wrapping the bound method for the duration of a call. It is
  the load-independent unit: latency divided by it is cost per operator.
* `Stage` records the itemised path so the report can say where the time went
  rather than quoting one end-to-end number.
"""
from __future__ import annotations
import gc, json, os, pathlib, resource, statistics, sys, time

ROOT = pathlib.Path('/home/brandonin/Documents/typed-crystallization-networks')
OUT = ROOT / 'research/inference-cost/out'
OUT.mkdir(parents=True, exist_ok=True)


class Counter:
    """Count typed operator applications inside a `with` block."""
    def __init__(self, registry):
        self.registry = registry; self.n = 0
    def __enter__(self):
        self._real = self.registry.exact
        def counted(op, args):
            self.n += 1; return self._real(op, args)
        self.registry.exact = counted; return self
    def __exit__(self, *a):
        self.registry.exact = self._real; return False


def timed(fn, repeats=3):
    """Median of `repeats` wall-clock samples, in ms, gc disabled."""
    samples = []
    gc.collect(); gc.disable()
    try:
        for _ in range(repeats):
            t = time.perf_counter_ns(); value = fn(); samples.append((time.perf_counter_ns() - t) / 1e6)
    finally:
        gc.enable()
    return statistics.median(samples), min(samples), value


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dump(name, obj):
    p = OUT / f'{name}.json'
    p.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    print(f'-> {p}')
    return p


def size_report(program, registry):
    """Serialized bits, and the same program with the JSON's slack removed.

    `description_bits` is `8 * len(json.dumps(to_dict(), sort_keys=True))` --
    pretty-printed keys, type records repeated at every node, decimal integers.
    Three progressively tighter measures are reported beside it so the reader can
    see how much of the headline number is punctuation:

    * `json_bits`      -- what `description_bits` returns.
    * `gzip_bits`      -- the same JSON, deflated. Removes repeated key names and
                          repeated type records; a lower bound on how much of the
                          figure was textual redundancy.

    The learned content itself is not a property of the frozen program -- it is
    log2 of the search space the selection was drawn from, which lives in each
    track's own recorded JSON, and is reported per artifact in RESULTS.md.
    """
    import gzip
    pruned = program.pruned()
    blob = json.dumps(pruned.to_dict(), sort_keys=True).encode()
    modules = {}
    def walk(p):
        for node in p.nodes:
            for c in node.candidates:
                for nm in (c.operator.name, dict(c.operator.parameters).get('module', '')):
                    if isinstance(nm, str) and nm.startswith('module:') and nm not in modules:
                        modules[nm] = registry.modules[nm]; walk(registry.modules[nm])
    walk(pruned)
    module_blobs = {k: json.dumps(v.to_dict(), sort_keys=True).encode() for k, v in modules.items()}
    total = blob + b''.join(module_blobs.values())
    return {
        'pruned_nodes': len(pruned.nodes),
        'module_count': len(modules),
        'module_nodes': {k: len(v.nodes) for k, v in modules.items()},
        'total_static_nodes': len(pruned.nodes) + sum(len(v.nodes) for v in modules.values()),
        'distinct_operators': len({c.operator.name for p in [pruned, *modules.values()]
                                   for n in p.nodes for c in n.candidates}),
        'description_bits': pruned.description_bits(registry),
        'json_bytes': len(total),
        'gzip_bytes': len(gzip.compress(total, 9)),
        'gzip_bits': len(gzip.compress(total, 9)) * 8,
        'estimated_operator_cost': pruned.execution_cost(registry),
    }
