"""Bit-identity check for the default `logic` stream across 192 seed/config pairs.

The gate-set observation channel is an *addition*: it must not perturb any
episode recorded without it. `Host.snapshot` embeds `source_fingerprint()`,
which changes whenever any `.py` under `tcn/` or `generators/` changes, so the
snapshot digest cannot be the comparison. Everything else can: this hashes the
episode with the `source` field removed, which covers state, observations,
probes, latents, rewards, action menus and inputs.

Usage:
    python check_default_stream.py capture out/default_stream_before.json
    python check_default_stream.py compare out/default_stream_before.json
"""
from __future__ import annotations
import hashlib
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tcn.generation import Host, Action
from tcn.types import BOOL, Value

# The same shape of sweep FINDINGS section 10 used for the earlier generator
# change: every default-relevant configuration axis crossed with several seeds.
CONFIGS = [
    {},
    {'depth': 1},
    {'depth': 2},
    {'depth': 3},
    {'depth': 8},
    {'depth': 1, 'fixed_inputs': True},
    {'depth': 3, 'fixed_inputs': True},
    {'depth': 1, 'table': 6},
    {'depth': 3, 'table': 6, 'fixed_inputs': True},
    {'depth': 2, 'inputs': 2},
    {'depth': 4, 'inputs': 6},
    {'depth': 3, 'nondegenerate': True},
    {'depth': 4, 'nondegenerate': True, 'min_relevant_inputs': 2},
    {'depth': 6, 'min_relevant_inputs': 3},
    {'depth': 2, 'table': 11},
    {'depth': 5, 'inputs': 3, 'nondegenerate': True},
]
SEEDS = (0, 1, 2)
SPLITS = ('train', 'test', 'validation', 'holdout')


def episode_hash(seed, index, split, configuration):
    host = Host.create('logic', seed=seed, index=index, split=split,
                       configuration=dict(configuration) | {'horizon': 4})
    for t in range(4):
        if host.records[-1].done:
            break
        host.step((Action('answer', arguments=(('value', Value.of(BOOL, t % 2 == 0)),)),), 1.)
    snapshot = host.snapshot()
    snapshot.pop('source')
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sweep():
    out = {}
    for (i, configuration), seed, split in itertools.product(enumerate(CONFIGS), SEEDS, SPLITS):
        key = f'{i}|{seed}|{split}'
        try:
            out[key] = episode_hash(seed, 100 + i, split, configuration)
        except Exception as exc:                       # a rejected configuration is part of the stream
            out[key] = f'error:{type(exc).__name__}:{exc}'
    return out


if __name__ == '__main__':
    mode, path = sys.argv[1], Path(sys.argv[2])
    current = sweep()
    if mode == 'capture':
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f'captured {len(current)} episodes -> {path}')
    else:
        before = json.loads(path.read_text())
        if set(before) != set(current):
            raise SystemExit('sweep key set changed')
        bad = {k: (before[k], current[k]) for k in before if before[k] != current[k]}
        print(f'{len(current)} episodes compared, {len(bad)} differ')
        if bad:
            for k, (a, b) in list(bad.items())[:10]:
                print(' ', k, a[:16], '!=', b[:16])
            raise SystemExit(1)
        print('default stream bit-identical')
