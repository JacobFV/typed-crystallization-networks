"""What each arm actually ships, run on a stock interpreter.

`/usr/bin/python3 -I` with no repository on the path, no virtualenv, no torch,
no numpy and -- for arms C and D -- no `tcn` either.  `cold start` is measured
with empty stdin so no inference runs: process spawn, import, artifact load.
`inference` is amortized over a warm batch and includes parsing the JSON input
line and serializing the JSON reply, i.e. the transport.

Arm A is `tcn.runtime.export_executable` unchanged.  Arm C is the generated
module plus a JSON front end that rebuilds native values from the artifact's own
interned type table.  Arm D is the hand-written function with the same front end.
"""
from __future__ import annotations

import gzip
import inspect
import json
import pathlib
import resource
import statistics
import subprocess
import sys
import time
import zipapp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import fixtures
from harness import dump
from tcn.compile import compile_program
from tcn.runtime import export_executable
from tcn.types import Value

PY = '/usr/bin/python3'

FRONT_END = '''import json, sys
{load}

def _from_json(tid, x):
    t = TYPES[tid]
    k = t["kind"]
    if k == "tuple":
        return tuple(_from_json(i, v) for i, v in zip(t["items"], x))
    if k == "set":
        return frozenset(_from_json(t["items"][0], v) for v in x)
    return x

def _to_json(x):
    if isinstance(x, frozenset):
        return sorted((_to_json(v) for v in x), key=repr)
    if isinstance(x, tuple):
        return [_to_json(v) for v in x]
    return x

state = None
for line in sys.stdin:
    row = json.loads(line)
    ins = {{k: _from_json(dict((a, b) for a, b in INPUTS)[k], v)
            for k, v in row["inputs"].items()}}
    out, state = run(ins, state)
    print(json.dumps({{"outputs": {{k: _to_json(v) for k, v in out.items()}}}},
                     sort_keys=True), flush=True)
'''


def build_c(name, res, out_dir):
    root = out_dir / f'{name}_c'
    if root.exists():
        for f in root.iterdir():
            f.unlink()
    root.mkdir(parents=True, exist_ok=True)
    (root / 'compiled.py').write_text(res.source)
    (root / '__main__.py').write_text(FRONT_END.format(
        load='from compiled import run, TYPES, INPUTS'))
    path = out_dir / f'{name}_c.pyz'
    zipapp.create_archive(root, path, interpreter='/usr/bin/env python3', compressed=True)
    return path


def build_d(name, fx, mod, out_dir):
    root = out_dir / f'{name}_d'
    if root.exists():
        for f in root.iterdir():
            f.unlink()
    root.mkdir(parents=True, exist_ok=True)
    body = inspect.getsource(fx['reference'])
    body = body[body.index('def '):].replace('def reference(c):', 'def _ref(c):', 1)
    extra = ''.join(f'    c[{k!r}] = {v!r}\n' for k, v in fx['cases'][0].items()
                    if k.startswith('_'))
    src = ['import math',
           'TYPES = %r' % (mod.TYPES,),
           'INPUTS = %r' % (mod.INPUTS,),
           '',
           body,
           '',
           'def run(c, state=None):',
           extra,
           '    return _ref(c), None']
    (root / 'compiled.py').write_text('\n'.join(src))
    (root / '__main__.py').write_text(FRONT_END.format(
        load='from compiled import run, TYPES, INPUTS'))
    path = out_dir / f'{name}_d.pyz'
    zipapp.create_archive(root, path, interpreter='/usr/bin/env python3', compressed=True)
    return path


def _spawn(path, text):
    """One child under `/usr/bin/time`, so its own peak RSS is attributable."""
    t = time.perf_counter()
    proc = subprocess.run(['/usr/bin/time', '-f', '%M', PY, '-I', str(path)],
                          input=text, capture_output=True, text=True, check=True)
    wall = (time.perf_counter() - t) * 1e3
    rss = int(proc.stderr.strip().splitlines()[-1]) / 1024.0
    return wall, rss, proc.stdout


def run_pyz(path, payload, repeats, samples=5):
    """Cold start with empty stdin, then amortized inference on a warm process."""
    colds, rsss = [], []
    for _ in range(samples):
        w, rss, _ = _spawn(path, '')
        colds.append(w)
        rsss.append(rss)
    cold = statistics.median(colds)
    # amortized cost is the *difference* between a 2N and an N batch, so process
    # spawn, import and artifact load cancel instead of being subtracted noisily
    walls, brss, stdout = {}, [], ''
    for n in (repeats, 2 * repeats):
        lines = ''.join(payload for _ in range(n))
        ws = []
        for _ in range(3):
            w, rss, stdout = _spawn(path, lines)
            ws.append(w)
            brss.append(rss)
        walls[n] = statistics.median(ws)
    wall = walls[repeats]
    n_out = len([x for x in stdout.splitlines() if x.strip()])
    return {'cold_start_ms': cold, 'batch_wall_ms': wall, 'repeats': repeats,
            'batch_wall_2n_ms': walls[2 * repeats],
            'amortized_inference_ms': (walls[2 * repeats] - walls[repeats]) / max(1, repeats),
            'peak_rss_mb_cold': statistics.median(rsss),
            'peak_rss_mb_batch': statistics.median(brss),
            'outputs': n_out, 'first_output_bytes': len(stdout.splitlines()[0])
            if stdout.strip() else 0}


def to_json(x):
    if isinstance(x, frozenset):
        return sorted((to_json(v) for v in x), key=repr)
    if isinstance(x, tuple):
        return [to_json(v) for v in x]
    return x


def main(name, arms='cd', repeats=None):
    fx = fixtures.FIXTURES[name]()
    p, r = fx['program'], fx['registry']
    res = compile_program(p, r)
    mod = res.module()
    out_dir = HERE / 'out'
    c0 = fx['cases'][0]
    types = dict(p.inputs)
    native_payload = json.dumps({'inputs': {k: to_json(c0[k]) for k in types}}) + '\n'
    # arm A is 2-4 orders slower per line, so it needs far fewer repeats for the
    # 2N-minus-N difference to clear the host's noise floor; arms C and D need many
    reps_a = repeats or {'mixed': 2000, 'language': 20, 'visual': 1}[name]
    reps_fast = repeats or {'mixed': 20000, 'language': 2000, 'visual': 20}[name]
    report = {'artifact': name, 'interpreter': PY, 'transport_in_bytes': len(native_payload)}

    if 'c' in arms:
        path = build_c(name, res, out_dir)
        report['C'] = dict(run_pyz(path, native_payload, reps_fast),
                           bytes_on_disk=path.stat().st_size,
                           gzip_bytes=len(gzip.compress(path.read_bytes(), 9)))
    if 'd' in arms:
        path = build_d(name, fx, mod, out_dir)
        report['D'] = dict(run_pyz(path, native_payload, reps_fast),
                           bytes_on_disk=path.stat().st_size,
                           gzip_bytes=len(gzip.compress(path.read_bytes(), 9)))
    if 'a' in arms:
        path = out_dir / f'{name}_a.pyz'
        export_executable(p, path, r)
        payload = json.dumps({'inputs': {k: Value.of(types[k], c0[k]).to_dict()
                                         for k in types}}) + '\n'
        report['transport_in_bytes_A'] = len(payload)
        report['A'] = dict(run_pyz(path, payload, reps_a),
                           bytes_on_disk=path.stat().st_size,
                           gzip_bytes=len(gzip.compress(path.read_bytes(), 9)))
    dump(f'deploy_{name}', report)
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'cd')
