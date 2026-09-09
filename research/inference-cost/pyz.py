"""The artifact a deployment would actually ship: `export_executable`'s `.pyz`.

For each capability this exports the frozen program with `tcn/runtime.py:export_executable`
and then runs it under the *system* interpreter with `python3 -I`: no repository
on the path, no virtualenv, no torch, no numpy, isolated from user site-packages.
That is the only configuration in this report that measures what a deployment costs.

Four numbers per artifact, each separated so none can hide inside another:
  cold_start_ms   process spawn + zipapp import + artifact load + digest check,
                  measured with an empty stdin so no inference runs at all
  transport       bytes and milliseconds to serialize one input as the JSON line
                  the `.pyz` reads on stdin, and to parse the JSON it writes back
  inference_ms    (total with N inputs - cold start) / N
  peak_rss_mb     the child process's maximum resident set
"""
import sys, json, os, re, resource, subprocess, time, statistics, pathlib
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT + '/research/inference-cost')
from harness import dump, OUT
from tcn.runtime import export_executable

SYS_PY = '/usr/bin/python3'


def run_pyz(path, lines, repeats=5):
    """Wall clock over `repeats` spawns, plus the child's own peak RSS from
    `/usr/bin/time -v`. The minimum is reported alongside the median because the
    host is shared: on a loaded machine the minimum is the closest available
    estimate of the cost, and the median is the closest to what a user sees."""
    payload = ''.join(lines)
    samples, out, rss = [], None, 0.0
    for _ in range(repeats):
        t = time.perf_counter_ns()
        r = subprocess.run(['/usr/bin/time', '-v', SYS_PY, '-I', str(path)],
                           input=payload, capture_output=True, text=True)
        samples.append((time.perf_counter_ns() - t) / 1e6)
        if r.returncode != 0:
            return {'failed': r.stderr[-800:]}
        out = r.stdout
        m = re.search(r'Maximum resident set size \(kbytes\): (\d+)', r.stderr)
        if m:
            rss = max(rss, int(m.group(1)) / 1024.0)
    return {'ms': statistics.median(samples), 'min_ms': min(samples), 'stdout': out, 'rss_mb': rss}


def measure(name, program, registry, inputs, repeats=3, batch=1):
    path = OUT / f'{name}.pyz'
    export_executable(program, path, registry)
    line_t = time.perf_counter_ns()
    line = json.dumps({'inputs': {k: v.to_dict() for k, v in inputs.items()}}, sort_keys=True) + '\n'
    serialize_ms = (time.perf_counter_ns() - line_t) / 1e6

    cold = run_pyz(path, [], repeats)
    one = run_pyz(path, [line], repeats)
    result = {'name': name,
              'pyz_bytes': path.stat().st_size,
              'interpreter': SYS_PY, 'flags': '-I (isolated: no repo, no venv, no torch)',
              'transport_bytes': len(line.encode()), 'transport_serialize_ms': serialize_ms,
              'cold_start_ms': cold.get('ms'), 'cold_start_min_ms': cold.get('min_ms'),
              'cold_start_peak_rss_mb': cold.get('rss_mb'),
              'one_inference_total_ms': one.get('ms'), 'one_inference_min_ms': one.get('min_ms'),
              'one_inference_peak_rss_mb': one.get('rss_mb')}
    if 'failed' in cold or 'failed' in one:
        result['failed'] = cold.get('failed') or one.get('failed')
        return result
    result['inference_ms'] = one['min_ms'] - cold['min_ms']
    result['reply_bytes'] = len(one['stdout'].encode())
    if batch > 1:
        many = run_pyz(path, [line] * batch, repeats)
        if 'ms' in many:
            result['batch_n'] = batch
            result['batch_total_ms'] = many['ms']
            result['amortized_inference_ms'] = (many['min_ms'] - cold['min_ms']) / batch
            result['batch_peak_rss_mb'] = many.get('rss_mb')
    return result


# ---------------------------------------------------------------- artifacts
def mixed():
    from tcn.runtime import load_program
    from tcn.types import Value, BOOL, floating
    p, r = load_program(ROOT + '/artifacts/demo/synthesis/program.json')
    F = floating()
    ins = {k: Value.of(t, True if t.kind == 'bool' else 0.13) for k, t in p.inputs}
    return 'mixed', p, r, ins, 2000


def language():
    sys.path.insert(0, ROOT + '/research/language-capability')
    import common as LC, scaffolds
    from run_stage_b import build_module
    sel = json.load(open(ROOT + '/research/language-capability/stage_b.json'))['enumeration']['selections']
    module, registry, _ = build_module()
    prog, _ = scaffolds.stage_b(module, registry)
    frozen = prog.harden(sel).pruned()
    ep = LC.dataset(1, seed0=100000, split='test')[0]
    return 'language', frozen, registry, {'text': ep['text']}, 20


def computer():
    sys.path.insert(0, ROOT + '/research/computer-capability')
    import program as P, task as T, closed_loop as CL
    from tcn.operators import Registry
    found = json.loads(open(ROOT + '/research/computer-capability/out/search.json').read())
    registry = Registry()
    t = P.transform_program(registry); pol = P.policy_program(registry)
    frozen = P.agent_program(registry, t, pol, found['transform']['enumeration']['selections'],
                             found['policy']['enumeration']['selections'])
    host = T.host('k', 7, 3, objective_path=T.TASK_PATH, seed=0, index=4000, split='test')
    host.step((T.read_action(),))
    view = host.view('agent_0')
    from tcn.types import Value
    types = dict(frozen.inputs)
    from tcn.generation import text_value
    ins = {'terminal': view.observations['terminal'],
           'action': Value.of(types['action'], (1., 0., 0.)),
           'action.2.text': text_value('', 512)}
    return 'computer', frozen.pruned(), registry, ins, 20


def visual():
    sys.path.insert(0, ROOT + '/research/visual-ladder')
    from common import FLAT, Registry, bytes_type, episode, load
    from tcn.types import Value
    import rung3_widgets as R
    found = load('rung3'); registry = Registry()
    probe = episode(0, 'train', **FLAT); W, H = probe['width'], probe['height']
    offsets = R.offset_pool(W)
    p0 = R.same_scaffold(registry, W, H); m0 = registry.register_module(p0.harden(found['s0']['chosen']))
    p1 = R.corner_scaffold(registry, W, H, m0, offsets); m1 = registry.register_module(p1.harden(found['s1']['chosen']))
    p2 = R.rect_scaffold(registry, W, H, m0, offsets); m2 = registry.register_module(p2.harden(found['s2']['chosen']))
    ep = episode(200, 'test', **FLAT)
    BT = bytes_type(ep['width'], ep['height'])
    parser = R.assembly(registry, BT, R.interior_positions(ep), m1, m2)
    return 'visual', parser, registry, {'observation': Value.of(BT, ep['pixels'])}, 1


if __name__ == '__main__':
    wanted = sys.argv[1:] or ['mixed', 'language', 'computer', 'visual']
    rows = []
    for key in wanted:
        name, prog, reg, ins, batch = {'mixed': mixed, 'language': language,
                                       'computer': computer, 'visual': visual}[key]()
        row = measure(name, prog, reg, ins, repeats=7 if key != 'visual' else 3, batch=batch)
        rows.append(row); print(json.dumps(row, indent=1)[:1400])
    dump('pyz' if len(wanted) > 1 else 'pyz_' + wanted[0], rows)
