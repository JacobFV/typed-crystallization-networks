"""In-process complete path for the language program and the mixed fixture.

`mixed` is the historical reference: `docs/VALIDATION.md`'s 0.0487 ms is the
`execute` cell of this table and nothing else. The other cells are what that
number excluded.

`language` is `research/language-capability`'s stage-B program, rebuilt from
`stage_b.json` exactly as `final_eval.py` rebuilds it (1.000 at lengths never
trained on). Its plain-Python reference is `common.balanced`, which is the
hand-written function the program was searched to match, so the two are
compared on identical inputs and their agreement is checked, not assumed.
"""
import sys, json, math, statistics
ROOT = '/home/brandonin/Documents/typed-crystallization-networks'
for p in (ROOT, ROOT + '/research/inference-cost', ROOT + '/research/language-capability'):
    sys.path.insert(0, p)
from harness import Counter, timed, dump, rss_mb, size_report


def language():
    import common as LC, scaffolds
    from run_stage_b import build_module
    from tcn.generation import text_value
    sel = json.load(open(ROOT + '/research/language-capability/stage_b.json'))['enumeration']['selections']
    module, registry, _ = build_module()
    prog, _ = scaffolds.stage_b(module, registry)
    frozen = prog.harden(sel).pruned()
    eps = LC.dataset(24, seed0=100000, split='test')
    eps = [e for e in eps if e['length'] not in (2, 4, 6)][:12]
    rows = []
    for e in eps:
        acquire_ms, _, _ = timed(lambda: LC.episode(e['seed'], 'test'), repeats=3)
        encode_ms, _, _ = timed(lambda: text_value(e['prompt'], LC.CAPACITY), repeats=11)
        with Counter(registry) as c:
            execute_ms, execute_min, out = timed(lambda: frozen.run({'text': e['text']}, registry=registry)[0], repeats=5)
        ops = c.n // 5
        decode_ms, _, answer = timed(lambda: out['answer'].decoded, repeats=11)
        py_ms, py_min, py_answer = timed(lambda: LC.balanced(e['string']), repeats=101)
        rows.append({'seed': e['seed'], 'length': e['length'], 'depth': e['depth'],
                     'acquire_ms': acquire_ms, 'encode_ms': encode_ms,
                     'execute_ms': execute_ms, 'execute_min_ms': execute_min,
                     'decode_ms': decode_ms,
                     'total_ms': acquire_ms + encode_ms + execute_ms + decode_ms,
                     'operator_applications': ops,
                     'us_per_operator': execute_ms * 1000 / max(1, ops),
                     'plain_python_ms': py_ms, 'plain_python_min_ms': py_min,
                     'agree': bool(answer) == bool(py_answer) == bool(e['label'])})
    def med(k): return statistics.median(r[k] for r in rows)
    return {'artifact': 'language-capability stage B (grammaticality from raw prompt bytes)',
            'size': size_report(frozen, registry),
            'search_space': {'stage_a_base': 41, 'stage_a_byte': 256, 'stage_b_symbols': 121,
                             'stage_b_plus': 5, 'stage_b_minus': 5, 'stage_b_rule': 15},
            'episodes': rows,
            'median': {k: med(k) for k in ('acquire_ms', 'encode_ms', 'execute_ms', 'decode_ms',
                                           'total_ms', 'plain_python_ms', 'operator_applications')},
            'all_agree': all(r['agree'] for r in rows), 'peak_rss_mb': rss_mb()}


def mixed():
    from tcn.runtime import load_program
    from tcn.types import Value
    prog, registry = load_program(ROOT + '/artifacts/demo/synthesis/program.json')
    types = dict(prog.inputs)
    cases = [(True, False, 0.13), (True, False, -0.43), (False, True, 0.3), (True, True, -0.2)]
    rows = []
    for a, b, x in cases:
        raw = {'a': a, 'b': b, 'x': x}
        encode_ms, _, ins = timed(lambda: {k: Value.of(types[k], raw[k]) for k in types}, repeats=101)
        with Counter(registry) as c:
            execute_ms, execute_min, out = timed(lambda: prog.run(ins, registry=registry)[0], repeats=101)
        ops = c.n // 101
        key = 'answer' if 'answer' in out else list(out)[0]
        decode_ms, _, value = timed(lambda: out[key].decoded, repeats=101)
        py_ms, py_min, py = timed(lambda: math.sin(float(a != b) + x), repeats=1001)
        rows.append({'inputs': [a, b, x], 'encode_ms': encode_ms,
                     'execute_ms': execute_ms, 'execute_min_ms': execute_min,
                     'decode_ms': decode_ms, 'total_ms': encode_ms + execute_ms + decode_ms,
                     'operator_applications': ops,
                     'us_per_operator': execute_ms * 1000 / max(1, ops),
                     'plain_python_ms': py_ms, 'plain_python_min_ms': py_min,
                     'output': value, 'python_output': py, 'abs_error': abs(value - py)})
    def med(k): return statistics.median(r[k] for r in rows)
    return {'artifact': 'mixed fixture (examples/mixed.py) -- the historical reference',
            'size': size_report(prog, registry),
            'search_space': {'enumerated_programs': 96},
            'episodes': rows,
            'median': {k: med(k) for k in ('encode_ms', 'execute_ms', 'decode_ms', 'total_ms',
                                           'plain_python_ms', 'operator_applications')},
            'max_abs_error_vs_python': max(r['abs_error'] for r in rows),
            'peak_rss_mb': rss_mb()}


if __name__ == '__main__':
    out = {}
    for key in (sys.argv[1:] or ['mixed', 'language']):
        out[key] = {'mixed': mixed, 'language': language}[key]()
        print(json.dumps(out[key].get('median'), indent=1))
    dump('inproc', out)
