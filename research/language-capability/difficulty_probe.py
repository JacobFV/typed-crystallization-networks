"""How far does the learned program reach past its declared capacity?

`Lesson.example` takes a `difficulty` that widens the nesting-depth span, and
`generators/language/generator.py` does not plumb it (proposed diff D3). This
probe reaches the same content through the lesson directly and renders it in the
lesson's own template, to measure where a 16-position scaffold stops. It is an
off-distribution diagnostic, not an accuracy on the shipped generator.
"""
from __future__ import annotations
import sys, os, json, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
import common, scaffolds
from run_stage_b import build_module
from generators.language.engine.registry import get

def main(reranked=False, outfile='difficulty_probe.json'):
    sel = dict(json.load(open(os.path.join(HERE, 'stage_b.json')))['enumeration']['selections'])
    if reranked:
        # the conforming member that a validation split of an unseen length keeps:
        # symbols = length - 101, steps -1/+1, rule eq(total, 0)
        sel.update({'symbols': 101, 'plus': 1, 'minus': 3, 'answer': 6})
    module, registry, _ = build_module()
    program, _ = scaffolds.stage_b(module, registry)
    lesson = get('context_free_language')
    per = collections.defaultdict(lambda: [0, 0])
    for s in range(600):
        ex = lesson.example(seed=s, difficulty=1.0)
        st = ex.metadata['hidden']['string']
        text, prompt = common.synthetic_text(st)
        out, _ = program.run({'text': text}, registry=registry, selections=sel)
        hit = bool(out['answer'].decoded) == (ex.answer == 'yes')
        per[len(st)][0] += hit; per[len(st)][1] += 1
    rows = {k: {'accuracy': v[0] / v[1], 'n': v[1]} for k, v in sorted(per.items())}
    report = {'note': 'difficulty=1.0 episodes rendered in the lesson template; '
                      'the scaffold declares 16 symbol positions',
              'reranked': reranked, 'selection': {k: sel[k] for k in ('symbols','plus','minus','answer')},
              'per_length': rows,
              'within_capacity': sum(v[0] for k, v in per.items() if k <= 16) /
                                 max(1, sum(v[1] for k, v in per.items() if k <= 16)),
              'beyond_capacity': sum(v[0] for k, v in per.items() if k > 16) /
                                 max(1, sum(v[1] for k, v in per.items() if k > 16))}
    json.dump(report, open(os.path.join(HERE, outfile), 'w'), indent=1)
    print(json.dumps(report, indent=1))

if __name__ == '__main__':
    main(reranked='--reranked' in sys.argv,
         outfile='difficulty_probe_reranked.json' if '--reranked' in sys.argv else 'difficulty_probe.json')
