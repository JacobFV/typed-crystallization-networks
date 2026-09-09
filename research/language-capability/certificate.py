"""What did the exported program actually learn: Dyck membership, or a count?

The lesson's docstring calls balanced brackets "the canonical test that a learner
has stack-like state". Its negatives are made by flipping one character of a
balanced string, which always breaks the count, so on the sampled distribution
`#( == #)` and `balanced` agree exactly. This interrogates the exported program
on strings the generator cannot emit, in the lesson's own template, to say which
of the two predicates it implements. These are diagnostic, never a reported
accuracy on the lesson.
"""
from __future__ import annotations
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import common, scaffolds
from run_stage_b import build_module
from tcn.search import candidate_counts

PROBE = ([')(', ')()(', '())(', '))((', ')(()', '()))((', ')))(((', '())()('],   # counts equal, not Dyck
         ['()', '()()', '(())', '((()))', '()(())', '(()())'])                    # Dyck

def main():
    sel = json.load(open(os.path.join(HERE, 'stage_b.json')))['enumeration']['selections']
    module, registry, _ = build_module()
    program, _ = scaffolds.stage_b(module, registry)
    rows = []
    for group, strings in (('counts_equal_not_dyck', PROBE[0]), ('dyck', PROBE[1])):
        for s in strings:
            text, prompt = common.synthetic_text(s)
            out, _ = program.run({'text': text}, registry=registry, selections=sel)
            rows.append({'group': group, 'string': s, 'program_says_balanced': bool(out['answer'].decoded),
                         'dyck': common.balanced(s), 'counts_match': common.counts_match(s)})
    agree_dyck = sum(r['program_says_balanced'] == r['dyck'] for r in rows) / len(rows)
    agree_count = sum(r['program_says_balanced'] == r['counts_match'] for r in rows) / len(rows)
    report = {'selection': sel, 'rows': rows,
              'agreement_with_dyck': agree_dyck, 'agreement_with_counting': agree_count}
    json.dump(report, open(os.path.join(HERE, 'certificate.json'), 'w'), indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != 'selection'}, indent=1))

if __name__ == '__main__':
    main()
