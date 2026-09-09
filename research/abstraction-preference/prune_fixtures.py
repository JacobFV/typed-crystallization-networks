"""R3 semantic preservation: pruning must not change what any fixture computes.

Every shipped fixture and every scaffold this track uses is hardened at many
selections, pruned, and executed side by side with the unpruned program on every
example. Any single disagreement, on any row, fails the check.
"""
import itertools, json, sys
import torch

sys.path.insert(0, '../recursive-abstraction-retest')
import common
from examples.mixed import problem as mixed_problem
from tcn.learning import SoftProgram
from tcn.operators import Registry
from tcn.types import BOOL, Value
from tcn.curriculum import Curriculum
from tcn.runtime import save_program, load_program


def compare(program, examples, registry, label, trials=25):
    rows = 0
    for t in range(trials):
        torch.manual_seed(t)
        m = SoftProgram(program, registry)
        with torch.no_grad():
            for p in m.choices: p.add_(torch.randn_like(p) * 6.)
        hard = m.export()
        lean = hard.pruned().validate(registry)
        for ex in examples:
            try:
                a, _, _ = hard.execute(ex['inputs'], registry=registry)
            except Exception as exc:                       # an illegal numeric domain
                try:
                    lean.execute(ex['inputs'], registry=registry)
                except Exception: continue
                return {'label': label, 'ok': False, 'why': f'unpruned raised {exc!r}, pruned did not'}
            b, _, _ = lean.execute(ex['inputs'], registry=registry)
            for k in a:
                if a[k].flat() != b[k].flat():
                    return {'label': label, 'ok': False, 'why': f'output {k} differs on trial {t}'}
            rows += 1
    return {'label': label, 'ok': True, 'rows_compared': rows,
            'dropped_nodes': len(program.nodes) - len(SoftProgram(program, registry).export().pruned().nodes)}


out = []
p, signals, examples = mixed_problem()
out.append(compare(p, examples, Registry(), 'examples/mixed'))

r = Registry()
name = r.register_module(common.minimal_module(r, 'maj'))
comp = common.composite_examples()
out.append(compare(common.wide_scaffold(r), comp, r, 'retest wide scaffold, arm A'))
out.append(compare(common.wide_scaffold(r, name), comp, r, 'retest wide scaffold, arm B'))
out.append(compare(common.tight_scaffold(r, name), comp, r, 'retest tight scaffold, arm B'))
r2 = Registry()
sub, _ = common.sub_scaffold(r2)
out.append(compare(sub, common.sub_examples(common.maj), r2, 'retest sub scaffold'))

# the curriculum fixtures, as the shipped pipeline builds them
from examples.joint import trainer as joint_trainer
from tcn.types import product
from tcn.scaffold import F
jt = joint_trainer(episodes=1)
jp = jt.model.program
view = jt.host.view() if hasattr(jt, 'host') else None
rows = []
for bits in ((False, False, False, False), (False, True, True, False), (True, True, False, True)):
    inputs = {'bits': Value.of(dict(jp.inputs)['bits'], bits),
              'goal': Value.of(dict(jp.inputs)['goal'], False),
              'action': Value.of(product(F, F), (0., 0.)), 'dt': Value.of(F, 1.)}
    rows.append({'inputs': inputs})
out.append(compare(jp, rows, Registry(), 'examples/joint'))

for row in out: print(row)
print('all preserved:', all(x['ok'] for x in out if x['ok'] is not None))
json.dump(out, open('prune_fixtures.json', 'w'), indent=1)
