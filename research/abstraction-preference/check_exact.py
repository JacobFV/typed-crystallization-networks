"""Does description_cost() equal description_bits() of the pruned hardened export at one-hot?

The surrogate is only defensible if it is a relaxation of a real quantity. The
claim is exact identity at every one-hot point, checked here on four scaffolds
including one with module candidates and one without.
"""
import sys, json, torch
sys.path.insert(0, '../recursive-abstraction-retest')
import common
from tcn.operators import Registry
from tcn.learning import SoftProgram


def check(program, registry, label, trials=6):
    rows = []
    for t in range(trials):
        torch.manual_seed(t)
        m = SoftProgram(program, registry)
        with torch.no_grad():
            for p in m.choices:
                p.add_(torch.randn_like(p) * 8.0)
                p.mul_(50.)                      # effectively one-hot
        surrogate = float(m.description_cost())
        truth = m.export().pruned().description_bits(registry)
        rows.append({'trial': t, 'surrogate': surrogate, 'truth': truth,
                     'delta': surrogate - truth,
                     'live_nodes': len(m.export().pruned().nodes),
                     'scaffold_nodes': len(program.nodes)})
    return {'label': label, 'rows': rows, 'max_abs_delta': max(abs(r['delta']) for r in rows)}


out = []
r = Registry()
out.append(check(common.wide_scaffold(r), r, 'wide arm A (no module)'))
r2 = Registry(); name = r2.register_module(common.minimal_module(r2, 'maj'))
out.append(check(common.tight_scaffold(r2, name), r2, 'tight arm B (module)'))
out.append(check(common.wide_scaffold(r2, name), r2, 'wide arm B (module)'))
r3 = Registry(); prog, _ = common.sub_scaffold(r3)
out.append(check(prog, r3, 'sub scaffold (5 gates, no module)'))

for o in out:
    print(o['label'], 'max |delta| =', o['max_abs_delta'],
          'live/scaffold =', [(x['live_nodes'], x['scaffold_nodes']) for x in o['rows']])
json.dump(out, open('exactness.json', 'w'), indent=1)
