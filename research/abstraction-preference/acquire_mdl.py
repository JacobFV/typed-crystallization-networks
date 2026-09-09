"""Does the description term make acquisition learn the *minimal* module?

Pruning removes dead nodes; it cannot remove a redundant live one. The learned
MAJ3 is a 5-gate live circuit where 4 gates suffice, and that surcharge is
charged at every future call site. This is the cleanest test of preference: the
same 5-node scaffold, the same seeds, the term on and off, run to the full budget
rather than stopping at the first conformant checkpoint, since a preference
applied after conformance cannot be seen by a stop-at-first criterion.
"""
import json, statistics, sys
import torch

sys.path.insert(0, '../recursive-abstraction-retest')
import common
import armlib
from tcn.operators import Registry

WEIGHTS = [0., 1e-6, 1e-5, 1e-4]
rows = []
for w in WEIGHTS:
    for seed in range(12):
        r = Registry()
        program, signals = common.sub_scaffold(r)
        ex = common.sub_examples(common.maj)
        rep = armlib.run(program, ex, signals, r, seed * 131, mdl_weight=w, steps=600,
                         eval_every=10, stop_on_success=False)
        rows.append({'weight': w, 'seed': seed, 'first_conformant_step': rep['first_conformant_step'],
                     'ever_conformant': rep['first_conformant_step'] is not None,
                     'final_conformant': rep['final_conformant'],
                     'found': rep['found'], 'final': rep['final'],
                     'final_task_loss': rep['final_task_loss']})
        print(f"w={w:g} seed={seed} first={rep['first_conformant_step']} "
              f"at-first live={rep['found']['live_nodes'] if rep['found'] else '-'} "
              f"final live={rep['final']['live_nodes']} cost={rep['final']['execution_cost']} "
              f"bits={rep['final']['description_bits']}", flush=True)

summary = {}
for w in WEIGHTS:
    got = [x for x in rows if x['weight'] == w]
    ok = [x for x in got if x['final_conformant']]
    # a run only counts as acquired if the program it ends on is still exact
    summary[f'{w:g}'] = {
        'acquired_ever': sum(1 for x in got if x['ever_conformant']), 'exact_at_end': len(ok), 'of': len(got),
        'median_first_step': statistics.median(x['first_conformant_step'] for x in ok) if ok else None,
        'median_live_at_first': statistics.median(x['found']['live_nodes'] for x in ok) if ok else None,
        'median_live_at_end': statistics.median(x['final']['live_nodes'] for x in ok) if ok else None,
        'minimal_at_end': sum(1 for x in ok if x['final']['live_nodes'] <= 4),
        'median_cost_at_end': statistics.median(x['final']['execution_cost'] for x in ok) if ok else None,
        'median_bits_at_end': statistics.median(x['final']['description_bits'] for x in ok) if ok else None,
    }
    print(w, summary[f'{w:g}'], flush=True)
json.dump({'rows': rows, 'summary': summary}, open('acquire_mdl.json', 'w'), indent=1)
