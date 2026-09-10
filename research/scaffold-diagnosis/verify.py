"""Post-hoc verification of the two headlines that could be less than they look.

1. The one case where the exploratory B-allholes control is **wrong**
   (`C_max2_add`, where it proposes `sub` at the `acc_fold` hole alongside the
   known `min`/`max`): score that proposal on held-out data, to show that
   "conforms on 24 training episodes" is not the same as "repairs the scaffold".

2. The §47 headline itself, re-confirmed on the **real typed program** rather
   than on the simulator: the first-in-order conforming member of the `fold=min`
   repair of `R1`, executed by `Program.execute` on all three splits.
"""
from __future__ import annotations
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import cases as case_mod, langfam, sim2
from run_stage_b import examples, accuracy, baselines
from tcn.search import evaluate

OUT = os.path.join(HERE, 'out')


def overfit_check():
    case = case_mod.CASES['C_max2_add']()
    opens = langfam.opens_cache(case['train'], case['positions'])
    out = {}
    for acc_fold, second_fold in (('sub', 'add'), ('add', 'max'), ('add', 'min')):
        r = langfam.sweep(case['train'], case['labels'], case['positions'], opens,
                          acc_fold, second_fold, collect_conforming=True)
        if not r['conforming_on_train']:
            out[f'acc={acc_fold},second={second_fold}'] = {
                'conforming_on_train': 0,
                'best_train_accuracy': r['best_train_accuracy']}
            continue
        per = {}
        for name, (eps, labs) in case['heldout'].items():
            accs = [langfam.accuracy_on(eps, labs, case['positions'], m,
                                        acc_fold, second_fold)
                    for m in r['conforming_members']]
            n = len(eps); yes = sum(labs)
            per[name] = {'n': n, 'majority_constant': max(yes, n - yes) / n,
                         'random': 0.5, 'min': min(accs), 'max': max(accs),
                         'mean': sum(accs) / len(accs),
                         'enumeration_order_first': accs[0]}
        out[f'acc={acc_fold},second={second_fold}'] = {
            'conforming_on_train': r['conforming_on_train'],
            'best_train_accuracy': r['best_train_accuracy'], 'heldout': per}
    return out


def real_program_check():
    """The `fold=min` repair of R1, on the REAL typed program."""
    case = case_mod.CASES['C_bal_min']()
    prog, registry, signals = case['program'], case['registry'], case['signals']
    opens = langfam.opens_cache(case['train'], case['positions'])
    r = langfam.sweep(case['train'], case['labels'], case['positions'], opens,
                      'add', 'min', collect_conforming=True)
    first = min(r['conforming_members'], key=lambda m: langfam.global_index(m, True))
    sel = langfam.selections(prog, first)
    err = evaluate(prog, sel, examples(case['train']), signals, registry,
                   tolerance=1e-6)
    out = {'conforming_on_train': r['conforming_on_train'],
           'first_in_order': langfam._readable(first, True),
           'global_enumeration_index': langfam.global_index(first, True),
           'space_size': r['space_size'],
           'fraction_through_enumeration':
               langfam.global_index(first, True) / r['space_size'],
           'train_max_error': err,
           'conforms_on_train': err is not None and err <= 1e-6}
    s = {'train': case['train']}
    for name, (eps, labs) in case['heldout'].items():
        s[name] = eps
    for name, eps in s.items():
        acc, per_len = accuracy(prog, sel, registry, eps)
        out[name] = {'accuracy': acc, 'per_length': per_len, **baselines(eps)}
    return out


if __name__ == '__main__':
    rep = {'allholes_overfit_case_C_max2_add': overfit_check(),
           'real_program_confirmation_of_the_fold_min_repair': real_program_check()}
    json.dump(rep, open(os.path.join(OUT, 'verify.json'), 'w'), indent=1)
    print(json.dumps(rep, indent=1))
