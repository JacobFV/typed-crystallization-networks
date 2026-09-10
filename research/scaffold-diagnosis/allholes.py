"""B-allholes: the strongest no-diagnosis baseline -- swap **every** hole, keep the best.

Not pre-registered.  `B-sweeponly` was pinned to a *fixed* site (the
output-adjacent accumulator), which is itself a localisation choice; this control
deletes localisation entirely.  For each case it enumerates every scaffold
reachable by changing **one** hole to any core operator of matching signature:

  * language scaffolds -- the `acc_fold` hole and the `second_fold` hole, nine
    candidates each, so at most 17 distinct scaffolds beyond the case's own;
  * Boolean scaffolds -- every node in turn, over the seven core operators that
    resolve at `BOOL x BOOL -> BOOL`.

Each is fitted on the **training episodes only** and ranked by best training
accuracy, exactly as Stage S is.  If this beats both the probe and B-sweeponly,
brute force over holes is what finds repairs and the information table is not
contributing.  Reported as exploratory, per §46's discipline.
"""
from __future__ import annotations
import sys, os, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import cases as case_mod, probe, langfam, boolfam

OUT = os.path.join(HERE, 'out')
FOLDS = case_mod.FOLDS


def lang_allholes(case):
    opens = langfam.opens_cache(case['train'], case['positions'])
    rows = []
    variants = [(f, case['second_fold']) for f in FOLDS]
    if case['second_fold'] is not None:
        variants += [(case['acc_fold'], f) for f in FOLDS]
    else:
        # the §47 template: adding a second accumulator is one hole too
        variants += [(case['acc_fold'], f) for f in FOLDS]
    seen = set()
    for acc_fold, second_fold in variants:
        if (acc_fold, second_fold) in seen:
            continue
        seen.add((acc_fold, second_fold))
        r = langfam.sweep(case['train'], case['labels'], case['positions'], opens,
                          acc_fold, second_fold)
        hole = 'acc_fold' if acc_fold != case['acc_fold'] else (
            'second_fold' if second_fold != case['second_fold'] else 'unchanged')
        rows.append({'hole': hole, 'acc_fold': acc_fold, 'second_fold': second_fold,
                     'best_train_accuracy': r['best_train_accuracy'],
                     'conforming_on_train': r['conforming_on_train'],
                     'usable_members_on_train': r['usable_members_on_train'],
                     'best_member': r['best_member']})
    return rows


def bool_allholes(case):
    rows = []
    for nd in case['program'].nodes:
        for name in probe.core_candidates(case, nd.name):
            ext = boolfam.extend_site(case['program'], case['registry'], nd.name, name)
            if not ext:
                continue
            r = boolfam.sweep(case['program'], case['registry'], case['train'],
                              override={nd.name: ext})
            rows.append({'hole': nd.name, 'operator': name,
                         'best_train_accuracy': r['best_train_accuracy'],
                         'conforming_on_train': r['conforming_on_train'],
                         'space_size': r['space_size']})
    return rows


def main():
    t0 = time.perf_counter()
    out = {}
    for cid in case_mod.CASES:
        case = case_mod.CASES[cid]()
        rows = (lang_allholes(case) if case['adapter'] == 'lang'
                else bool_allholes(case))
        usable = [r for r in rows if r['best_train_accuracy'] is not None]
        best = max((r['best_train_accuracy'] for r in usable), default=None)
        winners = [r for r in usable if r['best_train_accuracy'] == best]
        solving = [r for r in usable if r['conforming_on_train']]
        out[cid] = {
            'n_scaffolds_tried': len(rows),
            'best_train_accuracy': best,
            'winners': [{k: r.get(k) for k in
                         ('hole', 'acc_fold', 'second_fold', 'operator',
                          'conforming_on_train')} for r in winners],
            'solving': [{k: r.get(k) for k in
                         ('hole', 'acc_fold', 'second_fold', 'operator',
                          'conforming_on_train')} for r in solving],
            'rows': rows}
        print(f"  {cid}: {len(rows)} scaffolds, best train {best}, "
              f"{len(solving)} conform", flush=True)
    out['seconds'] = time.perf_counter() - t0
    p = os.path.join(OUT, 'allholes.json')
    json.dump(out, open(p, 'w'), indent=1)
    print('wrote', p)


if __name__ == '__main__':
    main()
