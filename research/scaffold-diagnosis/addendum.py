"""Addendum arms A1-A4, declared in `PREREGISTRATION.md` before this file ran.

A1/A2  `--c2`      establish family C2's repair set by exhaustive enumeration
                   over the `acc_fold` hole, and write it to `out/c2_answer.json`
                   BEFORE `score.py` reads it as the declared answer.
A3     `--member`  run the identical Stage D at §47's own member (§45's honest
                   counting program, `c=101, plus=+1, minus=-1`) and report
                   whether the recorded `acc21` constancy reproduces.
A4     `--spread`  held-out accuracy spread across the whole conforming set of
                   every fold that conforms on training, so that "conforms on 24
                   training episodes" is never reported as "solves the task".

Every number names its stream; `hardening` is pinned at every draw upstream in
`prepare.py` / `cases.py`.
"""
from __future__ import annotations
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import cases as case_mod, probe, langfam, sim2, common

OUT = os.path.join(HERE, 'out')
FOLDS = case_mod.FOLDS


def _case(cid):
    return case_mod.CASES[cid]()


def c2_answer():
    """Exhaustive sweep over the acc hole with `second_fold='min'`, task `bal`."""
    case = _case('C2_bal_add')
    opens = langfam.opens_cache(case['train'], case['positions'])
    rows = []
    for f in FOLDS:
        r = langfam.sweep(case['train'], case['labels'], case['positions'], opens,
                          f, 'min')
        rows.append({k: r[k] for k in
                     ('acc_fold', 'second_fold', 'space_size', 'exhausted',
                      'certificate', 'usable_members_on_train',
                      'best_train_accuracy', 'members_tied_at_best_train',
                      'conforming_on_train', 'best_member_readable')})
    solving = [r['acc_fold'] for r in rows if r['conforming_on_train']]
    rep = {'family': 'C2', 'task': 'bal', 'stream': case['stream'],
           'positions': case['positions'], 'second_fold': 'min',
           'hole': 'acc_fold', 'candidates': list(FOLDS),
           'folds_that_conform_on_train': solving,
           'train_majority_constant': case['train_majority_constant']
           if 'train_majority_constant' in case else None,
           'rows': rows}
    json.dump(rep, open(os.path.join(OUT, 'c2_answer.json'), 'w'), indent=1)
    print('C2 folds that conform on training:', solving)
    for r in rows:
        print(f"  acc={r['acc_fold']:5s} best train {r['best_train_accuracy']} "
              f"conforming {r['conforming_on_train']:6d} "
              f"usable {r['usable_members_on_train']}")
    return rep


def member_arm():
    """A3: Stage D at §47's member instead of the best-training member."""
    out = {}
    for cid, c, plus, minus in (('R1_counting_postaudit', 101, 1, -1),
                                ('C_bal_add', 101, 1, -1),
                                ('C_bal_min', 101, 1, -1)):
        case = _case(cid)
        ci = sim2.SUB.index(c); pi = sim2.STEPS.index(plus); mi = sim2.STEPS.index(minus)
        two = case['second_fold'] is not None
        # the readout is whatever the grid's first entry is; Stage D reads node
        # values, which do not depend on the readout choice
        member = (ci, pi, mi, 0, 0) if two else (ci, pi, mi, 0)
        sel = langfam.selections(case['program'], member)
        d = probe.stage_d(case, member_selections=sel,
                          member_note=f'section 47 member c={c}, plus={plus}, minus={minus}')
        rows = {r['node']: r for r in d['nodes']}
        acc = f"acc{case['positions'] - 1}"
        out[cid] = {
            'member': {'c': c, 'plus': plus, 'minus': minus},
            'terminal_accumulator': acc,
            'distinct': rows[acc]['distinct'],
            'distinct_values': rows[acc]['distinct_values'],
            'information_bits': rows[acc]['information_bits'],
            'label_entropy_bits': d['label_entropy_bits'],
            'fired': d['fired'], 'site': d['site'], 'rule': d['rule'],
            'n_fired_nodes': d['n_fired_nodes'],
            'acc_chain_information': {
                f'acc{i}': rows[f'acc{i}']['information_bits']
                for i in range(1, case['positions'])},
            'input_dependent_variant': d['input_dependent_variant']}
        print(cid, 'at section 47 member:', acc, 'distinct',
              rows[acc]['distinct'], 'bits', rows[acc]['information_bits'],
              '| site', d['site'], d['rule'])
    return out


def spreads():
    """A4: held-out spread over every conforming member of every solving fold."""
    out = {}
    specs = [('bal', 'C_bal_add', [('add', f) for f in FOLDS]),
             ('max2', 'C_max2_add', [('add', f) for f in FOLDS]),
             ('C2_bal', 'C2_bal_add', [(f, 'min') for f in FOLDS])]
    for tag, cid, variants in specs:
        case = _case(cid)
        opens = langfam.opens_cache(case['train'], case['positions'])
        rows = []
        for acc_fold, second_fold in variants:
            r = langfam.sweep(case['train'], case['labels'], case['positions'],
                              opens, acc_fold, second_fold, collect_conforming=True)
            if not r['conforming_on_train']:
                rows.append({'acc_fold': acc_fold, 'second_fold': second_fold,
                             'conforming_on_train': 0,
                             'best_train_accuracy': r['best_train_accuracy']})
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
                             'at_1.000': sum(1 for a in accs if a == 1.0),
                             'enumeration_order_first': accs[0]}
            rows.append({'acc_fold': acc_fold, 'second_fold': second_fold,
                         'conforming_on_train': r['conforming_on_train'],
                         'best_train_accuracy': r['best_train_accuracy'],
                         'heldout': per})
            u = per['heldout_unseen_lengths']
            print(f"  {tag} acc={acc_fold} second={second_fold}: "
                  f"{r['conforming_on_train']} conforming, unseen spread "
                  f"{u['min']:.4f}-{u['max']:.4f} (first in order {u['enumeration_order_first']:.4f}, "
                  f"majority {u['majority_constant']:.4f})")
        out[tag] = {'stream': case['stream'], 'positions': case['positions'],
                    'rows': rows}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--c2', action='store_true')
    ap.add_argument('--member', action='store_true')
    ap.add_argument('--spread', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    if a.c2:
        c2_answer()
    rep = {}
    if a.member:
        rep['A3_section47_member'] = member_arm()
    if a.spread:
        rep['A4_heldout_spread'] = spreads()
    if rep:
        rep['seconds'] = time.perf_counter() - t0
        p = os.path.join(OUT, 'addendum.json')
        if os.path.exists(p):
            old = json.load(open(p)); old.update(rep); rep = old
        json.dump(rep, open(p, 'w'), indent=1)
        print('wrote', p)


if __name__ == '__main__':
    main()
