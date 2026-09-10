"""Join the committed predictions against the known repairs.  The only file that
imports `answers.py`.

Scoring follows the pre-registration:

  * Stage D **fires** or does not.  A fire on a scaffold that contains a
    solution is a **false positive** (criterion F2).
  * Stage S's **prediction is correct** iff its tie set is *contained in* the
    known repair set (criterion, step 10).  A tie set that spans candidates
    which do not repair the scaffold is not a prediction, it is a shrug.
  * A prediction is **declined** when no candidate reaches training accuracy
    1.000 -- a visible failure rather than a confident wrong answer.  Declines
    are counted separately from wrong answers because they cost different
    things downstream.
  * **B-sweeponly** (the sweep with the diagnosis deleted) and **B-random**
    (uniform draw over the same candidate set) are scored identically, so
    criterion F3 is a comparison of three numbers rather than an argument.

Every number is derived from `out/predictions.json`, never recomputed here.
"""
from __future__ import annotations
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from answers import TRUTH

OUT = os.path.join(HERE, 'out')


def score_prediction(pred, truth):
    """(correct, declined, note) for one prediction against one known repair."""
    ops = pred.get('operators') or []
    solves = pred.get('solves_on_train') or []
    repair = set(truth['repair'])
    if truth['contains_solution']:
        # a solvable scaffold needs no repair; the honest outcome is silence
        return None, None, 'solvable scaffold: no repair is the right answer'
    if not ops:
        return False, True, 'no candidate proposed'
    if not solves:
        return False, True, 'no candidate reaches train 1.000'
    return set(ops) <= repair and bool(repair), False, ''


def main():
    rep = json.load(open(os.path.join(OUT, 'predictions.json')))
    rows = []
    for c in rep['cases']:
        t = TRUTH[c['id']]
        d = c['stage_d']
        idv = d.get('input_dependent_variant', {})
        r = {'id': c['id'], 'adapter': c['adapter'], 'task': c['task'],
             'stream': c['stream'], 'acc_fold': c['acc_fold'],
             'second_fold': c['second_fold'],
             'contains_solution': t['contains_solution'],
             'known_repair': t['repair'], 'repair_family_truth': t['repair_family'],
             'defect_kind': t['defect_kind'],
             'n_nodes': d.get('n_nodes'), 'n_fired_nodes': d.get('n_fired_nodes'),
             'd_fired': d['fired'], 'd_site': d.get('site'),
             'd_rule': d.get('rule'), 'd_site_distinct': d.get('site_distinct'),
             'd_fired_input_dependent': idv.get('fired'),
             'd_site_input_dependent': idv.get('site'),
             'terminal_site': c['b_terminal_site'],
             'site_equals_terminal': d.get('site') == c['b_terminal_site'],
             'best_train_accuracy_of_failed_scaffold':
                 (d.get('best_member') or {}).get('best_train_accuracy'),
             'train_majority_constant': c['train_majority_constant']}
        for tag, key in (('probe', 'prediction'),
                         ('input_dependent', 'prediction_input_dependent'),
                         ('sweeponly', 'prediction_sweeponly')):
            p = c[key]
            ok, dec, note = score_prediction(p, t)
            r[f'{tag}_family'] = p.get('repair_family')
            r[f'{tag}_operators'] = p.get('operators')
            r[f'{tag}_train'] = p.get('best_train_accuracy')
            r[f'{tag}_solves'] = p.get('solves_on_train')
            r[f'{tag}_correct'] = ok
            r[f'{tag}_declined'] = dec
            r[f'{tag}_note'] = note
        # B-random over the same candidate set the probe swept
        br = {}
        for k, v in c['b_random'].items():
            br[k] = {'n_candidates': v['n_candidates'],
                     'n_that_solve_on_train': v['n_that_solve_on_train'],
                     'p_uniform_draw_repairs': v['p_correct_uniform_draw']}
        r['b_random'] = br
        r['heldout_of_prediction'] = c.get('heldout_of_prediction', {})
        r['heldout_of_sweeponly'] = c.get('heldout_of_sweeponly', {})
        rows.append(r)

    failed = [r for r in rows if not r['contains_solution']]
    solvable = [r for r in rows if r['contains_solution']]
    tasks = sorted({(r['adapter'], r['task']) for r in rows})

    def tally(tag):
        att = [r for r in failed if r[f'{tag}_correct'] is not None]
        return {'failed_cases': len(att),
                'correct': sum(1 for r in att if r[f'{tag}_correct']),
                'declined': sum(1 for r in att if r[f'{tag}_declined']),
                'wrong_not_declined': sum(1 for r in att
                                          if r[f'{tag}_correct'] is False
                                          and not r[f'{tag}_declined']),
                'fired_on_solvable_scaffolds': sum(
                    1 for r in solvable
                    if (r['d_fired'] if tag != 'sweeponly' else bool(r['sweeponly_operators']))),
                'solvable_cases': len(solvable)}

    # F5: does the information table pick a different site from "blame the terminal node"?
    same_site = sum(1 for r in rows if r['site_equals_terminal'])
    summary = {
        'n_cases': len(rows), 'n_failed_scaffolds': len(failed),
        'n_solvable_controls': len(solvable),
        'independent_tasks': [f'{a}:{t}' for a, t in tasks],
        'n_independent_tasks': len(tasks),
        'probe': tally('probe'),
        'input_dependent_variant': tally('input_dependent'),
        'sweeponly': tally('sweeponly'),
        'F2_stage_d_fired_on_solvable_controls':
            [r['id'] for r in solvable if r['d_fired']],
        'F2_stage_d_input_dependent_fired_on_solvable_controls':
            [r['id'] for r in solvable if r['d_fired_input_dependent']],
        'F5_site_equals_terminal_node': f'{same_site}/{len(rows)}',
        'F1_sites_that_are_constant_nodes':
            sum(1 for r in rows if r['d_site_distinct'] == 1),
        'F1_failed_cases_where_d_did_not_fire':
            [r['id'] for r in failed if not r['d_fired']],
    }
    out = {'summary': summary, 'rows': rows}
    json.dump(out, open(os.path.join(OUT, 'scored.json'), 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    hdr = ('case', 'solvable', 'known', 'D fired', 'D site', 'probe',
           'sweeponly', 'probe ok')
    print('\n' + ' | '.join(hdr))
    for r in rows:
        print(' | '.join(str(x) for x in (
            r['id'], r['contains_solution'], ','.join(r['known_repair']) or '-',
            r['d_fired'], r['d_site'],
            f"{r['probe_family']}:{','.join(r['probe_operators'])}",
            f"{r['sweeponly_family']}:{','.join(r['sweeponly_operators'])}"
            f"@{r['sweeponly_train']}",
            r['probe_correct'])))


if __name__ == '__main__':
    main()
