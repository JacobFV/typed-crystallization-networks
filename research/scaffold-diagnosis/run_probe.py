"""Run the probe over every case and commit its predictions -- BLIND.

This script imports `cases`, never `answers`.  It writes `out/predictions.json`,
which is committed to git **before** `score.py` joins it against the known
repairs.  Nothing in here can consult the answer, and the commit order is the
evidence for that.

For each case it records:

  * the identity checks (scaffold digests against §45's and §47's builders);
  * the simulator validation for the family it decides;
  * Stage D's full per-node information table and whether it fired;
  * Stage S at Stage D's site, in each declared repair family;
  * **B-sweeponly** -- Stage S at the fixed default site with Stage D deleted;
  * **B-terminal** -- the site "always blame the output-adjacent accumulator";
  * **B-random** -- the uniform-draw rate over the same candidate set.

Held-out accuracy of the predicted repair is computed and stored, but *after*
the prediction is fixed and never as an input to it.
"""
from __future__ import annotations
import sys, os, json, time, argparse
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import cases as case_mod, probe, langfam, boolfam, sim2, scaffolds2, common, prepare
from tcn.search import space_size

OUT = os.path.join(HERE, 'out')


def identity_checks():
    """Assert this track's builder reproduces §45's and §47's programs exactly."""
    import importlib.util, accum_scaffold
    module, registry = case_mod._lang_base()
    spec = importlib.util.spec_from_file_location(
        'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
    lc = importlib.util.module_from_spec(spec); spec.loader.exec_module(lc)
    out = {}
    for pos in (16, 22):
        a, _ = lc.stage_b(module, registry, positions=pos)
        b, _ = scaffolds2.stage_b_gen(module, registry, positions=pos,
                                      acc_fold='add', second_fold=None)
        out[f'counting_positions{pos}'] = {
            'language_capability_digest': a.digest, 'this_track_digest': b.digest,
            'identical': a.digest == b.digest, 'space_size': space_size(a)}
    a, _ = accum_scaffold.stage_b_accum(module, registry, positions=22, fold='min')
    b, _ = scaffolds2.stage_b_gen(module, registry, positions=22,
                                  acc_fold='add', second_fold='min')
    out['accum_min_positions22'] = {
        'dyck_learnability_digest': a.digest, 'this_track_digest': b.digest,
        'identical': a.digest == b.digest, 'space_size': space_size(a)}
    return out


def validations(case, folds_touched):
    """Validate the simulator against the real typed program, per variant."""
    if case['adapter'] != 'lang':
        return {'bool': boolfam.validate(case['program'], case['registry'],
                                         case['train'], n=150)}
    out = {}
    for acc_fold, second_fold in folds_touched:
        prog, registry, _ = case_mod.lang_program(acc_fold, second_fold, case['positions'])
        key = f'acc={acc_fold},second={second_fold}'
        out[key] = sim2.validate(prog, registry, case['train'], case['positions'],
                                 acc_fold, second_fold, n=40)
    return out


def heldout_of(case, acc_fold, second_fold, member):
    """Score a fitted member on the held-out splits.  After the prediction."""
    if member is None or not case['heldout']:
        return {}
    out = {}
    for name, (eps, labs) in case['heldout'].items():
        n = len(eps)
        yes = sum(labs)
        out[name] = {
            'accuracy': langfam.accuracy_on(eps, labs, case['positions'], member,
                                            acc_fold, second_fold),
            'n': n, 'majority_constant': max(yes, n - yes) / n, 'random': 0.5}
    return out


TOUCHED = set()


def _predict(bucket):
    best = None
    for fam, r in bucket.items():
        if r['best_train_accuracy'] is None:
            continue
        if best is None or r['best_train_accuracy'] > best[1]:
            best = (fam, r['best_train_accuracy'], r['tie_set'], r['solves_on_train'])
    if best is None:
        return {'repair_family': None, 'operators': [], 'best_train_accuracy': None,
                'solves_on_train': []}
    return {'repair_family': best[0], 'operators': best[2],
            'best_train_accuracy': best[1], 'solves_on_train': best[3]}


def run_case(cid, verbose=True):
    t0 = time.perf_counter()
    case = case_mod.CASES[cid]()
    d = probe.stage_d(case)
    families = (['swap', 'fold'] if case['adapter'] == 'lang' else ['swap'])
    labels = case['labels']
    rec = {'id': cid, 'adapter': case['adapter'], 'kind': case['kind'],
           'task': case['task'], 'stream': case['stream'],
           'positions': case.get('positions'), 'arm': case.get('arm'),
           'acc_fold': case.get('acc_fold'), 'second_fold': case.get('second_fold'),
           'n_train': len(case['train']),
           'train_positive_rate': sum(labels) / len(labels),
           'train_majority_constant': max(sum(labels), len(labels) - sum(labels)) / len(labels),
           'default_site': case['default_site'],
           'stage_d': d, 'stage_s': {}, 'b_sweeponly': {},
           'stage_s_input_dependent': {}, 'b_random': {},
           'b_terminal_site': case['default_site']}
    if case['adapter'] == 'lang':
        TOUCHED.add((case['positions'], case['acc_fold'], case['second_fold']))

    def do_sweep(site, bucket):
        if not site:
            return
        for fam in families:
            r = probe.stage_s(case, site, fam)
            bucket[fam] = r
            for row in r['rows']:
                if 'acc_fold' in row:
                    TOUCHED.add((case['positions'], row['acc_fold'], row['second_fold']))

    if d['fired'] and d.get('site'):
        do_sweep(d['site'], rec['stage_s'])
    idv = d.get('input_dependent_variant', {})
    if idv.get('fired') and idv.get('site'):
        if idv['site'] == d.get('site'):
            rec['stage_s_input_dependent'] = rec['stage_s']
        else:
            do_sweep(idv['site'], rec['stage_s_input_dependent'])
    if case['default_site'] == d.get('site'):
        rec['b_sweeponly'] = rec['stage_s']
    else:
        do_sweep(case['default_site'], rec['b_sweeponly'])

    for tag, bucket in (('probe', rec['stage_s']),
                        ('input_dependent', rec['stage_s_input_dependent']),
                        ('sweeponly', rec['b_sweeponly'])):
        for fam, r in bucket.items():
            rec['b_random'][f'{tag}:{fam}'] = probe.b_random(case, r['site'], fam, r)

    # ---- predictions, fixed here ------------------------------------------
    rec['prediction'] = dict(_predict(rec['stage_s']), fired=d['fired'],
                             site=d.get('site'))
    rec['prediction_input_dependent'] = dict(
        _predict(rec['stage_s_input_dependent']),
        fired=idv.get('fired'), site=idv.get('site'))
    rec['prediction_sweeponly'] = dict(_predict(rec['b_sweeponly']),
                                       site=case['default_site'])

    # ---- after the prediction: held-out scoring ---------------------------
    rec['heldout_of_prediction'] = {}
    p = rec['prediction']
    if case['adapter'] == 'lang' and p['repair_family']:
        r = rec['stage_s'][p['repair_family']]
        for row in r['rows']:
            if row.get('operator') in p['operators'] and 'acc_fold' in row:
                rec['heldout_of_prediction'][row['operator']] = heldout_of(
                    case, row['acc_fold'], row['second_fold'], row['best_member'])
    ps = rec['prediction_sweeponly']
    rec['heldout_of_sweeponly'] = {}
    if case['adapter'] == 'lang' and ps['repair_family']:
        r = rec['b_sweeponly'][ps['repair_family']]
        for row in r['rows']:
            if row.get('operator') in ps['operators'] and 'acc_fold' in row:
                rec['heldout_of_sweeponly'][row['operator']] = heldout_of(
                    case, row['acc_fold'], row['second_fold'], row['best_member'])
    if case['adapter'] != 'lang':
        rec['bool_validation'] = boolfam.validate(case['program'], case['registry'],
                                                  case['train'], n=150)
    rec['seconds'] = time.perf_counter() - t0
    if verbose:
        print(f"  {cid}: fired={d['fired']} site={d.get('site')} rule={d.get('rule')}"
              f" | probe -> {p['repair_family']} {p['operators']}"
              f" train {p['best_train_accuracy']}"
              f" | sweeponly@{case['default_site']} -> {ps['repair_family']}"
              f" {ps['operators']} train {ps['best_train_accuracy']}"
              f" [{rec['seconds']:.0f}s]", flush=True)
    return rec


def all_validations():
    out = {}
    for positions, acc_fold, second_fold in sorted(TOUCHED, key=lambda t: (t[0], t[1], t[2] or '')):
        prog, registry, _ = case_mod.lang_program(acc_fold, second_fold, positions)
        eps = (prepare.load()['train'] if positions == case_mod.POSITIONS
               else case_mod._preaudit_split()['train'])
        out[f'positions={positions},acc={acc_fold},second={second_fold}'] = sim2.validate(
            prog, registry, eps, positions, acc_fold, second_fold, n=40)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None)
    ap.add_argument('--out', default='predictions.json')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t0 = time.perf_counter()
    ident = identity_checks()
    print(json.dumps(ident, indent=1), flush=True)
    ids = ([a.only] if a.only else list(case_mod.CASES))
    recs = []
    for cid in ids:
        recs.append(run_case(cid))
    vals = all_validations()
    print('simulator validation mismatches:',
          {k: v['mismatches'] for k, v in vals.items()}, flush=True)
    rep = {'identity_checks': ident, 'aliases': case_mod.ALIASES,
           'simulator_validation': vals,
           'probe': {'mi_zero': probe.MI_ZERO, 'h_label_min': probe.H_LABEL_MIN,
                     'stages': 'D = per-node information on the training batch; '
                               'S = swap sweep over core operators at the flagged site'},
           'streams_used': sorted({r['stream'] for r in recs}),
           'seconds': time.perf_counter() - t0, 'cases': recs}
    path = os.path.join(OUT, a.out)
    json.dump(rep, open(path, 'w'), indent=1)
    print('wrote', path, f"{rep['seconds']:.0f}s")


if __name__ == '__main__':
    main()
