"""Q2: put the min-prefix arm beside its counting-only control, arm by arm.

The comparison is the point. Section 45 proves a solution exists in the
min-prefix family and proves none exists in the counting-only family, so a
gradient path that conforms at the same rate on both is fitting noise, not
finding structure. Every row carries the majority constant (0.5262 on the unseen
split) and random (0.5) beside its accuracy, and the uniform-random-selection
null from addendum A2 is reported beneath.
"""
from __future__ import annotations
import sys, os, json, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)


def main():
    arms = []
    for path in sorted(glob.glob(os.path.join(HERE, 'out', 'q2_*_tau*_s*.json'))):
        d = json.load(open(path))
        row = {'scaffold': d['scaffold'], 'solution_exists_in_family': d['solution_exists_in_family'],
               'space_size': d['space_size'], 'tau_lt': d['tau_lt'], 'steps': d['steps'],
               'seeds': d['seeds'], 'conforming_seeds': d['conforming_seeds'],
               'conforming_rate': d['conforming_rate'],
               'first_conforming_steps': d['first_conforming_steps'],
               'mean_heldout_unseen': d['mean_heldout_unseen'],
               'median_heldout_unseen': d['median_heldout_unseen'],
               'max_heldout_unseen': d['max_heldout_unseen'],
               'min_heldout_unseen': d['min_heldout_unseen'],
               'majority_constant': d['baselines_heldout_unseen']['majority_constant'],
               'random': 0.5,
               'min_ok_status_counts': d.get('min_ok_status_counts', {}),
               'first_step_choice_gradients_seed0': d['runs'][0]['first_step_choice_gradients'],
               'chosen_per_seed': [r['chosen'] for r in d['runs']],
               'seconds_total': sum(r['seconds'] for r in d['runs'])}
        arms.append(row)
    order = {'dyck': 0, 'counting': 1}
    arms.sort(key=lambda r: (r['steps'], r['tau_lt'], order[r['scaffold']]))

    pairs = []
    for steps in sorted({a['steps'] for a in arms}):
        for tau in sorted({a['tau_lt'] for a in arms}):
            g = {a['scaffold']: a for a in arms if a['steps'] == steps and a['tau_lt'] == tau}
            if len(g) == 2:
                pairs.append({
                    'steps': steps, 'tau_lt': tau,
                    'dyck_conforming': g['dyck']['conforming_seeds'],
                    'counting_conforming': g['counting']['conforming_seeds'],
                    'dyck_beats_its_control': g['dyck']['conforming_seeds'] > g['counting']['conforming_seeds'],
                    'dyck_median_unseen': g['dyck']['median_heldout_unseen'],
                    'counting_median_unseen': g['counting']['median_heldout_unseen'],
                    'majority_constant': g['dyck']['majority_constant'],
                    'dyck_min_ok_status': g['dyck']['min_ok_status_counts']})

    rep = {'question': 'Q2 -- does the gradient path find the running minimum, and does it '
                       'beat its own control?',
           'arms': arms, 'paired_comparisons': pairs,
           'any_arm_conforms_on_dyck': any(a['conforming_seeds'] > 0 for a in arms
                                           if a['scaffold'] == 'dyck'),
           'any_arm_conforms_on_counting': any(a['conforming_seeds'] > 0 for a in arms
                                               if a['scaffold'] == 'counting'),
           'dyck_ever_beats_control': any(p['dyck_beats_its_control'] for p in pairs),
           'min_ok_status_over_all_dyck_arms': dict(sum(
               (collections.Counter(a['min_ok_status_counts']) for a in arms
                if a['scaffold'] == 'dyck'), collections.Counter()))}
    null_path = os.path.join(HERE, 'out', 'q2_null.json')
    if os.path.exists(null_path):
        rep['random_selection_null'] = json.load(open(null_path))['rows']
    out = os.path.join(HERE, 'out', 'q2_summary.json')
    json.dump(rep, open(out, 'w'), indent=1)
    for a in arms:
        print(f"{a['scaffold']:9s} tau_lt={a['tau_lt']:5.0f} steps={a['steps']:5d} "
              f"conforming {a['conforming_seeds']}/{a['seeds']} "
              f"median unseen {a['median_heldout_unseen']:.4f} "
              f"(majority {a['majority_constant']:.4f}) min_ok {a['min_ok_status_counts']}")
    print()
    print(json.dumps({k: v for k, v in rep.items()
                      if k not in ('arms', 'random_selection_null')}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
