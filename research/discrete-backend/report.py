"""Regenerate every table in RESULTS.md from the raw JSON in `out/`.

Run after the three experiment scripts. Nothing in RESULTS.md is transcribed by
hand; if a number there is not printed by this file it is not a measurement.
"""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).parent / 'out'


def load(name):
    return json.loads((OUT / name).read_text())


def recurrent():
    d = load('recurrent_depth.json')
    print('## recurrent (research/discrete-backend/recurrent_depth.py)')
    print(f"horizon {d['horizon']} settle {d['settle_window']} "
          f"train depths {d['train_depths']} eval depths {d['eval_depths']}")
    print('| arm | space | free nodes | train episodes | fit conforming | fit s | '
          'recurrent conforming | certificate | recurrent s | node evaluations |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for a in d['arms']:
        f, r = a['enumerate_fit'], a['enumerate_recurrent']
        free = ' '.join(f"{k}:{v}" for k, v in a['free_nodes'].items())
        print(f"| {a['arm']} | {a['space_size']} | {free} | {a['train_episodes']} | "
              f"{f['conforming']} | {f['seconds']:.2f} | {r['conforming']} | {r['certificate']} | "
              f"{r['seconds']:.2f} | {r['node_evaluations']:,} |")
    print()
    print('| arm | selection | depth | episodes | probe max error | mean settled return | exact |')
    print('|---|---|---|---|---|---|---|')
    for a in d['arms']:
        if not a.get('holdout'): continue
        sel = ' '.join(f"{k}={v}" for k, v in a['selection_free'].items())
        for depth, h in a['holdout'].items():
            print(f"| {a['arm']} | {sel} | {depth} | {h['episodes']} | {h['probe_max_error']} | "
                  f"{h['mean_settled_return']:.2f} | {h['exact_episodes']}/{h['episodes']} |")
        print(f"| {a['arm']} | | *total* | {a['environment_episodes_total']} episodes, "
              f"{a['environment_steps_total']} steps | | | |")
    print(f"\nwall clock {d['seconds']:.1f} s\n")


def environment():
    d = load('environment_policy.json')
    s = d['staged']
    print('## environment (research/discrete-backend/environment_policy.py)')
    print(f"scaffold space {s['space_size']} programs, horizon {d['horizon']}")
    print('| stage | mode | episodes | space | evaluated | conforming | certificate | s |')
    print('|---|---|---|---|---|---|---|---|')
    a, b = s['supervised'], s['reward']
    print(f"| 1 supervision | {a['mode']} | {s['stage1_episodes']} | {a['space_size']} | "
          f"{a['evaluated']} | {a['conforming']} | {a['certificate']} | {a['seconds']:.2f} |")
    print(f"| 2 reward | {b['mode']} | {s['stage2_episodes']} | {b['space_size']} | "
          f"{b['evaluated']} | {b['conforming']} | {b['certificate']} | {b['seconds']:.2f} |")
    print(f"| **total** | | **{s['search_episodes']}** | | | | | |")
    print()
    print(f"selection {s['selection']}, {s['search_steps']} environment steps")
    print(f"held out {s['held_out_mean_return']:.2f}/{d['horizon']} over "
          f"{d['evaluation_episodes']} episodes, {s['held_out_perfect']}/"
          f"{d['evaluation_episodes']} perfect; " +
          ', '.join(f"{k} {v:.2f}" for k, v in s['references'].items()))
    print()
    print('| unstaged, episodes per program | episodes | programs | conforming | certificate | s |')
    print('|---|---|---|---|---|---|')
    for row in d['unstaged']:
        print(f"| {row['episodes_per_program']} | {row['episodes']} | "
              f"{row['evaluated']}/{row['space_size']} | {row['conforming']} | "
              f"{row['certificate']} | {row['seconds']:.1f} |")
    print()


def beam():
    d = load('beam_search.json')
    print('## prefix and beam (research/discrete-backend/beam_search.py)')
    p = d['perception']
    print(f"rung 3 at R={p['resolution']}: {p['space_size']:,} programs, {p['records']} "
          f"supervised pixels, observation width {p['observation_width']}, "
          f"probes {p['probes']}")
    print('| mode | evaluated | conforming | certificate | node evaluations | s |')
    print('|---|---|---|---|---|---|')
    for label in ('enumerate_fit', 'enumerate_prefix'):
        r = p[label]
        print(f"| {label} | {r['evaluated']:,} | {r['conforming']} | {r['certificate']} | "
              f"{r['node_evaluations']:,} | {r['seconds']:.1f} |")
    print(f"\nidentical result: {p['identical']}; speedup {p['speedup']:.1f}x; "
          f"viability {p['viability']}")
    print()
    print('| beam | solved | conforming | same pick | discarded | space examined | certificate | s |')
    print('|---|---|---|---|---|---|---|---|')
    for r in p['beam']:
        print(f"| {r['beam']} | {r['solved']} | {r['conforming']} | "
              f"{r['matches_exhaustive_pick']} | {r['discarded']:,} | "
              f"{r['fraction_of_space_examined']:.4f} | {r['certificate']} | {r['seconds']:.1f} |")
    print()
    c = d['chain']
    print(f"chain control: {c['space_size']:,} programs, {c['examples']} examples, "
          f"reference {c['reference']}")
    print('| supervision | mode | conforming | certificate | node evaluations | s | reference conforms |')
    print('|---|---|---|---|---|---|---|')
    for label, entry in c['supervision'].items():
        for mode in ('enumerate_fit', 'enumerate_prefix'):
            r = entry[mode]
            print(f"| {label} | {mode} | {r['conforming']} | {r['certificate']} | "
                  f"{r['node_evaluations']:,} | {r['seconds']:.2f} | {entry['reference_conforms']} |")
    print()
    print('| supervision | beam | solved | conforming | recovers reference | discarded | space examined | certificate |')
    print('|---|---|---|---|---|---|---|---|')
    for label, entry in c['supervision'].items():
        for r in entry['beam']:
            print(f"| {label} | {r['beam']} | {r['solved']} | {r['conforming']} | "
                  f"{r['recovers_reference']} | {r['discarded']:,} | "
                  f"{r['fraction_of_space_examined']:.4f} | {r['certificate']} |")
    print(f"\nwall clock {d['seconds']:.1f} s\n")


if __name__ == '__main__':
    recurrent()
    environment()
    beam()
