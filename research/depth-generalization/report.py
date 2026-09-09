"""Render out/*.json into the tables in RESULTS.md."""
from __future__ import annotations
import json
import sys
from pathlib import Path

OUT = Path(__file__).parent / 'out'
DEPTHS = ('1', '2', '3', '4', '6', '8')
LABEL = {'record': 'record (`bits`,`goal`)',
         'interpreter': 'interpreter, wire choice free',
         'interpreter_fixed': 'interpreter, wire lookup pinned',
         'interpreter_nomux': 'interpreter, no settle mux (ablation)'}


def load(name):
    p = OUT / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def main():
    base = load('baselines')
    print('### Baselines, on the evaluation episodes and the same scored window\n')
    print('| depth | always true | always false | uniform random | best constant | fraction target true |')
    print('|---|---|---|---|---|---|')
    for d in DEPTHS:
        b = base[d]
        print(f"| {d} | {b['always_true']['mean']:.2f} | {b['always_false']['mean']:.2f} | "
              f"{b['uniform_random']['mean']:.2f} | **{b['best_constant']:.2f}** | "
              f"{b['fraction_target_true']:.3f} |")

    for which, title in (('exact', 'Exported exact frozen program'), ('soft', 'Soft model, deterministic')):
        print(f'\n### {title} — mean settled return over 8 seeds x 64 held-out episodes\n')
        print('| scaffold | space | ' + ' | '.join(f'd{d}' + (' *(seen)*' if d in '12' else '') for d in DEPTHS) + ' |')
        print('|---|---|' + '---|' * len(DEPTHS))
        for kind in ('record', 'interpreter_nomux', 'interpreter', 'interpreter_fixed'):
            r = load(kind)
            if not r:
                continue
            cells = []
            for d in DEPTHS:
                a = r['aggregate'][which][d]
                cells.append(f"**{a['mean']:.2f}** (sd {a['sd']:.2f})")
            print(f"| {LABEL[kind]} | {r['space_size']} | " + ' | '.join(cells) + ' |')
        print(f"| best constant | — | " + ' | '.join(f"{base[d]['best_constant']:.2f}" for d in DEPTHS) + ' |')
        print(f"| uniform random | — | " + ' | '.join(f"{base[d]['uniform_random']['mean']:.2f}" for d in DEPTHS) + ' |')

    print('\n### What each seed selected\n')
    print('| scaffold | seed | selections | exact d8 |')
    print('|---|---|---|---|')
    for kind in ('interpreter', 'interpreter_fixed', 'interpreter_nomux', 'record'):
        r = load(kind)
        if not r:
            continue
        for row in r['per_seed']:
            print(f"| {LABEL[kind]} | {row['seed']} | `{row['selections']}` | {row['exact']['8']['mean']:.2f} |")

    print('\n### Enumeration reference\n')
    print('| scaffold | space | evaluated | exhausted | unique | selection | seconds | held-out d1..d8 |')
    print('|---|---|---|---|---|---|---|---|')
    for kind in ('interpreter_fixed', 'interpreter'):
        r = load(f'enumerate_{kind}')
        if not r:
            continue
        hold = ' / '.join(f"{r['holdout'][d]['mean']:.2f}" for d in DEPTHS)
        print(f"| {LABEL[kind]} | {r['space_size']} | {r['evaluated']} | {r['exhausted']} | "
              f"{r['unique']} | `{r['selection']}` | {r['seconds']:.0f} | {hold} |")

    g = load('gradients')
    if g:
        print('\n### Gradient reaching each free choice, one training episode at initialization\n')
        print('| node | interpreter (task+regularizer) | interpreter (task only) | '
              'no settle mux (task+reg) | no settle mux (task only) |')
        print('|---|---|---|---|---|')
        def fmt(x):
            return 'None (unreachable)' if x is None else f'{x:.3g}'
        for node in ('va', 'vb', 'relation', 'goal_relation'):
            print(f"| `{node}` | {fmt(g['interpreter'][node])} | {fmt(g['interpreter_task_only'][node])} | "
                  f"{fmt(g['interpreter_nomux'][node])} | {fmt(g['interpreter_nomux_task_only'][node])} |")


if __name__ == '__main__':
    main()
