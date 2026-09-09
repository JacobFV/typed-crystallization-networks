"""Render out/*.json into the markdown tables used by RESULTS.md."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
OUT = HERE / 'out'

LABEL = {
 'record_fixed_xor': ('record (bits+goal)', 'fixed table 6 (XOR), fixed wiring'),
 'record_multitable_affine': ('record (bits+goal)', '8 affine tables {0,3,5,6,9,10,12,15}'),
 'record_multitable_nonaffine': ('record (bits+goal)', '8 non-affine tables {1,2,4,7,8,11,13,14}'),
 'lookup_affine': ('lookup (bits+goal+program, 17 cand.)', '8 affine tables'),
 'lookup_nonaffine': ('lookup (bits+goal+program, 17 cand.)', '8 non-affine tables'),
 'interpreter_affine': ('interpreter (bits+goal+program, 1 cand.)', '8 affine tables'),
 'interpreter_nonaffine': ('interpreter (bits+goal+program, 1 cand.)', '8 non-affine tables'),
 'lookup_wired_affine': ('lookup+wiring (17 cand.)', '8 affine tables, random wiring'),
 'interpreter_wired_affine': ('interpreter+wiring (1 cand.)', '8 affine tables, random wiring'),
 'record_depth12': ('record (bits+goal)', 'depth 1-2, random wiring, random tables'),
}
EVAL_LABEL = {
 'fixed_xor': 'fixed XOR, held-out episodes (64)',
 'fixed_xor_16': 'fixed XOR, held-out episodes (16, record budget)',
 'train_tables_affine': 'affine tables (seen), held-out episodes',
 'heldout_tables_nonaffine': 'NON-AFFINE tables (unseen)',
 'train_tables_nonaffine': 'non-affine tables (seen), held-out episodes',
 'heldout_tables_affine': 'AFFINE tables (unseen)',
 'train_tables_affine_wired': 'affine tables + random wiring (seen)',
 'heldout_tables_nonaffine_wired': 'NON-AFFINE tables + random wiring (unseen)',
 'depth_1': 'depth 1 (seen)', 'depth_2': 'depth 2 (seen)',
 'depth_3': 'DEPTH 3 (unseen)', 'depth_4': 'DEPTH 4 (unseen)',
}
ORDER = list(LABEL)

rows = []
for name in ORDER:
    p = OUT / f'{name}.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    scaffold, train = LABEL[name]
    for ename, s in d['summary'].items():
        b = d['baselines'][ename if ename in d['baselines'] else ename.replace('_16', '')]
        rows.append((scaffold, train, EVAL_LABEL.get(ename, ename), s, b, len(d['seeds'])))

print('| train scaffold | train condition | eval condition | mean return (8 seeds) | sd across seeds | range | always-True | always-False | uniform random | best constant |')
print('|---|---|---|---|---|---|---|---|---|---|')
for scaffold, train, ev, s, b, n in rows:
    print(f"| {scaffold} | {train} | {ev} | **{s['mean']:.2f}** | {s['sd']:.2f} | {s['min']:.2f}-{s['max']:.2f} | "
          f"{b['always_true']['mean']:.2f} | {b['always_false']['mean']:.2f} | {b['uniform_random']['mean']:.2f} | {b['majority_constant']:.2f} |")

print()
print('Per-seed detail:')
for name in ORDER:
    p = OUT / f'{name}.json'
    if not p.exists():
        continue
    d = json.loads(p.read_text())
    for ename in d['summary']:
        vals = [r['evaluations'][ename]['mean'] for r in d['runs']]
        print(f"  {name} / {ename}: " + " ".join(f"{v:.2f}" for v in vals))
    sel = [r['selections'].get('relation') for r in d['runs']]
    print(f"  {name} relation-node selection per seed: {sel}")
    print(f"  {name} goal_relation selection per seed: {[r['selections'].get('goal_relation') for r in d['runs']]}")
    print(f"  {name} train return (last 32 eps): " + " ".join(f"{r['train_return_last32']:.2f}" for r in d['runs']))
