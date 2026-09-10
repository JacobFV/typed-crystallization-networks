"""The answers: whether each scaffold contains a solution, and its known repair.

This file is imported by `score.py` **only**.  `probe.py` and `run_probe.py` do
not import it, and `out/predictions.json` is committed before `score.py` runs, so
the probe's commitment to a prediction is enforced by construction rather than by
intention.  Every entry names where the answer comes from in the record.
"""
from __future__ import annotations
from cases import FOLDS

TRUTH = {
    'R1_counting_postaudit': {
        'contains_solution': False,
        'certificate': 'complete, 0 conforming of 45,375 (§45, re-derived here)',
        'defect_kind': 'insufficient statistic: the accumulator is a bracket count',
        'repair_family': 'fold',
        'repair': ['min', 'max'],
        'source': 'FINDINGS §45 (min) and §47 Q3c (max is its sign-flipped twin)'},
    'S3_counting_preaudit': {
        'contains_solution': True,
        'certificate': 'solved, §19 / §45 pre-audit control (0.9986 on n=724)',
        'defect_kind': None, 'repair_family': None, 'repair': [],
        'source': 'FINDINGS §19, reproduced by §45 control_preaudit.json'},
    'R2_tight_flat': {
        'contains_solution': False,
        'certificate': 'complete, 0 conforming of 230,400 (§44 arm 1)',
        'defect_kind': 'missing abstraction; three 2-input gates reach at most '
                       '4 of the 6 inputs, so no operator swap can repair it',
        'repair_family': 'module',
        'repair': ['MAJ3'],
        'source': 'FINDINGS §44 arm 3 (144 conforming, complete)'},
    'R3_tight_wrong_module': {
        'contains_solution': False,
        'certificate': 'complete, 0 conforming of 2,709,504 (§44 arm 4)',
        'defect_kind': 'wrong module inherited',
        'repair_family': 'module',
        'repair': ['MAJ3'],
        'source': 'FINDINGS §44 arm 3 vs arm 4'},
    'S2_tight_maj3': {
        'contains_solution': True,
        'certificate': 'complete, 144 conforming of 2,709,504 (§44 arm 3)',
        'defect_kind': None, 'repair_family': None, 'repair': [],
        'source': 'FINDINGS §44 arm 3'},
}
for _f in FOLDS:
    TRUTH[f'C_bal_{_f}'] = {
        'contains_solution': _f in ('min', 'max'),
        'certificate': 'complete (§47 Q3c: exactly min and max conform, 110 each)',
        'defect_kind': None if _f in ('min', 'max') else f'fold hole pinned to `{_f}`',
        'repair_family': None if _f in ('min', 'max') else 'swap',
        'repair': [] if _f in ('min', 'max') else ['min', 'max'],
        'source': 'FINDINGS §47 Q3c, re-derived here'}
    TRUTH[f'C_max2_{_f}'] = {
        'contains_solution': _f == 'max',
        'certificate': 'established here by the same exhaustive nine-fold sweep',
        'defect_kind': None if _f == 'max' else f'fold hole pinned to `{_f}`',
        'repair_family': None if _f == 'max' else 'swap',
        'repair': [] if _f == 'max' else ['max'],
        'source': 'construction: `ge 2` on a running maximum, plus the sweep'}
