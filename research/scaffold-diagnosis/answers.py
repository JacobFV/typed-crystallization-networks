"""The answers: whether each scaffold contains a solution, and its known repair.

This file is imported by `score.py` **only**.  `probe.py` and `run_probe.py` do
not import it, and `out/predictions.json` is committed before `score.py` runs, so
the probe's commitment to a prediction is enforced by construction rather than by
intention.  Every entry names where the answer comes from in the record.
"""
from __future__ import annotations
import json as _json, os as _os
from cases import FOLDS

# Addendum A2: the repair set for family C2 is decided by the exhaustive sweep in
# `establish_c2.py`, written to `out/c2_answer.json` before `score.py` is run.
_C2 = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'out', 'c2_answer.json')
C2_SOLVING = (frozenset(_json.load(open(_C2))['folds_that_conform_on_train'])
              if _os.path.exists(_C2) else frozenset(('add',)))

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
    # ADDENDUM A1.  The pre-registration declared `max2`'s repair as {max} "by
    # construction" and said the nine-fold sweep would establish it.  The sweep
    # establishes {min, max}: with the step values sign-flipped, a running minimum
    # of the negated prefix sums IS a running maximum of the prefix sums, so `min`
    # conforms on 527 training members exactly as `max` does.  The declared answer
    # was wrong; the enumeration is the authority and the correction is recorded
    # in PREREGISTRATION.md (addendum A1) and in RESULTS.md.
    TRUTH[f'C_max2_{_f}'] = {
        'contains_solution': _f in ('min', 'max'),
        'certificate': 'exhaustive nine-fold sweep here: 527 conforming on '
                       'training for each of min and max, 0 for the rest',
        'defect_kind': None if _f in ('min', 'max') else f'fold hole pinned to `{_f}`',
        'repair_family': None if _f in ('min', 'max') else 'swap',
        'repair': [] if _f in ('min', 'max') else ['min', 'max'],
        'source': 'the exhaustive sweep here; the pre-registered "{max} by '
                  'construction" was wrong -- addendum A1'}
    # ADDENDUM A2.  The defect in the FIRST accumulator, `second_fold` pinned to
    # `min`.  The declared answer is whatever the exhaustive sweep returns, written
    # to out/c2_answer.json before `score.py` is run against family C2.
    TRUTH[f'C2_bal_{_f}'] = {
        'contains_solution': _f in C2_SOLVING,
        'certificate': 'exhaustive nine-fold sweep over the acc_fold hole here',
        'defect_kind': None if _f in C2_SOLVING else f'acc hole pinned to `{_f}`',
        'repair_family': None if _f in C2_SOLVING else 'swap',
        'repair': [] if _f in C2_SOLVING else sorted(C2_SOLVING),
        'source': 'established by enumeration in out/c2_answer.json'}
