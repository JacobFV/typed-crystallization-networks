"""The cases: failed scaffolds, solvable scaffolds, and (separately) the answers.

`CASES` is what the probe sees: a scaffold, a registry, training episodes and
their labels.  The answers -- whether a conforming program exists, with its
certificate, and what the known repair is -- live in `answers.py`, which this
file does not import.

`probe.py` and `run_probe.py` import `cases` only.  `score.py` is the single
file that imports `answers`, and it runs after `out/predictions.json` is
committed.

Stream discipline (§39): `hardening` is pinned explicitly at every draw.  The
post-audit split is §45's, drawn once by `prepare.py`; the pre-audit control
split is drawn here by the same vendored `splits.py` with `hardening='none'`.
"""
from __future__ import annotations
import sys, os, importlib.util, functools
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import common, prepare, splits as splits_mod, scaffolds2
from run_stage_b import build_module

_spec = importlib.util.spec_from_file_location(
    'lc_scaffolds', os.path.join(ROOT, 'research', 'language-capability', 'scaffolds.py'))
lc = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(lc)

EA = os.path.join(ROOT, 'research', 'earned-abstraction')

POSITIONS = 22
PREAUDIT_POSITIONS = 16
PREAUDIT_TRAIN_LENGTHS = (2, 4, 6)
PREAUDIT_HELDOUT_LENGTHS = (8, 10, 12, 14, 16, 18, 20, 22)


# --------------------------------------------------------------------- labels

def label_bal(e):
    """The stream's own answer probe: is the string balanced?"""
    return bool(e['label'])


def max_prefix(s: str) -> int:
    d = m = 0
    for ch in s:
        d += 1 if ch == '(' else -1
        m = max(m, d)
    return m


def label_max2(e):
    """Constructed label: does the string ever reach nesting depth 2?

    Threshold 2 because the scaffold's readout grid is `{eq,ge,le} x {-2..2}`,
    so `ge 2` on a running maximum with `plus=+1, minus=-1` is the only
    max-prefix threshold the family can express at all.
    """
    return max_prefix(e['string']) >= 2


TASK_LABELS = {'bal': label_bal, 'max2': label_max2}


# ------------------------------------------------------------------ language

@functools.lru_cache(maxsize=1)
def _lang_base():
    module, registry, frozen = build_module()
    return module, registry


@functools.lru_cache(maxsize=None)
def lang_program(acc_fold, second_fold, positions):
    module, registry = _lang_base()
    prog, signals = scaffolds2.stage_b_gen(module, registry, positions=positions,
                                           acc_fold=acc_fold, second_fold=second_fold)
    return prog, registry, signals


@functools.lru_cache(maxsize=1)
def _preaudit_split():
    return splits_mod.build(hardening=common.STREAM_PRE_AUDIT, n_train=24,
                            train_lengths=PREAUDIT_TRAIN_LENGTHS,
                            heldout_lengths=PREAUDIT_HELDOUT_LENGTHS)


def _lang_case(cid, acc_fold, second_fold, positions, stream, task):
    def build():
        s = prepare.load() if stream == common.STREAM_POST_AUDIT else _preaudit_split()
        lab = TASK_LABELS[task]
        prog, registry, signals = lang_program(acc_fold, second_fold, positions)
        return {'id': cid, 'adapter': 'lang',
                'kind': 'counting' if second_fold is None else 'accum',
                'acc_fold': acc_fold, 'second_fold': second_fold,
                'positions': positions, 'stream': stream, 'task': task,
                'program': prog, 'registry': registry, 'signals': signals,
                'train': s['train'], 'labels': [lab(e) for e in s['train']],
                'heldout': {k: (s[k], [lab(e) for e in s[k]])
                            for k in ('heldout_seen_lengths', 'heldout_unseen_lengths')},
                # B-sweeponly's fixed default site: the output-adjacent accumulator
                'default_site': (f'acc{positions - 1}' if second_fold is None
                                 else f'lo{positions - 1}')}
    return build


# ------------------------------------------------------------------- boolean

@functools.lru_cache(maxsize=1)
def _ea():
    sys.path.insert(0, EA)
    import later, arms                                     # noqa: E402
    return later, arms


def _bool_case(cid, arm):
    def build():
        later, arms = _ea()
        registry, module_name = arms.build(arm)
        prog = later.tight_scaffold(registry, module_name)
        ex = later.later_examples()
        return {'id': cid, 'adapter': 'bool', 'kind': 'tight', 'arm': arm,
                'module_name': module_name,
                'stream': 'complete truth table (64 rows), no sampling',
                'task': 'maj(a,b,c) xor maj(d,e,f)',
                'program': prog, 'registry': registry, 'signals': later.SIGNALS,
                'train': ex, 'labels': [bool(e['targets']['out'].decoded) for e in ex],
                'heldout': {}, 'default_site': 'y'}
    return build


# ---------------------------------------------------------------------- table

FOLDS = ('add', 'idiv', 'max', 'min', 'mod', 'mul', 'shl', 'shr', 'sub')

CASES = {}
CASES['R1_counting_postaudit'] = _lang_case(
    'R1_counting_postaudit', 'add', None, POSITIONS,
    common.STREAM_POST_AUDIT, 'bal')
CASES['S3_counting_preaudit'] = _lang_case(
    'S3_counting_preaudit', 'add', None, PREAUDIT_POSITIONS,
    common.STREAM_PRE_AUDIT, 'bal')
for _t in ('bal', 'max2'):
    for _f in FOLDS:
        CASES[f'C_{_t}_{_f}'] = _lang_case(
            f'C_{_t}_{_f}', 'add', _f, POSITIONS, common.STREAM_POST_AUDIT, _t)
# Addendum A2: the defect moved to the FIRST accumulator, so that not every
# failed language scaffold has the same correct answer.
for _f in FOLDS:
    CASES[f'C2_bal_{_f}'] = _lang_case(
        f'C2_bal_{_f}', _f, 'min', POSITIONS, common.STREAM_POST_AUDIT, 'bal')
CASES['R2_tight_flat'] = _bool_case('R2_tight_flat', 'arm1_none')
CASES['R3_tight_wrong_module'] = _bool_case('R3_tight_wrong_module', 'arm4_wrong_authored')
CASES['S2_tight_maj3'] = _bool_case('S2_tight_maj3', 'arm3_authored')

# `S1_dyck_min` is `C_bal_min` by construction -- the same program, asserted by
# digest in `run_probe.py` -- so it is named as an alias rather than duplicated.
ALIASES = {'S1_dyck_min': 'C_bal_min'}
