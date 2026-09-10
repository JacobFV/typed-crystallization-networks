"""Exact semantics of `scaffolds2.stage_b_gen`, with both holes open, and its validation.

This is `research/dyck-learnability/family.py` generalised in exactly one way:
the accumulator's own fold is a parameter too, so the same simulator covers
§45's counting scaffold, §47's two-accumulator scaffold, and both of the probe's
repair families.  The operator table -- including core's raise conditions for
`shl`/`shr` (`0 <= b < bits`, the bug §47 found in its own simulator and fixed)
and for `idiv`/`mod` -- is imported from `family.py` rather than restated, so it
cannot drift.

`validate()` follows §45's `bound.py` discipline in §47's per-episode form: the
simulator is checked against `Program.execute` example by example on random
members, and both the answer and the raise/no-raise verdict are compared.
"""
from __future__ import annotations
import random, sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)
import family

BASE, OPEN_BYTE = family.BASE, family.OPEN_BYTE
POS_HI = family.POS_HI
CNT_LO, CNT_HI = family.CNT_LO, family.CNT_HI
STEPS = family.STEP_VALUES
RULES = family.RULES
SUB = family.SUB
FOLDS = family.FOLDS
opens = family.opens
rule = family.rule


def values(e, c, plus, minus, positions=22, acc_fold='add', second_fold=None, b=None):
    """(acc_last, lo_last or None), or None if this member is unusable here."""
    length = e['prompt_bytes']
    if c > length:
        return None                     # unsigned `sub` underflow
    symbols = length - c
    if not 0 <= symbols <= POS_HI:
        return None                     # outside POS's declared bounds
    fa = FOLDS[acc_fold]
    fs = FOLDS[second_fold] if second_fold is not None else None
    bb = b if b is not None else opens(e, positions)
    acc = lo = None
    try:
        for i in range(positions):
            m = (plus if bb[i] else minus) if i < symbols else 0
            acc = m if i == 0 else fa(acc, m)
            if not CNT_LO <= acc <= CNT_HI:
                return None
            if fs is not None:
                lo = fs(acc, 0) if i == 0 else fs(lo, acc)
                if not CNT_LO <= lo <= CNT_HI:
                    return None
    except (family.Unusable, OverflowError, ValueError, ZeroDivisionError):
        return None
    return acc, lo


def validate(program, registry, episodes, positions=22, acc_fold='add',
             second_fold=None, n=150, seed=0):
    """Check the simulator against the real typed program, per episode."""
    names = [nd.name for nd in program.nodes]
    two = second_fold is not None
    rnd = random.Random(seed)
    checked = mismatches = 0
    ep_checked = ep_sim = ep_real = 0
    detail = []
    for _ in range(n):
        ci = rnd.randrange(len(SUB)); pi = rnd.randrange(5); mi = rnd.randrange(5)
        ti = rnd.randrange(15); oi = rnd.randrange(15)
        sel = {k: 0 for k in names}
        sel.update({'symbols': ci, 'plus': pi, 'minus': mi})
        sel.update({'total_ok': ti, 'min_ok': oi} if two else {'answer': ti})
        c, plus, minus = SUB[ci], STEPS[pi], STEPS[mi]
        tot, mn = RULES[ti], RULES[oi]
        checked += 1
        bad = False
        for e in episodes:
            got = values(e, c, plus, minus, positions, acc_fold, second_fold)
            if got is None:
                sim = None
            else:
                acc, lo = got
                sim = rule(*tot, acc) and (rule(*mn, lo) if two else True)
            try:
                _, _, trace = program.execute({'text': e['text']}, registry=registry,
                                              selections=sel)
                real = bool(round(trace['answer'].flat()[0]))
            except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                    ArithmeticError, IndexError):
                real = None
            ep_checked += 1
            ep_sim += sim is None
            ep_real += real is None
            if sim != real:
                bad = True
                if len(detail) < 10:
                    detail.append({'c': c, 'plus': plus, 'minus': minus,
                                   'total_ok': list(tot), 'min_ok': list(mn) if two else None,
                                   'simulator': sim, 'real_program': real,
                                   'string': e['string']})
        mismatches += bad
    return {'acc_fold': acc_fold, 'second_fold': second_fold,
            'members_checked': checked, 'mismatches': mismatches,
            'episode_evaluations_checked': ep_checked,
            'episode_evaluations_unusable_in_simulator': ep_sim,
            'episode_evaluations_unusable_in_real_program': ep_real,
            'episodes': len(episodes),
            'compared_via': 'Program.execute, per episode',
            'mismatch_detail': detail}
