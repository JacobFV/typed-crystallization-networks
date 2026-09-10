"""Exact semantics of the two-accumulator family, and the validation that it is exact.

Section 45's `bound.py` established the discipline this file follows: a fast
simulator of a scaffold's family is only usable once it has been checked against
the **real typed program**, member by member, with the number of members checked
and the number of mismatches reported. `validate()` does that and every script
that uses this module calls it and records the result.

The family is `dyck_scaffold.stage_b_dyck` with the fold operator left as a hole,
so the same code covers the shipped `min` accumulator and the mechanical sweep
over every other core operator of the same signature (Q3c).

Semantics, read off the scaffold rather than assumed:

    length  = project(text, 0)          # prompt_bytes
    symbols = length - c                # POS is unsigned with bounds (0,128)
    b_i     = (byte at 14 + i) == '('   # the frozen stage-A module, base 14
    m_i     = (b_i ? plus : minus) if i < symbols else 0
    acc_i   = m_0 + ... + m_i
    lo_0    = X(acc_0, 0);  lo_i = X(lo_{i-1}, acc_i)
    answer  = and( op_t(acc_{P-1}, v_t), op_m(lo_{P-1}, v_m) )

A member that would raise -- `sub` underflowing unsigned POS, a POS or CNT value
outside its declared range, an illegal numeric domain -- is *unusable*, exactly
as `tcn.search.evaluate` treats it (it returns None and the search moves on), and
`values()` returns None for it.
"""
from __future__ import annotations
import sys, os, math
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE)); sys.path.insert(0, ROOT)

BASE = 14                 # the frozen stage-A selection, certificate 'unique'
OPEN_BYTE = 40            # '('
POS_HI = 128              # POS = integer(32, signed=False, bounds=(0, CAPACITY))
CNT_BITS = 16                                  # CNT = integer(16), signed
CNT_LO, CNT_HI = -(1 << 15), (1 << 15) - 1
STEP_VALUES = (-2, -1, 0, 1, 2)
RULE_CONSTS = (-2, -1, 0, 1, 2)
RULES = [(op, v) for v in RULE_CONSTS for op in ('eq', 'ge', 'le')]
SUB = list(range(0, 121))

FOLDS = {
    'min': lambda a, b: min(a, b),
    'max': lambda a, b: max(a, b),
    'add': lambda a, b: a + b,
    'sub': lambda a, b: a - b,
    'mul': lambda a, b: a * b,
    'mod': lambda a, b: (_die() if b == 0 else a % b),   # core: `a % b`, Python semantics
    'idiv': lambda a, b: (_die() if b == 0 else a // b),
    # core raises "shift outside bit width" unless 0 <= b < bits (CNT is 16 bits),
    # not merely on a negative shift -- `tcn/operators.py`, the BINARY branch of
    # `exact`. Modelling only `b < 0` disagreed with the real program on 8 of 150
    # `shl` members and 31 of 150 `shr` members; with the width bound, 0.
    'shl': lambda a, b: (_die() if not 0 <= b < CNT_BITS else a << b),
    'shr': lambda a, b: (_die() if not 0 <= b < CNT_BITS else a >> b),
}


class Unusable(Exception):
    pass


def _die():
    raise Unusable()


def opens(e, positions=22):
    """b_0..b_{positions-1} for one episode, from the raw bytes the module reads."""
    raw = e['text'].raw[1]
    return tuple(raw[BASE + i] == OPEN_BYTE for i in range(positions))


def values(e, c, plus, minus, positions=22, fold='min', b=None):
    """(acc_last, lo_last) or None if this member is unusable on this episode."""
    length = e['prompt_bytes']
    if c > length:
        return None                     # unsigned `sub` underflow
    symbols = length - c
    if not 0 <= symbols <= POS_HI:
        return None                     # outside POS's declared bounds
    f = FOLDS[fold]
    bb = b if b is not None else opens(e, positions)
    acc = lo = None
    try:
        for i in range(positions):
            m = (plus if bb[i] else minus) if i < symbols else 0
            acc = m if i == 0 else acc + m
            lo = f(acc, 0) if i == 0 else f(lo, acc)
            if not (CNT_LO <= acc <= CNT_HI and CNT_LO <= lo <= CNT_HI):
                return None
    except (Unusable, OverflowError, ValueError, ZeroDivisionError):
        return None
    return acc, lo


def rule(op, v, x):
    return {'eq': x == v, 'ge': x >= v, 'le': x <= v}[op]


def predict(e, c, plus, minus, total_ok, min_ok, positions=22, fold='min', b=None):
    """Member's boolean answer, or None if unusable on this episode."""
    got = values(e, c, plus, minus, positions, fold, b)
    if got is None:
        return None
    acc, lo = got
    return rule(*total_ok, acc) and rule(*min_ok, lo)


# ---------------------------------------------------------------------------
# validation against the real typed program
# ---------------------------------------------------------------------------

def validate(n=150, seed=0, positions=22, fold='min', episodes=None, program=None,
             registry=None):
    """Check the simulator against the real program, **per episode**, on `n` random members.

    The comparison is deliberately not routed through `tcn.search.evaluate`: that
    function abandons an example once the tolerance is exceeded, so a member whose
    later example would raise can come back with a finite error, and a
    simulator-versus-`evaluate` comparison then reports a mismatch that is an
    artifact of the short-circuit rather than a semantic disagreement. (Measured:
    routing the check through `evaluate` reported 8 mismatches in 150 for the
    `shl` fold, all of that kind.) `Program.execute` is called once per episode
    instead, and the simulator's answer and its raise/no-raise verdict are both
    compared exactly.

    Returns the numbers section 45 reported for its own simulator: members
    checked, mismatches, and how many episode evaluations each account found
    unusable. A mismatch is any per-episode disagreement, in value or in whether
    the member runs at all.
    """
    import random
    import prepare
    from run_stage_b import build_module
    import accum_scaffold

    eps = episodes if episodes is not None else prepare.load()['train']
    if program is None:
        module, registry, _ = build_module()
        program, _ = accum_scaffold.stage_b_accum(module, registry, positions=positions,
                                                  fold=fold)
    names = [n_.name for n_ in program.nodes]
    rnd = random.Random(seed)
    checked = mismatches = 0
    ep_checked = ep_unusable_sim = ep_unusable_real = 0
    detail = []
    for _ in range(n):
        ci = rnd.randrange(len(SUB)); pi = rnd.randrange(5); mi = rnd.randrange(5)
        ti = rnd.randrange(15); oi = rnd.randrange(15)
        sel = {k: 0 for k in names}
        sel.update({'symbols': ci, 'plus': pi, 'minus': mi, 'total_ok': ti, 'min_ok': oi})
        c, plus, minus = SUB[ci], STEP_VALUES[pi], STEP_VALUES[mi]
        tot, mn = RULES[ti], RULES[oi]
        checked += 1
        bad = False
        for e in eps:
            sim = predict(e, c, plus, minus, tot, mn, positions, fold)
            try:
                _, _, trace = program.execute({'text': e['text']}, registry=registry,
                                              selections=sel)
                real = bool(round(trace['answer'].flat()[0]))
            except (ValueError, TypeError, OverflowError, ZeroDivisionError,
                    ArithmeticError, IndexError):
                real = None
            ep_checked += 1
            ep_unusable_sim += sim is None
            ep_unusable_real += real is None
            if sim != real:
                bad = True
                if len(detail) < 10:
                    detail.append({'c': c, 'plus': plus, 'minus': minus,
                                   'total_ok': list(tot), 'min_ok': list(mn),
                                   'simulator': sim, 'real_program': real,
                                   'string': e['string']})
        mismatches += bad
    return {'fold': fold, 'members_checked': checked, 'mismatches': mismatches,
            'episode_evaluations_checked': ep_checked,
            'episode_evaluations_unusable_in_simulator': ep_unusable_sim,
            'episode_evaluations_unusable_in_real_program': ep_unusable_real,
            'episodes': len(eps), 'compared_via': 'Program.execute, per episode',
            'mismatch_detail': detail}
