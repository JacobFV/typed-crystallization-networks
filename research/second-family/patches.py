"""Teach §54's `pool.is_window` about the second family, without editing it.

`research/reuse-ranking/pool.py` has one family-specific function -- it maps a
family name to the window's truth table so that a ranked row can be labelled
`is_window` -- and it knows only §52's two names (`maj`, `off`).  Nothing else
in `pool`, `objectives` or `context` is family-specific.

Rather than fork those files (which would break the "same code, different
family" comparison this track exists to make), the two functions are re-pointed
here.  The majority family's behaviour is preserved exactly by delegating to
the originals for any name this track does not define, and `is_window` is
otherwise the identical predicate with the arity read off the table width
instead of hard-coded to 3.

Import this module once, before `objectives.rank` is called.
"""
from __future__ import annotations

import _paths  # noqa: F401

import pool

import family

_UPSTREAM_TABLE = pool.window_table
_UPSTREAM_IS_WINDOW = pool.is_window


def window_table(family_name="maj"):
    if family_name in family.WINDOW_FN:
        return family.window_table(family_name)
    return _UPSTREAM_TABLE(family_name)


def is_window(k, family_name="maj"):
    if family_name not in family.WINDOW_FN:
        return _UPSTREAM_IS_WINDOW(k, family_name)
    tbl = window_table(family_name)
    arity = len(tbl).bit_length() - 1
    return bool(k.holes == arity and tuple(bool(v) for v in k.key[1]) == tbl)


pool.window_table = window_table
pool.is_window = is_window
