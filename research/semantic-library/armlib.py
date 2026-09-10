"""Assemble the arms: which library the evaluation task inherits, and nothing else.

Everything except the library contents is held fixed -- same scaffold builder,
same examples, same signals, same objective, same seeds.  Modules mined by this
track are loaded from a real `tcn.library.Library` on disk under `policy="strict"`;
the two hand-authored modules are `later.hand_authored`, imported unchanged.
"""
from __future__ import annotations

from pathlib import Path

import _paths  # noqa: F401

from tcn.library import Library
from tcn.operators import Registry

import later

HERE = Path(__file__).resolve().parent
LIB = HERE / "library"

# arm -> library label under `library/`, or a hand-authored kind, or None.
L1_ARMS = {
    "arm1_none": None,
    "arm2_syntactic": ("lib", "syn_minall"),
    "arm2s_semantic": ("lib", "sem_minall"),
    "arm2s_trace": ("lib", "sem_trace"),
    "arm3_authored": ("hand", "maj"),
    "arm4_wrong_authored": ("hand", "distractor"),
    "arm4b_wrong_mined": ("lib", "syn_minall_runnerup"),
    "arm4s_runnerup": ("lib", "sem_minall_runnerup"),
    "arm4s_matched": ("lib", "sem_minall_matched"),
    "arm4s_offfamily": ("lib", "sem_off"),
}

DESCRIPTION = {
    "arm1_none": "no library; the flat space",
    "arm2_syntactic": "rank-1 of the rule under digest identity, C-minall",
    "arm2s_semantic": "rank-1 of the rule under (arity, truth table) identity, C-minall",
    "arm2s_trace": "the same, on C-trace -- §46's exact configuration",
    "arm3_authored": "the hand-authored MAJ3 -- the ceiling",
    "arm4_wrong_authored": "the hand-authored D134, same size and arity",
    "arm4b_wrong_mined": "the syntactic rule's runner-up, same machinery",
    "arm4s_runnerup": "the top arity-3 non-majority *pooled* class, same corpus",
    "arm4s_matched": "the top arity-3 non-majority pooled class of equal node count",
    "arm4s_offfamily": "the rank-1 arity-3 pooled class mined from the off-family F'",
}

HELDOUT_ARMS = ("arm1_none", "arm2_syntactic", "arm2s_semantic", "arm3_authored",
                "arm4_wrong_authored", "arm4s_offfamily", "arm2s_window")

DESCRIPTION["arm2s_window"] = (
    "EXPLORATORY, not pre-registered: the highest-ranked *majority* class in the "
    "same leave-one-out pooled table, whatever its rank. Separates 'the corpus "
    "does not contain it' from 'the ranking does not pick it'.")


def runnable_l1_arms():
    """The L1 arms whose library actually exists.

    `arm4s_matched` is defined by a pre-registered rule -- "the highest-ranked
    arity-3 non-majority pooled class whose representative has the same node
    count as rank 1" -- and on the primary corpus **no such class exists**: the
    majority class is the only 4-node arity-3 class in the eligible set.  The
    arm is therefore reported as *no candidate*, which is a measurement, not a
    failure, and it is skipped here rather than silently dropped.
    """
    out = []
    for arm, spec in L1_ARMS.items():
        if spec is None or spec[0] == "hand":
            out.append(arm)
            continue
        try:
            build(spec)
        except (KeyError, FileNotFoundError):
            continue
        out.append(arm)
    return out


def build(spec, policy="strict"):
    """(registry, module operator name or None) for one library spec."""
    r = Registry()
    if spec is None:
        return r, None
    kind, name = spec
    if kind == "hand":
        return r, r.register_module(later.hand_authored(r, name))
    _, aliases = Library(LIB).load([name], registry=r, policy=policy)
    return r, aliases[name]


def l1_spec(arm):
    return L1_ARMS[arm]


def heldout_spec(arm, held_out):
    """The library an arm offers when `held_out` is removed from the mining corpus."""
    short = held_out.split("_")[0]
    loo = held_out in ("t1_maj_abc_xor_d", "t2_maj_abc_and_d", "t3_maj_bcd_or_a",
                       "t4_maj_acd_xor_b", "t5_maj_abd_or_c")
    if arm == "arm2_syntactic":
        return ("lib", f"syn_wo_{short}" if loo else "syn_minall")
    if arm == "arm2s_semantic":
        return ("lib", f"sem_wo_{short}" if loo else "sem_minall")
    if arm == "arm2s_window":
        return ("lib", f"sem_wo_{short}_win" if loo else "sem_minall")
    return L1_ARMS[arm]
