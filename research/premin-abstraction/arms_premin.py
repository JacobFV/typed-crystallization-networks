"""The arms. The only thing that differs between them is the library.

§44's five arms verbatim -- same later task, same scaffolds, same candidate
construction, same seeds, same objective -- plus the arms this track adds, each
inheriting the rank-1 abstraction mined from one non-minimised corpus through a
real `tcn.library.Library`.
"""
from __future__ import annotations

from pathlib import Path

from tcn.library import Library
from tcn.operators import Registry

import later

HERE = Path(__file__).parent
LIB = HERE / "library"
EA_LIB = HERE.parent / "earned-abstraction" / "library"

PRIMARY = ("arm1_none", "arm2_earned", "arm2p_trace", "arm3_authored",
           "arm4_wrong_authored", "arm4b_wrong_mined")
SECONDARY = ("arm2p_minall", "arm2p_plus1", "arm2p_plus1one")
#: Not pre-registered. Added after seeing the ranked table, and labelled as
#: exploratory everywhere it appears: the highest-ranked *majority-computing*
#: abstraction C-trace offers (rank 18 of 182), inherited through the same
#: library path. It separates "the corpus does not contain the abstraction"
#: from "the ranking does not pick it".
#: Also not pre-registered: the rank-1 proposal of the *semantic*-identity
#: variant of the rule (`mine_semantic.py`) on `C-trace`. Its module is the
#: same content digest as `arm3p_mined_maj3`'s, which is itself the finding;
#: the arm is run anyway so the number is measured and not inferred.
EXPLORATORY = ("arm3p_mined_maj3", "arm2s_semantic_trace")
ARMS = PRIMARY + SECONDARY + EXPLORATORY

MINED = {"arm2p_trace": "trace", "arm2p_minall": "minall",
         "arm2p_plus1": "plus1", "arm2p_plus1one": "plus1one",
         "arm3p_mined_maj3": "trace_maj3",
         "arm2s_semantic_trace": "semantic_trace"}

DESCRIPTION = {
    "arm1_none": "no library; the flat space",
    "arm2_earned": "module mined from the minimised corpus C-min (this is §44's earned arm)",
    "arm2p_trace": "module mined from C-trace (all programs at length k and k+1)",
    "arm3_authored": "hand-authored MAJ3 -- the ceiling",
    "arm4_wrong_authored": "truth table 134, hand-authored, same size and arity",
    "arm4b_wrong_mined": "§44's runner-up abstraction, same machinery",
    "arm2p_minall": "module mined from C-minall (every minimum-length program)",
    "arm2p_plus1": "module mined from C-plus1 (every length k+1 program)",
    "arm2p_plus1one": "module mined from C-plus1-one (one length k+1 program per task)",
    "arm3p_mined_maj3": "EXPLORATORY, not pre-registered: C-trace's rank-18 majority abstraction",
    "arm2s_semantic_trace": "EXPLORATORY, not pre-registered: rank-1 of the semantic-identity rule on C-trace",
}


def build(arm, policy="strict"):
    """Return (registry, module operator name or None) for one arm."""
    r = Registry()
    if arm == "arm1_none":
        return r, None
    if arm == "arm2_earned":
        _, aliases = Library(EA_LIB).load(["earned"], registry=r, policy=policy)
        return r, aliases["earned"]
    if arm == "arm4b_wrong_mined":
        _, aliases = Library(EA_LIB).load(["earned_runner_up"], registry=r, policy=policy)
        return r, aliases["earned_runner_up"]
    if arm in MINED:
        label = MINED[arm]
        _, aliases = Library(LIB).load([label], registry=r, policy=policy)
        return r, aliases[label]
    kind = "maj" if arm == "arm3_authored" else "distractor"
    return r, r.register_module(later.hand_authored(r, kind))
