"""Assemble the arms: which library the later task inherits, and nothing else.

Everything except the library contents is held fixed -- same scaffold builder,
same examples, same signals, same objective, same seeds.
"""
from __future__ import annotations

from pathlib import Path

from tcn.library import Library
from tcn.operators import Registry

import later

HERE = Path(__file__).parent
LIB = HERE / "library"

ARMS = ("arm1_none", "arm2_earned", "arm3_authored",
        "arm4_wrong_authored", "arm4b_wrong_mined")

DESCRIPTION = {
    "arm1_none": "no library; the flat space",
    "arm2_earned": "library holds the module the rule earned from the earlier tasks",
    "arm3_authored": "library holds the hand-authored MAJ3 of the existing demonstration",
    "arm4_wrong_authored": "library holds truth table 134, hand-authored, same size and arity",
    "arm4b_wrong_mined": "library holds the rule's runner-up abstraction, same machinery",
}


def build(arm, policy="strict"):
    """Return (registry, module operator name or None) for one arm."""
    r = Registry()
    if arm == "arm1_none":
        return r, None
    if arm == "arm2_earned":
        _, aliases = Library(LIB).load(["earned"], registry=r, policy=policy)
        return r, aliases["earned"]
    if arm == "arm4b_wrong_mined":
        _, aliases = Library(LIB).load(["earned_runner_up"], registry=r, policy=policy)
        return r, aliases["earned_runner_up"]
    kind = "maj" if arm == "arm3_authored" else "distractor"
    name = r.register_module(later.hand_authored(r, kind))
    return r, name
