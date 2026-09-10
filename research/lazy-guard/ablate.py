"""Criterion 3, checked as an ablation matrix.

The pre-registration says: *the target algorithm does not appear verbatim in the
candidate set or rule set; the search must compose it.  This is checked by
ablation -- removing the specific construct must be the only thing that prevents
the result.*

So the matrix removes each construct **one at a time** and reports the resulting
expected-work ratio.  If exactly one removal collapses the ratio to 1.0 and the
others leave it intact, the gain is attributable to that construct and the
search composed the rest.  If several removals collapse it, or none does, the
claim is weaker and the table says so.

A separate row supplies the target expression verbatim to a grammar that lacks
the construct.  That row is what template instantiation looks like; it is
reported so the two are distinguishable rather than conflated.
"""
from __future__ import annotations

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import ir
import runner
import synth


def matrix(fx, domain, engine, spec_expected, base_constructs=synth.ABLATABLE,
           verbose=True, **kw):
    rows = {}
    full = tuple(base_constructs)
    for removed in (None,) + full:
        cons = tuple(c for c in full if c != removed)
        label = "full" if removed is None else "minus_%s" % removed
        if verbose:
            print("   ablation %s (%s)" % (label, ",".join(cons) or "none"))
        res = engine(fx, constructs=cons, domain=domain, verbose=False, **kw) \
            if engine is synth.toplevel_search else \
            engine(fx, grammar_constructs=cons, domain=domain, verbose=False, **kw)
        prog = res.get("program")
        if prog is None:
            rows[label] = {"solved": False, "expected_ops": None,
                           "expected_ratio_vs_spec": None,
                           "constructs_used": [],
                           "candidates_considered": res.get("candidates_considered")}
            continue
        prof, _ = runner.alg_profile(fx, prog, domain)
        rows[label] = {"solved": True,
                       "expected_ops": prof["expected_ops"],
                       "worst_ops": prof["worst_ops"],
                       "expected_ratio_vs_spec": spec_expected / prof["expected_ops"],
                       "constructs_used": sorted(prog.constructs()),
                       "result": repr(prog.output) if not prog.bindings else
                                 [("%s = %r" % (k, v)) for k, v in prog.bindings][-4:],
                       "candidates_considered": res.get("candidates_considered")}
    rows["_none"] = None
    del rows["_none"]
    return rows


def summarize(rows, threshold=3.0, joint=None):
    """Which removals actually prevent the result.

    Two verdicts, because the pre-registration's phrasing ("removing the
    specific construct must be the only thing that prevents the result")
    presumes a unique culprit and a search can legitimately find two different
    ways to the same win.  `single_blocking` is the literal reading;
    `capability_blocking` is the honest one when more than one construct in the
    IR expresses the same capability and the *set* is what is necessary.
    """
    full = rows.get("full", {})
    base = full.get("expected_ratio_vs_spec")

    def blocks(v):
        return (not v["solved"] or v["expected_ratio_vs_spec"] is None
                or v["expected_ratio_vs_spec"] < threshold)

    blocked = [k[len("minus_"):] for k, v in rows.items()
               if k.startswith("minus_") and blocks(v)]
    intact = [k[len("minus_"):] for k, v in rows.items()
              if k.startswith("minus_") and not blocks(v)]
    out = {"full_expected_ratio": base, "threshold": threshold,
           "single_removals_that_block": blocked,
           "single_removals_that_do_not_block": intact,
           "single_blocking": bool(base and base >= threshold and len(blocked) == 1)}
    if joint is not None:
        out["joint_removal"] = joint
        out["capability_blocking"] = bool(base and base >= threshold and not blocked
                                          and joint.get("blocked"))
    return out
