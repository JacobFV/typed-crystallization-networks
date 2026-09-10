"""The three identity relations of `PREREGISTRATION.md`.  All exact.  No sampling.

D   syntactic          `Program.digest` of the canonical fragment (§44's identity).
S   type-exact semantic `(input types, output type, exhaustive I/O map)`, computed
                        only when every hole type and the output type is finite and
                        the product of the carrier sizes is <= EXHAUST_CAP.
S*  carrier-abstracted  the fragment re-typed at a common placeholder carrier and
                        exhausted at TWO widths; admitted only if both agree.

A carrier is finite here when the `Type` pins it: `bool` (2), an integer whose
width or whose semantic `bounds` leave at most EXHAUST_CAP values.  `bounds` are
enforced by `tcn.types.validate_raw`, so `POS = integer(32, bounds=(0,128))` has
129 inhabitants, not 2**32; that is the type's carrier, not an approximation of
it.  Floating carriers, products and sets are **not** exhausted and are reported
as unexhaustible with the reason.

Partiality: operators in this algebra are partial (unsigned `sub` underflow,
`idiv` by zero, `log` of a non-positive).  A row whose evaluation raises is
recorded as the distinguished symbol `ERR` rather than dropped, so the map stays
total over the carrier and the identity stays exact.  (Amendment A1.)
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.graph import Program, Node, Candidate
from tcn.operators import Registry
from tcn.types import BOOL, Type, Value, integer

EXHAUST_CAP = 1 << 20
ERR = "ERR"


# ------------------------------------------------------------------ carriers
def carrier(t, cap=EXHAUST_CAP):
    """The finite inhabitants of `t`, or None with a reason."""
    if t.kind == "bool":
        return (False, True), None
    if t.kind == "int":
        if t.encoding.kind != "integer":
            return None, f"non-integer encoding ({t.encoding.kind})"
        lo = -(2 ** (t.bits - 1)) if t.encoding.signed else 0
        hi = 2 ** (t.bits - int(t.encoding.signed)) - 1
        if t.bounds is not None:
            lo, hi = max(lo, int(t.bounds[0])), min(hi, int(t.bounds[1]))
        n = hi - lo + 1
        if n > cap:
            return None, f"integer carrier {n} > cap"
        return tuple(range(lo, hi + 1)), None
    if t.kind == "tuple":
        return None, f"product type ({len(t.items)} fields)"
    return None, f"{t.kind} type"


def domain_size(types, cap=EXHAUST_CAP):
    total = 1
    for t in types:
        u, why = carrier(t, cap)
        if u is None:
            return None, why
        total *= len(u)
        if total > cap:
            return None, f"joint domain > cap ({cap})"
    return total, None


# ------------------------------------------------------------ S: exhaustion
def _run_row(canon, registry, values):
    try:
        out, _ = canon.run({k: v for (k, _), v in zip(canon.inputs, values)},
                           registry=registry)
        return next(iter(out.values())).raw
    except Exception:
        return ERR


def signature(canon, registry, cap=EXHAUST_CAP):
    """The exhaustive I/O map, or (None, reason)."""
    types = [t for _, t in canon.inputs]
    outp = canon.nodes[-1].output if canon.nodes else None
    size, why = domain_size(types, cap)
    if size is None:
        return None, why
    if outp is not None:
        _, owhy = carrier(outp, cap)
        if owhy is not None and outp.kind != "int":
            return None, f"output {owhy}"
    universes = [carrier(t, cap)[0] for t in types]
    rows = []
    idx = [0] * len(universes)
    import itertools
    for combo in itertools.product(*universes):
        vals = []
        ok = True
        for t, x in zip(types, combo):
            try:
                vals.append(Value.of(t, x))
            except Exception:
                ok = False
                break
        rows.append(_run_row(canon, registry, vals) if ok else ERR)
    return tuple(rows), None


def s_class(canon, registry, cap=EXHAUST_CAP):
    """`(input types, output type, signature)` hashed, or (None, reason)."""
    sig, why = signature(canon, registry, cap)
    if sig is None:
        return None, why
    payload = {
        "inputs": [t.to_dict() for _, t in canon.inputs],
        "output": canon.nodes[-1].output.to_dict(),
        "sig": [x if x is ERR else x for x in sig],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24], None


# --------------------------------------------------- S*: carrier abstraction
def _abstract_type(t, width):
    """Every scalar integer carrier replaced by one placeholder of `width` bits."""
    if t.kind == "bool":
        return t
    if t.kind == "int" and t.encoding.kind == "integer":
        return integer(width, signed=t.encoding.signed, overflow=t.encoding.overflow)
    return None


def retype(canon, width):
    """Rebuild the fragment with every integer carrier at `width` bits.

    Returns (program, registry) or (None, reason).  Any fragment carrying a
    product, a set, a float or a frozen `module:` call cannot be re-typed and is
    reported as such -- a structural match only, never a semantic one.
    """
    reg = Registry()
    tmap = {}
    for _, t in canon.inputs:
        a = _abstract_type(t, width)
        if a is None:
            return None, f"input {t.kind} not abstractable"
        tmap[id(t)] = a
    inputs = tuple((k, _abstract_type(t, width)) for k, t in canon.inputs)
    port_t = {k: t for k, t in inputs}
    nodes = []
    for n in canon.nodes:
        cand = n.candidates[0]
        if cand.operator.name.startswith("module:"):
            return None, "frozen module call"
        try:
            ts = tuple(port_t[s] for s in cand.sources)
        except KeyError:
            return None, "unbound port"
        try:
            op = reg.resolve(cand.operator.name, ts, None, dict(cand.operator.parameters) or None)
        except Exception as e:
            return None, f"re-resolve failed: {type(e).__name__}"
        port_t[n.name] = op.output
        nodes.append(Node(n.name, op.output, (Candidate(op, cand.sources),),
                          n.region, n.depth, 0))
    try:
        p = Program(inputs, tuple(nodes), ((canon.outputs[0][0], canon.outputs[0][1]),)).validate(reg)
    except Exception as e:
        return None, f"revalidate failed: {type(e).__name__}"
    return (p, reg), None


def sstar_class(canon, widths=(4, 8), cap=EXHAUST_CAP):
    """Signature at two placeholder widths.  Admitted only if both are computed.

    The class key is the pair of signatures, so two fragments are S*-equal only
    when they agree at **both** widths -- one width certifies nothing (§53 arm F,
    §55).
    """
    sigs = []
    for w in widths:
        got, why = retype(canon, w)
        if got is None:
            return None, why
        p, reg = got
        sig, why = signature(p, reg, cap)
        if sig is None:
            return None, f"w={w}: {why}"
        sigs.append((w, [t.to_dict() for _, t in p.inputs],
                     p.nodes[-1].output.to_dict(), list(sig)))
    blob = json.dumps(sigs, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:24], None


# ------------------------------------------------------------- triviality
def is_trivial(canon, sig):
    """Pre-registered rule: one node, or a projection / constant / negated projection."""
    if len(canon.nodes) <= 1:
        return True, "single node"
    if sig is None:
        return False, "multi-node, signature unavailable"
    types = [t for _, t in canon.inputs]
    universes = [carrier(t)[0] for t in types]
    import itertools
    combos = list(itertools.product(*universes))
    if len(set(sig)) == 1:
        return True, "constant"
    for i, t in enumerate(types):
        proj = tuple(c[i] for c in combos)
        try:
            enc = tuple(Value.of(t, x).raw for x in proj)
        except Exception:
            continue
        if enc == sig:
            return True, f"projection x{i}"
        if t.kind == "bool" and tuple(not x for x in proj) == sig:
            return True, f"negated projection x{i}"
    return False, "non-trivial"


# ------------------------------------------------------- compact type labels
def tshort(t):
    """A short readable label for a `Type.to_dict()`."""
    if t["kind"] == "bool":
        return "BOOL"
    if t["kind"] == "int":
        e = t.get("encoding", {})
        if e.get("kind") == "float":
            return f"f{t['bits']}"
        lab = ("i" if e.get("signed") else "u") + str(t["bits"])
        if t.get("role"):
            lab += f"[{t['role']}]"
        if t.get("bounds"):
            lab += f"<={int(t['bounds'][1])}"
        return lab
    if t["kind"] == "tuple":
        inner = tshort(t["items"][0]) if t.get("items") else ""
        return f"({len(t.get('items', []))}x{inner})"
    return t["kind"]


def tag(t):
    """`label#sha` -- compact but exact: the sha is over the full `to_dict()`.

    Stored records use this instead of the full type dict.  A 3072-field product
    type serialises to ~250 kB, which would put the inventory over a hundred
    megabytes for no gain: the sha distinguishes any two distinct types exactly,
    and identity is computed in memory from the real `Type` objects.
    """
    d = t.to_dict() if hasattr(t, "to_dict") else t
    h = hashlib.sha256(json.dumps(d, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    return f"{tshort(d)}#{h}"
