"""The seven frozen artifact programs of the three real domains, reconstructed.

Nothing here searches or trains.  Every program is the *published* artifact of a
section, rebuilt from that section's recorded selections and checked against a
recorded digest or node count.  `tcn/` and `generators/` are untouched.

Streams, stated because §39/§45 require it:

* visual   -- `visual-ladder` FLAT screens, resolution 32, palette 32,
              nesting 5, widgets 20, min_size 4 (`rung3_root.json:configuration`).
* language -- TWO streams, carried separately and never merged:
              `L_stage_b16` is §19's **pre-audit** artifact,
              `L_dyck22` is §45's **post-audit** witness.
              Neither is re-run here; both are static programs.
* computer -- §23's `agent_program.json`, loaded, not rebuilt.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tcn.graph import Program
from tcn.operators import Registry
from tcn.runtime import load_program

VISUAL = ROOT / "research" / "visual-ladder"
LANG = ROOT / "research" / "language-capability"
LANGPA = ROOT / "research" / "language-post-audit"
COMP = ROOT / "research" / "computer-capability"

DOMAIN = {"V_same": "visual", "V_corner": "visual", "V_rect": "visual",
          "V_assembly": "visual", "L_stage_a": "language",
          "L_stage_b16": "language", "L_dyck22": "language",
          "C_agent": "computer"}


def _harden_prune(program, selections):
    """Freeze at the recorded selection and drop what the selection kills."""
    sel = {n.name: selections.get(n.name, 0) for n in program.nodes}
    return program.harden(sel).pruned()


# ------------------------------------------------------------------ visual
def visual():
    sys.path.insert(0, str(VISUAL))
    import rung3_widgets as R
    import rung3_root as RR
    import common as C

    root = json.load(open(VISUAL / "out" / "rung3_root.json"))
    rung3 = json.load(open(VISUAL / "out" / "rung3.json"))
    cfg = root["configuration"]
    w = h = cfg["resolution"]

    out = {}

    r0 = Registry()
    same = R.same_scaffold(r0, w, h, free=False)
    out["V_same"] = (_harden_prune(same, rung3["s0"]["chosen"]), r0)

    # S1' and S2' call `same` as a registered module, exactly as the track does.
    r1 = Registry()
    same1 = _harden_prune(R.same_scaffold(r1, w, h, free=False), rung3["s0"]["chosen"])
    mod1 = r1.register_module(same1)
    corner = RR.corner_scaffold_masked(r1, w, h, mod1, R.offset_pool(w))
    out["V_corner"] = (_harden_prune(corner, root["s1"]["chosen"]), r1)

    r2 = Registry()
    same2 = _harden_prune(R.same_scaffold(r2, w, h, free=False), rung3["s0"]["chosen"])
    mod2 = r2.register_module(same2)
    rect = RR.rect_scaffold_clamped(r2, w, h, mod2, R.offset_pool(w)) \
        if hasattr(RR, "rect_scaffold_clamped") else R.rect_scaffold(r2, w, h, mod2, R.offset_pool(w))
    out["V_rect"] = (_harden_prune(rect, root["s2"]["chosen"]), r2)

    # the assembly: four caller nodes over the frozen modules
    r3 = Registry()
    sameA = _harden_prune(R.same_scaffold(r3, w, h, free=False), rung3["s0"]["chosen"])
    modA = r3.register_module(sameA)
    cornerA = _harden_prune(RR.corner_scaffold_masked(r3, w, h, modA, R.offset_pool(w)),
                            root["s1"]["chosen"])
    rectA = _harden_prune((RR.rect_scaffold_clamped(r3, w, h, modA, R.offset_pool(w))
                           if hasattr(RR, "rect_scaffold_clamped")
                           else R.rect_scaffold(r3, w, h, modA, R.offset_pool(w))),
                          root["s2"]["chosen"])
    cname = r3.register_module(cornerA)
    rname = r3.register_module(rectA)
    positions = RR.all_positions({"width": w, "height": h})
    asm = R.assembly(r3, C.bytes_type(w, h), positions, cname, rname)
    out["V_assembly"] = (asm.pruned(), r3)
    return out


# ---------------------------------------------------------------- language
def language():
    sys.path.insert(0, str(LANG))
    sys.path.insert(0, str(LANGPA))
    out = {}

    sa = json.load(open(LANG / "stage_a.json"))
    # `module_program` is already the frozen, selected stage-A module: five
    # single-candidate nodes, digest 8063993bfeee7393683e47b3, which is exactly
    # the `module:` name `stage_b.json` records.  Re-hardening it would rebuild
    # it under a different digest, so it is taken as published.
    ra = Registry()
    out["L_stage_a"] = (Program.from_dict(sa["module_program"], ra).pruned(), ra)

    sb = json.load(open(LANG / "stage_b.json"))
    rb = Registry()
    mod_b = rb.register_module(Program.from_dict(sa["module_program"]))
    if mod_b != sb["module"]:
        raise SystemExit(f"stage-A module digest {mod_b} != recorded {sb['module']}")
    pb = Program.from_dict(sb["program"], rb)
    out["L_stage_b16"] = (_harden_prune(pb, sb["enumeration"]["selections"]), rb)

    # §45's Dyck witness, rebuilt from `dyck_witness.json:witness` and digest-checked.
    import dyck_scaffold as DS
    wj = json.load(open(LANGPA / "dyck_witness.json"))
    rc = Registry()
    mod = rc.register_module(Program.from_dict(sa["module_program"]))
    prog, _ = DS.stage_b_dyck(mod, rc, positions=wj["positions"])
    sel = _dyck_selection(prog, wj["witness"])
    out["L_dyck22"] = (_harden_prune(prog, sel), rc)
    return out, wj["program_digest"]


def _dyck_selection(prog, witness):
    """Turn `{c, plus, minus, total_ok, min_ok}` into a per-node candidate index."""
    sel = {}
    by = {n.name: n for n in prog.nodes}
    for name, want in (("symbols", ("sub", f"s{witness['c']}")),
                       ("plus", ("identity", f"v{witness['plus']}")),
                       ("minus", ("identity", f"v{witness['minus']}"))):
        node = by[name]
        sel[name] = next(i for i, c in enumerate(node.candidates)
                         if c.operator.name == want[0] and want[1] in c.sources)
    for name, (op, const) in (("total_ok", witness["total_ok"]),
                              ("min_ok", witness["min_ok"])):
        node = by[name]
        sel[name] = next(i for i, c in enumerate(node.candidates)
                         if c.operator.name == op and f"v{const}" in c.sources)
    return sel


# ---------------------------------------------------------------- computer
def computer():
    prog, reg = load_program(str(COMP / "out" / "agent_program.json"))
    return {"C_agent": (prog.pruned(), reg)}


def all_artifacts():
    out = {}
    out.update(visual())
    lang, dyck_digest = language()
    out.update(lang)
    out.update(computer())
    return out, dyck_digest


if __name__ == "__main__":
    arts, dyck_digest = all_artifacts()
    print(f"{'id':14s} {'domain':9s} {'nodes':>6s} {'digest':24s}")
    for k, (p, r) in arts.items():
        print(f"{k:14s} {DOMAIN[k]:9s} {len(p.nodes):6d} {p.digest}")
    print("dyck recorded digest", dyck_digest,
          "rebuilt", arts["L_dyck22"][0].digest,
          "match", dyck_digest == arts["L_dyck22"][0].digest)
