"""Driver for rung 2 (relations)."""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.search import space_size, evaluate
from common import (run_seeds_local, summarise, enumerate_reference, random_reference,
                    dump, held_out_error, accuracy)
from rung2_relations import (reachability_program, signals, examples, closure_wall,
                             ENTITIES, POOL4, POOL16, AND, OR, EDGES, EDGE)

ARMS = {
    "R0_fold_only":   dict(pool=POOL16, free_p=False, free_o=True),
    "R1_pool4":       dict(pool=POOL4,  free_p=True,  free_o=True),
    "R2_pool16":      dict(pool=POOL16, free_p=True,  free_o=True),
    "R3_free_wiring": dict(pool=POOL16, free_p=True,  free_o=True, free_mediators=True),
}


def reference(program, kw):
    sel = {}
    pool = kw["pool"]
    for n in program.nodes:
        if n.name.startswith("P"): sel[n.name] = pool.index(AND) if kw.get("free_p", True) else 0
        elif n.name.startswith("O"): sel[n.name] = pool.index(OR) if kw.get("free_o", True) else 0
        elif kw.get("free_mediators") and (n.name.startswith("l") or n.name.startswith("rr")):
            sel[n.name] = int(n.name.lstrip("lr"))
        else: sel[n.name] = 0
    return sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what")
    ap.add_argument("--seeds", type=int, default=12); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--ntrain", type=int, default=24)
    a = ap.parse_args(); seeds = list(range(a.seeds))
    tr = examples(range(0, a.ntrain), dense=True)
    te = examples(range(5000, 5000 + 64), dense=True)

    if a.what == "wall":
        dump("rung2_closure_wall", closure_wall()); print(json.dumps(closure_wall(), indent=1))

    elif a.what == "gradients":
        # Per-node gradient norms: does anything below a set operator learn?
        from tcn.learning import SoftProgram, tensor
        r = Registry(); p = reachability_program(r, **ARMS["R3_free_wiring"])
        m = SoftProgram(p, r)
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            for q in m.choices: q.add_(torch.randn(q.shape, generator=g) * .5)
        inputs = {k: torch.stack([tensor(e["inputs"][k]) for e in tr]) for k, _ in p.inputs}
        targets = {"target": torch.stack([tensor(e["targets"]["target"]) for e in tr])}
        _, _, trc = m(inputs, return_trace=True)
        loss = m.probe_loss(trc, targets, signals())
        loss.backward(); lv = float(loss.detach())
        norms = {n.name: (None if q.grad is None else float(q.grad.abs().sum()))
                 for n, q in zip(p.nodes, m.choices)}
        dump("rung2_gradients", {"loss": lv, "choice_grad_l1": norms,
                                 "regions": {n.name: n.region for n in p.nodes},
                                 "candidates": {n.name: len(n.candidates) for n in p.nodes},
                                 "note": "logits perturbed with sigma=0.5 to break the uniform truth-table symmetry"})
        for n in p.nodes:
            print(f"  {n.name:6s} region={n.region:6s} cands={len(n.candidates):3d} |grad|_1={norms[n.name]}")

    elif a.what == "arms":
        out = {}
        for name, kw in ARMS.items():
            r = Registry(); p = reachability_program(r, **kw); n = space_size(p)
            ref = reference(p, kw)
            ref_err = evaluate(p, ref, tr, signals(), r)
            enum = enumerate_reference(p, tr, signals(), r, max_programs=1 << 20) if n <= 1 << 20 else None
            rnd = random_reference(p, tr, signals(), r, draws=100000)
            res = {}
            for dense in (True, False):
                def build(seed, kw=kw, dense=dense):
                    rr = Registry()
                    return reachability_program(rr, **kw), signals(dense=dense), tr, te, rr
                rows = run_seeds_local(build, seeds, steps=a.steps, freeze=False,
                                       label=f"{name} dense={dense}", polish=0)
                # exact conformance is judged on the OUTPUT signal only, both arms
                for row in rows:
                    if "selections" in row:
                        rr = Registry(); q = reachability_program(rr, **kw)
                        row["output_train_err"] = evaluate(q, row["selections"], tr, signals(), rr)
                        row["output_test_err"] = evaluate(q, row["selections"], te, signals(), rr)
                        row["ok"] = row["output_train_err"] is not None and row["output_train_err"] <= 1e-3
                        row["test_err"] = row["output_test_err"] if row["output_test_err"] is not None else float("inf")
                        row["test_accuracy"] = accuracy(q, row["selections"], te, signals(), rr)
                acc = [r.get("test_accuracy", 0.) for r in rows]
                res["dense" if dense else "output_only"] = {"rows": rows, "summary": summarise(rows),
                                                            "mean_test_accuracy": sum(acc) / max(1, len(acc))}
                print(name, "dense" if dense else "output_only", summarise(rows),
                      "mean test acc", round(sum(acc) / max(1, len(acc)), 4))
            rr0 = Registry(); q0 = reachability_program(rr0, **kw)
            out[name] = {"space": n, "reference_train_error": ref_err,
                         "reference_test_accuracy": accuracy(q0, ref, te, signals(), rr0),
                         "enumeration": enum,
                         "random": rnd, **res}
            print(name, "space", n, "enum", None if enum is None else (enum["solved"], round(enum["seconds"], 2), enum["unique"]))
        dump("rung2_arms", out)
