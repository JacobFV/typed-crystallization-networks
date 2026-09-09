"""Driver for rung 4 (raster_text)."""
from __future__ import annotations
import sys, json, argparse, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.search import space_size, evaluate
from tcn.learning import SoftProgram, tensor
from common import (run_seeds_local, summarise, enumerate_reference, random_reference,
                    dump, local_fit, accuracy)
from rung4_raster import eq_program, threshold_program, examples, SIGNAL, W, H

THR_POOL = tuple(x + .5 for x in range(256))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what")
    ap.add_argument("--seeds", type=int, default=8); ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args(); seeds = list(range(a.seeds))
    tr, te = examples(range(0, 12)), examples(range(500, 524))
    print(f"{len(tr)} train / {len(te)} test examples, canvas {W}x{H}", flush=True)
    out = {}

    if a.what == "eq":
        r = Registry(); p = eq_program(r); n = space_size(p)
        enum = enumerate_reference(p, tr, SIGNAL, r, max_programs=1 << 22)
        def build(seed):
            rr = Registry(); return eq_program(rr), SIGNAL, tr, te, rr
        rows = run_seeds_local(build, seeds, steps=a.steps, freeze=False, polish=0, label=f"eq ({n})")
        for row in rows:
            if "selections" in row:
                rr = Registry(); row["test_accuracy"] = accuracy(eq_program(rr), row["selections"], te, SIGNAL, rr)
        dump("rung4_eq", {"space": n, "enumeration": enum, "rows": rows, "summary": summarise(rows)})
        print("eq route: space", n, "| enumeration solved:", enum["solved"], "exhausted:", enum["exhausted"],
              f"{enum['seconds']:.2f}s | gradient", summarise(rows), flush=True)

    elif a.what == "threshold":
        # (1) which parameters can a gradient even reach?
        r = Registry(); p = threshold_program(r); n = space_size(p)
        m = SoftProgram(p, r)
        inputs = {"pixels": torch.stack([tensor(e["inputs"]["pixels"]) for e in tr])}
        targets = {"is_b": torch.stack([tensor(e["targets"]["is_b"]) for e in tr])}
        _, _, trc = m(inputs, return_trace=True)
        loss = m.probe_loss(trc, targets, SIGNAL); loss.backward()
        out["choice_grad_l1"] = {nd.name: (None if q.grad is None else float(q.grad.abs().sum()))
                                 for nd, q in zip(p.nodes, m.choices)}
        out["threshold_grad_l1"] = {k: (None if v.grad is None else float(v.grad.abs().sum()))
                                    for k, v in m.constants.items()}
        print("choice-logit gradients:", out["choice_grad_l1"], flush=True)
        print("threshold gradient:", out["threshold_grad_l1"], flush=True)

        # (2) pure gradient, threshold trainable, address free.  Two initialisations.
        out["gradient"] = {}
        for init, lr in ((128., .05), (128., 2.), (70., .05)):
            def build(seed, init=init):
                rr = Registry(); return threshold_program(rr, init=init), SIGNAL, tr, te, rr
            rows = run_seeds_local(build, seeds, steps=a.steps, lr=lr, freeze=False, polish=200,
                                   label=f"threshold init={init} lr={lr}")
            for row in rows:
                if "selections" in row:
                    rr = Registry()
                    row["test_accuracy"] = accuracy(threshold_program(rr, init=init).harden(row["selections"]),
                                                    {}, te, SIGNAL, rr)
            out["gradient"][f"init{init}_lr{lr}"] = {"rows": rows, "summary": summarise(rows)}
            print(f"gradient init={init} lr={lr}:", summarise(rows), flush=True)

        # (3) address pinned to a byte that is known to separate: can gradient fit only thr?
        for init, lr in ((128., .05), (128., 2.)):
            def build(seed, init=init):
                rr = Registry(); return threshold_program(rr, free_address=False, address=180, init=init), \
                    SIGNAL, tr, te, rr
            rows = run_seeds_local(build, seeds, steps=a.steps, lr=lr, freeze=False, polish=200,
                                   label=f"thr-only init={init} lr={lr}")
            out["gradient"][f"pinned_address_init{init}_lr{lr}"] = {"rows": rows, "summary": summarise(rows)}
            print(f"threshold-only (address pinned) init={init} lr={lr}:", summarise(rows), flush=True)

        # (4) the discrete backend over the same program family
        r2 = Registry(); q = threshold_program(r2, thr_pool=THR_POOL); nq = space_size(q)
        enum = enumerate_reference(q, tr, SIGNAL, r2, max_programs=1 << 22)
        if enum["solved"]:
            enum["held_out_accuracy"] = accuracy(q, enum["selections"], te, SIGNAL, r2)
        rnd = random_reference(q, tr, SIGNAL, r2, draws=20000)
        out["enumeration_threshold_pool"] = {"space": nq, "enumeration": enum, "random": rnd}
        print("enumeration over address x comparison x 256 thresholds:", nq, enum["solved"],
              f"{enum['seconds']:.1f}s unique={enum['unique']} held-out acc={enum.get('held_out_accuracy')}",
              "| random", rnd, flush=True)
        dump("rung4_threshold", out)
