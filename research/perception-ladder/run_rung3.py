"""Driver for rung 3 (geometry)."""
from __future__ import annotations
import sys, json, argparse, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.search import space_size, evaluate
from common import (run_seeds_local, summarise, enumerate_reference, random_reference,
                    dump, local_fit)
from rung3_geometry import (centre_program, mask_program, centre_signal, mask_signals,
                            allbg_signals, examples, image_type, BG, AND, OR)

RES = (1, 2, 3, 4, 6, 8)


def datasets(R, n_train=24, n_test=48, dense=False):
    return (examples(range(0, n_train), R, dense=dense),
            examples(range(9000, 9000 + n_test), R, dense=dense))


def ref_sel(R, free_address, pool):
    p = (R * R) // 2
    return {"bytes": 0, "hit_px0": 3 * p if free_address else 0,
            "hit_px1": 3 * p + 1 if free_address else 0,
            "hit_eq0": list(pool).index(BG[0]), "hit_eq1": list(pool).index(BG[1]), "hit": 0}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what")
    ap.add_argument("--seeds", type=int, default=12); ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--free-address", type=int, default=1)
    ap.add_argument("--pool", type=str, default="")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args(); seeds = list(range(a.seeds))

    if a.what == "surrogate":
        # How far can `eq`'s declared surrogate see on byte-valued inputs?
        rows = []
        for d in (0, 1, 2, 3, 5, 8, 10, 12, 16, 24, 32, 64, 128, 255):
            a_ = torch.tensor([[0.]], requires_grad=True); b_ = torch.tensor([[float(d)]])
            y = torch.exp(-((a_ - b_) ** 2).sum(-1, keepdim=True) / 1.)
            y.backward()
            rows.append({"delta": d, "surrogate": float(y), "d_surrogate_d_input": float(a_.grad)})
        dump("rung3_eq_surrogate", rows)
        for r_ in rows: print(f"  |a-b|={r_['delta']:4d}  eq_relaxed={r_['surrogate']:.3e}  grad={r_['d_surrogate_d_input']:.3e}")

    elif a.what == "width":
        out = {}
        for R in RES:
            tr, te = datasets(R)
            r = Registry(); p = centre_program(r, R, free_address=True, pool=(BG[0], BG[1]))
            n = space_size(p)
            enum = enumerate_reference(p, tr, centre_signal(), r, max_programs=1 << 22)
            if enum["solved"]:
                enum["held_out_error"] = evaluate(p.harden(enum["selections"]), {}, te, centre_signal(), r) \
                    if False else evaluate(p, enum["selections"], te, centre_signal(), r)
            rnd = random_reference(p, tr, centre_signal(), r, draws=min(100000, 20 * n))
            def build(seed, R=R, tr=tr, te=te):
                rr = Registry(); return centre_program(rr, R, free_address=True, pool=(BG[0], BG[1])), \
                    centre_signal(), tr, te, rr
            rows = run_seeds_local(build, seeds, steps=a.steps, freeze=False, polish=0,
                                   label=f"geometry R={R} ({n})")
            out[f"R{R}"] = {"resolution": R, "image_width": image_type(R).width,
                            "addresses": 3 * R * R, "space": n, "enumeration": enum,
                            "random": rnd, "rows": rows, "summary": summarise(rows)}
            print("R", R, "width", image_type(R).width, "space", n, summarise(rows),
                  "enum", (enum["solved"], round(enum["seconds"], 2), enum["unique"], enum.get("held_out_error")))
        dump("rung3_width", out)

    elif a.what == "colour":
        # addresses pinned, the background colour searched over the whole byte alphabet
        out = {}
        for R in (2, 4):
            tr, te = datasets(R)
            pool = tuple(range(256))
            r = Registry(); p = centre_program(r, R, free_address=False, pool=pool)
            n = space_size(p)
            enum = enumerate_reference(p, tr, centre_signal(), r, max_programs=1 << 22)
            def build(seed, R=R, tr=tr, te=te, pool=pool):
                rr = Registry(); return centre_program(rr, R, free_address=False, pool=pool), \
                    centre_signal(), tr, te, rr
            rows = run_seeds_local(build, seeds, steps=a.steps, freeze=False, polish=0,
                                   label=f"colour R={R} ({n})")
            out[f"R{R}"] = {"space": n, "enumeration": enum, "rows": rows, "summary": summarise(rows)}
            print("colour R", R, "space", n, summarise(rows), "enum",
                  (enum["solved"], round(enum["seconds"], 2), enum["unique"]))
        dump("rung3_colour", out)

    elif a.what == "dense":
        out = {}
        POOL = tuple(int(x) for x in a.pool.split(",")) if a.pool else (BG[0], BG[1])
        FREE = bool(a.free_address)
        for head, sigfn in (("mask", mask_signals), ("allbg", allbg_signals)):
            for R in (2, 3):
                tr, te = datasets(R, dense=True)
                base = sum(e["targets"]["allbg"].decoded for e in tr) / len(tr)
                for dense in (True, False):
                    def build(seed, R=R, head=head, dense=dense, tr=tr, te=te):
                        rr = Registry()
                        return (mask_program(rr, R, free_address=FREE, pool=POOL,
                                             and_menu=(AND, OR), head=head),
                                sigfn(R, dense=dense), tr, te, rr)
                    rows = run_seeds_local(build, seeds, steps=a.steps, freeze=False, polish=0,
                                           label=f"{head} R={R} dense={dense}")
                    # judge on the output signal alone
                    for row in rows:
                        if "selections" in row:
                            rr = Registry()
                            q = mask_program(rr, R, free_address=FREE, pool=POOL,
                                             and_menu=(AND, OR), head=head)
                            e_tr = evaluate(q, row["selections"], tr, sigfn(R, dense=False), rr)
                            e_te = evaluate(q, row["selections"], te, sigfn(R, dense=False), rr)
                            row["ok"] = e_tr is not None and e_tr <= 1e-3
                            row["train_err"] = e_tr if e_tr is not None else float("inf")
                            row["test_err"] = e_te if e_te is not None else float("inf")
                    rr = Registry()
                    q = mask_program(rr, R, free_address=FREE, pool=POOL, and_menu=(AND, OR), head=head)
                    out[f"{head}_R{R}_{'dense' if dense else 'output_only'}"] = {
                        "space": space_size(q), "positive_rate_allbg": base,
                        "rows": rows, "summary": summarise(rows)}
                    print(head, "R", R, "dense" if dense else "output_only", "space", space_size(q),
                          summarise(rows))
        dump("rung3_dense" + (("_" + a.tag) if a.tag else ""), out)
