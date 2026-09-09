"""Driver for rung 1 (signal)."""
from __future__ import annotations
import sys, json, argparse, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.operators import Registry
from tcn.search import space_size, evaluate
from common import (run_seeds_local, summarise, enumerate_reference, random_reference,
                    dump, local_fit, held_out_error)
from rung1_signal import (frequency_program, future_program, sample_examples, LADDER,
                          FREQ_SIGNAL, FUTURE_SIGNAL, F, WINDOW, ALL_FREE)

TRAIN, TEST = range(0, 60), range(1000, 1120)


def datasets(n_train=10, n_test=24):
    return sample_examples(TRAIN)[:n_train], sample_examples(TEST)[:n_test]


def reference_selection(program, free):
    sel = {n.name: 0 for n in program.nodes}
    for k, v in (("a", WINDOW - 1), ("b", WINDOW - 2), ("d", WINDOW - 3)):
        if k in free and k in sel: sel[k] = v
    return sel


def gradient_arm(builder, signals, seeds, steps, label, init_noise=.5, anneal=None, polish=0):
    tr, te = datasets()
    def build(seed):
        r = Registry(); return builder(r), signals, tr, te, r
    return run_seeds_local(build, seeds, steps=steps, freeze=False, label=label,
                           init_noise=init_noise, anneal=anneal, polish=polish)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("what")
    ap.add_argument("--seeds", type=int, default=12); ap.add_argument("--steps", type=int, default=600)
    a = ap.parse_args(); seeds = list(range(a.seeds)); tr, te = datasets()

    if a.what == "ladder":
        out = {}
        for name, free in LADDER.items():
            r = Registry(); p = frequency_program(r, free=free)
            n = space_size(p)
            enum = enumerate_reference(p, tr[:6], (FREQ_SIGNAL,), r) if n <= 1 << 21 else None
            if enum is not None:                       # confirm the winner generalises
                enum["held_out_error"] = held_out_error(p.harden(enum["selections"]), te, (FREQ_SIGNAL,), r) \
                    if enum["solved"] else None
            rows = gradient_arm(lambda rr, f=free: frequency_program(rr, free=f), (FREQ_SIGNAL,),
                                seeds, a.steps, f"freq {name} ({n})")
            rnd = random_reference(p, tr[:6], (FREQ_SIGNAL,), r, draws=min(200000, 20 * n))
            out[name] = {"space": n, "free": list(free), "enumeration": enum,
                         "random": rnd, "rows": rows, "summary": summarise(rows)}
            print(name, "space", n, summarise(rows), "enum",
                  None if enum is None else (enum["solved"], round(enum["seconds"], 2), enum["unique"]))
        dump("rung1_ladder", out)

    elif a.what == "future":
        out = {}
        arms = {
            "dense_freq+future": (dict(with_frequency_branch=True), (FUTURE_SIGNAL, FREQ_SIGNAL)),
            "output_only_same_space": (dict(with_frequency_branch=True), (FUTURE_SIGNAL,)),
            "output_only_no_branch": (dict(with_frequency_branch=False), (FUTURE_SIGNAL,)),
        }
        for name, (kw, sigs) in arms.items():
            for free_rec in (False, True):
                key = f"{name}{'' if free_rec else '_pinned_recurrence'}"
                r = Registry(); p = future_program(r, free_recurrence=free_rec, **kw)
                n = space_size(p)
                rows = gradient_arm(lambda rr, kw=kw, fr=free_rec: future_program(rr, free_recurrence=fr, **kw),
                                    sigs, seeds, a.steps, f"{key} ({n})")
                # judge both arms on the OUTPUT signal alone, so the dense arm is not
                # penalised for also having to satisfy its extra probe
                for row in rows:
                    if "selections" in row:
                        rr = Registry(); q = future_program(rr, free_recurrence=free_rec, **kw)
                        e_tr = evaluate(q, row["selections"], tr, (FUTURE_SIGNAL,), rr)
                        e_te = evaluate(q, row["selections"], te, (FUTURE_SIGNAL,), rr)
                        row["output_train_err"] = e_tr; row["output_test_err"] = e_te
                        row["ok"] = e_tr is not None and e_tr <= 1e-3
                        row["test_err"] = e_te if e_te is not None else float("inf")
                out[key] = {"space": n, "signals": [s.target for s in sigs],
                            "rows": rows, "summary": summarise(rows)}
                print(key, "space", n, summarise(rows))
        dump("rung1_future", out)

    elif a.what == "future-enum":
        out = {}
        for name, kw in {"no_branch_pinned": dict(with_frequency_branch=False, free_recurrence=False),
                         "no_branch_free": dict(with_frequency_branch=False, free_recurrence=True)}.items():
            r = Registry(); p = future_program(r, **kw); n = space_size(p)
            out[name] = {"space": n,
                         "enumeration": enumerate_reference(p, tr[:4], (FUTURE_SIGNAL,), r, max_programs=1 << 22),
                         "random": random_reference(p, tr[:4], (FUTURE_SIGNAL,), r, draws=200000)}
            print(name, n, out[name]["enumeration"]["solved"], out[name]["enumeration"]["seconds"],
                  out[name]["random"])
        dump("rung1_future_enum", out)

    elif a.what == "gap":
        # Is the relaxation tight?  Compare the relaxed loss the optimiser reaches
        # against the relaxed loss of the one-hot reference program.
        from tcn.learning import SoftProgram, tensor
        res = {}
        for name, free in LADDER.items():
            r = Registry(); p = frequency_program(r, free=free)
            model = SoftProgram(p, r)
            inputs = {"samples": torch.stack([tensor(e["inputs"]["samples"]) for e in tr])}
            targets = {"frequency": torch.stack([tensor(e["targets"]["frequency"]) for e in tr])}
            ref = reference_selection(p, free)
            with torch.no_grad():
                for node, logit in zip(p.nodes, model.choices):
                    logit.zero_(); logit[ref[node.name]] = 30.
                _, _, trc = model(inputs, return_trace=True)
                onehot = float(model.probe_loss(trc, targets, (FREQ_SIGNAL,)))
            best = []
            for seed in range(8):
                torch.manual_seed(seed)
                try:
                    m, rep = local_fit(p, tr, (FREQ_SIGNAL,), steps=a.steps, freeze=False, registry=r,
                                       polish=0, init_noise=.5, seed=seed)
                    best.append((rep["relaxed_loss"], rep["exact_max_error"]))
                except Exception as exc: best.append((float("nan"), repr(exc)[:60]))
            res[name] = {"onehot_reference_relaxed_loss": onehot, "optimiser": best}
            print(name, "one-hot relaxed loss", f"{onehot:.4g}", "| optimiser", 
                  [(round(x, 5) if isinstance(x, float) else x, y if isinstance(y, str) else round(y, 4)) for x, y in best])
        dump("rung1_relaxation_gap", res)
