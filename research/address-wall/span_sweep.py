"""Brackets the D1 temperature: is 2^bits a plateau or a lucky point?

The 144-program pinned-colour benchmark of `eq_temperature.py`, swept over
temperatures around `2^8 = 256`.  Written after the fact to make the two numbers
quoted in RESULTS section 4.4 reproducible; the results are in
`out/span_scaled_eq.json`.

Run:  .venv/bin/python research/address-wall/span_sweep.py
"""
from tcn.operators import Registry
from eq_temperature import build, data, tensors, synthesise, pointing_at, R
from instrument import BUDGET, dump

def main():
    train, test = data()
    r = Registry()
    p, sig = build(r, R)
    inp, tgt = tensors(p, train, sig)
    out = {}
    for tau in (64., 128., 256., 512.):
        rows = [synthesise(tau, p, r, inp, tgt, sig, s) for s in range(12)]
        pt = pointing_at(tau, p, r, inp, tgt, sig)
        out[str(tau)] = {"hits": sum(x["hit"] for x in rows), "n": 12,
                         "picks_reference": pt["picks_reference"], "rows": rows}
        print(f"tau={tau:7.0f} synthesis {out[str(tau)]['hits']}/12  "
              f"pointing {pt['picks_reference']:.3f}", flush=True)
    BUDGET.stop(); out["budget"] = BUDGET.to_dict()
    dump("span_scaled_eq", out)

if __name__ == "__main__":
    main()
