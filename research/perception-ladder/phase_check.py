"""The `phases` latent IS expressible: `atan2` wraps to [-pi, pi] by construction.

    w      = acos(c)                      (c from the recurrence, see rung1_signal)
    A sin(phi_t) = x[t]
    A cos(phi_t) = (x[t] cos w - x[t-1]) / sin w
    p      = atan2(A sin(phi_t) cos(w t) - A cos(phi_t) sin(w t),
                   A cos(phi_t) cos(w t) + A sin(phi_t) sin(w t))

`atan2` is invariant to a positive scale, so the amplitude (which the generator
does not expose as a latent) is never needed.  Every step uses only operators in
`tcn/operators.py`: div, mul, sub, sqrt, abs, atan2, sin, cos.
"""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from tcn.generation import Host
from common import dump

if __name__ == "__main__":
    rows = []
    for seed in range(40):
        h = Host.create("signal", seed=seed, configuration={"components": 1, "window": 4, "horizon": 32})
        for _ in range(8): h.step()
        view = h.view(); s = view.observations["samples"].decoded; t = view.observations["time"].decoded
        if min(abs(x) for x in s) < .05: continue
        c = (s[-1] + s[-3]) / (2 * s[-2]); w = math.acos(max(-1, min(1, c)))
        As = s[-1]; Ac = (s[-1] * math.cos(w) - s[-2]) / math.sin(w)
        ct, st = math.cos(w * t), math.sin(w * t)
        p = math.atan2(As * ct - Ac * st, Ac * ct + As * st)
        rows.append({"seed": seed, "predicted": p, "latent": h.state["phase"][0],
                     "error": abs(p - h.state["phase"][0])})
    worst = max(r["error"] for r in rows)
    dump("phase_reference", {"episodes": len(rows), "worst_error": worst, "rows": rows})
    print(f"phase reference program: {len(rows)} episodes, worst error {worst:.3e}, "
          f"exact (<=1e-3) on {sum(r['error'] <= 1e-3 for r in rows)}")
