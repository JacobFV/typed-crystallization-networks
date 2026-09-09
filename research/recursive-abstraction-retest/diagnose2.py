"""Two follow-ups to `diagnose.py`.

P4b  `diagnose.py`'s partial-credit table freezes nodes, which routes them
     through `exact_tensor` and hard-thresholds their inputs; the resulting
     3.45 loss is partly the BCE clamp on a confidently wrong hard prediction,
     not the landscape gradient descent actually traverses. Repeat it by
     *peaking the softmax* instead, which is what the optimizer does.

P7   How wide is the basin? `diagnose.py` showed that a +3.0 logit boost on the
     three correct choices converges and stays. Sweep the boost down to zero --
     zero is the experiment's own random initialization -- and find where the
     search stops being able to finish the job.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from tcn.operators import Registry
from tcn.learning import SoftProgram

import common as C
from diagnose import build, loss_of, correct_index

HERE = Path(__file__).parent


def main():
    out = {}
    r, prog, ex, inputs, targets = build("tight")
    n1, n2, y = prog.nodes
    i1 = correct_index(n1, ("a", "b", "c"))
    i2 = correct_index(n2, ("d", "e", "f"))
    iy = next(i for i, c in enumerate(y.candidates)
              if c.operator.name == "xor" and c.sources == ("n1", "n2"))
    idx = {"n1": i1, "n2": i2, "y": iy}

    # ---- P4b : partial credit with a peaked softmax, no freezing ---------
    def soft_loss(peaked, boost=8.0):
        m = SoftProgram(prog, r)
        with torch.no_grad():
            for k, node in zip(("n1", "n2", "y"), range(3)):
                if k in peaked:
                    m.choices[node][idx[k]] += boost
        return round(float(loss_of(m, inputs, targets).detach()), 5)

    out["P4b_partial_credit_soft"] = {
        "none": soft_loss(()),
        "n1": soft_loss(("n1",)),
        "y": soft_loss(("y",)),
        "n1_and_y": soft_loss(("n1", "y")),
        "n1_and_n2": soft_loss(("n1", "n2")),
        "n1_n2_y": soft_loss(("n1", "n2", "y")),
        "note": "softmax mass ~0.997 on each named choice; the rest stay uniform",
    }

    # ---- P7 : how wide is the basin? -------------------------------------
    rows = []
    for boost in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
        for seed in range(3):
            torch.manual_seed(seed)
            m = SoftProgram(prog, r)
            with torch.no_grad():
                for p in m.choices:
                    p.add_(torch.randn_like(p) * 0.5)
                m.choices[0][i1] += boost
                m.choices[1][i2] += boost
                m.choices[2][iy] += boost
            opt = torch.optim.Adam(m.parameters(), lr=0.05)
            first = None
            for step in range(150):
                opt.zero_grad()
                loss = loss_of(m, inputs, targets) + 0.001 * (step / 150) * m.entropy()
                loss.backward(); opt.step()
                if step % 10 == 0 or step == 149:
                    if C.conformant(m.export(), ex, C.COMP_SIGNALS, r):
                        first = step
                        break
            sel = m.selections()
            rows.append({"boost": boost, "seed": seed, "solved": first is not None,
                         "first_step": first,
                         "final_loss": round(float(loss_of(m, inputs, targets).detach()), 5),
                         "picked_module_at": [k for k, v in sel.items()
                                              if prog.nodes[[n.name for n in prog.nodes].index(k)]
                                              .candidates[v].operator.name.startswith("module:")]})
            print(rows[-1], flush=True)
    out["P7_basin_width"] = {
        "per_run": rows,
        "by_boost": {str(b): {"solved": sum(1 for x in rows if x["boost"] == b and x["solved"]),
                              "of": sum(1 for x in rows if x["boost"] == b),
                              "median_final_loss": statistics.median(
                                  [x["final_loss"] for x in rows if x["boost"] == b])}
                     for b in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)},
        "note": "boost is added to the logit of each of the three correct choices "
                "before training; boost 0.0 is the experiment's own initialization",
    }

    (HERE / "diagnose2.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
