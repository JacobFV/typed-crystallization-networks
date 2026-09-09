"""`local_fit(init_noise=0)` must be bit-identical to `tcn.synthesis.fit`."""
import sys, math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from tcn.synthesis import fit
from common import local_fit
from examples.mixed import problem

for steps, polish in ((120, 0), (300, 200)):
    p, sigs, ex = problem()
    torch.manual_seed(0); m1, r1 = fit(p, ex, sigs, steps=steps, freeze=False, polish=polish)
    p, sigs, ex = problem()
    torch.manual_seed(0); m2, r2 = local_fit(p, ex, sigs, steps=steps, freeze=False, polish=polish, init_noise=0.)
    same = (m1.selections() == m2.selections()
            and abs(r1['exact_max_error'] - r2['exact_max_error']) == 0.
            and abs(r1['relaxed_loss'] - r2['relaxed_loss']) < 1e-12)
    print(f"steps={steps} polish={polish}: identical={same} "
          f"err {r1['exact_max_error']:.3e} vs {r2['exact_max_error']:.3e} "
          f"loss {r1['relaxed_loss']:.8e} vs {r2['relaxed_loss']:.8e}")
