# Autopsy of the "generic arithmetic/sine scaffold" joint-training failure

Track 2. Branch `research/scaffold-autopsy`. Every number below was produced in
this worktree; scripts and raw logs are in this directory (`README.md` indexes
them). Interpreter:
`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`.

---

## Verdict

**Not a scaffold-quality effect, and not a bug in `tcn/`. The run was stopped
before its learning transition.**

The same scaffold, the same trainer, the same learning rate, the same task —
run for 5120 episodes instead of 640 — reaches **4/4** deterministic return on
held-out episodes with a held-out prediction loss of **4.4e-4**, on both seeds
tested. The transition happens near episode **750** (seed 0), **800** (seed 2)
and **2600** (seed 1). The recorded 640-episode run stopped before the earliest
of those.

**No patch to `tcn/` is required.** I implemented one (episode batching), measured
it, and rejected it on the evidence — see *The patch I tried and rejected*. The
minimal change that fixes the result is one number in the experiment
configuration: `episodes: 640 → 4096`. The representation is learned earlier than
that (prediction loss ≤9.4e-4 on all three seeds by 2048 episodes); the extra
budget is for the policy readout, which is the slow half.

`docs/VALIDATION.md` **needs correcting**, and is corrected on this branch.

---

## The claim under test

`docs/VALIDATION.md`, *Unsuccessful and unestablished results*:

> An earlier generic arithmetic/sine scaffold trained for 640 episodes stayed near
> chance: mean return **2.03125/4**, with final prediction loss around **0.2806**.
> [...] Replacing that scaffold with the typed logic experiment established a
> working joint-learning path; it was not evidence that arbitrary scaffolds learn
> equally well.

`artifacts/joint-long/` is gitignored and absent, so the run was rebuilt from the
description: `tcn.scaffold.arithmetic_scaffold` substituted for the hand-wired
logic graph of `examples/joint.py`, everything else identical — `logic` generator
with `{'depth':1,'table':6,'fixed_inputs':True}`, horizon 4, objectives
`invert=False/True`, `lr=.04`, Adam, `prediction`/`value`/`policy` heads of width
2/1/2 (`arith_joint.py`). 277 trainable parameters, 152 nodes.

Reproduction, 640 episodes, seed 0 (`repro.py`, `repro640.log`):

| | recorded | reproduced |
|---|---|---|
| mean return | 2.03125 / 4 | **1.9625 / 4** |
| final prediction loss | ~0.2806 | **0.1826** (mean of final 8) |
| deterministic eval return | — | 2.75 / 4 (16 held-out episodes) |

Chance is 2/4. The failure reproduces.

---

## Before / after

Same scaffold, same trainer, same `lr=.04`, same task. The only change is the
episode budget.

| configuration | prediction loss | deterministic eval return |
|---|---|---|
| **640 episodes (as recorded), seed 0** | 0.183 | **2.75 / 4** |
| 1024 episodes, seed 0 | 6.7e-4 | **4.00 / 4** |
| 1024 episodes, seed 1 | 0.161 | 2.50 / 4 |
| 1024 episodes, seed 2 | 1.9e-3 | **4.00 / 4** |
| 2048 episodes, seed 0 | 9.4e-4 | **4.00 / 4** |
| 2048 episodes, seed 1 | 1.8e-4 | 2.50 / 4 |
| 2048 episodes, seed 2 | 1.1e-4 | **4.00 / 4** |
| **5120 episodes, seed 0** | 5.9e-5 | **4.00 / 4** |
| **5120 episodes, seed 1** | 8.1e-4 | **4.00 / 4** |

The 2048-episode rows (`stock_2048.log`, via `repro.py`) were produced against
unmodified `tcn/` after the rejected patch was reverted; the 1024/5120 rows come
from `fixed.py` at `batch=1`, which is the identical code path.

Note the split at seed 1: by 2048 episodes its **prediction loss is already
1.8e-4** — the relation is learned — while its return is still 2.5/4, because the
policy readout is still pointing the wrong way and REINFORCE has to unwind
saturated logits to fix it. It reaches 4/4 by ~episode 2600. The representation is
the fast half; the policy readout is the slow half, exactly as the SNR numbers in
§4 predict.

Training curve, 5120 episodes, seed 0 (256-episode blocks; `fixed_ep5120_b1_s0.json`):

```
episode  pred_loss  return  value_loss  cos(pred,policy)  grad_norm
      0    0.30342   2.031       1.941            -0.095      3.361
    256    0.20648   1.984       2.011            -0.477      2.891
    512    0.13499   2.441       1.695            -0.407      2.303   <- record stops just here
    768    0.00322   3.965       0.775            +0.003      0.441
   1024    0.00102   4.000       0.728            +0.006      0.123
   3072    0.00012   4.000       0.688            -0.019      0.043
```

The 640-episode cutoff falls inside the plateau, one block before the transition.
Seed 2 transitions at ~episode 820, seed 1 at ~episode 2600.

---

## Why the transition is late — measured, not assumed

The scaffold is slower than the typed logic graph (which reaches 4/4 in 160
episodes) for reasons that can be measured. None of them is a defect in `tcn/`.

### 1. It is not capacity, initialization, or the encode/decode path (`capacity.py`)

The only operator choice in `arithmetic_scaffold` is `sin` vs `identity` at the
eight activations, so this is a small sin-activated network with typed plumbing.
Take the *identical* program, drop the RL entirely, and fit its `prediction` head
on the same two targets (`probes['gate'] = bit0 xor bit1`,
`probes['target'] = gate xor invert`) by full-batch supervision over all 32 input
settings:

| hidden | lr | seeds | final MSE | accuracy |
|---|---|---|---|---|
| 8 | .04 | 0,1,2 | 1e-5, 0, 0 | 1.000 |
| 8 | .01 | 0,1,2 | 1e-5, 0, 5e-5 | 1.000 |
| 32 | .04 | 0,1,2 | 0, 0, 0 | 1.000 |
| 32 | .01 | 0,1,2 | 1e-5, 1e-5, 1e-5 | 1.000 |

12 of 12 settings fit exactly. Convergence trace (hidden=8, lr=.04, seed 0):

```
step    0  loss 0.809417
step   60  loss 0.231845
step  120  loss 0.002385
step  180  loss 0.000121
step  600  loss 0.000003
```

**~120 clean optimizer steps** at the *same* learning rate solve it. Gradients
flow through the BOOL→float `decode` conversion and the sin/identity mixture
without saturation or detachment; the optimum is reachable and reached.

### 2. The value baseline is fine — hypothesis 1 is refuted (`diagnose.py`)

`examples/joint.py`'s `value` node reads a trainable constant: a state-independent
baseline sharing no parameters with anything else (its prediction·value gradient
cosine is exactly 0.000). The arithmetic scaffold gets a real linear value head on
the shared hidden layer, and that head works:

```
ep   0  V=+1.295  G=1.262  advantage=-0.034
ep  64  V=+1.081  G=1.166  advantage=+0.085
ep 160  V=+0.902  G=1.110  advantage=+0.208
ep 288  V=+1.146  G=1.131  advantage=-0.015
```

V tracks the actual discounted return to within ~0.2 from the first block onward.
Advantage estimation is neither broken nor systematically biased. The value term
is nevertheless the largest single gradient contributor and is what triggers the
5.0 gradient clip in 10% of early episodes.

### 3. Per-term gradient norms and objective conflict (`instrument.py`, `run_instrument.py`)

96 episodes from initialization, each weighted term differentiated separately
against the shared trainable parameters:

| term | ‖grad‖, arithmetic | ‖grad‖, `examples/joint.py` |
|---|---|---|
| prediction | 2.047 | 0.073 |
| actor | 0.573 | 0.317 |
| value | 3.091 | 0.749 |
| entropy | 0.006 | 0.003 |
| crystal | 0.0005 | 0.0007 |
| mean pre-clip total | **3.819** (10.4% clipped at 5) | 0.875 (0% clipped) |
| mean max policy logit | **1.933** | 0.542 |

Pairwise cosines on shared parameters:

| pair | arithmetic | `examples/joint.py` |
|---|---|---|
| prediction · actor | −0.006 | +0.106 |
| prediction · value | −0.084 | 0.000 (no shared parameters) |
| actor · value | +0.018 | 0.000 |

`JointTrainer` already computes and logs `prediction_policy_gradient_cosine`, so
AGENTS.md's conflict-reporting requirement is met by the existing code. Over the
640-episode run that metric degrades monotonically:

```
episodes    0- 63   cos = +0.015
episodes  192-255   cos = -0.203
episodes  320-383   cos = -0.508
episodes  576-639   cos = -0.676
```

This is real conflict, and it is the most striking number in the failing run — but
it is a **transient of the plateau, not the cause**. In the 5120-episode run the
same metric goes −0.095 → −0.477 → −0.407 → **+0.003** and stays near zero for the
remaining 4000 episodes. Once the shared trunk actually represents the relation,
the two objectives stop fighting on their own. Nothing needed to be added to
resolve it.

### 4. Low per-episode gradient signal-to-noise, and policy saturation (`snr.py`, `diagnose.py`)

`JointTrainer.run` takes one optimizer step per episode. For each term, over 64
episodes, `SNR = ‖mean gᵢ‖ / mean ‖gᵢ‖`; `1/SNR²` is how many episodes must be
averaged before the mean direction dominates the noise.

Arithmetic scaffold at initialization:

| term | ‖mean g‖ | mean ‖g‖ | SNR | episodes needed |
|---|---|---|---|---|
| prediction | 1.935 | 3.140 | 0.616 | 2.6 |
| **actor** | 0.279 | 1.266 | **0.221** | **20.5** |
| value | 4.218 | 4.977 | 0.848 | 1.4 |

After 200 training episodes — i.e. on the plateau — the supervised term degrades
too:

| term | ‖mean g‖ | mean ‖g‖ | SNR | episodes needed |
|---|---|---|---|---|
| prediction | 0.435 | 1.589 | **0.274** | 13.3 |
| actor | 0.059 | 0.164 | 0.359 | 7.8 |
| value | 0.569 | 2.474 | 0.230 | 18.9 |

A single-episode step on the plateau is ~73% noise for prediction and ~78% noise
for the policy at the start. Meanwhile the policy saturates on that noise:

```
ep   0  |logit|max=1.28  ‖g actor‖=1.10  ‖Wh‖=3.19  max|Wh|=1.08  ‖Wpred‖=1.26
ep  64  |logit|max=2.48  ‖g actor‖=0.21  ‖Wh‖=3.13  max|Wh|=1.28  ‖Wpred‖=0.98
ep 192  |logit|max=2.62  ‖g actor‖=0.12  ‖Wh‖=3.13  max|Wh|=1.21  ‖Wpred‖=0.78
ep 288  |logit|max=2.45  ‖g actor‖=0.26  ‖Wh‖=3.67  max|Wh|=1.60  ‖Wpred‖=1.13
```

Policy logits reach ±2.5 within ~30 episodes (≈0.99 action probability), the
actor gradient collapses 5× because `∂ log p/∂ logits ∝ (1−p)`, `‖Wpred‖` shrinks
toward the constant-0.5 predictor (MSE ≈0.25 — the plateau value in the record),
and the hidden weights inch along (parity needs weights near π; they are at 1.3).
The escape at ~episode 750 is the trunk finally accumulating enough coherent
signal to make the policy readout worth following.

### 5. Why `examples/joint.py` is ~10× faster on the same trainer

Not because its gradients are cleaner. Its actor SNR is 0.238 at initialization
and *falls to 0.044* after 40 episodes — noisier than the arithmetic scaffold's.
It wins for two structural reasons:

- its policy is not learned by REINFORCE at all. The logits are `±(2z−1)` where
  `z` is the relaxed Boolean of the `goal_relation` node, so once the dense
  prediction loss picks the right table the policy is correct by construction;
- what the prediction loss has to learn is a **discrete 16-way operator choice**,
  and categorical evidence accumulates robustly across noisy single episodes. Its
  prediction SNR *rises* with training (0.390 → 0.485), where the arithmetic
  scaffold's *falls* (0.616 → 0.274).

That is a genuine and interesting scaffold effect — it buys sample efficiency and
a shorter search. It is not what the recorded sentence says, and it does not mean
the generic scaffold fails to learn the task.

---

## The patch I tried and rejected

The SNR numbers say the trainer offers no way to average away a term that is 78%
noise, so I added one: a declared `TrainConfig.batch`, defaulting to 1, with
`run()` accumulating that many episodes per optimizer step, clipping and stepping
on the averaged gradient, and always completing a batch before a checkpoint.

It passed everything: `tests/test_training_curriculum.py` and
`tests/test_integration_complete.py`, 11 passed; and `examples/joint.py` at the
default `batch=1` reproduced the recorded validation numbers exactly —
prediction loss `0.24884 → 0.00223`, deterministic return `4.0000`.

**It does not help, so it is not on this branch.** At a fixed episode budget,
averaging buys update quality by spending optimizer steps, and the policy readout
is the part that needs steps:

| configuration | optimizer steps | held-out prediction loss | eval return |
|---|---|---|---|
| batch=1, 640 episodes | 640 | 0.183 | 2.75 / 4 |
| batch=8, 640 episodes (2 seeds) | 80 | 0.274 | 1.94 / 4 |
| batch=8, 5120 episodes (2 seeds) | 640 | **2.1e-4** | 2.69 / 4 |
| **batch=1, 5120 episodes (2 seeds)** | 5120 | 4.4e-4 | **4.00 / 4** |
| `examples/joint.py`, batch=8, 160 episodes | 20 | 0.206 | 2.00 / 4 |

Batching does exactly what the SNR analysis predicts for the supervised half —
`batch=8` drives the prediction loss to 2.1e-4 in 640 updates, where `batch=1`
needs roughly 3000 — and it starves the policy head in the process. Net effect at
equal episodes: worse. `tcn/training.py` is therefore left untouched. The rejected
diff is reproduced below for the record, and `batched.py` implements the same
accumulation against the unmodified trainer if anyone wants to re-measure it.

<details><summary>rejected diff against <code>tcn/training.py</code></summary>

```diff
@@ class TrainConfig:
     episodes: int = 64
+    batch: int = 1
     horizon: int = 8
@@ def __post_init__(self):
         if self.episodes<1 or self.horizon<1 or self.dt<=0:raise ValueError('invalid training budget')
+        if self.batch<1:raise ValueError('an update must average at least one episode')
@@
-    def episode(self,index,train=True,split='train',loss_only=False):
+    def episode(self,index,train=True,split='train',loss_only=False,step=True,scale=1):
@@
-            self.optimizer.zero_grad();loss.backward()
-            metrics['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.))
-            self.optimizer.step()
+            (loss/scale).backward()
+            if step:
+                metrics['gradient_norm']=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),5.))
+                self.optimizer.step();self.optimizer.zero_grad()
         return metrics,host
     def run(self,outdir=None):
+        """One update averages `batch` episodes; a batch always completes before a checkpoint."""
         torch.set_num_threads(1)
         if self.completed==0:torch.manual_seed(self.config.seed)
-        for i in range(self.completed,self.config.episodes):
-            metrics,_=self.episode(i);self.history.append(metrics);self.completed=i+1
-            if outdir is not None and (self.completed%8==0 or self.completed==self.config.episodes):self.save(outdir)
+        self.optimizer.zero_grad();saved=self.completed
+        while self.completed<self.config.episodes:
+            n=min(self.config.batch,self.config.episodes-self.completed)
+            for k in range(n):
+                metrics,_=self.episode(self.completed,step=k==n-1,scale=n);self.history.append(metrics);self.completed+=1
+            if outdir is not None and (self.completed-saved>=8 or self.completed==self.config.episodes):self.save(outdir);saved=self.completed
         return self.history
```
</details>

The one thing worth keeping from it: the `batch=8` rows above were produced with
that patch applied. Every `batch=1` row, including all of the *Before / after*
table, is the stock trainer (`stock_2048.log` re-runs the confirmation against
unmodified `tcn/` for provenance; the patched code is bit-identical at `batch=1`,
which the exact `examples/joint.py` reproduction confirms).

---

## Hypotheses from the brief, resolved

| # | hypothesis | verdict |
|---|---|---|
| 1 | broken/high-variance value baseline | **Refuted.** V tracks the return to ±0.2 from the first block; mean advantage ≈0. `examples/joint.py`'s constant baseline is the *decoupled* one, and it is the arithmetic scaffold that has the proper state-dependent head. |
| 2 | REINFORCE variance vs. the prediction term | **Confirmed as a slowing factor, not the cause.** actor SNR 0.221 at batch 1 (~20 episodes to average out); ‖g value‖ 3.09 > ‖g prediction‖ 2.05 > ‖g actor‖ 0.57; 10% of early episodes clipped, mostly by the value term. |
| 3 | prediction/policy gradient conflict | **Real and reported by the code already.** Cosine falls to −0.68 on the plateau — then returns to ~0 by itself once the trunk learns. A transient, not the cause. |
| 4 | BOOL encode/decode gradient path | **Refuted.** Full-batch supervision through the same path reaches MSE 1e-5 in 12/12 settings. |
| 5 | initialization / cancellation / dead sin units | **Refuted.** Same 12/12 result across 3 seeds and 2 widths; the trunk is simply slow to leave the near-linear regime, not stuck in it. |
| 6 | learning rate / episode budget | **This is the cause — the budget half of it.** `lr=.04` is fine (full-batch supervision converges at both .04 and .01). 640 episodes is short of every seed's transition. |

---

## Correction applied to `docs/VALIDATION.md`

The paragraph now records the reproduction, the transition episodes, the
full-batch capacity control, and states the scaffold effect accurately: it buys
sample efficiency and a shorter search, and the recorded episode budget — not the
scaffold — produced the near-chance number. The original 640-episode figures are
kept, since they are what that run measured.

## Limitations

- Two seeds at 5120 episodes, three at 2048 and three at 1024; seed 1's transition
  at ~2600 episodes shows the spread is wide, so "≥4096 episodes" is a floor read
  off three seeds, not a characterized distribution.
- The original `artifacts/joint-long/` is gone; the reproduction matches the
  recorded return to 0.07 and the prediction loss to within seed noise, but it is
  a reconstruction from the prose, not the original configuration file.
- This is one 4-bit depth-1 truth-table task with held-out episode addresses. It
  says nothing about the generic scaffold on harder relations, and the ~10×
  sample-efficiency gap in favour of the typed logic graph is real.
- The machine was heavily loaded throughout; wall-clock seconds in the JSON
  summaries are not a performance measurement.
