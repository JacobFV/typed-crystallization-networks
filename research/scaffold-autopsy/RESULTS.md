# Autopsy of the "generic arithmetic/sine scaffold" joint-training failure

Track 2. Branch `research/scaffold-autopsy`. Every number below was produced in
this worktree; the scripts and raw logs are in this directory (`README.md` indexes
them). Interpreter: `/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`.

## The claim under test

`docs/VALIDATION.md`, *Unsuccessful and unestablished results*:

> An earlier generic arithmetic/sine scaffold trained for 640 episodes stayed near
> chance: mean return **2.03125/4**, with final prediction loss around **0.2806**.
> [...] Replacing that scaffold with the typed logic experiment established a
> working joint-learning path; it was not evidence that arbitrary scaffolds learn
> equally well.

`artifacts/joint-long/` is gitignored and absent, so the run was rebuilt from the
description: `tcn.scaffold.arithmetic_scaffold` substituted for the hand-wired
logic graph in `examples/joint.py`, everything else identical — `logic` generator
with `{'depth':1,'table':6,'fixed_inputs':True}`, horizon 4, objectives
`invert=False/True`, `lr=.04`, Adam (`research/scaffold-autopsy/arith_joint.py`).

Reproduction, 640 episodes, seed 0:

| | recorded | reproduced |
|---|---|---|
| mean return | 2.03125 / 4 | **1.9625 / 4** |
| final prediction loss | ~0.2806 | **0.1826** (mean of final 8) |
| deterministic eval return | — | 2.75 / 4 (16 held-out episodes) |

Chance is 2/4. The failure reproduces.

## Root cause

**It is not the scaffold. It is the optimizer schedule: `JointTrainer.run` took
exactly one Adam step per episode, and at batch size one the policy-gradient term
has a signal-to-noise ratio of 0.22.** The policy saturated on noise inside the
first ~30 episodes, the policy gradient then collapsed 5×, and the shared
representation was never pulled toward the task relation afterwards.

### 1. The scaffold has no capacity or search problem (`capacity.py`)

The failing program is a small sin-activated network — the only operator choice
is `sin` vs `identity` at the eight activations. Take that *identical* program,
drop the RL entirely, and fit its `prediction` head on the same two targets
(`probes['gate'] = bit0 xor bit1`, `probes['target'] = gate xor invert`) by
full-batch supervision over all 32 input settings:

| hidden | lr | seed | final MSE | accuracy |
|---|---|---|---|---|
| 8 | .04 | 0 | 0.00001 | 1.000 |
| 8 | .04 | 1 | 0.00000 | 1.000 |
| 8 | .04 | 2 | 0.00000 | 1.000 |
| 8 | .01 | 0 | 0.00001 | 1.000 |

Convergence trace (hidden=8, lr=.04, seed 0):

```
step    0  loss 0.809417
step   60  loss 0.231845
step  120  loss 0.002385
step  180  loss 0.000121
step  600  loss 0.000003
```

**~120 clean optimizer steps** at the *same* learning rate solve the task exactly.
The failing run took **640** steps and never went below ≈0.16. Capacity,
initialization scale, the sin/identity mixture, and the BOOL→float `decode` path
are all exonerated: gradients flow and the optimum is reachable and reached.

### 2. Per-term gradient norms and objective conflict (`instrument.py`, `run_instrument.py`)

96 episodes from initialization, each objective term differentiated separately
against the shared trainable parameters:

| term | ‖grad‖, arithmetic scaffold | ‖grad‖, `examples/joint.py` |
|---|---|---|
| prediction | 2.047 | 0.073 |
| actor | 0.573 | 0.317 |
| value | 3.091 | 0.749 |
| entropy | 0.006 | 0.003 |
| crystal | 0.0005 | 0.0007 |
| mean pre-clip total | **3.819** (10.4% of episodes clipped at 5) | 0.875 (0% clipped) |
| mean max policy logit | **1.933** | 0.542 |

Pairwise gradient cosines on shared parameters:

| pair | arithmetic | `examples/joint.py` |
|---|---|---|
| prediction · actor | −0.006 | +0.106 |
| prediction · value | −0.084 | 0.000 (no shared parameters) |
| actor · value | +0.018 | 0.000 |

`JointTrainer` already computes and logs `prediction_policy_gradient_cosine`, so
AGENTS.md's reporting requirement is met by the code. Over the 640-episode run
that metric degrades monotonically:

```
episodes    0- 63   cos = +0.015
episodes  192-255   cos = -0.203
episodes  320-383   cos = -0.508
episodes  576-639   cos = -0.676
```

Conflict is real but it is a *consequence*, not the cause: it only becomes large
after ~300 episodes, long after the policy has already locked (see 4). At the
start the two objectives are simply orthogonal (−0.006) — the shared trunk carries
no task information for either of them to agree about.

### 3. The value baseline is fine — hypothesis 1 is refuted (`diagnose.py`)

`examples/joint.py`'s `value` node reads a trainable constant, so its baseline is
state-independent and shares no parameters with anything else (hence the exact
0.000 cosines above). The arithmetic scaffold instead gets a real linear value
head over the shared hidden layer. That head *works*:

```
ep   0  V=+1.295  G=1.262  advantage=-0.034
ep  64  V=+1.081  G=1.166  advantage=+0.085
ep 160  V=+0.902  G=1.110  advantage=+0.208
ep 288  V=+1.146  G=1.131  advantage=-0.015
```

V tracks the actual discounted return to within ~0.2 from the first block onward,
so advantage estimation is not broken and not systematically biased. The value
term is nevertheless the single largest gradient contributor (‖g‖ 3.09 vs 2.05 for
prediction) and it, not the task objectives, is what triggers the 5.0 gradient
clip in 10% of episodes.

### 4. What actually kills the run: policy saturation on noise (`diagnose.py`, `snr.py`)

Same 320-episode trace:

```
ep   0  |logit|max=1.28  ‖g actor‖=1.10  ‖Wh‖=3.19  max|Wh|=1.08  ‖Wpred‖=1.26
ep  64  |logit|max=2.48  ‖g actor‖=0.21  ‖Wh‖=3.13  max|Wh|=1.28  ‖Wpred‖=0.98
ep 192  |logit|max=2.62  ‖g actor‖=0.12  ‖Wh‖=3.13  max|Wh|=1.21  ‖Wpred‖=0.78
ep 288  |logit|max=2.45  ‖g actor‖=0.26  ‖Wh‖=3.67  max|Wh|=1.60  ‖Wpred‖=1.13
```

Three things happen at once:

- the policy logits reach ±2.5 within ~30 episodes (≈0.99 action probability) —
  the categorical entropy bonus at `entropy_weight=.01` is far too weak to hold it
  open, and Adam at `lr=.04` moves each policy weight ~0.04 *per episode*;
- the actor gradient consequently collapses by 5×, from 1.10 to ~0.2, because
  `∂ log p / ∂ logits ∝ (1 − p)`. Exploration is over;
- the hidden layer barely moves. `‖Wh‖` starts at 3.19 (the scaffold's
  `N(0, 1/sqrt(features))` init) and is still 3.67 after 288 episodes, and
  `‖Wpred‖` *shrinks* from 1.26 to 0.78 — the prediction head is collapsing onto
  its bias, i.e. onto the constant-0.5 predictor, which is exactly the MSE ≈0.25
  plateau in the record. Parity needs weights near π; they are near 1.3.

The reason a single episode cannot move the trunk usefully is measured directly.
For each term, over 64 fresh episodes, `SNR = ‖mean gᵢ‖ / mean ‖gᵢ‖`; `1/SNR²`
is the number of episodes that must be averaged before the mean direction
dominates the noise:

| term | ‖mean g‖ | mean ‖g‖ | SNR | episodes needed |
|---|---|---|---|---|
| prediction | 1.935 | 3.140 | 0.616 | 2.6 |
| **actor** | **0.279** | **1.266** | **0.221** | **20.5** |
| value | 4.218 | 4.977 | 0.848 | 1.4 |
| entropy | 0.0023 | 0.0030 | 0.753 | 1.8 |

The policy-gradient update at batch size one is ~78% noise. `JointTrainer.run`
offered no way to average it: the loop was hard-coded to one optimizer step per
episode.

Why `examples/joint.py` survives the same schedule: its policy is not learned by
REINFORCE at all. Its logits are `±(2z−1)` where `z` is the relaxed Boolean of the
`goal_relation` node, so once the *dense prediction loss* picks the right truth
table out of 16 discrete candidates — a categorical choice that accumulates
evidence robustly across noisy single episodes — the policy is correct by
construction. The generic scaffold has to discover a continuous readout with a
term that is four-fifths noise, and it saturates before the trunk has anything
worth reading.

## The patch

`tcn/training.py` only. It adds a declared number of episodes to average per
optimizer step; the default is 1, so every existing configuration, checkpoint and
test behaves exactly as before. No new operator, no type escape hatch, no loss
penalty, no domain branch.

```diff
--- a/tcn/training.py
+++ b/tcn/training.py
@@ -32,6 +32,7 @@ class TrainConfig:
     generator_config: dict
     objectives: tuple[dict,...] = ({},)
     episodes: int = 64
+    batch: int = 1
     horizon: int = 8
     dt: float = 1.
     lr: float = .01
@@ -51,6 +52,7 @@ class TrainConfig:
         if any(b.template>=len(self.action_templates) for b in self.action_bindings):raise ValueError('action binding index out of range')
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

Notes on the shape of the change:

- gradient clipping and the reported `gradient_norm` now apply to the averaged
  update rather than to a single episode, which is what the 5.0 threshold was
  presumably meant to bound;
- a batch always completes before `save`, so checkpoints never carry half of an
  accumulated gradient and `run` resumes correctly;
- `TrainConfig.from_dict` supplies `batch=1` for checkpoints written before the
  field existed.

*(pending: before/after table, regression checks, and the VALIDATION.md verdict)*
