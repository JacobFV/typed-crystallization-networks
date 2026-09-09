# A task with genuine delayed credit assignment, and what can solve it

Research track `credit-assignment`, 2026-09-09. Everything below was produced in this
workspace with the repository `.venv` (Python 3.13.15, torch 2.14.0+cpu, Node 24.21.0,
aarch64, `torch.set_num_threads(1)`).

`tcn/` was **not modified**. `generators/computer/` gained one gated configuration
(`interface='panel'`) and one new engine file (`engine/session.ts`); both are verified
byte-identical to the pre-change behaviour on every default configuration, and the
verification is in section 2. Raw data is in `out/`; every table is regenerated from
it by `aggregate.py`. Environment episodes are counted by `panel.Counter`, which wraps
every `Host.create` and every `Host.step`, so no budget figure here is a configuration
number.

The checkout is shared with several agents and ran at load 50-85 on 20 cores
throughout, which is why wall-clock figures are noisy; episode counts are not.

---

## 0. Verdict

**Yes on both counts, and the second answer splits in two.**

**The task poses real credit assignment, and that is proved rather than asserted.**
The panel configuration of `generators/computer` has state that actions change (the
terminal, a register the agent cannot see, and termination), a reward that arrives
only on the terminating `commit` step, and an optimal policy whose first action --
`look` -- is never itself rewarded. Solved exactly by belief-state dynamic programming
over the task's own generative distribution: **V\*(6) = 1.0000** against a myopic
(gamma = 0) optimum of **0.0625**, a **16x** gap; the optimal first action becomes
`look` at **horizon 3** and its reward arrives **2 to 5 steps later**; the optimal
policy *is* the myopic one for any discount below **0.394**. Both identities that made
the `logic` generator's horizon vacuous fail here: `V*(H)` is **not** `H * V*(1)`
(1.0000 vs 0.375 at H = 6) and the myopic policy is **not** optimal at any horizon
above 1. I ran that detector against myself before running any learner; the task does
not decompose.

**It is reachable by exploration.** The narrow typed argument is the whole difference:
`write.text` is 1026 policy parameters scored by exact string equality and one neutral
write is rewardable with probability **5.65e-06**; `dial(value: int[4])` is **two**
parameters and one neutral commit is rewarded with probability **6.24e-02**, an
**11,043x** improvement, **28.9 episodes per reward**, about 45 seconds.

**Reward alone solves the delayed part.** With no supervision of any kind, plain
REINFORCE on the typed program recovers the exact situation-to-verb map --
`look` when nothing is showing, `dial` when the task record is showing, `commit` after
dialling -- on **6 / 6 seeds** in 600 episodes, none of those three actions being
rewarded on the step it is taken. Its return under its own sampling policy rises from
**0.06 to 0.59**, against a myopic reference measured at **PLACEHOLDER-MYOPIC** and an
exact oracle at **1.00**.

**Reward alone does not solve the argument.** It never selects the sweeping sub-action
`next_slot = add(previous slot, 1)`; on every seed it instead widens its own slot
sampler and searches slots at random, which caps it at the i.i.d.-slot ceiling of
**0.714** and, under deterministic evaluation -- which destroys that exploration --
reads **0.24**. The model-based route has no such problem: enumerating action
sequences against a frozen exact model, with the perception found by exhaustive
enumeration against the probe channel and **no policy at all**, reaches the exact
optimum **1.00 / 1 on 64 / 64 held-out episodes in 89 environment episodes**. It is
again the cheapest arm by an order of magnitude, exactly as in
`research/policy-learning/RESULTS.md`.

PLACEHOLDER-VERDICT-MACRO

---

## 1. The action hierarchy, and why it is not an oracle menu

The project lead's requirement is that an agent can *discover* meaningful sub-actions.
AGENTS.md forbids an oracle action menu as a model input. The design below has two
levels and neither is a menu.

### Level 0 -- primitives with a narrow typed argument

The shell interface's `write` takes `path` (a 129-byte text) and `text` (a 513-byte
text). `tcn/policy.py:parameter_width` charges a text `2 + 2*capacity` policy
parameters, so `write.text` alone is **1026 parameters** sampled independently and
scored by exact string equality: `research/computer-capability/RESULTS.md` measured
the probability that one neutral write is rewardable at **5.65e-06**, about 176,942
write actions, about 27 days.

The panel interface replaces that with what the type algebra already has --
`ARCHITECTURE.md` section 1's `int[n]`, and `tcn/policy.py:numeric_bounds`, which
already reads the declared bit width as the argument's range:

| verb | argument | values | policy parameters |
|---|---|---|---|
| `wait` | -- | 1 | 0 |
| `look` | `slot: int[2]`, wrapping | 4 | **2** (mean, log sigma) |
| `dial` | `value: int[4]` | 16 | **2** |
| `commit` | -- | 1 | 0 |

Four parameters against 1026, and no operator, no type and no sampler was added: an
`int[n]` argument goes down `sample_typed`'s existing numeric branch, which is why it
costs two parameters rather than `n` bits, and why a program can *compute* the value
it wants to emit -- a `role="category"` argument is sampled bit by bit and, being
outside `Type.numeric`, admits no arithmetic at all.

`SLOT` is declared `overflow='wrap'`, which makes it a 2-bit ring on which
`add(slot, 1)` is total. That one declaration is what makes a sub-action expressible.

### Level 1 -- composite actions as crystallized modules

A composite action here is **not** an extra verb the environment offers. It is a
frozen sub-program from the executed-action ports to a primitive action's typed
argument, registered with `Registry.register_module` and offered to the outer policy
as one candidate at one node. That is `ARCHITECTURE.md` section 4's recursive
abstraction applied to the action-emitting region instead of the perception region,
and it needs nothing new: `register_module` already turns a crystallized `Program`
into a callable operator with `gradient="none"`, so *choosing between* macros is an
ordinary candidate softmax while each macro's internals are a hard gradient boundary
-- exactly the abstraction boundary section 4 asks for.

The composite this task calls for is the panel sweep: `next_slot = add(previous slot, 1)`,
a one-node program `SLOT -> SLOT`. Section 8 builds it, installs it beside a wrong
sibling (`identity(previous slot)`, "stare at the same slot"), and measures whether
reward alone picks the right one.

### Why this is not an oracle action menu

1. **The environment's menu never changes.** `StepRecord.available_actions` is
   `('wait','look','dial','commit')` at every tick of every panel episode and
   `ActorView` carries nothing else (`tests/test_panel_interface.py`). An oracle menu
   is environment-side and *state-dependent*: it tells the agent which actions are
   useful here. The macro library is agent-side, fixed for the episode, and says
   nothing about the state.
2. **The library contains a wrong member and does not say which.** Section 8 measures
   both, and the wrong one is worse. A menu that has to be evaluated by acting is not
   an oracle.
3. **It is charged for.** `Program.description_bits` charges each distinct module
   definition once and every call site individually, so a macro that does not pay for
   itself loses under `L_program_description`. An oracle menu is free.
4. **Its content comes from crystallizing a program the agent ran.** Nothing outside
   the agent supplies the body. The one thing that *is* declared is the pool the
   sub-action is drawn from -- twelve type-legal candidates over the executed-slot
   port and two constants, of which two sweep -- and that is a declared candidate
   pool like every other scaffold in this repository, stated and enumerable.

---

## 2. What was added to `generators/computer`, and the proof nothing else moved

Two changes, both gated, both verified.

### 2.1 `interface='panel'`

`generators/computer/generator.py` gains three verbs in `action_schema`
(`look`, `dial`, `commit`), a `panel_setup` that draws the episode's hidden content on
its **own named random stream** (`address.rng('panel')`), a panel branch in `advance`
and `observe`, and three privileged probes. Everything is behind
`configuration['interface'] == 'panel'`.

One change was necessary to keep the default recorded stream fixed and is worth
naming: `observe` used to compute the shell menu as `tuple(self.action_schema)`, so
adding the three panel verbs to the schema would have silently grown every
pre-existing episode's `available_actions`. The menu is now the literal
`SHELL_MENU = ('wait','command','type','key','read','write')`, and the panel verbs are
refused outside the panel interface rather than merely unlisted.

**Verification.** `equivalence.py` loads the pre-change `generator.py` from
`generator_before.py` and the current one, drives both through the same addresses,
configurations, action sequences, objectives, splits and probe settings, and compares
every serialized `StepRecord` field for field.

PLACEHOLDER-EQUIV-TABLE

`tests/test_panel_interface.py` adds ten invariants: the two menus and their mutual
refusal, the narrow argument widths (2 and 2, against 1026 for `write.text`), that
reward is zero on every step before the commit and 1.0 on it, that the reward is read
back off the filesystem rather than off the register, that a myopic commit scores far
below the plan, that the panel probes are invisible to `actor_view`, replay and
restore, transport transparency, the separate random stream, and that an out-of-range
slot is an observable failed attempt rather than an error.

### 2.2 `engine/session.ts`, and the cost that made it necessary

`bridge.ts` boots a fresh kernel and replays the whole event log for **every**
transition. Measured here: 1.86 s per call regardless of log length, so a horizon-6
panel episode costs about 4 subprocess boots. A credit-assignment study is
arms x seeds x hundreds of episodes; at that price the whole study is weeks.

`session.ts` is a long-lived sibling: it reads the same requests as newline-delimited
JSON, keeps the runtime alive, and applies only the events that extend the log it has
already applied, rebuilding from scratch whenever the seed changes or the prefix does
not match -- which is exactly what `bridge.ts` does on every call. It is reached only
when `configuration['session']` is set. Measured: a horizon-6 panel episode costs
**0.48 s** through the session against **about 8 s** through `bridge.ts` (2.3 s to
initialize plus 1.9 s for each of the three kernel-changing actions), and **1.5 s**
including the training forward and backward passes at the load this study ran under.
That is what made the study possible at all.

**One real bug, found by the contract and fixed.** `bridge.ts` boots in a fresh
process, so the logical clock is always at the epoch when the runtime is constructed
and every boot-time stamp is t = 0. A rebuild inside a live session inherited the clock
from the previous request, which moved inode times, process start times, `uptimeMs` and
the first trajectory entry -- and `Host.replay()` caught it, because a replayed panel
episode diverged. `session.ts:boot` now resets the clock. The scope of the bug is on
record: a field-by-field diff of the two snapshots shows **only** timestamp and uptime
fields inside `state['result']`, and **no** field of any `StepRecord` -- no
observation, no probe, no latent, no reward -- so the learning arms that were in flight
when it was fixed measured exactly what they claim to.

PLACEHOLDER-SESSION-EQUIV

---

## 3. The task

`generators/computer`, `configuration = {'interface': 'panel', 'horizon': 6}`.

**Setup.** Four files `/home/agent/slot0.txt` .. `slot3.txt`. Exactly one, at an index
drawn uniformly per episode, holds `<key> = <digit>` with a key beginning with `t`
(`tmp`, `task`, `tally`, `target`) and a digit uniform on 0..8. The other three hold
`<key> = <digit>` with a key that does not begin with `t` (`note`, `memo`, `log`,
`data`, `ref`, `aux`). A hardware register, which the agent cannot observe, starts at
a value drawn uniformly on 0..15.

**Actions.** `look(slot)` reads that slot's record into the terminal. `dial(value)`
sets the register and touches nothing else. `commit` writes `str(register)` to
`/home/agent/out.txt` **and ends the episode**. `wait` does nothing.

**Reward.** One component, `goal`. On the commit step the generator reads
`/home/agent/out.txt` back off the filesystem and pays `1.0` iff it holds
`str(task digit + 1)`. Every other step pays `0.0`. There is no other reward and no
shaping.

**Observations.** `terminal` (`text[4096]`, `role="byte"`) and `pixels`, exactly as the
shell interface. At tick 0 the terminal shows the last setup write's JSON,
`{"path": "/home/agent/slot3.txt", "bytes": 8}` -- a length, not a content, so the
digit is not visible until the agent looks. The register is never visible.

**Privileged channels** (probes and latents only, unreachable from `actor_view`):
`answer` (the digit + 1), `task_slot`, `showing_task`, `register`, and the file
contents. These exist so a *staged* arm and an *enumerative* arm are expressible; the
reward-only arms use none of them.

**What the agent has to find.** Three things, and only the first is perception:

* the digit is the last byte of a record whose length varies (6 to 9), so its address
  must be computed, not constant -- the same shape as the shell task's `sub(length,1)`;
* `terminal[0] == 't'` is an exact predicate for "the panel is showing the task record";
* **when** to look, when to dial and when to commit, and **which slot to look at next**.
  Neither is ever rewarded on the step it is taken.

---

## 4. Does it pose credit assignment? Exactly, before anything is trained

`analysis.py` solves the task's belief-state MDP exactly, in rationals, over its own
generative distribution (task slot uniform over 4, digit uniform over 0..8, initial
register uniform over 0..15). Nothing here is sampled.

### 4.1 The myopic policy is strictly worse, at every horizon above 1

`V*` is the optimal undiscounted return. `myopic` is the optimum at `gamma = 0`, i.e.
the policy that maximises immediate reward. `bandit` is what a task that decomposed
into independent per-step bandits would give, `H * V*(1)`.

| H | V* | optimal first action | myopic (gamma=0) | ratio | bandit prediction |
|---|---|---|---|---|---|
| 1 | 0.0625 | commit | 0.0625 | 1.00 | 0.0625 |
| 2 | 0.1111 | **dial** | 0.0625 | 1.78 | 0.125 |
| 3 | 0.3333 | **look** | 0.0625 | 5.33 | 0.1875 |
| 4 | 0.5556 | **look** | 0.0625 | 8.89 | 0.25 |
| 5 | 0.7778 | **look** | 0.0625 | 12.44 | 0.3125 |
| 6 | **1.0000** | **look** | 0.0625 | **16.00** | 0.375 |
| 7 | 1.0000 | dial/look/wait (tied) | 0.0625 | 16.00 | 0.4375 |
| 8 | 1.0000 | dial/look/wait (tied) | 0.0625 | 16.00 | 0.5 |

* **The gap appears at H = 2** and the *credit-assignment* gap -- the one that requires
  an action which is never itself rewarded -- **appears at H = 3**, where the optimal
  first action becomes `look`.
* **The optimal first action depends on a reward arriving k steps later.** `look` pays
  nothing, ever. Under the optimal policy the reward lands on the commit step, which
  is the (number of looks + 2)-th tick; the number of looks is uniform on 1..4, so the
  first action's credit is delayed by **3 to 6 steps**, and the look that actually
  reveals the digit is credited **2 steps** later.
* The myopic value is **flat in the horizon**. A myopic agent commits at tick 0 and
  collects 1/16 whatever the horizon is.

### 4.2 It is not a bandit, and the detector says so

`bandit_check` tests the two identities that hold on `generators/logic` -- where
`research/policy-learning/RESULTS.md` found the previous horizon results to be
vacuous -- and both fail here:

| identity | `logic` | panel |
|---|---|---|
| `V*(H) = H * V*(1)` (independent repetitions of one bandit) | true | **false** (1.00 vs 0.375 at H=6) |
| the myopic policy is optimal | true | **false** at every H > 1 |
| state changes with actions | no | yes: the terminal, the register, and termination |
| reward on the step the action is taken | yes | **no**: only on the commit step |

This is the check the brief asked me to run against myself. The task does not
decompose.

### 4.3 The discount at which credit assignment bites

Sweeping `gamma` and re-solving: the optimal first action is `commit` for
`gamma < 0.3940` and `look` for `gamma > 0.3940` (bisection to 1e-18, H = 6). Below
that discount the exact optimal policy *is* the myopic one. This is the sharpest
statement the task supports: a learner that discounts harder than 0.394 cannot want
the right first action, whatever its estimator.

### 4.4 Delay is intrinsic; the myopic *trap* is a dial

The initial register is a stated hand-initialisation (`panel_setup`'s docstring names
it). It is what makes `commit` pay 1/16 immediately and so makes the myopic optimum
strictly *wrong* rather than merely uninformed. With `panel_register=0` -- the answer
is always 1..9, so a zero register is never right -- the same DP gives:

| H | V* | optimal first | myopic |
|---|---|---|---|
| 1 | 0.0000 | (all tied) | 0.0000 |
| 2 | 0.1111 | dial | 0.0000 |
| 3 | 0.3333 | **look** | 0.0000 |
| 6 | **1.0000** | **look** | 0.0000 |

So the **delay is intrinsic to the task** -- the reward still arrives strictly after
the actions that earn it, and a gamma = 0 learner still gets nothing -- while the
*size* of the myopic baseline is a dial. Both configurations are shipped.

---

## 5. Exploration probability

The number the computer track computed for the shell interface was **5.65e-06**: the
probability that one neutral `write` is rewardable. The same quantity here, computed
the same way and exactly rather than by sampling (`analysis.exploration`, a closed-form
forward chain over the neutral sampler's own distribution):

| quantity | panel | shell `write.text` |
|---|---|---|
| policy parameters in the rewarded argument | **2** | 1026 |
| P(one neutral commit is rewarded) | **6.239e-02** | 5.65e-06 |
| ratio | **11,043x** | 1 |
| P(a neutral episode at H=6 is rewarded) | **3.457e-02** | -- |
| episodes per reward | **28.9** | 176,942 write *actions* |
| wall clock to the first reward | **about 45 s** at the measured 1.5 s/episode | about **27 days** |

The neutral policy is uniform over the four templates with `slot` and `value` drawn by
`tcn/policy.py:sample_typed` from zero parameters, which is what a freshly initialised
typed policy actually emits: P(slot) = (0.2105, 0.2895, 0.2895, 0.2105) and P(value)
is the tanh-squashed normal over 0..15. Its exact reward probability per horizon:

| H | P(episode rewarded) | episodes per reward |
|---|---|---|
| 3 | 2.935e-02 | 34.1 |
| 4 | 3.213e-02 | 31.1 |
| 6 | 3.457e-02 | 28.9 |
| 8 | 3.534e-02 | 28.3 |

**One caveat, stated because it is the interesting one.** Most of that reward is
*luck*: a neutral policy that never looks still commits a randomly dialled register and
is right 1 time in 16. The probability that a neutral episode is rewarded *after having
seen the task record* is 6.9e-04. So the reward is dense enough to give the estimator a
signal at a sane budget -- which is the thing `write.text` could not do -- but the
signal that distinguishes "look, then dial what you read" from "dial anything" has to
be extracted from an advantage difference, not from a first success. Section 7 is where
that bites.

---

## 6. Baselines, on the real generator

All on held-out episode indices 10000+, `split='test'`, horizon 6, on the real
generator through `panel.Counter`. `oracle` reads `Host.state['panel']` -- it is a
reference, not a model; the typed programs in section 7 see `terminal` and the
executed-action ports and nothing else.

PLACEHOLDER-REFS-TABLE

The exact values from section 4 for comparison: optimal 1.0000, myopic 0.0625,
blind dial-then-commit 1/9 = 0.1111. The measured references land on them.

**Two ceilings that matter for reading section 7.** The scaffold's `look` argument is
one number computed from the executed-slot port, and three policy classes are
separated by which number it is (`analysis.look_ceilings`, exact):

| policy class | value at H=6 |
|---|---|
| sweeps to a different unseen slot each look | **1.0000** |
| draws slots i.i.d. from the neutral sampler (a wide argument sampler) | 0.7141 |
| looks at the same slot every time (any non-sweeping program, deterministically evaluated) | **0.3333** |

A program that has learned everything about *when* to act but nothing about *which
slot* is capped at 0.3333. That number is the one to hold against the reward-only arms.

---

## 7. Learning

### 7.1 What is declared and what each arm has to find

The scaffold is `program.py`; its docstring names every declaration. In the arms whose
subject is credit assignment the perception chain is **declared** (one candidate per
node) and only the policy is searched. That is deliberate and it is the only way to
isolate delay: `research/computer-capability/RESULTS.md` established that the byte
address on this observation receives `grad is None`, because `unpack` -- the only
declared exit from `role="byte"` -- is `gradient="none"`. Leaving it searched under
reward re-measures a known gradient boundary, not credit assignment. The
`reward_percept` arm leaves it searched anyway, so the boundary is re-measured rather
than assumed.

What reward has to supply, in every reward-only arm:

* **three logit vectors**, one per situation (`nothing showing`, `task record showing`,
  `just dialled`), each zero-initialised, so the policy starts uniform over the four
  templates. This is where the delayed credit lives: `look` and `dial` are never
  rewarded on the step they are taken.
* **`next_slot`**, a 12-candidate choice over the executed-slot port and two constants.
  Two candidates sweep the panel; the rest stare at one slot or repeat the last.
* **`logstd_slot`**, the width of the slot sampler, trainable from 0. The dial's width
  is fixed at `exp(-5)` because the dial value is *computed* from the terminal and
  nothing about it is explored; that is a stated hand-setting and section 8's
  `dial_explores` sibling is the ablation.

### 7.2 Reward only, and the myopic control

PLACEHOLDER-ARM-TABLES

### 7.3 The model-based route: enumerate against a frozen exact model, learn no policy

The cheapest approach in `research/policy-learning/RESULTS.md` (27 environment
episodes, against 400 for reward-only REINFORCE) is the cheapest here too, and it is
the only arm that reaches the exact optimum.

**Stage 1, perception by enumeration** (`perception.py`). 25 recorded episodes driven
through one fixed sweep, targets taken from `StepRecord.probes` only:

| search | space | evaluated | conforming | selected | held-out max error |
|---|---|---|---|---|---|
| `terminal -> answer` (address x arithmetic) | **2700** | 2700 (exhausted) | 2 | `pos = sub(length, 1)`, `shift = sub(code, 47)` | **0.0** |
| `terminal -> showing task` (address x constant) | **168** | 168 (exhausted) | 10 | `qpos = 0`, `brand = eq(byte, 't')` | **0.0** |

Both spaces are exhausted, so this is a certificate rather than an argument: the
conforming programs are enumerated exactly, and the address the search returns is the
*computed* one over fifteen constant alternatives -- the same result the shell task
got, at a narrower action interface. The two conforming transform programs are the two
spellings of the same constant (`one_len` and `k0` are both 1); the ten conforming
predicates are the ten spellings of address 0.

**Stage 2, planning by enumeration** (`mpc.py`). No policy, no trained parameter. At
each tick the planner enumerates **every** action sequence to the end of the episode
over four abstract actions (`wait | look at the lowest-index unseen slot | dial the
current best answer | commit`), scores each by expectation under a *declared* exact
transition and reward model, and executes the first action of the best. The reduction
from "look at slot k" to "look at an unseen slot" is exact because the model is
symmetric in unseen slots. 349,440 sequences were evaluated over the 64 held-out
episodes.

| | |
|---|---|
| held-out return | **1.0000 / 1**, **64 / 64 solved** |
| modal trace | `look, look, look, look, dial, commit` |
| environment episodes, **total** | **89** (25 perception + 64 evaluation) |
| trained parameters | **0** |

This is the exact optimum of section 4.1, reached with no policy at all.

### 7.4 Cost, side by side

PLACEHOLDER-COST-TABLE

---

## 8. Composite actions from crystallized modules

`macro.py`. Two sub-actions are crystallized as one-node programs `SLOT -> SLOT` and
registered with `Registry.register_module`:

* **`sweep`** = `add(previous slot, 1)` -- the sub-action that solves the panel;
* **`stare`** = `identity(previous slot)` -- the one that does not.

They become the only two candidates of `next_slot`. Because `register_module` gives a
frozen module `gradient="none"`, the *choice between* them is an ordinary candidate
softmax while each body is a hard gradient boundary: this is section 4's abstraction
applied to actions, with nothing added to the algebra.

**The description-length prior points the wrong way, which is the sharpest evidence
that this is not an oracle menu.** `Program.description_bits` on the hardened, pruned
program:

| program | description bits | execution cost |
|---|---|---|
| `next_slot` = call `stare` | 29,930,880 | 46.0 |
| `next_slot` = call `sweep` | 29,931,832 | 46.0 |
| `next_slot` = `add(slot, 1)` inlined, no module | **29,923,704** | 46.0 |

At one call site a module costs **more** than inlining its body -- exactly what
`Program.description_bits` is built to say, since it charges the definition once *and*
every call site -- and the wrong sub-action is **952 bits cheaper** than the right one.
So `L_program_description` prefers no macro at all, and among macros it prefers the
useless one. Nothing but the delayed reward can separate them.

PLACEHOLDER-MACRO-TABLE

---

## 9. Core changes required, as diffs

Nothing under `tcn/` was changed. Two changes are required for someone else to
reproduce this work inside the shipped trainer, and one is a bug report.

### 9.1 `TrainConfig` cannot express a reward-only, a supervision-only or a gamma = 0 arm

This is the blocking defect `research/policy-learning/RESULTS.md` already reported, and
it is unchanged. `run.py` re-implements `JointTrainer.episode` for exactly this reason.

```diff
--- a/tcn/training.py
+++ b/tcn/training.py
@@ class TrainConfig:
     def __post_init__(self):
         if not self.objectives or not self.action_templates:raise ValueError('goals and actions must be nonempty')
-        if self.prediction_weight<=0 or self.policy_weight<=0:raise ValueError('integrated training requires prediction and policy objectives')
+        # A weight of zero is a *declared* ablation, not a malformed configuration:
+        # a reward-only arm, a supervision-only arm and a gamma = 0 myopic arm are
+        # all objectives this repository has now measured, and none of them is
+        # expressible while a zero weight is rejected. Reject negatives, and require
+        # that at least one objective is active.
+        for name in ('prediction_weight','probe_weight','policy_weight','value_weight','entropy_weight','mdl_weight','description_weight','crystal_weight'):
+            if getattr(self,name)<0:raise ValueError('loss weights must be non-negative')
+        if max(self.prediction_weight,self.probe_weight,self.policy_weight)<=0:raise ValueError('training requires at least one active objective')
         if len({(b.template,b.argument) for b in self.action_bindings})!=len(self.action_bindings):raise ValueError('duplicate action binding')
```

`discount` is already a free field, so no change is needed for the myopic arm beyond
this one.

### 9.2 `relaxed`'s `index` and `gt` are unusably wide at the default temperature

Not a proposed diff -- a finding that anyone building a byte-perception scaffold needs.
At `temperature = 1`, `relaxed`'s `index` is
`softmax(-(address - position)^2 / tau)`, so "the byte at `length - 1`" comes back as a
blend of three neighbouring bytes and **no `eq` against it can fire**. The reference
program -- the one whose choices are exactly right -- scored **0.00 / 1** until the
temperature on the two `index` nodes and the two threshold nodes was set to 0.02, and
**1.00 / 1** after. The program's *choices* were correct in both cases; its relaxed
*perception* was not.

Sharpening is what `ARCHITECTURE.md` section 5's anneal already does, and
`SoftProgram.temperatures` is already per-node, so no core change is needed. What is
missing is that nothing warns you: a scaffold can be exactly right and score at chance,
and the failure looks like a policy failure. `run.py:SHARP` names every node it
touches, and none of them is a searched choice.

### 9.3 `torch.remainder` is the wrong wrap for a relaxed ring

Also a finding rather than a diff. `SLOT` is declared `overflow='wrap'`, so
`add(slot, 1)` wraps in exact execution; the relaxed mixture does not, and the obvious
repair -- a `mod` node -- is wrong, because `torch.remainder` is discontinuous at the
modulus: a mixture that has sharpened to `4 - 1e-8` comes back as `3.99999999` rather
than `0`, and the sweep silently sticks on one slot. The scaffold writes the wrap as
`x - 4 if x > 3.5 else x` out of `gt`, `sub` and `mux` instead. Anyone relaxing a
modular type will hit this.

---

## 10. Limitations

* **One task, one generator.** Everything here is the panel configuration of
  `generators/computer`. The exact DP, the exploration probability and the
  bandit-decomposition detector are properties of this task; they say nothing about
  whether another delayed task in this substrate would behave the same way.
* **The perception is declared in the reward-only arms.** That is deliberate, and it
  is the only way to isolate credit assignment: `research/computer-capability`
  established that the byte address gets `grad is None` because `unpack` is
  `gradient="none"`, so leaving it searched in a reward-only arm measures a known
  gradient boundary rather than delay. The `reward_percept` arm leaves it searched
  anyway and reports what happens.
* **The `SHARP` temperatures are a hand-setting**, named in `run.py` and in section
  9.2, and they are load-bearing: without them the reference program itself scores
  0.00. They are not a task hint -- they change no choice and no candidate -- but a
  reader should know the arms would all read as failures without them.
* **The initial register is a dial.** Section 4.4 separates what it does (the size of
  the myopic baseline) from what it does not (the delay, which is intrinsic).
* **The machine was shared.** Load ran 50-95 on 20 cores for the whole study, so every
  wall-clock number is an upper bound and the per-episode costs in section 2.2 are
  measured, not budgeted.
* **Seeds.** The two headline arms have 6 and 4 seeds and the ablations have 3; the
  policy-learning track used 8. The reduction is a compute decision under the load
  above, and the seed-level numbers are in `out/arm_*.json` rather than only their
  means.
