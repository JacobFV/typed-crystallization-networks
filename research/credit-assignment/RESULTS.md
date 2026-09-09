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

> **STATUS: substantially complete, with two gaps, both named.** The session ended and
> killed (a) the `staged_enum` arm's three seeds mid-run and (b) a re-run of the
> baselines at n = 192. Section 10 lists the exact commands.
>
> **UPDATE 2026-09-09, a later session: the composite-action arm has now been run.**
> `bash research/credit-assignment/run_macro.sh`, six seed-runs, results in section 8.1
> and in `out/macro_*.json`; that subsection and the two paragraphs marked **NEW** in
> section 0 are the only newly measured numbers in this report. Everything
> else below is measured as it stands, and every number in it comes from a file in `out/`: the
> exact analysis, the 384-combination equivalence check, the 64-combination session
> check, the n = 48 baselines, the perception enumeration, the model-based arm, and
> seven learning arms over 24 seed-runs.

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
**0.06 to 0.59**, against a myopic reference measured at **0.0664 (learned) / 0.0208 (reference at n=48) / 0.0625 (exact)** and an
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

**NEW -- the composite action closes that gap, and the hierarchy claim is now measured.**
Offered the same sub-action as one of **two crystallized modules** -- `sweep` and the
wrong sibling `stare` -- instead of inside the 12-candidate arithmetic pool, reward-only
learning **picks `sweep` on 4 / 4 seeds** and reaches **0.9648 deterministic (3 / 4 seeds
at exactly 1.0000, 64/64 held-out episodes)** in 400 training episodes, against
**0.2396** for the same arm searching the 12-candidate pool, **0.2969** for the
`stare`-only control forced onto the wrong module, a myopic reference of **0.0208
measured / 0.0625 exact**, constant policies and uniform random at **0.0000**, and an
exact oracle at **1.0000**. The selection is learned rather than an initial tie: the
argmax at `next_slot` sits on `stare` through episode 200 on two of the four seeds and
switches to `sweep` by 300, with the training reward rate rising as it switches. So the
12-candidate arithmetic pool was the obstacle, and the abstraction boundary -- a frozen
module chosen by an ordinary candidate softmax -- is what made the sub-action
discoverable from delayed reward alone. Section 8.1.

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

| | |
|---|---|
| combinations (4 documents x 4 seeds x 3 action sequences x 2 objectives x 2 splits x 2 probe settings) | **384** |
| records identical, field for field | **384 / 384** |
| mismatches | **0** |

That is twice the 192-combination precedent in
`research/computer-capability/equivalence.py`, and it includes the probe dimension,
which that check did not.

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

**Verification.** `session_equivalence.py` drives the same addresses, configurations
and action sequences through both transports and compares every serialized
`StepRecord`: **64 / 64 identical, 0 mismatches**, over four panel action sequences x
6 episode indices x 2 splits plus two shell sequences x 4 indices x 2 documents. The
same check is in the suite as `test_session_transport_is_transparent`.

**Note for anyone running `bridge.ts` in parallel.** Under load its `finally { rm(...) }`
intermittently raises `ENOTEMPTY` and kills the transition; it took out one of the six
equivalence chunks and had to be re-run. `session.ts`'s teardown retries and then gives
up, because failing to delete a scratch directory must not end an episode.

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

| policy | mean return | sd | solved / 48 | exact value (section 4) |
|---|---|---|---|---|
| `always_wait` (constant) | 0.0000 | 0.000 | 0/48 | 0 |
| `look_only` (constant) | 0.0000 | 0.000 | 0/48 | 0 |
| `uniform_random` | 0.0000 | 0.000 | 0/48 | ~0.045 |
| `neutral_typed` (the sampler a fresh policy uses) | 0.0833 | 0.276 | 4/48 | 0.0346 |
| `commit_now` (**the myopic reference**) | 0.0208 | 0.143 | 1/48 | **0.0625** |
| `dial_then_commit` (non-myopic, uninformed) | 0.1042 | 0.305 | 5/48 | 0.1111 |
| `fixed_slot_plan` (plan, one fixed slot) | 0.2500 | 0.433 | 12/48 | 0.3333 |
| `oracle` (**the exact plan**) | **1.0000** | 0.000 | **48/48** | **1.0000** |

At n = 48 a single success is 0.021 of the mean, so the references are consistent with
the exact values but individually noisy (`commit_now` drew 1 where 3 were expected,
`neutral_typed` 4 where 1.7 were). **A re-run at n = 192 was launched and was killed with
the session before it wrote anything**: both `out/refs.json` and `out/refs_48.json` are
the n = 48 table above. Re-run `.venv/bin/python research/credit-assignment/refs.py 192`
to tighten it.

Horizon sweep, same references:

| H | oracle (sweep plan) | commit_now (myopic) | uniform_random | neutral_typed |
|---|---|---|---|---|
| 1 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0208 | 0.0000 | 0.0208 |
| 3 | 0.3750 | 0.0208 | 0.0000 | 0.0417 |
| 4 | 0.5208 | 0.0208 | 0.0000 | 0.0833 |
| 6 | **1.0000** | 0.0208 | 0.0000 | 0.0833 |
| 8 | **1.0000** | 0.0208 | 0.0000 | 0.1042 |

The oracle here is the *sweep plan*, not the horizon-conditional DP optimum, which is
why it is 0 at H = 1 and 2 where the DP optimum is `commit` and `dial, commit`. From
H = 3 it tracks the exact V\* (0.375 measured against 0.3333 exact, 0.5208 against
0.5556, 1.0 against 1.0).

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

600 training episodes per seed, `lr = .05`, `discount = .95` unless stated, Adam,
grad-clip 5, value baseline, entropy weight .02, evaluated on 64 held-out episodes at
indices 10000+. **Deterministic** evaluation is `argmax` template with each argument at
its mean, which is what `tcn/agent.py` does at `deterministic=True`. **Stochastic**
evaluation samples, which is what the policy itself does -- and it matters here,
because several arms learn a deliberately *wide* argument sampler and deterministic
evaluation destroys exactly that.

| arm | seeds | eval (deterministic) | sd | eval (stochastic) | seeds at 1.00 | env episodes |
|---|---|---|---|---|---|---|
| `flat` -- no state-dependent policy | 3 | 0.0260 | 0.037 | 0.1146 | 0/3 | 856 |
| `probe_only` -- supervision, no actor term | 2 | **0.0000** | 0.000 | 0.0781 | 0/2 | 524 |
| `myopic` -- **gamma = 0** | 4 | **0.0664** | 0.023 | 0.0820 | 0/4 | 856 |
| `reward_percept` -- perception searched under reward | 3 | 0.0625 | 0.044 | 0.0677 | 0/3 | 856 |
| `reward_g50` -- gamma = 0.5 | 3 | 0.1979 | 0.064 | 0.3281 | 0/3 | 856 |
| **`reward`** -- reward only, gamma = 0.95 | 6 | **0.2396** | 0.058 | **0.5938** | 0/6 | 856 |
| **`reward_sweep_given`** -- the same, sub-action supplied | 3 | **1.0000** | 0.000 | 0.6927 | **3/3** | 856 |
| **model-based** (section 7.3) | -- | **1.0000** | 0.000 | -- | **64/64 episodes** | **89** |
| references | -- | myopic **0.0625**, oracle **1.0000** | | | | |

**The decisive comparison is `reward` against `myopic`, and it is a one-parameter
difference.** The two arms share every line of code, every seed range and every budget;
they differ in `discount`. What each arm's policy learned, read off its argmax logit per
situation:

| arm | nothing showing | task record showing | just dialled | seeds agreeing |
|---|---|---|---|---|
| `reward` (gamma = .95) | **`look`** | **`dial`** | **`commit`** | **6 / 6** |
| `myopic` (gamma = 0) | **`commit`** | commit / dial / wait | `commit` | **4 / 4 on `commit` first** |

That is the credit-assignment result. Every seed of the gamma = 0.95 arm recovers the
exact three-situation plan `look -> dial -> commit`, in which **neither of the first two
actions is ever rewarded on the step it is taken**; every seed of the gamma = 0 arm
falls into the myopic trap and commits immediately. The exact DP predicted the flip at
gamma = 0.394 (section 4.3), and `reward_g50` at gamma = 0.5 -- just above the threshold
-- sits between them at 0.1979 / 0.3281, which is the shape a discount just past a
policy-flip threshold should have.

The training reward rate says the same thing without any evaluation at all:

| arm | 50 | 150 | 300 | 450 | 600 |
|---|---|---|---|---|---|
| `reward` | 0.11 | 0.20 | 0.50 | 0.56 | **0.58** |
| `reward_sweep_given` | 0.08 | 0.19 | 0.64 | 0.61 | **0.74** |
| `reward_g50` | 0.12 | 0.13 | 0.21 | 0.28 | 0.35 |
| `myopic` | 0.14 | 0.07 | 0.08 | 0.10 | **0.07** |
| `flat` | 0.10 | 0.09 | 0.08 | 0.10 | 0.07 |
| `reward_percept` | 0.07 | 0.06 | 0.07 | 0.11 | 0.11 |

**What reward only does not learn: the argument-computing sub-action.** On 6 / 6 seeds
`next_slot` settles on a *constant* slot (`add(one_slot,one_slot)` or
`identity(one_slot)`) and the policy instead **widens its own slot sampler**
(`logstd_slot` rises from 0 to 0.26-2.32), i.e. it searches slots at random rather than
sweeping. That is a coherent strategy -- it is worth 0.7141 (section 6) against 0.3333
for a fixed slot -- and it is why the stochastic evaluation reads 0.5938 while the
deterministic one, which collapses the sampler to its mean, reads 0.2396.

**Supply the sub-action and reward finishes the job.** `reward_sweep_given` is the same
arm with `next_slot` reduced to its one sweeping candidate: **1.0000 on 3 / 3 seeds**,
crossing to 1.000 between episode 100 and 200. So the delayed part is learnable from
reward and the argument part, inside a 12-candidate arithmetic pool, is not -- at this
budget.

**Three controls behave as they must.**

* `flat` (one state-independent logit vector) reaches **0.0260**: the situation-conditioned
  structure is load-bearing, and the arms above are not winning on the templates alone.
* `probe_only` (dense probe supervision, `w_actor = 0`) reaches **0.0000** while its
  probe loss falls -- the F-init control of `research/policy-learning/RESULTS.md`
  reproduces exactly: supervision alone produces no behaviour.
* `reward_percept` (the four perception choices left searched under reward) reaches
  **0.0625**, i.e. the myopic value. This re-measures FINDINGS section 23's boundary
  rather than assuming it: `unpack` is `gradient="none"`, the address choice gets no
  gradient, and with perception unresolved the policy has nothing to condition on.

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

Environment episodes to reach the stated return on 64 held-out episodes, counted by
`panel.Counter` (creations and steps, evaluation included):

| approach | environment episodes | held-out return |
|---|---|---|
| **frozen exact model + enumeration, no policy at all** | **89** | **1.0000 (64/64)** |
| reward only, sweeping sub-action supplied | ~290 (200 training + evaluation) | **1.0000 (3/3 seeds)** |
| **reward only, sub-action chosen between two crystallized modules** (section 8.1, **new**) | **528** | **0.9648 (3/4 seeds at 1.0000)** |
| reward only, sub-action searched in the 12-candidate arithmetic pool | 856 | 0.2396 deterministic / 0.5938 sampling |
| reward only, gamma = 0 | 856 | 0.0664 |
| supervision only, no reward term | 524 | 0.0000 |

The ranking reproduces `research/policy-learning/RESULTS.md`'s: the model-based route is
cheapest by an order of magnitude, and it is still the only route to the exact optimum on
every seed. The composite-action arm is the cheapest *learned* route that gets there
without being handed the sub-action -- 528 episodes against 856 for a worse result -- and
it is the row that turns section 1's level 1 from an argument into a measurement.

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

### 8.1 Measured. The composite action closes the gap.

**NEW -- run 2026-09-09 in a later session by `bash research/credit-assignment/run_macro.sh`;
everything above this subsection is inherited from the original study.** Four
`sweep+stare` seeds and two `stare`-only controls, 400 training episodes each, the same
`Cfg` as `reward` (lr .05, gamma .95, entropy .02, value baseline), evaluated on the same
64 held-out episodes at indices 10000+. 528 environment episodes per seed (400 training +
128 evaluation, both eval modes), counted by `panel.Counter`. About 4 minutes wall clock
for all six concurrently, not the 25 estimated.

The prediction, stated in advance in the original report: `reward` selects a constant slot
out of the 12-candidate arithmetic pool (6/6 seeds) but reaches 1.0000 when the sweep is
the only candidate (3/3 seeds), so the question was whether reducing the choice to *two
crystallized modules* -- one right, one wrong -- is enough for reward alone to pick the
right one. **It is.**

| arm | seeds | eval (deterministic) | sd | eval (stochastic) | chose `sweep` | seeds at 1.0000 | env episodes |
|---|---|---|---|---|---|---|---|
| **`sweep+stare` (the composite-action arm)** | 4 | **0.9648** | 0.061 | 0.6250 | **4 / 4** | **3 / 4** | 528 |
| `stare` only (wrong module forced, control) | 2 | 0.2969 | 0.016 | 0.4219 | 0/2 | 0/2 | 528 |
| `reward` (12-candidate arithmetic pool) | 6 | 0.2396 | 0.058 | 0.5938 | 0/6 | 0/6 | 856 |
| `reward_sweep_given` (sub-action supplied) | 3 | 1.0000 | 0.000 | 0.6927 | -- | 3/3 | ~290 |
| **baselines** | | `always_wait` 0.0000, `look_only` 0.0000, `uniform_random` 0.0000, `commit_now` (myopic) **0.0208** measured / **0.0625** exact, `neutral_typed` 0.0833, `fixed_slot_plan` 0.2500, `oracle` **1.0000** | | | | | |
| **exact ceilings** (section 6) | | sweeping **1.0000**, i.i.d. slots 0.7141, constant slot **0.3333** | | | | | |

Per seed, deterministic: **1.0000 (64/64), 1.0000 (64/64), 1.0000 (64/64), 0.8594 (55/64)**.
The `stare`-only control: 0.3125 and 0.2812, i.e. it sits on the exact constant-slot
ceiling of 0.3333 and cannot do better, which is what makes the library's wrong member a
real control rather than a decoration.

**The selection is learned, not an argmax tie.** `SoftProgram` zero-initialises choice
logits, so at episode 0 both modules are at 0.5 and the relaxed `next_slot` is
`slot + 0.5` -- a nonsense slot, which is why a tie cannot score. The logged argmax at
`next_slot` moves during training: seeds 0 and 2 hold `stare` through episode 200 and
switch to `sweep` by 300; seed 1 switches by 200; seed 3 is on `sweep` by 100. The
training reward rate rises with the switch (seed 0: 0.12, 0.23, 0.64, 0.76 at episodes
100/200/300/400), which is the delayed reward doing the separating.

Every seed also recovers the three-situation verb map `look -> dial -> commit`, as
`reward` did (argmax logits: `idle` -> look, `found` -> dial, `dialled` -> commit,
4/4 seeds).

**What this establishes.** The 12-candidate arithmetic pool *was* the obstacle. The
delayed reward is strong enough to separate two crystallized modules but not strong
enough to find the sweep inside a 12-way arithmetic search, at this budget. The
level-1 claim of section 1 is therefore measured rather than argued: a sub-program
crystallized into a frozen module and offered as one candidate at one node is
selected by reward alone, against a wrong sibling, and against a
`Program.description_bits` prior that prefers the wrong sibling by 952 bits and
prefers no module at all by a further 8,128.

**One qualification, stated because the numbers say it.** Stochastic evaluation reads
**0.6250**, *below* the deterministic 0.9648, and on the three seeds at 1.0000 it reads
0.53, 0.52 and 0.64. The policy keeps a wide slot sampler alongside the sweep, and
sampling perturbs the sweep it has just learned. That is the reverse of `reward`, whose
stochastic number (0.5938) was the *higher* one because random slot search was its whole
strategy. The composite-action arm is better where it matters -- the deterministic
policy, which is what `tcn/agent.py` deploys -- by 0.9648 against 0.2396.

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

## 10. What is not done, in priority order

1. ~~**Run the composite-action arm.**~~ **DONE, 2026-09-09**, by
   `bash research/credit-assignment/run_macro.sh` (about 4 minutes at 6 concurrent
   processes on a quiet machine, not the 25 estimated) and
   `.venv/bin/python research/credit-assignment/aggregate.py`. Results in section 8.1:
   4/4 seeds select `sweep`, 0.9648 deterministic. Section 1's level-1 claim is now a
   measurement. What is *not* done here is seed count -- four `sweep+stare` seeds and
   two controls -- and a budget sweep; 400 episodes was the figure the original run
   script chose.
2. **Collect the `staged_enum` arm.** Three seeds were in flight when the session ended
   and produced nothing; re-run
   `.venv/bin/python research/credit-assignment/arms.py staged_enum <seed> 600` for
   seeds 0, 1, 2 (they can run concurrently; each takes about 25 minutes). Two
   short-budget runs exist at seeds 98 and 99 and are excluded from the tables by the
   `seed >= 90` filter in `aggregate.py`. The expectation is that it matches
   `reward` (the enumerated perception is what the declared arms already have), so its
   value is confirming the handoff rather than the perception.
3. **Collect the n = 192 baselines.** `out/refs.json` will overwrite `out/refs_48.json`
   when the re-run finishes; section 6 currently reports n = 48 and says so.
4. **Run the full test suite.** `.venv/bin/python -m pytest tests/ -q`.
   `tests/test_panel_interface.py` passed 10/10 on its own; the rest of the suite was
   not re-run after the last edit to `generators/computer/engine/session.ts` (a retry
   around a scratch-directory delete, reachable only from the gated session transport).
5. **Then, if the budget exists**: `reward` at 2,000 episodes rather than 600, to see
   whether the sweep is found late rather than never; and the same study at
   `panel_register=0` (section 4.4), where the myopic baseline is 0 rather than 1/16.

---

## 11. Files changed outside `research/credit-assignment/`

Nothing under `tcn/` was touched. Three files:

| file | change |
|---|---|
| `generators/computer/generator.py` | gated `interface='panel'`: three verbs added to `action_schema`, `panel_setup`, panel branches in `initialize` / `advance` / `observe`, three privileged probes and one latent, an opt-in `Session` transport, and the shell menu frozen as the literal `SHELL_MENU` |
| `generators/computer/engine/session.ts` | **new file**, a long-lived sibling of `bridge.ts`; nothing reaches it unless `configuration['session']` is set |
| `tests/test_panel_interface.py` | **new file**, 10 invariants, all passing |

**Confirmed, and it is the load-bearing one:** the three panel verbs are declared in
`Implementation.action_schema` (so `Host.validate_actions` accepts them) and are **gated
out of the default `available_actions`**. `observe` now returns the literal
`SHELL_MENU = ('wait','command','type','key','read','write')` for the shell interface
instead of the old `tuple(self.action_schema)`, which would have silently grown. The
panel verbs are additionally **refused** outside the panel interface, and the shell verbs
refused inside it, rather than merely being unlisted. This is asserted by
`test_panel_menu_and_default_shell_menu_are_disjoint_and_stable` and by all 384
combinations of `equivalence.py`, which compare full `StepRecord`s -- `available_actions`
included -- against a pre-change copy of the generator.

---

## 12. Limitations

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
* **The composite-action arm has four seeds and one budget.** 400 episodes, four
  `sweep+stare` seeds and two `stare` controls, all newly measured (section 8.1); one of
  the four reads 0.8594 rather than 1.0000. The comparison it is read against, `reward`
  at 0.2396, ran at 600 episodes rather than 400, so the composite arm wins on a
  *smaller* budget -- but the two are not budget-matched, and no budget sweep was run.
* **Seeds.** The two headline arms have 6 and 4 seeds and the ablations have 3; the
  policy-learning track used 8. The reduction is a compute decision under the load
  above, and the seed-level numbers are in `out/arm_*.json` rather than only their
  means.
