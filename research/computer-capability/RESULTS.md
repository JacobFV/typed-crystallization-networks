# Computer use: a typed program that reads a file and writes a transformed value

Research track `computer-capability`, 2026-09-09. Everything below was produced in
this workspace with the repository `.venv` (Python 3.13.15, Node 24.21.0, aarch64).
Raw data is in `out/`; every script named here regenerates its own file.

`tcn/` was **not modified**. `generators/computer/generator.py` and
`generators/computer/engine/bridge.ts` gained one gated probe channel, described and
verified in section 3. The full Python suite is **179 passed** with the change in
the tree.

---

## 0. Verdict

**A typed program now acts usefully in the computer environment, and it generalizes.**
The exact frozen `Agent` of `tcn/agent.py` scores **2.00 / 2 mean return on 10
held-out episodes** whose file contents it never saw, against **0.30, 0.20 and 0.60**
for three baselines and **0.10** for six random points of its own search space. It
holds 2.00 / 2 across six reading commands it was not searched on and five document
formats it was not searched on.

Three things had to be true for that, and each is the real finding:

1. **The shipped supervision channels carry zero task information.** Over eight
   documents driven through one action sequence, the `state_counts` probe and the
   `event_count` latent take **1 distinct value series** while `terminal` takes 8.
   They are counters of the agent's own actions. Nothing dense could be attached to
   them, and dense supervision is what `research/FINDINGS.md` section 1 measures as
   the thing that trains this substrate. **The generator needed a richer probe
   channel**, and section 3 adds one.
2. **The program was found by enumeration, not by gradient descent.** On the
   perception/transform region the exhaustive sweep of 7,480 programs returns the
   reference program with **held-out max error 0.0**, while the relaxed arm returns a
   program with exact error **52.0** and held-out accuracy **0.0**. The address choice
   receives **`grad is None`** — no gradient at all, because `unpack`, the only
   declared exit from `role="byte"`, is `gradient="none"`.
3. **The scalar reward cannot start.** The `write.text` argument is a 1026-parameter
   typed text. At a neutral policy the probability that one write is rewardable is
   **5.65e-06**, i.e. **176,942 write actions**, i.e. about **27 days** of wall clock
   at the measured 13.3 s per episode. This is measured analytically and checked by
   20,000 draws (0 hits).

**The blocking constraint is not the environment.** The kernel is a real
transition system. It is the *contract over it*: one bit of reward, an action
argument that is 1026 continuous samples, byte observations whose only predicate has
a surrogate that is exactly zero at the distances that occur, and — before this
track — no privileged state channel to hang dense supervision on.

---

## 1. What the environment is, and what a transition costs

`out/audit.json`, `research/computer-capability/audit.py`.

| | |
|---|---|
| Observations | `pixels` width **18435** (`role="byte"`), `terminal` width **4097** (`role="byte"`) |
| Latent | `event_count`, width 1 |
| Probe | `state_counts`, width 3 |
| Reward | one component, `goal`, from an exact content match on a configured path |
| Actions | `wait`, `command` (text 513), `type` (text 513), `key` (code 1), `read` (path 129), `write` (path 129 + text 513) |
| Initialize | 1.58 s |
| Step | **1.96 / 3.13 / 2.77 / 3.88 s** for read / write / wait / command |
| Episode, horizon 4 | **13.3 s** |

Cost structure, from the source: `advance` re-executes the **whole event log** in a
fresh Node process each time an action changes state, and then, when an objective is
configured, executes the whole log **again** with an appended read to score the
reward. Two subprocesses per acting step, each replaying the episode from boot.

**Reward causality, verified rather than assumed** (`docs/LESSONS.md`'s fidelity
lesson: interface coverage does not prove execution semantics):

| trajectory | reward per tick |
|---|---|
| read, write the correct successor | 0.0, 0.0, **1.0** |
| read, write a wrong digit | 0.0, 0.0, 0.0 |
| read, wait | 0.0, 0.0, 0.0 |

The reward responds to the agent's write and to nothing else, and it is *stateful*:
once the file holds the target content every later tick also scores 1.0.

---

## 2. Step 1 — what is learnable before anything is trained

### 2.1 The probe and latent channels carry no task information

Eight documents, one fixed action sequence `read, write, wait`, everything else
held constant:

| channel | distinct value series over the 8 documents |
|---|---|
| `terminal` observation | **8** |
| `state_counts` probe | **1** |
| `event_count` latent | **1** |

`state_counts` is `(number of computers, number of packets, trajectory length)`. The
first two are constants of the topology; the third counts the agent's own events. A
prediction or probe target built from it is a function of the action sequence alone.
`TrainConfig.Target` can only read `probes`, `latent_states` or `observations`, so
**as shipped there is nothing on this generator for `L_future_latents` or
`L_intermediate` to attach to.** That is the honest answer to step 1: a richer probe
channel is required, not optional.

### 2.2 The `eq` surrogate is dead at the distances this task actually uses

`role="byte"` keeps the terminal out of `Type.numeric`, so `eq` is the only
predicate available on it. Measured at the operating distances of this task
(`out/audit.json`):

| comparison | byte distance | surrogate at shipped `tau=1` | at `tau=2^bits=256` |
|---|---|---|---|
| terminal byte 0, tick 0 vs tick 1 (`{` vs `c`) | 24 | **7.0e-251** (exactly 0.0 in float32) | 1.05e-01 |
| digit `0` vs digit `9` | 9 | 6.6e-36 | 7.29e-01 |
| digit `7` vs digit `8` | 1 | 3.68e-01 | 9.96e-01 |
| uniform byte mixture (127) vs `c` | 28 | **exactly 0.0** even in float64 | 4.68e-02 |

This is `research/FINDINGS.md` section 16 reproduced on a new domain, and the
one-line `tau = 2^bits` fix proposed there is **still not applied to `tcn/`**. Every
number in this report is against the shipped surrogate.

### 2.3 Three structural limits, measured

`out/limits.json`, `limits.py`.

* **L1 — the episode address does not reach the task.** Four episode indices with one
  configuration produce **1 distinct terminal** before and after a read. `initialize`
  uses `address.rng()` only to seed the kernel; the document is a fixed configuration
  string. Held-out episode addresses therefore hold out nothing about the task.
* **L2 — the objective is not consumable.** `ActorView.objective` carries
  `{'path': ..., 'content': '8'}`, but `Agent.act` and `JointTrainer.inputs` build
  program inputs from `view.observations` alone, and no observation names the goal.
  Goal-conditioned generalization over held-out objectives is not expressible here.
* **L3 — `TrainConfig.generator_config` is per run, not per episode.** The only thing
  `JointTrainer` varies with the episode index is the objective, which by L2 the
  actor cannot see. **The shipped integrated trainer cannot present this generator
  with a distribution of tasks.** This is why the experiment below drives its own
  episode loop and varies `configuration` per episode through the public interface.

### 2.4 A reward that can only modify, never create

An objective naming a path that does not yet exist raises
`RuntimeError: ... no such file: /home/agent/answer.txt` out of the kernel and
terminates the episode — at *every* tick before the agent creates the file, which is
every tick where a create-a-file task is still unsolved. Reproduced in
`out/audit.json` (`objective_on_nonexistent_path`) and again in the closed loop
(`out/closed_loop.json`, `F_unseen_objective_path`). **The objective mechanism can
express "change this file" and cannot express "create this file."**

---

## 3. The gated probe channel

Added to `generators/computer/generator.py` and `generators/computer/engine/bridge.ts`.
Emitted **only** when `configuration['probe']` is present; absent by default, in
which case the bridge receives no `probe` key and its payload is byte-identical to
the payload it produced before the channel existed.

Following the `gates` channel of `research/FINDINGS.md` section 15: capacity-declared
relations whose **type does not depend on episode content**, with a load-bearing index
field because a set is duplicate-free, and deterministic truncation at the declared
capacity with the true count reported separately.

| key | channel | type | width at the capacities used |
|---|---|---|---|
| `filesystem` | probe | `set[(index, parent, name hash, kind, size, mode)]` cap 24 | 168 |
| `processes` | probe | `set[(pid, ppid, state, executable hash)]` cap 8 | 40 |
| `content_0` | probe | text at declared capacity 64 | 65 |
| `content_0_present` | probe | `bool` | 1 |
| `goal_reached` | probe | `bool` | 1 |
| `file_count`, `process_count` | latents | `int[32]` | 1 each |

Everything is derived from state the kernel already holds. `vfs.list` and
`vfs.readFile` record no trajectory event (unlike `execute` and `writeTextFile`), so
enabling the channel cannot move the simulation or the existing `state_counts` probe;
`out/probe_channel.json` shows `state_counts` unchanged with the channel on. Name
hashes are computed in Python, not JavaScript, so they replay across interpreters.

**Probes and latents only.** `out/probe_channel.json` records
`actor_view_keys = ['pixels', 'terminal']` and an empty `leak_check`: none of
`filesystem`, `processes`, `content_0`, `goal_reached`, `file_count`,
`process_count` reaches `StepRecord.actor_view`. The agent still sees the terminal
and must compute everything else.

### 3.1 The default stream is bit-identical, over 192 combinations

`equivalence.py`, `out/equivalence.json` and `out/equivalence_wide.json`. A verbatim
copy of the pre-change generator (`generator_before.py`) and the current one are
driven through the same addresses, configurations and action sequences, and **every
serialized `StepRecord`** — observations, latents, probes, transitions, rewards,
available actions, times — is compared.

| sweep | combinations | identical |
|---|---|---|
| 3 documents x 3 seeds x 2 action sequences x 2 objectives | 36 | **36** |
| 4 documents x 4 seeds x 3 action sequences x 2 objectives x 2 splits | **192** | **192** |

466 s and 2,621 s respectively; 456 live episodes in total. This matches the
192-combination precedent of `research/FINDINGS.md` section 10. Note that
`Host.digest` necessarily changes anyway, because `source_fingerprint()` hashes all
of `tcn/` and `generators/`; what is verified here is the semantic stream, which is
the thing a recorded episode's *meaning* depends on.

### 3.2 One gated behaviour change, stated separately

When the channel is on **and** the objective's path is in its content list — which
`probe_request` always arranges — `advance` scores the reward from the probe read
instead of spawning a second Node process. Same comparison on the same state; it
halves the subprocess cost of an acting step, and a missing file reads as
*unsatisfied* rather than raising, which is a fix for section 2.4 inside the gated
path only. The default path is untouched, and the 192-combination sweep above covers
configurations with and without an objective.

### 3.3 The channel actually supplies the supervision

`probe_supervision.py`, `out/probe_supervision.json`. The synthesis of section 5 is
re-run with every target read out of `StepRecord.probes` — the byte to write is
`ord(content_0[-1])` at the tick after the write — instead of out of the
configuration this harness chose.

* identical selections to the harness-labelled run (`agrees_with_harness_labelled_run: true`),
* 2 conforming programs out of 7,480, held-out accuracy **1.0**,
* over the same 5 documents, `content_0` takes **5 distinct series** and
  `state_counts` takes **1**.

---

## 4. The task, and why it is causal

`task.py`. `/home/agent/task.txt` holds `<name> = <digit>`. The objective is
satisfied when the file's whole content is `str(digit+1)`.

* At tick 0 the terminal shows the setup write's JSON result,
  `{"path": "/home/agent/task.txt", "bytes": 9}` — the file's **length**, not its
  content. The digit becomes visible only after the agent has itself issued a read.
  An agent that writes at tick 0 cannot know what to write.
* The digit varies per episode, so no constant text is right more than 1 time in 9;
  measured, the best constant byte scores **0.20** on both splits.
* The name varies in length (5 to 15 characters), so no constant byte address finds
  the digit; measured, the conforming address is **unique** in a 136-candidate
  address space (`out/tie_check.json`).
* The reward is the shipped one — the kernel reads the file back and compares bytes —
  and section 1 shows it responds to the write and to nothing else.

Horizon 3, maximum return 2. The reference trajectory `read, write, wait` scores
2.0 on all 15 episodes collected.

Held out: names never seen (`k, xy, idx, amount, accumulator, q, zz, sum, length,
registers`), digits never seen, and lengths inside and outside the trained range.
Length does not determine the digit in the test set (`k = 7` and `q = 5` are both 5
characters).

---

## 5. Synthesis: enumeration solves it, relaxation does not

`program.py`, `search_run.py`, `out/search.json`. Both arms search the **identical**
candidate space. Every choice logit starts at zero, so the relaxed arm has a uniform
prior and the enumerative arm has none.

**Declared** (stated in `program.py`'s docstring, per AGENTS.md): the graph shape;
the byte/numeric representation boundary, where `unpack` is the only declared exit
from `role="byte"`; the policy-parameter encoding, which is the algebraic inverse of
`sample_typed`'s bounded decode, `mean = (log x - log(255-x))/2`, written from `log`,
`sub` and `mul` because no `atanh` operator exists; the three action-logit constants.
**Searched**: which byte holds the digit, what arithmetic transforms it, which byte
decides whether the terminal shows file content or a command result, and against
which constant.

### 5.1 Perception and transform — 7,480 programs

| arm | result |
|---|---|
| **Enumeration** (`enumerate_fit`, rank `description`) | exhausted 7,480 in **168 s**; **2 conforming** (density 2.67e-04); train accuracy **1.0**, held-out accuracy **1.0**, held-out max error **0.0** |
| Selected program | `pos = sub(length, 1)`, `shift = add(value, 1)` — a **computed** address, chosen over 15 constant addresses in the pool |
| The tie | both conforming programs use the *same, unique* address; they differ only by the commutation of `add`, and both score held-out max error 0.0 (`out/tie_check.json`) |
| **Gradient** (`synthesis.fit`, 300 steps, neutral init) | exact max error **52.0**, exact conformance **false**, train accuracy **0.0**, held-out accuracy **0.0**, 210 s |
| Random baseline | **0 / 400** draws conform |
| Constant baseline | best constant byte **0.20** train, **0.20** held out |

**Why the gradient arm fails, measured rather than inferred.** At initialization the
choice-logit gradients are:

| node | gradient max abs |
|---|---|
| `pos` (which byte to read) | **None** — no gradient at all |
| `shift` (what arithmetic to apply) | 34.88 |

`pos` sits upstream of `unpack`, the declared byte-to-numeric conversion, which is
`gradient="none"`; `relaxed` evaluates it with `exact_tensor`, which is built from
detached lists. The value path is severed there, so no temperature and no surrogate
scaling reaches the address. This is the address wall of `FINDINGS` sections 11/14/16
in a **fourth and different form**: not a saturated surrogate (section 16's diagnosis)
and not kernel locality, but a hard `gradient="none"` representation boundary that the
type algebra forces every byte program to cross.

The relaxed intermediate values show the same thing from the other side: at the
uniform address mixture the blended `byte` reads `1e-06, 7.7e-04, 9.6e-02, 21.9, 42.3`
across the five examples — the 4,096-wide `index` softmax spreads over mostly-zero
padding, so the mixture denotes nothing near a digit.

### 5.2 Policy — 448 programs

| arm | result |
|---|---|
| **Enumeration** | exhausted 448 in **26 s**; **21 conforming** (density 4.69e-02); train accuracy **1.0**, held-out accuracy **1.0** on 30 held-out decision points |
| Selected program | `ppos = 0`, `brand = eq(terminal[0], '{')` — write when the terminal is not a JSON result, wait when the previous action was a write, otherwise read |
| Random baseline | **20 / 400** = 5.0%, which matches the enumerated density 4.69% |
| Constant baseline | **0.333** (labels balanced 5/5/5) |
| **Gradient** | **failed**: `ValueError: value outside semantic bounds` |

**The whole conforming set, scored held out** (`conforming_set.py`,
`out/conforming_set.json`), because `FINDINGS` section 14 records this tie-break
being measurably wrong once before: of the 21 conforming programs, **19 reach
held-out accuracy 1.0 and 2 reach 0.933**; the returned program is one of the 19. The
tie-break was not wrong here, but it was not certified either — only exhaustion plus
a held-out score establishes that.

The gradient failure is informative. `legal_candidates` admits address candidates
whose exact execution is *partial* — `sub(q3, length)` is negative for a short
terminal. Enumeration handles partiality correctly, by discarding the candidate for
that input (`evaluate` returns `None`). Relaxation **mixes** them, and the mixed
address leaves the declared domain of its own type, so the crystallizer's hard-forward
trial cannot be encoded at all. At initialization the surviving gradients are
`ppos` 3.98e-02 and `brand` 1.40e-03, and the `brand` gradient is an accident: the
blended probe byte lands at 100.71 and the constant `c` = 99 happens to be 1.71 away,
which is the only reason the `eq` surrogate is not exactly zero.

### 5.3 A substrate fault found on the way

`SoftProgram.__init__` treats any node with `selected is not None` as **frozen**, and
`SoftProgram.forward` computes a frozen node as `exact_tensor(...).detach()`. A
scaffold that pins its single-candidate plumbing nodes therefore severs the autograd
graph at each of them: the first version of `program.py` did exactly that, and the
relaxed policy loss came back with **no `grad_fn` at all**, so `backward()` raised
`RuntimeError: element 0 of tensors does not require grad`. Leaving one-candidate
nodes unselected — a one-way softmax, which is the identity — restores the gradient
and changes nothing for enumeration. Every gradient number above is from the
corrected scaffold, so the failures reported are the surrogate's and the
representation boundary's, not the scaffold's. This is worth a line in `tcn/`: a
node with exactly one candidate is not a *decision* and pinning it should not be a
gradient boundary.

---

## 6. Closed loop: the exact frozen program in the live kernel

`closed_loop.py`, `out/closed_loop.json`. The two searched selections are composed
into one 23-node program, exported with `tcn.runtime.save_program` and reloaded
(`reload_digest_matches: true`), then driven by `tcn.agent.Agent` with
`deterministic=True`. Every number is the reward the shipped generator returned for
the action the frozen program chose. About 110 live episodes, 870 s.

| arm | episodes | mean return / 2 | solved |
|---|---|---|---|
| **Held-out documents** | 10 | **2.00** | **10 / 10** |
| Training documents | 5 | 2.00 | 5 / 5 |
| Baseline: always write `"5"` | 10 | 0.30 | 0 / 10 |
| Baseline: read, then write the modal digit `"5"` | 10 | 0.20 | 1 / 10 |
| Baseline: uniform random verb, random digit | 10 | 0.60 | 1 / 10 |
| **Ablation: 6 random points of the same search space** | 30 | **0.10** | 0 / 30 |

The ablation (`ablation.py`, `out/ablation.json`) resamples only the searched nodes
and leaves the declared plumbing alone. Five of six arms crash mid-episode on an
out-of-domain address and are scored the return they had accumulated, which is zero;
the sixth runs and scores 0.60. **The searched content is what produces the return.**

The frozen program's actual trajectory on a held-out episode is
`read -> write("8") -> wait`, with the written text recorded in
`out/closed_loop.json`.

---

## 7. Generalization outside the training structure

Held-out seeds alone would not be a generalization claim here at all — by L1 the
episode address does not reach the task. Every arm below varies the *structure*.

**Unseen documents** (section 6): 10 names and digits never searched on, **2.00 / 2**.

**Unseen reading commands.** The searched program was found on episodes where the
observation came from the `read` action. Swapping template 1 for a shell `command`
changes how the text reaches the terminal:

| command | episodes | mean return / 2 |
|---|---|---|
| `cat /home/agent/task.txt` | 5 | **2.00** |
| `head -n 1 /home/agent/task.txt` | 5 | **2.00** |
| `tail -n 1 /home/agent/task.txt` | 5 | **2.00** |
| `grep = /home/agent/task.txt` | 5 | **2.00** |
| `sed -n 1p /home/agent/task.txt` | 5 | **2.00** |
| `awk {print} /home/agent/task.txt` | 5 | **2.00** |
| `wc -c /home/agent/task.txt` | 5 | **0.00** |

Six of seven transfer. `wc -c` correctly does not: its stdout is a byte count, not
the counter, and the program does exactly what it learned — increments the last byte
of whatever the terminal shows. That is a scope limit of the learned program, stated,
not a defect hidden.

**Unseen document formats**, four digits each, none of these shapes searched on:

| format | mean return / 2 |
|---|---|
| `count=7` (no spaces) | **2.00** |
| `7` (bare digit, length 1) | **2.00** |
| `value: 7` | **2.00** |
| `  total = 7` (leading whitespace) | **2.00** |
| `answer -> 7` | **2.00** |

The computed address `length - 1` is what carries this; a constant address would not,
and enumeration certified that no constant address in the pool conforms.

**Unseen objective location.** Blocked, not failed: pointing the objective at
`/home/agent/Desktop/notes.txt` raises out of the kernel at tick 1 because the file
does not exist (section 2.4). Objective-location generalization is not testable on
this generator until that is fixed.

---

## 8. Why the shipped integrated trainer cannot do this, quantified

`reinforce_feasibility.py`, `out/reinforce_feasibility.json`.

The generator's `write.text` argument is a 512-byte text, so `sample_typed` emits
**1026** parameters. `read_text` reads only the first `length` bytes, so the effective
target is two draws — but both are continuous, squashed through `tanh` onto `[0,512]`
and `[0,255]`, and the reward is exact string equality.

| quantity | value |
|---|---|
| P(length field decodes to 1) | 1.514e-03 |
| P(byte field decodes to the successor) | 3.733e-03 |
| **P(one neutral write action is rewardable)** | **5.652e-06** |
| Expected write actions to the first reward | **176,942** |
| Expected episodes at horizon 3, uniform verb policy | **176,942** |
| At the measured 13.3 s per episode | **27.2 days** of wall clock |

Checked by 20,000 draws of `sample_exact` at neutral parameters: 38 landed on
length 1 (analytic 30.3) and **0** produced the target text. With the searched
program supplying the two means, 20,000 / 20,000 land on length 1 and 12,067 produce
the exact target under sampling. The frozen agent runs the `deterministic=True` path,
which takes the mean rather than sampling, and section 6 measures it emitting the
exact target on 10 of 10 held-out episodes.

Together with L1-L3 of section 2.3, this is the certificate: **on this generator the
shipped `JointTrainer` cannot vary the task across episodes, cannot show the actor
which objective it is being scored against, and would need ~177,000 episodes before
its policy gradient saw a single nonzero reward.** Dense supervision through a probe
channel is not a convenience here, it is the only available route.

---

## 9. Faults found

| # | fault | evidence |
|---|---|---|
| C1 | An objective naming a file that does not exist raises out of the kernel and kills the episode, so the reward can express "modify a file" but not "create a file" | `out/audit.json`, `out/closed_loop.json` F |
| C2 | `state_counts` and `event_count` are functions of the action sequence alone; 8 documents give 1 distinct series | `out/audit.json` A2 |
| C3 | `SoftProgram` treats a one-candidate node as frozen and detaches it, so pinning plumbing severs the autograd graph and `backward()` raises | section 5.3 |
| C4 | `legal_candidates` admits candidates that are partial at execution; enumeration discards them per input, relaxation mixes them and produces a value outside the declared domain of the node's own type | section 5.2 |
| C5 | `advance` re-executes the entire event log twice per acting step, in two fresh Node processes; episode cost is quadratic in horizon | section 1 |
| C6 | `description_bits` on a program over a 4,096-wide tuple reads **3.4e07 bits**, essentially all of it the recursive serialization of 4,096 identical byte type dictionaries. It measures JSON verbosity, as `FINDINGS` section 6 already records | `out/closed_loop.json` |
| C7 | The `tau = 2^bits` surrogate fix that `FINDINGS` section 16 derived is still not applied in `tcn/learning.py`; on this domain it is the difference between 7.0e-251 and 1.05e-01 | section 2.2 |

---

## 10. What this does not show

* **It is one small task.** Read one byte, add one, write one byte, in three steps.
  It exercises reading a file, a computed address, a representation conversion, a
  perceptual predicate over raw bytes, and an action argument that is a function of
  the observation. It does not exercise multi-file work, processes, navigation,
  package management, git, or anything the kernel's other 20,000 lines support.
* **The gradient path contributed nothing.** Every program reported here came from
  exhaustive enumeration. This is `FINDINGS` section 8's recommended framing holding
  on a new domain — discrete search where the space is small and exactly checkable —
  but it means no claim is made for relaxation on computer use.
* **The scaffold is authored.** The graph shape, the representation boundary and the
  action-parameter encoding are declared, and stated in `program.py`. What was
  searched is the address, the transform, the probe position and the probe constant,
  and enumeration certifies the answer lies in that space with the address unique.
* **No integrated `JointTrainer` run was performed.** Section 8 explains why one
  cannot be informative on this generator as it stands; that is a certificate, not a
  substitute for the run, and if L1-L3 are fixed the run should be done.
* **`pixels` was not used.** The instruction was to prefer `terminal`, and section 2.2
  is why: pixels are the same `role="byte"` carrier with the same dead surrogate, at
  4.5x the width.
* **Synthetic only.** Everything is the local kernel. Nothing here says anything about
  a real operating system.

---

## 11. Episode and time budget

Live environment episodes, since wall clock alone is not the right unit:

| script | live episodes | seconds |
|---|---|---|
| `audit.py` | ~16 | 120 (section A2 alone) |
| `probe_channel.py` | 6 | 23 |
| `task.py` (example collection) | 15 | 153 |
| `equivalence.py` (36 combinations, both arms) | 72 | 466 |
| `equivalence.py --wide` (192 combinations, both arms) | 384 | 2,621 |
| `limits.py` | 6 | 15 |
| `probe_supervision.py` | 15 | 56 |
| `closed_loop.py` | ~110 | 870 |
| `ablation.py` | 30 | 87 |
| **total** | **~650** | **~4,400 s** |

Non-environment compute: enumeration 168 s (transform) and 26 s (policy); the relaxed
arms 210 s and 234 s; the conforming-set sweep 30 s. The host was heavily loaded
throughout (load average 74-88 on 20 cores), so all wall-clock figures are upper
bounds.

---

## 12. Reproduce

```
.venv/bin/python research/computer-capability/audit.py
.venv/bin/python research/computer-capability/probe_channel.py
.venv/bin/python research/computer-capability/limits.py
.venv/bin/python research/computer-capability/equivalence.py          # 36 combinations
WIDE=1 .venv/bin/python research/computer-capability/equivalence.py   # 192 combinations
.venv/bin/python research/computer-capability/task.py                 # collects examples
.venv/bin/python research/computer-capability/search_run.py
.venv/bin/python research/computer-capability/tie_check.py
.venv/bin/python research/computer-capability/conforming_set.py
.venv/bin/python research/computer-capability/probe_supervision.py
.venv/bin/python research/computer-capability/closed_loop.py
.venv/bin/python research/computer-capability/ablation.py
.venv/bin/python research/computer-capability/reinforce_feasibility.py
.venv/bin/python -m pytest tests/ -q                                  # 179 passed
```

`generator_before.py` is a verbatim copy of `generators/computer/generator.py` as it
stood before this track, with only its engine path repointed; it exists so
`equivalence.py` can compare the two streams in one process.
