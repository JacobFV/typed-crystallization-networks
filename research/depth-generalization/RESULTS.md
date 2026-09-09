# Depth generalization — 2026-09-09

**Verdict: depth generalization is achievable with the current algebra. No
operator, type, relaxation or loss term was added, and none is proposed.**

Two things were open. The first is representational: the `program` observation is
a tuple of `3 x depth` scalars, so its *type* changes with depth and a
fixed-width typed program cannot accept an episode of unseen depth. The second is
computational: evaluating a depth-d circuit is inherently sequential, and `map`
is parallel over a set and carries no fold.

Both are answered inside the existing algebra. The gate list is emitted as a
**relation of indexed gates** whose type does not depend on depth, and the fold
is the graph's **explicit recurrence** — one gate per tick over `Program.state`,
which `ARCHITECTURE.md` section 4 already specifies. The resulting program text
is one fixed 60-node graph, byte-identical at every depth, that scores **4.00/4
at depths 3, 4, 6 and 8 in 8/8 seeds having trained only at depths 1-2**, against
a best constant of 2.06-2.50 on the same episodes and the same scored window.
The table-conditioned interpreter candidate — one of 17 at one node — is selected
in 8/8 seeds. Enumeration over the scaffold's own 272-program space exhausts in
25 s, certifies the solution unique, and selects the same program.

Two things are load-bearing and neither is free. A two-node `mux` is what keeps
the `relation` choice differentiable at all: without it every choice in the
recurrence has `grad is None` under the task objective, and the arm is at chance.
And when the *input binding* of a gate operand is searched rather than declared,
2 of 8 seeds mis-pick it, which is the address-relaxation wall of FINDINGS
sections 11 and 14 reappearing on a task with no images in it. Enumeration gets
the binding right, uniquely, in 98 s.

Everything below was measured on this worktree with the repository `.venv`.
`tcn/` is unmodified; `generators/logic/generator.py` gained one gated
observation channel, verified bit-identical on the default stream.

---

## 1. The representation

### 1.1 Encoding and semantics

A depth-d circuit over w inputs is emitted as the relation

```text
gates : set[(index, wire_a, wire_b, table)]      capacity C, fields int[8] unsigned
```

with one element per gate. The declared semantics:

* wire `j` for `j < w` is input bit `j`; wire `w + i` is the output of gate `i`.
* element `(i, a, b, t)` means `wire[w + i] = t(wire[a], wire[b])`, where `t` is a
  two-input truth table read as bit `2*wire[a] + wire[b]` of `t`, exactly the
  convention `generators/logic/generator.py:evaluate` already uses.
* the generator draws `a, b < w + i`, so every referenced wire is defined before
  it is read; the circuit's value is wire `w + d - 1` where `d = |gates|`.
* the `index` field is what makes this a faithful *sequence*: two gates may share
  wiring and table, and a set is duplicate-free, so without it they would
  collapse and the order would be unrecoverable. `ARCHITECTURE.md` section 1
  names exactly this shape — "Sequences are indexed values plus length; relations
  are sets of tuples" — and the length is `count(gates)`.

**Capacity bound.** `C` is a declared property of the type, not of the episode:
the channel exists only when `gate_capacity` is configured, and a draw with
`depth > gate_capacity` is refused rather than truncated. The largest wire index
is `w + C - 1`, and the generator's own limits (`inputs <= 16`, `depth <= 64`)
keep that inside the `int[8]` field; the field refuses anything that would not
fit rather than wrapping.

**Round trip.** `gate_set_value` / `gates_from_set` are inverse: sorting the
relation by its index field restores the generator's own gate list exactly, at
every depth tested, including when two gates are identical apart from their
index. Verified in `tests/test_gate_observation.py`.

**Not a new privilege.** The relation carries exactly the fields the `program`
channel already carried — `gates_from_set(obs['gates'])` equals `obs['program']`
reshaped into triples, asserted at depths 1, 2, 3 and 8. What changes is the
type, not the information.

### 1.2 Measured: one type at every depth

`run.py representation` -> `out/representation.json`.

| depth | `program` tuple fields | `program` flat width | `gates` capacity | `gates` flat width | `gates` cardinality |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 3 | 8 | 40 | 1 |
| 2 | 6 | 6 | 8 | 40 | 2 |
| 3 | 9 | 9 | 8 | 40 | 3 |
| 4 | 12 | 12 | 8 | 40 | 4 |
| 6 | 18 | 18 | 8 | 40 | 6 |
| 8 | 24 | 24 | 8 | 40 | 8 |

Six distinct `program` types; **one** `gates` type, compared by full type digest.
Depth moves cardinality, not width.

The blocker restated as an executable check: `research/structure-generalization`
`common.py:program_scaffold` — the step-4 table-conditioned scaffold, unmodified
— accepts depth 1 and refuses depths 2, 3, 4, 6 and 8 with
`ValueError: table-conditioned scaffold is typed for a single-gate program
observation`. That is the wall, and it is a type error, not a training failure.

### 1.3 The default stream is unchanged

`check_default_stream.py` hashes each episode's complete snapshot — state,
observations, probes, latents, rewards, action menus, recorded inputs — with only
the `source` fingerprint removed, over **192 combinations**: 16 configurations
(default, five depths, fixed wiring, explicit tables, two widths, `nondegenerate`,
`min_relevant_inputs`) x 3 seeds x 4 splits, four steps each. Captured before the
edit and compared after: **192 episodes, 0 differ.**

The channel is emitted only when `gate_capacity` is configured, and
`gate_capacity` is written into the generator state only in that case, so an
episode drawn without it has the same state dict — and therefore the same
snapshot — it had before the channel existed. `source_fingerprint()` hashes every
`.py` under `tcn/` and `generators/`, so it changes for any edit at all and
cannot be the comparison; that is why it is excluded and everything else is
included. This is the same precedent as the 192-config check in FINDINGS
section 10.

---

## 2. Is sequential evaluation expressible? Yes, through the recurrence

### 2.1 The difficulty, stated before building

Gate `i`'s inputs may be gate `i-1`'s output, so evaluation is a fold, and the
algebra has no fold. `map` applies a module to every element of a set
independently; nothing carries a value from one application to the next. A
bounded unroll would express it, but only by writing `D_max` blocks into the
program text, which reintroduces exactly the dependence on depth that the
encoding removed — and a module cannot help, because each call site is still a
node.

`ARCHITECTURE.md` section 4 supplies the third option directly: "Within a tick it
is a DAG; recurrence reads old state and commits new state through explicit
delays." One gate per tick over `Program.state` is a fold whose *program text*
does not grow with depth. Section 4 also says execution cost is charged per use,
which is the honest price: the text is flat, the number of ticks is not.

**So the answer is: expressible, no addition needed.** As in the positional-reuse
track, the supposed blocker was expressible after all — and the shape of the
answer is the same one section 1 already gives for sequences.

### 2.2 The program

`interpreter.py`. Two state ports and a fixed graph:

```text
state values  : set[(wire, bool)]     wires computed so far
state counter : int                   which gate this tick evaluates

held     = insert({}, counter)                          set[int], capacity 1
records  = pair(gates, held)                            set[(gate, counter)]
current  = filter(records; module: gate.index == counter)
wire_a   = sum(map(current; module: gate.a))            0 once the circuit is finished
va       = mux(wire_a < w, index(bits, wire_a mod w), member(values, (wire_a, true)))
...      likewise wire_b -> vb, and table
relation = index(<truth_0..truth_15(va, vb)>, table)    <- the searched choice
is_final = eq(counter + w, w + count(gates) - 1)
answer   = mux(is_final, relation, member(values, (w + count(gates) - 1, true)))
values'  = insert(values, (w + counter, relation))
counter' = counter + 1
```

Every operator is already in `tcn/operators.py`. Three details were forced by
real constraints and are worth recording, because each is the kind of thing that
looks like a missing operator until it is not:

* **`sum`, not `reduce_max`, extracts the singleton.** `Registry.exact` raises
  `empty reduction` for min/max/mean on an empty set, and the filtered set *is*
  empty once the counter passes the last gate. `sum` of the empty set is 0, so
  the recurrence stays total after settling instead of crashing. The junk wire it
  then writes has index `w + counter >= w + d`, which can never collide with the
  answer wire `w + d - 1`.
* **The wire lookup is a `mux` over two total branches.** A wire below `w` is an
  input bit and is read with `index` over `bits`; a wire at or above `w` is a gate
  output and is read with `member` over the state relation. A graph evaluates both
  branches, so the `index` address is taken `mod w` to stay in range
  unconditionally rather than relying on the mux to guard it.
* **The value relation stores `(wire, bool)`.** A bare set of true wires would
  conflate "false" with "not yet computed"; tagging keeps them distinct and the
  membership test asks for `(wire, true)`. This is the same tagging positional
  reuse needed, for the same reason.

The counter, the mod, and the capacity are all plain integers and sets. Nothing
here is specific to Boolean circuits: the same three-node dispatch would step any
indexed instruction relation.

### 2.3 Measured: the program text does not depend on depth

`measure_cost.py` -> `out/cost.json`. The scaffold built against a depth-1
episode and against a depth-8 episode is the **same program**, by digest:

| depth | nodes | structural symbols | discrete space | execution cost / tick | description bits | program digest |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 60 | 72 | 1,088 | 136 | 484,640 | `c21698de19e06dc5b7f00c47` |
| 2 | 60 | 72 | 1,088 | 136 | 484,640 | `c21698de19e06dc5b7f00c47` |
| 3 | 60 | 72 | 1,088 | 136 | 484,640 | `c21698de19e06dc5b7f00c47` |
| 4 | 60 | 72 | 1,088 | 136 | 484,640 | `c21698de19e06dc5b7f00c47` |
| 6 | 60 | 72 | 1,088 | 136 | 484,640 | `c21698de19e06dc5b7f00c47` |
| 8 | 60 | 72 | 1,088 | 136 | 484,640 | `c21698de19e06dc5b7f00c47` |

What *does* grow with depth is the number of ticks: the answer is settled from
tick `d - 1`, exactly as the construction predicts.

### 2.4 Measured: it computes the circuit, exactly

`check_exact.py` -> `out/exactness.json`. The two searched nodes are hardened to
the reference program and the graph is stepped through `Program.execute` only —
no relaxation, no training. Against the generator's own final wire value, over 40
episodes per depth with `nondegenerate` and `min_relevant_inputs=2`:

| depth | episodes | exact on the settled window | first settled tick, max | first settled tick, mean |
|---:|---:|---:|---:|---:|
| 1 | 40 | 40/40 | 0 | 0.00 |
| 2 | 40 | 40/40 | 1 | 0.60 |
| 3 | 40 | 40/40 | 2 | 0.85 |
| 4 | 40 | 40/40 | 3 | 1.20 |
| 6 | 40 | 40/40 | 5 | 2.75 |
| 8 | 40 | 40/40 | 7 | 3.15 |

The maximum settling tick is exactly `d - 1` at every depth, and the mean is
lower because a circuit whose last gate does not change the answer settles early.

---

## 3. Depth generalization, measured

### 3.1 Method

* Generator `logic`, `inputs=4`, `nondegenerate=True`, `min_relevant_inputs=2`,
  `gate_capacity=8`. FINDINGS section 10 documents why the default draw is
  unusable here: 34% of depth-8 targets are constants. `min_relevant_inputs=2`
  removes constant targets by construction, and the measured baselines below
  confirm no degenerate depth.
* **Train at depths 1 and 2**, alternating every two episodes so each depth meets
  both objectives; 320 episodes; the joint prediction/probe/policy objective and
  the `z/world/policy/prediction/value` tail of `examples/joint.py`, verbatim.
* **Evaluate at depths 1, 2 (seen) and 3, 4, 6, 8 (unseen)** on 64 held-out
  episode addresses (`index 10000..10063`, `split='test'`), deterministic policy.
* **Horizon 12, scored on the trailing 4 ticks.** The recurrence needs `d - 1`
  ticks before the answer exists, so a return summed from tick 0 charges the
  interpreter for its own settling. The trailing window measures the answer once
  the fold has run, on the same 0..4 scale every other logic result in this
  repository uses. **Every baseline is rolled on exactly the same episodes and
  the same window.** This is a measurement convention and it is not free — see
  section 5.
* **8 seeds, with explicit initialization noise (sd 0.05) on the choice logits.**
  `SoftProgram` zero-initializes every logit, so `torch.manual_seed` does not vary
  synthesis at all (FINDINGS section 11, fault P2). Without the noise the eight
  seeds would be one outcome repeated eight times. The `record` arm below is the
  illustration: even *with* noise it produces one identical outcome in all eight
  seeds.

### 3.2 Scaffolds

| name | observations | free choices | space |
|---|---|---|---|
| **record** | `bits`, `goal` | `relation` (16), `goal_relation` (16) | 256 |
| **interpreter, wire lookup pinned** | `bits`, `goal`, `gates` | `relation` (17: the 16 fixed tables plus the table-conditioned `index`), `goal_relation` (16) | **272** |
| **interpreter, wire choice free** | as above | additionally `va`, `vb` (2 each: the general `mux` lookup, or a fixed `project(bits, i)`) | **1,088** |
| **interpreter, no settle mux** | as above | as above | 1,088 |

The 272-program space is deliberately the same size as step 4's lookup scaffold,
so the enumeration reference is directly comparable. The interpreter candidate is
made *legal*, not supplied: it is candidate 16 of 17 at one node.

### Baselines, on the evaluation episodes and the same scored window

| depth | always true | always false | uniform random | best constant | fraction target true |
|---|---|---|---|---|---|
| 1 | 1.88 | 2.12 | 1.89 | **2.12** | 0.469 |
| 2 | 1.88 | 2.12 | 1.92 | **2.12** | 0.469 |
| 3 | 2.38 | 1.62 | 2.08 | **2.38** | 0.594 |
| 4 | 2.50 | 1.50 | 2.11 | **2.50** | 0.625 |
| 6 | 1.94 | 2.06 | 2.11 | **2.06** | 0.484 |
| 8 | 1.56 | 2.44 | 1.86 | **2.44** | 0.391 |

### Exported exact frozen program — mean settled return over 8 seeds x 64 held-out episodes

| scaffold | space | d1 *(seen)* | d2 *(seen)* | d3 | d4 | d6 | d8 |
|---|---|---|---|---|---|---|---|
| record (`bits`,`goal`) | 256 | **2.12** (sd 0.00) | **2.12** (sd 0.00) | **1.62** (sd 0.00) | **1.50** (sd 0.00) | **2.06** (sd 0.00) | **2.44** (sd 0.00) |
| interpreter, no settle mux (ablation) | 1088 | **2.24** (sd 0.33) | **2.20** (sd 0.22) | **1.75** (sd 0.35) | **1.66** (sd 0.46) | **2.13** (sd 0.20) | **2.48** (sd 0.11) |
| interpreter, wire choice free | 1088 | **3.77** (sd 0.43) | **3.69** (sd 0.58) | **3.66** (sd 0.64) | **3.70** (sd 0.55) | **3.66** (sd 0.64) | **3.69** (sd 0.58) |
| interpreter, wire lookup pinned | 272 | **4.00** (sd 0.00) | **4.00** (sd 0.00) | **4.00** (sd 0.00) | **4.00** (sd 0.00) | **4.00** (sd 0.00) | **4.00** (sd 0.00) |
| best constant | — | 2.12 | 2.12 | 2.38 | 2.50 | 2.06 | 2.44 |
| uniform random | — | 1.89 | 1.92 | 2.08 | 2.11 | 2.11 | 1.86 |

### Soft model, deterministic — mean settled return over 8 seeds x 64 held-out episodes

| scaffold | space | d1 *(seen)* | d2 *(seen)* | d3 | d4 | d6 | d8 |
|---|---|---|---|---|---|---|---|
| record (`bits`,`goal`) | 256 | **2.12** (sd 0.00) | **2.12** (sd 0.00) | **1.62** (sd 0.00) | **1.50** (sd 0.00) | **2.06** (sd 0.00) | **2.44** (sd 0.00) |
| interpreter, no settle mux (ablation) | 1088 | **2.20** (sd 0.20) | **2.20** (sd 0.22) | **1.75** (sd 0.35) | **1.65** (sd 0.42) | **2.13** (sd 0.20) | **2.45** (sd 0.04) |
| interpreter, wire choice free | 1088 | **3.06** (sd 0.24) | **3.16** (sd 0.26) | **3.09** (sd 0.27) | **3.15** (sd 0.29) | **3.12** (sd 0.31) | **3.02** (sd 0.29) |
| interpreter, wire lookup pinned | 272 | **3.20** (sd 0.02) | **3.31** (sd 0.00) | **3.25** (sd 0.00) | **3.31** (sd 0.00) | **3.30** (sd 0.02) | **3.19** (sd 0.00) |
| best constant | — | 2.12 | 2.12 | 2.38 | 2.50 | 2.06 | 2.44 |
| uniform random | — | 1.89 | 1.92 | 2.08 | 2.11 | 2.11 | 1.86 |

### What each seed selected

| scaffold | seed | selections | exact d8 |
|---|---|---|---|
| interpreter, wire choice free | 0 | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire choice free | 1 | `{'va': 0, 'vb': 1, 'relation': 16, 'goal_relation': 6}` | 2.75 |
| interpreter, wire choice free | 2 | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire choice free | 3 | `{'va': 0, 'vb': 1, 'relation': 16, 'goal_relation': 6}` | 2.75 |
| interpreter, wire choice free | 4 | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire choice free | 5 | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire choice free | 6 | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire choice free | 7 | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 0 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 1 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 2 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 3 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 4 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 5 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 6 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, wire lookup pinned | 7 | `{'relation': 16, 'goal_relation': 6}` | 4.00 |
| interpreter, no settle mux (ablation) | 0 | `{'va': 1, 'vb': 0, 'relation': 14, 'goal_relation': 12}` | 2.44 |
| interpreter, no settle mux (ablation) | 1 | `{'va': 0, 'vb': 1, 'relation': 4, 'goal_relation': 9}` | 2.44 |
| interpreter, no settle mux (ablation) | 2 | `{'va': 0, 'vb': 1, 'relation': 16, 'goal_relation': 6}` | 2.75 |
| interpreter, no settle mux (ablation) | 3 | `{'va': 0, 'vb': 1, 'relation': 9, 'goal_relation': 6}` | 2.44 |
| interpreter, no settle mux (ablation) | 4 | `{'va': 1, 'vb': 0, 'relation': 2, 'goal_relation': 6}` | 2.44 |
| interpreter, no settle mux (ablation) | 5 | `{'va': 0, 'vb': 1, 'relation': 10, 'goal_relation': 1}` | 2.44 |
| interpreter, no settle mux (ablation) | 6 | `{'va': 0, 'vb': 1, 'relation': 13, 'goal_relation': 6}` | 2.44 |
| interpreter, no settle mux (ablation) | 7 | `{'va': 0, 'vb': 1, 'relation': 9, 'goal_relation': 6}` | 2.44 |
| record (`bits`,`goal`) | 0 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 1 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 2 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 3 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 4 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 5 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 6 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |
| record (`bits`,`goal`) | 7 | `{'relation': 3, 'goal_relation': 3}` | 2.44 |

### Enumeration reference

| scaffold | space | evaluated | exhausted | unique | selection | seconds | held-out d1..d8 |
|---|---|---|---|---|---|---|---|
| interpreter, wire lookup pinned | 272 | 272 | True | True | `{'relation': 16, 'goal_relation': 6}` | 25 | 4.00 / 4.00 / 4.00 / 4.00 / 4.00 / 4.00 |
| interpreter, wire choice free | 1088 | 1088 | True | True | `{'va': 0, 'vb': 0, 'relation': 16, 'goal_relation': 6}` | 98 | 4.00 / 4.00 / 4.00 / 4.00 / 4.00 / 4.00 |

### Gradient reaching each free choice, one training episode at initialization

| node | interpreter (task+regularizer) | interpreter (task only) | no settle mux (task+reg) | no settle mux (task only) |
|---|---|---|---|---|
| `va` | 2.11e-05 | 4.65e-08 | 2.11e-05 | None (unreachable) |
| `vb` | 0.000148 | 3.05e-05 | 9.13e-06 | None (unreachable) |
| `relation` | 0.0102 | 0.0087 | 1.04e-05 | None (unreachable) |
| `goal_relation` | 0.152 | 0.343 | 0.0844 | 0.183 |

---

## 4. Reading the numbers

**1. Depth generalization holds, and it is the same number at every depth.**
Trained only at depths 1-2, the exported exact frozen program of the pinned-wire
interpreter scores **4.00 at depths 3, 4, 6 and 8 in 8/8 seeds, sd 0.00** —
identical to its score at the trained depths, against a best constant of
2.06-2.50 measured on the same episodes and the same window. That is what
generalization over a generating structure looks like: a program that reads the
circuit performs the same on any circuit. Compare step 4, where the equivalent
statement could only be made at depth 1, because no program could accept a
depth-3 episode at all.

**2. The record scaffold is at chance at every depth, including the trained
ones, and cannot do better.** 1.50-2.44 against constants of 2.06-2.50. It never
observes the circuit, and its gate choice is a global logit vector, so one frozen
program computes one Boolean relation. All eight seeds produce the identical
selection and identical returns *despite* the initialization noise — this loss
surface has one attractor and the seed does not move it, which is worth recording
next to fault P2: adding noise is necessary for seed statistics to mean anything,
and it is not sufficient for them to vary.

**3. The searched choice is discovered, not supplied.** The table-conditioned
`index` candidate is candidate 16 of 17 at the `relation` node, and it is
selected in **8/8 seeds in both interpreter arms** — the same count step 4
reported at depth 1, now at every depth. `goal_relation` converges to `truth_6`
(xor with the objective bit) in 8/8 as well. Enumeration over the same space
confirms this pair is the *unique* optimum, which a gradient run cannot
establish.

**4. The `is_final` mux is load-bearing, and the ablation is decisive.** Without
it the answer is read only out of the state relation — equally exact, one tick
slower, and gradient-dead. Measured on one training episode at initialization
with the discreteness regularizer removed so that only the task objective is
measured, the `relation`, `va` and `vb` logits have `grad is None`: not merely
small, but unreachable in the autograd graph. Under the *regularized* objective
`relation` shows 1.04e-05, which is the entropy term and not learning — precisely
the mechanism FINDINGS section 3 identifies as defeating the connectivity guard,
reproduced here in a scaffold built for an unrelated purpose. End to end the
ablation is at chance at every depth (1.66-2.48) and picks the interpreter
candidate in 1/8 seeds, by noise. With the mux, `relation` receives 8.7e-03 of
task gradient and 8/8 seeds find the interpreter.

**5. The wire *binding* choice is gradient-starved, exactly as the perception
ladder predicted.** `relation` chooses among operators at fixed inputs and gets
8.7e-03. `va`/`vb` choose between two different *input bindings* and get 4.7e-08
and 3.1e-05 — three to five orders of magnitude less. The consequence is visible
in the seed table: with the binding free, **6/8 seeds reach 4.00 and 2/8 land on
2.75** because `vb` picked the fixed `project(bits, 1)` shortcut, giving a mean of
3.66-3.77; with the binding pinned, 8/8 reach 4.00. Both arms pick the
interpreter candidate. FINDINGS sections 11 and 14 state this as "relaxation is
strong on value choices at a fixed address and useless on address choices"; this
is a third independent instance, on a task with nothing to do with rasters, and
the two arms differ *only* in whether one binding is searched.

**6. The soft model is well below its own exported program, and that is the
`index` relaxation.** 3.19-3.31 soft against 4.00 exact for the pinned arm. Step
4 saw the same gap for the same reason: `index` relaxes as
`softmax(-(x-k)^2/tau)`, which blurs neighbouring table indices, while the
exported exact program has no blur. The soft number is flat across depth too, so
the gap is relaxation, not depth.

**7. Enumeration settles it, again.** The 272-program space exhausts in 25 s and
the 1,088-program space in 98 s, both certifying a unique optimum and both
reaching 4.00 held-out at every depth — including the binding choice the gradient
path fumbles in 2/8 seeds. Consistent with FINDINGS section 8: where the space is
small, discrete and exactly checkable, the discrete backend is the right one. The
comparison does not isolate the differentiable path's one measured advantage on
this family, environment sample efficiency, since the enumerator spends
`space x 16` rollouts to score candidates.

---

## 5. What remains blocked

Stated precisely, because each is a real limit and none of them is fixed by
trying harder.

1. **Depth is generalized up to a declared capacity, not without bound.** A `set`
   type carries a finite capacity by construction (`ARCHITECTURE.md` section 1:
   "Sets are finite"), the flat width of the observation is `5C`, and the program's
   input port is typed for one `C`. So the result converts "one program per depth"
   into "one program per capacity bound" — a real change in kind, since `C = 8`
   already covers eight distinct depths and the capacity can be declared once at
   the top of a curriculum, but it is not unbounded depth. Unbounded depth would
   need a carrier whose capacity is not part of its type, which is a change to
   section 1 and is **not** proposed here: nothing measured requires it.
2. **The program text is depth-independent; the execution is not.** The answer
   exists from tick `d - 1`, so the horizon must cover the deepest circuit plus
   the scored window. Scoring the trailing 4 ticks is what makes the returns
   comparable to every other logic number in this repository, and it is a
   deliberate choice that hides settling cost. `measure_settling.py` quantifies
   what it hides — the same reference program, the same 64 episodes, three
   scoring conventions:

   | depth | trailing 4 ticks | first 4 ticks | whole horizon, rescaled to /4 | best constant |
   |---:|---:|---:|---:|---:|
   | 1 | 4.00 | 4.00 | 4.00 | 2.12 |
   | 2 | 4.00 | 3.53 | 3.84 | 2.12 |
   | 3 | 4.00 | 3.06 | 3.69 | 2.38 |
   | 4 | 4.00 | 2.59 | 3.53 | 2.50 |
   | 6 | 4.00 | 2.06 | 3.19 | 2.06 |
   | 8 | 4.00 | 2.44 | 3.09 | 2.44 |

   Scored from tick 0 over four ticks the interpreter decays to exactly the
   constant baseline by depth 6, because it has not finished computing. That is a
   latency measurement, not a correctness one. Both readings are honest; they
   answer different questions, and this report answers "does one program compute
   circuits of unseen depth".
3. **Gradient descent does not reliably settle the input-binding choice.**
   Section 4 item 5: 6/8 seeds against 8/8, on two arms that differ only in
   whether one operand binding is searched or declared, with a task gradient
   three to five orders of magnitude smaller at the binding node. The mitigations
   used here are to declare the binding and search the operator choice, or to
   enumerate — which gets it right and certifies it unique. Neither is a fix for
   the relaxation, and this is now the third track to hit the same wall.
4. **`tcn/search.py` cannot address a recurrent program at all.** `evaluate`
   calls `Program.execute(example['inputs'], ...)` with no state and no sequence,
   so the shipped enumerator cannot score any scaffold that uses `Program.state`
   — which is precisely the family that expresses depth generalization. The
   reference in section 3 is a return-scored walk over `candidate_counts`, written
   here because `enumerate_fit` could not be used. Diff proposed in section 6.
5. **`SoftProgram` conflates a settled choice with a gradient boundary.** A node
   declared `selected` is routed through `exact_tensor(...).detach()`, so
   declaring the fixed plumbing of this scaffold as settled severed every choice
   upstream of it and took `relation`'s gradient to `None`. The workaround here is
   to leave every plumbing node `selected=None`, which works but means the
   scaffold cannot say what it means. Diff proposed in section 6.
6. **This scaffold cannot be batched, for the reason FINDINGS section 11 records
   as fault P1.** `relaxed` implements `tuple` as a bare `torch.cat(xs, dim=-1)`,
   which does not broadcast, and the interpreter packs a batched intermediate
   alongside the unbatched constants `true`, `width` and `one` at `a_key`,
   `b_key`, `new_entry` and `answer_key`. `JointTrainer` runs one episode at a
   time so nothing here is batched and the fault never fires; a batched forward
   pass over the same program raises
   `RuntimeError: Tensors must have same number of dimensions: got 2 and 1`,
   confirmed directly. Any attempt to speed this up by batching episodes hits it
   first.
7. **Scope.** One generator, one input width (4), one gate universe (16 two-input
   tables), horizon 12, capacity 8, 64 held-out episodes, 8 seeds. The circuit is
   evaluated, not synthesised: this says nothing about learning a *program* of
   unseen depth, only about interpreting one. And nothing here speaks to
   non-Boolean domains.

---

## 6. Proposed `tcn/` changes, not applied

`tcn/` was not modified — `tcn/search.py` and `tcn/learning.py` are being changed
concurrently by another agent — so both are diffs against the tree as read.

### D1. `SoftProgram` severs the graph at every pre-selected node

`Node.selected` settles a node's discrete choice. `SoftProgram.forward` reads it
as a gradient boundary as well, and `exact_tensor` detaches its inputs, so a
scaffold that declares its fixed plumbing as selected disconnects every soft node
upstream of it. Measured here: the `relation` choice went from 8.7e-03 of task
gradient to `grad is None`. That is the disconnection `ARCHITECTURE.md` section 5
explicitly forbids ("Outside-in freezing must never silently disconnect all
learning signal to the interior"), arriving from the scaffold rather than from
the crystallizer. Keep the two meanings apart:

```diff
--- a/tcn/learning.py
+++ b/tcn/learning.py
@@ class SoftProgram(nn.Module):
         self.temperatures={n.name:1. for n in program.nodes}
         self.frozen={n.name:n.selected for n in program.nodes if n.selected is not None}
+        # A settled choice is not a gradient boundary. `frozen` records which
+        # candidate a node uses; `hard` records which nodes are executed as an
+        # exact, detached boundary, and only crystallization creates one.
+        self.hard=set()
         self.trials={}
@@ def forward
             if n.name in self.frozen:
                 c=n.candidates[self.frozen[n.name]]
-                values[n.name]=exact_tensor(self.registry,c.operator,[values[s] for s in c.sources]).detach()
+                xs=[values[s] for s in c.sources]
+                values[n.name]=(exact_tensor(self.registry,c.operator,xs).detach() if n.name in self.hard
+                                else relaxed(self.registry,c.operator,xs,self.temperatures[n.name]))
                 continue
@@ def freeze
         self.frozen[name]=int(self.choices[i].argmax()) if index is None else index
+        self.hard.add(name)
         self.choices[i].requires_grad_(False); self.trials.pop(name,None)
```

For a `gradient="none"` operator `relaxed` already dispatches to `exact_tensor`,
so every frozen module and every set operator keeps the boundary it has today;
what changes is that a settled *relaxable* node stays in the graph.
`JointTrainer.save`/`load` would need to persist `hard` alongside `frozen`, and
`Crystallizer`'s use of `m.frozen` is unaffected. Not applied, and it should be
re-measured against the crystallization fixtures before landing, because it
changes what a partially crystallized forward pass computes.

### D2. `tcn/search.py:enumerate_fit` cannot drive a recurrence

```diff
--- a/tcn/search.py
+++ b/tcn/search.py
 def evaluate(program, selections, examples, signals, registry, tolerance=None):
     worst = 0.
     for example in examples:
+        # An example may be a sequence of ticks rather than a single input, so
+        # that a program with explicit recurrence can be scored at all. Without
+        # this, a scaffold using `Program.state` has no discrete reference --
+        # and that is the family that expresses folds and depth generalization.
+        steps = example.get('sequence') or [example['inputs']]
+        truths = example.get('target_sequence') or [example['targets']]
+        state = None
         try:
-            _, _, trace = program.execute(example['inputs'], registry=registry, selections=selections)
+            for inputs, targets in zip(steps, truths):
+                _, state, trace = program.execute(inputs, state, registry, selections)
+                for signal in signals:
+                    a = trace[signal.source].flat(); b = targets[signal.target].flat()
+                    worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
         except (ValueError, TypeError, OverflowError, ZeroDivisionError, ArithmeticError, IndexError):
             return None
-        for signal in signals:
-            a = trace[signal.source].flat(); b = example['targets'][signal.target].flat()
-            worst = max(worst, max((abs(x - y) for x, y in zip(a, b)), default=0.))
         if tolerance is not None and worst > tolerance:
             return worst
     return worst
```

A second, smaller gap worth recording rather than patching blind: `enumerate_fit`
scores supervised signals only, so a scaffold whose objective is *return* needs
its own scorer. Step 4 wrote one; so did this track. A return-scored entry point
in `tcn/search.py` would stop that being rewritten per experiment.

### No `ARCHITECTURE.md` amendment is proposed

The condition for one — that sequential evaluation be inexpressible — is not met.
Section 1 already names indexed values plus length as the representation of a
sequence, and section 4 already names explicit recurrence as the mechanism for a
computation that is not a within-tick DAG. Both were used as written. Section 4's
"execution cost is charged per use" is the correct account of the tick budget.

One thing is worth *recording* rather than amending: section 4 says a
crystallized module is registered as one typed candidate operator, "this is
recursive abstraction", and the positional-reuse track added that its positional
form is `hold / pair / map` in three nodes. The *sequential* form is
`insert / pair / filter` over a counter in state, and its output is committed
back into an indexed relation — three nodes to dispatch, one to commit, and the
text does not grow with the number of steps.

---

## 7. Generator change

`generators/logic/generator.py` gains `gate_capacity` (1..64) and, when it is
set, the `gates` observation channel plus the `gate_set_type`, `gate_set_value`
and `gates_from_set` helpers. `initialize` refuses `depth > gate_capacity` and
records `gate_capacity` in the state **only** when configured, so an episode
drawn without it has exactly the state — and snapshot — it had before. Verified
bit-identical over 192 seed/config/split combinations (section 1.3).

`tests/test_gate_observation.py`, 10 tests: the channel absent by default and
present when configured; one type across six depths while `program` has six;
lossless round trip; the index field distinguishing two otherwise identical
gates; capacity declared, enforced, and refusing an overflowing wire index;
element semantics reproducing the generator's own evaluator; the type helpers
agreeing; and the channel carrying exactly the information `program` already
carried.

Full suite: **125 passed**.

---

## 8. Files

| file | what it does |
|---|---|
| `check_default_stream.py` -> `out/default_stream_before.json` | 192-combination bit-identity check of the default stream, capture and compare. |
| `interpreter.py` | The recurrent interpreter scaffold, the record control, the training harness and the baselines. |
| `check_exact.py` -> `out/exactness.json` | The reference program stepped through `Program.execute` against the generator's own wire values, per depth. |
| `measure_cost.py` -> `out/cost.json` | Program text, cost and digest against depth. |
| `measure_settling.py` -> `out/settling.json` | The same program under three scoring conventions, so the trailing-window choice is visible. |
| `research_step4_probe.py` | Asks step 4's own scaffold builder to accept each depth. |
| `run.py` -> `out/*.json`, `out/*.log` | `representation`, `baselines`, `gradients`, the four training arms, and the enumeration reference. |
| `report.py` | Renders `out/*.json` into the tables above. |
