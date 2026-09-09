# Track 8: enumerative, random, and constraint-solver baselines

**The question.** TerpreT (Gaunt et al., 2016, [arXiv 1608.04428](https://arxiv.org/abs/1608.04428))
benchmarked one program-induction specification against four back-ends and found that
*"constraint solvers dominate the gradient descent and LP-based formulations."*
`research/literature/RESULTS.md` §5.5 lists **"TerpreT already answered this. Where is
your constraint-solver baseline?"** as the single most damaging objection to TCN's
framing. TCN's premise is that differentiable relaxation is the right way to search a
typed operator space. This track tests that premise against the obvious
non-differentiable alternatives, over the identical candidate space `tcn/learning.py`
searches, on this repo's own tasks.

---

## 0. Headline

**Brute force recovers the entire discrete content of the repo's flagship joint result —
`relation = truth_6`, `goal_relation = truth_6`, byte-identical to
`artifacts/joint-final/program.json` — by sweeping all 256 candidate programs in
0.081 milliseconds.** The same sweep run through the repo's own exact runtime
(`Program.execute`, 369 program executions) takes 143 ms. One gradient run of
`examples/joint.py` takes 10–36 seconds and delivers the same two-operator program.
The solution is unique, so enumeration also returns a completeness certificate that the
gradient path cannot.

**Crossover verdict: there is no crossover in the gradient method's favour anywhere in
reach, and as difficulty rises the gradient method is the *first* of the three to break,
not the last.** On the hard end of the depth sweep (targets that depend on all four
input bits):

| depth | space | exhaustive enumeration | CDCL SAT | gradient (best known config) |
|---|---|---|---|---|
| 3 | 2^25.8 | 0.53 / 0.86 s ✓ | 0.019 / 0.024 s ✓ | **0.67 / 0.33 success**, ~41 s per run |
| 4 | 2^35.4 | 0.97 / 1.21 s ✓ | 0.019 / 0.029 s ✓ | **0.00 / 0.33 success**, ~118 s per run |
| 5 | 2^45.4 | 2.20 s ✓ / **60 s TIMEOUT** | 0.12 / 0.47 s ✓ | not attempted (already failing at 4) |
| 6 | 2^55.8 | 5.90 / 3.11 s ✓ | 0.73 / 0.77 s ✓ | not attempted |

The gradient path degrades from depth 3 and is at zero on one depth-4 instance.
Enumeration first fails at depth 5. The SAT solver — a hand-written CDCL, i.e. the
*weakest possible* stand-in for a real constraint solver — never failed, and was 4× to
64× faster than enumeration on every hard instance both finished (and closed in 0.471 s
the depth-5 instance enumeration could not finish in 60 s). TerpreT's finding reproduces
on this repo, on this repo's own tasks and its own candidate space.

**The one thing the differentiable path clearly wins.** On the joint RL task, at a
matched budget of **704 environment steps** (what one `JointTrainer` run spends), random
search over the same 256 candidates succeeds **0/20** while the gradient path succeeds
**5/5**. Enumeration needed **16,384 env steps** for its full sweep. Where environment
interaction is the scarce resource rather than CPU, the ordering flips — and that is a
statement about sample efficiency, not about search.

---

## 1. What was built

Everything is under `research/enumerative-baseline/`. Nothing in `tcn/` or `generators/`
was modified and no package was installed into `.venv`.

| file | what it is |
|---|---|
| `sat.py` | a from-scratch **CDCL SAT solver**: two-watched-literal propagation, 1-UIP clause learning, non-chronological backjumping, VSIDS activity with decay, phase saving, Luby restarts |
| `mixed_task.py` | `examples/mixed.py`: exhaustive enumeration, random search, the repo's gradient path |
| `joint_task.py` | `examples/joint.py`: probe-conformance enumeration (direct and through `Program.execute`), live-environment enumeration, random search at two budgets, the repo's gradient path |
| `mixed_constants.py` | mixed discrete/continuous variant: enumeration over structures + derivative-free constant fitting, against `tcn.synthesis.fit` |
| `scaling.py` | the `depth` sweep: enumeration, random search, CDCL SAT on a circuit-synthesis encoding, gradient |
| `noise.py` | partial credit under corrupted targets |
| `headline.py` | independent re-measurement of the load-bearing arms in one process |
| `common.py`, `instrument.py`, `report.py` | accounting, non-invasive instrumentation of the gradient path, table rendering |

Raw measurements are in `out/*.json`; console transcripts in `out/*.log`.

**On the SAT solver.** Neither `python-sat` nor `z3` is present in `.venv`
(`.venv/bin/python -c "import z3"` → `ModuleNotFoundError`; same for `pysat`), and the
brief forbids installing into the project environment, so `sat.py` is written from
scratch. It is validated by (a) a **400-instance random-CNF fuzz against brute force with
0 verdict mismatches on SAT and UNSAT and 0 invalid models**, and (b) pigeonhole
instances, correctly UNSAT. **Every SAT wall-clock number below is an upper bound on what
MiniSat/CaDiCaL/z3 would take**, plausibly by one to two orders of magnitude. The
constraint-solver side of this comparison is therefore handicapped, and it wins anyway.
Its two measured weaknesses — no MaxSAT, so it has no answer to corrupted targets, and
stalls on UNSAT proofs for uniform-random 4-input functions at four or more gates — are
documented in §7 and §10.

## 2. How work is counted

Three counters, because the two sides do different units of work.

- **programs** — complete discrete candidate programs evaluated. Enumeration's natural
  unit; not meaningful for a gradient step, so reported for the discrete methods.
- **op applications** — applications of one candidate operator to one example. This *is*
  comparable. Enumerating a `P`-node program over `N` examples with `C` combinations
  costs `C·P·N`. One forward pass of the soft graph over `N` examples costs
  `Σ_v |K_v| · N`, because **every node evaluates every legal candidate on every forward
  pass** (`tcn/learning.py:100`). A backward pass is charged at 2× a forward.
- **env steps** — generator `step()` calls. The only currency that matters for the RL
  task and the one on which the two sides differ most.

Forward passes, backward passes, `torch.autograd.grad` calls and `Program.execute` calls
in the gradient arms are counted by wrapping the model instance and the `torch.autograd`
entry points (`instrument.py`); nothing in `tcn/` is patched on disk.

**Machine and load.** `Linux-6.17.0-1026-nvidia-aarch64`, Python 3.13.15, torch 2.14.0
CPU, `torch.set_num_threads(1)` on both sides. **The machine carried heavy concurrent
load from other tracks throughout** (`research/search-scaling/run_sweeps.py` with 18
workers, `research/structure-generalization/`, plus unrelated jobs; 1-minute load average
72–86 during the headline run). Absolute milliseconds are inflated for *both* sides.
Every comparison below is between arms that ran **in the same process, back to back**, so
the ratios are the load-bearing numbers; `headline.py` re-measured the two headline
comparisons independently and reproduced the enumeration figure to the microsecond
(0.081 ms both times) and the ratios to within a factor of two. The conclusions are
separated by three to six orders of magnitude and are nowhere near close.

**Fairness to the gradient side.** Every gradient arm uses the configuration the repo or
the relevant research track already settled on — `tcn/cli.py:mixed`'s
`fit(..., steps=300, tolerance=.005)` for `examples/mixed.py`; `examples/joint.py`'s own
`trainer(episodes=160)` for the joint task; and for the depth sweep, the settings
`research/search-scaling/harness.py` uses (lr .05, 500 steps, τ annealed 1.0 → 0.1
geometrically, `init_scale` .1, ramped entropy weight). The joint gradient timing
*excludes* the `Crystallizer` pass that `tcn/cli.py:joint` runs afterwards, which is
generous to the gradient side. §5 additionally sweeps budget and learning rate before
drawing any conclusion about a gradient failure.

---

## 3. Task A — `examples/mixed.py` (pure discrete)

The candidate space `tcn/learning.py` searches for this program:

```
logic       16 candidates   truth_0..truth_15 over (a, b)
conversion   1 candidate    encode : bool -> float
algebra      3 candidates   add | sub | mul over (conversion, x)
analytic     2 candidates   sin | identity over (algebra)
                            ---------------------------------
                            16 · 1 · 3 · 2 = 96
```

`examples/mixed.py` declares **no trainable constants**, so those 96 combinations are the
*entire* search space and exhaustive enumeration is complete and exact. Success = exact
conformance at the repo's own tolerance (`.005`) on all 16 examples, checked with
`Program.execute`.

| method | n | solved | median wall clock | programs | op applications |
|---|---|---|---|---|---|
| **exhaustive enumeration** (stop at first) | 1 | 1/1 | **15.3 ms** | 37 | 272 |
| **exhaustive enumeration** (all 96) | 1 | 1/1 | **40.8 ms** | 96 | 620 |
| random search | 10 | 10/10 | 16.2 ms | 40 | 250 |
| gradient (`tcn.synthesis.fit`, repo config) | 10 | 10/10 | 1524 ms | 80 | 365,728 |

Re-measured in `headline.py`: enumeration 28.1 / 64.2 ms, gradient median 639 ms.

- The solution is **unique**: `logic=truth_6 (xor), algebra=add, analytic=sin`. All
  methods find it.
- **Sweeping the entire space costs 40.8 ms and 620 operator applications; the gradient
  path costs 1524 ms and 365,728** — 37× the wall clock and **590× the operator
  applications** for the same unique program in a space small enough to print.
- Random search is not merely competitive, it is equal: median 40 draws out of 96.
- Gradient seed variance here is timing only. `SoftProgram` initialises choice logits to
  zeros (`tcn/learning.py:79`) and `fit` is otherwise deterministic, so all 10 "seeds"
  produce the identical program. Ten seeds on this task is one deterministic run.

## 4. Task B — `examples/joint.py` (the flagship joint result)

The scaffold has 13 nodes. **Exactly two carry operator choice**, both 16-way truth
tables, so the discrete space is **16 × 16 = 256**. The five constants
(`w0, w1, bias0, bias1, baseline`) are trainable floats.

This task has two honestly different readings and this report keeps them apart. **The
joint RL task is not a fair enumeration target in the same way the pure synthesis task
is**, and the two sub-sections below say exactly where the line falls.

### 4a. The supervised sub-problem — squarely inside enumeration's reach

Which `(relation, goal_relation)` make the program's probe outputs `z` and `world` equal
the generator's probe labels `target` and `gate` on all 8 reachable
`(bit_0, bit_1, goal)` rows? No environment interaction; this is a truth-table
conformance question.

| method | n | solved | median wall clock | programs | op applications |
|---|---|---|---|---|---|
| **exhaustive enumeration** (stop at first) | 1 | 1/1 | **0.034 ms** | 103 | 306 |
| **exhaustive enumeration** (all 256) | 1 | 1/1 | **0.081 ms** | 256 | 738 |
| **exhaustive enumeration through `Program.execute`** (all 256) | 1 | 1/1 | **143 ms** | 256 | 4,797 |
| random search (256 draws) | 20 | 10/20 | 0.142 ms | — | 685 |
| gradient (`examples/joint.py`, 160 episodes) | 5 | 5/5 | 35,924 ms | — | 101,824 |

- The solution is **unique**: `relation = truth_6 (xor)`, `goal_relation = truth_6 (xor)`
  — exactly the program in `artifacts/joint-final/program.json`.
- **Brute force recovers the discrete content of the flagship joint result in 81
  microseconds** with a direct evaluator, or 143 ms if every candidate is executed
  through the repo's own exact runtime. The gradient path takes 10–36 seconds. That is
  a factor of **~4 × 10^5** or **~10^2** respectively, and the gradient figure excludes
  crystallization.
- Random search is the one place systematic enumeration wins outright: with one solution
  in 256 and 256 draws with replacement it hits only ~63% of the time (10/20 measured).
  Enumeration beats it by 0.06 ms.

### 4b. The RL problem as posed — a different problem

Harden each of the 256 assignments, run the frozen agent in the **live generator**, score
actual return over 16 episodes, then verify the selection on 64 disjoint held-out
episodes. Maximum return is 4.

| method | n | solved | median wall clock | env steps | op applications |
|---|---|---|---|---|---|
| **enumeration in the environment** (stop at first) | 1 | 1/1 | 10,470 ms | 6,592 | 171,392 |
| **enumeration in the environment** (all 256) | 1 | 1/1 | 23,832 ms | 16,384 | 425,984 |
| random search, 256 draws | 20 | 14/20 | 12,694 ms | 7,136 | 185,536 |
| **random search at the gradient run's env budget (704 steps)** | 20 | **0/20** | 1,384 ms | 704 | 18,304 |
| gradient (`examples/joint.py`, 160 episodes) | 5 | **5/5** | 35,924 ms | **704** | 101,824 |

Re-measured in `headline.py`: full env sweep 9,437 ms, gradient median 23,740 ms.

Four findings. Three cut against the flagship result; the fourth is the clearest win for
the differentiable path in this whole track.

1. **Enumeration still wins on wall clock** — a complete sweep of the space in 9.4–23.8 s
   against 10–36 s for one gradient run — and it returns a certificate: it has checked
   every program in the space.
2. **The continuous part of this task is already solved at initialisation.**
   `examples/joint.py` declares `w0=-2, w1=2, bias0=1, bias1=-1`, so
   `logit0 = -2z+1` and `logit1 = 2z-1`, and `argmax` is already the correct decision
   rule at both `z=0` and `z=1` before any training
   (`joint_task.py:constants_probe()` → `policy_correct_at_initialization = true`). The
   only thing learning must supply is the 256-way discrete choice. The enumeration arm
   inherits the same initialisation, which is what makes it a fair substitute here —
   and it is also why this task is weaker evidence for joint learning than it looks.
3. **Reward alone cannot identify the reference program.** The full environment sweep
   finds **two** behaviourally optimal assignments: `(6,6)` and `(9,9)`. `(9,9)` is
   `xnor` twice, which computes `z = gate XOR goal = target` correctly while computing
   `world = NOT gate` — wrong on the `world` probe, invisible to the reward. The
   supervised probe formulation has exactly one solution; the reward formulation has two.
   Any claim that the joint task "recovers the program" is a claim about the probe
   objective, not about the reward.
4. **The gradient path is 23× more environment-efficient, and that is a real advantage.**
   At the gradient run's own budget of 704 environment steps, random search over the same
   256 candidates succeeds **0/20**, because scoring even 11 candidates on 16 episodes
   exhausts the budget. Enumeration needed 16,384 env steps; the gradient run needed 704
   and succeeded 5/5. **Where environment interaction is the scarce resource rather than
   CPU, the ordering flips.**

**How many episodes does enumeration need to pick correctly?** Scoring by return is
noisy, and the sweep quantifies it:

| selection episodes | assignments scoring a perfect 4.0 | first one found | full 256-sweep time |
|---|---|---|---|
| 2 | 64 | `(0,3)` ✗ | 4.4 s |
| 4 | 12 | `(1,11)` ✗ | 3.9 s |
| 8 | 6 | `(1,11)` ✗ | 21.1 s |
| 16 | **2** | **`(6,6)` ✓** | 30.1 s |
| 32 | 2 | `(6,6)` ✓ | 74.0 s |
| 64 | 2 | `(6,6)` ✓ | 118.3 s |

The set converges to the two reward-equivalent programs at 16 episodes and stays there.
Below that, a stop-at-first enumerator returns the **wrong** program: at 2 episodes a
quarter of the space scores a perfect 4.0 and the first hit is `(0,3)`, a constant-`True`
answerer that got lucky. This is why the environment arm verifies its pick on 64 disjoint
held-out episodes rather than trusting the selection score, and it is a concrete way in
which environment-scored enumeration is *not* the same procedure as truth-table
enumeration.

## 5. Task C — discrete structure plus a trainable continuous constant

`examples/mixed.py` has no trainable constants, so the hybrid arm the brief asks for was
built on a variant assembled from the same operators, with one:
`scale = mul(conversion, k)` with `k` trainable, target `sin(1.7·[a xor b] + x)`,
supervised **only at the output** — dense per-node probes would hand the constant to both
methods and make the comparison vacuous. Space: 96 discrete structures × 1 continuous
parameter. The hybrid arm enumerates the 96 structures and fits `k` inside each by a
**derivative-free** 1-D search (161-point grid then golden section) on the exact forward
map; no gradients anywhere.

| method | n | solved | median wall clock | notes |
|---|---|---|---|---|
| **hybrid: enumeration + 1-D fit** (stop at first) | 1 | 1/1 | **287 ms** | `k` recovered to 2.0 × 10⁻⁸ |
| **hybrid: enumeration + 1-D fit** (all 96) | 1 | 1/1 | 726 ms | same solution |
| gradient (`fit`, 600 steps, lr .05) | 10 | **0/10** | 3426 ms | correct structure, `k = 1.7395`, error 3.9 × 10⁻² |

The gradient path **finds the right discrete structure every time** (`truth_6`, `add`,
`sin`) and then misses exact conformance because the constant is 3.9 × 10⁻² away from
1.7 — far outside the 10⁻³ tolerance — so `fit` reports `exact_conformance=False` and
`fully_frozen=False`. This is a constant-fitting failure, not a search failure. Before
concluding anything, the budget and learning rate were swept:

| steps | lr | exact conformance | learned `k` | error | seconds |
|---|---|---|---|---|---|
| 600 | .05 | ✗ | 1.739464 | 3.9 × 10⁻² | 3.2 |
| 600 | .01 | ✗ | 1.892987 | 1.9 × 10⁻¹ | 3.2 |
| 600 | .005 | ✗ | 1.988367 | 2.9 × 10⁻¹ | 3.2 |
| 2000 | .05 | ✗ | 1.707319 | 7.3 × 10⁻³ | 5.3 |
| 2000 | .01 | ✗ | 1.758549 | 5.9 × 10⁻² | 7.8 |
| 2000 | .005 | ✗ | 1.827551 | 1.3 × 10⁻¹ | 6.4 |
| **6000** | **.05** | **✓** | 1.699888 | 1.1 × 10⁻⁴ | 7.7 |
| 6000 | .01 | ✗ | 1.705837 | 5.8 × 10⁻³ | 25.7 |
| 6000 | .005 | ✗ | 1.715562 | 1.6 × 10⁻² | 13.2 |

**Given 10× the default budget, the gradient path does converge** — 6000 steps at lr .05,
7.7 s, error 1.1 × 10⁻⁴. So the honest statement is not "gradient cannot fit constants";
it is that **a derivative-free 1-D search inside an enumerated structure hit the constant
to 8 significant figures in 287 ms, and the relaxation needed 6000 steps and 7.7 s to
reach 4 significant figures — 27× slower, on one parameter.** `synthesis.fit` has no
learning-rate schedule, which is the proximate cause and is a cheap fix.

The scaling caveat matters here and points the other way: a grid-plus-golden-section fit
is `O(grid^p)` in the number of continuous parameters, so this hybrid does not survive
past one or two constants. The standard answer (NEAR, DreamCoder, `∂4`) is to use
gradients for the *parameters inside a fixed structure* while searching the structure
discretely — which is not the same thing as relaxing the discrete choice, and is exactly
what these numbers support.

## 6. Task D — scaling with `depth`

Task family, identical to `research/search-scaling/harness.py` so the two tracks'
numbers are comparable (that file is not imported; `scaling.py` is self-contained): a
real `logic` episode at configuration `{'depth': d}` supplies the target — 4 input bits,
`d` randomly wired two-input gates with random 16-way tables, output = last gate — and
the scaffold is `d` Boolean nodes whose candidate sets are exactly what
`tcn/graph.py:legal_candidates` produces for a bool-output node over the bounded
predecessor pool:

```
|K_k| = 16 tables × (4+k)^2 ordered source pairs
|space(d)| = Π_{k<d} 16 (4+k)^2
d=1  2.6e2    d=2  1.0e5    d=3  5.9e7    d=4  4.6e10    d=5  4.7e13    d=6  6.1e16
```

Random search is given **the gradient run's own evaluation budget**:
`3 · steps · Σ_k |K_k| / d` draws (384k at d=1 rising to 1.08M at d=6).

### 6a. The generator's own `depth` dial does not make the problem harder

| d | log₂ space | median solution density | enumeration | SAT | random @ matched budget | gradient success | gradient s/run |
|---|---|---|---|---|---|---|---|
| 1 | 8.0 | 7.8 × 10⁻³ | < 1 ms ✓ | 0.002–0.015 s ✓ | 2/2 | 1.00 | 28–42 |
| 2 | 16.6 | 1.9 × 10⁻² | ≤ 9 ms ✓ | 0.022–0.054 s ✓ | 2/2 | 0.83 | 49–84 |
| 3 | 25.8 | 2.1 × 10⁻² | < 1 ms ✓ | 0.018–0.020 s ✓ | 2/2 | 1.00 | 89–117 |
| 4 | 35.4 | 8.7 × 10⁻² | < 1 ms ✓ | 0.016–0.037 s ✓ | 2/2 | 1.00 | 89–101 |
| 5 | 45.4 | 9.2 × 10⁻² | < 1 ms ✓ | 0.017–0.033 s ✓ | 2/2 | — | — |
| 6 | 55.8 | 9.7 × 10⁻² | < 1 ms ✓ | 0.033–0.050 s ✓ | 2/2 | — | — |

**Solution density goes *up* with depth on the generator's own distribution.** Random
`logic` circuits collapse: at depth 5 and 6 two of the four targets drawn were the
constant function or a single input bit (`mask=0x0000`, `0xff00`). More gates means more
ways to realise the same simple function, so the search space explodes while the problem
gets *easier*. `depth` is a space-size dial, not a difficulty dial.

A separate measurement makes this precise. For each depth `d`, targets were drawn from
the generator and the SAT solver was asked whether the same function is realisable with
`d-1` gates; UNSAT is a certificate that the draw genuinely needs `d`:

| depth drawn | draws | draws that genuinely need `d` gates |
|---|---|---|
| 2 | 33 | 3 |
| 3 | 200 | 1 |
| 4 | 200 | **0** |
| 5 | 200 | **0** |
| 6 | 200 | **0** |

At depth 4 and above, **not one draw in 200 needed as many gates as the generator used**.
Any scaling claim that varies `generators/logic`'s `depth` and reports accuracy is
varying candidate-set size while target complexity stays flat or falls. This is worth
flagging to track 3.

### 6b. Hard targets — where the crossover actually is

To get a real difficulty gradient inside the same distribution, targets are
rejection-sampled to those that depend on **all four input bits** (`relevant_inputs == 4`;
impossible at d ≤ 2, since two gates can reach at most three inputs).

| d | log₂ space | density | enumeration (to first solution) | leaf programs | SAT | SAT conflicts | random @ matched budget | gradient success | gradient s/run |
|---|---|---|---|---|---|---|---|---|---|
| 3 | 25.8 | 2 × 10⁻⁵ / < 10⁻⁵ | 0.528 s ✓ / 0.862 s ✓ | 4.7M / 7.8M | **0.019 s ✓ / 0.024 s ✓** | 226 / 308 | 2/2 (18k / 395k draws) | 0.67 / 0.33 | 40 / 41 |
| 4 | 35.4 | < 10⁻⁵ / 3 × 10⁻⁵ | 1.210 s ✓ / 0.965 s ✓ | 10.6M / 7.9M | **0.019 s ✓ / 0.029 s ✓** | 109 / 211 | 2/2 (634k / 16k draws) | **0.00** / 0.33 | 124 / 112 |
| 5 | 45.4 | < 10⁻⁵ | 2.196 s ✓ / **60 s TIMEOUT** | 18.7M / 441M | **0.121 s ✓ / 0.471 s ✓** | 205 / 2867 | 1/2 (**912k draws, miss**) | — | — |
| 6 | 55.8 | < 10⁻⁵ / 2 × 10⁻⁵ | 5.901 s ✓ / 3.107 s ✓ | 50.7M / 26.9M | **0.726 s ✓ / 0.771 s ✓** | 4657 / 4745 | 2/2 | — | — |

Every returned program was verified against the target truth table
(`verified: true` on every solved row).

**Order of failure, which is the whole answer to the framing question:**

1. **Gradient fails first**, at depth 3–4. Its expected wall clock to a *verified*
   solution, accounting for restarts (`time_per_run / success_rate`), is **60–123 s at
   depth 3**; at depth 4 it is **335 s** on the one instance where it ever succeeded and
   **unbounded** on the other, where 3 of 3 seeds failed.
2. **Enumeration fails second**, at depth 5, where one instance exhausted a 60 s budget
   after 441 million leaf programs.
3. **SAT never failed.** On the seven hard instances both methods finished it was
   **4× to 64× faster than enumeration**; on the depth-5 instance enumeration could not
   finish in 60 s, SAT closed it in 0.471 s. Its hardest instance cost 0.77 s and 4,745
   conflicts — nowhere near the ceiling. And this is a hand-rolled solver.

Measured enumeration throughput on the hard instances (the ones large enough to time
reliably) is **8.63 × 10⁶ leaf programs/s**, giving the projected cost of a *complete*
sweep:

| d | space | full sweep |
|---|---|---|
| 1 | 2.6 × 10² | 0.03 ms |
| 2 | 1.0 × 10⁵ | 12 ms |
| 3 | 5.9 × 10⁷ | 6.8 s |
| 4 | 4.6 × 10¹⁰ | 1.5 h |
| 5 | 4.7 × 10¹³ | 64 days |
| 6 | 6.1 × 10¹⁶ | 2.3 × 10² years |
| 7 | 9.8 × 10¹⁹ | 3.6 × 10⁵ years |
| 8 | 1.9 × 10²³ | 7.0 × 10⁸ years |

So enumeration's *guarantee* becomes unaffordable somewhere around depth 4–5. **That is
where a better search is needed — and the method that takes over is the constraint
solver, not the relaxation.** There is no depth at which the gradient path is the best
available method on this family: it is dominated by enumeration where enumeration is
cheap, and it has already stopped working by the time enumeration stops being cheap.

## 7. Partial credit under noisy targets

The strongest fair-to-gradient argument is that a relaxation gives partial credit when
nothing conforms exactly, whereas a decision procedure only answers yes/no. That is true
of SAT *as a decision procedure* and false of enumeration in general — enumeration with a
scoring objective is a MaxSAT-style optimiser. Protocol: draw a depth-2 target from
`generators/logic`, flip `f` of its 16 truth-table rows, hand the corrupted table to each
method, and ask whether it recovers the **uncorrupted** function.

| flipped rows | enumeration (min Hamming) recovers | enumeration time | SAT returns a model | gradient recovers | gradient time |
|---|---|---|---|---|---|
| 0 | 4/4 | 20 ms | 4/4 | 1.00 | 15.7 s |
| 1 | **4/4** | 29 ms | 0/4 (UNSAT, correctly) | 0.50 | 29.4 s |
| 2 | **3/4** | 28 ms | 0/4 | 0.12 | 31.3 s |
| 3 | **2/4** | 23 ms | 0/4 | 0.12 | 27.3 s |

**Partial credit is not an advantage of the relaxation.** Enumeration with a
minimum-Hamming objective recovers the clean function strictly more often than the
gradient path at every noise level, ~1000× faster. What *is* true is that a pure SAT
decision encoding returns UNSAT the moment any row is corrupted; the standard fix is
MaxSAT, which was not available here and is the one measured weakness of the
constraint-solver arm.

## 8. What the differentiable path buys that enumeration does not

Stated plainly, because the rest of this report is one-sided.

1. **Environment sample efficiency (measured, real).** §4b: 704 env steps and 5/5, versus
   0/20 for random search at the same budget and 16,384 env steps for a full enumerative
   sweep. When rollouts are the expensive resource — the regime the architecture is
   ultimately aimed at — gradient wins by more than an order of magnitude, and neither
   enumeration nor SAT has an answer for a live non-differentiable environment beyond
   "score every candidate by running it".
2. **Continuous parameters (partly).** §5: gradient recovers a trainable constant to
   1.1 × 10⁻⁴ given enough budget. The derivative-free hybrid beat it here on one
   parameter, but it is `O(grid^p)` and will not survive more than one or two.
3. **Anytime behaviour on structures nobody can enumerate.** At depth 8 a complete sweep
   is 7 × 10⁸ years; the gradient path returns *something* in 500 steps. It just was not
   the right program at depth 4.
4. **Partial credit under noise — no.** §7 measures this and enumeration wins.
5. **Typing as a search prior — shared, not differentiable-specific.** The typed candidate
   sets are what keep `|K_k|` at 16(4+k)² instead of the untyped product, and all four
   methods benefit identically. That is a claim about `tcn/graph.py`, not about
   `tcn/learning.py`.

## 9. Verdict

**Differentiable search is not currently earning its keep on this repo's own synthesis
benchmarks.**

- On `examples/mixed.py` (96 programs) and the discrete content of `examples/joint.py`
  (256 programs), **exhaustive enumeration is complete, exact, and 37× to 4 × 10⁵ times
  faster**, and returns uniqueness certificates the gradient path cannot. Both flagship
  results are inside brute force's reach by three to five orders of magnitude. This must
  be stated plainly in any document that presents them as evidence for the approach.
- On the depth sweep, **the gradient path is the first method to break** — degrading at
  depth 3 and hitting zero at depth 4 on hard targets — while enumeration holds to depth 5
  and a hand-written CDCL solver holds through depth 6 at under a second. TerpreT's
  conclusion reproduces here, on TCN's own tasks and TCN's own candidate space.
- The premise "differentiable relaxation is the right way to search a typed operator
  space" is **not supported by any measurement in this track**. What the measurements do
  support is a different and narrower claim: relaxation is worth its cost **when the
  objective is only reachable through a non-differentiable environment and rollouts are
  the scarce resource**, which is precisely §4b and precisely the regime
  ARCHITECTURE.md §8 describes.

Recommended consequences:

1. **Add a discrete baseline to every synthesis experiment in this repo.** `scaling.py`'s
   enumerator and `sat.py` are self-contained and cost milliseconds; there is no excuse
   for reporting a gradient synthesis result without one alongside.
2. **Reframe the claim.** The defensible version of TCN's thesis is NEAR-shaped: the
   relaxation is a *heuristic guiding discrete search*, or the mechanism for the parts
   of a program that are continuous or only reachable through an environment — not the
   search itself. `research/literature/RESULTS.md` §4.4 already identifies NEAR as the
   correct architectural answer to TerpreT; this track's numbers are the local evidence
   for that recommendation.
3. **Stop using `generators/logic`'s `depth` as a difficulty dial** without also
   reporting solution density or target complexity. §6a shows it currently inflates the
   candidate space while making the target *easier*.
4. **Give `synthesis.fit` a learning-rate schedule.** §5 shows a trainable constant
   missing a 10⁻³ tolerance by 3.9 × 10⁻² purely because Adam at a fixed lr .05 never
   settles; the fix is one line and turns a reported failure into a success.
5. **Where the joint task is cited, cite the probe objective.** §4b finding 3: the reward
   admits two optimal programs, `(6,6)` and `(9,9)`; only the probe objective identifies
   the reference one.

## 10. Limitations

- **The SAT solver is hand-written.** It is fuzz-validated for correctness (§1) but is
  not competitive with MiniSat/CaDiCaL/z3; its numbers are upper bounds. It also stalls
  on UNSAT proofs for uniform-random 4-input functions at ≥ 4 gates, which is why §6b
  uses "all four inputs relevant" as the hardness filter rather than certified minimal
  circuit size. A production solver would sharpen §6b, not soften it.
- **Small n.** The depth sweep uses 2 targets per depth and 3 gradient seeds per target
  (12 gradient runs per source), constrained by the ~100 s cost of one gradient run under
  the machine's load. Track 3's sweeps carry 8–12 seeds on the same family and should be
  read alongside §6.
- **Heavy concurrent machine load throughout** (§2). Ratios are within-process; absolute
  times are inflated on both sides.
- **`gradient_max_depth = 4`** in `scaling.py`: depth 5 and 6 gradient runs were not
  attempted, because the method was already at 0.00–0.33 success at depth 4 and each run
  would cost several minutes. This is stated rather than extrapolated.
- **§6b's hard targets are still generator-drawn**, filtered for input relevance. They
  are not certified minimal circuits, so "hard" here means "low measured solution
  density" (< 10⁻⁵ at the 100k-draw resolution), which is the operative quantity for
  enumeration and random search but only indirectly so for SAT.
- **The joint environment arm inherits the declared constant initialisation** rather than
  learning it (§4b finding 2). This is stated, quantified, and is the reason the
  supervised and RL readings are reported separately rather than merged.

## 11. Reproduction

```bash
bash research/enumerative-baseline/run_all.sh    # everything, in order

# or one arm at a time:
.venv/bin/python research/enumerative-baseline/mixed_task.py
.venv/bin/python research/enumerative-baseline/joint_task.py
.venv/bin/python research/enumerative-baseline/mixed_constants.py
.venv/bin/python research/enumerative-baseline/scaling.py generator 1 2 3 4 5 6
.venv/bin/python research/enumerative-baseline/scaling.py hard      3 4 5 6
.venv/bin/python research/enumerative-baseline/noise.py 2
.venv/bin/python research/enumerative-baseline/headline.py
.venv/bin/python research/enumerative-baseline/report.py     # renders the tables above
```

All scripts write JSON to `research/enumerative-baseline/out/`. They import from `tcn/`
and `generators/` read-only and modify nothing.
