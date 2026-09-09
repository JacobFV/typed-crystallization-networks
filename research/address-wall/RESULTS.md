# The address wall — 2026-09-09

**Verdict: soft addressing is fixable, and the finding that motivated this track
was misattributed.** The perception ladder's failing arm does not fail because a
mixture of addresses denotes nothing. It fails because `eq`'s surrogate is
`exp(-(a-b)^2/1)`, which is exactly `0.0` in float32 at `|a-b| >= 11`, while the
uniform mixture over 12 raw `geometry` bytes sits 25.6 away from the constant it
is compared against. **Every address logit gradient in that arm is exactly zero.**
Scaling that one surrogate to the declared width of `int[8]` — a temperature of
`2^8 = 256`, derived from the representation, not tuned — takes the recorded
`rung3:centre_free_address` arm from **0/12 to 12/12 held-out-exact at 9,216
programs** and from 0/12 to 7/12 at 576. The proposed diff was applied literally
(section 7, **D1**), not emulated, and it puts **both free addresses on the
centre pixel in 24 of 24 runs across the two resolutions**, against 0 of 24
shipped.

That is one of **three** distinct mechanisms found and separated here. Only one
of them matches the stated explanation, and it is not the one the ladder hit:

| mechanism | where it bites | landscape character | fixable |
|---|---|---|---|
| **M1 mean confound** | a discrete address mixed *linearly*, when addresses differ in mean | convex, reference is the unique global optimum, **first step misleading** | yes, and it self-corrects — Adam recovers 1.000 unless the offset is byte-scale |
| **M2 surrogate saturation** | a discrete address behind a nonlinearity whose surrogate is scaled wrong | **flat: gradient exactly 0.0**, or misleading when partly alive | **yes** — scale the surrogate to the representation (0/12 → 12/12, addresses 0/24 → 24/24) |
| **M3 kernel locality** | the `index` operator's continuous address | **multi-modal**, 5–25 local minima, basin 1.3–2.5 addresses wide | partly — annealing gives 0/12 → 12/12 on uncorrelated data, 0/12 → 2/12 on byte-scale data |

The stated explanation is **falsified for the discrete mechanism and confirmed
for the `index` operator**. On the decisive control — a smooth array against the
*same array with its positions permuted*, identical values, identical marginals,
only the arrangement changed — the discrete route is 11/12 and 12/12 in **both**
arrangements, unchanged; the `index` route goes 0.806 → 0.117.

Everything below was measured on this worktree with the repository `.venv`.
Nothing under `tcn/` or `generators/` was modified; four proposed diffs are in
section 7. Total cost of every measurement in this document: **4,596,212 node
evaluations, 874 s** of single-threaded compute on a host at load average 23–25.

---

## 1. The instrument, and the measurement warnings

`instrument.py` builds one node, one address choice, a known reference address,
and data whose two candidate causal factors can be dialled independently:

* **neighbour correlation** — a Gaussian-process length scale over positions,
  from 0 (independent addresses, the "unrelated values" regime) to 8 (a smoothly
  varying array over 16 positions);
* **per-address mean spread** — a constant offset per address, fixed across
  examples, which changes only how much the addresses differ in mean and leaves
  the correlation structure of the fluctuations untouched.

Two mechanisms are instrumented, because TCN spells an address two ways:

```text
DISCRETE   one node, candidates project(arr, index=i) for every i.
           SoftProgram mixes them with softmax(logits).  This is what
           graph.legal_candidates produces for a tuple port, and it is the
           mechanism the perception ladder measured.
INDEX      one node index(arr, addr), addr a trainable constant.
           learning.relaxed softmaxes -(addr - arange)^2 / temperature.
```

and two downstreams: **direct** (the target is the addressed value) and **eq**
(the target is `eq(addressed value, constant)`, the ladder's own shape).

**Warnings honoured, all re-verified here** (`checks.py`, `out/checks.json`):

* `SoftProgram` zero-initialises every choice logit. Four `torch.manual_seed`
  values give one identical all-zero initialisation — confirmed. **Every
  per-seed number in this document adds explicit Gaussian logit noise at a
  stated scale (0.5 unless said otherwise) and varies the data seed.** Nothing
  here is a seed statistic over identical initialisations.
* `relaxed("tuple", ...)` raises
  `RuntimeError: Tensors must have same number of dimensions: got 2 and 1` on a
  batched value and an unbatched constant, while `add` on the same pair
  broadcasts to `[8, 1]` — confirmed. Nothing here tuples a constant with an
  intermediate.
* `eq`'s surrogate reaches exactly `0.0` at gap **11** (1.6e-28 still
  representable at 8, 3.8e-44 at 10) — confirmed. This is load-bearing below.
* Node evaluations are reported with wall clock everywhere.

Two further defects were found and are new (section 7): `relaxed("index")` does
not reduce to `Registry.exact` at an integer address, and `SoftProgram` uses one
temperature per node for two unrelated jobs.

---

## 2. Landscape characterisation

"Worse than chance" conflates three situations with three different fixes. They
are separated here by mapping the loss over the whole parameter space.

### 2.1 The discrete route with direct supervision is **convex, and the reference is the unique global optimum**

With direct supervision the relaxed loss of a discrete address node is *exactly*
a quadratic in the mixture weights:

```text
L(p) = p'Cp - 2 (C e_k)'p + C_kk,    C = E[x x'],  k = the reference address
```

`C` is a positive-definite second-moment matrix (measured minimum eigenvalue
0.27 on the centred regime), so `e_k` is the unique minimiser over the simplex.
Measured, not assumed:

* **the reference vertex has rank 0 among all 16 vertices in 48/48 direct
  configurations and 32/32 `eq` configurations**;
* 400 Dirichlet mixtures per configuration, 38,400 in total: **not one interior
  mixture beat the reference vertex** at 16 addresses;
* an **exhaustive** grid over the whole 3-simplex at 4 addresses (3,276 points
  per map, 0.04 spacing): the reference vertex is the global minimum in 9/9
  direct maps and 10/12 `eq` maps. The two exceptions are real but tiny —
  `[0, 0, 0.96, 0.04]` beating the vertex by 4.8e-4 and 3.1e-3.

So the failure is **not** `DISPLACED`. The relaxed optimum is where it should
be. This confirms and sharpens the ladder's `excursion.py` result from the
other side.

### 2.2 What the first gradient step actually ranks

Differentiating the quadratic through the softmax, the candidate the first Adam
step raises is

```text
argmax_i [ C_ik - mean_j C_ij ]
  = argmax_i [ mu_i (mu_k - mean_j mu_j)   +   Sigma_ik - mean_j Sigma_ij ]
             \______ MEAN term ______/       \____ COVARIANCE term ____/
```

The covariance term is the one that identifies the address (it peaks at `i = k`
because `Sigma_kk` is the largest entry of its row). The mean term identifies
nothing — it ranks addresses by their mean. **The closed form agrees with
autograd's argmin in 48/48 measured cases.** And the two terms are not
comparable in size on real data:

| regime | mean term | covariance term | gradient picks the reference |
|---|---|---|---|
| `iid_centred` | 0.014 | 0.166 | **8/8** |
| `iid_mean_spread` | 6.63 | 0.166 | 2/8 |
| `bytelike` (offset 128) | 5,010 | 0.166 | 1/8 |

This is **M1**: a first step that is `MISLEADING`, deterministically, by a
signal that has nothing to do with addressing. It is also the mildest mechanism,
because the landscape underneath is convex — full optimisation recovers the
reference at rate 1.000 in **35 of the 36 sweep cells** of section 3.2 (12 data
seeds each), and fails only at byte-scale offsets (3/8).

### 2.3 The `index` route is multi-modal, and its relaxation is not exact anywhere

The `index` address is one scalar, so the landscape can be mapped *exhaustively*:
3,001 grid points at 0.005 spacing over `b ∈ [0, 15]`, at two reference
addresses on opposite sides of the midpoint (the softmax kernel is truncated at
both ends of the tuple, which raises the loss there and would otherwise be
mistaken for signal).

| regime | downstream | local minima | basin width | global min at reference | Adam from 15 starts reaches it |
|---|---|---|---|---|---|
| `smooth_centred` | direct | **1.4** | **6.21** | 8/8 | **0.833** |
| `iid_centred` | direct | 5.2 | 2.50 | 8/8 | 0.350 |
| `iid_mean_spread` | direct | 4.9 | 1.54 | 6/8 | 0.200 |
| `bytelike` | direct | 4.9 | 1.50 | **1/8** | 0.100 |
| `smooth_centred` | eq | 2.2 | 5.13 | 8/8 | 0.600 |
| `iid_centred` | eq | 7.6 | 1.27 | 8/8 | 0.125 |
| `iid_mean_spread` | eq | **25.4** | 0.48 | 2/8 | 0.083 |

This is **M3**, and the cause is visible in the operator: at the shipped
temperature of 1.0 over 16 positions, `softmax(-(b-i)^2/1)` puts 0.564 of its
mass on the addressed element and 0.208 on each immediate neighbour. The kernel
spans about two addresses, so the loss as a function of `b` is essentially a
resampled copy of the data — smooth and single-basined when the array is smooth,
white noise when it is not.

The same fact is a correctness defect independent of search: **`relaxed("index")`
does not agree with `Registry.exact` even at an exactly integer address**
(mean absolute error 0.332 on unit-variance data at temperature 1.0; 0.0015 at
0.145; 3e-10 at 0.05). The relaxation does not reduce to the operator it relaxes
at any point of its own domain.

### 2.4 The arm that actually fails: `rung3:centre_free_address`, autopsied per node

`pointing.py` reproduces exactly as recorded on this worktree (0.250 against
chance 0.292 for `centre_free_address_R2`, 0.812/0.004 for the colour search,
0.667/0.333 for the pinned mask). But that aggregate mixes node roles and
silently drops every node whose gradient is exactly zero. Split by role
(`ladder_autopsy.py`, 8 perturbed starts, 24 episodes):

| arm | node role | nodes | **dead gradient** | scored | picks reference | chance | median \|grad\| |
|---|---|---|---|---|---|---|---|
| free addresses | **address** | 16 | **6** | 10 | **0.000** | 0.083 | **4.55** |
| free addresses | colour | 16 | 6 | 10 | 0.500 | 0.500 | 0.349 |
| pinned addresses | colour | 64 | 0 | 64 | **1.000** | 0.250 | 0.151 |
| pinned addresses | logic | 32 | 0 | 32 | 0.000 | 0.500 | 0.145 |

The address nodes are not near chance and not flat-when-scored: they pick the
reference **0 of 10 times** with a gradient **13x larger** than the colour
nodes'. And the mechanism is legible in one forward pass:

```text
uniform mixture over the 12 bytes ....... 49.64
the reference byte ...................... 40.25   (mean over episodes)
the constant it is compared against ...... 24
eq surrogate at the mixture ............. 3.18e-31
eq surrogate at the reference ........... 0.833
```

The mixture lands 25.6 from the constant, where `exp(-d^2/1)` is numerically
dead. In that regime the only way the loss can fall is by moving the mixture's
**value** toward the constant, so the gradient ranks addresses by *how close
their value is to the constant*, not by whether they are the right address.
Directly confirmed: with the colour constants pinned so the addresses are the
only free choice, the picks at temperature 10 are `{0, 3, 6, 9}` — and the four
addresses ranked nearest the constant 24 by mean are `9, 6, 10, 7, 0`, while the
per-episode nearest address is 0 in 18 of 24 episodes and 3 in the other 4.
**The gradient returns the value-proximity ranking.** That is **M2**.

With the colours pinned (144 programs, `eq_temperature.py`), at the shipped
temperature the ranking is not even available:

| `eq` temperature | address gradients dead | scored | picks reference | chance | median \|grad\| |
|---|---|---|---|---|---|
| **1** (shipped) | **16/16** | 0 | — | 0.083 | 0 |
| 10 | 0 | 16 | 0.125 | 0.083 | 3.08 |
| 100 | 0 | 16 | 0.250 | 0.083 | 1.51 |
| 1,000 | 0 | 16 | **0.438** | 0.083 | 0.39 |
| 10,000 | 0 | 16 | **0.438** | 0.083 | 0.12 |
| 1,000,000 | 0 | 16 | 0.375 | 0.083 | 0.10 |

**At the shipped temperature the address gradient is exactly zero — the
landscape is `FLAT`, not misleading.** The "worse than chance 0.25" in
`FINDINGS` section 11 is the average of dead nodes being dropped and the
surviving, partly-alive nodes returning a value-proximity ranking.

### 2.5 Does M1 explain the ladder? No — and saying so matters

`real_data.py` takes the actual `geometry` bytes and measures the two factors:

| | addresses | value range | per-address mean sd | within-address sd | **mean spread / sd** |
|---|---|---|---|---|---|
| R=2 | 12 | 16–207 | 8.45 | 35.45 | **0.238** |
| R=4 | 48 | 14–219 | 10.51 | 36.38 | **0.289** |

That is a *low* mean spread — pixels are mostly background, so the addresses
have nearly equal means. Consequently a plain direct read of a real pixel
address works: gradient picks the reference in 3 of 4 probed addresses at R=4
and 4 of 4 at R=2, full optimisation 4/4 in both, and **per-address centring
changes nothing** (identical picks, "argmax decided by covariance" in 16/16
cases). M1 is real and decisive on synthetic byte-offset data; it is **not** the
ladder's disease.

---

## 3. The stated explanation, tested

> "a mixture of candidate operators is a blend of functions at a valid input,
> while a mixture of candidate addresses is a blend of unrelated values and
> denotes nothing"

read as a prediction about data, says: **E1** the failure worsens as the
addressed values become less correlated; **E2** it largely vanishes when
neighbouring addresses hold similar values.

### 3.1 The decisive control: the same array, permuted

A smooth array (neighbour correlation +0.945) against **the same array with its
positions randomly permuted** (+0.366). Identical value multiset, identical
per-address marginals, identical spectrum of `C`; only the arrangement differs.
12 data seeds, explicit logit noise.

| arrangement | neighbour corr | discrete: gradient → reference | discrete: optimised → reference | index: descent reaches reference |
|---|---|---|---|---|
| smooth | +0.945 | 11/12 | 12/12 | **0.806** |
| permuted | +0.366 | **11/12** | **12/12** | **0.117** |

**E1 and E2 are false for the discrete mechanism and true for the `index`
operator.** Scrambling the arrangement has literally no effect on the mechanism
the perception ladder measured, and a 7x effect on the one it did not use.

### 3.2 The two-factor sweep

36 cells, 12 data seeds each, 16 addresses, chance 0.0625.

**Discrete route — the first gradient step picks the reference** (rows =
correlation length, columns = per-address mean spread):

| corr \ spread | 0.0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| **0.0** | 1.000 | 1.000 | 1.000 | 0.833 | 0.417 | 0.250 |
| **0.5** | 1.000 | 1.000 | 1.000 | 0.833 | 0.417 | 0.250 |
| **1.0** | 1.000 | 1.000 | 1.000 | 0.833 | 0.333 | 0.250 |
| **2.0** | 1.000 | 1.000 | 0.917 | 0.667 | 0.167 | 0.167 |
| **4.0** | 0.417 | 0.917 | 0.833 | 0.333 | 0.167 | 0.167 |
| **8.0** | **0.000** | 0.417 | 0.333 | 0.250 | 0.167 | 0.167 |

Read down a column: correlation makes the pick **worse**, the opposite of E2, and
at correlation 8 with equal means the pick is *never* the reference. (Honest
caveat, in the direction of the explanation: at correlation 8 over 16 positions
the addresses are nearly interchangeable, so a miss costs almost nothing in
loss. E2 is right about the loss and wrong about the pick.) Read across a row:
mean spread is what destroys it, monotonically, everywhere.

**Discrete route — full optimisation reaches the reference** in **35 of 36
cells at 1.000**, the exception being corr 8 / spread 4 at 0.833. A single
free address is not the wall.

**`index` route — Adam from 8 starts reaches the reference:**

| corr \ spread | 0.0 | 0.25 | 0.5 | 1.0 | 2.0 | 4.0 |
|---|---|---|---|---|---|---|
| **0.0** | 0.365 | 0.333 | 0.323 | 0.312 | 0.260 | 0.167 |
| **1.0** | 0.521 | 0.521 | 0.438 | 0.333 | 0.208 | 0.167 |
| **4.0** | 0.927 | 0.781 | 0.438 | 0.250 | 0.167 | 0.125 |
| **8.0** | **1.000** | 0.500 | 0.240 | 0.167 | 0.156 | 0.125 |

Here correlation helps monotonically, from 0.365 to 1.000 — **E1 and E2 hold,
for the `index` operator only**, and mean spread hurts on this route too.

### 3.3 What the explanation should say instead

The blend is not meaningless. It is a **legal value of the addressed type that
no candidate produces**, and that is the whole problem: the optimiser can reduce
the loss by moving the blend's *value* instead of by concentrating on an
address, and it will, because that is the larger gradient. Three consequences
follow, and each is one of M1–M3:

* under a **linear** downstream, moving the blend's value means matching means
  (M1);
* under a **saturating** downstream, moving the blend's value means walking
  toward the constant (M2) — or nowhere at all when the surrogate has underflowed;
* under the `index` **kernel**, the blend is a local average, so the search only
  ever sees two addresses at a time (M3).

---

## 4. Remedies, and what each one buys

Five synthetic benchmarks (16 addresses, 64 examples, 12 data seeds, explicit
logit noise) and the real 144-program `geometry` benchmark. Successes and mean
node evaluations per run. `logit_noise_0.5` is the baseline for discrete arms.

### 4.1 Synthetic (`remedies.py`, 470,880 node evals, 225 s)

| remedy | T1 discrete direct, byte-scale | T2 discrete `eq`, mean spread | T3 `index` direct, byte-scale | T4 `index` direct, iid | T5 two coupled addresses (256) |
|---|---|---|---|---|---|
| baseline / `logit_noise_0.5` | 6/12 · 600 | 2/12 · 1200 | 0/12 · 600 | 0/12 · 600 | 2/12 · 3000 |
| `logit_noise_0` (zero init) | 7/12 | 2/12 | — | — | 1/12 |
| `logit_noise_2` | 7/12 | 2/12 | — | — | 1/12 |
| choice-softmax anneal 8 → 0.25 | **12/12** | 2/12 | — | — | 0/12 |
| **`index` temperature anneal 64 → 0.5** | — | — | 2/12 | **12/12** | — |
| `index` fixed wide temperature | — | — | 0/12 | 1/12 | — |
| `index` init at centre | — | — | 1/12 | 4/12 | — |
| `index` init at random | — | — | 2/12 | 6/12 | — |
| straight-through hard address | 7/12 | 3/12 | 0/12 | 0/12 | 0/12 |
| sampled address per step | 7/12 | **8/12** | 0/12 | 1/12 | 0/12 |
| **perturbation scoring** | **12/12 · 16** | **12/12 · 16** | **12/12 · 16** | **12/12 · 16** | 5/12 · 32 |
| per-address centred values (diagnostic) | **12/12** | n/a | 0/12 | 0/12 | n/a |
| *enumeration reference* | *16 programs, unique, 0.00 s* | *16, unique* | *n/a (continuous)* | *n/a* | *256, unique, 0.08 s* |

`centred_values` is only definable where the supervision is the addressed value
itself; with an `eq` downstream the target is computed against a raw value, so
removing a per-address mean would require already knowing the address. Marked
n/a rather than silently measuring a different task.

### 4.2 Real, 144 programs, raw `geometry` bytes, colour constants pinned (`real_remedies.py`)

| arm | successes | node evals |
|---|---|---|
| baseline (`eq` temperature 1) | **0/12** | 4,800 |
| `eq` temperature 100 | 4/12 | 4,800 |
| choice-softmax anneal | 0/12 | 4,800 |
| choice-softmax anneal + temperature 100 | 4/12 | 4,800 |
| straight-through hard address | 0/12 | 4,800 |
| straight-through + temperature 100 | 4/12 | 4,800 |
| sampled address per step | 0/12 | 4,800 |
| sampled + temperature 100 | 0/12 | 4,800 |
| **`eq` temperature 256 (= `2^bits` for `int[8]`)** | **8/12** | 4,800 |
| **perturbation, coordinate descent to a fixpoint** | **12/12** | **294–402** |
| *enumeration* | *solved, **unique**, 0.007 s* | *20,736* |

### 4.3 The recorded arm, `rung3:centre_free_address` (`ladder_fix.py`)

Colours free as well as addresses — so this also exercises the one-temperature
defect, since scaling `eq`'s surrogate flattens the same node's colour choice.
"Compensated" multiplies only those nodes' learning rate by the temperature,
which is what separating the two temperatures (diff **D2**) would buy. Success
is **held-out exact on 24 fresh episodes**, which is the criterion that matters;
address identity is reported separately because several conforming programs
exist.

| arm | R=2, **576 programs** | R=4, **9,216 programs** |
|---|---|---|
| shipped (`eq` temperature 1) | **0/12** | **0/12** |
| node temperature 256, uncompensated | 0/12 | 0/12 |
| node temperature 256, compensated (emulating D1+D2) | 6/12 | 12/12 |
| **the D1 diff applied literally** (`d1_patch.py`) | **7/12** | **12/12** |
| *— of which both addresses on the centre pixel* | ***12/12*** | ***12/12*** |

`FINDINGS` section 11 records this arm at 1/12 (576) and 0/12 (46,656). The
"compensated" row raises the node temperature, which under the current
`SoftProgram` also flattens that node's candidate softmax, and then multiplies
those nodes' learning rate back up; it is an emulation. The last row applies the
proposed diff itself, by rebinding `tcn.learning.relaxed` inside the research
script for the duration of the run — nothing under `tcn/` is edited, and the
rebound function is asserted bit-identical to the shipped one on `add`, `mul`,
`lt`, `not` and `eq` over a float carrier. **D1 alone is enough; D2 is not
required for this result.**

Both addresses land on the centre pixel's red or green byte in 24 of 24 runs.
The five R=2 runs that find the right addresses without held-out exactness pick
the wrong colour constant. The R=4 successes select byte 25 — the centre pixel's
green channel — and test it against 30, a single-channel program exact on all 24
held-out episodes: the address is recovered and the program is one of the
conforming non-unique solutions (enumeration at R=4 reports `unique=False`).

### 4.4 What each remedy is actually worth

* **Scaling the downstream surrogate to the representation: the fix.** 0/12 →
  12/12 at 9,216 programs on the recorded arm. It is not a hyperparameter hunt:
  256 is `2^bits` for `int[8]`, and the bracket 64/128/256/512 reads
  3/12, 5/12, 8/12, 8/12 on the 144-program benchmark — a plateau starting at the
  representation's own width, not a tuned peak. It only applies where a saturating surrogate sits
  downstream of the address, which is M2.
* **`index` temperature annealing: the fix for M3 on well-behaved data.** 0/12 →
  12/12 on uncorrelated values. It buys nothing against M1 (2/12 at byte scale),
  and a *fixed* wide temperature buys nothing at all (1/12) — the anneal, not the
  width, is what works, which is what a multi-modal landscape predicts.
* **Perturbation scoring: 12/12 everywhere a single node decides**, at 16 node
  evaluations. But this is not a repair of soft addressing — it is one node's
  enumeration, and it degrades to 5/12 the moment two addresses are coupled
  through a single Boolean (T5). On the real coupled task, coordinate descent to
  a fixpoint recovers 12/12 at 294–402 node evaluations against enumeration's
  20,736, so it is a **70x-cheaper heuristic without a uniqueness certificate**.
* **Choice-softmax annealing: 12/12 on M1, nothing anywhere else.** It rescales
  a convex problem; it cannot repair a dead or misleading gradient.
* **Straight-through hard address: null, everywhere.** 0/12 on both `index`
  benchmarks, 7/12 vs 6/12 on T1, 0/12 on the real task. Worth recording as a
  documented boundary: the forward value is not what is broken.
* **Sampling an address per step: helps once (T2, 2/12 → 8/12), null or harmful
  elsewhere** (0/12 on the real task at both temperatures, 1/12 on T4).
* **Initialisation is nearly irrelevant on the discrete route** — logit noise 0,
  0.5 and 2 give 7, 6, 7 on T1 and 2, 2, 2 on T2. On the `index` route the start
  point matters somewhat (0/12 → 4/12 at centre, 6/12 at random) and is
  dominated by annealing.
* **Per-address centring: 12/12 on T1, and irrelevant on the real data**, whose
  mean spread is only 0.24 sd. It is a clean confirmation of M1's mechanism and
  not a usable remedy, since it is preprocessing the substrate is not allowed
  to hide.

---

## 5. Answering the three-way question the track posed

For each mechanism, "worse than chance" resolves to a different thing:

* **M1 — signal, pointing the wrong way.** Not flat (the gradient is large), not
  displaced (the landscape is convex with the reference as its unique optimum).
  The first step is decided by a term that ranks means. *Fix: none needed at one
  node; the optimiser recovers. Anneal the choice softmax if it does not.*
* **M2 — no signal at all, then signal pointing the wrong way.** At the shipped
  `eq` temperature, 16/16 address gradients are exactly zero. Widen the
  surrogate and the gradient appears, initially ranking addresses by value
  proximity (0.125 at temperature 10) and reaching 0.438 against chance 0.083 at
  temperature 1,000. *Fix: scale the surrogate to the declared representation.*
* **M3 — multi-modal.** 5–25 local minima on an exhaustively mapped
  one-dimensional space, with a basin 1.3–2.5 addresses wide, because the kernel
  spans two addresses. *Fix: anneal the kernel from wide to narrow; the fixed
  wide kernel does not work, and the shipped narrow one is not even exact at an
  integer address.*

---

## 6. The construction rule

Soft addressing is fixable, so the rule below is **not** "never relax an
address". It is a rule about *when* relaxing one is worth the trouble, and it is
stated so it can be followed without rerunning any of this.

> **Rule A — compute an address, do not choose one, whenever the address is a
> function of something already in the program.**
>
> If the address can be derived — from a constant set of positions paired with
> the observation, or read out of a relation the observation already carries —
> derive it. The scaffold then has no address choice at all, and its size stops
> growing with the number of addressable positions.
>
> *Precedents, all in this repository.* The positional-reuse pattern pairs a
> constant `set[Index]` with a singleton set holding the whole observation and
> `map`s one module over the product, so the module reads its own position out
> of the record with `project` and indexes with `index`: **3 caller nodes at any
> width**, and the discrete-perception track measured the space staying at
> **32,000 programs from R=8 to R=48** against the ladder's `(3R^2)^2`. The
> depth-generalization track reads the wire index out of the `gates` relation
> instead of searching a binding: **272 programs and 4.00 at every depth,
> against 1,088 programs and 3.66–3.77 with the binding free**.
>
> *What it costs.* The position set is a declared structural constant and cannot
> itself be learned. `insert`, `pair`, `map` and `filter` all declare
> `gradient="none"`, so nothing upstream of the map is trainable through it, and
> `pair` replicates the observation once per position. Positional reuse buys a
> scaffold that does not grow; it does not buy a differentiable path.

> **Rule B — if an address must be relaxed, make the downstream operator's
> relaxation valid at the mixture before blaming the address.**
>
> A mixture of addresses is a legal value of the addressed type that no
> candidate produces. Before concluding the address is unlearnable, check that
> every operator the address feeds still has a live, correctly-scaled gradient
> *at that mixture*. Concretely, for each such operator:
>
> 1. evaluate its relaxation at the uniform mixture and at the reference vertex,
>    and compare — 3.2e-31 against 0.83 is the diagnosis, not the symptom;
> 2. set any surrogate temperature from the **declared representation**, not
>    from 1.0: `2^bits` for an `int[n]` carrier, the bound width for a bounded
>    numeric one;
> 3. check the relaxation reduces to `Registry.exact` at a vertex. `index` at
>    temperature 1.0 does not (0.332 mean error), and any relaxation that does
>    not agree with its operator on its own domain is measuring a different
>    program.
>
> Measured payoff of doing exactly this on the recorded failing arm: **0/12 →
> 12/12 held-out exact at 9,216 programs**.

> **Rule C — a relaxed address is worth it only above the enumeration
> crossover, and it never returns a certificate.**
>
> On the 144-program real benchmark, enumeration is exhaustive in **0.007 s**
> and certifies the solution **unique**; the best gradient arm is 8/12 at 4,800
> node evaluations, and perturbation coordinate descent is 12/12 at ~350 with no
> certificate. Relaxing an address earns its place only where the address space
> is out of enumeration's reach *and* Rule B has been satisfied. Below that,
> enumerate; that is section 8 of `FINDINGS`, unchanged.

> **Rule D — do not relax an address across positions whose values are
> unrelated, when the mechanism is the `index` operator.**
>
> This is the one place the original explanation is correct, and it is
> quantified: on the identical values, arrangement alone moves `index` from
> 0.806 to 0.117. If the tuple is a raster, a time series or anything else whose
> neighbouring entries are similar, `index` with an annealed kernel is
> reasonable (12/12 on uncorrelated data with annealing, and correlated data is
> easier still). If the tuple is a record of unrelated fields, use `project`
> candidates and let discrete search have it.

---

## 7. Proposed changes to `tcn/`, as diffs. **None applied.**

Ordered by measured payoff. Nothing under `tcn/` or `generators/` was touched.

### D1 (highest payoff). `eq`'s surrogate temperature ignores the representation it compares

`tcn/learning.py:38`. `exp(-(a-b)^2/temperature)` at `temperature=1` is exactly
`0.0` in float32 at a gap of 11, so on `int[8]` data it is dead for 96% of value
pairs. The temperature should be scaled by the declared carrier width — that is
a representation fact, available from `op.inputs[0]`, not a domain fact.

```diff
     if n in COMPARE:
-        if n=="eq": return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/temperature)
+        if n=="eq":
+            # `temperature` is dimensionless; the squared gap is not.  Scale by
+            # the declared representation so the surrogate is alive across the
+            # carrier's own range: an int[8] byte pair 25 apart is as "close" as
+            # a unit-scale float pair 0.1 apart, and at temperature 1 the former
+            # underflows to exactly 0.0 while the latter does not.
+            t=op.inputs[0]
+            span=float(2**t.bits) if t.kind=="int" and t.encoding.kind=="integer" else 1.
+            return torch.exp(-((a-b)**2).sum(-1,keepdim=True)/(temperature*span))
         d=b-a if n in {"lt","le"} else a-b
         return torch.sigmoid(d/temperature)
```

**Measured with this exact diff applied** (`d1_patch.py` rebinds
`tcn.learning.relaxed` for the run; the rebound function is asserted identical
to the shipped one wherever D1 does not apply), on `rung3:centre_free_address`
with 24 training and 24 held-out episodes and explicit logit noise:

| | R=2, 576 programs | R=4, 9,216 programs |
|---|---|---|
| shipped | 0/12 exact, 0/12 addresses | 0/12 exact, 0/12 addresses |
| **with D1 alone** | **7/12 exact, 12/12 addresses** | **12/12 exact, 12/12 addresses** |

D2 is *not* needed for this. On the 144-program pinned-colour benchmark the same
temperature gives 0/12 → 8/12, with the address gradient going from 16/16 dead
to picking the reference 0.438 against chance 0.083. Note the compare family's *other* members take `d/temperature` through a
sigmoid, which has the same units problem in linear rather than squared form;
that half is not changed here because nothing in this track measures it.

**Risk, stated:** this changes an operator contract's behaviour, so every
conformance case that pins `eq`'s relaxed value changes with it, and every
recorded relaxed loss on integer data becomes incomparable to the old one. It
should land with a spec note in `ARCHITECTURE.md` section 2 and a regression
test asserting the surrogate is nonzero across the full carrier range.

### D2. `SoftProgram` uses one temperature per node for two unrelated jobs

`tcn/learning.py:81,99–102`. The same scalar divides the candidate softmax and
the operator's own relaxation, so `eq`'s surrogate cannot be widened at a node
that also has a real choice without flattening that choice. Measured directly:
at temperature 10,000 the surrogate at gap 5 goes from 0.0 to 0.998 while the
candidate distribution collapses from `[0.881, 0.119]` to `[0.50005, 0.49995]`.
This is why D1's payoff at R=2 needed a compensating learning rate (0/12
uncompensated, 6/12 compensated).

```diff
-        self.temperatures={n.name:1. for n in program.nodes}
+        self.temperatures={n.name:1. for n in program.nodes}
+        # The candidate softmax and the operator's relaxation are unrelated
+        # quantities: one is a search annealing schedule over a discrete choice,
+        # the other is part of an operator's declared contract and carries the
+        # units of its inputs.  Sharing one scalar means a scaffold cannot fix a
+        # saturated surrogate without also destroying its own choice signal.
+        self.relaxations={n.name:1. for n in program.nodes}
...
             tau=self.temperatures[n.name]
-            ys=[relaxed(self.registry,c.operator,[values[s] for s in c.sources],tau) for c in n.candidates]
+            rho=self.relaxations[n.name]
+            ys=[relaxed(self.registry,c.operator,[values[s] for s in c.sources],rho) for c in n.candidates]
```

`freeze`, `export`, `entropy` and `complexity` are untouched; `relaxations`
defaults to the current behaviour, so this is behaviour-preserving until a
caller sets it.

### D3. `relaxed("index")` does not reduce to `exact` at an integer address

`tcn/learning.py:53–56`. At temperature 1.0 over 16 positions the kernel puts
0.564 of its mass on the addressed element — measured mean absolute error 0.332
against `Registry.exact` on unit-variance data, at an exactly integer address.
`ARCHITECTURE.md` section 1 requires each lift to declare a *valid* relaxation;
a relaxation that disagrees with its operator everywhere on its own domain does
not meet that. A straight-through form makes the forward exact while keeping the
soft gradient:

```diff
     if n=="index":
         width=op.output.width; count=len(op.inputs[0].items)
         weights=torch.softmax(-(b-torch.arange(count,device=a.device))**2/temperature,dim=-1)
-        return (a.reshape(*a.shape[:-1],count,width)*weights.unsqueeze(-1)).sum(-2)
+        cells=a.reshape(*a.shape[:-1],count,width)
+        soft=(cells*weights.unsqueeze(-1)).sum(-2)
+        # At the shipped temperature the kernel keeps only 0.564 of its mass on
+        # the addressed element, so the relaxed read disagrees with `exact` even
+        # at an integer address.  Take the exact read in the forward pass and the
+        # kernel's gradient in the backward one.
+        hard=(cells*torch.nn.functional.one_hot(weights.argmax(-1),count)
+              .unsqueeze(-1).to(cells.dtype)).sum(-2)
+        return ste(soft,hard)
```

**Measured honestly: this does not improve search.** Straight-through addressing
is 0/12 on both `index` benchmarks and 0/12 on the real task. It is a
correctness fix — it removes a relaxed/exact disagreement that makes every
relaxed loss on an `index` node incomparable with the exported program's — and
it should be landed as such, not as a search improvement. **What does improve
search is annealing the temperature from wide to narrow (0/12 → 12/12 on
uncorrelated data), which needs no core change at all**, only a caller that sets
`model.temperatures[node]` on a schedule — and after D2, `model.relaxations`.

### D4 (optional). A perturbation coordinate-descent selector in `tcn/search.py`

Scoring each candidate of one node by hard substitution and descending to a
fixpoint recovered the reference in 12/12 at **294–402 node evaluations** on the
real 144-program benchmark, against enumeration's 20,736 — 70x cheaper. It is a
heuristic, not a certificate: on the coupled synthetic benchmark T5 it is 5/12
where enumeration is exact over 256 programs in 0.08 s. If added, it must report
that it forfeits the uniqueness certificate, exactly as `stop_at_first` already
does. No diff is proposed here because the measured envelope is one benchmark
wide; the numbers are in `out/real_remedies.json` for whoever wants to widen it.

---

## 8. Limits of this track, stated

* **One free address at a time is not the ladder's problem.** Every synthetic
  discrete benchmark here has one or two address nodes over 16 positions.
  Section 3.2's 35-of-36 cells at 1.000 say a single relaxed address is easy;
  the ladder's difficulty is the conjunction plus M2, and the conjunction alone
  (T5) is 2/12 with a working gradient. Nothing here measures more than two
  coupled addresses.
* **The D1 payoff is measured on one generator.** 0/12 → 12/12 is `geometry`
  pixels at R=4 with 24 training episodes, and 0/12 → 7/12 at R=2. It has not been run on `raster_text`
  rung 4, where `FINDINGS` records the solution living behind `pack`
  (`gradient="none"`) — D1 cannot help there, because there is no gradient to
  rescale.
* **Held-out sets here are 24 episodes**, against the ladder's 48. Every
  reported exact success is max error 0.0 over 24 fresh episodes, not a
  generalisation claim at the ladder's resolution.
* **The `index` sweep uses 8 starts per cell at stride 2**, not all 16, and 300
  Adam steps; the permutation-pair and landscape numbers use all 15 starts and
  400–600 steps. Cells are comparable within a table, not across.
* **`tcn/search.py` still cannot score a recurrent program**, so none of the
  enumeration references here cover the depth-generalization scaffold, and Rule
  A's depth-scaffold precedent is quoted from that track rather than re-measured.
* **No claim is made about `lt/le/gt/ge`.** They carry the same units problem as
  `eq` in linear form and were not measured.

---

## 9. Files

| file | what it does |
|---|---|
| `instrument.py` | the controlled instrument: data regimes, both address mechanisms, budget accounting |
| `landscape.py` | task 1 — vertices, Dirichlet interiors, an exhaustive 4-address simplex grid, an exhaustive 1-D `index` map |
| `explanation.py` | task 2 — the permutation control and the 36-cell two-factor sweep |
| `remedies.py` | task 3 — five synthetic benchmarks x every remedy, with enumeration references |
| `real_data.py` | the two data factors measured on the actual `geometry` bytes |
| `ladder_autopsy.py` | `pointing.py` split by node role, with dead gradients reported as data |
| `eq_temperature.py` | the 144-program pinned-colour benchmark and the surrogate-temperature sweep |
| `real_remedies.py` | every remedy on that benchmark, against its enumeration reference |
| `ladder_fix.py` | D1+D2 on the recorded `rung3:centre_free_address` arm at R=2 and R=4 |
| `span_sweep.py` | brackets D1's temperature at 64/128/256/512 on that benchmark |
| `d1_patch.py` | the proposed D1 diff applied literally, with a faithfulness assertion |
| `checks.py` | the three measurement warnings, plus defects D2 and D3 |
| `out/*.json` | every number above, as written by the scripts |

Reproduce with `PYTHONPATH=.:research/address-wall .venv/bin/python research/address-wall/<script>.py`.
