# Three substrate defects, fixed — 2026-09-09

**Verdict: all three are fixed, both shipped fixtures reproduce bit-identically,
and the cost is one measured regression that is named rather than hidden.**

Three defects in `tcn/learning.py` were blocking gradient learning across four
tracks. All three are now fixed with regression tests that fail before and pass
after, and every shipped fixture reproduces to every printed digit.

| defect | fixed | cost |
|---|---|---|
| **D1** a declared / one-candidate node severed the upstream gradient | yes, unconditionally | none measured |
| **D2** one temperature drove both the choice softmax and the operator relaxation | yes, adopting the surrogate branch's interface verbatim | none; the carrier width it enables has a cost, below |
| **D3** `relaxed("tuple")` did not broadcast | yes, unconditionally | none measured |

The derived surrogate scale is applied to **`eq` only**, opt-in through
`SoftProgram.scale_surrogates()`. That restriction is a measurement, not an
omission, and section 4 gives the numbers: on the ordering comparisons the
carrier width restores a gradient while **moving the loss minimum onto a wrong
answer**, which is worse than the dead surrogate it replaces.

Everything below was measured on this worktree with a fresh `uv sync --locked
--extra test` venv, single-threaded, on a host at load average 31–78. Node
evaluations are reported beside wall clock because the host is shared.

---

## 1. What changed in `tcn/`

`tcn/learning.py`, `tcn/crystallize.py` (two lines), `tcn/training.py` (two
lines). Nothing else.

### D1 — three fixed-choice states, two of which are boundaries

`SoftProgram.__init__` built `self.frozen` from `Node.selected` and the forward
pass evaluated every frozen node as `exact_tensor(...).detach()`. So declaring a
node's operator — the natural way to write plumbing whose choice is already
decided, and the mode AGENTS.md explicitly endorses — cut the gradient to
everything upstream of it.

The fix separates the three ways a node's choice can be settled. `self.frozen`
keeps its meaning for every existing caller ("this node's choice is settled");
a new `self.pinned` records the subset that is **not** a gradient boundary.

1. **Declared** (`Node.selected` is set). The choice is a hand-supplied prior or
   fixed plumbing. Its choice logit does not train, but its value path is
   evaluated through the operator's declared relaxation, so it behaves *exactly*
   as the same node written with that one candidate and no selection. Tested as
   an equality, not an approximation.
2. **Declared with no relaxation** (`gradient="none"`: every module call, the set
   operations). `relaxed()` routes those to exact execution, so the boundary is a
   property of the **operator contract**, which is where ARCHITECTURE section 4
   puts it. A frozen module still stops gradients, unchanged.
3. **Crystallized** (`SoftProgram.freeze`, called by the scheduler). *This stays a
   gradient boundary, and that is the third case's decided semantics.* Freezing is
   an irreversible commitment to the exported discrete program, so the node
   executes exactly and detached, per ARCHITECTURE section 5's "freezing removes
   that internal gradient machinery". Weakening it would weaken the connectivity
   guard, whose whole job is to check that a commitment has not disconnected the
   interior. `freeze()` therefore drops the node from `pinned`, and
   `Crystallizer.try_freeze` snapshots and restores `pinned` on rollback.

Both routes into `self.frozen` are covered: a one-candidate node and a
multi-candidate node pinned to index *k* both stay differentiable, and the pinned
index is honoured by `selections()`, `distributions()`, `entropy()` and
`export()`.

### D2 — choice temperature and surrogate temperature, separated

Adopted **verbatim from `worktree-agent-afc0484c7c0dc4724`** rather than written
a second time, so the two branches merge instead of competing:

- `SoftProgram.temperatures[n]` — the candidate softmax temperature, unchanged.
- `SoftProgram.surrogate_scale[n]` — a multiplier; the relaxation sees
  `temperatures[n] * surrogate_scale[n]`. Defaults to 1., so every shipped run is
  bit-identical and the crystallizer's single 0.8×-per-round anneal still sharpens
  both, because it scales the shared term.
- `relaxed(registry, op, xs, temperature=1., carrier_scaled=False)` — the
  carrier-derived `eq` width, opt-in.
- `carrier_temperature(t)` — `2**bits` for an integer-encoded `int[n]`, 1. for
  `bool`, 1. for a floating encoding; a composite takes its widest leaf, since
  `eq` sums the squared difference over the whole flattened width.
- `SoftProgram.scale_surrogates()` — turns the width on and returns the nodes it
  reaches.

My only additions to that interface are the `pinned` interaction in `forward`
(D1) and checkpoint persistence of `surrogate_scale` / `carrier_scaled` /
`pinned` in `tcn/training.py`, which that branch does not touch.

### D3 — `tuple` now broadcasts

`torch.cat(xs, dim=-1)` does not broadcast, so packing a batched intermediate
alongside an unbatched trainable constant raised where every arithmetic operator
on the identical pair succeeds. It now broadcasts the leading batch axes and
concatenates only the declared widths — the same rule `exact_tensor` already
applies on the exact path.

---

## 2. Every shipped fixture reproduces

Before = `main` at `b9088a9`. After = this branch. Both run in this worktree's
own venv.

### `tcn train --episodes 160`

| | before | after |
|---|---|---|
| initial prediction loss | 0.248835613951087 | **0.248835613951087** |
| final prediction loss | 0.0022308224288281053 | **0.0022308224288281053** |
| deterministic evaluation return | 4.0 / 4 | **4.0 / 4** |
| frozen evaluation return | 4.0 / 4 | **4.0 / 4** |
| fully frozen | True | **True** |
| wall clock | 38.3 s | 25.1 s (host load, not a speedup claim) |

Rounded as the brief states them: **0.24884 → 0.00223, 4/4 deterministic, 4/4
frozen**, in both arms.

### `tcn synthesize` (the mixed fixture)

| | before | after |
|---|---|---|
| exact conformance | True | **True** |
| fully frozen | True | **True** |
| exact max error | 0.0 | **0.0** |
| relaxed loss | 1.0066399909192114e-06 | **1.0066399909192114e-06** |
| enumeration | 96 / 96 exhausted, unique | **96 / 96 exhausted, unique** |
| enumeration selection | `logic=6` (xor) | **`logic=6`** |
| agrees with enumeration | True | **True** |
| accepted freezes, in order | algebra, logic, analytic, conversion | **identical** |

The relaxed loss agreeing to 17 significant figures is the strongest statement
available that the forward and backward paths are unchanged where nothing was
meant to change: neither fixture contains an `eq`, an ordering comparison, an
`index`, or a `selected` node.

### Test suites

| | before | after |
|---|---|---|
| `pytest -q` | 179 passed (185 s) | **194 passed** (96 s) |
| `npm --prefix generators/computer/engine test` | — | 372 passed, 10 files |
| `npm ... run typecheck` | — | clean |

The 15 new tests are `tests/test_gradient_boundaries.py`. Each fails on `main`.

---

## 3. D1 and D3 re-measured

Both "before" columns below were **produced, not quoted**: `main`'s
`tcn/learning.py` and `tcn/crystallize.py` were swapped into this worktree and
the same reproduction script run against them, then the fixed files restored and
the script re-run. `tests/test_gradient_boundaries.py` cannot even be collected
against `main` (no `carrier_temperature`), which is a weak form of "fails
before", so this stronger check was run instead.

### D1 — the exact reproduction from FINDINGS section 18

A trainable constant feeding `mul` through an identity node. Gradient at the
constant:

| node | before | after |
|---|---|---|
| single candidate, unselected | `tensor([1.])` | `tensor([1.])` |
| single candidate, `selected=0` | **`None`** | **`tensor([1.])`** |
| two candidates, `selected=0` | `None` | `tensor([1.])` |
| committed by `freeze()` | `None` | `None` — *by design, case 3* |

Forward value is `3.0` in every row, so nothing about the exact semantics moved.
A module call (`gradient="none"`) has no gradient path to its input at either
selection state, before and after: that boundary belongs to the operator, and it
is untouched.

### D3 — the exact reproduction

Packing an `(8,1)` batched value with a `(1,)` constant:

| | result |
|---|---|
| `add` on the identical pair | `(8, 1)` — broadcasts, always did |
| `tuple`, before | **`RuntimeError: Tensors must have same number of dimensions: got 2 and 1`** |
| `tuple`, after | `(8, 2)`, constant gradient `[8.0]` |

---

## 4. D2 re-measured — the surrogates, and where the minimum sits

### The coupling itself

One node, candidates `eq | lt | le` over `int[8]`, logits `[1,0,0]`:

| | choice entropy | max choice gradient |
|---|---|---|
| choice 1, surrogate 1 (shipped) | 0.97533 | 0.16702 |
| choice 1, surrogate 8 (**now possible**) | **0.97533** | **0.16650** |
| choice 8, surrogate 8 (the old coupling) | 1.09683 | **0.02706** |

Widening the surrogate alone leaves the choice distribution bit-identical and the
choice gradient within 0.3%. Widening it the only way that used to be available
flattens the distribution and costs 6.2× of the choice gradient — the mechanism
behind the language track's recorded 9e-2 → 2e-5 collapse.

### `eq` — the surrogate is alive again

`eq` on a byte carrier, `carrier_temperature = 256`:

| \|a−b\| | value before | grad before | value after | grad after |
|---|---|---|---|---|
| 1 | 3.68e-01 | 7.36e-01 | 9.96e-01 | 7.78e-03 |
| 10 | 3.78e-44 | 7.57e-43 | 6.77e-01 | 5.29e-02 |
| **11** | **0.0** | **0.0** | 6.23e-01 | 5.36e-02 |
| **25.6** | **0.0** | **0.0** | **7.73e-02** | 1.55e-02 |
| 48 | 0.0 | 0.0 | 1.23e-04 | 4.63e-05 |
| 128 | 0.0 | 0.0 | 1.60e-28 | 1.60e-28 |
| 255 | 0.0 | 0.0 | **0.0** | **0.0** |

7.73e-02 at 25.6 reproduces FINDINGS section 16's 7.7e-02 exactly. Note the last
row: the carrier width does **not** cover the whole carrier. It buys roughly
`|a−b| < 150` on a byte and nothing beyond.

### Liveness, read per candidate per example

Pooling over a batch calls a node differentiable whenever *any* candidate is live
on *any* example, which is exactly how a node whose reference candidate can never
be selected passes for healthy. Read per candidate per example instead — 256
`eq(x, c)` candidates against 12 byte examples:

| | before | after |
|---|---|---|
| candidates live on **no** example | **92 of 256** | **0 of 256** |
| candidates live on **every** example | 0 of 256 | 93 of 256 |
| mean examples live per candidate | **0.94 / 12** | **10.20 / 12** |
| pooled: "node is differentiable" | **True** | True |

The pooled reading is True in both columns. That is the fault: before the fix,
36% of the alphabet could not be selected at all and the aggregate said the node
was fine.

### `rung3:centre_free_address` — the failing benchmark now passes

The arm FINDINGS section 16 records at 0/12. Same seeds, same explicit
initialization noise (0.5; `SoftProgram` zero-initializes logits, so seeds vary
nothing on their own), the only difference being `scale_surrogates()`.

| | before | after |
|---|---|---|
| R=2, 576 programs | 0/12 held-out exact, 0/12 addresses | **7/12 exact, 12/12 addresses** |
| R=4, 9,216 programs | 0/12 held-out exact, 0/12 addresses | **12/12 exact, 12/12 addresses** |

230,400 node evaluations, 63 s. This reproduces section 16's table exactly, on
the shipped code path rather than a monkeypatch.

### `eq`'s cost: the carrier width biases a constant-selection minimum

Choosing the byte constant `c` in `eq(x, c)` against a BCE target, true byte 40,
64 examples of which a quarter match:

| effective τ | argmin | correct | surrogate grad at Δ=25 |
|---|---|---|---|
| 1 (before) | **40** | **yes** | **0.0** |
| 12.8 (the anneal floor, 0.05×256) | **40** | **yes** | 2.4e-21 |
| 32 | 41 | no | 5.1e-09 |
| 64 | 41 | no | 4.5e-05 |
| **256 (the carrier)** | **42** | **no** | **1.7e-02** |
| 65536 (carrier squared) | 255 | no | 7.6e-04 |

**This is a real cost and it is the same pattern as the ordering comparisons: the
shipped temperature has a dead gradient and a correct optimum.** For `eq` it is
mild — the minimum drifts by two bytes, not across the alphabet — and it is
bounded by two things. First, `exp(-‖a−b‖²/τ)` is strictly decreasing in
‖a−b‖ at every τ, so the *surrogate's own* ranking is the distance ranking at
every temperature (asserted directly across all 256 bytes in the tests); the
drift comes from the loss trading positives against negatives, not from the
kernel. Second, the crystallizer already anneals to τ = 12.8, where the minimum
is exact again. The usable reading is **wide to find, annealed to commit**, and
the end-to-end benchmark above is what settles that the trade is worth taking on
that task.

### `lt` / `le` / `gt` / `ge` — deliberately **not** scaled

Their gradient is just as dead:

| Δ | `lt`/`le` grad | `gt`/`ge` grad |
|---|---|---|
| 1 | 1.97e-01 | 1.97e-01 |
| 16 | 1.19e-07 | 1.13e-07 |
| **17** | **0.0** | 4.14e-08 |
| 48 | **0.0** | 1.43e-21 |
| 96 | 0.0 | **0.0** |

reproducing section 19's 1.97e-01 at Δ=1, 1.19e-07 at 16, exactly zero from 17,
against a task operating near Δ=48. **They are still not scaled.** Choosing the
threshold `c` in `le(x, c)`, true threshold 128, 96 byte operands of which 94.8%
sit past the dead point and only two straddle the boundary:

| effective τ | argmin | hard predicate correct | loss spread |
|---|---|---|---|
| **1 (shipped, kept)** | **128** | **yes** | 10.009 |
| 8 | 120 | no | 5.423 |
| 32 | 93 | no | 1.419 |
| **256 (the carrier)** | **0** | **no** | **0.214** |
| 65536 | 0 | no | 9.3e-04 |

The shipped temperature is numerically dead and puts its minimum on exactly the
right threshold — a plateau with cliffs, not a misleading slope. Every widening
moves the minimum, and the carrier width collapses the loss spread by 47×. A
live gradient pointed at a wrong answer is worse than a dead one, so no constant
is derived for this family. The temperature that does work is the task's
**decision margin**, and a margin is a property of the decision being learned
rather than of the declared type — estimating it from the operands is circular,
because it is measured from a boundary the search has not found yet. The
deliverable for this family is therefore the *separation*: a caller sets
`SoftProgram.surrogate_scale[node]` to a task margin, which the split now makes
possible without flattening that node's candidate softmax. That was impossible
before and is the whole point.

### `index` — also not scaled, and not fixed by any temperature

`index` over 12 positions of byte-scale values in an arbitrary arrangement,
descending the address from each of the 12 integer starts:

| effective τ | kernel mass on the addressed element | strict local minima | descent hits |
|---|---|---|---|
| **1 (shipped, kept)** | 0.564 | 4 | **2 / 12** |
| 12 (= positions) | 0.165 | 1 | **0 / 12** |
| 0.1 (annealed) | 0.99991 | 7 | 1 / 12 |

0.564 reproduces section 16's M3 figure. Widening removes the local minima and
makes descent *worse*, because the single remaining minimum sits where the local
average matches the target rather than on the addressed element. Sharpening
concentrates the kernel to 0.99991 and makes the landscape more jagged. **No
temperature makes relaxed addressing work here**, consistent with section 16's
"partly — 0/12 → 2/12 on byte-scale data". `index`'s temperature is a *mixture
weight*, so it changes the value the node returns and therefore certainly moves
the optimum; it is the clearest case for keeping the shipped scale. What the
split adds is that annealing it no longer concentrates the node's choice
distribution as a side effect, which is how the language track's `index0.1` arm
was confounded.

---

## 5. The language benchmarks: still failing

FINDINGS section 19 records gradient descent conforming in **0 of 44 runs** on
the spaces enumeration settles, and calls separating the temperatures "a
prerequisite for the whole family". The prerequisite is now met. **It was not
sufficient: both benchmarks still conform in 0 runs.** Same seeds, same explicit
initialization noise (0.01), the only difference being `scale_surrogates()`.

### Stage A — find `open(text,i) = eq(index(bytes, base+i), c)`

10,496 programs; enumeration certifies the unique answer `base=14, c=40`
(which is `(`). 8 seeds × 400 steps.

| arm | conformant | exact train accuracy | max `eq` choice gradient | bytes chosen across seeds |
|---|---|---|---|---|
| before | **0/8** | 0.528 | 9.01e-02 | 39, 41, 68, 88, 108, 110 |
| after (carrier `eq`) | **0/8** | 0.523 | 5.14e-03 | **36, 37, 38, 39** |
| after + annealed surrogate | **0/8** | 0.500 | 5.14e-03 | 32, 33, 42, 43, 70, 110, 114 |

Not fixed. The improvement is directional only, and it is the *same* bias the
constant-selection landscape in section 4 predicts: with the surrogate alive,
every seed lands within four bytes of the truth instead of scattering across a
quarter of the alphabet — but it lands *beside* it, not on it. Annealing the
surrogate during training (an arm that only exists because the temperatures are
now separate) scatters them again and is the worst of the three: **the
"wide to find, annealed to commit" reading does not carry to this task**, and I
report that against my own expectation.

### Stage B — the answer rule over the frozen stage-A module

45,375 programs, the family whose surrogate is `lt` (deliberately unscaled).
6 seeds × 300 steps, held-out on 724 episodes of unseen lengths.

| arm | exact-train conformant | mean held-out (unseen lengths) | per-seed train accuracy |
|---|---|---|---|
| before | **0/6** | 0.4383 | 0.292 ×4, 0.625, 0.000 |
| after | **0/6** | 0.5104 | **0.625 ×5**, 0.000 |
| majority constant | — | **0.5483** | — |

Not fixed, and **still below the majority baseline**. Train accuracy improves for
five of six seeds (0.292 → 0.625) and held-out rises 0.438 → 0.510, but 0.510 is
worse than answering "yes" every time. This arm is the direct consequence of the
decision in section 4: `lt`'s surrogate is dead at the Δ≈48 this task operates
at, and it is not scaled because scaling it would move the minimum. The gain
visible here comes from `eq` elsewhere in the same scaffold.

**So the headline stays 0 of 44.** Section 19 called the separation a
prerequisite; it was right that it is one, and this measurement says it is not
the remedy. Enumeration still settles both stages — 3.5 s and 363 s — and
remains the method that works on this task.

Cost of these two benchmarks: 350,400 node evaluations, 502 s single-threaded on
a host at load average 25–48.

---

## 6. What these fixes do **not** fix

- **Relaxed addressing is still harder than relaxed values.** Section 16's M3 is
  untouched: the `index` table above shows no temperature makes descent on a
  continuous address work on byte-scale data, and the *shipped* temperature is
  the best of the three. Compute addresses where you can. The address-wall win in
  section 4 is a discrete address chosen through a `gradient`-carrying `eq`, not
  a relaxed `index`.
- **`eq` is still dead past ‖a−b‖ ≈ 150 on a byte**, and exactly 0.0 at 255. The
  carrier width buys most of the carrier, not all of it.
- **The ordering comparisons are still dead past Δ=17.** Nothing here changes
  that; the fix is a margin the caller supplies, and this branch supplies the
  mechanism, not the margin.
- **The unreachable part of every scaffold is unchanged.** `truth_0` and
  `truth_15` are constant functions whose relaxation has identically zero
  derivative in both inputs, so no gradient can ever select them, while
  enumeration searches all 16:

  | | space | reachable by relaxation | fraction |
  |---|---|---|---|
  | the 16-table family | 16 | 14 | **0.875** |
  | `examples/mixed.py` | 96 | 84 | **0.875** |
  | `examples/joint.py` | 256 | 196 | **0.766** |

  Confirmed independently here, and **these fixes do not change it.** It has been
  true of every shipped scaffold since the first commit. It is not a bug in the
  relaxation — a constant function genuinely has no input gradient — but it is a
  standing gap between what gradient descent can reach and what enumeration
  searches, and any comparison of the two should quote it.
- **`carrier_temperature` has two type cases it gets wrong**, adopted as-is from
  the surrogate branch to avoid divergence, and flagged here for the merge:
  `role="category"` / `"symbol"` returns `2**bits` although such a carrier
  flattens to `bits` independent bits whose maximum squared separation is `bits`
  (so `int[8]` categories are scaled 32× too wide); and a **fixed-point**
  encoding returns 1. although its decoded span is `2**bits / scale`. Neither is
  exercised by any measurement in this document or on that branch.
- **The language track is not unblocked.** 0 of 44 before, 0 of 44 after
  (section 5). FINDINGS section 19 called the temperature separation "a
  prerequisite for the whole family"; it is one, and it is not the remedy.
- **Nothing here makes gradient descent competitive with enumeration.** STATUS.md's
  verdict stands: brute force settles the mixed fixture's 96 programs in 7 ms,
  and enumeration still settles both language stages where descent does not.

---

## 7. Costs, stated plainly

1. **The `eq` carrier width biases a constant-selection minimum by two bytes** at
   τ=256 (section 4). It is opt-in, it is annealed away by the crystallizer's own
   schedule, and it buys 0/24 → 24/24 on the address-wall benchmark. It is a
   trade, not a free win.
2. **Widening `index` or the ordering comparisons is worse than not widening
   them**, measured in both cases as a moved loss minimum, which is why neither is
   shipped. If a future change scales them by carrier width "for consistency",
   two tests in `tests/test_gradient_boundaries.py` will fail, deliberately.
3. **D1 changes forward values for scaffolds that use `selected` on an operator
   with a temperature-bearing relaxation** — the node now relaxes rather than
   executing exactly. That is the intended semantics (a declared node behaves as
   the same node written alone), and no shipped fixture sets `selected`, so
   nothing recorded moves. A scaffold that *relied* on `selected` to force exact
   execution mid-training should call `SoftProgram.freeze` instead, which is what
   that means.
4. **Annealing the surrogate during training was worse than holding it wide** on
   language stage A — 7 scattered bytes against 4 clustered ones (section 5).
   That arm exists only because of D2, and it is reported against expectation:
   the constant-selection landscape says the annealed temperature has the exact
   minimum, and the task disagrees.
5. No convergence slowdown was observed on either fixture: identical losses,
   identical freeze order, identical enumeration agreement.

---

## 8. Reproducing

```
uv sync --locked --extra test
.venv/bin/python -m pytest -q                                   # 194 passed
.venv/bin/python -m tcn.cli synthesize --out artifacts/mixed    # 96/96 unique, exact
.venv/bin/python -m tcn.cli train --episodes 160 --out artifacts/joint
.venv/bin/python research/core-gradient-fixes/measure.py        # -> measure.json
.venv/bin/python research/core-gradient-fixes/benchmarks.py     # -> benchmarks.json
.venv/bin/python research/core-gradient-fixes/language.py       # -> language.json
```

`generators/computer/engine/node_modules` is gitignored; the four
`generators/computer` tests need it symlinked from a checkout that has run
`npm ci`. It was symlinked for the run above and removed before committing.
