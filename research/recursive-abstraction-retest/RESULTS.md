# Track 5 retest — does recursive-abstraction reuse pay once F1 and F2 are fixed?

Track 5 ([`../recursive-abstraction/RESULTS.md`](../recursive-abstraction/RESULTS.md))
concluded that *"recursive abstraction currently delivers no measurable benefit,
and by the repo's own accounting it is a net cost at the scale the system can
search"*. It named two faults as the reason it could not test that claim:

* **F1** — a single-output module resolved to `product(BOOL)`, so every call site
  paid a `project` node and abstraction was strictly costlier than inlining for
  **all** call counts;
* **F2** — `exact_tensor` re-ran `Program.validate` per batch row, making module
  candidates 16–98× a primitive and capping the input width the search could
  afford. Track 5 wrote: *"fixing F2 is a prerequisite for testing the headline
  claim at all"*, because a non-collapsing composite needs ≥ 6 Boolean inputs.

Both are fixed on `main`. This is the re-run: costs re-verified from scratch,
targets whose flat minimum is **proved** rather than assumed, matched arms at 8
seeds on the scaffold where both routes fit and 24 on the scaffold where only the
abstracted one does, the module-on-output-path measurement repeated, and an
exhaustive enumeration cross-check that certifies what the search found.

Nothing in `tcn/` or `generators/` was modified. Proposed core changes are
diffs in §8. Reproduce with `PYTHONPATH=<repo>:. .venv/bin/python <script>` from
this directory.

---

## VERDICT

**Track 5's conclusion does not survive. Recursive abstraction pays now — and
the thing that was wrong was the accounting, not the search. But it pays because
it makes an otherwise unreachable target reachable, not because it is cheaper:
nothing in the system's objective can see the size saving, so reuse is *used*
where it is necessary and never *preferred* where it is merely economical.**

1. **The cost verdict is overturned, and it was an artifact of F1.** Re-measured
   from scratch: a single-output module call is `BOOL`-typed and needs no
   `project` node; **execution cost is at exact parity with inlining at every
   call count**; on track 5's own construction the description-size crossover
   moved from **4 call sites to 2**, and an 8-gate body at 8 sites is **0.38×**
   the inlined size. For the actual target the abstracted program is **19 496
   description bits against the flat program's 20 800 with the module definition
   fully charged, at identical execution cost, at only two call sites** (§1, §3).
   Track 5's "net cost at every scale the system can search" was measuring the
   projection node F1 forced on every call site.

2. **Track 5's methodological finding survives and was indispensable.** Its
   collapse — `MAJ3(a,b,c) xor MAJ3(b,c,d)` is a 3-gate function, not 9 —
   reproduces exactly with independent machinery. The retest target uses disjoint
   windows and its flat minimum is **proved** to be ≥ 7 by gate elimination over
   the full binary basis, against a verified 9-gate circuit, with the abstracted
   route at 3 nodes (§2).

3. **Its headline finding does not reproduce.** Track 5: the module on the output
   path in **0 of 20** arm-B runs, 0 of 10 successes. Here: **27 of 27
   successes**, on both scaffolds. Arm B is
   **8/8** on the wide scaffold where arm A is **0/8** (Fisher p = 1.6·10⁻⁴), and
   **19/24** on the tight scaffold where arm A is **0/24** (p = 7.4·10⁻⁹). Every
   arm-B success *is* the abstracted program (§4).

4. **The effect is not "more candidates helps", and it is not a search
   artifact.** Arm C — a module of the same verified size, the same interface,
   the same candidate count, verified to shorten MAJ3 by nothing — is **0/32**
   across both scaffolds, while having its module on the output path at exactly
   the measured chance rate. Exhaustive enumeration certifies the rest: arm A's
   230 400-program space contains **no** solution, arm C's 2 709 504-program
   space contains **no** solution, and arm B's contains **144, all of which use
   the module** (§5). At three nodes the abstraction is not an optimization, it
   is the only way to express the target.

5. **What still does not pay is *preference*.** Track 5's **F3 is unfixed and F1
   has made it decisive**: the only MDL term the repo has, `SoftProgram.complexity()`,
   is cost-weighted, and F1 put module cost at *exact* parity with inlining — it
   measures **20.43 for both routes**, so switching it on could not prefer
   abstraction. The quantity that does prefer it, description bits, has no
   differentiable surrogate, and `enumerate_fit` ranks by declaration order
   rather than by cost (§5.1, §8 R2). The visible consequence is in the runs:
   one arm-B success calls MAJ3 three times where two would do, and is accepted
   because it conforms. **Reuse is currently a capability, not an objective.**

6. **F2's fix was real but smaller than the handoff implies, and the gap it left
   is not closed.** The "4.0× a primitive" figure is batch-one and reproduces
   (3.8×); at the batch this experiment actually needs, a module candidate is
   still **105× a primitive**, and arm B costs **5.1 s per optimizer step against
   arm A's 0.23 s**. The fix bought roughly 3× per row, which was enough to reach
   6 inputs — and 6 inputs is the minimum for a non-collapsing target, so it was
   genuinely the prerequisite track 5 said it was. A further measured **7.45×**
   is available from deduplicating repeated argument rows (§8 R1).

---

## 1. Re-verified costs (`costs.py`, `costs_detail.py`, `crossover_track5.py`)

None of the handoff's "after" numbers were taken on trust. Every quantity below
was measured again from the current tree.

### 1.1 F1 is fixed, and it is the change that matters for the accounting

| check | result |
|---|---|
| a single-output module call's inferred type | `bool` — **not** `product(bool)` |
| projection node needed per call site | **no** |
| a two-output module still forms a product | yes, `tuple(bool, bool)` |
| module operator `cost` | equals the body's `execution_cost` (4.0 for minimal MAJ3) |
| module operator `gradient` | `none` (unchanged) |
| exact semantics of the call | verified against all 8 rows |

The consequence is structural, not just numeric: **a module is now a drop-in
candidate at an ordinary typed node.** Track 5 had to build tuple-typed "call
slots" plus projection nodes to offer one at all, which is why its scaffolds
carried extra machinery in both arms. Here every node is `BOOL` and the arms
differ only in whether that node's candidate list also contains module calls.

On **track 5's own crossover construction** (N independent MAJ3 calls on the same
three inputs, exposed as N outputs), rebuilt against the fixed tree:

| | description-bits crossover | execution-cost vs inlining |
|---|---|---|
| track 5, as recorded in its `crossover.json` | **4 call sites** | **never** crosses; 5 vs 4 per use, permanently |
| current tree (`crossover_track5.json`) | **2 call sites** | **parity at every N** (4·N vs 4·N) |

Both handoff claims reproduce. On this retest's own construction (N calls on
disjoint input windows, combined by `xor` — the shape the experiment actually
uses), the crossover sits one site later because the combine nodes are shared by
both forms:

| body | 1 site | 2 | 3 | 4 | 6 | 8 |
|---|---|---|---|---|---|---|
| 3-gate, module bits ÷ flat bits | 1.52 | 1.03 | **0.88** | 0.80 | 0.73 | 0.69 |
| 4-gate | 1.41 | **0.92** | 0.77 | 0.69 | 0.62 | 0.58 |
| 8-gate | 1.21 | **0.73** | 0.58 | 0.50 | 0.42 | **0.38** |
| execution cost ratio, every body, every N | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

The handoff's "8-gate body at 8 call sites is 0.28× inlined" measures the
same direction; on this construction it is 0.38×, the difference being the
shared combine chain. The two forms were checked to compute the same function
over all 64 rows before their sizes were compared.

### 1.2 F2 is fixed, but the fix is smaller than the headline and the gap it left is not closed

The claim "a 5-node module call is 4.0× a primitive (was 27.4×)" is a
**batch-one** comparison, and at batch one it reproduces:

| measurement (batch 1, minimal 4-gate MAJ3) | µs |
|---|---|
| module call via `exact_tensor` | 22.2 |
| primitive `xor` via `relaxed` | 5.8 → **3.8×** |
| primitive `xor` via `exact_tensor` | 12.7 → 1.7× |

And the memoization does exactly what it claims: `Program.validate` on the
module body costs **33.95 µs**, the memoized body run costs **5.80 µs**, so
removing the per-row revalidation is a **6.85×** speedup of the body call
(`costs_detail.json`, `validation_is_memoized: true`).

**But the soft search does not run at batch one.** A module candidate is
evaluated over the whole batch, and `exact_tensor` is still an O(batch) Python
loop while a primitive is one vectorized tensor op. Measured on the current tree:

| batch | primitive `xor` | MAJ3 module | ratio | module µs/row |
|---|---|---|---|---|
| 8 | 6.6 µs | 88 µs | 13× | 11.0 |
| 16 | 6.4 | 159 | 25× | 9.9 |
| 32 | 6.4 | 300 | 47× | 9.4 |
| **64** (this experiment's batch) | 6.5 | 682 | **105×** | 10.7 |
| 128 | 6.9 | 1263 | 184× | 9.9 |

So the per-row cost fell from track 5's ~29 µs to ~10 µs — a real **~3×** — and
the per-row cost is now flat in batch, confirming the revalidation is gone. What
did **not** change is the shape: the ratio to a primitive still grows linearly
with batch, and at the batch this experiment needs it is **105×**, worse than
the 16–98× track 5 reported at its smaller batches. About 54% of what remains is
the module body's own execution; the rest is the per-row `Value.unflat` /
`Value.flat` marshalling.

In the actual experiment this shows up as **arm B costing 5.11 s per optimizer
step against arm A's 0.23 s — 22×, for a 1.63× larger candidate set** (medians
over the 8 wide runs; an unloaded probe of the same scaffolds gave 3.29 s against
0.17 s, the same 19–22× ratio).

That is enough to make the 6-input target reachable, which is why this retest
could run at all: track 5's F6 projected 2 × 216 × 64 ≈ 27 600 exact module
executions per step at ~29 µs, "roughly 0.8 s", and called that regime "out of
reach today"; the same two call slots now cost **0.30 s**. It is not "module
calls are affordable". §8 R1 proposes a further **7.45×**, measured.

### 1.3 Batch-one latency

| program (2 MAJ3 calls + xor, 6 inputs) | p50 ms | p95 ms | description bits | operator cost |
|---|---|---|---|---|
| abstracted | 0.0166 | 0.0172 | 19 760 | 9.0 |
| inlined flat | 0.0116 | 0.0118 | 21 376 | 9.0 |

Abstracted is **1.43× slower** in wall clock while being **0.92×** the size and
**exactly equal** in the repo's own cost metric. Track 5 measured 1.73–1.75×
slower *and* 1.40–1.44× larger *and* 1.40–1.62× costlier. The size and cost
penalties are gone; a real-time penalty remains, and it is the same O(rows)
Python path as §1.2 rather than anything intrinsic to abstraction.

---

## 2. Targets whose flat minimum is proved, not assumed (`min_program.py`)

Track 5's key methodological finding was that its "9-gate" composite
`MAJ3(a,b,c) xor MAJ3(b,c,d)` is really a **3-gate** function — overlapping
windows of a symmetric sub-function collapse — so its headline experiment had
nothing for abstraction to save. It concluded that a non-collapsing composite
needs **disjoint** argument windows, hence ≥ 6 Boolean inputs.

Track 5's tool is a depth-first exhaustive search over straight-line programs. It
is exact but costs roughly (branching)^depth, so it terminates only when the
answer is *small*: it can confirm a collapse to 3 gates and can never confirm the
*absence* of one at 6 inputs, where the branching factor is 69 and the
interesting depth is 7–9. Assuming a target does not collapse is precisely the
error being corrected, so this retest proves it instead.

**Lower bound — gate elimination over the full binary basis B2** (all sixteen
two-input functions, negations free). If `f` depends on variable `v` then in any
circuit `v` feeds at least one gate; substituting a constant for `v` makes that
gate a function of its other input alone, so it can be deleted:

> `C(f) ≥ 1 + C(f | v = c)` for every live `v` and every constant `c`.

Recursing and maximising over `(v, c)` bounds the number of B2 gates, and every
`and`/`or`/`xor`/`not` node is at most one B2 gate, so it lower-bounds the *node*
count of any flat program in the scaffold basis too. The recursion bottoms out at
three live variables, where an exhaustive enumeration of every B2 circuit with at
most three gates gives the exact minimum for each function that has one and a
certified `≥ 4` for each that does not.

**Upper bound — an explicit circuit**, executed against the complete truth table.

| target | arity | rows | verified flat minimum | abstracted route |
|---|---|---|---|---|
| **MAJ3** (the module) | 3 | 8 | **exactly 4** (lower bound 4, 4-gate circuit exhibited) | — |
| **G = MAJ3(a,b,c) xor MAJ3(d,e,f)** | 6 | 64 | **≥ 7, ≤ 9** — does not collapse | **3 nodes** |
| `H = MAJ3(a,b,c) and MAJ3(d,e,f)` | 6 | 64 | ≥ 7, ≤ 9 | 3 nodes |
| *control:* track 5's `MAJ3(a,b,c) xor MAJ3(b,c,d)` | 4 | 16 | **3** — collapses, reproduced | — |
| **arm-C distractor**, truth table 134 | 3 | 8 | **exactly 4** | — |

All six inputs of `G` are live (verified by exact sensitivity), and the 9-gate
circuit was executed against all 64 rows.

**The bound tool is checked against ground truth wherever ground truth is
computable.** On all three targets small enough for track 5's exhaustive search
to terminate, the B2 lower bound and the exhaustively-searched scaffold minimum
**agree exactly**:

| target | exhaustive scaffold minimum | B2 lower bound |
|---|---|---|
| MAJ3 | 4 | 4 |
| track 5's collapsing composite | 3 | 3 |
| arm-C distractor (table 134) | 4 | 4 |

Re-deriving track 5's collapse is the tool's own control: a method that could not
see that collapse would report something above 3 on that row and fail visibly.
The bound is a lower bound, so it can only under-state; on these three it does
not under-state at all.

**The arm-C distractor is selected, not assumed.** Of the 256 three-variable
functions, 24 need ≥ 4 B2 gates; 12 of those have exactly 4 in the scaffold basis
and depend on all three inputs. The control is the first of those that is (a) not
in the majority family (majority under any input negation, or its complement) and
(b) **verified useless**: offering it as a free extra value leaves the minimum
circuit for MAJ3 at 4 gates, exactly as without it. Track 5's own arm C failed
this test — it noted `(a∧b, a∨b)` is "genuinely useful for a full adder", so its
control was not clean. Here it is measured:

| extra value available for free | minimum gates to compute MAJ3 |
|---|---|
| nothing | 4 |
| **the arm-C distractor (table 134)** | **4** — buys nothing |
| MAJ3 itself | 0 |

Truth table 134 is `xor(xor(a,b), and(c, or(a,b)))`.

---

## 3. Design (`common.py`, `solvability.py`)

**Target** `G(a,b,c,d,e,f) = MAJ3(a,b,c) xor MAJ3(d,e,f)`. The full 64-row truth
table is both the training set and the conformance set, so "success" is exact
program equivalence rather than a held-out estimate.

**Arms**, sharing one scaffold with identical nodes, depths and predecessor pools:

* **A** — no module candidates; the flat route only.
* **B** — MAJ3 offered as a `BOOL`-valued candidate at **every** node, over the
  six inputs (216 bindings per node). The scaffold makes no guess about which
  node is a "call site"; this is the faithful reading of *"register it as one
  typed candidate operator for later synthesis"*, and the dilution it causes is
  itself measured (§6).
* **C** — identical, with the verified-useless module of the same verified size.

**Two scaffolds:**

| | nodes | arm A candidates | arm B candidates | arm A space | arm B space |
|---|---|---|---|---|---|
| **wide** | 8 workhorse + output | 3 060 | 5 004 (+63.5%) | 1.9·10²² | 3.5·10²⁴ |
| **tight** | 3 | 256 | 696 | 230 400 | 2 709 504 |

The **wide** scaffold fits both routes: every node sees the six inputs and every
earlier node, so the verified 9-gate flat program has room. The **tight**
scaffold has 3 nodes, below the verified flat minimum of 7, so *only* the
abstracted route can solve it — and it is small enough to enumerate exhaustively
(§5). It is the abstraction-mandatory regime track 5 never reached.

**Modules are supplied by construction, and acquisition is measured separately.**
Both are the *minimum* circuit for their sub-function, verified in §2 and
re-checked against the truth table at construction. Folding acquisition into the
arms would have confounded two things that are separately interesting:

* the learned MAJ3 keeps a dead gate even after pruning in 4 of 8 seeds (5 live
  nodes where 4 suffice, cost 5 vs 4) — charging that transitively at every call
  site forever would tax abstraction for track 5's **F4**, not for abstraction;
* the arm-C sub-function is **not learnable by this synthesizer at all**
  (§4.1), so a learned arm C could not have been matched to arm B.

**Solvability gate.** Before running anything, every intended solution was
constructed by hand inside the scaffold the search actually gets, hardened, and
checked on all 64 rows. All four are conformant, so a 0% success rate cannot be a
scaffold bug.

| | conformant | live nodes | description bits (pruned, library charged) | without library | execution cost |
|---|---|---|---|---|---|
| wide, arm A, 9-gate flat | yes | 9 | 20 800 | 20 800 | 9.0 |
| wide, arm B, flat route still available | yes | 9 | 20 800 | 20 800 | 9.0 |
| **wide, arm B, module route** | yes | **3** | **19 496** | 9 440 | 9.0 |
| **tight, arm B, module route** | yes | **3** | **19 496** | 9 440 | 9.0 |

**This is the first configuration in either track where the abstracted program is
not more expensive than the flat one by the repo's own accounting**: 0.94× the
description bits with the module definition fully charged, and exactly equal
execution cost, at only two call sites. Track 5's equivalent rows were 1.10×
larger and 1.18× costlier.

All size and cost figures for discovered programs are measured **after pruning
nodes unreachable from the outputs**. `SoftProgram.export()` hardens and keeps
the whole scaffold, so an unpruned export describes the scaffold, not the
program (track 5's F4, still unfixed).

---

## 4. Learning results (`experiment.py` → `results.json`, `results_tight24.json`)

### 4.1 Module acquisition, measured separately (`--acquire`)

| sub-function | verified minimum | acquired | median attempts | median steps | live nodes after pruning | execution cost |
|---|---|---|---|---|---|---|
| **MAJ3** | 4 gates | **8 / 8** | 1 | **166** (range 66–2 541) | 5 (minimum 4) | 5 (minimum 4) |
| **truth table 134** (arm C) | 4 gates | **1 / 8** | 8 | 5 906 | 5 | 5 |

Two things to read off this. Track 5 measured a median of 699 steps and 2
restarts for MAJ3; here it is **166 steps and 1 attempt**, so acquisition is no
longer 10× the composite budget it is supposed to accelerate — the composite runs
below take a median of 50–85 steps, putting acquisition at 2–3× a single
composite search rather than 10×. And the learned module keeps a dead gate after
pruning in 5 of 8 seeds (§8 R3), which is why the arms are run with verified
minimal bodies instead.

### 4.2 The wide scaffold — both routes fit, 8 seeds, 300 steps

| | arm A (flat only) | arm B (MAJ3 offered) | arm C (useless module) |
|---|---|---|---|
| candidates in scaffold | 3 060 | 5 004 | 5 004 |
| **success (exact conformance)** | **0 / 8** | **8 / 8** | **0 / 8** |
| Fisher exact vs arm A | — | **p = 1.6·10⁻⁴** | p = 1.00 |
| module on the output path (chance 68.9%) | — | 8 / 8 | 5 / 8 |
| **module on the output path, of successes** | — | **8 / 8** | — |
| median steps to conformance | — | **85** | — |
| median final relaxed loss | 0.431 | 0.265 | 0.432 |
| lowest relaxed loss reached | 0.390 | 0.115 | 0.394 |
| median seconds / optimizer step | 0.231 | 5.11 | 6.24 |
| median live nodes, successes | — | **3** | — |
| median description bits, pruned, successes | — | **19 496** | — |
| median execution cost, pruned, successes | — | 9.0 | — |

### 4.3 The tight scaffold — only the abstracted route exists, 24 seeds, 300 steps

| | arm A | arm B | arm C |
|---|---|---|---|
| candidates in scaffold | 256 | 696 | 696 |
| **success (exact conformance)** | **0 / 24** | **19 / 24** | **0 / 24** |
| Fisher exact vs arm A | — | **p = 7.4·10⁻⁹** | p = 1.00 |
| module on the output path (chance 81.6%) | — | 24 / 24 | 23 / 24 |
| **module on the output path, of successes** | — | **19 / 19** | — |
| median steps to conformance | — | **50** (range 20–280) | — |
| median final relaxed loss | 0.483 | 0.371 | 0.393 |
| lowest relaxed loss reached | 0.483 | 0.118 | **0.002** |
| median seconds / optimizer step | 0.021 | 1.24 | 1.10 |
| median live nodes, successes | — | **3** | — |

The main run repeated seeds 0–7 of this scaffold in a separate process, and all
24 of those runs came back **bit-identical** — same conformance, same first
conformant step, all three arms. That is a reproducibility check, not an
independent replication: the seeds are the same, so those runs are **not** added
to any count above.

### 4.4 Track 5's decisive measurement, repeated

> Track 5: *"In **0 of 20** arm-B runs across both experiments did the module end
> up **on the output path** of the discovered program — 0 of 10 successes
> included it. When arm B succeeded, it succeeded by rediscovering the flat
> program and leaving the module calls in dead branches."*

**That does not reproduce. Here it is 27 of 27 successes** — 8 of 8 on the wide
scaffold and 19 of 19 on the tight one. Every single arm-B success put the module
on the output path, because in every case the discovered program *was* the
abstracted route. Two examples, exactly as recorded:

```
wide, seed 0, conformant at step 70, 3 live nodes:
    w1 = module:8ceedf7b792a7(a, c, b)
    w6 = module:8ceedf7b792a7(d, f, e)
    y  = xor(w6, w1)

wide, seed 1, conformant at step 50, 5 live nodes:
    w0 = module:8ceedf7b792a7(d, e, f)
    w1 = module:8ceedf7b792a7(b, c, a)
    w4 = module:8ceedf7b792a7(e, d, f)
    w7 = or(w0, w4)
    y  = xor(w1, w7)
```

The first is the intended 3-node program. The second is a **redundant** one: it
calls MAJ3 three times, `or`s two calls of the same function together, and is
exactly conformant anyway. Nothing in the objective prefers the smaller program,
so the search stops at the first conformant one it reaches, dead weight and all
(§8 R2).

**Arm C behaves exactly as a control should.** Its useless module was on the
output path in 5/8 wide and 23/24 tight runs — indistinguishable from the
measured chance baselines of 68.9% and 81.6% — and it produced **0 successes in
32 runs**. So "a module ends up on the output path" is nearly free and means
nothing on its own; "a module ends up on the output path *of a correct program*"
happened only with the right module, in every arm-B success and no arm-C run.

### 4.5 Relaxed loss is again not evidence of synthesis

Arm C on the tight scaffold reached a relaxed loss of **0.00204** while
`enumerate_fit` has certified, by exhausting all 2 709 504 programs, that **its
space contains no solution at all** (§5). A soft mixture fitting the data to
2·10⁻³ in a space where no discrete program can fit it is the sharpest available
demonstration of the gap `exact_max_error` was added to expose (FINDINGS §10,
F-soft). Every success figure above is exported-argmax exact conformance on all
64 rows, never relaxed loss.

### 4.6 What the search does with the module over time

Softmax mass at the arm-B wide nodes, median over 8 seeds (`tables_wide.json`):

| step | mass on module candidates (max over nodes) | mass on the *correct binding* |
|---|---|---|
| uniform prior | 0.403 | **0.00187** |
| 0 | 0.634 | 0.0043 |
| 20 | 0.673 | 0.0139 |
| 40 | 0.906 | 0.0411 |
| 60 | 0.977 | 0.101 |
| 80 | 0.995 | **0.138** |

Mass on *some* module saturates almost immediately — 39% of the candidates are
module calls, so that number is near-uninformative. The informative column is the
second: the correct binding climbs from 0.19% (uniform) to **74× that** by step
80, and runs that finish do so around there. The search is concentrating on the
right binding, not stumbling onto it.

---

## 5. The enumeration cross-check (`enumerate_check.py`)

`tcn/search.py` enumerates exactly the space `SoftProgram` relaxes, using only
`Program.execute` over the declared node candidates. That separates two very
different failures: *the abstraction does not pay* from *the gradient search
cannot find it*. Run on the tight scaffold, exhaustively:

| arm | module offered | programs evaluated | exhausted | solved | seconds |
|---|---|---|---|---|---|
| **A** | none | **230 400** | yes | **no** | 9.8 |
| **B** | MAJ3 | **2 709 504** | yes | **yes** | 630 |
| **C** | the verified-useless module | **2 709 504** | yes | **no** | 353 |

The program enumeration returns for arm B is exactly the intended one:

```
n1 = module:8ceedf7b792a7(a, b, c)
n2 = module:8ceedf7b792a7(d, e, f)
y  = xor(n1, n2)
```

3 live nodes, 19 496 description bits with the module definition charged (9 440
without), execution cost 9.0.

Three things follow, and they are the strongest results in this retest:

1. **The abstraction is not merely cheaper here, it is necessary.** Arm A's
   entire 230 400-program space was exhausted with no solution. That is a
   certificate, not a search outcome: at three nodes the target is unreachable
   without the module.
2. **It is the *right* module that matters, not more candidates.** Arm C has the
   identical candidate count (696), the identical space size (2 709 504), a
   module of the identical verified size — and exhausting the space finds
   nothing. Any effect in arm B cannot be "enlarging the candidate set helps".
3. **The optimum genuinely uses the module.** An independent count, from packed
   truth tables with no `tcn` execution at all, finds **144 solutions in
   2 709 504 programs (1 in 18 816), and all 144 use a module.** That count
   reproduces `tcn/search.py`'s own `space_size` exactly, so the two methods
   agree. The degeneracy is MAJ3's argument symmetry: 6 orderings per window × 2
   window assignments × 2 `xor` argument orders = 144.

The wide scaffold cannot be enumerated (1.9·10²² programs in arm A, 3.5·10²⁴ in
arm B), which is why the tight scaffold exists.

### 5.1 But enumeration does not *prefer* the module — it has no objective beyond conformance

"Does enumeration select the module when it is genuinely optimal?" has a second
half. `enumerate_fit` appends every conforming selection and returns `found[0]`:
it ranks by **enumeration order**, not by description bits or execution cost.
Demonstrated on a two-candidate scaffold where a registered `and` module and the
primitive `and` compute the same function (`enumerate_preference.json`):

| candidate order | what enumeration returns | description bits of the choice |
|---|---|---|
| primitive declared first | `and` | 3 760 |
| module declared first | `module:c6f94b361fab5` | 7 744 |

Both are conformant, both have `cost` 1.0, and enumeration takes whichever comes
first. So on the tight scaffold enumeration selects the module because **every**
solution uses one (144 of 144), not because it costs less. Neither the discrete
reference nor the gradient search has any term that prefers the cheaper program
— see §8 R2.

---

## 6. Why the search can now reach the module (`diagnose.py`, `diagnose2.py`)

Track 5 measured the outcome — the module on the output path in 0 of 20 runs —
and attributed it to the gradient boundary. That attribution needs revisiting now
that the outcome has changed, so each candidate mechanism was measured
separately.

### 6.1 The gradient boundary is unchanged, and it is not the obstacle it looked like

| | measured |
|---|---|
| a module candidate's output `requires_grad` | **False** (a primitive `xor` on the same tensors: True) |
| module operator `gradient` declaration | `none` |
| arguments hard-thresholded at 0.5 before exact execution | yes: 0.49 → False, 0.51 → True |

All of track 5's G1–G3 reproduce exactly. But there is a distinction its report
did not draw, and it decides the outcome:

> **A module bound to the program *inputs* receives exact 0/1 arguments, so it is
> exact during soft search, not a step function of anything.**

The threshold only bites for a module fed by another *soft* node. Every module
binding in both this retest and track 5's scaffolds is drawn from the program
inputs, so in both cases the module candidate contributes a **constant, exactly
correct** column to the mixture. The choice logit over it is fully
differentiable (track 5's own G4 measured 0.021 of gradient on it). So
"gradient descent has strictly more signal along the flat route" is true of a
module *consuming* soft values and false of the module calls these experiments
actually offer. The gradient boundary was not what stopped track 5.

### 6.2 Dilution is real but survivable

| node | candidates | of which module calls | uniform mass on the one correct binding |
|---|---|---|---|
| tight `n1` / `n2` | 336 | 216 (64%) | **0.30%** |
| tight `y` | 24 | 8 | 4.2% |
| wide scaffold, all nodes | 5 004 | 1 944 (39%) | — |

The correct binding starts with 0.3% of the mass at each call node. What matters
is whether the loss gradient distinguishes it. Measured over 8 initializations,
ranking every candidate at each node by its logit gradient (most negative first,
i.e. the direction descent will raise):

| node | median rank of the correct module binding | of | percentile |
|---|---|---|---|
| `n1` | 52.5 | 336 | top **16%** |
| `n2` | 45.5 | 336 | top **14%** |
| `y` | 5.0 | 24 | top 21% |

So at initialization the correct choice is a *plausible* direction — comfortably
in the top sixth — but not the leading one at any node. There is signal, and it
is not decisive on its own.

### 6.3 There is real partial credit, and the combine node carries most of it

Peaking the softmax on individual correct choices (≈0.997 mass each, the rest
left uniform, no freezing — freezing routes the node through `exact_tensor` and
hard-thresholds its inputs, so the BCE clamp on a confidently wrong hard
prediction dominates and the reading is not the landscape descent traverses):

| choices held correct | loss | change from uniform |
|---|---|---|
| none (uniform mixture) | 0.887 | — |
| `n1` only (one module call) | 0.856 | −3.6% |
| **`y` only (the `xor` combine)** | **0.595** | **−33%** |
| `n1` and `y` | 0.509 | −43% |
| `n1` and `n2` (both calls, combine still uniform) | 0.719 | −19% |
| all three | 0.078 | −91% |

The composite is not the credit desert it looks like. Getting one module call
right buys little on its own (−3.6%), but the descent does not have to acquire
the calls first: **the combine node alone is worth −33%**, and from there each
call adds more. There is a monotone path from the uniform mixture to the
solution, which is why the search can walk it at all.

### 6.4 The module solution is a stable attractor once reached

Adding +3.0 to the logits of the three correct choices and then training
normally: the argmax is the intended program at **every** checkpoint, the mass on
each correct choice rises monotonically (n1 0.035 → 0.79, n2 0.029 → 0.81, y 0.39
→ 0.98), the loss falls to 0.0055, and the exported program is exactly
conformant. Nothing pulls the search back toward the flat route once it is on the
abstracted one — consistent with §5, where the flat route does not exist at three
nodes at all.

### 6.5 How wide is the basin? A logit prior of +1.5 makes it reliable

Sweeping the boost added to the three correct logits before training, 3 seeds
each, 150 steps (`diagnose2.json`):

| boost on the correct choices | resulting prior mass per choice | solved |
|---|---|---|
| **0.0** (the experiment's own initialization) | 0.30% | **1 / 3** |
| 0.5 | 0.49% | 1 / 3 |
| 1.0 | 0.81% | 2 / 3 |
| **1.5** | **1.33%** | **3 / 3** |
| 2.0 | 2.17% | 3 / 3 |
| 3.0 | 5.6% | 3 / 3 |

In **every** run at every boost, including the ones that did not reach exact
conformance, the argmax picked a module call at both `n1` and `n2`. The search
is not avoiding the module; it is failing to land on the right *binding* of it —
which of 216 argument triples — within the budget. Raising the correct binding's
prior mass from 0.3% to 1.3% is enough to make that reliable.

That localizes the remaining difficulty precisely, and it is not the gradient
boundary, not the accounting, and not dilution across *operators*: it is
**dilution across bindings**. A module of arity k offered over a pool of p ports
contributes p^k candidates to a node — 216 here, and 6³ is the smallest
interesting case. That is a scaffold-construction problem, not a claim about
recursive abstraction.

---

## 7. Limitations

* **Boolean domain, complete truth tables, supervised synthesis only**
  (`SoftProgram` + argmax export). The composite arms run search + argmax with no
  crystallization phase, following track 5's E2; track 5's E1 included one and
  found the scheduler accepted a median of **0** freezes per run, so its
  conformance numbers were effectively search + argmax anyway. No RL path.
* **Success is an anytime argmax criterion** — the hardened export is checked
  every 10 optimizer steps and the run stops at the first conformant checkpoint.
  This is track 5's criterion, is more generous than the repo's single
  end-of-training check, and is applied identically to every arm.
* **The arms are supplied with verified minimal modules**; acquisition is
  measured separately (§8 R4). This deliberately gives abstraction its best case:
  a correct, minimum-size, exactly-matching module with no acquisition debt
  charged against it. A negative result under those conditions is strong; a
  positive one is an upper bound on what reuse can do here.
* **Module bindings are drawn from the six program inputs only** (216 per node),
  not from every live port at that depth. This is an advantage handed to arms B
  and C — it puts the correct binding in a 216-element pool rather than a much
  larger one — and it keeps arm B's step cost tractable (§1.2).
* **The chance rate of "module on the output path" is high** and was measured
  rather than assumed (`baseline.json`): under uniform random argmax it is
  **68.9%** on the wide scaffold and **81.6%** on the tight one, because a module
  is a candidate at every node. The raw rate therefore carries little
  information on its own; the informative quantities are the rate *among
  successes* and the contrast with arm C, whose module can never be part of a
  correct program (§5).
* **Sample sizes** are 8 seeds per arm on the wide scaffold and 24 on the tight
  one. Only large effects are detectable at 8.
* **The exact flat minimum of the target is not determined** — it is proved to lie
  in [7, 9]. That is sufficient for the experiment (the abstracted route is 3
  nodes) but the bound is not tight.
* **The machine was heavily loaded throughout** (load average 37–52 on 20 cores;
  other agents working in the same tree). Wall-clock figures are ratios, not
  absolutes. Step counts are load-independent and are the primary cost metric.

---

## 8. Findings against `tcn/` (proposed diffs; no core file was modified)

### R1. `exact_tensor` should evaluate each *distinct* row once — measured 7.45×

F2's memoization removed the per-row `Program.validate`. What remains is the
O(batch) Python loop, and it is now the dominant cost of a module candidate
(§1.2: 105× a primitive at batch 64; arm B pays 3.29 s per step against arm A's
0.17 s). But exact execution is a pure function of the row, and a typed row over
finite carriers repeats: **a 3-input Boolean module has at most 8 distinct
argument rows whatever the batch**, so a 64-row truth table contains 8 real calls
and 56 repeats.

```diff
 # tcn/learning.py
 def exact_tensor(registry,op,xs):
     shape=torch.broadcast_shapes(*(x.shape[:-1] for x in xs)) if xs else ()
     batch=math.prod(shape) if shape else 1
     flat=[x.detach().expand(*shape,x.shape[-1]).reshape(batch,-1).cpu().tolist() for x in xs]
-    vals=[]
+    # Exact execution is a pure function of the row, and a typed row over finite
+    # carriers repeats: a 3-input Boolean module has at most 8 distinct argument
+    # rows whatever the batch, so a 64-row truth table holds 8 calls and 56
+    # repeats. Evaluate each distinct row once.
+    cache={}; vals=[]
     for i in range(batch):
-        args=[Value.unflat(t,x[i]) for t,x in zip(op.inputs,flat)]
-        vals.append(registry.exact(op,args).flat())
+        key=tuple(tuple(x[i]) for x in flat)
+        got=cache.get(key)
+        if got is None:
+            args=[Value.unflat(t,x[i]) for t,x in zip(op.inputs,flat)]
+            got=cache[key]=registry.exact(op,args).flat()
+        vals.append(got)
```

Measured (`proposed_dedup.json`, `dedup_overhead.json`), output verified
bit-identical in every case:

| case | distinct rows | shipped | deduped | ratio |
|---|---|---|---|---|
| MAJ3 over `(a,b,c)`, 64-row table | 8 / 64 | 723 µs | 126 µs | **5.7× faster** |
| MAJ3 over `(a,a,b)` | 4 / 64 | 740 µs | 87 µs | **8.5× faster** |
| every module candidate at every node of arm B's scaffold | — | 2.11 s | 0.28 s | **7.45× faster** |
| `int[8]` `add` module, 64 random rows (worst case) | 64 / 64 | 1106 µs | 1248 µs | 1.13× **slower** |
| `int[8]` `add` module, 512 random rows | 502 / 512 | 9050 µs | 9432 µs | 1.04× slower |

So it is a large win exactly where module candidates are expensive (small finite
carriers, which is where crystallized Boolean modules live) and a 4–13% penalty
where rows do not repeat. It is a memoization of a pure function: no new
semantics, no new operator, digests unchanged.

A second-order version would go further. A frozen module bound to *program
inputs* is constant across the whole search — the inputs do not change between
optimizer steps — yet `SoftProgram.forward` recomputes all 1 944 of arm B's
module candidates every step. Caching per (candidate, input batch) would remove
essentially all of it. That is a larger change to `SoftProgram`'s contract and is
flagged rather than drafted.

### R2. F3 stands, and F1 has made it decisive

Track 5's F3 — description size and cost never enter the synthesis objective —
is **unchanged on main**. `SoftProgram.complexity()` is referenced exactly once,
in `tcn/training.py`, behind `mdl_weight` which defaults to `0.`;
`tcn/synthesis.py::fit`, the path that learns and crystallizes modules and
synthesizes composites, optimizes `probe_loss + 0.001*(step/steps)*entropy` and
nothing else.

F1 turned this from a gap into the binding constraint, and the reason is precise.
`complexity()` is a softmax-weighted sum of operator **cost**, and F1 put a
module call's cost at exact parity with its inlined body. Measured on the wide
scaffold (`objective_check.json`):

| route | `SoftProgram.complexity()` | pruned execution cost | pruned description bits |
|---|---|---|---|
| 9-gate flat | **20.43** | 9.0 | 20 800 |
| 2 calls + xor | **20.43** | 9.0 | **19 496** |

**The one MDL term the repo has is exactly equal for the two routes, so enabling
it would not prefer abstraction.** The quantity that does prefer abstraction is
description bits, and nothing differentiable measures it: `description_bits` is
consulted only by `tcn.runtime.benchmark`, after the fact. If reuse is meant to
be *preferred* rather than merely *possible*, the objective has to say so, and
`complexity()` is not the term that can say it.

### R3. F4 stands — modules are registered with dead gates, charged forever

`SoftProgram.export()` hardens and keeps the whole scaffold and nothing prunes
nodes unreachable from the outputs, so a crystallized module carries dead gates
that are charged transitively at every call site for the life of the module.
Measured over 8 acquisition seeds: the learned MAJ3 keeps **5 live nodes where 4
suffice in 5 of 8 seeds** (12 088 bits / cost 5 against the minimum 10 056 /
cost 4) — a 20% permanent surcharge on every future call, for a defect that has
nothing to do with abstraction. `common.py::prune` is a 10-line implementation.

```diff
 # tcn/graph.py, on Program
+    def pruned(self):
+        """Drop nodes unreachable from the outputs and the state updates.
+
+        A hardened export keeps the whole scaffold, so its description bits and
+        execution cost describe the scaffold rather than the program. A module's
+        dead gates are charged transitively at every call site forever.
+        """
+        by={n.name:n for n in self.nodes}; keep=set()
+        stack=[v for _,v in self.outputs]+[u for _,_,u in self.state]
+        while stack:
+            k=stack.pop()
+            if k in by and k not in keep:
+                keep.add(k); stack+=list(by[k].candidates[by[k].selected or 0].sources)
+        return replace(self,nodes=tuple(n for n in self.nodes if n.name in keep))
```

with `Registry.register_module` and `runtime.save_program` calling it. It changes
digests, so it needs a spec note and a version bump.

### R4. The synthesizer cannot reliably acquire a 4-gate 3-ary sub-function

Not a module finding, but it bounds what recursive abstraction can be built from.
Over 8 seeds each, with an identical 5-node scaffold, up to 8 restarts and 800
steps per attempt:

| sub-function | verified minimum | acquired | median steps (successes) |
|---|---|---|---|
| MAJ3 | 4 gates | **8 / 8** | 166 |
| truth table 134 | 4 gates | **1 / 8** | 5 906 |

Both are 4-gate functions of three Boolean variables on the same scaffold. One is
learned in 161 steps; the other fails 7 times out of 8 after 6 400 steps. Whatever
distinguishes them, "the sub-function has 4 gates" does not predict acquirability,
and a library of reusable modules can only contain what stage 1 can actually
learn. This is why both arms here are supplied with verified minimal bodies.

---

## 9. Files

| file | what it does | output |
|---|---|---|
| `min_program.py` | verified flat minima: B2 gate-elimination lower bounds, explicit upper bounds, arm-C distractor selection | `min_program.json` |
| `costs.py` | F1 checks, module-vs-primitive candidate cost, crossover against inlining, batch-one latency | `costs.json` |
| `costs_detail.py` | reconciles the "4.0× a primitive" claim; isolates what the F2 memoization removed | `costs_detail.json` |
| `crossover_track5.py` | track 5's own crossover construction, re-run against the fixed tree | `crossover_track5.json` |
| `common.py` | scaffolds, targets, minimal module bodies, training loop, `prune`, `module_on_output_path` | — |
| `solvability.py` | proves every intended solution exists in the scaffold each arm gets; per-step timing | `solvability.json` |
| `experiment.py` | the matched arms A/B/C on both scaffolds, plus module acquisition | `results*.json`, `run*.log` |
| `enumerate_check.py` | `tcn.search.enumerate_fit` on the tight scaffold; independent solution count | `enumeration.json`, `enumeration_c.json` |
| `baseline.py` | chance rate of "module on the output path" under uniform random argmax | `baseline.json` |
| (inline, see §3) | discrete space size of each scaffold via `tcn.search.space_size` | `space.json` |
| `diagnose.py` | gradient boundary, dilution, partial credit, selection-gradient rank, basin | `diagnose.json` |
| `diagnose2.py` | partial credit without freezing; basin-width sweep | `diagnose2.json` |
| `enumerate_preference.py` | whether `enumerate_fit` ranks by cost or by declaration order | `enumerate_preference.json` |
| `objective_check.py` | whether the existing MDL term could prefer abstraction | `objective_check.json` |
| `proposed_dedup.py`, `dedup_overhead.py` | measures the R1 diff, including its worst case | `proposed_dedup.json`, `dedup_overhead.json` |
| `tables.py`, `render_arms.py` | summary tables, two-sided Fisher exact, section 4 rendering | `tables_*.json` |
