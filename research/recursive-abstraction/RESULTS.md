# Track 5 — Does crystallized-module reuse measurably help?

ARCHITECTURE.md sec. 4 makes the headline architectural claim:

> A crystallized module is an immutable, versioned, discrete callable program
> with **no internal gradients**. Register it as one typed candidate operator
> for later synthesis: this is recursive abstraction. Its internal description,
> precision, latency, and storage still count toward complexity/cost; a call is
> not free. Sharing counts a module definition once plus its call sites, with
> execution cost charged per use.

The existing tests check the *mechanism* — a frozen module is callable, its
output carries no gradient, and `description_bits(registry) > description_bits()`.
Nobody had shown that reuse **helps**. This track ran the first end-to-end
experiment that could show a benefit.

Everything runs against unmodified `tcn/` and `generators/`. All code, logs and
raw JSON are under `research/recursive-abstraction/`.

---

## VERDICT

**Recursive abstraction currently delivers no measurable benefit, and by the
repo's own accounting it is a net cost at the scale the system can search.**

1. **Search.** Over 8–12 seeds per arm in two matched experiments, giving the
   search a crystallized, correct, exactly-matching sub-module as an extra typed
   candidate did not improve success rate or steps-to-conformance beyond noise:
   E1 **3/12 flat vs 2/12 with the module** (Fisher exact **p = 1.00**), E2
   **5/8 vs 7/8** (**p = 0.57**). In **0 of 20** arm-B runs across both
   experiments did the module end up **on the output path** of the discovered
   program — 0 of 10 successes included it. When arm B succeeded, it succeeded
   by rediscovering the flat program and leaving the module calls in dead
   branches.
2. **Cost.** For the intended 2-call solutions, the abstracted program is
   **1.40–1.44× larger** in description bits, **1.40–1.62×** more expensive by
   the repo's operator-cost estimate, and **1.73–1.75× slower** at batch-one
   exact execution than the flat program. The description-size crossover for a
   4-gate module is **4 call sites**; the execution-cost crossover is **never**
   (§3.1).
3. **Accounting.** The accounting *claim itself is correctly implemented*:
   definition-once, call-sites-each, cost-per-use, transitive through nesting —
   all four verified numerically (§3). What is missing is that none of it enters
   any synthesis objective (F3), and exports are never pruned, so the numbers
   describe the scaffold rather than the program (F4).
4. **Mechanism cost.** A module candidate is 16–98× more expensive to evaluate
   than a primitive candidate inside the soft graph (batch-dependent, ~29 µs per
   row), because it runs the module exactly, row by row, in Python, re-validating
   the module body every row (F2). This caps the input width the search can
   afford, and at the widths it can afford, no target is deep enough for
   abstraction to pay (F6).
5. **Amortization is never reached.** Arm B must first *acquire* the module.
   Median acquisition: 31 optimizer steps for the half adder, **699 steps and 2
   restarts** for MAJ3 — about **10× the composite search budget** it is
   supposed to accelerate. With a 4-call-site description-size crossover and a
   cost crossover that never arrives, a 2-call composite cannot repay that.

The mechanism works exactly as specified — registration, typing, digests, the
gradient boundary, and the full sec. 4 accounting are all correctly implemented
and were verified numerically. What has not been shown is that using it helps,
and the measurements point at three concrete reasons why it currently cannot
(F1, F2, F3).

---

## 1. Experimental design

### 1.1 Why these targets

Two constraints narrow the choice sharply:

* `truth_k` (k = 0..15) already provides **every** 2-input Boolean function as a
  primitive, so any 2-ary sub-function is already a primitive and reusing it
  proves nothing. A genuine sub-function must be ≥ 3-ary or produce a tuple.
* A module call resolves to `product(*module_outputs)`, so a one-output module
  yields `product(BOOL)`, **never** `BOOL` (measured, §3, M1). Every call site
  needs an explicit `project` node.

| | sub-function (the module) | composite target | measured flat minimum | abstracted route |
|---|---|---|---|---|
| **E1** | `HA(a,b) = (a xor b, a and b)`, 2 gates, 2 outputs | full adder `FA(a,b,cin) -> (sum,carry)`; HA twice + one `or` | **5 gates** | 2 calls + 4 projections + 1 `or` |
| **E2** | `MAJ3(a,b,c)`, 4 gates, 1 output | `H(a,b,c,d) = MAJ3(a,b,c) xor MAJ3(b,c,d)` | **3 gates** | 2 calls + 2 projections + 1 combine |

"Measured flat minimum" is not an assumption: `min_program.py` exhaustively
searches straight-line programs over the same vocabulary the scaffolds use
(`and`, `or`, `xor`, `not`, full sharing) and returns the shortest one.

* full adder: 5 gates — `t1=a∧b; t2=a⊕b; t3=cin∧t2; sum=cin⊕t2; carry=t1∨t3`
* half adder: 2 gates
* MAJ3: 4 gates
* **E2 composite: 3 gates.** `MAJ3(a,b,c) ⊕ MAJ3(b,c,d) = (a⊕d) ∧ (b⊕c)`.

That last line is a result in its own right and it was **not** the intended
design. E2 was built expecting a 9-gate flat target (4 + 4 + 1). Arm A found a
3-gate program, and the exhaustive search confirms 3 is optimal: a composite of
two overlapping majorities *collapses algebraically*. See finding **F6** — the
"reuse saves you the sub-function's gates" intuition needs a composite that
provably does not simplify, and in the Boolean regime this system can actually
search, such composites are hard to come by.

So both experiments landed in the same regime: **the abstracted route needs at
least as many correct choices as the flat route.** E1: 8 vs 7. E2: 5 vs 3. That
regime is worth reporting precisely because it is the regime the system is
currently in — the honest read is not "reuse hurts", it is "at the scale this
system can search, there is nothing for reuse to save".

### 1.2 The arms

Both arms share **one scaffold**: identical node set, identical depths,
identical bounded predecessor pools, identical Boolean vocabulary
(`and`, `or`, `xor`, `not`, `identity` — functionally complete). The scaffold
contains two tuple-valued "call slots".

* **Arm A (from scratch).** Call slots offer only `tuple(...)`, an inert
  pass-through. Arm A builds the composite from Boolean gates.
* **Arm B (with abstraction).** Stage 1 learns and crystallizes the
  sub-function on its own simpler task with the same machinery (`SoftProgram` +
  `Crystallizer`), registers it via `Registry.register_module`, and the call
  slots additionally offer `module:<digest>(...)` over every binding in the
  pool. A **fresh module is learned per seed**, so acquisition cost is measured
  per seed and no single lucky module is shared across runs.
* **Arm C (E1 only, control).** Identical to arm B except the registered module
  computes `(a and b, a or b)` — same typed interface, same internal size, same
  candidate count, learned by the same pipeline, but not the half adder. A vs C
  isolates the cost of enlarging the candidate set. *Caveat:* this control is
  weaker than intended — `and` and `or` of two bits are not orthogonal to a full
  adder, and in 2 of 8 arm-C runs (including the single success) the distractor
  module ended up on the output path. Read C as "a same-size, same-interface,
  differently-decomposed module", not "a useless one".

Nothing else differs, and arm A never loses expressive power.

### 1.3 Success criterion

`success` = at some checkpoint (every 5–10 optimizer steps) within the budget,
the **argmax-hardened** program (`SoftProgram.export()`, exactly what
`tcn.synthesis.fit` reports as `exact_conformance`) was exactly conformant on
the complete truth table (8 rows for E1, 16 for E2). `steps to conformance` is
the first such checkpoint; the run stops there. This is an *anytime* criterion —
strictly more generous than the repo's single end-of-training check — but it is
applied identically to every arm.

### 1.4 Solvability gate (`solvability.py` → `solvability.json`)

Before running anything, the intended solution for **each** arm was constructed
by hand inside the shared scaffold, hardened, and checked for exact conformance.
Without this, a 0% success rate would be indistinguishable from a scaffold bug.
All four are conformant.

| | conformant | choices the search must get right | live nodes | description bits (with library) | execution cost |
|---|---|---|---|---|---|
| E1 arm A flat | yes | 7 | 7 | 16 320 | 7 |
| E1 arm B module | yes | 8 | 9 | 31 656 (23 744 without library) | 13 |
| E2 arm A flat | yes | 11 (the 9-gate route I designed for) | 11 | 24 208 | 11 |
| E2 arm B module | yes | 5 | 5 | 26 592 (14 504 without library) | 13 |

Even in E2, where the module cuts the *designed* flat program from 11 nodes to
5, the total description with the library charged is 10% **larger** than flat and
the execution cost is 18% **higher**. And the real flat optimum in E2 is 3 gates,
not 9 (§1.1), so the true comparison is even less favourable. (`live nodes` =
reachable from the outputs; tcn exports the whole hardened scaffold, so pruning
is done in the analysis — finding F4.)

---

## 2. Learning results

### 2.1 E1 — full adder from half adders

12 seeds for arms A/B, 8 for the control C. 250-step search budget, then the
full `Crystallizer` schedule (8 rounds, 5 retrain steps, tolerance 0.005).

| | arm A (flat) | arm B (HA module available) | arm C (control module) |
|---|---|---|---|
| candidates in scaffold | 1 169 | 1 203 (+2.9%) | 1 203 |
| success (exact conformance) | **3 / 12** | **2 / 12** | **1 / 8** |
| Fisher exact vs A | — | **p = 1.00** | **p = 0.62** |
| steps to conformance (median) | 140 | 165 | 110 |
| steps to conformance (range) | 130 – 180 | 100 – 230 | 110 |
| final loss, median | 0.1397 | 0.1398 | 0.1391 |
| final loss on failures, median | 0.1423 | 0.1398 | 0.1394 |
| seconds/optimizer step (median) | 0.151 | 0.170 | 0.231 |
| exported description bits (median) | 24 304 | 24 320 | 32 552 |
| exported execution cost (median) | 10 | 10 | 14 |
| live nodes in exported program (median) | 6 | 5 | 5–6 |
| module selected *somewhere* in argmax | — | 1 / 12 | 8 / 8 |
| **module on the output path** | — | **0 / 12** | 2 / 8 |
| module on output path when successful | — | **0 / 3** | 1 / 1 |

Crystallization phase (arms A and B only):

| | arm A | arm B |
|---|---|---|
| conformant after crystallization | 3 / 12 | 2 / 12 |
| fully frozen | 2 / 12 | 2 / 12 |
| freeze attempts (median per run) | 54 | 55 |
| accepted freezes (median per run) | 0 | 0 |
| rejection reasons (totals) | `runtime conformance` 465, `degradation` 34, accepted 29 | `runtime conformance` 509, `degradation` 55, `disconnected remaining region` 1, accepted 20 |

Module acquisition (stage 1, arm B only): 12/12 half adders acquired, median
**31 steps** (range 6–56), 7 912 bits, execution cost 3 (where the minimum is 2 —
one dead internal gate, see F4). 11 distinct digests from 12 seeds, and the two
seeds that learned the identical program produced the identical digest: content
addressing works.

Reading:

* **No difference between arms** on success (3/12 vs 2/12, p = 1.00), on final
  loss (0.1397 vs 0.1398), or on steps to conformance. The control arm C is in
  the same band.
* Arm B's module was on the output path in **0 of 12** runs, including 0 of its
  2 successes. Unlike E2, arm B's argmax rarely even *selected* a module
  (1/12) — the E1 call slot has 9 `tuple` and 9 module candidates, so the coin
  is fair here, and the module lost it.
* The exported description-bit medians are an independent confirmation: arm B
  (24 320) is indistinguishable from arm A (24 304) because no hardened arm-B
  program selected a module, so no library was charged; arm C (32 552) carries
  its module's ~8 000-bit definition because 8/8 of its hardened programs did
  select one. The accounting tracks reality.
* Arm C's distractor module *was* on the output path in 2/8 runs, including its
  single success — see the §1.2 caveat: `(a∧b, a∨b)` is genuinely useful for a
  full adder, so C is not a clean "useless module" control.
* **The crystallization scheduler almost never accepts a freeze in this
  scaffold.** Median accepted freezes per run: 0 in both arms. The dominant
  rejection reason is `runtime conformance` (465 + 509 of 1 112 attempts) — the
  freeze passes the loss-degradation test but the hardened program fails exact
  conformance, so it is rolled back. `disconnected remaining region` fired once.
  This is arm-independent and belongs to track 1, but it means E1's
  conformance results are effectively "search + argmax", the same regime as E2.

### 2.2 E2 — sliding-window majority from MAJ3

8 seeds per arm, 300-step budget, identical scaffold, no crystallization phase
(search + argmax only).

| | arm A (flat) | arm B (MAJ3 module available) |
|---|---|---|
| candidates in scaffold | 2 330 | 2 458 (+5.5%) |
| success (exact conformance) | **5 / 8** | **7 / 8** |
| Fisher exact, A vs B | — | **p = 0.569** |
| steps to conformance (median) | 50 | 70 |
| steps to conformance (range) | 30 – 80 | 30 – 90 |
| final loss, median | 0.063 | 0.078 |
| final loss on failures, median | 0.0006 | 0.0008 |
| seconds/optimizer step (median) | 0.200 | 0.517 (**2.6×**) |
| exported description bits (median) | 29 000 | 42 384 (**1.46×**) |
| exported execution cost (median) | 13 | 21 (**1.6×**) |
| module selected *somewhere* in argmax | — | 8 / 8 |
| **module on the output path** | — | **0 / 8** |
| module on output path when successful | — | **0 / 7** |

Module acquisition (stage 1, arm B only, charged to arm B): 8/8 modules acquired,
median **2 search attempts** and **699 optimizer steps** (range 106–1901) to get
a crystallized MAJ3 — i.e. arm B spends roughly **10× the composite budget**
just obtaining the module, before the composite search begins.

Three things to read off this table:

1. **The 7/8 vs 5/8 gap is noise** (p = 0.57), and it is not attributable to the
   module: in every arm-B run, the module was selected at both call slots but
   *neither call slot was reachable from the output*. That "8/8 selected
   somewhere" figure is an artifact — 64 of the 68 call-slot candidates are
   module calls, so an untrained argmax lands on one with 94% probability. The
   informative number is the output-path row: **0/8**.
2. Arm B pays 2.6× the wall-clock per step for a 5.5% larger candidate set,
   entirely because of the module-candidate evaluation path (F2).
3. The `final loss on failures` rows (0.0006 / 0.0008) show a distinct failure
   mode in both arms: the **soft mixture fits the data almost perfectly while
   the argmax program does not conform**. Three arm-A runs and one arm-B run
   ended with near-zero relaxed loss and a non-conformant discrete program. That
   is the crystallization gap, and it is arm-independent.

---

## 3. Description-size and cost accounting (`accounting_check.py` → `accounting.json`)

Checked on a registered half adder (2 internal nodes, definition = 5 904 bits,
unit cost 2).

| call sites N | description bits (with registry) | without registry | difference | execution cost |
|---|---|---|---|---|
| 1 | 10 712 | 4 808 | 5 904 | 2 |
| 2 | 13 888 | 7 984 | 5 904 | 4 |
| 4 | 20 240 | 14 336 | 5 904 | 8 |
| 8 | 32 944 | 27 040 | 5 904 | 16 |

**A1 — "a module definition counted once plus its call sites": CORRECT.** The
library surcharge is exactly 5 904 bits at every N (one copy), and each extra
call site adds a constant 3 176 bits — its own serialized operator, smaller than
the definition. Sharing amortizes as claimed.

**A2 — "execution cost charged per use": CORRECT.** Cost is exactly
`N × module.execution_cost` (2, 4, 8, 16); `Registry.resolve` sets a module
operator's `cost` to `m.execution_cost(self)`.

**A3 — transitive through nesting: CORRECT.** For top → middle → inner (inner
called twice inside middle), `top.description_bits(registry) −
top.description_bits()` = 16 640 = middle (13 264) + inner (3 376), exactly.
Execution cost is likewise transitive.

**A4 — a module reachable from two levels at once is still counted once:
CORRECT.** A program calling both a wrapper and the inner module it contains is
charged 10 640 = 4 736 + 5 904, one copy each.

**A5 — merely *offering* a module already charges its whole definition.**
`description_bits` scans all candidates, not just the selected one, so an
unfrozen scaffold pays for a module it never selects. Defensible (the library is
part of the system description), but it means "description size of a scaffold"
and "description size of the discovered program" are different quantities, and
only the latter is a fair arm comparison.

**M1 — a one-output module call resolves to `product(BOOL)`, not `BOOL`.** See
finding F1.

### 3.1 How many call sites before abstraction pays? (`crossover.py`)

Minimal 4-gate MAJ3 (definition 10 048 bits, cost 4), N call sites vs N inlined
copies:

| call sites | module-form bits | flat bits | module cost | flat cost |
|---|---|---|---|---|
| 1 | 17 152 | 10 168 | 5 | 4 |
| 2 | 22 424 | 18 504 | 10 | 8 |
| 3 | 27 696 | 26 840 | 15 | 12 |
| **4** | **32 968** | **35 176** | 20 | 16 |
| 8 | 54 056 | 68 520 | 40 | 32 |

* **Description-size crossover: 4 call sites.** Both experiments use 2 — the
  natural arity of the targets — so both sit on the losing side.
* **Execution-cost crossover: never.** A call is charged exactly the module's
  internal cost with no call overhead, but the module form additionally pays one
  `project` node per call site (5 vs 4 per use, permanently). For a one-output
  module the abstracted form is more expensive for *every* N (finding F1).

### 3.2 Batch-one latency (`latency.py`, via `tcn.runtime.benchmark`)

| program | p50 ms | p95 ms | description bits | operator cost |
|---|---|---|---|---|
| E1 flat | 0.109 | 0.113 | 23 864 | 10 |
| E1 abstracted | 0.190 | 0.196 | 33 536 | 14 |
| E2 flat | 0.136 | 0.140 | 28 928 | 13 |
| E2 abstracted | 0.236 | 0.249 | 41 672 | 21 |

Abstracted is **1.73–1.75× slower**, **1.40–1.44×** larger, **1.40–1.62×**
costlier. The extra latency is real work: `Registry.exact` on a module runs
`Program.run → execute → validate` on the module body at every invocation.

---

## 4. What a frozen-module candidate does to learning signal (`gradient_probe.py`)

ARCHITECTURE.md sec. 5 warns that "outside-in freezing must never silently
disconnect all learning signal to the interior". Measured:

* **G1.** A module operator has `gradient="none"`, so `relaxed()` routes it
  through `exact_tensor`, which detaches: `module_output.requires_grad = False`
  where a primitive `xor` on the same tensors gives `True`. No surrogate exists.
* **G3.** Arguments are hard-thresholded at 0.5 by `Value.unflat`
  (`bool(next(it) >= .5)`) before exact execution. Measured: `(0.49, 0.51)` →
  `HA(False, True) = (1, 0)`; `(0.51, 0.51)` → `HA(True, True) = (0, 1)`. During
  soft search a module candidate is a **step function** of the mixture, not a
  relaxation of it.
* **G2.** If a node's only consumer is frozen onto a module, the whole objective
  loses its gradient: after `freeze("call", module_index)` the loss has
  `requires_grad = False`. `Crystallizer.try_freeze` **does** catch this
  (`reason = "disconnected remaining region"`) and rolls back, so the safeguard
  is real and works. In the MAJ3 module-acquisition runs it fired 3 times across
  43 freeze attempts.
* **G4.** The *choice* is still learnable: the softmax weight over a module
  candidate receives gradient (measured 0.021). The search **can** select a
  module; it just cannot propagate any signal through one.

So in arm B the module path contributes a value that is constant with respect to
its arguments, while the flat path is differentiable end to end. Gradient descent
has strictly more signal along the flat route — which is exactly what the
output-path result (0/16) shows in practice.

---

## 5. Findings against `tcn/`

Proposed diffs only; no core file was modified (other agents are working in this
tree).

### F1. A one-output module can never be cheaper than inlining

`Registry.resolve` infers a module call's output as
`product(*(m.port_types()[v] for _, v in m.outputs))`, so a single-output module
resolves to `product(BOOL)`. Every call site must then spend a `project` node,
permanently adding 1 to execution cost per use, making the abstracted form more
expensive than inlining for **all** N (§3.1). It also means a module can never
be a drop-in candidate at an existing typed node — it needs its own tuple-typed
slot plus projections, which is what forced the two-node call-slot structure in
these scaffolds.

```diff
 # tcn/operators.py, in the module: branch of Registry.resolve
-            inferred = product(*(m.port_types()[v] for _,v in m.outputs))
+            outs = tuple(m.port_types()[v] for _,v in m.outputs)
+            inferred = outs[0] if len(outs) == 1 else product(*outs)
```

```diff
 # tcn/operators.py, in Registry.exact
         if n.startswith("module:"):
             m=self.modules[n]; out,_=m.run(dict(zip((k for k,_ in m.inputs),args)),registry=self)
-            return Value(op.output,tuple(v.raw for v in out.values()))
+            vs=list(out.values())
+            return vs[0] if len(vs)==1 else Value(op.output,tuple(v.raw for v in vs))
```

A semantic simplification (unit-arity products collapse), not a new primitive.
It changes existing digests, so it needs a spec note and a version bump.

### F2. Module candidates are 1–2 orders of magnitude costlier to evaluate

Measured (`candidate_cost.json`), median over 30 evaluations:

| batch | primitive `xor` | half-adder module | ratio | module µs/row |
|---|---|---|---|---|
| 8 | 46 µs | 758 µs | 17× | 95 |
| 16 | 8.8 µs | 456 µs | 52× | 29 |
| 32 | 10.7 µs | 902 µs | 84× | 28 |
| 64 | 20.0 µs | 1 970 µs | 98× | 31 |

(The batch-8 primitive figure includes first-call warm-up; from batch 16 up the
ratio grows cleanly with batch.) The ratio grows because the primitive is one
vectorized op while the module path is O(batch) Python: `exact_tensor` loops over rows and calls
`Registry.exact` → `Program.execute` → **a full `Program.validate` on every
single row**, re-resolving every operator contract from its dict. This is why
arm B costs 2.6× per step for a 5.5% larger candidate set.

`Program` is an immutable frozen dataclass, so validity cannot change — validation
should be memoized per (program, registry):

```diff
 # tcn/graph.py
+_VALIDATED = weakref.WeakKeyDictionary()   # program -> set of registry ids
     def execute(self, inputs, state=None, registry=None, selections=None):
-        r=registry or Registry(); self.validate(r)
+        r=registry or Registry()
+        seen=_VALIDATED.setdefault(self,set())
+        if id(r) not in seen: self.validate(r); seen.add(id(r))
```

(`Program` is hashable and frozen, so a `WeakKeyDictionary` keyed on the program
works; the point is that re-validating an immutable program per row is pure
overhead.)

### F3. Description size and cost are never part of the synthesis objective

`SoftProgram.complexity()` — the only place operator cost enters a loss — is
referenced exactly once, in `tcn/training.py` (`c.mdl_weight*self.model.complexity()`).
`tcn/synthesis.py::fit`, the path used to learn and crystallize modules and to
synthesize composites, optimizes `probe_loss + 0.001*(step/steps)*entropy` and
nothing else. `description_bits` is consulted only by `tcn.runtime.benchmark`,
i.e. after the fact.

So sec. 8's `lambda_mdl * L_program_description` is not applied in supervised
synthesis. Nothing pushes the search toward reusing a module to save description
size, and nothing penalises a module whose internals are dead weight. If reuse is
meant to be *preferred*, the objective has to say so. Adding
`+ mdl_weight * model.complexity()` to `fit`'s loss is a one-line change and is
the single most direct way to give the central claim a mechanism.

### F4. Exports keep dead scaffold nodes; crystallized modules keep dead gates

`SoftProgram.export()` hardens every node and keeps them all. Nothing prunes
nodes unreachable from the outputs, so `description_bits`, `execution_cost` and
`benchmark` of an export measure the **scaffold**, not the discovered program.
Measured on E1's intended flat solution: 23 864 bits / cost 10 as exported;
16 320 bits / cost 7 after pruning — a 32% overstatement.

This compounds through abstraction. The half adder learned in stage 1 has 3
internal nodes where 2 suffice (cost 3 vs 2); the learned MAJ3 has 5 where 4
suffice (cost 5 vs 4). Those dead gates are charged **transitively at every call
site, forever**. A `prune()` pass before `register_module` and before
`save_program` would be small and safe (implementation: `solvability.py::prune`).

### F5. "Relearning creates a new version and revalidates dependents" is half implemented

`register_module` names modules by content digest, so relearning does produce a
new version — verified: independently learned half adders produced several
distinct digests, and two seeds that learned the identical program produced the
identical digest, so content addressing works. But there is no dependent
registry: nothing tracks which programs reference `module:<old digest>`, so
nothing revalidates or migrates them. A stale dependent silently keeps working
against the old definition.

### F6. Composites of a reused sub-function collapse; the tractable envelope is too small

E2 was designed as a 9-gate flat target (two 4-gate majorities plus a combine).
`min_program.py` proves it is a **3-gate** function: `MAJ3(a,b,c) ⊕ MAJ3(b,c,d)
= (a⊕d) ∧ (b⊕c)`. Overlapping-window composites of a symmetric sub-function
degenerate. Non-degenerate composites need *disjoint* argument windows, hence
≥ 6 Boolean inputs for a 3-ary sub-function, hence ≥ 64 truth-table rows.

At 6 inputs a call slot has 6³ = 216 module bindings, so one optimizer step
would evaluate 2 × 216 × 64 ≈ 28 000 exact module executions at ~29 µs each
(F2) — roughly **0.8 s per call slot per step**, before any Boolean candidate is
touched. That regime is out of reach today. Conversely, inside the reachable
regime the two targets I could afford have measured flat minima of 5 and 3
gates, against abstracted routes of 8 and 5 learned choices — so there was
nothing for abstraction to save in either.

This is the structural reason the headline claim has no evidence yet: **fixing
F2 is a prerequisite for being able to test it at all.** F1 and F3 then decide
whether it can win once testable.

### F7. The freeze scheduler rejects almost everything on `runtime conformance`

Not a module finding, but it dominated E1 and it changes how E1's numbers should
be read. Across 24 E1 runs the `Crystallizer` made ~1 100 freeze attempts and
accepted a median of **0** per run; 974 of those rejections were
`runtime conformance` — the freeze passed the loss-degradation tolerance but the
hardened program then failed exact conformance, so it rolled back. Only 2/12
runs per arm ended fully frozen. Progressive crystallization therefore
contributed essentially nothing here, and E1's results are effectively
"search + argmax", the same regime as E2. This is track 1's question; flagged
here because any future module-reuse experiment that relies on the
crystallization phase will hit it first.

---

## 6. Limitations

* Two call sites per composite — the natural arity of both targets, but §3.1
  shows the description-size crossover is at 4 sites, so this study never
  observes the regime where MDL favours abstraction.
* Boolean domain only, complete truth tables, supervised synthesis
  (`SoftProgram` + `Crystallizer`); no RL and no `tcn/training.py` path.
* E2 ran search + argmax only (no crystallization phase), for runtime reasons.
  E1 includes it, but the scheduler accepted a median of 0 freezes per run
  (rejecting 465+509 attempts on `runtime conformance`), so E1's conformance
  numbers are also effectively search + argmax. Whether progressive
  crystallization would change the arm comparison is untested here; that
  interacts with track 1.
* Arm C is a weaker control than intended (§1.2 caveat).
* Success is an *anytime* argmax criterion (§1.3), more generous than the repo's
  end-of-training check, applied identically across arms. Several successes were
  found at very early steps with high relaxed loss, i.e. argmax luck; this
  inflates all arms equally.
* The machine was heavily loaded throughout (load average 40–60 on 20 cores).
  Wall-clock figures are ratios, not absolutes; step counts are load-independent
  and are the primary cost metric.
* Success rates are far from 1.0 and sample sizes are 8–12 per arm, so only
  large effects would be detectable. The output-path result (0/20) is the
  robust finding, not the success-rate comparison.

---

## 7. Files

| file | what it does |
|---|---|
| `experiment.py` | E1: half adder → full adder, arms A/B, plus crystallization |
| `experiment2.py` | E2: MAJ3 → sliding-window majority, arms A/B |
| `control_arm.py` | E1 arm C control (same-size, different-semantics module) |
| `solvability.py` | proves both arms' intended solutions exist in the shared scaffold; `prune()` |
| `min_program.py` | exhaustive minimum straight-line program length for each target |
| `accounting_check.py` | A1–A5, M1, M2 accounting checks |
| `crossover.py` | call-site crossover for description bits and execution cost |
| `latency.py` | batch-one exact latency of flat vs abstracted programs |
| `gradient_probe.py` | G1–G4 gradient-boundary measurements |
| `tables.py`, `fisher.py` | summary tables and the two-sided Fisher exact test |
| `candidate_cost.json` | module-vs-primitive candidate evaluation cost by batch |
| `*.json`, `*.log` | raw results |

Reproduce with `PYTHONPATH=<repo>:. .venv/bin/python <script>` from this
directory.
