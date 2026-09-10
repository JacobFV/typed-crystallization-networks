# Pre-registration — the integrated flagship

Written and committed **before any arm runs** (house standard since §44). Any
change after the first arm is a numbered amendment in `RESULTS.md` naming the
item it replaces and the number it moves. Track directory:
`research/integrated-flagship/`. `generators/` and `tcn/` are **not modified**;
if a core hook proves unavoidable it will be minimal, off by default and
justified from a measurement recorded here first.

**What was run before this document, disclosed.** Three feasibility probes, used
for schedulability only and reported in `RESULTS.md`: (a) the GUI generator at the
configuration below realizes all 16 (colour, relation) pairs, 27–53 per 400
screens, 3 or 4 widgets per screen; (b) the language engine's English realization
of `press(<relation>(<colour>, box))` for 16 relation predicates (the four used
here render as "press to the right of the red box", "press to the left of …",
"press above …", "press below …"); (c) the computer kernel answers one
`execute()` in ~0.3–0.9 s on the default bridge and ~3 ms on the opt-in session
transport. No program was searched, fitted or scored.

---

## 1. The question and the criterion

> **Does the accumulated instruction set make increasingly integrated intelligence cheaper to acquire?**

The success criterion, verbatim:

> The inherited library must make a previously unseen integrated visual-language-computer task materially cheaper to acquire than the same substrate without the library, while a matched distractor library does not.

**"Materially", fixed now.** Let `cost(A)` be arm A's expected number of programs
evaluated to its first conforming program, and `episodes(A)` the environment
episodes that search consumes (§6). The criterion is **met** iff all of:

* **C1 (material saving).** `cost(N″) ≤ cost(N)/10` **and** `episodes(N″) ≤ episodes(N)/10`.
* **C2 (distractor).** The matched distractor library, run under the identical N″
  protocol, either has **no conforming program in its space** (exhausted,
  certificate `complete`, 0 conforming) **or** its cost is `> cost(N)/10` — i.e. it
  does *not* itself achieve a 10× saving.
* **C3 (it is a solution, not a fit).** The program N″ finds scores ≥ 0.90 on the
  held-out episodes, measured live in the environment, and beats every baseline
  of §7.

Two further pre-registered readings, from the owner's ladder:

* **N′ vs N.** If `cost(N′) > cost(N)/10` the schema is **organisational reuse,
  not learned intelligence**, stated as such.
* **N″ vs N′.** The learned specialization prior "earns its name" iff
  `cost(N″) ≤ cost(N′)/10`.

---

## 2. The integrated task

A research-side `Generator` subclass, `integrated`, composes the three shipped
generators **by import, unmodified**:

| part | source | what it contributes |
|---|---|---|
| screenshot | `generators/gui` `Implementation.initialize` / `observe` / `advance` | raw pixels (`image_value`), widget layout, owner map, `press` |
| instruction | `generators/language` engine, `get_language('english').render` | raw text bytes realized from the term `press(<relation>(<colour>, box))` |
| actuation + reward | `generators/computer` `execute()` kernel | the click is written into the kernel filesystem; the reward is read back from it |

**Screen** (`generators/gui` configuration): `resolution 16, widgets 4, nesting 1,
min_size 3, palette 8, palette_levels 2, gap 0, colour_mode random, horizon 1`.
At `palette_levels 2` the palette is the 8 corner colours of the RGB cube with
channel values {30, 240}; with at most 4 widgets and 2 colours each, the colour
assignment is injective (`palette ≥ 2 × widgets`), so a colour names a widget.

**Vocabulary.** Colours {red, green, blue, yellow} — the four words that are both
in the language generator's own `COLORS` and in the palette. Relations
{right_of, left_of, above}. Three relations, not four, because with the engine's
realization these three are separated by **one** byte (`T[13]` ∈ {r, l, h}) while
adding `below` needs a second address; this is the smallest vocabulary that
exercises both screen axes, and the choice is stated here rather than discovered.

**Episode.** Episode `(seed, index, split)` requests the combination
`index mod 12` (4 colours × 3 relations). Sub-draws `k = 0, 1, …` of the GUI
generator's own address are taken until the screen has a child of the requested
colour with a neighbour in the requested direction (a deterministic rejection
rule, reported with its rejection counts). The target is that neighbour.
Achieved difficulty — widgets per screen, anchor and target sizes, text lengths,
rejections — is reported, never the request.

**Action.** One action type, `click(pos)`, `pos` a pixel index. The environment
resolves `pos` through the GUI generator's own owner map, applies the GUI
generator's `press` when the widget kind accepts one, writes the clicked widget
id to `/home/agent/pressed.txt` in the computer kernel, and reads it back.
**Reward 1 iff the kernel's file content equals the target id.** Horizon 1.

**Splits.** Training: 12 episodes, train split, indices 0–11 (one per
combination). Held-out: 36 episodes, test split, indices 0–35 (three per
combination). One seed set (`seed 0`), one palette, one relation vocabulary —
**single-configuration evidence**, labelled as such wherever it is cited.

**Second configuration (robustness, pre-registered).** Identical except `gap 1`,
which puts one pixel of the parent between siblings, so the clicks that land in a
neighbour move from 1 pixel / 1 row to 2 pixels / 2 rows. It exists to test
whether an inherited schema or prior is knowledge or a coincidence of constants.

**Supervision for acquisition: the environment reward only.** The composed task
declares no intermediate probe. The GUI generator's `hierarchy`/`owner` probes
and the language construction are used for diagnostics and for the equivalence
checks of §6.3, never as a search target and never as a program input. This is
the decision that makes the flat search genuinely expensive: without
intermediate targets the search cannot be staged, so its cost is the *product* of
its parts, not their sum (§64 charged the staged sum and found the saving bounded
by the largest stage).

---

## 3. The program family (the coarse graph, identical in every arm)

The scaffold is authored once (hand-initialisation **H1**, as in every scaffold in
this repository; not ablated — without a graph there is no scaffold). Inputs:
`text` (the language observation) and `pixels`. Fifteen choice slots:

| slot(s) | kind | meaning |
|---|---|---|
| `cpos` | ADDR | address into the text for the colour word's discriminating byte |
| `lc1, lc2, lc3` | LIT | literals compared with that byte by `eq` |
| `K1 … K4` | GROUND | the target colour (an RGB byte triple) chosen for each colour class |
| `ra` | ADDR | address into the text for the relation's discriminating byte |
| `lr1, lr2` | LIT | literals compared with that byte |
| `M` | MATCH | the per-pixel test "this pixel has the target colour" (a registered module used by `filter`) |
| `X1, X2, X3` | STEP | the click, computed from the anchor's extent |

Semantics, all universal operators:

```
c   = index(text, cpos)
t   = mux(eq(c,lc1), K1, mux(eq(c,lc2), K2, mux(eq(c,lc3), K3, K4)))
A   = filter(pair(positions, {(pixels, t)}); module = M)      # anchor pixels
lo  = reduce_min(A);  hi = reduce_max(A)                      # empty A is an error
r   = index(text, ra)
clk = mux(eq(r,lr1), X1, mux(eq(r,lr2), X2, X3))              # X_j = op(base, off), base ∈ {lo, hi}
pos = idiv(clk, 3)
```

plus a fixed, declared encoder that turns `pos` into the typed action argument
through `tcn/policy.py`'s bounded decode (the computer track's pattern, §23).
Address arithmetic on text uses the text length's declared type
(`u32 ≤ 64`: out of bounds is an error, as in every text track); raster
arithmetic uses `u16` with `overflow='wrap'` (declared here so that a click
expression is total; an out-of-raster click is a miss). The program is **eager**:
every node executes, so an address slot that errors on any episode makes the
program unusable.

---

## 4. The arms

Each arm is the scaffold of §3 with a candidate pool per slot and an
enumeration order. Arms differ **only** in pools and order.

### 4.1 N — flat substrate, no library

Pools derived from types by a fixed rule, with no domain knowledge beyond the
declared constant ranges:

* **ADDR**: `tcn.graph.legal_candidates` over operators
  `identity, add, sub, mul, min, max`, ports `{length} ∪ {0,…,15}`, arities 1–2:
  **1,462** candidates per slot.
* **LIT**: `eq` against each lowercase letter a–z: **26**.
* **GROUND**: the 8 palette colours: **8**.
* **MATCH**: channel strides `(s_g, s_b) ∈ {1,2,3}²` × two 16-way truth tables
  combining the three channel tests (the §33 S0 shape with its strides freed):
  **2,304** registered modules.
* **STEP**: `op(base, off)` for `op ∈ {add, sub, mul, min, max}`,
  `base ∈ {lo, hi}`, `off ∈ {1,…,128}`: **1,280**.

Order: uniform random. `|S_N| = 1462² · 26⁵ · 8⁴ · 2304 · 1280³`.

### 4.2 N′ — schema only

The parametric structural schemas already in the repository, instantiated at this
task's widths, with **no** specialization:

* **ADDR ← §58/§60's M3 schema** (`research/motif-unification/schema.py`,
  `eq(index(buf, comb(x0,x1)), lit)`): `comb ∈ {add, sub, mul, min, max}` over
  holes bound to the same ports: 1,445.
* **MATCH ← §33/§64's S0 schema** (`rung3_widgets.same_scaffold`): channel strides
  fixed at +1, +2 (its plumbing), the two truth tables free: 256.
* **STEP ← §33/§64's step schema** (`rung3_widgets.offset_pool`, S1'/S2'):
  `op ∈ {add, sub}`, `base ∈ {lo, hi}`, `off ∈ offset_pool(W) = (6, 3W+3, 3, 3W, 9)`: 20.
* LIT and GROUND: no inherited schema covers them — flat pools.

Order: uniform random.

### 4.3 N″ — schema + learned specialization prior (the key arm)

N′'s pools, enumerated best-first by a **prior learned from the earlier domains'
instantiations** (§5). No hard vector is copied.

### 4.4 Controls and ablations

* **P — prior only**: N's flat pools with the learned prior. Separates prior from schema.
* **H — hard-transferred vector**: one program. Each slot takes the source
  domain's certified selection *by value* where the target pool contains it:
  ADDR ← §23's `sub(length, 1)`; LIT ← §19's `(` (byte 40, not in the pool →
  the pool's first candidate); MATCH ← §33's S0 vector `{rg: 7, same: 2}`;
  STEP ← §33's S2' step vector (`add`, pixel stride for `X1`, row stride for
  `X2`, `X3`); GROUND ← index 0. Its task quality is reported; §60 predicts it fails.
* **D — matched distractor library**: the same number of schemas and a prior of
  the same form, the same pool sizes, wrong content:
  ADDR `comb ∈ {mul, min, max, idiv, mod}` (1,445);
  MATCH strides fixed at (+2, +1) — channels swapped (256);
  STEP offsets `(4, 3W+2, 5, 3W−1, 7)` (20);
  prior = the §5 estimator fitted on the same source rows **with survival labels
  permuted within each source slot** (5 permutation seeds, each reported).
  Run as **D′** (uniform order, the N′ counterpart) and **D″** (distractor prior,
  the N″ counterpart).
* **R — right schemas, permuted prior**: N′ pools with D's permuted prior. Isolates
  whether the prior's *content* matters.
* **Prior feature ablation**: the prior without its observation-conditioned
  features (§5), under N″'s pools.

---

## 5. The learned specialization prior

**Data — the earlier domains' instantiations, every candidate labelled by whether
it survives in some conforming program of its own exhausted search:**

| kind | source slots | provenance |
|---|---|---|
| ADDR | §19 stage A `base` (41); §23 transform `pos` (136); §23 policy `ppos` (56) | re-run here from the tracks' own scaffolds; stage A on `hardening='none'` (the stream §19 used), recorded |
| LIT | §19 stage A `open` (256) and §23 policy `brand` (8), at each surviving address | same |
| TRUTH | §33 S0 `rg`, `same`; S1' `corner` | `research/visual-width-reuse/out/enum_32.json` (committed) |
| STEP | §33 S1' `back_a`, `back_b` (`sub(pos, off)`); S2' `step_w`, `step_h` (`add`) | same |
| GROUND | none — no earlier domain grounds a word in a colour | uniform |

**Features** (domain-independent, computed identically on source and target):

* ADDR: `(op, operand kinds ⊆ {length, constant, position}, V)` where **V** =
  the addressed byte is not constant across the training episodes' *unlabelled*
  observations (other inputs held fixed).
* LIT: `(occurs)` = the literal occurs at the chosen address in the training
  episodes' unlabelled observations. Conditional on the address.
* TRUTH: the 2-input truth-table id.
* STEP: `(op, offset class)`, offset class ∈ {pixel stride 3, row stride 3W,
  2·pixel, row+pixel, 3·pixel, other}.

**Estimator.** Per kind, `P(survive | feature tuple)` by Laplace smoothing
(α = 1, prior mean = the kind's overall survival rate), backing off to a coarser
tuple (drop the last feature) when a tuple is unseen. **Hand-initialised and
ablated**: the feature set (ablation of §4.4), α, and the back-off order.

**Search order (tiers).** Tier `t` restricts each slot to candidates whose score
is ≥ (the slot's best score) / 2ᵗ, `t = 0, 1, 2, …` until the full pool.
Programs in tier `t` not in tier `t−1` are visited in uniform random order.
`cost = Σ_{t<t*} |shell_t| + (|shell_{t*}| + 1)/(K_{shell_{t*}} + 1)`,
`t*` the first tier whose shell contains a conforming program. This is exact
given the conforming counts of the nested tiers.

---

## 6. Costs, certificates and how every number is obtained

### 6.1 Currencies

* **Programs to first solution**, expected under the arm's order: uniform random
  over a space of size `S` with `K` conformers gives `(S+1)/(K+1)` exactly.
* **Environment episodes**: `tcn.search.enumerate_environment` runs every
  training episode for every program, so `episodes = programs × 12`.
* **Compiled execution** of each arm's found program (`tcn/compile.py`,
  `compile_program`) — latency as a measurement only (no compiler work).

### 6.2 Certificates

The flat space is too large to enumerate program by program. Conformer counts
are computed by an **exact factorized counter**: candidates are grouped by their
per-episode behaviour, the relation side and colour side are summed separately,
and the click slots are counted by a superset transform over training-episode
subsets. It decides every program in the space, so its result carries
certificate `complete` (`unique` when the count is 1) in the sense of
`tcn/search.py`. Tier runs that stop at a first solution carry `none`.

### 6.3 Validation of the counter and the evaluator — required before any arm is reported

* **V1** a fast vectorized evaluator agrees with `Program.execute` on the real
  `tcn` program (clicks, or error, per episode) for ≥ 500 random programs across
  the pools of every arm, on all 48 episodes;
* **V2** the counter equals brute force on ≥ 20 random sub-spaces, and equals
  `tcn.search.enumerate_fit` (with the clicked pixel's packed colour scored
  against the target's — equivalent to the reward because colours are injective,
  checked per episode) on ≥ 3 sub-spaces;
* **V3** live rewards through the kernel equal the evaluator's hits for every
  found program and ≥ 50 random programs, and `tcn.search.enumerate_environment`
  (through an `EpisodeLedger` that builds this generator) returns the same
  conforming set on one small sub-space.

A failure of any of V1–V3 is reported and blocks the numbers it underwrites.

---

## 7. Baselines beside every accuracy

Best constant click (best single pixel on training, applied to held-out); oracle
constant (best single pixel chosen *on* held-out); uniform random pixel (exact
expectation); uniform random child widget (exact expectation). Plus, for N and
N′, the fraction of their conforming programs that also conform on the held-out
episodes (the chance their first found solution generalizes).

---

## 8. Falsification — declared now, honoured whatever happens

* **F1** `cost(N′) ≈ cost(N)` (within 10×) **and** `cost(N″) ≈ cost(N)`:
  inheritance buys nothing on the integrated task. **A first-class result.**
* **F2** the distractor saves as much as the library (C2 fails): the effect is
  scaffold shape, not learned knowledge.
* **F3 — the saving equals the space size.** Every N′/N″ pool is a subset of N's
  pool, so `K_arm ≤ K_N` and **the programs-saving can never exceed the ratio of
  space sizes** `|S_N| / |S_arm|` — this is a theorem, stated in advance. The
  finding is whether it *equals* it: the solution-retention fraction
  `K_arm / K_N` is reported, and a saving that equals the space-size ratio means
  §55's caveat stands (the library saves exactly the enumeration it skips).
  Episodes and wall clock are compared the same way.
* **F4** the task cannot be built from the existing generators without modifying
  them — reported with the smallest core change, not hidden.
* **F5** no specialization prior can be learned from the earlier domains (the
  estimator is flat, or the source rows carry no signal) — reported, and no
  hand-written prior is substituted.
* **F6** an inherited schema fails its conformance check at this task or at the
  second configuration (0 conforming, exhausted) — a counterexample in the CEGIS
  sense: recorded against the schema class, and the class is refined or split
  rather than extrapolated.

**Predictions, not criteria** (stated so a null cannot be re-narrated): the ADDR
schema pool is ~the flat pool, so N′'s saving comes from MATCH and STEP only; the
colour grounding (8⁴) and the literal chains are untouched by any inherited
object and will dominate N″'s cost; H fails; D′ has no solution; at `gap 1` the
STEP schema's offset pool lacks the needed 2-row step, so N′ and N″ have **no
solution there** while N and P do.

---

## 9. Instrumentation of *why*, for every inherited arm

Which inherited schema / class / prior feature was selected and whether it is on
the execution path of the found program (`Program.pruned` reachability); how much
re-specialization it needed (the found candidate's rank in its slot's prior
order, and whether it differs from H); how many candidate programs disappeared
relative to N (`|S_N| − |S_arm|`, and the expected-cost difference); how many
episodes disappeared; whether compiled execution got cheaper.

The library is evidence-carrying, not scored: `out/library.json` records per
class its semantic identity, schema reference, implementations (digests),
observed domains and widths, and reuse successes and failures **including this
task's**. No scalar abstraction score is computed.

---

## 10. Evidence standard and resources

`verify.py` recomputes every headline figure in `RESULTS.md` from committed files
under `out/` and prints PASS/FAIL per claim, exiting non-zero on any FAIL. Every
number in `RESULTS.md` is emitted by a script from `out/`. Heavy jobs run through
`capped.sh` (MemoryMax 20G, CPUQuota 800%, single-threaded BLAS, ≤ 4 workers,
memory floor 25 GB checked before each phase and logged in `out/memory_floor.log`).
Commit after every completed arm. Large artifacts stay out of git.
