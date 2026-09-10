# Pre-registration — depth encoding / width polymorphism

Committed **before any arm is run**. House standard since FINDINGS §44.
Amendments are appended with the number they replace; nothing above is edited.

Environment: repository `.venv`
(`/home/brandonin/Documents/typed-crystallization-networks/.venv/bin/python`),
worktree `research/depth-encoding/`. `tcn/` and `generators/` are **not**
modified by this track; every scaffold is built from the public API.

Baseline recorded before writing any arm:

* `pytest -q` → **324 passed, 13 failed**, all 13 in `tests/test_panel_interface.py`
  and `tests/test_generators.py::…[computer|embodied_world]` — the documented
  gitignored-`node_modules` failures (FINDINGS §45, §47). 337 collected.
* `python -m tcn train --episodes 160` → `initial_prediction_loss`
  **0.248835613951087**, `final_prediction_loss` **0.0022308224288281053**,
  `evaluation_mean_return` **4.0**, `fully_frozen` **true**,
  `frozen_evaluation_mean_return` **4.0**. Reproduces 0.248836 → 0.002231 at 4/4.

---

## The question

The `logic` generator's `program` observation is `tuple[3 × depth]` of scalars
(`generators/logic/generator.py:128`), so its **type**, not merely its content,
varies with depth. FINDINGS §15 sidestepped this with a second typed view
(`gates`, one `set` type at every depth) and did **not** answer whether a
varying-width observation can be consumed at all. §30 established the general
form: a module's input type names the observation, so a hardened artifact is
width-specific, and what transfers is the **selections**.

Two things are asked, in order, and step 3 is entered only if step 2 fails.

## Step 1 — width-entry audit (descriptive, no hypothesis)

Enumerate every site in `tcn/` where a type carrying a width is compared for
equality, or where a width is derived into a type or a parameter family, with
`file:line`. Complete means: for each load-bearing site, an **executable**
demonstration that a depth-1-typed artifact meets a depth-2 observation there
and the exact exception text is recorded. Written to `out/audit.json`.

## Step 2 — do selections alone transfer across widths?

### The schema

A depth-parametric builder `interpreter_scaffold(host, depth)` over the
**`program`** channel (the width-varying one). It unrolls the circuit:

* `bit_j = project(bits, j)` for `j < w`;
* for gate `i < depth`: `project(program, 3i / 3i+1 / 3i+2)` → `encode` →
  `int[8]`; `wires_i = tuple(bit_0…bit_{w-1}, gate_0…gate_{i-1})`;
  `wa_i = index(wires_i, ia_i)`, `wb_i = index(wires_i, ib_i)`;
  `tab_i = tuple(truth_j(wa_i, wb_i) for j < 16)`; `gate_i = index(tab_i, it_i)`;
* `relation`: **17** candidates — the 16 fixed `truth_j(wa_{d-1}, wb_{d-1})`
  plus `identity(gate_{d-1})`, the table-conditioned answer;
* `goal_relation`: **16** candidates, `truth_j(relation, goal)`;
* the `examples/joint.py` policy tail verbatim, constants at their declared
  values.

Exactly **two** nodes carry choice, with candidate counts **(17, 16)** at every
depth: space **272** at every depth. The *selection vector is therefore
depth-independent by construction*; whether it is **semantically** correct at an
unseen depth is what is measured. Everything else is single-candidate.

This is the "generic schema + concrete instantiated artifacts" direction of
`research/algorithm-resynthesis/DESIGN.md` §7. **No type check is relaxed**: each
instantiated `Program` is exactly typed for its own width and is validated by
`Program.validate`; only the integer selection vector crosses widths.

### Configuration, fixed now

* `BASE = {'inputs': 4, 'nondegenerate': True, 'min_relevant_inputs': 2}`
  (`fixed_inputs` absent → gate wiring is drawn at random), matching §15's
  difficulty axis.
* `HORIZON = 4`, the whole episode scored (the scaffold is feed-forward, so
  there is no settling window). Returns are on the same 0–4 scale as §15.
* fit episodes: `range(16)`, `split='train'`; held-out: `range(10000, 10064)`
  (64 episodes), `split='test'`. Objectives cycle `{invert: False/True}`
  positionally. Both taken from §15 unchanged.
* depths `{1, 2, 3, 4, 6, 8}`; corresponding `program` widths 3, 6, 9, 12, 18, 24.

### Arms

* **A1 — fit at depth 1, transfer to 2, 3, 4, 6, 8.** Exhaustive enumeration of
  all 272 discrete programs at depth 1, scored by actual environment return on
  the 16 fit episodes; certificate from `tcn.search.certificate_of` semantics
  (`unique` / `complete`). The **whole conforming set** is carried forward, not
  only the first member. Each conforming selection is applied to a scaffold
  rebuilt at each `d′` and scored on the 64 held-out episodes.
* **A2 — fit at depth 2, transfer to 1, 3, 4, 6, 8.** The same, the other
  direction, to test that transfer is not an artefact of the smallest width.
* **B — negative control, the record scaffold** (`bits` + `goal` only, no
  `program` port; choice nodes `relation` and `goal_relation`, counts (16, 16)).
  Its selection vector is width-free trivially. Expected: it transfers
  syntactically and carries **no** capability, at or below the best constant —
  demonstrating that "the vector is applicable" is not the claim being tested.
* **C — random-selection baseline.** 64 selections drawn uniformly from the
  272-program space at each `d′`, scored on the same held-out episodes.
* **D — hardened-artifact control (§30 on the width axis).** The depth-1
  hardened `Program` is handed a depth-`d′` observation; the exact exception and
  the `file:line` that raised it are recorded.
* **E — per-depth exhaustion.** All 272 programs enumerated independently at
  each `d′` on held-out episodes, to establish the achievable optimum and
  whether the transferred selection is inside the conforming set at that depth.
  This is what separates "no solution exists in this family at `d′`" from
  "transfer failed".
* Baselines beside every return: `always_true`, `always_false`,
  `uniform_random`, and `fraction_target_true`, rolled on the **same** episode
  addresses over the **same** scored window.

### Pre-registered success criterion (all must hold)

1. A1's depth-1 enumeration is exhausted and carries certificate `unique` or
   `complete`.
2. Every conforming depth-1 selection, rebuilt at each `d′ ∈ {2,3,4,6,8}`,
   scores mean held-out return **≥ 3.9 / 4**.
3. That is strictly above the best constant baseline and above arm C's mean at
   every `d′`.
4. Arm B stays at or below its best constant at every `d′`.
5. Arm D raises a type error — the hardened artifact does **not** cross widths.
6. No `tcn/` file is modified; the 337-test result and the shipped fixture are
   unchanged from the baseline above.

### Pre-registered falsification

* If any conforming depth-1 selection scores at or below the best constant at
  any `d′`, **selections do not transfer**, arm E says whether a solution exists
  at that `d′` at all, and step 3 opens.
* If arm E shows the conforming set at `d′` is empty, the finding is
  "no solution exists in this family at that width", not "transfer failed".
* If A1's conforming set has more than one member and they disagree at `d′`,
  the honest statement is that **enumeration order** decides what transfers, and
  the transfer is under-determined rather than established. This will be
  reported as such, following §47's treatment of enumeration order.
* If the only way to make an arm pass is to weaken a type check, that arm is
  reported as a **failed approach**, per the project owner's standing
  instruction. Any such attempt is recorded even when abandoned.

## Step 3 — entered only if step 2 fails

Evaluate the candidate directions (type variables, shape variables, bounded
dependent dimensions, generic schema + instantiated artifacts) against the
measurements, cost each, and prototype only the smallest. Not begun unless the
falsification above fires.

## Known traps, not to be rediscovered

* The 13 failing tests are environmental (`node_modules` symlink); verified at
  baseline **before** this directory existed.
* Probes are supervision, never model inputs.
* Surrogates are dead at operating distance (§16): `eq` is exactly 0.0 past
  |a−b| ≥ 11, `lt` past 17. Enumeration is the primary method here for that
  reason; any gradient arm is secondary and reported as such.

---

## Amendment 1 — three further controls (appended after the step-2 arms ran,
## before arms F, G and H were written or run)

Nothing above is edited or withdrawn. Arms A1, A2, B, C, D and E ran as
pre-registered and their outputs are committed at `59477f3`. This amendment
**adds** three arms; it replaces no criterion.

Why: A1/A2 passed with certificate `unique` at every depth, so the step-2
falsification did not fire. That makes the *reason* it passed the thing that now
needs a control. The transfer succeeded because the **schema** is depth-parametric
in a semantics-preserving way, and a passing result cannot by itself show that
this was load-bearing rather than automatic.

* **Arm F — wrong-schema control.** An interpreter variant identical in every
  respect except that `relation`'s 17th candidate reads `gate_0` instead of
  `gate_{d-1}`. At depth 1 those are the same node, so the depth-1 fit **cannot
  distinguish** the two schemas: it must return the same selection with the same
  certificate. Pre-registered expectation: F fits at depth 1 at 4.00 with
  certificate `unique` and the same vector `{relation: 16, goal_relation: 6}`,
  and then **collapses toward the baseline at every `d′ ≥ 2`**. If it does, then
  "the vector is applicable at another width" demonstrably does **not** imply
  "the capability transfers", and depth-1 evidence alone cannot certify a
  schema. If instead F also scores 4.00 at every `d′`, the depth-8 task is not
  discriminating and arms A1/A2 are correspondingly weakened; that would be
  reported as a defect in this track's task, not as a success.
* **Arm G — achieved difficulty, measured rather than requested.** Per depth,
  over the 64 held-out episodes: distinct circuits, distinct final-gate truth
  tables, achieved `relevant_inputs` counts, and the fraction of episodes whose
  answer differs from the majority answer. Reported instead of the requested
  configuration.
* **Arm H — what the width-invariant part actually costs.** Per depth:
  `Program.description_bits` and node count of the frozen artifact, against the
  bit length of the selection vector (`log2(17) + log2(16)`). This quantifies the
  ergonomic claim rather than asserting it.

No `tcn/` change is contemplated by any of these.
