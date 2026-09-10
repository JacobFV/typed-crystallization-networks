# Pre-registration — does §52 + §54 replicate on a second task family?

Committed **before any arm ran**. House standard since §44. Amendments, if any,
are appended at the bottom with the number each one replaces, as §51, §53 and
§55 did.

`tcn/` and `generators/` are not modified by this track. Nothing under
`research/residual-gap/` is touched.

---

## 1. The question

§52 established that **semantic pooling** — identity by `(arity, truth table)`
instead of `Program.digest` — fixes abstraction *identity*, reaching the
hand-authored ceiling while both pooled wrong-module controls score 0. §54
established that a **breadth-weighted ranking objective** `O2 = |T(c)|·Σ s_e − D`
fixes *transfer*, helping 5 of 5 held-out tasks against the incumbent
`description_bits` objective's 2 of 5 and a frequency-count baseline's 0 of 5.

Every number in both sections comes from **one** family: §44's six Boolean
tasks built around a `MAJ3(a,b,c)` window over four inputs. §54's own closing
caveat is *"one family, seven tasks — anecdote-strength on breadth"*, and the
partial second-family check it ran was the same six task *shapes* with the
window swapped for the §44 distractor — a relabelling, not an independent
family.

**A method demonstrated on one family is not a method.** This track builds a
genuinely different second family and runs §52's and §54's arms on it,
unchanged, to decide whether the two results are a method or an artifact of
one task family.

---

## 2. The second family, and why it is a fair test rather than a rerun

### The shared abstraction

    W4(w, x, y, z) = (w XOR x) AND (y XOR z)          "both pairs differ"

over **five** Boolean inputs `a..e`. It differs from `MAJ3` in every dimension
that could plausibly break the method:

| | family 1 (§44/§52/§54) | family 2 (this track) |
|---|---|---|
| window | `MAJ3(a,b,c)` | `W4(w,x,y,z) = (w⊕x)∧(y⊕z)` |
| arity | 3 | **4** — exactly `mine.MAX_HOLES` |
| algebraic character | symmetric, monotone, threshold | **non-monotone, non-threshold, a conjunction of two parities** |
| symmetry group on holes | all of S3 (6 elements) | an 8-element subgroup of S4 (`w↔x`, `y↔z`, pair swap) |
| window size in the basis | 4 gates | **3 gates** |
| corpus inputs | 4 | **5** |
| task minimum length | 5 | **4** |
| pooling key width | 8 rows | **16 rows** |
| module call-site cost | `ports³` | **`ports⁴`** |

The symmetry group is the property that makes this a *fair* test and not a
rigged one. `MAJ3` is invariant under **every** ordering of its holes, so §52's
pooling key could not possibly split it however a task realised it. `W4` is
invariant under only 8 of the 24 orderings, so hole ordering is not free: a
task that realises the window with its pairs discovered in a different order
can, in principle, land in a different pooled class. That is a live way for
semantic pooling to fail here that did not exist in family 1.

`W4` is also not reachable by any single gate of the basis and is not a
sub-function of any 2-gate program, so it is a genuine abstraction rather than
a renamed primitive.

### The tasks

Six corpus tasks, each `W4` over an ordered 4-subset of `a..e` combined with
the remaining input by one 2-input gate, each using a **different pairing**:

| task | definition | pairing |
|---|---|---|
| `u1_w_abcd_xor_e` | `W4(a,b,c,d) ⊕ e` | `{ab\|cd}` |
| `u2_w_acbd_and_e` | `W4(a,c,b,d) ∧ e` | `{ac\|bd}` |
| `u3_w_bcde_or_a` | `W4(b,c,d,e) ∨ a` | `{bc\|de}` |
| `u4_w_acde_xor_b` | `W4(a,c,d,e) ⊕ b` | `{ac\|de}` |
| `u5_w_abde_or_c` | `W4(a,b,d,e) ∨ c` | `{ab\|de}` |
| `u6_w_abce_and_d` | `W4(a,b,c,e) ∧ d` | `{ab\|ce}` |

`u1..u5` are the five leave-one-out held-out tasks; `u6` is present in every
corpus — §52's protocol with `t6`.

A **seventh** in-family task is fixed now and never enters any mining corpus:

    L1_w4_bdae_xor_c  =  W4(b,d,a,e) XOR c            pairing {bd|ae}

It carries the §52-style arm table, the way §44's `L1` did. Its pairing is used
by no corpus task.

### Why every number can carry a certificate

Every task depends essentially on all five inputs. A **pruned** straight-line
program of 3 gates over five inputs can reach at most four distinct inputs (the
root reads two ports, each of which is a gate reading two more), so every task's
minimum length is ≥ 4 by construction; `minimal.scaffold_min` certifies it at
exactly **4** by exhausting lengths 0–3. The same argument makes the two-node
evaluation scaffold unable to express any of them without a module, so the
no-library arm must exhaust at zero — the structural property that made §44's
tight scaffold a fair test.

All corpus enumeration is **exhaustive** (`enumerate_programs.exhaustive`, §46's
code, `N` 4→5 the only change): every pruned program at the certified minimum
length and at minimum + 1, with the same per-task cap of 32 and the same
deterministic sha256-of-canonical-key subsample, every retained program rebuilt
as a `tcn.graph.Program` and executed against its full 32-row truth table
before it enters the corpus. All evaluation enumeration is a full sweep with
`max_programs` above the space size, so `exhausted` and `certificate` mean what
they say.

### The off-family twin and the two controls

`F''` is the same six shapes with the window replaced by

    X4(w, x, y, z) = (w AND x) XOR (y OR z)

— same arity, same 3-gate size, a different table. It supplies the off-family
mining corpus (`arm4s_offfamily`) and the hand-authored wrong module.

Two control tasks, §52's `H_par` / `H_d134` pattern:

| control | definition | role |
|---|---|---|
| `H_par5` | `a⊕b⊕c⊕d⊕e` | shares nothing with either family |
| `H_x4` | `X4(a,b,c,d) ⊕ e` | the *wrong* window in a family shape — the `X4` module must solve it and the `W4` module must not, which is what makes the control live |

**Every `W4`-family arm, including the hand-authored ceiling, must score 0 on
both controls.**

---

## 3. Corpus bands — both are declared primary

At five inputs the family's tasks have only **1 to 3** minimum-length programs
each, so the `C-minall` band is a 10-entry corpus and cannot exhibit the
phenomenon §46 discovered (many equal-length minima realising one abstraction
under different digests). The band whose *geometry* matches §52's primary
corpus — 130 entries over 6 tasks — is `C-trace` (min ∪ min+1), at **136
entries over 6 tasks**.

Rather than pick one after seeing the tables, **both bands are declared primary
and both are reported in full**, with every falsification criterion below
applied to each. `C-trace` is the entry-count match to §52 and is the band the
headline verdict is read from; `C-minall` is the second configuration the §53 /
§55 caution demands ("a certificate at one config is not evidence"). If the two
bands disagree, the disagreement is the result.

**Disclosure — what was measured before this document was written.** The family
was constructed and its feasibility checked first: the six tasks' certified
minimum lengths (all 4), the two bands' program counts, the four full-corpus
ranked mining tables (`C-minall` and `C-trace` × syntactic and semantic
identity), and six single-cell enumerations on the compact scaffold
(`arm1_none`, hand-`W4` and hand-`X4` on `u1`, hand-`W4` on both controls,
hand-`X4` on `H_x4`) establishing that the scaffold is tight, exhaustible and
that the ceiling is non-zero. Those numbers are family construction, not arms;
they are all reported in `RESULTS.md`, and the two-band decision above exists
precisely so that nothing in the arm design is contingent on having seen them.
No leave-one-out corpus was mined and no objective other than the rule's own
was scored before this document was committed.

---

## 4. The arms — §52's structure, unchanged

Evaluated on `L1_w4_bdae_xor_c`, mined from the full six-task corpus, on the
two-node compact scaffold, exhaustively, per band.

| arm | library |
|---|---|
| `arm1_none` | none — the flat space |
| `arm2_syntactic` | rank 1 of the rule under `Program.digest` identity (`mine_multi`) |
| `arm2s_semantic` | rank 1 of the rule under `(arity, truth table)` identity (`mine_semantic`) |
| `arm3_authored` | the hand-authored `W4` — the ceiling |
| `arm4_wrong_authored` | the hand-authored `X4` — same arity, same size |
| `arm4b_wrong_mined` | highest-ranked arity-4 non-window class under **digest** identity |
| `arm4s_runnerup` | highest-ranked arity-4 non-window **pooled** class, same corpus |
| `arm4s_matched` | highest-ranked arity-4 non-window pooled class whose representative has the window's node count |
| `arm4s_offfamily` | rank-1 arity-4 pooled class mined from the off-family `F''` corpus |

The arity-4 restriction on the wrong-module rules is the counterpart of §52's
arity-3 restriction: a control must cost the search the same as the module it
controls for, or the comparison is confounded by space size.

**§55's caution is a declared reporting requirement, not an afterthought.** §52's
`arm2_syntactic` and `arm4s_runnerup` turned out to be one class under two
names. The digest and pooling key of every arm's module are recorded and the
table reports the **count of distinct classes**, not the count of arm names.

---

## 5. The ranking objectives — §54's, unchanged

Six corpora per band: the full six-task corpus and the five leave-one-out
corpora that remove one task **entirely, at every length**. `pool.build`,
`context.build_context` and `objectives.rank` are imported from §54 verbatim;
only the family changes.

| id | score | role |
|---|---|---|
| **O1** | `description_bits` summed over entries | the incumbent |
| **O2** | `\|T(c)\|·Σ s_e − D` breadth-weighted | §54's winner |
| **B1** | number of distinct tasks | §54's frequency control |
| O3 | per-task mean of per-entry saving | secondary |
| O4 | in-corpus leave-one-out CV saving | secondary |
| B2 | number of occurrences | secondary |

**O5 (measured execution cost) is excluded**, and the reason is declared here
rather than discovered later: §54 measured it to help 0 of 5 and to be an
objective about execution cost rather than transfer, and it answers none of the
three questions this track asks.

Beside every objective: the floor (`arm1_none`), the ceiling (`arm3_authored`),
and the two off-family controls. An exploratory `arm2s_window` — the
highest-ranked *window* class in the same table whatever its rank — is run in
the leave-one-out panel as §52 ran it, and is labelled exploratory.

---

## 6. Falsification criteria

Each fires on its own; each is reported plainly whether it fires or not. **A
negative is a first-class deliverable here and is the more useful outcome.**

* **F1 — identity is family-specific.** `arm2s_semantic` fails to strictly beat
  *every* pooled wrong-module control (`arm4s_runnerup`, `arm4s_matched`,
  `arm4s_offfamily`) in conforming count on `L1_w4_bdae_xor_c`. → §52's identity
  result does not replicate.
* **F2 — ranking is family-specific.** O2 does not help strictly more of the 5
  held-out tasks than O1. → §54's ranking result does not replicate. *This is
  the outcome that would matter most.*
* **F3 — the breadth term was doing nothing.** B1 helps as many held-out tasks
  as O2. → frequency was enough and §54's central control flips. Checked
  hardest: B1's rank of the window class is reported on every corpus, and the
  `|T|^α` endpoints are swept as §54 swept them.
* **F4 — the controls are dead.** Any `W4`-family arm, the ceiling included,
  scores above 0 on `H_par5` or `H_x4`; or the hand-authored `X4` fails to solve
  `H_x4`. → no transfer number in the table means anything.
* **F5 — the family is not exhaustible.** Any declared enumeration fails to
  exhaust, or any corpus band is sampled rather than exhaustive. → report what
  could be certified and report **no** uncertified numbers as if they were
  certified.
* **F6 — the scaffold is not tight.** `arm1_none` conforms on any family task. →
  the comparison is degenerate and every arm's margin is meaningless.
* **F7 — the arms are not distinct.** Two or more arms resolve to the same class.
  → report the count of distinct controls, not the count of arm names (§55).

## 7. What counts as "§52 + §54 replicates"

All four, jointly:

1. `arm2s_semantic` reaches the hand-authored ceiling's conforming count on
   `L1_w4_bdae_xor_c`, and every pooled wrong-module control is 0;
2. O2 helps strictly more held-out tasks than O1;
3. O2 helps strictly more held-out tasks than B1;
4. `H_par5` and `H_x4` are 0 for every `W4`-family arm including the ceiling,
   while the `X4` module solves `H_x4`.

Anything less is reported as a partial or a failure, in those words.

## 8. Integrity checks

* **IC1** — `pool.build`'s re-derivation must agree with `mine_semantic.propose`
  exactly (digest, tasks, entries, occurrences, `sum s_e − D` vs `saving_bits`)
  on every corpus. §54's own check, reused. If IC1 fails the track stops.
* **IC2** — O1's rank 1 must equal the rule's own rank 1 on every corpus.
* **IC3** — every corpus program is verified in `tcn` against its full 32-row
  truth table before entering a corpus.
* **IC4** — every reported enumeration must satisfy `evaluated == space_size`
  and `exhausted == true`.

## 9. Sensitivity

The rule's three parameters are swept on the primary band as §52 §8 swept them:
`MIN_TASKS ∈ {2, 3}`, `MAX_NODES ∈ {4, 5}`, `MAX_HOLES ∈ {4}` (the window's
arity is 4, so a lower ceiling removes the window by construction and is
reported as such rather than run). O2's rank-1 class is reported for every
cell. Nothing is tuned: every declared arm uses the rule's defaults
`MAX_NODES=5, MAX_HOLES=4, MIN_TASKS=2`.

## 10. Constraints honoured

Repo `.venv`; all 337 tests run and the count of failures compared against a
baseline taken on this worktree **before** any file of this track existed; the
shipped fixture must still reproduce 0.248836 → 0.002231 at 4/4 frozen; corpus
and enumeration artifacts stay out of git where they are large.
