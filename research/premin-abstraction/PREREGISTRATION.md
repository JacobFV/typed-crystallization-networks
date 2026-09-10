# Pre-registration — written before any arm of this track was measured

Written 2026-09-09. At the time of writing, the only things that had been run
in this directory were (a) a **cost probe** of the program enumerator — how
long it takes to exhaust length 3, 4 and 5 over four Boolean inputs, and how
many programs come out — and (b) a still-running probe of length 6. No corpus
had been mined, no module published, and no arm of the later task executed.
The cost probe is a budgeting measurement, not an outcome: it fixes the caps
below and nothing else.

This track follows directly from FINDINGS §44 / `research/earned-abstraction`.
§44 established that the closed abstraction loop works mechanically and does
not pay, and located the cause: **exact minimisation is adversarial to
abstraction mining**. Each earlier task's minimum-gate program factors
differently, so `MAJ3` — the fragment that would have paid — is fused into its
wrapper in five of six solved programs, and the frequency statistics the rule
reads never see it.

## The hypothesis

> If the corpus is not minimised — or not *only* minimised — the reusable
> abstraction survives in enough programs for a frequency rule to find it.

## What is fixed in advance

### The later task, the scaffolds, the arms

Unchanged from §44, so every number is directly comparable. Imported from
`research/earned-abstraction/later.py` and `arms.py`, not restated:
`maj(a,b,c) xor maj(d,e,f)` over six Boolean inputs; tight (3-node) and wide
(9-node) scaffolds; `tcn.search.enumerate_fit` with tolerance 1e-3,
`rank='order'`, `max_programs = 2^24`, full sweep so the certificate survives;
gradient runs with `SoftProgram` + Adam, lr 0.05, 400 steps, conformance of the
argmax export checked every 10 steps — 24 seeds tight, 8 seeds wide. The
constant and uniform-random baselines are both 0.5 exact accuracy.

| arm | library |
|---|---|
| `arm1_none` | none |
| `arm2_earned` | §44's module, mined from the **minimised** corpus `C-min` |
| **`arm2p_trace`** | **the module mined from the primary non-minimised corpus `C-trace`** |
| `arm3_authored` | the hand-authored `MAJ3` (the ceiling) |
| `arm4_wrong_authored` | hand-authored truth table 134, same size and arity |
| `arm4b_wrong_mined` | §44's runner-up abstraction, same machinery |

`arm2_earned` is a **reproduction check**: it must give 0 conforming in the
2 709 504-program tight space with certificate `complete`. If it does not, the
run stops and that is reported as the finding.

Secondary arms are declared here so they cannot be chosen after the fact: if
the rank-1 proposal of `C-minall`, `C-plus1` or `C-plus1-one` differs from
`C-trace`'s **and** from §44's, it is run as `arm2p_<corpus>` under exactly the
same protocol and reported in the same table. If they agree, that is reported
as agreement and no extra arm is run.

### The corpora

All variants are declared now. Every variant is built over the same six
earlier 4-input tasks as §44 (`research/earned-abstraction/corpus.py`,
imported), in the same basis (`and`, `or`, `xor`, `not`, no constants, no value
recomputed twice). A program is *pruned*: every gate other than the last is
read by a later gate, and the last gate is the output. `k_t` is the certified
minimum length for task `t` (5 for `t1`–`t5`, 3 for `t6`; §44's figure, reused).

| corpus | definition |
|---|---|
| `C-min` | §44's corpus: **one** program per task, the first minimum-length program in that track's enumeration order. Loaded from `earned-abstraction/out/corpus_exact.json`, not re-derived. |
| `C-minall` | **every** pruned program of length `k_t`. Still minimal, but the tie-break is gone. |
| `C-plus1` | **every** pruned program of length `k_t + 1`. Genuinely non-minimal conforming programs. |
| `C-trace` | `C-minall` ∪ `C-plus1` — "the programs enumerated on the way, not just the minimum". **This is the primary non-minimised corpus and it feeds `arm2p_trace`.** |
| `C-plus1-one` | **one** program per task, the first of `C-plus1` in enumeration order. A same-size control for `C-min`, so a multiset effect can be told apart from a non-minimality effect. |

**Caps and budget.** Per task and per length band the mined corpus is capped at
`CAP = 32` programs. When the enumeration exceeds the cap, the retained subset
is the deterministic, enumeration-order-independent subsample "sort by the
sha256 of the canonical key, take the first 32". Uncapped counts are always
reported beside capped ones, and retention is reported over **both** the full
enumerated set and the capped subsample.

Exhaustion of a length band is attempted first. If a band takes longer than
**30 minutes for one task**, that band falls back to the declared sampler in
`enumerate_programs.sample` (uniform random prefix over the growing pool,
rejecting any gate that recomputes an existing value, then every single-gate
completion to the target; seeded, reproducible) and is labelled `sampled`,
never `exhaustive`. Which method was used is recorded per task and per band.

Every program in every corpus is rebuilt as a `tcn.graph.Program` and executed
through `tcn` against the full 16-row truth table before it enters the corpus.

### The rule

§44's rule, `mine.propose`, with `MAX_NODES = 5`, `MAX_HOLES = 4`,
`MIN_TASKS = 2`. R1, R2 and the rewriting are **imported unchanged**. Exactly
two changes are forced by a corpus holding several programs per task, and both
are declared here rather than tuned later:

1. **R3 counts base tasks, not corpus entries** — an abstraction is eligible
   when it occurs in at least `MIN_TASKS` distinct earlier *tasks*. On a
   one-program-per-task corpus this is identical to §44's rule, which is the
   check that `C-min` reproduces §44's ranked table exactly.
2. **R4 sums over corpus entries** — `saving(F) = Σ_e bits(P_e) −
   Σ_e bits(rewrite_e(F)) − bits(F)`. The definition is still charged once, so
   a larger corpus amortises it over more call sites. Absolute bit figures are
   therefore comparable only *within* a corpus; `saving_per_entry` is reported
   alongside.

Change 2 has a foreseeable side effect and it is named in advance: amortising a
one-time definition cost over more sites can make small fragments look better
in a bigger corpus. In §44's accounting a one-gate module *costs* ~224 bits at
every call site, so no amount of amortisation can make one positive — but
`C-plus1-one` is in the design precisely so the multiset effect and the
non-minimality effect can be separated, and it will be reported whichever way
it comes out.

### The mechanism measurement

For every corpus, and computed the fair way — over **all 24 ordered 3-subsets
of the four inputs**, not just `(a,b,c)`:

* how many programs retain a node whose **value** is `MAJ3`;
* how many retain a node whose value is §44's rank-1 fragment
  `M(x0,x1,x2) = x2 ∨ (x0 ∧ x1)`, and its runner-up `M'`;
* how many *tasks* have at least one such program;
* **structural retention**: whether a single canonical abstraction (one
  `Program.digest`, which is what a frequency rule actually ranks) computing
  majority is legal in the rule's R1/R2 sense, and in how many tasks and
  entries. Two programs can both hold `maj` and still offer the rule nothing if
  they realise it with different circuits.

### The compute cost

Reported for every corpus: wall seconds and DFS nodes expanded to build it,
and wall seconds to mine it, beside §44's cost for `C-min`. A rule that needs
100× the search to find a module that saves 10× is not a win and will be said
so.

## What would falsify the claim

Stated as the brief states them, and honoured:

* **The corpus was not the problem; the rule is** — if `arm2p_trace` **ties**
  `arm2_earned` (same conforming count in the same exhausted space, same
  gradient conformance).
* **Any module of that size helps and the selection still contributes nothing**
  — if `arm2p_trace` ties `arm4_wrong_authored` or `arm4b_wrong_mined`.
* **Mining does not reach what a human picks, even from a richer corpus** — if
  `arm2p_trace` still loses badly to `arm3_authored` (0 vs 144 conforming,
  0/24 vs 18/24).
* **The rule is degenerate** — if the rank-1 proposal on a non-minimised corpus
  is a 1-node fragment, or if nothing clears the reuse gate.

Any of these is a legitimate deliverable and is reported as it comes out.

## Discipline

* Enumeration beside every synthesis number, with its certificate
  (`unique` / `complete` / `none`), and `exhausted` reported separately from
  `evaluated`.
* Constant and random baselines beside every return.
* "No solution exists in this family" is distinguished from "search failed".
* The achieved configuration is reported, never the requested one.
* **The rule is not re-tuned until a number improves.** Its three parameters
  are swept once as a reported-regardless robustness check, exactly as §44 did,
  and the sweep is reported whatever the headline says.
* If a recorded §44 number does not reproduce here, it is reported as a
  finding, not worked around.
* Every corpus definition tried is reported, including any that is abandoned,
  with the reason.
