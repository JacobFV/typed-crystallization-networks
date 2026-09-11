# Pre-registration — scaffold induction as a cross-domain outer loop

Written and committed **before any arm runs** (house standard since §44). Any
change after the first arm is a numbered amendment in `RESULTS.md` naming the
item it replaces and the number it moves. Track directory:
`research/scaffold-induction/`. `tcn/` and `generators/` are **not modified**; if
a core change proves unavoidable it is reported as the smallest diff and not
committed.

This is priority 4 of the post-§65 direction. §65's failures are binding on it,
and each is answered by a numbered clause below: the hand-written baseline is
mandatory (§4.4, C3), the criterion is **held-out** conformance (§5), the
distractor is checked to be able to succeed (§4.6, V3), both directions are
bounded (§6.3), and every figure is script-rendered with a `verify.py` (§10).

---

## 0. What was run before this document, disclosed

Four **schedulability probes** (`probe_timing.py`, committed): for each domain,
build the base scaffold, enumerate its typed edits, and time one family decision
on a sample of them. Their purpose was to fix the resource plan and the two
edit-space bounds of §3.3. Recorded here so the numbers cannot be re-used as
findings: **no repair label, no arm ordering and no criterion outcome was
computed, and nothing from those runs is quoted in `RESULTS.md`.** The base
scaffolds were found to be solvable (a conforming member exists on all
episodes), which is a precondition of the corpus construction in §2.3 and is
re-derived there under certificate.

---

## 1. The question and the criterion

> When a scaffold has no exact solution, does an **inherited body of edit
> knowledge** make the repair cheaper on a **genuinely new domain** — enumerating
> typed structural edits without knowing the repair in advance?

§50 is the incumbent: a plain operator sweep over the holes of a failed scaffold
repaired **21 of 24** cases with no diagnosis at all. That result is scored on
**training** conformance and on a corpus of 3 tasks in 2 domains. This track
must beat it, or report that it did not.

**The criterion, fixed now.** Let `E(A)` be arm `A`'s **exact expected number of
typed edits enumerated before the first repair**, averaged over the corpus
(§6.1), where *repair* is defined by **held-out** conformance (§5). The
inherited edit prior earns its name iff **all** of:

* **C1 (beats no library).** `E(N″) ≤ E(N)/2` on the held-out domain.
* **C2 (beats the structure alone).** `E(N″) ≤ E(N′)/2`.
* **C3 (beats the obvious hand-written prior).** `E(N″) ≤ E(H1)/2` **and**
  `E(N″) ≤ E(H2)/2`. §65's entire effect vanished against this comparison and it
  was missing from its pre-registration; it is a criterion here, not a control.
* **C4 (the distractor does not).** The label-permuted distractor `D1`, run under
  the identical protocol, does **not** satisfy C1 — and `D1`'s edit space is
  shown to contain a repair before C4 is read (validity check V3).

Factor 2, not 10: the edit space per scaffold is 10²–10³ (§3.4), so a 10×
separation is not available at this corpus size and demanding it would make the
criterion untestable. This is fixed now and is not revisited.

**What a null looks like, stated in advance.** `E(N″) ≈ E(N)` **or**
`E(N″) ≥ E(H1)` is the null. It is a first-class result and will be reported as
the finding, with the failed criterion kept and no metric substituted. Given
§65, §57 and §50, **I expect the null**: the specific prediction recorded here is
that the hand-written minimality-and-novelty prior H1 is within a factor 2 of the
learned prior N″, and that the schema-only arm N′ is within a factor 2 of the
flat arm N.

---

## 2. The corpus

### 2.1 Domains

Four, each a task this repository already runs, none of them a Boolean
microbenchmark venue in §58's sense (they span four different output types and
four different input structures):

| id | task | source | held-out episodes |
|---|---|---|---|
| `bool` | `maj(a,b,c) xor maj(d,e,f)` over six BOOL inputs | §44/§46's task, `research/earned-abstraction/later.py` | odd rows of the complete 64-row truth table |
| `arith` | two int16 operands and a one-hot goal selecting add / sub / max | `generators/arithmetic`, unmodified | test-split seeds 1000–1031 |
| `rel` | membership in the two-step reachability relation of a 4-entity directed graph | `generators/relations`, unmodified | test-split seeds 1000–1063 |
| `lang` | bracket grammaticality | `generators/language`, §19/§45's family via `research/scaffold-diagnosis/splits.py`, **post-audit stream pinned** (`hardening='context_free_language'`) | unseen lengths 14 and 16 |

Every base scaffold is built here from `tcn.graph` and `legal_candidates`; the
`bool` scaffold restricts MAJ3's ordered triples to the 20 combinations (MAJ3 is
symmetric), which is a disclosed scaffold-design choice and is what keeps the
base space exhaustible. Every edit re-derives its pool from the registry and may
re-introduce the orderings.

**A domain that cannot supply a corpus under §2.3 is dropped and the reason
disclosed in `RESULTS.md`,** with its partial numbers kept. Three domains are the
minimum for the leave-one-domain-out protocol of §4.3; if fewer than three
survive, the cross-domain claim is not made at all and the track reports the
within-domain result only.

### 2.2 The defect generator

A **defect** is one of four typed narrowings of a base scaffold, applied at one
site, enumerated deterministically over the base's own sites and operators:

| defect | what it removes |
|---|---|
| `drop_operator(site, op)` | every candidate at `site` whose operator is `op` |
| `drop_source(site, port)` | every candidate at `site` that reads `port` |
| `keep_prefix(site, k)` | all but the first `k` candidates at `site` |
| `delete_node(site)` | the node, re-pointing its consumers at one of its own sources |

`delete_node` is included precisely because its repair is **not** the inverse of
the defect: re-adding the node is an `ADD_NODE`/`ADD_PATH` edit over a pool the
defect did not name. The fraction of first repairs that *are* the syntactic
inverse of the defect is a mandatory corpus-validity disclosure (V1).

### 2.3 Admission

A defective scaffold enters the corpus iff, with `tolerance = 1e-6`:

1. it has **0 members conforming on the training episodes**, `exhausted`,
   certificate `complete` — this is what "no exact solution" means, and it is a
   proof, not a timeout; and
2. its typed edit space (§3) contains **at least one repair** under §5 — else no
   arm could succeed and the case measures nothing. Cases failing (2) are counted
   and reported, never silently dropped.

Target: **12–24 admitted cases per domain**, taken in the defect generator's own
deterministic order, capped at 24 per domain so no domain dominates the mean.

---

## 3. The typed structural edit space

### 3.1 Families

`edits.py`, generic over `tcn.graph.Program`; every produced program is put
through `Program.validate(registry)` and discarded if it does not type-check, so
no edit can widen the type system.

| family | what it does |
|---|---|
| `SUBST(site, op)` | replace the site's candidate tuple with every legal wiring of `op` over the site's own source pool |
| `WIDEN(site, op)` | union those candidates into the site's existing tuple |
| `REWIRE(site, port)` | union in every wiring of the site's **existing** operator names that reads `port` |
| `ADD_NODE(site, op, t)` | insert a new node of type `t` computing `op` over the pool below `site`, shift the site and everything deeper down one depth, and widen the site to admit the new node |
| `ADD_PATH(op)` | insert a new output-adjacent node combining the current output with another value in scope, and re-point the program's output at it |

This is materially wider than §50's sweep, which is `SUBST` at two named holes
only. A site's source pool is derived from the program (inputs, constants and
nodes of strictly smaller depth), never hand-listed. Operator names are the union
of `tcn.operators`'s own name sets plus the extras `Registry.resolve` accepts
(`identity, not, mux`, the five reductions), never a curated per-domain list.
**Nothing in `edits.py` knows what any scaffold's defect is.**

### 3.2 Order

Edits are enumerated in a fixed lexicographic order — family, then site, then
operator/port/type, all sorted. This order is a property of the enumerator and is
identical for every arm; the arms differ only in how they *re-order* it.

### 3.3 Declared bounds

`MAX_NEW = 48` candidates may be added at an existing site by one edit and
`MAX_NEW_NODE = 12` candidates may be carried by an inserted node, truncating in
`legal_candidates` order. `MAX_SPACE = 400,000` bounds the selection space of an
edited scaffold that will be decided. These bounds are deterministic, applied
identically to every arm, and fixed now. Reported: how many edits were truncated
and how many exceeded `MAX_SPACE`. An over-`MAX_SPACE` edit is **undecided**, and
every headline is stated **both** with undecided edits counted as non-repairs and
with them counted as repairs (§6.3) — the two together bound the truth.

### 3.4 Size, as probed

`bool` 160 edits, `arith` 542, `rel` 118 (schedulability probe, §0). The corpus
is therefore ~10²–10³ edits per case.

---

## 4. The arms

Every arm searches **the same edit space**; they differ only in the order. A
library that only re-orders cannot make a repair unreachable, and that is what
makes the distractor of C4 able to succeed by construction (V3).

### 4.1 `N` — no library
Uniform random over the whole typed edit space. Exact expected cost
`(S + 1)/(K + 1)` for `S` edits of which `K` are repairs (§6.1).

### 4.2 `N′` — the structure alone
The inherited **edit-family class**: the subset of families that repaired
anything in the source domains, uniform within it, then the rest uniformly. No
learned content beyond "which of the five families ever works".

### 4.3 `N″` — schema + learned edit prior (the key arm)
A per-edit score from features that are available in any domain, fitted on the
**source domains only** under leave-one-domain-out: for a held-out domain `d`,
the estimator sees `(edit, repaired?)` rows from every domain except `d`, and is
evaluated only on `d`. The held-out domain contributes **nothing** to the fit —
not its rows, not its feature statistics, not its corpus size.

Features, all derived from the program and the registry, none of them naming a
domain or an answer:

1. edit family (one-hot over the five)
2. `log` of the number of candidates the edit adds, and of the resulting space
3. the operator's algebraic class, read off `tcn.operators`' name sets:
   arithmetic / selective (`min`, `max`, the reductions) / logical / comparison /
   structural / conversion / identity
4. whether the operator already occurs anywhere in the scaffold (novelty)
5. the site's depth, normalised by the program's depth, and whether it is the
   output node
6. whether the site's value is **constant on the training batch** (§47's probe —
   §50 showed it does not generalise; it is included so that this track's own
   measurement of it is on the record, not because it is expected to help)
7. the arity of the added candidates

Estimator: additive per-feature empirical log-odds of repair (naive Bayes with
Laplace smoothing), fitted on source-domain rows only. Auditable, and small
enough that its whole table is committed to `out/`. **No scalar abstraction
score is computed** (§57, §64): the estimator ranks edits for one slot, it does
not rank library entries.

### 4.4 `H1` and `H2` — the hand-written priors, mandatory
Fixed here, before any result, and not tuned afterwards.

* **`H1` (primary), "smallest and newest first"**: order by (candidates added,
  ascending), then (operator not already in the scaffold, first), then the
  enumerator's own order. One line of code; a human would write it without
  looking at any data.
* **`H2`, "output-adjacent first"**: order by distance from the output node,
  ascending, then the enumerator's own order. This is the localisation choice
  §50 deleted, restored as a baseline.

If `E(H1) ≤ E(N″)` the inherited rows add nothing measurable beyond a prior a
human would write, and that is the finding (§65's lesson, made a criterion).

### 4.5 `ORACLE` — the ceiling
Rank 1 by construction. Reported so every arm's distance from perfect ordering
is visible; not a criterion.

### 4.6 `D1`, `D2` — the distractors
* **`D1`** — the identical estimator on **permuted repair labels** from the same
  source rows (permutation seed fixed at 0, 20 permutations, mean and spread
  reported).
* **`D2`** — the learned estimator, reversed.

Both order the **same** edit space as `N″`, so each contains every repair `N″`
has. **V3 verifies this by counting repairs in each distractor's space before C4
is read**; §65's distractor could not have succeeded and this one is checked.

---

## 5. What counts as a repair — held-out, not training

Edit `e` on case `c` is a **repair** iff the edited scaffold has at least one
member conforming, at `tolerance = 1e-6`, on **every training *and* held-out
episode of the domain**. Existence is proved by a witness (the first conforming
member); non-existence is proved by exhaustion, `exhausted = true`, certificate
`complete`. The decision is made by `tcn.search.enumerate_prefix` — core
machinery, not a track simulator — so there is no simulator to validate; a
sample of decisions is nevertheless cross-checked against `tcn.search.enumerate_fit`
and against `Program.execute` (V2).

**Repair labels are ground truth and are computed in a separate file
(`truth.py`) that no arm imports.** Priors, orderings and features are computed
from the base scaffold, the failed scaffold's certificate and the *training*
episodes only, exactly as §50 did.

**Secondary metric M2 — the deployed protocol.** Enumerate in the arm's order and
stop at the first edit whose scaffold has a **training** conformer; report
whether that edit is a repair. This is the §50 protocol and the §65 trap in one
number: it measures what a system that cannot see held-out data would actually
accept. M1 (§6.1) is the pre-registered criterion; M2 is reported beside it for
every arm and is **not** substituted for it.

**Secondary arm S1 — re-scoring §50's own corpus on held-out conformance.**
§50's `allholes` arm reports 21 of 24 repairs judged on training accuracy 1.000.
Its cases are re-decided here on held-out conformance with §50's own validated
decider (`research/scaffold-diagnosis/langfam.py`, `boolfam.py`) and its own
`answers.py`. Pre-registered prediction: the count **falls**. If it does not, that
is recorded too. A disagreement with §50's recorded number is reported as a
disagreement, not routed around.

---

## 6. Costs, certificates and how every number is obtained

### 6.1 The currency
`E(A, c)` = expected number of edits enumerated before the first repair, for arm
`A` on case `c`. An arm supplies a **tiered** order: a sequence of tiers, uniform
within each. With `s_i` edits and `k_i` repairs in tier `i`,

    E = Σ_{j < i*} s_j  +  (s_{i*} + 1) / (k_{i*} + 1),   i* = first tier with k_i > 0

which is the flagship's exact formula — the expected number of draws to the
first of `k` good items among `s`, drawn uniformly without replacement. **Every count is exact**: the edit space is
enumerated in full and every edit is decided, so no arm's cost is sampled and no
0/N bound is quoted (§65's failure to separate N″ from N came from exactly that).
The corpus figure is the mean of `E(A, c)` over admitted cases, reported with its
per-case table.

### 6.2 Certificates
`complete` (exhausted, whole conforming set reported), `unique`, `none`, exactly
`tcn.search.certificate_of`. Every admission decision (§2.3) and every
non-existence claim carries one.

### 6.3 Bounding both directions
Undecided edits (over `MAX_SPACE`) are counted as non-repairs in the headline and
as repairs in a parallel column. A criterion that flips between the two columns
is reported as **inconclusive**, never as a pass.

---

## 7. Baselines beside every conformance number

For every domain: the majority-constant and uniform-random accuracy on the
held-out episodes, and the accuracy of the *base* (undefective) scaffold's own
conforming member. A repaired scaffold that merely reaches the majority constant
is not a repair — conformance at `1e-6` on every episode already excludes this,
and the baselines are reported so the exclusion is visible.

---

## 8. Falsification — declared now, honoured whatever happens

* **F1** `E(N″) ≈ E(N)` (within 2×): inherited edit knowledge buys nothing. **The
  expected outcome, and a first-class result.**
* **F2** `E(H1) ≤ E(N″)`: the hand-written prior reproduces or beats the learned
  one. **Also expected**, and it is C3 failing.
* **F3** `E(N′) ≈ E(N)`: the edit-family class is organisational reuse, not
  learned intelligence — §64's reading, applied to edits.
* **F4** `D1` satisfies C1: the effect is the shape of the tiering, not the
  learned content.
* **F5** the estimator is flat (no feature separates repairs from non-repairs in
  the source domains). Reported; no hand prior is substituted for it.
* **F6** more than 60% of first repairs are the syntactic inverse of the defect
  (V1): the corpus is easy and **every** arm's number must be read that way. The
  criterion is then reported as resting on an easy corpus.
* **F7** fewer than three domains supply a corpus: the cross-domain claim is not
  made.
* **F8** §50's 21/24 does **not** fall under held-out conformance (S1).

**Predictions, not criteria** (so a null cannot be re-narrated afterwards): H1
lands within 2× of N″; N′ lands within 2× of N; feature 6 (§47's constant-node
probe) carries no weight; `SUBST`/`WIDEN` dominate the repairs and `ADD_NODE`
repairs only the `delete_node` cases.

---

## 9. Validity checks, run and reported before any criterion is read

* **V1** fraction of first repairs that are the syntactic inverse of the defect,
  per domain and overall (F6).
* **V2** the decider: for a sample of 40 (case, edit) pairs per domain,
  `enumerate_prefix` is checked against `enumerate_fit` for the same conforming
  set, and each reported witness is re-executed through `Program.execute` on
  every episode. Mismatches reported; a non-zero count blocks the headline.
* **V3** the distractor can succeed: repairs counted in `D1`'s and `D2`'s edit
  space (identical to `N″`'s by construction, and verified equal).
* **V4** no leakage: the held-out domain's rows, features and corpus size enter
  no estimator; asserted by a test that re-fits with the held-out domain's rows
  deleted and compares the estimator tables byte for byte.
* **V5** `tcn/` and `generators/` diffs against `main` are empty.

---

## 10. Evidence standard and resources

Every figure in `RESULTS.md` is rendered from `out/` by `report.py` inside
`<!-- BEGIN:x -->` / `<!-- END:x -->` blocks. `verify.py` has two layers: it
re-renders every block and compares, and it re-derives every headline claim from
raw JSON **without importing the run or report scripts**; it writes
`out/verify.json` (`{"pass": N, "fail": M, "claims": [...]}`) and
`out/headline.json`, and exits non-zero on any FAIL. Single-configuration
evidence is labelled explicitly wherever it occurs.

Heavy jobs run as
`systemd-run --user --scope -q -p MemoryMax=20G -p CPUQuota=400% env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 <cmd>`,
at most **4 workers**, with `MemAvailable` checked against a **25 GB** floor
before each phase and logged to `out/memory_floor.log`. The other project's GPU
processes are never touched.

---

# Amendments, written before any arm ran

The sections above are **unaltered**: what they originally said still stands in
the text, and each amendment below quotes the clause it replaces. All three were
written before the first arm, and the corpus runs they govern were restarted
under them. Nothing here is back-dated — where the original wording was wrong it
is recorded as wrong, not quietly replaced by the wording I now prefer.

## A8 — the training-episode budget is fixed by a stated rule

§2.1 named a seed range per domain and said nothing about how the size of that
range was chosen. That is a hole: the number of admitted cases depends strongly
on it — `rel` admits one defect at 96 training episodes and nineteen at 384 — so
a budget picked after seeing those counts would be a corpus parameter tuned on
the quantity the corpus exists to measure, and no reader could tell it apart
from tuning.

**The rule, adopted now and applied identically to every domain:** the training
budget is the **smallest budget on a doubling ladder whose admissible defect
*set* — the set itself, not its size — is identical at `B`, `2B` and `4B`.**

The justification is that stability across two consecutive doublings is a
property of the domain rather than of the budget: it says the admission decision
has stopped depending on how many episodes were drawn, so a defect still counted
`solvable on train` is solvable because a conforming program genuinely exists,
not because too few episodes were drawn to exclude one. Where no budget on the
ladder is stable, the largest swept budget is used and **the instability is
reported as a limitation of that domain**, not hidden.

`episode_sweep.py` runs the ladder; `out/episode_sweep.json` records **every**
budget tried, including those the rule rejects, and the full sweep appears in
`RESULTS.md`. `bool`'s ladder is bounded by its data: its episodes are rows of a
complete 64-row truth table, so past half the table the held-out set is smaller
than the training set, and at the whole table there is no held-out set at all.

## A9 — the 12–24 case floor was unreachable, and that is a pre-registration error

§2.3 set a target of "**12–24 admitted cases per domain**". I set that number
without checking that §2.2's defect generator could produce it, and for `bool` it
cannot. Exhaustive enumeration of every defect (`admissible.py`,
`out/admissible_bool.json`) gives 46 defects, of which 3 are invalid and 32 leave
the training episodes still solvable, so **11 are admissible at most; 3 of those
contain no repair, leaving a ceiling of 8**. The ceiling does not move with the
episode budget — it is 11 at 32, 42 and 48 training rows — so it is a property of
the scaffold and the defect generator, not of the resources spent on it.

**The floor is amended to: as many cases as the domain admits, up to 24, with the
exhaustive admissible-defect enumeration reported per domain.** A domain that
cannot reach twelve is reported at its ceiling together with the enumeration that
proves the ceiling. This is a weakening of the pre-registration and is recorded
as one: the original number was not reachable and should not have been written
without checking that it was.

## A10 — F7 counts domains where it should count cases

**F7** as written reads: "fewer than three domains supply a corpus: the
cross-domain claim is not made." That counts *domains*, not *cases*, so it would
have passed unmoved with `rel` contributing a single case — a leave-one-domain-out
fold resting on one case, which cannot carry a cross-domain claim. The weakness
is in my wording and is recorded here rather than repaired by pretending a better
condition had been pre-registered.

**F7 stands as written**, and whether it fires is reported against its original
text. Beside it, and explicitly **not** as a pre-registered criterion, this track
states the stronger condition it would use instead: *no cross-domain claim unless
every domain contributes at least three admitted cases, with the per-domain case
counts printed beside every cross-domain figure.* Any reading of the results
against that stronger condition is labelled post-hoc wherever it appears.

## A11 — the candidate-truncation bound is removed, because it was deleting repairs

§3.3 declared `MAX_NEW = 48` candidates addable at a site and `MAX_NEW_NODE = 12`
for an inserted node, "truncating in `legal_candidates` order", and argued the
bound was safe because it is "deterministic, applied identically to every arm".
**That argument was wrong, and the bound was not safe.** Applying the same
truncation to every arm keeps the *comparison* fair while changing *which program
family the edit denotes* — the edit stops meaning "widen this site with `xor`"
and starts meaning "widen it with the first forty-eight `xor` wirings", which is
not a typed structural edit anyone would propose.

It bit immediately and hard. On `bool`, `WIDEN(y, xor)` needs the wiring
`(n1, n2)`; `legal_candidates` enumerates the eight-port pool in order and emits
that wiring at **index 55 of 64**, past the bound. The whole `bool` corpus ran to
completion under the bound and admitted **2** cases: 9 of its 11 admissible
defects were recorded "no repair in the edit space" **for that reason alone**.
Re-deciding them without the bound turns 8 of those 9 into repairs
(`out/notrunc_bool.json`).

**The bound is removed.** The work is bounded by `MAX_SPACE` alone: an edit whose
selection space exceeds it is recorded **undecided**, and §6.3 already requires
every headline to be stated both with undecided edits counted as non-repairs and
as repairs, so the cost of the remaining bound is visible and two-sided rather
than silent. Measured price on `bool`: over-cap edits rise from 0–17 per case to
10–43 of 160, and median edited space rises from about 10³–10⁴ to roughly double.
Removing the bound invalidates every corpus built under it; all three domains are
re-run, and the superseded `bool` and `rel` corpora are kept in `out/` rather than
deleted.

The general lesson, which is the part worth carrying: **a bound that is applied
uniformly across arms is not thereby harmless.** Uniformity protects the
comparison between arms; it does nothing to protect the meaning of the object
being compared. This one was declared in the pre-registration, flagged in the
code as a risk, and still went unnoticed until a domain returned an implausible
number of "no repair" verdicts.

## A12 — the headline is per domain, with a stated pooling rule

Written before any arm was costed, and before the site structure could be known
to favour any arm.

`bool`'s admitted cases are **all defects at one node**, its output `y`; `rel`'s
are spread over four sites. A domain whose defects sit at a single site gives an
ordering prior almost nothing to discriminate on, because every candidate edit
attaches to the same place: what separates the arms there is the choice of
operator, not the choice of where to act. Averaging such a domain with a
multi-site one produces a number that is neither domain's behaviour, weighted by
corpus sizes that are artifacts of each domain's defect ceiling rather than of
its importance. §65's lesson was not that a number came out wrong; it was that a
true number answered a question nobody had asked.

**1. Every criterion is resolved per domain.** C1, C2, C3a, C3b and C4 are
computed separately for `bool`, `rel` and `arith`, and all three verdicts appear
in the headline table. A pooled figure may appear **alongside** them and never
instead of them.

**2. The pooling rule, fixed now: equal weight per domain (macro-average).** Any
pooled cost is the unweighted mean of the per-domain mean costs. The question is
cross-domain transfer, and per-case weighting would let the domain with the
largest ceiling decide the answer — `rel` admits nineteen cases and `bool` at
most eight, a ratio reflecting how many defects each scaffold tolerates, not how
much either domain should count. The per-case (micro) average is reported beside
it, so the difference between the two is visible rather than buried in a choice
of denominator.

**3. The site structure is disclosed as a rendered block**, before any verdict:
defects per site, distinct sites per domain, and repairs per case. A reader
should be able to see that `bool` is single-site before reading `bool`'s verdict.

**4. What a split verdict means, stated before it is known.** If the criteria
pass on some domains and fail on others, the pre-registered verdict is **NOT
MET** — the criterion is about transfer to a genuinely new domain, and a prior
that helps only where it happens to fit has not transferred. The result is then
reported as *which structure the inherited knowledge fits*, not as a qualified
success. In particular, a pass confined to multi-site domains would indicate the
prior is discriminating on **where to edit** rather than on **what edit to make**;
that reading is testable by knocking out the site-derived features, and any such
test is labelled exploratory and post-hoc. The converse — a pass confined to the
single-site domain — would indicate the opposite, that the prior carries operator
knowledge and no localisation. Neither is the pre-registered success condition,
and neither will be presented as one.


## A13 — three splits, not two: `admission` is selection data, `final` is not

**Raised by the owner, reviewing the §65 handoff.** That handoff says
"pre-register on held-out conformance, not training conformance". His objection:
a held-out set that is *used to make admission decisions* has become selection
data. Counterexamples drawn from it may legitimately refine a schema — that is
what CEGIS is — but once used they are training information. Keep calling the
same set "held-out" afterwards and the project ends up optimising against
something it believes is blind. §65's own lesson was that the metric, not the
number, was the defect; this is the same failure one level up.

He is right, and it applies directly to this track as §5 originally wrote it.
§5 defined a repair as a member conforming on "every training *and* held-out
episode", and §2.3 admitted a defect on the training episodes. So the set that
decided what counts as a repair was also the only set left to report
generalization on. There was nothing blind.

**The structure, replacing §2.3's admission rule and §5's repair definition:**

    train  →  admission / CEGIS validation  →  untouched final test

* **`train`** — what a repaired scaffold must fit to be a candidate at all.
* **`admission`** — decides both that a defect is admissible (no member conforms
  on `train ∪ admission`, exhausted) and that an edit is a repair (at least one
  member does). Admission and repair are deliberately complements of one
  predicate on one split, so no decision straddles two criteria.
* **`final`** — read by nothing until the arms have been ordered and scored.
  It is used for exactly one thing: whether the edit an arm actually selected
  generalizes. Every number derived from it is labelled as coming from it.

**What this invalidates, stated plainly.** The corpora built under the two-way
split are **superseded, not re-labelled**, and are kept under
`out/superseded-twoway/`. They cannot be re-scored into the new structure: the
new repair predicate (`train ∪ admission`) is *weaker* than the old one
(`train ∪ all evaluation episodes`), so edits recorded as non-repairs may be
repairs under it, and those are precisely the cases where `final` would bite.
Re-using the stored labels would silently keep the old, stricter criterion while
claiming the new one. Every per-edit decision is therefore re-run. The
`episode_sweep` ladder and the admissible-defect ceilings are re-run too, because
A8's rule and `admissible.py` both evaluate the admission predicate, which has
changed.

**Enforcement, not intention.** `run_domain.py` fingerprints `final` and then
**deletes it from the domain dictionary** before the corpus loop begins, so a
later read raises rather than quietly succeeding; the corpus records
`final_untouched`, `n_final` and the fingerprint. Scoring against `final` happens
in a separate step that records the same fingerprint, and `verify.py` fails if
the fingerprints disagree, if a corpus artifact lacks `final_untouched`, or if any
arm's ordering was computed from anything but `train`.
