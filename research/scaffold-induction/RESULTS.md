# Scaffold induction as a cross-domain outer loop — results

`PREREGISTRATION.md` was committed before the first arm. `tcn/` and
`generators/` are untouched. Every figure below sits inside a
`<!-- BEGIN:x -->` block rendered from `out/` by `report.py`; `verify.py`
re-renders each block, re-derives every headline from raw JSON without importing
the run or report scripts, and writes `out/verify.json` and `out/headline.json`.

**[Single-configuration evidence.]** One seed set per domain, one base scaffold
per domain, one defect generator, one edit enumerator, one estimator family, one
train/held-out split per domain. The domain count is small enough that the
cross-domain result is anecdote-strength and is labelled so wherever it is
stated — the same disclosure §50 made about its own task count.

---

## Amendments to the pre-registration

Every change to `PREREGISTRATION.md` after it was committed, in order, with what
it replaces and **the commit that made it**, so a reader can check for himself
that each predates the arms rather than taking this paragraph's word for it.
The pre-registration is `11d86b4`, the first commit on the track; the amendments
land in `a73107a` and `0705225`; the first corpus artifact is committed after
both. **All of them were made during corpus construction, before any arm
ordering or arm cost was computed**, and none moves a number that had already
been measured — the arms had not run. They are listed anyway, because a
pre-registration that is quietly re-read is not one.

| amendment | commit | what it touches |
|---|---|---|
| A1 `rel` becomes a three-step scaffold over three entities | `a73107a` | `domains.py` |
| A2 `rel` and `bool` episode counts | `a73107a` | `domains.py` |
| A3 `keep_prefix` k values and defect sites | `a73107a` | `run_domain.py` |
| A4 `lang` dropped | `a73107a` | `domains.py`, `analyse.py` (`DOMAINS`) |
| A5 the training decision is skipped behind a witness | `0705225` | `run_domain.py` |
| A6 §3.4's probe sizes moved with A1 | `a73107a` | prose only |
| A7 the expectation formula corrected | `0705225` | `PREREGISTRATION.md`, `analyse.py`, `verify.py` |
| A8 the training budget is fixed by a stated rule | `a635469`, `PREREGISTRATION.md` | `episode_sweep.py`, `domains.py` |
| A9 the 12–24 case floor was unreachable | `PREREGISTRATION.md` | `admissible.py` |
| A10 F7 counts domains where it should count cases | `PREREGISTRATION.md` | wording only |
| A11 the candidate-truncation bound removed | `PREREGISTRATION.md` | `edits.py`, all three corpora re-run |
| R1 the memory floor scales to the measured peak | `8fd4673` | `kit.py`, `run_all.sh` |
| R2 the four-worker cap is enforced in code | `kit.py` | every launch path |

A8 to A11 are written out in full at the end of `PREREGISTRATION.md`, each
quoting the clause it replaces, and all four were written before the first arm.
Three of them are corrections to my own pre-registration rather than adaptations
to results: A9 records a floor I set without checking it was reachable, A10
records a falsification condition I wrote counting the wrong thing, and A11
records a declared bound that was deleting the repairs it was supposed to be
neutral about.

**R1 and R2 are resource-policy amendments, not pre-registration ones**, and are
written up in the Resources section rather than here. They are listed in the same
table because they change what the track is allowed to run and when, and a reader
tracing why a phase started or did not should find them in one place.

---

### The amendments in full

**A1 — `rel` is a three-step scaffold over three entities, replacing §2.1's
"two-step reachability relation of a 4-entity directed graph".** The generator's
target is membership in the full transitive closure. A two-step scaffold cannot
express it: the base scaffold itself was decided as having **no** conforming
member over all episodes, exhausted, certificate `complete`, so no edit of any
defect of it could have been a repair and the domain would have contributed
nothing. Three edge steps is what the closure needs at three entities (a simple
path between distinct vertices is at most two edges, a cycle at most three). The
base's certificate on all episodes is recorded in `out/cases_rel.json` and
checked by `verify.py`.

**A2 — `rel` uses more episodes than the pre-registered seed ranges, and `bool`
trains on every even row rather than a subset.** With the smaller splits, almost
every defect still had a member conforming on the training episodes and the
domain admitted one case. The admission rule (§2.3) is unchanged; only the
episode counts moved. Both counts are rendered in the corpus block.

**A3 — `keep_prefix`'s `k` values are {1, 2, 3, 4, 6, 8}, and defects are
enumerated over every node of the base scaffold.** §2.2 fixed neither. Stated
here so the defect enumeration is reproducible from the document.

**A4 — the `lang` domain is dropped**, under §2.1's own drop clause. The reason
is in the corpus section below.

**A5 — the training-only decision is skipped when the all-episode decision
returns a witness.** A member conforming on every episode conforms on the
training ones, so the second decision would be redundant; the field records that
it was implied by the witness rather than measured. No metric changes.

**A6 — §3.4's edit-space sizes were probe figures and `rel`'s has moved** with
A1. The measured sizes are in the corpus block; nothing in §3.4 is a criterion.

**A7 — §6.1's expectation formula was written down wrong and is corrected.** The
document said `(s − k + 1)/(k + 1)` for a uniform tier; the expected number of
draws to the first of `k` good items among `s`, without replacement, is
`(s + 1)/(k + 1)` — the formula §6.1 also named as "the flagship's". The typo
was in the pre-registration and in the first draft of both `analyse.py` and
`verify.py`, which is exactly the failure mode a verifier that shares a
mis-derivation cannot catch; it was found by checking the degenerate case
(`s = k = 1` must cost one draw, not a half) before any arm was costed. No
measured number changes, because none had been produced. The cost metric and
the independent check it now carries are set out in the next section.

---

## The training-episode budget, fixed by rule

The number of admitted cases depends strongly on how many training episodes are
drawn — `rel` admits one defect at 96 and nineteen at 384 — so choosing that
number after seeing those counts would be tuning a corpus parameter on the
quantity the corpus exists to measure. A8 fixes it by rule instead: **the
smallest budget on a doubling ladder whose admissible defect *set* is identical
at `B`, `2B` and `4B`**. Every budget tried is below, including the ones the rule
rejects.

<!-- BEGIN:sweep -->
<!-- END:sweep -->

`bool` is the interesting row and the rule earns its keep there: its ladder is
bounded by its own data — episodes are rows of a complete 64-row truth table, so
past half the table the held-out set is smaller than the training set and at the
whole table there is none — and **no budget on that ladder is stable**. The rule
therefore does not pick a budget for `bool`; it falls back to the largest swept
value and hands back a disclosure. Had I chosen `bool`'s budget by yield I would
have written down 32 and said nothing.

---

## What each domain can admit at all

The pre-registration asked for 12–24 admitted cases per domain without checking
that the defect generator could produce that many. It cannot, and A9 records that
as a pre-registration error. These are exhaustive enumerations, not samples: every
defect the generator produces is applied and put through the admission rule.

<!-- BEGIN:ceilings -->
<!-- END:ceilings -->

---

## Why one domain costs fourteen times another per case

`arith` costs more than an order of magnitude more per case than `bool`, on the same
machinery, and the reason is not the size of its space — it is **where its
supervision sits**. `arith` carries a single probe, on the output node.
`enumerate_prefix` prunes a prefix only when a *probed* node's value already
misses its target, so with the only probe at the end of the program there is
nothing to prune against: every selection in the space is walked to the last node
before it can be rejected. `bool`'s scaffold is three nodes deep, so the walk is
short whatever happens; `arith`'s is nine, and every one of them is paid for
every candidate program.

This is §41 and §45's "dense supervision is what makes search tractable" showing
up as a wall-clock bill rather than as an argument, and it is worth recording
because it inverts the usual intuition about cost: the expensive domain here is
not the one with the biggest space — `bool`'s edited scaffolds reach a larger
maximum than `arith`'s — but the one whose supervision cannot reject a partial
program. Anyone budgeting an enumeration in this substrate should count
probed nodes before counting candidates.

<!-- BEGIN:cost -->
<!-- END:cost -->

## The cost metric, and how it is checked without re-using its own derivation

An arm supplies a **tiered** ordering of a case's edits: a sequence of tiers,
uniform within each. Enumeration draws from the first tier until it is exhausted,
then the second, and so on. For a tier of `s` edits containing `k` repairs, drawn
uniformly without replacement, the expected number of draws to the first repair
is

    (s + 1) / (k + 1)

and the cost of the whole ordering is the sizes of every repair-free tier ahead
of it, plus that term for the first tier that contains a repair:

    E = Σ_{j < i*} s_j  +  (s_{i*} + 1) / (k_{i*} + 1),   i* = the first tier with k_i > 0

**Worked example**, small enough to check by hand. Three edits in one tier, one
of them a repair. The repair is equally likely to sit in each position, so the
number of draws to reach it is one, two or three with probability a third each,
and the mean is two — which is `(3 + 1) / (1 + 1)`. The formula the
pre-registration originally carried, `(s − k + 1) / (k + 1)`, gives one and a
half for the same tier, and gives a half for a single edit that is certainly a
repair, which is not a possible number of draws.

**Why the verifier's re-derivation is not enough on its own, and what replaces
it.** `verify.py` re-implements the expectation rather than importing it, but
re-implementing a formula I had already written down wrong is not an independent
check — it is the same derivation typed twice, and it is precisely what let the
error through the first time. So the verifier now checks the closed form against
a **different route to the same quantity**: it draws uniformly random
permutations of a tier and averages the position of the first repair, and
requires the empirical mean to agree with `(s + 1) / (k + 1)`. Simulation shares
no algebra with the closed form, so a mistake in the algebra shows up as a
disagreement. The degenerate cases are asserted outright beside it. Both are
FAIL-able claims in `out/verify.json`, not comments.

**And a second guard that was not guarding, found while writing the above.**
`verify.py` also refuses any number in RESULTS prose that is neither a declared
structural constant nor present in a rendered block. Two defects made that check
close to vacuous. Its block-stripping pattern did not tie the closing marker to
the opening one and required a newline inside the block, so on an unfilled block
it ran forward to a later marker and deleted the prose in between; then its
code-span pattern was allowed to cross newlines, so a single unpaired backtick
paired across paragraphs. Together they were discarding about ninety-six percent
of the document before the check ever ran — the check passed because there was
almost nothing left to check. Both patterns are fixed, the block marker is now
tied by a backreference, and a **guard on the guard** now fails if the strips
remove most of the document. A third defect in the same check: it compared each
prose number against the blocks by substring, so a stray digit passed whenever
some block happened to contain it inside a longer number. It now compares
tokens, and that change immediately surfaced two real strays this document had
been carrying.

---

## The pre-registered verdict

<!-- BEGIN:criteria -->
<!-- END:criteria -->

<!-- BEGIN:costs -->
<!-- END:costs -->

<!-- BEGIN:ratios -->
<!-- END:ratios -->

---

## The corpus

Defects are typed narrowings of a base scaffold; a defective scaffold is
admitted only when it has **no** member conforming on the training episodes,
exhausted, certificate `complete` — a proof, not a timeout — and its typed edit
space contains at least one repair.

<!-- BEGIN:corpus -->
<!-- END:corpus -->

**The `lang` domain was dropped, and why.** §19/§45's bracket-grammaticality
family was pre-registered as a fourth domain and does not survive contact with
the generic enumerator. The post-audit stream emits only long strings, so the
scaffold needs its full width; at that width the program carries a large
constant pool, and `legal_candidates` exceeds its enumeration budget on almost
every operator, so the typed edit space collapses to a handful of edits and
takes longer to *enumerate* than a whole corpus takes to decide in the other
domains. §2.1's drop clause is exercised: the domain is dropped, disclosed here,
and its partial numbers are in `out/resource_gate.log` and the probe scripts. The
cross-domain claim therefore rests on the remaining domains, which is the
pre-registered minimum and no more.

<!-- BEGIN:percase -->
<!-- END:percase -->

---

## What the inherited prior learned

<!-- BEGIN:estimator -->
<!-- END:estimator -->

<!-- BEGIN:families -->
<!-- END:families -->

---

## The deployed protocol (M2), and the §65 trap

M1 above is the pre-registered criterion: expected edits to the first edit whose
scaffold conforms on the **held-out** episodes. M2 is what a system that cannot
see held-out data would actually do — enumerate in the arm's order and stop at
the first edit with a **training** conformer — and it is reported beside M1, not
substituted for it.

<!-- BEGIN:deployed -->
<!-- END:deployed -->

---

## Validity checks, read before the criteria

<!-- BEGIN:validity -->
<!-- END:validity -->

**V2 — the decider.** Every (case, edit) pair is decided by
`tcn.search.enumerate_prefix`, core machinery rather than a track simulator, so
there is nothing to validate against `Program.execute` in the way §45 and §47
had to validate theirs. It is checked anyway, two ways: a deterministic sample
of edits per case is re-decided by the flat `enumerate_fit` walk over the same
space, and **every** recorded witness is re-executed through `Program.execute` on
**every** episode of its domain. `validate_decider.py` also asserts that the edit
enumeration is reproducible — that re-applying the defect and re-enumerating
yields exactly the committed key set — because an ordering claim over an
irreproducible list would mean nothing.

<!-- BEGIN:decider -->
<!-- END:decider -->

---

## S1 — re-scoring §50's own corpus on held-out conformance

**The pre-registered prediction F8 was that §50's count would fall. It does
not.** Every case that had a repair keeps one under the stricter metric, and the
variant a training-only protocol stops at conforms on the held-out episodes too.
§50's baseline is stronger than this track predicted, not weaker. The prediction
is kept and marked wrong rather than removed.

**But the stricter metric is not inert on §50's own corpus, and the one place it
bites is the argument for using it.** One case, `C_max2_add`, has three variants
that conform on its training episodes and only two that conform on all of them:
`sub/add` fits the training set and fails held out. It is a single variant out of
the whole corpus, and the block below gives the exact counts — but it is direct
in-corpus evidence that "conforms on training" admits spurious repairs, the §65
defect turning up inside §50's own material rather than only in this track's.

**It is worse at the level of family members than at the level of variants.**
The `sub/add` scaffold does not merely contain a spurious member: **every** member
of it that conforms on the training episodes fails on the held-out ones, 24 to
zero. A protocol that accepted that scaffold on training conformance would have
had nothing to select from, at any tie-break. The two genuine repairs lose
members too — 527 conforming on training, 324 surviving the held-out check — so
even where the repair is real, training conformance overstates the conforming set
by roughly two thirds. That is the same shape §65 recorded and it is here in a
family this project already believed it understood.

**Two things that single case shows, and one it does not.** It shows the metric
choice is not academic. It also shows *why* the training-stop column reads
cleanly: `sub/add` sits in the second half of the variant enumeration, behind
`add/max`, so the stop rule never reaches it. The 22/22 is therefore a fact about
the enumeration order as much as about the family — reorder the variants and a
training-only protocol would accept a variant that does not generalize. What the
case does **not** show is that §50 was wrong: §50's own scoring compares a tie
set against a known repair set, which is a different and stricter question than
"does some variant conform", and its recorded 21 of 24 is not contradicted by
anything here.

This is worth holding beside the corpus above, where the same protocol on this
track's own domains produces far more edits that fit training and fail held out.
The difference is a property of the *family*, not of the method: §50's language
scaffold family is narrow enough that fitting the training episodes very nearly
pins the program, while the edit spaces here are wide enough to contain many fits
that do not.

<!-- BEGIN:s50 -->
<!-- END:s50 -->

---

## Resources

**A process-management near-miss, recorded rather than tidied away.** Trying to
keep the worker count at its cap, this track sent `SIGSTOP` to a PID selected by
`pgrep -f "run_domain.py rel"` — and the pattern matched the shell running the
command, which stopped itself. This is exactly the failure
`docs/CORRECTIONS.md` already records ("`pkill` patterns matched the running
shell; identify processes by PID and `/proc/<pid>/cwd`, never by command-line
substring"). It cost one hung command and no data. The rule is not new; it was
not followed. Nothing was paused after that, and the S1 arm was run after a
corpus job had finished rather than beside four.

Every job ran inside a capped scope (`MemoryMax=20G`, `CPUQuota=100%` per
worker, single-threaded BLAS, at most four workers), with `MemAvailable` checked
against the floor before each phase and logged to `out/resource_gate.log`. The
other project's GPU processes were never touched.

**This track's own footprint is small enough that the floor is about the host,
not about it.** Measured peak resident memory for the heaviest operation —
building a domain, enumerating a whole case's typed edits, and deciding them — is
**0.025 GB**, and the live corpus workers sat between 0.03 and 0.24 GB. An early guess that `enumerate_edits` was expensive because it
materialises every edited program at once was wrong: holding every edited program of a case at once costs a few megabytes,
as the table below records. Four shards would total roughly 0.15 GB.

**R1 — the flat floor is replaced by one that scales to the measured peak.**
This is a resource-policy amendment and is recorded as visibly as the
pre-registration ones. The original rule was a flat 25 GB of `MemAvailable`
before any phase. It was written for heavy phases, and it cannot tell a 50 MB job
from a 50 GB one; on a host loaded by *other* projects it blocks work that could
not possibly tip the host. The replacement:

> A phase may start when `MemAvailable` ≥ max(2 GB, 20 × the phase's measured
> peak RSS) **and** at least 1 GB of headroom remains once the phase is resident.
> A phase with no measured peak keeps the 25 GB floor.

For `arith` that is max(2 GB, 20 × 0.025 GB) = **2 GB**, two orders of magnitude
of slack over the measurement, and the 1 GB headroom test is what actually
protects a nearly-full host. Per-shard `MemoryMax` is set to **2 GB** rather than
20 GB — about eighty times the measured peak — so a genuine runaway is killed
alone instead of taking the host with it, and the `SIGTERM` handler records the
kill. The rule lives in `kit.check_floor`, which logs the requirement, the
arithmetic behind it and the headroom to `out/resource_gate.log` at every phase.
Unmeasured phases are unaffected: they keep the original floor, which is what
`kit.measured_peak` returning `None` selects.

**R2 — the four-worker cap is enforced in code, because enforcing it by
attention failed within the hour.** The cap had always been stated and was
restated in the exchange that produced R1; minutes later it was breached. Two
`arith` shards were launched while `bool` and two `rel` shards were still
running, putting **five** workers on the host. The combined footprint was the 0.23 GB the block below records,
so in substance it was never a host risk — but *"it looked fine"* is the argument
that preceded this project's crash, and a cap that depends on someone
remembering the other three jobs is not a cap. It is now `kit.check_workers`,
which counts this track's live workers and **fails closed** before any phase
starts. Processes are identified by PID and `/proc/<pid>/cwd`, never by a
command-line substring — `docs/CORRECTIONS.md` records `pkill` patterns matching
the running shell and killing the session twice, and this track reproduced the
same mistake once before writing the gate.

It was tested against the breach it exists to prevent: with the four workers
live it refuses the fifth, naming the PIDs, and logs `REFUSED`. Every launch path
goes through it — `run_domain.py`, `episode_sweep.py`, `admissible.py`,
`validate_decider.py`, `s50_rescore.py`, `analyse.py` — so no phase can start on
anyone's say-so.

**Both resource rules now fail closed in code, and one of them exists because the
human-enforced version failed within the hour.** That is the part worth carrying
out of this track: the memory floor and the worker cap were equally well known,
equally well intentioned and equally stated; the one that held was the one a
program checked. Their decisions share `out/resource_gate.log`, renamed from
`memory_floor.log` now that it records both kinds — admissions, refusals, the
requirement and the arithmetic behind it.

**The floor was breached under the old rule, and the launch was held.** During the corpus phase `MemAvailable` fell from 79.0 GB to 13.2 GB. None of it was this track:
its running jobs held a fraction of a gigabyte between them, against the other
projects' processes tabulated below. The `arith` phase was **not started**, the hold is recorded
in `out/resource_gate.log` with the cause, and no other project's process was
touched — including the one holding twenty gigabytes, which is the only one whose
removal would have helped. Running jobs were left alone deliberately: stopping
them would have freed 0.23 GB and lost an hour of work, and they checkpoint after
every defect, so even a host-level kill costs one defect rather than a corpus.

**A killed worker leaves a record rather than a gap.** `run_domain.py` installs a
`SIGTERM`/`SIGINT` handler that writes the partial corpus with
`stopped_by_signal` and `last_defect_started` set; `verify.py` fails if any
corpus an arm is costed on carries that marker. This was verified by killing a
real shard, not by inspection: it recorded `stopped_by_signal: SIGTERM` at
`delete_node:inner:None`. The path exists because an earlier timing probe died
leaving no output and no diagnosable cause — measurement later ruled memory out,
but by then the evidence of what had happened was gone.

<!-- BEGIN:pressure -->
<!-- END:pressure -->

Every resource decision this track made, admitted or refused, is in
`out/resource_gate.log`:

<!-- BEGIN:gate -->
<!-- END:gate -->

<!-- BEGIN:resources -->
<!-- END:resources -->
