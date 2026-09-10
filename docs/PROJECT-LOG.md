# Project log

Chronological record of the whole project, reconstructed from git history
(111 commits, `7a30d6b` 2026-09-08 16:38 → `1257715` 2026-09-09 13:40) and
cross-checked against `research/FINDINGS.md`. **Append new phases; do not rewrite
earlier ones.** Where a later phase overturned an earlier claim, the earlier
entry is left standing and the reversal is recorded in its own right — the
reversals are the most useful part of this file.

Per-session narrative lives in [`handoffs/`](handoffs/). Per-result measurement
lives in [`../research/FINDINGS.md`](../research/FINDINGS.md), which is
authoritative wherever it disagrees with this summary. Every claim reversed at
any point is indexed in [`CORRECTIONS.md`](CORRECTIONS.md).

---

## Phase 1 — Implementation and the first measurement pass
**2026-09-08 16:38 → 19:25**, `7a30d6b` … `dadf3fd`

The system is implemented in one commit (`7a30d6b`): typed data algebra over
`bool | int[n] | set[T] | tuple[T...]`, operator registry, soft program with
candidate mixtures, crystallization scheduler, synthetic generators, curriculum
runner, exact export runtime.

Eight parallel research tracks then measure it rather than assume it. The
consolidated result (`936ed80`) is uncomfortable and sets the project's tone:
**exactness and dense hierarchical supervision hold up; the flagship mechanisms
do not.** Track 8 concludes differentiable search is not earning its keep —
enumeration settles both flagship results in milliseconds.

A correction lands within the hour (`b032a33`): a "Mario null finding" was an
artifact of the search reaching only indexed sources. First entry in what
becomes a long list.

## Phase 2 — The instruments were wrong, not (only) the method
**2026-09-08 19:50 → 23:31**, `a273474` … `69e87e1`

The measurement pass is turned on itself and finds **instrumentation faults**
(`a273474`). Two capabilities written off in Phase 1 are re-tested and work:

- **Recursive abstraction** (`9f4f4f6`) — track 5's verdict does not survive
  fixing F1 and F2. Module on the output path in 27/27 successes; the flat space
  of 230,400 exhausted with no solution.
- **Structural generalization** (`64fc865`) — on a benchmark that earns the claim.

Also here: **positional reuse is expressible today** with `insert`/`pair`/`map`
(`bed5b5c`, measured `55803ac`) — one module at thousands of positions via three
caller nodes, width-independent.

And a self-inflicted defect found and fixed (`1923e91`): **an explicit `table`
config was silently ignored by a restricted pool**, so a held-out-generalization
run had been evaluating on its own training distribution. The tell was `seen`
and `unseen` identical to two decimals. The invalid log is preserved rather than
deleted — a decision that pays off much later (§40).

The merge queue and its standing rule are established (`27ffc5a`): **never merge
a change to `tcn/` or `generators/` while an agent is measuring against main**,
because `source_fingerprint()` hashes both.

## Phase 3 — Crystallization refuted, twice more; substrate defects found
**2026-09-09 00:30 → 06:12**, `bb31560` … `c1d011a`

`ec953bc` gates crystallization on the task loss and measures it against argmax.
It loses. That is refutation two and three (after the Phase 1 ablation):
DARTS-PT-style perturbation selection and loss-gated eligibility both lose to
plain argmax at equal compute. The diagnosis is stable: **the interval in which
the better measurement wins is exactly the interval in which committing is a
mistake.**

`a45a57f` is the session's largest single correction: **the "address wall" was a
dead surrogate, not a relaxation limit.** `eq`'s surrogate is exactly 0.0 past
|a−b| ≥ 11 and `lt`'s past 17; the failing experiment's gradients were
identically zero and a probe was averaging over dead nodes. This had been
reported three times and briefed to four agents before it was caught.

Then **four substrate defects** that had been silently preventing gradient
learning are found and fixed (`fddbc79`, `3c17250`): operator-surrogate
temperature conflated with candidate-softmax temperature; one-candidate trainable
nodes treated as frozen; the byte→numeric boundary; tuple broadcasting.

`be32576` adds the **byte commitment operator** — a byte is uncommitted, not
un-numeric — which makes convolution expressible; 3×3 Sobel-x is then recovered
exactly by 4/4 seeds.

## Phase 4 — Capabilities, and a curriculum that was cheating
**2026-09-09 06:15 → 09:34**, `be32576` … `2cda074`

The capability results land in sequence:

| commit | result |
|---|---|
| `0115056` | **63 of 179 language lessons were exploitable**; 31 solvable by copying. 14 re-drawn |
| `06f1cb3` | Persistent module library + curriculum artifact flow; three-stage chain measured |
| `8b8764d` | A task posing **real credit assignment**, with two solutions and honest controls |
| `6f57fa6` | The composite action closes the hierarchy gap |
| `626b608` | **Raw pixels → exact hierarchy**: 227/227 links, 12/12 trees |
| `0b538ec` | **Code generation**: a netlist from behaviour alone, certified unique |
| `98cdca8` | CORRECTION: a policy **can** be learned from reward; the earlier claim came from a baseline that never ran REINFORCE |

`c0f5c4c` records a limit that constrains everything after it: **a hardened
module is width-specific**; what transfers is the *selections*.

Then two measurements that reframe the project. `025d7b2`: the 117.71 MB
`visual.pyz` is **99.68% repeated type declarations** — envelope, not content;
learned content is 21.3 bits. And `2cda074`: the complete inference path is
**146×–90,400× slower than plain Python**, with **97.7%** in
decode/encode/validate and **0.56%** in operator semantics. `execution_cost` is
blind to the dominant term.

## Phase 5 — The efficiency question, settled
**2026-09-09 10:04 → 13:40**, `8afd6aa` … `1257715`

`8afd6aa` is refutation **four**: seasons — reversible winter/summer
crystallization, the form the third refutation explicitly asked for — repairs the
incumbent scheduler and still loses to plain argmax at equalized compute. The
branch is deliberately **not merged**.

`e74aa87` rewrites the constitutional narrative accordingly: **crystallization is
a product, not a schedule.** New ARCHITECTURE §8.1 separates four costs
(operator, value movement, complete-path latency, environment) and three sizes
(21.3 bits learned / 41 KB canonical / 117.7 MB shipped), and forbids quoting
`execution_cost` as latency until it predicts measured latency.

Then the decisive experiment, with criteria stated in advance:

- **§42 — the overhead was the interpreter.** Compiling a frozen program to
  standalone stdlib Python takes the visual parse from **103,487,972** element
  operations to **6,144**, none internal; attribution flips 97.7% marshalling →
  66.7% operator work. *The one clear positive of the phase.*
- **§41 — but ranking cannot shorten it.** Certified by exhaustion: the
  description-minimal program is the **bytecode-maximal** one, and
  `execution_cost` is **constant (148.0)** across ten programs that differ in
  real executed work. Residual gap is program length: 11.1× more bytecodes ×
  2.25× per bytecode.
- **§43 — matched neural baselines exist, and the answer is mixed.** Typed wins
  quality on all three artifacts (nine CNN arms score **0 exact trees** at every
  width and budget); the CNN wins visual execution 300–556× and size 250×.
- **§44 — the abstraction loop closes and fails at the selection.** The earned
  module ties the *wrong-module control* exactly. Cause: exact minimisation
  dissolves the useful fragment — `MAJ3` survives in **1 of 6** minimised
  programs.

Two shipped numbers are corrected in the same phase: the language capability is
**0.9986, not 1.000**, and it reproduces **only** on the pre-audit stream (§39).

## Phase 6 — Both remaining limits turn out to be expressiveness
**2026-09-09 13:40 → 19:00**, `4a0ad71` … `a1a5553`

Two independent measurements land on the same diagnosis, which is the phase's
result rather than either measurement alone.

**§45 — the language capability was counting, not balancedness.** Run unchanged
on an honestly-posed split of the re-drawn (non-exploitable) stream, the shipped
stage-B scaffold contains **no conforming program**: space 45,375, exhausted,
**0 conforming, certificate `complete`**. A proved non-existence, not a timeout.
§24's re-draw made bracket counts identical across both classes, so the honest
counting program scores **exactly the majority, 0.5262**. Post-audit,
`balanced ⟺ min prefix ≥ 0`, and a minimum is not a sum. The repair needs **no
new operator**: a running `min` beside the running `add` scores **1.000 on all
859 held-out episodes** at unseen lengths 16–22. The control that makes the
negative trustworthy is that the same code path reproduces §19 to every digit
(0.9986187845303868, n=724) on the pre-audit stream — the harness reproduces
where the result holds and proves non-existence where it does not.

**§19 is bounded, not withdrawn.** Its program is real and solves the task as
that lesson posed it. What is refuted is the implicit claim that the capability
was about balancedness.

**The convergence.** §41/§42 had already located the visual residue in
expressiveness: `Program.execute` is an unconditional loop over every node and
the depth scaffold *rejects* within-tick back-edges, so totality is a validator
invariant and early exit is **inexpressible**, not merely unfound. §45 locates
the language limit in expressiveness too — the scaffold cannot say "running
minimum". Two tracks, different domains, same class of limit.

`research/algorithm-resynthesis/DESIGN.md` is the design study that follows: an
expressivity audit of 23 constructs cited to code, a recommendation of two
languages (specification IR unchanged, separate algorithm IR) with oracle-guided
synthesis, e-graphs kept as semantic-identity engine rather than algorithm
inventor, Lean deferred to the parametric case where exhaustion is impossible,
and a pre-registered miniature experiment. Its most consequential finding is that
the **objective cannot currently see the fix** — `filter` is charged at declared
capacity, `execution_cost` is constant across programs differing by 48 executed
bytecodes, and early exit never improves worst-case count — so a cost model
distinguishing expected from worst-case is a **precondition** for that direction,
not a follow-on.

**Documentation restructured** so history accumulates: dated append-only session
handoffs under `docs/handoffs/`, this log, and `CORRECTIONS.md`. The previous
single `HANDOFF.md` was being overwritten each session, destroying the prior
record; session 1's was recovered from git verbatim.

---

## Standing shape of the project, as of 2026-09-09

**Holds up:** exact typed execution; dense hierarchical supervision (the
mechanism that actually works); recursive abstraction as a *mechanism*; the
persistent module library and curriculum flow; compilation to ordinary software.

**Refuted, four times independently:** the progressive irreversible freezing
schedule, most recently in the reversible form its third refutation asked for.

**The binding constraint, as of Phase 6:** not search, not compute, but **what
the representation can express**. Established independently in two domains (§41,
§42, §45).

**Open:** whether the min-prefix fix is reachable by the *system* rather than by a
human reading the diagnosis — enumeration without a hand-chosen window, and the
gradient path against its own control; whether scaffold design can be proposed
from pre-hoc evidence at all, or is currently a human input; abstraction
*selection* from non-minimised corpora; a cost model that distinguishes expected
from worst-case work.
