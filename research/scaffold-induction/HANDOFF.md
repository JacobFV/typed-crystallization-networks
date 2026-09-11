# Handoff — what to reuse, what to throw away, what this cost to learn

This track asked: **can cross-domain edit history accelerate repair of
synthetically narrowed scaffolds?** The answer is no (`RESULTS.md`), and the
more useful output is the machinery and the rules. A fresh agent drafting the
structural-expressibility pre-registration around §65's `gap 1` should start
here.

---

## 1. Reusable, and worth reusing

| piece | file | why it transfers |
|---|---|---|
| **Typed structural-edit enumerator** | `edits.py` | Generic over `tcn.graph.Program`. Five families; every produced program goes through `Program.validate(registry)`, so no edit can widen the type system. Source pools are derived from the program's own depth scaffold, never hand-listed. Nothing in it knows what a defect is. |
| **Three-split harness** | `domains.py`, `run_domain.py`, `score_final.py` | `train → admission → final`. Admission and repair are complements of **one** predicate on **one** split. `final` is fingerprinted, replaced by a sentinel that raises on any access, and read by exactly one script. |
| **Both resource gates** | `kit.py` | `check_floor` scales to a phase's measured peak; `check_workers` counts live workers by PID and `/proc/<pid>/cwd`. Both fail closed. Both log every decision, admitted or refused, to `out/resource_gate.log`. |
| **Provenance stamping** | `kit.provenance`, `verify.py` | Five identities per artifact: base digest, split digest, grammar digest, producing commit, pre-registration revision. The verifier compares identities across artifacts rather than trusting a file that says PASS. |
| **Promise registry** | `verify.py` `REQUIRED_CLAIMS` | Lists the checks the documents claim exist; fails for each one missing. A claim about the verifier lives inside the verifier. |
| **Outer-loop cost block** | `report.py` `outerloop` | Edits proposed → decided → selection space covered → episodes consumed → wall clock → downstream search size. Makes "moving combinatorics up a level" checkable instead of assumed. |
| **Exact tiered expectation** | `analyse.py`, `verify.py` | `(s+1)/(k+1)` per uniform tier, checked against a *simulation* rather than a re-implementation, because the first draft of the formula was wrong in the pre-registration and in both copies of the code. |
| **Signal-stop discipline** | `run_domain.py` | Checkpoint after **every** defect; `SIGTERM` writes the partial corpus with `stopped_by_signal` and `last_defect_started`; `verify.py` fails if an arm is costed on a signal-stopped corpus. |

## 2. Generator-specific — do not inherit

**The narrowing defect families** (`drop_operator`, `drop_source`, `keep_prefix`,
`delete_node`) are the part to throw away, and the reason matters more than the
fact.

They only ever **remove options from a scaffold that already worked**. The
inverse of a narrowing is a widening, so `SUBST` and `WIDEN` repair everything
and **`ADD_NODE` and `ADD_PATH` repair nothing** — a third of every edit
proposed, zero repairs, in both domains. The mutation grammar was nominally five
families wide and effectively two. Any prior learned on this corpus has learned a
fact about the generator.

Two measurements bound it further:

* **62% of rejected narrowings were not defects at all.** Of 32 `bool` defects
  rejected as still solvable, 20 left the base scaffold's own conforming member
  completely intact. A narrowing at a uniformly random site mostly misses,
  because a conforming member uses a small fraction of the candidates available.
* **`bool` is single-site.** All nine of its admitted defects sit at the output
  node, which is why a one-line rule ("prefer output-adjacent edits") beats every
  learned prior there: the rule encodes the generator's shape.

**What to build instead:** defects that remove *expressiveness*, not *options* —
a scaffold that **cannot say** what the task needs. §45's counting scaffold has no
way to express a running minimum, so no selection of its candidates conforms and
the repair must **add** a reduction. §65's `gap 1` is the same failure one level
up: the schema's STEP pool lacked the two-row displacement, giving 2.6×10¹¹
training conformers and zero over all episodes. The prediction worth
pre-registering there is that `ADD_NODE` becomes the only family that repairs
anything — making that corpus and this one exact complements.

## 3. Rules this track paid for

1. **An amendment ships its check in the same commit.** A13 promised three
   mechanical guarantees about the blind split and implemented none; they stood
   for hours while the artifacts were written and nothing read them. An
   amendment is a claim like any other.
2. **Every derived artifact carries a fingerprint of what it was derived from.**
   A superseded validation was nearly reported as current: `out/` *was* cleaned,
   but a still-running job wrote its output afterwards, so a stale file
   reappeared in an emptied directory with nothing inside it naming its corpus.
   Cleaning a directory does not remove artifacts not yet produced.
3. **A guard with nothing to compare must fail.** The prose-number check was
   discarding ~96% of the document before scanning, and passed because almost
   nothing was left. Guards need a guard: assert that the check still has
   material to check.
4. **Uniformity across arms does not make a bound harmless.** `MAX_NEW = 48`
   was applied identically to every arm and was still deleting the repairs —
   `WIDEN(y, xor)` needs a wiring `legal_candidates` emits at index 55 of 64.
   Uniformity protects the comparison; it does nothing to protect the meaning of
   the thing compared.
5. **Do not ask whether a mechanism solves tasks built in its image.** Ask
   whether it discovers structure whose necessity was not encoded into the
   generator. This generator rewarded inverse edits; §65's prior rediscovered a
   one-line heuristic. Same warning, two tracks, independently.
6. **Fix the reporting unit before the arms.** `bool` (1 site, 9 cases) and
   `rel` (4 sites, 19) behave oppositely; pooling them would have produced a mean
   that is neither domain's behaviour. A12 fixed per-domain reporting and the
   macro pooling rule in advance, and the split verdict it anticipated is what
   happened.
7. **A budget chosen by its yield is tuning.** A8 fixes the episode budget by a
   stated rule — the smallest budget whose admissible defect *set* is stable
   across two doublings — applied identically to every domain, with every
   rejected rung recorded.

## 4. Things a successor should know that are not in the files

* **`arith` was never run to completion** and nothing about it is known except
  its cost (~89 min/case). It is not a negative result; it is an absence. Its
  supervision sits only on the output node, so prefix enumeration cannot reject a
  partial program — if you reuse it, add an intermediate probe first and expect
  the cost to fall by more than the probe costs.
* **The `lang` domain was dropped** because at its required width the constant
  pool makes `legal_candidates` exceed its budget on nearly every operator, and
  the edit space collapses to a handful. If you want a sequence domain, either
  bound the constant pool or give the enumerator a pool-sampling policy — but
  note that sampling a pool makes the edit space a random variable, which the
  exact cost metric assumes away.
* **`ORACLE = 1.00` against a best arm of 18.59** is the most interesting
  unexplained number here. The signal needed to order edits well exists; nothing
  tried captured more than a few percent. Before building a better prior, it is
  worth asking what an oracle knows that these features cannot express — the
  feature set is deliberately small and domain-general, and that may be the
  binding constraint rather than the estimator.
* **The superseded two-way corpora are kept** under `out/superseded-twoway/`.
  They cannot be re-scored into the three-way structure: the new repair predicate
  is strictly weaker, so stored non-repairs may now be repairs, and those are
  exactly the edits where the blind split bites.
* **`run_all.sh` reproduces the whole pipeline** in order, under both gates. It
  has not been run end-to-end in one go — the track was built incrementally — so
  treat it as a specification of the order rather than a tested script.
