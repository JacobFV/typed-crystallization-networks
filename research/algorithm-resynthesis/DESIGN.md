# Post-crystallization algorithm resynthesis — design study

**Status: design only. Nothing implemented. No operator added.** Deliverables 1–10
of the brief; item 11 (implementation) awaits approval of the §9 experiment.

Every claim about present expressivity below is derived from code and cited to
file and line, not from intuition.

---

## 0. Three findings that reframe the question

**0.1 The eagerness is structural, not incidental.** `Program.execute`
(`tcn/graph.py:105-110`) is an unconditional loop over `self.nodes`, assigning
every node exactly once per tick. There is no mechanism by which a node is *not*
evaluated. The depth scaffold (`tcn/graph.py:72`) rejects any edge whose source
depth ≥ the node's depth, so the graph is a strict DAG and cannot express a
back-edge within a tick. **Totality is enforced by the validator, not merely
unimplemented.**

**0.2 The compiler's `if` is syntactically lazy and semantically eager.**
`tcn/compile.py:384-386` (branch `compiled-runtime`) emits

```python
v19 = v17 if v16 else v18        # mux
```

which *looks* like a lazy conditional. It is not. `v17` and `v18` are SSA
variables assigned by earlier emitted statements, because the compiler lowers in
dependency order. Both branches have already been computed when the selection
runs. This is the single most load-bearing fact in the study: **the gap is not
that the compiler discards laziness — it is that the specification never had
any.**

**0.3 The cost model cannot see sparsity, so it cannot reward the fix.**
`map`/`filter` cost is `max(1, ts[0].capacity) * m.execution_cost(self)`
(`tcn/operators.py:118`) — charged at declared **capacity**, not at realized
size, and identically whether the predicate passes once or never. Combined with
§41's finding that `execution_cost` is *constant* (148.0) across ten programs
differing by 48 executed bytecodes, this means: **an early-exit algorithm is
invisible to every objective this repository currently optimizes.** Worst-case
executed-operation count also does not improve under early exit — only *expected*
cost over an input distribution does.

**Consequence for sequencing.** A cost model that distinguishes expected from
worst-case cost is a **precondition** for this research direction, not a
follow-on task. Without it, a successful resynthesizer would produce an algorithm
the system scores as no better. This reorders the brief: §11 (cost model) comes
before §8 (experiment).

---

## 1. Expressivity audit

Derived from `tcn/graph.py`, `tcn/operators.py`, `tcn/compile.py`
(branch `compiled-runtime`), `research/earned-abstraction/mine.py`.

Legend — *Lazy?*: whether unneeded work can be skipped. *Search?*: reachable by
the differentiable/discrete search. *Module?*: reusable via `register_module`.

| construct | expressible today? | how represented | lazy? | search? | module? | compiler |
|---|---|---|---|---|---|---|
| conditional branch | **value selection only** | `mux` (`operators.py:199`) | **no** — both operands precomputed | yes | yes | `a if c else b` over prior SSA (0.2) |
| short circuit | **no** | — | — | — | — | — |
| bounded loop | **unrolled only** | N replicated nodes in the DAG | no | yes | yes | N straight-line statements |
| while / until | **across ticks only** | `Program.state` + external tick loop | n/a | yes | via `lift_state()` | per-tick function |
| fold / reduce | **fixed kernels only** | `sum`,`mean`,`reduce_min`,`reduce_max`,`count` (`operators.py:210`) | no | yes | yes | Python builtin |
| fold with *learned* body | **no** | — | — | — | — | — |
| scan (prefix) | **no** | — | — | — | — | — |
| find-first | **no** | `filter` evaluates every element (`operators.py:227`) | no | yes | yes | full comprehension |
| sparse iteration | **no** | `map`/`filter` touch all elements | no | — | — | full comprehension |
| recursion | **no** | forbidden by depth scaffold (`graph.py:72`) | — | — | — | — |
| finite state machine | **yes, across ticks** | `state` triples + update node | n/a | yes | `lift_state()` | per-tick step |
| memoisation | **no** | — | — | — | — | — |
| indexed lookup | **partial** | `join` = full cross product + equality filter (`operators.py:225`), **O(n·m)**; `index` = dynamic tuple index | no | `join` no-grad | yes | nested comprehension |
| sort | **no** | — | — | — | — | — |
| grouping / partition | **multi-pass only** | repeated `filter` | no | yes | yes | repeated comprehension |
| dynamic programming | **no within tick** | across ticks via `state`, awkwardly | — | — | — | — |
| divide and conquer | **no** | — | — | — | — | — |
| streaming transducer | **yes, across ticks** | state machine | n/a | yes | yes | per-tick step |
| parser / state machine | **yes, across ticks** | as above | n/a | yes | yes | per-tick step |
| work queue | **no** | — | — | — | — | — |
| branch and bound | **no** | — | — | — | — | — |
| procedure call | **yes** | `module:<digest>` operator (`operators.py:44`) | eager | yes (as candidate) | yes | inlined or called |
| polymorphic procedure | **no** | module input type names the observation width (§30) | — | — | — | — |

**Summary of the boundary.** The algebra is a *total, eager, first-order,
finite-depth dataflow language with across-tick recurrence and procedure call*.
It has: composition, procedure abstraction, bounded structural iteration
(`map`/`filter`), fixed reductions, dynamic indexing, and state. It lacks
entirely: **laziness, data-dependent iteration, accumulation with a learned
body, and any data structure with better-than-linear lookup.**

The visual gap is exactly this boundary. The hand-written reference
(`research/compiled-runtime/fixtures.py`) wins by two early `continue`s and a
contiguous run scan — a *find-first* and a *scan*, the two most conspicuous
"no"s above — exploiting that ~20 of 961 interior positions are corners.

**Abstraction identity is syntactic.** `mine.py` enumerates single-exit sub-DAGs
(`mine.py:12`), canonicalizes to a `Program`, and treats sameness as equal
`Program.digest` (`mine.py:22-23`). Two implementations of one transformation
with different factorizations are, by construction, different abstractions. §44
measured the consequence: `MAJ3` survives in 1 of 6 minimised programs and is
never proposed.

---

## 2. Candidate architectures

### A. Extend the single algebra
Add lazy/iterative constructs directly to the training algebra.

*Against:* every added operator enlarges the search space the differentiable path
must cover, and §7/§12/§37 show that path is already the weak link (the gradient
path solves 1 of 6 corpus tasks, §44). It also breaks the property that makes the
substrate auditable. **Rejected**, and this is what the brief's §2 anticipates.

### B. Two languages: specification IR → algorithm IR
Keep the current algebra as the *specification* language. Introduce a separate
*algorithm* IR, used only after a behaviourally correct frozen program exists,
carrying control flow. Resynthesis is a search from spec to algorithm IR.

*For:* the training algebra stays simple; exactness is preserved because the spec
remains the oracle; the algorithm IR can have constructs no learner ever has to
search over.

### C. Equality saturation over an extended basis
An e-graph whose rules include loop/scan introduction forms.

*Assessment:* e-graphs excel at *canonicalizing within* a basis and are the right
tool for level 1 and parts of level 2. They do **not** invent algorithms: the
rule that rewrites "evaluate predicate at all N positions, then filter" into
"scan with early exit" is precisely the hard part, and supplying it is supplying
the answer. **Role: canonicalizer and semantic-identity engine (§6), not the
level-3 proposal engine.**

### D. Oracle-guided synthesis (CEGIS) against the frozen program
The frozen program is an *executable, exact* specification: unlimited labelled
I/O for free, and a decision procedure for counterexamples. Synthesize an
algorithm-IR program from examples; verify by exhaustion (finite carriers) or
SMT; on failure, feed the counterexample back.

*For:* uses the project's strongest asset — exactness — directly. Carriers are
finite and often small, so verification is frequently *cheaper than proving*.

### Recommendation

**B + D**: two languages, with oracle-guided enumerative/CEGIS synthesis in the
algorithm IR, C as the semantic-identity substrate. Lean deferred (§10).

---

## 3. A minimal algorithm IR — three constructs, not twenty

Start with exactly what the measured gap requires. Every construct needs typed
exact semantics, an explicit cost semantics with an *expected* case, and a
lowering to standalone stdlib code.

```
guard  c then e else e'        -- lazy; only the taken branch is evaluated
find   i in 0..n-1 where p(i)  -- first index satisfying p, or none
scan   i from s while p do f   -- bounded accumulation, explicit finite bound
```

Plus `let` (sharing) and the existing module call. **Not** initially: `while`
with a proved variant, memo tables, sort, recursion, general data-structure
synthesis. Those are level-3 constructs the current evidence does not yet demand;
adding them now would be inventing an instruction set rather than discovering
one, which is the failure mode the brief warns against.

`find` and `scan` carry an explicit finite bound so termination is structural,
not proof-obligated. That deliberately keeps the first version inside decidable
territory.

---

## 4. Proposal engines × equivalence backends

| proposal engine | invents control flow? | needs the answer supplied? | fit here |
|---|---|---|---|
| enumerative synthesis over algorithm IR | yes | no | **primary**, spaces are small if the IR is small |
| CEGIS | yes | no | **primary**, spec is a perfect counterexample oracle |
| syntax-guided synthesis (grammar-restricted) | yes | grammar is a strong prior | good second stage |
| equality saturation | **no** | rules encode the answer | canonicalization / semantic identity |
| stochastic superoptimization | yes | no | fallback when enumeration blows up |
| MCTS / beam over algorithms | yes | no | later, when IR is larger |
| anti-unification | partially (generalizes) | no | **abstraction discovery (§6)**, not resynthesis |
| ILP | yes | needs relational encoding | poor fit to typed carriers |
| partial evaluation / deforestation / fusion | **no** (level 1–2) | no | complements, does not replace |
| loop / invariant synthesis | yes | no | needed only when `while` is admitted |

| equivalence backend | when it is right | cost |
|---|---|---|
| **exhaustive execution** over the finite domain | small declared carriers — the common case | cheapest; already implemented; yields a *certificate*, matching repo convention |
| **SMT / SAT** | finite but too large to exhaust; bitvector/boolean carriers | moderate; no proof engineering |
| random + structural held-out | never as the sole check | weak; use only alongside |
| **Lean / proof assistant** | *parametric* algorithms (any n, any width), where exhaustion is impossible | high; justify per §10 |

**The separation the brief asks for holds:** the proposal engine should be
enumerative/CEGIS; the equivalence checker should be exhaustion first, SMT
second, Lean only where parametricity forces it; the objective is the §11 cost
model. Lean is a plausible *oracle*, an implausible *optimizer*.

---

## 5. Semantic equivalence classes, not one minimum

§44's measured lesson — exact per-task minimisation destroys shared structure —
generalizes to: **local program minimality ≠ global library minimality.**

Proposed object:

```
SemanticClass
  semantic_id     canonical behavioural signature (below)
  interface       typed domain -> codomain, plus shape parameters
  members[]       implementations, each with:
                    program (spec IR) or algorithm (algorithm IR)
                    cost vector: worst_ops, expected_ops(dist), bytecodes,
                                 allocations, live_width, measured_latency
                    description_bits, factorization fingerprint
                    provenance: task(s), search that found it, certificate
```

`semantic_id` for a finite carrier is the exhaustive input→output table hashed
canonically — cheap, exact, and it merges `a∧(b∨c)` with `(a∧b)∨(a∧c)`
immediately. For carriers too large to exhaust, an SMT-checked representative
plus a behavioural hash over a declared sample, marked as *unverified class
membership* until proved. **Never silently merge on a sampled hash.**

Retain a **Pareto front per class**, not one member. §44's failure is a direct
consequence of keeping only the minimum.

---

## 6. Semantic abstraction mining

Successor to `mine.py`, in the order the evidence supports:

1. **Semantic normalization before counting** (B/C in the brief). Cluster
   fragments by exhaustive truth table rather than `digest`. This alone would
   have merged the factorizations §44 measured as distinct — the cheapest
   available test of the whole thesis.
2. **Port/constant parameterization** (D). Count instances differing only in
   which inputs they read as one abstraction.
3. **Anti-unification** (E) to generalize related transformations into a schema.
4. **Retain non-minimal members** (F) — the population of §5 is what supplies
   the common factors minimisation removes.
5. **Refactor-then-re-optimize loop** (G/H): rewrite the corpus using candidate
   abstractions, re-measure the joint objective, iterate.

The joint objective from the brief is the right shape:

```
objective(L, {P_i}) = task_error + λ1·compiled_runtime + λ2·bits(L)
                    + λ3·Σ bits(P_i | L) + λ4·search_cost + λ5·interface_complexity
```

with definitions charged once and execution charged per use. **λ1 must use the
§11 expected-cost model**, or this reduces to description-bit ranking, which §41
certified picks the bytecode-*maximal* program.

---

## 7. Feeding discovered algorithms back into `tcn.library`

**§55 REFUTES part of this section:** "the class holds the schema" is wrong —
`tcn.library` stores *programs* and a schema is *code*. A class record holds a
schema **reference**, bounded by `source_fingerprint`. §55 also measured that a
`semantic_id` field is silently erased by `_save()`, so it is a real core change
and nothing yet needs it.

`tcn/library.py` is content-addressed with fixtures and a source policy. Minimal
change, additive:

- keep `digest` as **artifact** identity (unchanged, so existing verification and
  `strict`/`revalidate` still hold);
- add `semantic_id` as **class** identity;
- a library entry becomes a class with several artifacts and a declared *preferred*
  member per cost axis;
- a resynthesized algorithm-IR member is registrable as a candidate operator only
  once it has (a) an equivalence certificate against the spec member and (b) a
  fixture that re-executes exactly.

This preserves the §30 constraint (a hardened module is width-specific): the
class holds the *schema*, artifacts hold instantiations. That is the brief's §12
"generic schema + instantiated typed artifacts", and it needs no type-system
weakening.

---

## 8. Cross-domain story

The candidate universal factors — `equality`, `neighborhood`, `locate`,
`partition`, `group`, `connectedness`, `ordering`, `scan`, `accumulate`, `count`,
`bind`, `lookup`, `parse`, `transduce`, `iterate-until`, `maintain-state` —
should **not** be hand-added. The test for promoting any of them is the brief's
own: *it must help a later held-out task*, measured as §44's arms measure it
(against no-library, against a wrong module of the same size, against a
hand-authored one).

What the audit says about plausibility: `locate`/`find-first`, `scan`,
`group`/`partition` and `lookup` are exactly the constructs missing from the
algebra **and** the ones that recur across all four domains — sparse detection in
vision, span-finding in language, control-location in computer use, entity
lookup in embodied worlds. That coincidence is the reason to believe this
direction is about ontology and not just speed. It is also why the first
experiment should target `find-first` and `scan` specifically.

---

## 9. The smallest falsifiable experiment

**Not the parser.** A miniature with the same structure, exhaustively verifiable.

**Setup — guard-dominated sparse evaluation.** A frozen spec program over a small
finite carrier: for each of N positions, compute a cheap predicate `p_i` and an
expensive value `e_i`, and output `{(i, e_i) : p_i}`. Written in today's algebra
this necessarily evaluates all N `e_i` (§1). Choose the input distribution so
that `p_i` holds at a small expected fraction, mirroring 20-of-961.

**Task:** a resynthesizer must produce an algorithm-IR program computing the same
transformation with substantially fewer executed primitives, **without the
algorithm being supplied as a candidate**.

**Staged, per the brief:**
- **Stage 1** — recover a lazy `guard` avoiding an expensive continuation.
- **Stage 2** — recover `find-first`/bounded scan from an unrolled predicate family.
- **Stage 3** — compose them on one visual-parser subroutine.

### Pre-registered success criteria (all must hold)

1. **Exactness:** the resynthesized algorithm agrees with the frozen spec on
   **every** input of the declared finite domain — exhaustive, yielding a
   certificate, not a sample.
2. **Expected work:** ≥ **3×** fewer executed primitive operations than the
   frozen spec, measured over the declared input distribution.
3. **Not supplied:** the target algorithm does not appear verbatim in the
   candidate set or rule set; the search must compose it. This is checked by
   ablation — removing the specific construct must be the only thing that
   prevents the result.
4. **Held out:** exactness survives structural cases not used during synthesis.
5. **Deployable:** generated standalone code runs stdlib-only and is bit-identical
   to arm A, per the existing `research/compiled-runtime` harness.
6. **Honest worst case:** the report states worst-case executed operations, which
   is expected **not** to improve. A win reported without this is a false win.

### Pre-registered falsification

- Expected-work gain < 3× → the representational gap is not primarily laziness.
- The result appears only when the construct is supplied verbatim → this is
  template instantiation, not algorithm discovery. **Report as negative.**
- Exhaustive equivalence not obtainable at the chosen size → the domain was mis-chosen;
  shrink it rather than weakening the check to sampling.
- Level confusion: if the gain is traceable to DCE/CSE/constant folding, it is
  **level 1** and must be reported as level 1 (brief §9).

**Precondition, from §0.3:** the cost model must report expected-vs-worst
separately *before* this runs, or criterion 2 is unmeasurable.

---

## 10. Where Lean would earn its place

Deferred, with a concrete test rather than a presumption.

The decisive consideration: **for finite carriers, exhaustive execution is both
cheaper than proving and already implemented**, and it produces the same kind of
certificate this repository already uses (`unique`, `complete`). Lean's advantage
appears only where exhaustion is impossible — **parametric** algorithms (any `n`,
any width). That is exactly the regime §12/§30 pushes toward, so the question is
live but not yet urgent.

Run the brief's experiments A/B/C **only after** the §9 experiment, and record
proof size, wall clock, manual theorem engineering, whether proof generation
automates, and — the deciding column — **whether SMT or exhaustion would have
been cheaper**. Lean earns a role only as a scalable equivalence backend for the
parametric case. It is not needed in the deployed runtime, and adding it there
would break the stdlib-only property that §42 established as a genuine win.

---

## 11. Cost model — the precondition

Replace `execution_cost` for this purpose with measured, compiled work:

| metric | why |
|---|---|
| executed primitive count | comparable to the hand-written reference |
| **expected** executed count over a declared distribution | the only metric under which early exit is a win |
| worst-case executed count | must be reported alongside, and will not improve |
| bytecodes / instructions | §41 showed this is the metric that tracked reality |
| allocations, live-value width | §42 measured 48.4 MB → 0.121 MB; representation choices show up here |
| measured batch-one latency | the ground truth `execution_cost` failed to predict |

All three of static worst-case, expected-over-distribution, and measured-realized
are reported separately. ARCHITECTURE §8.1 already forbids quoting
`execution_cost` as latency; this extends that to forbidding **any** single
scalar standing in for the three.

---

## Answer to the core question

*Can a system start from a correct low-level learned program, discover a
semantically equivalent better algorithm, certify it, promote its reusable
factors, and make future intelligence cheaper?*

**The pieces that exist are stronger than expected:** an exact executable
specification, a compiler to ordinary code, exhaustive certificates, a
content-addressed library, and a measured target with a known-good reference.

**Two things block it today, and both are precisely located.** The specification
language is *provably total and eager* — the validator rejects the structure a
faster algorithm needs — so the better algorithm is not merely unfound but
**inexpressible**, which is why no amount of rewriting within the basis can
reach it. And the objective is *blind to the improvement*: `filter` is charged at
capacity, and `execution_cost` is constant across programs that differ in real
work, so an early-exit algorithm would score as no better even if found.

**The honest sequencing** is therefore: cost model → algorithm IR with three
constructs → oracle-guided synthesis on the §9 miniature → semantic (not
syntactic) abstraction identity → library induction → cross-domain test. The
first two are small. The §9 experiment is the one that decides whether the rest
is worth building, and it is designed to fail loudly if the answer is no.
