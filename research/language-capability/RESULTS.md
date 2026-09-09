# Language capability: what a typed program can learn from generated text

`docs/VALIDATION.md` records "179 executable lessons x two seeds generated
nonempty prompt/answer". That is a sampling check. No program in this repository
had ever been trained on `generators/language`. This track trains one, and
bounds the rest of the catalogue.

Everything here was produced with `tcn/` and `generators/` **unmodified**. The
agent input is `text` only; `construction` (latent) and `answer` (probe) are
supervision channels and never enter a program's input ports. Core changes are
proposed as diffs in section 10, none applied. Nothing is committed.

Scripts and raw data are in this directory: `bound.py`/`bounds.json`,
`realization.py`/`realization.json`, `surrogate.py`/`surrogate.json`,
`descent_direction.py`/`.json`, `common.py`, `scaffolds.py`, `splits.json`,
`baselines.py`/`.json`, `run_stage_a*.py`/`stage_a*.json`,
`run_stage_b*.py`/`stage_b*.json`, `conforming_sweep.py`/`conforming.json`,
`certificate.py`/`.json`, `final_eval.py`/`.json`, `difficulty_probe.py`/`.json`.

## 0. Result in one paragraph

A typed program learns a real language task end to end from raw prompt bytes: it
**discovers its own lexical unit** (which byte denotes an opening bracket, over
the full 0-255 alphabet, and where the symbol field starts) and then a
**grammaticality rule over the recovered symbol sequence**, reaching 1.000 on
string lengths and nesting depths never trained on -- against a 0.548 constant
and a 0.648 best-fitted-feature baseline -- once the conforming set is
tie-broken on a validation split; the program `enumerate_fit` returns by
declaration order instead reaches 0.9986, failing only at the longest unseen
length. It is found by exhaustive enumeration with
a uniqueness certificate at stage A and a 10-member conforming set at stage B;
gradient descent on the identical spaces conforms in **0 of 44 runs**. Three
separate findings qualify the claim: the lesson named `context_free_language`
does not exercise a stack (its negatives always break the bracket count, 0 of
20,000 seeds), so what was learned is counting, not recursion; only 6 of 179
lessons have prompts byte-predictable at a fixed offset, which bounds this method
to a small corner of the catalogue; and `construction` fails to determine
`answer` in 39 of 179 lessons, so dense staging is not uniformly available.

---

## 1. Bounding the catalogue before searching

`bound.py`, 179 implemented lessons x 300 seeds. For each lesson: the answer
distribution, and the accuracy of the *optimal deterministic predictor* from the
observation and from the latent -- an upper bound on any program reading it.

| question | answer |
|---|---|
| Is `answer` determined by `text`? | **Yes, in all 179 lessons.** No prompt was ever seen with two different answers. 13 lessons repeat prompts across seeds; none conflicted. |
| Is `answer` determined by `construction`? | **No -- in 39 of 179 it is not.** Mean bound 0.929. |
| Majority-class baseline | 0.010 to 0.557 by lesson; the binary lessons sit at 0.51-0.56. |

The 39 lessons where the latent does not determine the answer are a finding about
the generator. `generators/language/generator.py` forwards `metadata['hidden']`
as `construction`, and what a lesson puts there is its own business: sometimes
the whole pre-realization structure, sometimes only the *question*.

| lesson | best possible from `construction` | majority baseline |
|---|---|---|
| `finite_state_language` | 0.540 | 0.540 (the probe is worth **nothing**) |
| `protocol_discovery` | 0.267 | 0.230 |
| `unification` | 0.280 | 0.203 |
| `negation` | 0.697 | 0.523 |
| `symbol_discrimination` | 0.630 | 0.557 |

`finite_state_language` is the sharp case: its `hidden` is `{symbol, parity}` --
the induced *rule* -- and not the query string the rule is applied to, so a probe
on `construction` cannot beat the majority class. **Dense intermediate
supervision is available for 140 of 179 lessons and structurally unavailable for
39.** A curriculum that stages on `construction` has to check this per lesson.

## 2. What a typed program can address in text, catalogue-wide

`text` is `tuple(int32 length, tuple[capacity x int[8] role="byte"])`.
`role="byte"` puts bytes outside `Type.numeric`, so the whole arithmetic,
ordering and aggregation family is type-illegal on them; `eq`, `index`, `pack`,
`project` and `tuple` are the entire reachable inventory, and there is no
tuple-to-set conversion, so `map`, `filter` and `member` cannot reach text
either. Addresses are therefore constants, or computed from the one numeric
handle the observation carries -- the length field.

`realization.py` measures how much of each prompt sits at a predictable offset
(mean modal-byte frequency per position, aligned left and right, 200 seeds):

| | lessons |
|---|---|
| either alignment >= 0.90 | **6 of 179** |
| either alignment >= 0.75 | 22 of 179 |
| whole variable region drawn from <= 4 distinct bytes | **2** (`context_free_language`, `parse_depth`) |

That is the structural certificate for the rest of the catalogue. The grammar
engine varies the realization -- the same lesson renders a list inline
(`The symbols are e; a; a; d.`) or as a bulleted block -- so content offsets move
with vocabulary and list length. A constant-address program cannot follow that;
relaxed addressing is measured at chance in section 5.

## 3. The task, and a benchmark finding that changes what it means

**`context_free_language`**: `The string is ((((()())).\nIs string balanced?`
with the answer `yes` or `no`. The program's `bool` output is the generator's own
answer under a stated bijection -- no re-encoding of the target.

Why this lesson: it is ARCHITECTURE section 9's `L` stage (grammar and synthetic
language); the answer is binary with a 0.52 majority baseline; `construction`
determines the answer (bound 1.000), so dense staging is available; and it has
the shortest prompt in the catalogue (103-117 bytes; text capacity set to 128).
It is a single template with the symbol field always at byte 14 and prompt length
exactly `101 + L`, which is what makes it reachable at all.

**The finding: as sampled, this lesson is not context-free.** Its negatives are
made by flipping one character of a balanced string, which always breaks the
count, so `#( == #)` and `balanced` agree on **20,000 of 20,000 seeds**. A
counting predicate scores 100%; no stack is needed. The lesson's own docstring
calls it "the canonical test that a learner has stack-like state". On its sampled
distribution it is not that test. This is the F-bench pattern again: the
difficulty the name asserts is absent from the draws.

The consequence is stated up front: what follows is **learned lexical perception
plus a counting/agreement judgment over a symbol sequence, from raw prompt
bytes** -- not recursion.

## 4. What was learned, staged on the privileged latent

Two stages, per ARCHITECTURE section 3 and findings sections 3, 11 and 14.

### Stage A -- lexical perception (the tokenizer the agent is not given)

AGENTS.md forbids a tokenizer as preprocessing, and `role="byte"` means the agent
has no notion of a symbol. Stage A *learns* one:
`open(text, i) = eq(index(bytes, base + i), c)`, with **`c` searched over the
whole 0-255 byte alphabet** and `base` over 0-40. The address is computed
(`base + i`), so one module serves every position. The probe is the per-position
bit read off `construction`'s `string` -- section 7's intermediate supervision.

| | |
|---|---|
| space | 10,496 programs |
| training | 12 episodes, lengths {2,4,6}, 44 supervised positions |
| enumeration | exhausted in **3.5 s**, `conforming = 1`, **uniqueness certified** |
| selected | `base = 14`, open byte = **40 = `'('`** |
| held-out positions, seen lengths | 1.000 (166) |
| held-out positions, unseen lengths {8,10,12,14} | **1.000** (422) |

The module is hardened, pruned and registered as a frozen operator -- immutable,
no internal gradients (ARCHITECTURE section 4), `execution_cost` 5.0. It was
supervised only *inside* the string; at the 1,180 positions past the string end
that stage B relies on, it reports "not open" 1,180 / 1,180 times, which is an
extrapolation that was checked rather than assumed.

### Stage B -- the grammaticality rule

The frozen module is called at each of 16 positions with a computed address. The
search chooses the length relation `symbols = length - c` (121 candidates), the
two values the symbol-to-step map emits (5 x 5), and the rule that reads the
accumulated count (15: `eq`/`ge`/`le` against -2..2). 84 nodes.

| | |
|---|---|
| space | 45,375 programs |
| training | 24 episodes, lengths {2,4,6} -- only **12 distinct strings** |
| enumeration | exhausted in **363 s**, `conforming = 10`, **not unique** |

**Exact exported program against matched baselines on the same episodes:**

| split | n | lengths | program | majority constant | best fitted feature | random |
|---|---|---|---|---|---|---|
| train | 24 | 2,4,6 | 1.000 | 0.625 | 1.000 (`bigram_multiset`) | 0.5 |
| held-out, seen lengths | 120 | 2,4,6 | 1.000 | 0.567 | 0.858 (`open_count_and_length`) | 0.5 |
| **held-out, unseen lengths** | 724 | 8,10,12,14,16 | **1.000** / 0.9986 | 0.548 | 0.648 (`first_and_last`) | 0.5 |

The two program figures are the validation-reranked conforming member and the
one `enumerate_fit` returns by declaration order; they differ only at length 16,
and the next subsection is why.

Baselines are the accuracy of the *optimal* predictor from a stated feature
fitted on the training episodes: constant, length, first symbol, last symbol,
first-and-last, bigram multiset, and (open count, length). The two that reach
1.000 on training collapse to the constant, 0.548, on unseen lengths -- they are
lookup tables over structures that split does not contain. The typed program does
not collapse. The unseen-length split shares **0%** of its strings with training
(`splits.json`); the seen-length split shares 68%, which is why it is the weaker
of the two tests and is reported separately.

### Non-uniqueness, and the tie-break that is wrong again

10 of 45,375 programs conform on the training episodes, so a conforming program
is not the right one. `conforming_sweep.py` scores all ten on held-out structure:

| conforming family | count | unseen lengths 8-14 | **length 16** |
|---|---|---|---|
| `symbols = length - 101`, steps +-1 or +-2, rule `eq(total, 0)` | 4 | 1.000 | **1.000** |
| shifted: `length - 99` or `- 100`, rule `eq(total, 2)` / `eq(total, 1)` / ... | 6 | 1.000 | **0.500** |

The shifted family counts one or two positions past the string, each contributing
a constant, and compensates in the rule -- exactly right until L = 16, where the
16-position scaffold has no positions to spare. `enumerate_fit` ranks by
declaration order and returns a shifted member. **Requiring exactness on a
validation split that contains an unseen length is what separates them.** This
reproduces findings section 14's tie-break result on language, on the first
attempt, without looking for it.

### How far the reranked program reaches

`difficulty_probe.py` renders `difficulty = 1.0` episodes (nesting spans the
lesson widens to lengths 18-26) in the lesson's own template -- an
off-distribution diagnostic, since `generators/language/generator.py` does not
plumb `difficulty` (diff D3).

| string length | 6 | 8 | 10 | 12 | 14 | **16** | 18 | 20 | 22 | 24 |
|---|---|---|---|---|---|---|---|---|---|---|
| reranked program | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | **1.000** | 0.542 | 0.288 | 0.364 | 0.500 |
| order-ranked program | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.220 | 0.333 | 0.538 | 0.591 | 0.000 |

1.000 at every length up to the scaffold's **declared 16-position capacity**,
including a difficulty setting never trained on, and chance beyond it. Depth is
generalized up to a declared capacity, not unboundedly -- findings section 15's
conclusion, reproduced here.

## 5. The differentiable path, measured on the same spaces

Every gradient number is reported beside the enumeration of the identical
candidate space. `SoftProgram` zero-initializes all choice logits (finding P2),
so explicit initialization noise (sd 0.01) was added to obtain seed variation.

### The shipped surrogates at this task's operating distances (`surrogate.json`)

| measurement | value |
|---|---|
| `eq` at \|a-b\| = 1, i.e. `'('` vs `')'`, tau=1 | 0.3679 -- **alive** |
| `eq` at \|a-b\| = 10 | 3.78e-44 |
| `eq` at \|a-b\| = 11 | **exactly 0.0** (findings section 16's threshold confirmed) |
| `eq` at \|a-b\| = 44 (a template byte against `'('`) | exactly 0.0 |
| `index` kernel mass on the addressed byte, 128 positions, tau=1 | 0.5641 |
| ... on the two neighbours | 0.4151 |
| separation of `'('` from `')'` after that blur | 0.954 vs 0.542 |
| `lt` (`sigmoid`) gradient at distance 17 | **exactly 0.0** |
| `lt` gradient at this task's operating distance 48 | exactly 0.0 |
| `lt` gradient at distance 48, tau = 128 | 1.9e-03 |

So on *this* task the `eq` surrogate is not dead at the distance that decides the
answer -- the two symbols differ by 1 -- and it **is** exactly dead for 235 of the
256 candidate byte constants.

### Stage A by gradient descent -- 0 of 32 runs, against enumeration's 3.5 s

| arm | temperatures | conformant / 8 | exact train accuracy | byte chosen |
|---|---|---|---|---|
| shipped | tau = 1 | **0/8** | 0.528 | 39, 41, 68, 88, 108, 110 |
| eq256 | tau = 2^bits = 256 at `eq` (section 16's derived fix) | **0/8** | 0.523 | 47, 48 |
| index0.1 | tau = 0.1 at `index` (sharpens M3) | **0/8** | 0.614 | 32, 41, 115 |
| both | | **0/8** | 0.523 | 33, 34, 36, 57, 60-62 |
| **enumeration** | -- | **1/1, unique, 3.5 s** | 1.000 | **40** |

Two mechanisms, measured rather than inferred:

- **Section 16's fix cannot be applied here, and D2 is why.** Raising `eq`'s
  temperature to the derived 256 widens the surrogate *and* flattens the same
  node's 256-way choice softmax, because `SoftProgram` uses one temperature for
  both. Choice gradients fall from 9e-2 to 2e-5 and all 8 seeds collapse to the
  same wrong byte. **This is the first task where D2 is the binding constraint
  rather than a note**, which makes separating the two temperatures a
  prerequisite for the fix rather than an independent cleanup.
- **Zero gradient is not the failure mode; uninformative gradient is.** No
  candidate had exactly-zero *choice* gradient in any run, although 235 of the 256
  `eq` surrogates are exactly 0.0: the softmax gives a dead candidate a nonzero
  logit gradient through the mixture mean, carrying no information about whether
  it is right. Findings section 4's connectivity-guard defeat, in a new place.

### Where the descent direction points at initialization

| choice | candidates | reference picked | chance | reference's mean rank | chance mean rank |
|---|---|---|---|---|---|
| value (`eq` byte constant) | 256 | 0/8 | 0.004 | **25.9** | 127.5 |
| address (`base` offset) | 41 | 0/8 | 0.024 | **21.5** | 20.0 |

The value choice is informative but not decisive -- the reference sits in the top
10% by steepest descent and is never first. The address choice is **exactly at
chance**. That is the address/value asymmetry of findings sections 11, 14, 15 and
16, measured on text with no image involved, in the weakened form section 16
established: harder, not broken.

### Stage B by gradient descent -- a third dead surrogate, on `lt`

| arm | conformant on training | mean held-out unseen | `symbols` choice gradient |
|---|---|---|---|
| shipped, 6 seeds x 300 steps | **0/6** | 0.438 (below the 0.548 constant) | **exactly 0.0, every run** |
| `lt` temperature 128, 6 seeds | **0/6** | 0.452 | 1.4e-09 to 8.0e-08 |
| **enumeration** | **10 conforming, exhaustive, 363 s** | **1.000** (0.9986 order-ranked) | -- |

The `symbols` choice -- which constant relates prompt length to symbol count --
receives exactly zero gradient because `lt`'s sigmoid is saturated at the diffuse
mixture's operating distance of about 48. Widening it to tau = 128 revives the
gradient but leaves it 5-7 orders of magnitude below the rule choice's 2.5e-2,
and no run conforms.

So `eq` dies at 11, `lt` dies at 17, and `index`'s kernel keeps only 0.564 of its
mass on the addressed element -- three defects with one cause: a temperature of 1
against integer carriers whose declared range is 2^8 or 2^32. One derived rule
(scale the surrogate temperature by the carrier's declared range) addresses all
three, and D2 blocks it.

## 6. What the program actually learned: a certificate, not an accuracy

The exported program was interrogated on strings the generator cannot emit,
written into the lesson's own template (`certificate.py`, diagnostic only):

| probe group | agrees with Dyck membership | agrees with `#( == #)` |
|---|---|---|
| 8 count-preserving non-Dyck strings (`)(`, `))((`, `)))(((`, ...) | 0/8 | 8/8 |
| 6 Dyck words | 6/6 | 6/6 |
| **total** | **0.429** | **1.000** |

It answers `yes` for `)(`. It has no stack, and it does not need one, because the
generator never asks for one. That is the difference between "solved the lesson"
and "has the capability the lesson is named after", made checkable.

## 7. Cost of the complete inference path

| measurement | value |
|---|---|
| exported program | 84 live nodes, 16 frozen-module call sites |
| `execution_cost` | 148.0 |
| `description_bits` | 4,043,552 -- serialized JSON length, the measure findings section 3 already refutes. The learned *content* is one byte constant, one offset, two step values and one rule: about 30 bits. |
| batch-one latency, exact path | **8.65 ms per episode** |
| undecomposed search (stage A and B choices free together) | 476,256,000 programs, projected **50.7 days** at the measured 9.2 ms/program |

Staging is what makes this reachable: 3.5 s + 363 s against 50.7 days, a factor
of about 1.2e4. Findings section 14's positional-reuse result, on language.

## 8. What generalizes, and what does not

**Generalizes.** Unseen string lengths and unseen nesting depths with 0% string
overlap: 1.000 (0.9986 for the order-ranked member) against a 0.548 constant and
a 0.648 best fitted feature; and
1.000 at every length up to the declared capacity under a `difficulty` setting
never trained on. The stage-A module, supervised at 44 positions of 12 episodes,
is exact at every position of every held-out episode, because its address is
computed rather than chosen.

**Does not generalize.** Beyond the scaffold's 16-position declared capacity the
program is at chance (0.43). It is a counting rule and is provably wrong on Dyck
words the generator cannot produce. And nothing here transfers to the other 178
lessons: section 2 measures only 6 of 179 prompts as byte-predictable at a fixed
offset.

## 9. What blocks more, in order

1. **`role="byte"` leaves no aggregation over text.** `sum`, `mean`, `count` over
   a value, ordering comparisons and all arithmetic are type-illegal on bytes,
   and there is no tuple-to-set conversion, so `map`/`filter`/`member` cannot
   reach text either. The only route into arithmetic is `pack` over an explicitly
   constructed tuple of indexed bytes, which is `gradient="none"`. Any lesson
   needing a scan over a variable-length field is out of reach without the
   recurrence of ARCHITECTURE section 4 (which `tcn/search.py` cannot score at
   all) or a set-shaped second view of the text.
2. **Realization variance.** 173 of 179 lessons move their content with the
   vocabulary. Constant addresses cannot follow that; relaxed addresses are at
   chance.
3. **D2, now binding.** One temperature per node couples surrogate width to the
   choice softmax, so the derived surrogate fix cannot be applied where needed.
4. **Three saturated surrogates from one cause**: `eq` at 11, `lt` at 17, and
   `index` kernel locality, all from tau = 1 against wide integer carriers.
5. **`construction` is not uniformly a probe**: in 39 of 179 lessons it does not
   determine the answer, and in `finite_state_language` it is worth exactly
   nothing over the majority class.
6. **The benchmark.** `context_free_language` does not exercise a stack. Before
   any recursion claim is made on this generator, its negatives need to include
   count-preserving permutations.

## 10. Core changes proposed as diffs, none applied

**D1 -- separate the surrogate temperature from the choice temperature.**
Findings section 16's D2, promoted from a note to a blocker by section 5 above.

```diff
--- a/tcn/learning.py
+++ b/tcn/learning.py
@@ class SoftProgram(nn.Module):
         self.temperatures={n.name:1. for n in program.nodes}
+        # The candidate softmax and the operator relaxation are different knobs.
+        # Sharing one temperature means a surrogate cannot be widened without
+        # flattening that node's choice distribution.
+        self.surrogate_temperatures={n.name:1. for n in program.nodes}
@@ def forward
-            ys=[relaxed(self.registry,c.operator,[values[s] for s in c.sources],tau) for c in n.candidates]
+            sur=self.surrogate_temperatures.get(n.name,tau)
+            ys=[relaxed(self.registry,c.operator,[values[s] for s in c.sources],sur) for c in n.candidates]
```

Default behaviour unchanged (both dicts start at 1.0); every existing number
reproduces exactly.

**D2 -- derive the surrogate temperature from the declared carrier.** With D1 in
place, default `surrogate_temperatures` to the carrier's declared range for
integer inputs rather than 1.0 -- section 16's rule for `eq`, generalized to
`lt`/`le`/`gt`/`ge` and to `index`, since section 5 measures all three failing
for the same reason. Derived, not tuned. It changes recorded numbers, so it
belongs behind an explicit flag until re-measured.

**D3 -- plumb `difficulty` through the language generator.** Every lesson accepts
`difficulty` and uses it to widen its axes (`context_free_language` reads
`ctx.span((1,4),(3,7))` for nesting depth; verified here to take string lengths
from a 2-16 range to 6-26). `generators/language/generator.py` passes only
`lesson` and `language`, so the one knob that configures these lessons up or down
is unreachable from a TCN configuration -- which is why section 4's difficulty
probe had to go around the generator.

```diff
--- a/generators/language/generator.py
+++ b/generators/language/generator.py
@@ def initialize
-        lesson=configuration.get('lesson','unification');language=configuration.get('language','english')
-        ex=get(lesson).example(seed=address.rng().randrange(2**31),language=language)
+        lesson=configuration.get('lesson','unification');language=configuration.get('language','english')
+        difficulty=configuration.get('difficulty')
+        ex=get(lesson).example(seed=address.rng().randrange(2**31),language=language,difficulty=difficulty)
```

**D4 -- a set-shaped second view of text**, exactly as `generators/logic` gained
`gates` in findings section 15: `set[(index, byte)]` at a declared capacity,
emitted only when configured. `role="byte"` bars every aggregation from the tuple
view; a set view makes `member`, `filter` and `map` legal over positions without
adding an operator or a domain primitive. This is the change that would open the
other 173 lessons, and it is the one worth measuring next.

**D5 -- `context_free_language` needs count-preserving negatives.** Its negatives
are single-character flips, which always break the count. A negative branch that
permutes a balanced string, or swaps a matched pair, would make the lesson test
what it claims. Left as a proposal because the generator is concurrently owned.

## 11. Limitations of this track

- One lesson of 179. The choice was forced by section 2, and the certificate in
  section 2 is the general claim; the trained result is not.
- The positional application (16 call sites), the accumulation chain and the
  masking shape are **declared scaffold**, as in findings sections 14 and 15. What
  was **searched** is the byte constant over the full alphabet, the field offset,
  the length relation, the step values and the rule -- 10,496 x 45,375 programs.
- Enumeration is the method that worked. The gradient path was run on every
  space and is reported at 0/44, which is a measurement of the relaxations named
  in section 5, not of the tasks.
- The crystallization scheduler was not exercised: stage A's export is a hardened
  selection and stage B's is a discrete pick. Findings section 1 already records
  the scheduler as contributing nothing at working budgets, and nothing here
  gives a reason to revisit that.
- `difficulty` probes and the Dyck certificate use prompts assembled in the
  lesson's own template rather than drawn from the generator; both are labelled
  diagnostics and neither is reported as an accuracy on the lesson.
