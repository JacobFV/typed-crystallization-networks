# Lesson audit: how much of the language curriculum is answerable without the lesson

179 executable lessons, a battery of sixteen cheap exploits fitted on one set of
seeds and scored on a disjoint one, and a null control that says how much of a
score is just the maximum over sixteen fitted predictors.

**The finding.** 63 of 179 lessons admit a cheap heuristic that beats the
lesson's own floor by more than the selection band, and 14 of them are *solved*
by one — seven at exactly 1.000, two more above 0.90, five between 0.80 and 0.90. The dominant mechanism is
not subtle: in 31 of the 63 the winning exploit is a copy — the answer is a
token at a fixed offset, the option mentioned last, or the option that occurs
most. `context_free_language` is the case the language-capability track already
found (FINDINGS section 19); it is not the worst one.

**What was done about it.** The 14 solved lessons were re-drawn. Their best
cheap exploit falls from a mean of 0.934 to 0.394, none is above 0.63, and every
one still answers exactly from its own construction. The pre-audit draw is
preserved bit-for-bit as `hardening="none"` and checked against a recorded
fixture over 179 lessons x 6 configurations x 4 seeds; the default draw changes,
deliberately, in exactly those 14 lessons plus the 2 that compose them.

| | before | after |
|---|---|---|
| lessons solved by a cheap exploit (>= 0.90) | **9** | **0** |
| lessons at >= 0.80 | 14 | 0 |
| lessons exploitable (excess over null >= 0.10) | **63** | **54** |
| lessons exploitable (excess >= 0.30) | 24 | 14 |
| winning exploit is a copy / surface / distractor / memorisation | 31 / 18 / 9 / 5 | 27 / 16 / 8 / 3 |

Everything here was produced with `tcn/` unmodified. The changes are confined to
`generators/language/`.

---

## 0. Method, cost, and what "oracle" means

`battery.py` holds the exploits, `audit.py` drives them, `report.py` ranks,
`stream_digest.py` is the equivalence check.

**Episodes.** Per lesson: 600 training seeds (0-599) and 400 test seeds
(1,000,000-1,000,399). Every exploit is *fitted* on the training episodes and
*scored* on the test episodes, so a lookup table gets credit only for what
transfers.

| run | episodes | wall clock | workers |
|---|---|---|---|
| pre-fix audit | 179 x 1,000 = 179,000 | 198 s | 8 |
| pre-fix null control | 179,000 | 220 s | 8 |
| post-fix audit | 179,000 | 90 s | 8 |
| post-fix null control | 179,000 | 90 s | 8 |
| stream equivalence, 3 languages x 4 difficulties x 8 seeds | 17,184 x 3 regimes | 11 s each | 8 |
| `tests/test_language_hardening.py` | ~22,000 | 18 s | 1 |

Total: about 750,000 episodes in under 11 minutes of wall clock on a loaded
host. Episode generation is 0.7 ms median per lesson, so seeds are cheap and
the audit is run at 1,000 of them rather than at 30.

**Oracle.** The answer is computed from the construction, so the true answer
function is exact by definition. What is *measurable* is the best accuracy
available to anything that reads only the prompt: the column `oracle` is
`1 - P(two seeds produce the same prompt with different answers)`. It is 1.000
in 176 of 179 lessons and 0.975-0.998 in the other three
(`long_range_agreement`, `symbol_discrimination`, `expression_eval`), which is
the sampling-noise version of the language track's "answer is determined by
`text` in all 179 lessons" — those three repeat a prompt often enough that 1,000
seeds catch a collision.
`gap = oracle - best cheap exploit` is therefore how much of the lesson the
exploits leave on the table.

**Floor.** `max(majority share, mean 1/|choices|)` — what knowing nothing gets,
per lesson, computed from the episodes themselves.

**Null control.** Taking a maximum over sixteen fitted predictors inflates the
best score even when nothing is learnable. Every lesson was re-run with each
answer replaced by a uniformly random member of *its own* choice set, so nothing
is predictable by construction. Over 179 lessons the null lift is mean 0.020,
p95 0.048, **max 0.060**. `excess = best - floor - (null best - null floor)` is
the number to read: it is above the entire null band at 0.10.

### The battery

| exploit | what it fits |
|---|---|
| `constant` | the most frequent answer string in training |
| `const_choice_index` | the best fixed index into the episode's own choice list, from either end |
| `choice_by_length` | the r-th longest or shortest option, r fitted |
| `choice_length_outlier` | the option whose length is furthest from the others' median |
| `choice_global_prior` | the option that was most often the answer in training |
| `choice_in_observation` | the option that occurs earliest / latest / most / least / not at all in the prompt |
| `copy_salient_token` | the first / last / longest / shortest / commonest / rarest token of the prompt |
| `copy_fixed_offset` | the token at a fixed index from the left or the right, index fitted over 48 |
| `copy_near_keyword` | the token at a fixed offset from a fixed keyword, both fitted |
| `prompt_length` | prompt length -> majority answer, with a backoff |
| `byte_at_offset` | one byte at the best fixed offset -> majority answer (the language track's control) |
| `bag_of_chars` | nearest class centroid over character counts |
| `char_count_tree` | a depth-5 decision tree over character counts *and their pairwise differences* |
| `nearest_neighbour` | 1-NN over training prompts by character-count cosine |
| `train_lookup` | exact prompt -> training answer; pure memorisation |
| `choice_ranker` | an averaged perceptron over per-option surface features (length, occurrences, position, digits, overlap) |

Predictors that emit an option index rather than a string are applied to the
episode's own choice list, which is what makes them meaningful on the lessons
that invent their vocabulary per episode.

Two exploits are worth naming. `char_count_tree` carries pairwise *differences*
of character counts, so `#( - #) == 0` is one feature and not an unreachable
interaction — that is the family that defeats a bracket lesson. `choice_ranker`
is not a model of the lesson; it is a model of how the distractors were drawn,
and when it wins the answer is separable from its distractors on surface
statistics alone.

---

## 1. Mechanisms

Grouping the 63 exploitable lessons by *why*, from reading the generators of the
worst ones:

### A. The answer is a copy, and its position is fixed by the construction (31 lessons)

The single largest group. In 144 of 179 lessons the answer is literally a
substring of the prompt — which is fine, most of these lessons are reference or
retrieval tasks — but in 31 of them *which* substring is decided by something
positional rather than by the semantics.

* **`ellipsis` (1.000).** The docstring says "distractor clauses precede the
  antecedent so the answer is never the only verb in the discourse". They do,
  and that puts the antecedent **last**: `clauses.append(antecedent)` after the
  distractors are shuffled. The answer is therefore always the most recently
  mentioned verb or object, and "take the option that occurs latest" is exact.
  One exploit was closed and another opened in the same line.
* **`center_embedding` (1.000).** `N1 N2 N3 V3 V2 V1`, and the query is always
  `verb_of(N1)` — whose verb is `V1`, the last word. A trailing adverb was added
  so that "the last token" fails; "the last *option*" was not covered.
* **`tree_to_sequence` (1.000).** The query is `first_leaf` and the tree is
  rendered in order, so the answer is the first symbol printed. The question is
  the exploit.
* **`symbol_equivalence` (1.000).** One `means(alias, colour)` fact, printed
  after the scene: the answer is both the last colour named and the commonest.
* **`language_culture` (0.907).** The surviving word is the last one printed in
  the transmission log, so a token at a fixed offset from the end answers it.
* **`discourse_state` (0.750), `multi_perspective_modeling` (0.670),
  `knowledge_refactoring` (0.522), `deception_detection` (0.550)** — same shape:
  the entity the answer refers to is the last mentioned, or sits at a fixed
  offset from a template word.

### B. Negatives drawn to correlate with a surface feature (the bracket case)

* **`context_free_language` (1.000).** Negatives are single-character flips,
  which always change the bracket count, so `#( == #)` and `balanced` agree on
  every seed. `char_count_tree` reaches 1.000 and so does `nearest_neighbour`.
  This is FINDINGS section 19's D5, measured again from the other direction.
* **`scope_ambiguity` (0.785).** `neither` is realized by silencing one agent,
  so it has fewer `reads` edges than the other two labels; `both` shares a book,
  so one book occurs once per agent. The three labels have three different
  character-count signatures.
* **`paradigm_shift` (0.823).** Each falsified assumption leaves its own mark in
  the characters: `positivity` prints a minus sign, `boundedness` prints
  three-digit readings, `small_steps` prints a large gap. A depth-5 tree over
  character counts names the target without reading the framework.
* **`presupposition` (0.868).** The label is drawn first and the polarity chosen
  to realize it, so `asserted` implies `affirm` and `denied` implies `negate`;
  with no other utterance in the episode, the polarity word is a global feature.
* **`underspecification_reasoning` (1.000).** Six colours, three on cubes and
  three on boxes, leaves the `spare` distractor list **empty on every seed**, so
  the number of printed `fits` lines *is* the answer. The lesson prints its own
  answer as a tally.
* **`symbol_discrimination` (0.802).** The scale always runs from 0 and the
  boundary always sits in 3-7, so the queried number alone predicts the label
  0.80 of the time and the shown examples are decorative.
* **`nesting_depth_compare` (0.812).** The docstring says both strings are
  "padded with flat pairs so *length* carries no signal". Independent padding
  does not equalize anything: the deeper string is still the longer one, and
  "where does the word `right` start, as a fraction of the prompt" recovers it.

### C. Distractors that are never plausible (9 lessons)

`choice_ranker` wins when the correct option differs from its distractors in
some surface statistic.

* **`unification` (1.000).** `parent(X, bob)` against `parent(alice, bob)`: the
  binding is the one name printed **once**; every other name is printed twice,
  in the pattern and in the fact. A perceptron over occurrence counts is exact.
* **`external_memory_design` (0.730).** The composite design covers two fields
  and so contributes two `indexes` lines instead of one; it is also usually the
  cheapest. "The option that occurs most" is 0.73.
* **`symmetry_reasoning` (0.660), `social_convention_learning` (0.630),
  `compiler_construction` (0.522)** — same family.

### D. Episode spaces small enough to memorise (5 lessons)

* **`parse_depth` (0.998).** `_balanced(rng, depth)` emits four or five strings
  per depth, so **85% of prompts recur across seeds** and a lookup over training
  prompts is exact. It also makes the string longer when it is deeper, so
  `prompt_length` alone scores 0.615.
* **`context_free_language`** repeats 71% of prompts for the same reason.
* **`tree_to_sequence`** repeats 52% and **`unification`** 43%, which is why
  `train_lookup` runs close behind the winning exploit in both.
* **`thematic_roles`** (0.608, repeat 21%), **`expression_eval`** (23%) and
  **`variable_binding`** (15%) are the tail of the same problem.

### E. Floors that are not floors

Ten binary lessons sit at a majority share of 0.50-0.54, which is what the
catalogue's own `verify` gate checks and passes. That is fine. What `verify`
does not check is anything in groups A-D, which is why a lesson can pass
admission and still be answerable by counting characters.

---

## 2. The full table, worst first

`excess` is the number to rank on: best cheap exploit, minus the lesson's floor,
minus that lesson's own null-control lift. The whole null band is below 0.060,
so **excess >= 0.10 is exploitable** and excess >= 0.30 means most of the lesson
is surface. `copy` is the fraction of episodes whose answer is a substring of the
prompt; `repeat` is the fraction of prompts that recur across seeds.

| # | lesson | oracle | best cheap exploit | score | gap | floor | excess | family | #ans | H(bits) | copy | repeat |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `ellipsis` | 1.000 | `choice_in_observation` | **1.000** | 0.000 | 0.167 | +0.800 | copy | 12 | 3.58 | 1.00 | 0.00 |
| 2 | `tree_to_sequence` | 1.000 | `copy_near_keyword` | **1.000** | 0.000 | 0.182 | +0.800 | copy | 6 | 2.58 | 1.00 | 0.52 |
| 3 | `symbol_equivalence` | 1.000 | `choice_in_observation` | **1.000** | 0.000 | 0.188 | +0.797 | copy | 6 | 2.58 | 1.00 | 0.00 |
| 4 | `unification` | 1.000 | `choice_ranker` | **1.000** | 0.000 | 0.182 | +0.782 | distractor | 6 | 2.58 | 1.00 | 0.43 |
| 5 | `parse_depth` | 1.000 | `train_lookup` | **0.998** | 0.003 | 0.223 | +0.775 | memorisation | 5 | 2.32 | 0.00 | 0.85 |
| 6 | `underspecification_reasoning` | 1.000 | `char_count_tree` | **1.000** | 0.000 | 0.223 | +0.762 | surface-statistic | 5 | 2.32 | 0.39 | 0.00 |
| 7 | `language_culture` | 1.000 | `copy_fixed_offset` | **0.907** | 0.092 | 0.250 | +0.625 | copy | 6 | 2.56 | 1.00 | 0.00 |
| 8 | `center_embedding` | 1.000 | `choice_in_observation` | **1.000** | 0.000 | 0.361 | +0.608 | copy | 6 | 2.58 | 1.00 | 0.01 |
| 9 | `presupposition` | 1.000 | `char_count_tree` | **0.868** | 0.133 | 0.268 | +0.593 | surface-statistic | 4 | 2.00 | 1.00 | 0.29 |
| 10 | `next_symbol` | 1.000 | `copy_near_keyword` | **0.863** | 0.138 | 0.285 | +0.562 | copy | 4 | 2.00 | 1.00 | 0.41 |
| 11 | `paradigm_shift` | 1.000 | `char_count_tree` | **0.823** | 0.177 | 0.260 | +0.532 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 12 | `external_memory_design` | 1.000 | `choice_in_observation` | **0.730** | 0.270 | 0.215 | +0.530 | copy | 5 | 2.32 | 1.00 | 0.00 |
| 13 | `context_free_language` | 1.000 | `nearest_neighbour` | **1.000** | 0.000 | 0.500 | +0.500 | memorisation | 2 | 1.00 | 0.00 | 0.71 |
| 14 | `symmetry_reasoning` | 1.000 | `choice_ranker` | **0.660** | 0.340 | 0.185 | +0.468 | distractor | 6 | 2.58 | 1.00 | 0.00 |
| 15 | `discourse_state` | 1.000 | `copy_fixed_offset` | **0.750** | 0.250 | 0.250 | +0.460 | copy | 10 | 3.29 | 1.00 | 0.00 |
| 16 | `algorithm_analysis` | 1.000 | `byte_at_offset` | **0.770** | 0.230 | 0.338 | +0.420 | surface-statistic | 5 | 1.85 | 1.00 | 0.00 |
| 17 | `explanation_repair` | 1.000 | `choice_in_observation` | **0.690** | 0.310 | 0.250 | +0.407 | copy | 395 | 8.62 | 1.00 | 0.00 |
| 18 | `scope_ambiguity` | 1.000 | `char_count_tree` | **0.785** | 0.215 | 0.362 | +0.390 | surface-statistic | 3 | 1.58 | 0.00 | 0.00 |
| 19 | `multi_perspective_modeling` | 1.000 | `copy_near_keyword` | **0.670** | 0.330 | 0.273 | +0.383 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 20 | `proof_compression` | 1.000 | `choice_in_observation` | **0.670** | 0.330 | 0.250 | +0.383 | copy | 376 | 8.52 | 1.00 | 0.00 |
| 21 | `variable_binding` | 1.000 | `char_count_tree` | **0.560** | 0.440 | 0.182 | +0.372 | surface-statistic | 6 | 2.57 | 1.00 | 0.15 |
| 22 | `emergence_discovery` | 1.000 | `copy_near_keyword` | **0.625** | 0.375 | 0.263 | +0.362 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 23 | `social_convention_learning` | 1.000 | `choice_ranker` | **0.630** | 0.370 | 0.250 | +0.352 | distractor | 6 | 2.58 | 1.00 | 0.00 |
| 24 | `teaching` | 1.000 | `choice_in_observation` | **0.615** | 0.385 | 0.250 | +0.330 | copy | 398 | 8.63 | 1.00 | 0.00 |
| 25 | `knowledge_refactoring` | 1.000 | `copy_near_keyword` | **0.522** | 0.477 | 0.233 | +0.285 | copy | 5 | 2.31 | 1.00 | 0.00 |
| 26 | `algorithm_discovery` | 1.000 | `bag_of_chars` | **0.497** | 0.502 | 0.200 | +0.278 | surface-statistic | 8 | 2.99 | 1.00 | 0.00 |
| 27 | `deception_detection` | 1.000 | `copy_near_keyword` | **0.550** | 0.450 | 0.250 | +0.275 | copy | 6 | 2.57 | 1.00 | 0.00 |
| 28 | `instruction_following_micro` | 1.000 | `choice_in_observation` | **0.472** | 0.527 | 0.167 | +0.273 | copy | 12 | 3.55 | 1.00 | 0.00 |
| 29 | `nesting_depth_compare` | 1.000 | `choice_ranker` | **0.812** | 0.188 | 0.522 | +0.268 | distractor | 2 | 1.00 | 1.00 | 0.04 |
| 30 | `representation_selection` | 1.000 | `byte_at_offset` | **0.560** | 0.440 | 0.268 | +0.265 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 31 | `instruction_composition` | 1.000 | `choice_in_observation` | **0.463** | 0.537 | 0.167 | +0.260 | copy | 12 | 3.56 | 1.00 | 0.00 |
| 32 | `knowledge_update` | 1.000 | `choice_ranker` | **0.443** | 0.557 | 0.182 | +0.260 | distractor | 7 | 2.79 | 0.97 | 0.00 |
| 33 | `compiler_construction` | 1.000 | `choice_ranker` | **0.522** | 0.477 | 0.250 | +0.250 | distractor | 190 | 7.27 | 0.01 | 0.00 |
| 34 | `symbol_discrimination` | 0.984 | `byte_at_offset` | **0.802** | 0.181 | 0.540 | +0.245 | surface-statistic | 2 | 0.99 | 1.00 | 0.04 |
| 35 | `world_model_synthesis` | 1.000 | `choice_in_observation` | **0.515** | 0.485 | 0.285 | +0.245 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 36 | `knowledge_gap_detection` | 1.000 | `byte_at_offset` | **0.502** | 0.497 | 0.268 | +0.223 | surface-statistic | 4 | 2.00 | 0.00 | 0.00 |
| 37 | `pronoun_coreference` | 1.000 | `char_count_tree` | **0.728** | 0.273 | 0.500 | +0.198 | surface-statistic | 6 | 2.57 | 1.00 | 0.02 |
| 38 | `architecture_composition` | 1.000 | `copy_near_keyword` | **0.365** | 0.635 | 0.165 | +0.198 | copy | 8 | 2.97 | 1.00 | 0.00 |
| 39 | `logic_discovery` | 1.000 | `copy_near_keyword` | **0.453** | 0.547 | 0.268 | +0.185 | copy | 4 | 2.00 | 1.00 | 0.05 |
| 40 | `long_horizon_projects` | 1.000 | `choice_in_observation` | **0.323** | 0.677 | 0.118 | +0.180 | copy | 394 | 8.61 | 1.00 | 0.00 |
| 41 | `belief_state` | 1.000 | `copy_near_keyword` | **0.460** | 0.540 | 0.250 | +0.177 | copy | 6 | 2.58 | 1.00 | 0.00 |
| 42 | `contract_reasoning` | 1.000 | `prompt_length` | **0.410** | 0.590 | 0.223 | +0.172 | surface-statistic | 5 | 2.32 | 0.20 | 0.00 |
| 43 | `counting_quantifier` | 1.000 | `byte_at_offset` | **0.703** | 0.297 | 0.517 | +0.163 | surface-statistic | 2 | 1.00 | 0.00 | 0.00 |
| 44 | `question_answering` | 1.000 | `choice_in_observation` | **0.427** | 0.573 | 0.230 | +0.158 | copy | 27 | 4.54 | 1.00 | 0.00 |
| 45 | `expression_eval` | 0.998 | `train_lookup` | **0.395** | 0.603 | 0.200 | +0.155 | memorisation | 55 | 5.08 | 0.12 | 0.23 |
| 46 | `sequence_copy` | 1.000 | `copy_near_keyword` | **0.362** | 0.637 | 0.188 | +0.152 | copy | 6 | 2.58 | 1.00 | 0.02 |
| 47 | `tool_construction` | 1.000 | `choice_in_observation` | **0.448** | 0.552 | 0.275 | +0.147 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 48 | `scientific_civilization` | 1.000 | `copy_fixed_offset` | **0.502** | 0.497 | 0.333 | +0.145 | copy | 383 | 8.56 | 1.00 | 0.00 |
| 49 | `document_world` | 1.000 | `choice_ranker` | **0.427** | 0.573 | 0.250 | +0.145 | distractor | 6 | 2.57 | 1.00 | 0.00 |
| 50 | `uncertain_symbolic_reasoning` | 1.000 | `nearest_neighbour` | **0.410** | 0.590 | 0.250 | +0.140 | memorisation | 23 | 3.97 | 1.00 | 0.01 |
| 51 | `research_program` | 1.000 | `copy_near_keyword` | **0.355** | 0.645 | 0.200 | +0.140 | copy | 6 | 2.58 | 1.00 | 0.00 |
| 52 | `goal_generation` | 1.000 | `copy_near_keyword` | **0.330** | 0.670 | 0.180 | +0.130 | copy | 6 | 2.58 | 1.00 | 0.00 |
| 53 | `minimum_description_learning` | 1.000 | `choice_ranker` | **0.380** | 0.620 | 0.270 | +0.128 | distractor | 4 | 2.00 | 1.00 | 0.00 |
| 54 | `interactive_reference` | 1.000 | `nearest_neighbour` | **0.357** | 0.642 | 0.223 | +0.122 | memorisation | 5 | 2.32 | 0.60 | 0.00 |
| 55 | `source_provenance` | 1.000 | `bag_of_chars` | **0.340** | 0.660 | 0.223 | +0.115 | surface-statistic | 5 | 2.32 | 0.00 | 0.00 |
| 56 | `latent_rule_discovery` | 1.000 | `choice_ranker` | **0.340** | 0.660 | 0.200 | +0.113 | distractor | 72 | 5.54 | 0.32 | 0.04 |
| 57 | `narrative_modeling` | 1.000 | `copy_near_keyword` | **0.295** | 0.705 | 0.180 | +0.113 | copy | 7 | 2.75 | 1.00 | 0.00 |
| 58 | `metalinguistic_reasoning` | 1.000 | `char_count_tree` | **0.390** | 0.610 | 0.268 | +0.110 | surface-statistic | 4 | 2.00 | 0.00 | 0.00 |
| 59 | `collective_theory_building` | 1.000 | `choice_in_observation` | **0.415** | 0.585 | 0.250 | +0.107 | copy | 114 | 6.65 | 0.98 | 0.00 |
| 60 | `ambiguity_preservation` | 1.000 | `char_count_tree` | **0.405** | 0.595 | 0.268 | +0.107 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 61 | `palindrome` | 1.000 | `char_count_tree` | **0.650** | 0.350 | 0.537 | +0.105 | surface-statistic | 2 | 1.00 | 0.00 | 0.06 |
| 62 | `norm_reasoning` | 1.000 | `copy_near_keyword` | **0.492** | 0.507 | 0.362 | +0.105 | copy | 3 | 1.58 | 1.00 | 0.00 |
| 63 | `self_model` | 1.000 | `byte_at_offset` | **0.667** | 0.333 | 0.535 | +0.102 | surface-statistic | 2 | 1.00 | 0.00 | 0.00 |
| 64 | `string_reversal` | 1.000 | `char_count_tree` | **0.323** | 0.677 | 0.210 | +0.092 | surface-statistic | 6 | 2.57 | 1.00 | 0.00 |
| 65 | `belief_revision` | 1.000 | `byte_at_offset` | **0.297** | 0.703 | 0.220 | +0.090 | surface-statistic | 5 | 2.32 | 1.00 | 0.00 |
| 66 | `invariance_discovery` | 1.000 | `byte_at_offset` | **0.407** | 0.593 | 0.278 | +0.087 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 67 | `multi_objective_reasoning` | 1.000 | `copy_near_keyword` | **0.357** | 0.642 | 0.285 | +0.085 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 68 | `interpreter_learning` | 1.000 | `choice_global_prior` | **0.350** | 0.650 | 0.250 | +0.080 | distractor | 51 | 4.54 | 0.66 | 0.00 |
| 69 | `thematic_roles` | 1.000 | `train_lookup` | **0.608** | 0.393 | 0.500 | +0.070 | memorisation | 6 | 2.58 | 1.00 | 0.21 |
| 70 | `analogy` | 1.000 | `choice_in_observation` | **0.263** | 0.738 | 0.193 | +0.070 | copy | 6 | 2.58 | 1.00 | 0.00 |
| 71 | `quantification` | 1.000 | `nearest_neighbour` | **0.583** | 0.417 | 0.502 | +0.068 | memorisation | 2 | 1.00 | 0.13 | 0.00 |
| 72 | `lexicon_induction` | 1.000 | `char_count_tree` | **0.407** | 0.593 | 0.343 | +0.065 | surface-statistic | 3 | 1.58 | 1.00 | 0.00 |
| 73 | `default_reasoning` | 1.000 | `byte_at_offset` | **0.432** | 0.568 | 0.367 | +0.062 | surface-statistic | 3 | 1.58 | 1.00 | 0.00 |
| 74 | `procedural_language` | 1.000 | `copy_fixed_offset` | **0.182** | 0.818 | 0.117 | +0.062 | copy | 10 | 3.30 | 0.79 | 0.00 |
| 75 | `theory_transfer` | 1.000 | `choice_ranker` | **0.362** | 0.637 | 0.282 | +0.060 | distractor | 4 | 1.99 | 1.00 | 0.00 |
| 76 | `lemma_invention` | 1.000 | `choice_in_observation` | **0.315** | 0.685 | 0.250 | +0.055 | copy | 377 | 8.53 | 1.00 | 0.00 |
| 77 | `compositional_reference` | 1.000 | `char_count_tree` | **0.300** | 0.700 | 0.255 | +0.052 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 78 | `institution_design` | 1.000 | `copy_near_keyword` | **0.357** | 0.642 | 0.275 | +0.050 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 79 | `translation` | 1.000 | `choice_ranker` | **0.233** | 0.767 | 0.167 | +0.048 | distractor | 387 | 8.58 | 1.00 | 0.00 |
| 80 | `source_reliability_learning` | 1.000 | `copy_near_keyword` | **0.347** | 0.652 | 0.250 | +0.045 | copy | 8 | 2.99 | 1.00 | 0.00 |
| 81 | `strategy_transfer` | 1.000 | `choice_in_observation` | **0.390** | 0.610 | 0.293 | +0.040 | copy | 394 | 8.61 | 1.00 | 0.00 |
| 82 | `civilization_simulator` | 1.000 | `copy_near_keyword` | **0.360** | 0.640 | 0.293 | +0.040 | copy | 384 | 8.56 | 1.00 | 0.00 |
| 83 | `value_learning` | 1.000 | `char_count_tree` | **0.292** | 0.708 | 0.278 | +0.040 | surface-statistic | 4 | 1.99 | 1.00 | 0.01 |
| 84 | `open_world_language` | 1.000 | `char_count_tree` | **0.427** | 0.573 | 0.367 | +0.037 | surface-statistic | 3 | 1.58 | 0.37 | 0.00 |
| 85 | `semantic_compression` | 1.000 | `bag_of_chars` | **0.338** | 0.662 | 0.270 | +0.037 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 86 | `curriculum_invention` | 1.000 | `prompt_length` | **0.300** | 0.700 | 0.273 | +0.037 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 87 | `long_range_agreement` | 0.975 | `const_choice_index` | **0.568** | 0.407 | 0.500 | +0.035 | constant | 12 | 3.57 | 0.00 | 0.00 |
| 88 | `deformalization` | 1.000 | `choice_ranker` | **0.278** | 0.723 | 0.258 | +0.035 | distractor | 4 | 2.00 | 1.00 | 0.00 |
| 89 | `definitions` | 1.000 | `copy_fixed_offset` | **0.295** | 0.705 | 0.263 | +0.033 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 90 | `language_design` | 1.000 | `choice_ranker` | **0.292** | 0.708 | 0.268 | +0.033 | distractor | 4 | 2.00 | 1.00 | 0.00 |
| 91 | `negation` | 1.000 | `char_count_tree` | **0.580** | 0.420 | 0.510 | +0.030 | surface-statistic | 2 | 1.00 | 0.49 | 0.00 |
| 92 | `symbol_grounding` | 1.000 | `byte_at_offset` | **0.318** | 0.682 | 0.270 | +0.028 | surface-statistic | 5 | 2.16 | 1.00 | 0.00 |
| 93 | `conservation_law_discovery` | 1.000 | `byte_at_offset` | **0.290** | 0.710 | 0.235 | +0.028 | surface-statistic | 5 | 2.31 | 1.00 | 0.00 |
| 94 | `formalization` | 1.000 | `nearest_neighbour` | **0.285** | 0.715 | 0.258 | +0.028 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 95 | `architecture_selection` | 1.000 | `nearest_neighbour` | **0.297** | 0.703 | 0.265 | +0.025 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 96 | `coalition_formation` | 1.000 | `nearest_neighbour` | **0.320** | 0.680 | 0.270 | +0.022 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 97 | `scientific_model_induction` | 1.000 | `choice_ranker` | **0.307** | 0.693 | 0.250 | +0.022 | distractor | 101 | 5.96 | 0.46 | 0.00 |
| 98 | `institution_learning` | 1.000 | `copy_near_keyword` | **0.302** | 0.698 | 0.263 | +0.022 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 99 | `hierarchical_planning` | 1.000 | `bag_of_chars` | **0.292** | 0.708 | 0.270 | +0.022 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 100 | `symbolic_generalist` | 1.000 | `choice_length_outlier` | **0.155** | 0.845 | 0.125 | +0.022 | distractor | 242 | 7.63 | 0.89 | 0.00 |
| 101 | `reflective_goal_reasoning` | 1.000 | `copy_near_keyword` | **0.340** | 0.660 | 0.278 | +0.020 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 102 | `open_ended_concept_discovery` | 1.000 | `choice_ranker` | **0.300** | 0.700 | 0.260 | +0.020 | distractor | 4 | 2.00 | 1.00 | 0.00 |
| 103 | `capability_estimation` | 1.000 | `char_count_tree` | **0.285** | 0.715 | 0.265 | +0.020 | surface-statistic | 4 | 1.99 | 1.00 | 0.00 |
| 104 | `cultural_evolution` | 1.000 | `byte_at_offset` | **0.338** | 0.662 | 0.295 | +0.015 | surface-statistic | 4 | 1.99 | 1.00 | 0.00 |
| 105 | `natural_language_bridge` | 1.000 | `nearest_neighbour` | **0.292** | 0.708 | 0.270 | +0.015 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 106 | `dsl_invention` | 1.000 | `choice_in_observation` | **0.285** | 0.715 | 0.263 | +0.015 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 107 | `program_synthesis` | 1.000 | `byte_at_offset` | **0.287** | 0.713 | 0.275 | +0.013 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 108 | `compressed_language` | 1.000 | `choice_ranker` | **0.285** | 0.715 | 0.250 | +0.013 | distractor | 6 | 2.57 | 1.00 | 0.00 |
| 109 | `explanation` | 1.000 | `nearest_neighbour` | **0.282** | 0.718 | 0.260 | +0.013 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 110 | `falsification` | 1.000 | `nearest_neighbour` | **0.282** | 0.718 | 0.270 | +0.013 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 111 | `ontology_alignment` | 1.000 | `const_choice_index` | **0.278** | 0.723 | 0.280 | +0.010 | constant | 4 | 2.00 | 1.00 | 0.00 |
| 112 | `multiscale_modeling` | 1.000 | `choice_global_prior` | **0.228** | 0.772 | 0.200 | +0.010 | distractor | 44 | 5.10 | 0.03 | 0.00 |
| 113 | `comparatives` | 1.000 | `choice_global_prior` | **0.455** | 0.545 | 0.418 | +0.007 | distractor | 4 | 1.93 | 1.00 | 0.00 |
| 114 | `strategy_discovery` | 1.000 | `nearest_neighbour` | **0.328** | 0.672 | 0.320 | +0.007 | memorisation | 6 | 2.47 | 0.72 | 0.00 |
| 115 | `proof_translation` | 1.000 | `char_count_tree` | **0.300** | 0.700 | 0.233 | +0.007 | surface-statistic | 6 | 2.40 | 1.00 | 0.00 |
| 116 | `problem_reformulation` | 1.000 | `copy_fixed_offset` | **0.297** | 0.703 | 0.265 | +0.007 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 117 | `self_repair` | 1.000 | `bag_of_chars` | **0.287** | 0.713 | 0.270 | +0.007 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 118 | `implicature` | 1.000 | `char_count_tree` | **0.247** | 0.752 | 0.230 | +0.007 | surface-statistic | 6 | 2.48 | 1.00 | 0.00 |
| 119 | `goal_inference` | 1.000 | `const_choice_index` | **0.295** | 0.705 | 0.273 | +0.005 | constant | 4 | 1.99 | 1.00 | 0.00 |
| 120 | `abstraction_ladder` | 1.000 | `choice_ranker` | **0.287** | 0.713 | 0.287 | +0.005 | distractor | 4 | 1.99 | 1.00 | 0.00 |
| 121 | `ontology_revision` | 1.000 | `bag_of_chars` | **0.287** | 0.713 | 0.270 | +0.005 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 122 | `temporal_language` | 1.000 | `choice_in_observation` | **0.242** | 0.757 | 0.200 | +0.005 | copy | 8 | 2.99 | 1.00 | 0.00 |
| 123 | `self_error_diagnosis` | 1.000 | `char_count_tree` | **0.200** | 0.800 | 0.182 | +0.005 | surface-statistic | 6 | 2.58 | 1.00 | 0.00 |
| 124 | `finite_state_language` | 1.000 | `prompt_length` | **0.515** | 0.485 | 0.505 | +0.003 | surface-statistic | 2 | 1.00 | 0.94 | 0.00 |
| 125 | `few_shot_language_learning` | 1.000 | `choice_global_prior` | **0.372** | 0.627 | 0.333 | +0.003 | distractor | 6 | 2.57 | 1.00 | 0.00 |
| 126 | `negotiation_game` | 1.000 | `byte_at_offset` | **0.350** | 0.650 | 0.305 | +0.003 | surface-statistic | 7 | 2.68 | 0.69 | 0.00 |
| 127 | `goal_revision` | 1.000 | `nearest_neighbour` | **0.270** | 0.730 | 0.268 | +0.003 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 128 | `argumentation` | 1.000 | `char_count_tree` | **0.395** | 0.605 | 0.362 | +0.000 | surface-statistic | 3 | 1.58 | 0.00 | 0.00 |
| 129 | `logic_selection` | 1.000 | `byte_at_offset` | **0.305** | 0.695 | 0.282 | +0.000 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 130 | `mechanism_design` | 1.000 | `prompt_length` | **0.305** | 0.695 | 0.282 | +0.000 | surface-statistic | 4 | 1.99 | 1.00 | 0.00 |
| 131 | `dimensional_analysis` | 1.000 | `constant` | **0.297** | 0.703 | 0.297 | +0.000 | constant | 4 | 1.98 | 1.00 | 0.00 |
| 132 | `decomposition` | 1.000 | `prompt_length` | **0.292** | 0.708 | 0.278 | +0.000 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 133 | `general_language_agent` | 1.000 | `choice_in_observation` | **0.292** | 0.708 | 0.258 | +0.000 | copy | 222 | 7.41 | 1.00 | 0.00 |
| 134 | `predicate_logic` | 1.000 | `copy_fixed_offset` | **0.280** | 0.720 | 0.250 | +0.000 | copy | 6 | 2.57 | 1.00 | 0.00 |
| 135 | `open_ended_question_generation` | 1.000 | `constant` | **0.278** | 0.723 | 0.278 | +0.000 | constant | 4 | 2.00 | 1.00 | 0.00 |
| 136 | `cross_domain_unification` | 1.000 | `copy_fixed_offset` | **0.235** | 0.765 | 0.225 | -0.000 | copy | 5 | 2.32 | 1.00 | 0.00 |
| 137 | `ontology_construction` | 1.000 | `bag_of_chars` | **0.302** | 0.698 | 0.268 | -0.003 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 138 | `distributed_knowledge` | 1.000 | `choice_in_observation` | **0.290** | 0.710 | 0.270 | -0.003 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 139 | `paraphrase` | 1.000 | `nearest_neighbour` | **0.285** | 0.715 | 0.280 | -0.003 | memorisation | 4 | 1.99 | 1.00 | 0.00 |
| 140 | `causal_language` | 1.000 | `prompt_length` | **0.278** | 0.723 | 0.258 | -0.003 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 141 | `experimental_design` | 1.000 | `copy_near_keyword` | **0.270** | 0.730 | 0.273 | -0.003 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 142 | `grammar_induction` | 1.000 | `const_choice_index` | **0.268** | 0.733 | 0.250 | -0.003 | constant | 399 | 8.64 | 1.00 | 0.00 |
| 143 | `metareasoning` | 1.000 | `choice_in_observation` | **0.240** | 0.760 | 0.200 | -0.003 | copy | 7 | 2.80 | 1.00 | 0.00 |
| 144 | `resource_bounded_reasoning` | 1.000 | `choice_length_outlier` | **0.220** | 0.780 | 0.200 | -0.003 | distractor | 7 | 2.79 | 1.00 | 0.00 |
| 145 | `set_operations` | 1.000 | `nearest_neighbour` | **0.270** | 0.730 | 0.278 | -0.005 | memorisation | 4 | 1.99 | 1.00 | 0.00 |
| 146 | `program_explanation` | 1.000 | `char_count_tree` | **0.265** | 0.735 | 0.270 | -0.005 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 147 | `noisy_channel_language` | 1.000 | `nearest_neighbour` | **0.223** | 0.777 | 0.200 | -0.005 | memorisation | 6 | 2.57 | 1.00 | 0.00 |
| 148 | `counterfactuals` | 1.000 | `constant` | **0.295** | 0.705 | 0.295 | -0.007 | constant | 4 | 1.99 | 0.98 | 0.00 |
| 149 | `historical_reconstruction` | 1.000 | `byte_at_offset` | **0.287** | 0.713 | 0.292 | -0.007 | surface-statistic | 4 | 1.99 | 1.00 | 0.00 |
| 150 | `concept_invention` | 1.000 | `choice_in_observation` | **0.282** | 0.718 | 0.282 | -0.007 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 151 | `mechanism_discovery` | 1.000 | `nearest_neighbour` | **0.282** | 0.718 | 0.273 | -0.007 | memorisation | 4 | 2.00 | 1.00 | 0.00 |
| 152 | `theorem_proving` | 1.000 | `const_choice_index` | **0.278** | 0.723 | 0.250 | -0.007 | constant | 379 | 8.54 | 1.00 | 0.00 |
| 153 | `anytime_reasoning` | 1.000 | `nearest_neighbour` | **0.270** | 0.730 | 0.250 | -0.007 | memorisation | 395 | 8.62 | 1.00 | 0.00 |
| 154 | `conceptual_chunking` | 1.000 | `copy_fixed_offset` | **0.268** | 0.733 | 0.263 | -0.007 | copy | 4 | 2.00 | 1.00 | 0.00 |
| 155 | `question_generation` | 1.000 | `bag_of_chars` | **0.287** | 0.713 | 0.270 | -0.010 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 156 | `counterexample_generation` | 1.000 | `choice_in_observation` | **0.285** | 0.715 | 0.285 | -0.010 | copy | 4 | 1.99 | 1.00 | 0.00 |
| 157 | `event_semantics` | 1.000 | `bag_of_chars` | **0.312** | 0.688 | 0.280 | -0.015 | surface-statistic | 9 | 3.10 | 1.00 | 0.00 |
| 158 | `dialogue_game` | 1.000 | `choice_ranker` | **0.273** | 0.728 | 0.280 | -0.015 | distractor | 4 | 1.99 | 1.00 | 0.00 |
| 159 | `adversarial_argumentation` | 1.000 | `char_count_tree` | **0.265** | 0.735 | 0.250 | -0.015 | surface-statistic | 395 | 8.62 | 1.00 | 0.00 |
| 160 | `planning_language` | 1.000 | `choice_ranker` | **0.345** | 0.655 | 0.324 | -0.018 | distractor | 333 | 8.29 | 1.00 | 0.00 |
| 161 | `theory_comparison` | 1.000 | `bag_of_chars` | **0.285** | 0.715 | 0.292 | -0.018 | surface-statistic | 4 | 1.99 | 1.00 | 0.00 |
| 162 | `continual_language` | 1.000 | `choice_in_observation` | **0.280** | 0.720 | 0.250 | -0.018 | copy | 6 | 2.58 | 1.00 | 0.00 |
| 163 | `unknown_game` | 1.000 | `nearest_neighbour` | **0.347** | 0.652 | 0.333 | -0.020 | memorisation | 384 | 8.56 | 1.00 | 0.00 |
| 164 | `spatial_language` | 1.000 | `bag_of_chars` | **0.290** | 0.710 | 0.285 | -0.020 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 165 | `contradiction_tolerance` | 1.000 | `nearest_neighbour` | **0.280** | 0.720 | 0.250 | -0.020 | memorisation | 382 | 8.55 | 1.00 | 0.00 |
| 166 | `mathematical_definition_learning` | 1.000 | `choice_ranker` | **0.278** | 0.723 | 0.275 | -0.020 | distractor | 4 | 2.00 | 1.00 | 0.00 |
| 167 | `entailment` | 1.000 | `char_count_tree` | **0.400** | 0.600 | 0.372 | -0.022 | surface-statistic | 3 | 1.58 | 0.00 | 0.00 |
| 168 | `speaker_listener_game` | 1.000 | `choice_ranker` | **0.280** | 0.720 | 0.290 | -0.022 | distractor | 4 | 1.99 | 1.00 | 0.00 |
| 169 | `identity_continuity` | 1.000 | `nearest_neighbour` | **0.565** | 0.435 | 0.530 | -0.025 | memorisation | 2 | 1.00 | 0.00 | 0.00 |
| 170 | `recursive_self_application` | 1.000 | `choice_length_outlier` | **0.263** | 0.738 | 0.250 | -0.025 | distractor | 96 | 6.07 | 0.09 | 0.00 |
| 171 | `curriculum_design` | 1.000 | `constant` | **0.338** | 0.662 | 0.338 | -0.028 | constant | 4 | 1.97 | 1.00 | 0.00 |
| 172 | `universal_interface_transfer` | 1.000 | `choice_global_prior` | **0.307** | 0.693 | 0.294 | -0.028 | distractor | 380 | 8.54 | 1.00 | 0.00 |
| 173 | `anomaly_resolution` | 1.000 | `bag_of_chars` | **0.375** | 0.625 | 0.372 | -0.030 | surface-statistic | 3 | 1.58 | 0.00 | 0.00 |
| 174 | `representation_invention` | 1.000 | `const_choice_index` | **0.282** | 0.718 | 0.278 | -0.035 | constant | 4 | 2.00 | 1.00 | 0.00 |
| 175 | `protocol_discovery` | 1.000 | `byte_at_offset` | **0.233** | 0.767 | 0.220 | -0.037 | surface-statistic | 5 | 2.32 | 1.00 | 0.00 |
| 176 | `symbolic_world_builder` | 1.000 | `bag_of_chars` | **0.268** | 0.733 | 0.270 | -0.040 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 177 | `conjecture_generation` | 1.000 | `prompt_length` | **0.263** | 0.738 | 0.273 | -0.043 | surface-statistic | 4 | 2.00 | 1.00 | 0.00 |
| 178 | `problem_formulation` | 1.000 | `nearest_neighbour` | **0.278** | 0.723 | 0.300 | -0.062 | memorisation | 4 | 1.99 | 1.00 | 0.00 |
| 179 | `multimodal_symbolization` | 1.000 | `choice_by_length` | **0.198** | 0.802 | 0.200 | -0.062 | distractor | 8 | 2.99 | 1.00 | 0.00 |

---

## 3. The fixes

### How they are gated

A lesson is a pure function of an `rng` and a `GenerationContext`, and a context
is what a generator is told about the episode. It now carries a third field
beside `language` and `difficulty`:

```python
GenerationContext(language=..., difficulty=..., hardening=frozenset({...}))
ctx.hardens("ellipsis")   # -> bool, and raises on a name that is not declared
```

`context.HARDENING` names the fourteen draws; `Lesson.example(...,
hardening=...)` accepts `None` (the default set), `"none"` / `"off"` /
`"legacy"`, `"all"`, or an explicit list. A misspelled name raises rather than
silently doing nothing, and a test asserts that every declared name is actually
read by the lesson of that name. `generators/language/generator.py` forwards both
`hardening` and `difficulty` from its configuration, which also lands the
language track's D3 — the difficulty knob was previously unreachable from a TCN
configuration.

**The default is the hardened draw, and that changes the default distribution.**
This is deliberate and is the one place where the usual rule (gate a change and
leave the default alone) is overridden: a fix that is off by default fixes
nothing, and the task's own standard is that an exploitable lesson is worse than
a changed distribution. What is preserved instead is *reachability and
verifiability* of the old stream:

* `hardening="none"` reproduces the pre-audit generator **bit for bit**. Checked
  over 179 lessons x 12 configurations (3 languages x 4 difficulties) x 8 seeds
  = 17,184 episodes against a digest taken from the pre-audit generator
  *sources* -- recovered from the repository's `build/lib` copy, so the check is
  against an independent copy of the old code rather than against the legacy
  branch of the new code -- and again in `tests/test_language_hardening.py` over
  a 6-configuration grid against a committed fixture. Zero lessons differ.
* Under the default, exactly **16** of 179 lessons change: the 14 named draws,
  plus `general_language_agent` and `symbolic_generalist`, which compose
  sub-episodes and now draw them under the same regime. The other 163 are
  bit-identical, instance ids included.
* An episode drawn under a non-default regime gets a different `instance_id`,
  because it is a different problem. Ids under the default regime are unchanged.

Three of the fixes change what a lesson *asks*, not only how it draws, and that
is recorded here rather than buried:

* `tree_to_sequence` asks for a leaf at a drawn index instead of always the
  first leaf. The original question is not repairable — the first leaf of an
  in-order rendering is the first token printed.
* `center_embedding` asks for the verb of a subject drawn at random instead of
  always the outermost. Subject *i* still pairs with the verb at
  `depth + (depth - 1 - i)`, so the nesting still has to be unwound.
* `language_culture` asks about a drawn generation, and about meanings that may
  not have drifted, instead of always the final generation of a drifted meaning.

Two fixes cost a difficulty axis. `ellipsis` pins its clause count at the maximum
the six-name pool allows (4), because with two distractors the antecedent is
first or second and "the option that occurs earliest" is right half the time;
its `difficulty` knob has no range left. `presupposition` raises the floor of its
`n_extra` knob from 0 to 2 for the same reason.

### Before and after

Every lesson's oracle is 1.000 after the change: the answer is still a function
of the prompt, and the catalogue's own admission gate `verify_all()` passes for
all 179 implemented lessons.

| lesson | floor before | best exploit before | score | floor after | best exploit after | score | excess over null | oracle after |
|---|---|---|---|---|---|---|---|---|
| `center_embedding` | 0.361 | `choice_in_observation` | **1.000** | 0.361 | `char_count_tree` | **0.395** | -0.008 | 1.000 |
| `context_free_language` | 0.500 | `nearest_neighbour` | **1.000** | 0.500 | `train_lookup` | **0.610** | +0.045 | 1.000 |
| `ellipsis` | 0.167 | `choice_in_observation` | **1.000** | 0.167 | `choice_in_observation` | **0.207** | +0.015 | 1.000 |
| `language_culture` | 0.250 | `copy_fixed_offset` | **0.907** | 0.250 | `copy_fixed_offset` | **0.425** | +0.120 | 1.000 |
| `nesting_depth_compare` | 0.522 | `choice_ranker` | **0.812** | 0.522 | `byte_at_offset` | **0.627** | +0.092 | 1.000 |
| `next_symbol` | 0.285 | `copy_near_keyword` | **0.863** | 0.180 | `copy_near_keyword` | **0.370** | +0.182 | 1.000 |
| `paradigm_shift` | 0.260 | `char_count_tree` | **0.823** | 0.260 | `byte_at_offset` | **0.333** | +0.055 | 1.000 |
| `parse_depth` | 0.223 | `train_lookup` | **0.998** | 0.223 | `byte_at_offset` | **0.330** | +0.105 | 1.000 |
| `presupposition` | 0.268 | `char_count_tree` | **0.868** | 0.268 | `byte_at_offset` | **0.575** | +0.340 | 1.000 |
| `symbol_discrimination` | 0.540 | `byte_at_offset` | **0.802** | 0.540 | `byte_at_offset` | **0.580** | +0.037 | 1.000 |
| `symbol_equivalence` | 0.188 | `choice_in_observation` | **1.000** | 0.193 | `choice_in_observation` | **0.223** | +0.030 | 1.000 |
| `tree_to_sequence` | 0.182 | `copy_near_keyword` | **1.000** | 0.135 | `choice_ranker` | **0.295** | +0.160 | 1.000 |
| `underspecification_reasoning` | 0.223 | `char_count_tree` | **1.000** | 0.223 | `choice_in_observation` | **0.235** | -0.003 | 1.000 |
| `unification` | 0.182 | `choice_ranker` | **1.000** | 0.182 | `choice_in_observation` | **0.307** | +0.103 | 1.000 |
| `general_language_agent` | 0.258 | `choice_in_observation` | **0.292** | 0.257 | `choice_in_observation` | **0.292** | -0.010 | 1.000 |
| `symbolic_generalist` | 0.125 | `choice_length_outlier` | **0.155** | 0.124 | `choice_length_outlier` | **0.155** | +0.017 | 1.000 |

Each fix, one line on what changed:

| lesson | what the draw does now |
|---|---|
| `context_free_language` | the negative is the positive with **one bracket transposed** — same characters, same length, same first and last character, same local statistics; only the matching relation differs. The base string is drawn as a Dyck word with the number of pairs independent of the depth, which widens the string space |
| `parse_depth` | the string is a Dyck word with `pairs` drawn independently of `depth` (6-18) rather than `_balanced`'s four shapes, so length carries nothing and prompts stop recurring |
| `ellipsis` | the antecedent-and-gap pair is placed inside the discourse with at least one distractor clause after it, so recency is always wrong |
| `center_embedding` | the queried subject is drawn |
| `tree_to_sequence` | a leaf at a drawn index, over a tree built to a drawn leaf count from a 10-symbol alphabet |
| `unification` | two or three near-miss facts that disagree with the pattern in exactly one bound position, so occurrence counts stop isolating the binding |
| `symbol_equivalence` | every colour gets an alias (two colours get two, which is what "many-to-one" means), and the queried alias is drawn — so the lexicon narrows nothing |
| `underspecification_reasoning` | reversed compatibilities the instruction cannot instantiate, with the **total** number of printed lines drawn independently of the answer |
| `language_culture` | the transmission log is shown in shuffled order (each fact carries its own generation and position), and the question is drawn by choosing the answer word first and the (generation, meaning) pair second, so how often a word was heard says nothing |
| `nesting_depth_compare` | both strings redrawn with the **same** number of pairs and the flat pairs scattered through the nesting, so the two sides are identical in length and in bracket counts and no fixed offset reports the depth |
| `next_symbol` | six symbols instead of four, and the terminal cycle length drawn uniformly from {3,4,5} — 2 is excluded because every even offset then also lands on the answer |
| `symbol_discrimination` | the whole scale slides by a per-episode offset, so the queried magnitude carries nothing and only its position relative to the shown examples decides |
| `paradigm_shift` | a control series for **every** assumption that is not the target, so all four violation signatures are present in every prompt and the target can only be found by attributing one to the regime the rules name |
| `presupposition` | at least two competing utterances, and a `neither` proposition that shares one argument with the utterance |

### What is left

`presupposition` is the one fix that does not reach its floor, and the reason is
structural rather than fixable by re-drawing. The presupposed content is a
*different proposition* from the asserted content (`did_before` versus `does`),
so the query's predicate name partitions the four labels into
{presupposed, neither} and {asserted, denied, neither}. With uniform label
priors the best predictor from that partition alone is exactly **0.500**,
measured at 0.501 over 4,000 seeds; the battery reaches 0.575. The honest reading
is that `presupposition` should be reported against a 0.50 floor, not a 0.25
one — or given more layers.

`context_free_language` retains a `train_lookup` of 0.610 against a 0.500 floor:
19% of its prompts still recur, because the string space at depth 1-2 and 5-11
pairs is genuinely small and the prompt has to stay inside the 128-byte text
capacity the language track's scaffold uses (prompts are now 111-123 bytes, up
from 103-117). Widening further trades against that budget.

`nesting_depth_compare` retains a `byte_at_offset` of 0.627 against 0.522. A
single byte inside a bracket string does carry information about how deep that
string is; short of making the two sides identical in every local statistic —
which would make them equally deep — this is a real correlate rather than an
artifact.

---

## 4. What to fix next, in order

The audit ranks these; the mechanisms below come from reading the generators, so
they are diagnoses rather than scores.

| lesson | floor | best cheap exploit | score | excess | answer in prompt |
|---|---|---|---|---|---|
| `external_memory_design` | 0.215 | `choice_in_observation` | **0.730** | +0.510 | 1.00 |
| `discourse_state` | 0.250 | `copy_fixed_offset` | **0.750** | +0.468 | 1.00 |
| `symmetry_reasoning` | 0.185 | `choice_ranker` | **0.660** | +0.468 | 1.00 |
| `algorithm_analysis` | 0.338 | `byte_at_offset` | **0.770** | +0.455 | 1.00 |
| `explanation_repair` | 0.250 | `choice_in_observation` | **0.690** | +0.397 | 1.00 |
| `proof_compression` | 0.250 | `choice_in_observation` | **0.670** | +0.395 | 1.00 |
| `scope_ambiguity` | 0.362 | `char_count_tree` | **0.785** | +0.385 | 0.00 |
| `multi_perspective_modeling` | 0.273 | `copy_near_keyword` | **0.670** | +0.385 | 1.00 |
| `variable_binding` | 0.182 | `char_count_tree` | **0.560** | +0.373 | 1.00 |
| `social_convention_learning` | 0.250 | `choice_ranker` | **0.630** | +0.355 | 1.00 |
| `emergence_discovery` | 0.263 | `copy_near_keyword` | **0.625** | +0.330 | 1.00 |
| `teaching` | 0.250 | `choice_in_observation` | **0.615** | +0.325 | 1.00 |
| `representation_selection` | 0.268 | `byte_at_offset` | **0.560** | +0.300 | 1.00 |
| `instruction_following_micro` | 0.167 | `choice_in_observation` | **0.472** | +0.290 | 1.00 |
| `knowledge_refactoring` | 0.233 | `copy_near_keyword` | **0.522** | +0.272 | 1.00 |
| `deception_detection` | 0.250 | `copy_near_keyword` | **0.550** | +0.255 | 1.00 |
| `instruction_composition` | 0.167 | `choice_in_observation` | **0.463** | +0.255 | 1.00 |
| `compiler_construction` | 0.250 | `choice_ranker` | **0.522** | +0.255 | 0.01 |
| `algorithm_discovery` | 0.200 | `bag_of_chars` | **0.497** | +0.250 | 1.00 |
| `knowledge_update` | 0.182 | `choice_ranker` | **0.443** | +0.243 | 0.97 |
| `world_model_synthesis` | 0.285 | `choice_in_observation` | **0.515** | +0.205 | 1.00 |
| `knowledge_gap_detection` | 0.268 | `byte_at_offset` | **0.502** | +0.200 | 0.00 |
| `belief_state` | 0.250 | `copy_near_keyword` | **0.460** | +0.190 | 1.00 |
| `architecture_composition` | 0.165 | `copy_near_keyword` | **0.365** | +0.190 | 1.00 |
| `pronoun_coreference` | 0.500 | `char_count_tree` | **0.728** | +0.185 | 1.00 |
| `tool_construction` | 0.275 | `choice_in_observation` | **0.448** | +0.180 | 1.00 |
| `long_horizon_projects` | 0.118 | `choice_in_observation` | **0.323** | +0.177 | 1.00 |
| `question_answering` | 0.230 | `choice_in_observation` | **0.427** | +0.177 | 1.00 |

Four have a diagnosed mechanism and a fix of the same shape as one already
applied:

1. **`external_memory_design` (0.730).** The composite design contributes two
   `indexes` lines and one `design` line where every other design contributes
   two lines total, and it is usually the cheapest: "the design named on the
   most lines" is the answer in **0.727** of 1,500 seeds, measured directly.
   Give every design the same number of covered fields, or draw the coverage
   width independently of the cost — the `unification` fix.
2. **`discourse_state` (0.750).** Turns are shuffled and then *rendered in that
   order*, so the answer is the entity named in the last `mention` line —
   **0.767** of 1,500 seeds, measured directly. Each turn already carries its
   own index, so the rendering order can be shuffled independently — the
   `language_culture` fix.
3. **`scope_ambiguity` (0.785).** The three labels have three different edge
   counts by construction (`neither` silences an agent, `both` adds a shared
   book). Draw the number of `reads` edges first and realize the label within
   it — the `underspecification_reasoning` fix.
4. **`algorithm_analysis` (0.770).** Which search wins is mostly decided by where
   the target sits, and the target's *value* reveals its rank because the array
   is drawn from a fixed range. Normalize the value range per episode — the
   `symbol_discrimination` fix.

The remaining group (`explanation_repair`, `proof_compression`, `teaching`,
`scientific_civilization`) share one shape: the answer is one of four nonce
symbols that all occur in the prompt, and the correct one occurs slightly more
often because it participates in more of the structure being reasoned about.
That correlation is a *consequence* of the semantics rather than an artifact of
the draw, and closing it means equalizing occurrence counts across the four
candidates — worth doing, but it is a redesign of each lesson's candidate
sampler rather than a two-line change.

---

## 5. Lessons recommended for removal

**Short list, on purpose.** Seven lessons were solved outright by a cheap
heuristic and seven more were three-quarters solved. Two of the fourteen were
trivial *by specification* — the question itself names where the answer sits —
and the rest were trivial only in how they draw, which is repairable and was
repaired. The audit did **not** find a lesson that teaches nothing under any
draw. Recommending cuts the evidence does not support would be the same failure
as keeping lessons that copying solves, so the list below is short and two of
its four entries are conditional.

| lesson | recommendation | why |
|---|---|---|
| `tree_to_sequence` | **cut, unless the question may change** | "What is the first leaf?" over an in-order rendering is "copy the first symbol", and no re-draw changes that. It is repaired here by asking for a drawn leaf index — at which point it is `sequence_copy` over a tree rendering, and the catalogue already has `sequence_copy`. Cut it as redundant, or keep the repair and accept the redundancy. |
| `center_embedding` | **keep the repair; cut if the question is fixed** | Same shape: "the verb of the outermost subject" *is* the last word of an `N1..Nk Vk..V1` sentence. The repair drawing the queried subject preserves the capability the name claims; pinning the question back to the outermost subject makes the lesson a copy again. |
| `finite_state_language` | **keep, but stop reporting `construction` for it** | Not exploitable (best 0.515 against a 0.505 floor) but its privileged latent is `{symbol, parity}` — the induced rule, not the query — so a probe on `construction` is worth *exactly nothing* over the majority class, as the language track measured. It is a staging bug, not a lesson bug. |
| `next_symbol` (pre-fix) | **would have been cut** | Four symbols and a deterministic table give at most 1,024 distinct worlds, and the chain reaches a fixed point in **55.5%** of episodes -- the answer is then a symbol already printed at every position -- while 3,000 seeds produce only 691 distinct prompts. Widening the alphabet to six and drawing the cycle length makes it a real induction task; without that it is not worth a slot. |

Two structural recommendations that matter more than any single cut:

* **`verify_all` is not a sufficient admission gate.** It checks generation,
  determinism and the constant-guesser floor. All 179 lessons passed it while 63
  were exploitable, 14 answerable at >= 0.80 by a heuristic, and 7 at 1.000. The battery in this directory takes 90
  seconds over the whole catalogue and should run beside it.
* **Report `excess over null`, not accuracy.** Seven lessons sat at 1.000 for a
  cheap heuristic while their published floor said 0.17-0.50. A floor computed
  from the answer distribution does not bound a predictor that reads the prompt.

---

## 6. Limitations

* The exploits are cheap by construction: fixed offsets, character counts, a
  depth-5 tree, a linear ranker over eleven per-option features, 1-NN. A
  stronger surrogate would find more, and a lesson at excess 0.05 here is not
  certified clean — it is only not caught by this battery.
* `excess` subtracts a per-lesson null lift estimated from **one** null run.
  The band is tight (max 0.060 over 179 lessons) but it is one draw, not a
  distribution.
* 400 test episodes give about +-0.05 of sampling noise on a single exploit
  score, which is why the exploitability threshold is 0.10 and not 0.02.
* Everything is measured at `difficulty=None`. The knob widens most lessons'
  axes and some exploits will weaken with it; the audit accepts `--difficulty`
  and was not run across it.
* The fixes are measured by the same battery that found the problems. That is
  the right test for "is this exploit closed" and the wrong test for "is this
  lesson now hard" — the second question needs a learner, and the only learner
  ever run on this generator is the one in
  `research/language-capability/RESULTS.md`, on the one lesson this audit
  changed most.
* `research/language-capability/RESULTS.md` section 4's stage-A/stage-B result
  was obtained on the **pre-audit** `context_free_language`. It reproduces
  exactly under `hardening="none"`; under the default draw the counting rule it
  learned scores 0.5, which is the point.

---

## 7. Files

| file | what |
|---|---|
| `battery.py` | the sixteen exploits and the answer-set statistics |
| `audit.py` | drives the battery over the catalogue; `--null`, `--hardening`, `--difficulty` |
| `report.py` | ranks and renders the tables |
| `stream_digest.py` | per-lesson digests over a language x difficulty x seed grid |
| `audit_before.json`, `audit_null.json` | pre-fix audit and its null control |
| `audit_after.json`, `audit_null_after.json` | post-fix audit and its null control |
| `table_before.md`, `table_after.md` | the ranked tables, regenerated |
| `stream_preaudit.json` | digests of the generator before any edit (taken from the pre-audit sources) |
| `stream_legacy.json`, `stream_hardened.json` | the same grid under `hardening="none"` and under the default |
| `../../tests/test_language_hardening.py` | 23 tests: the regime API, the legacy-stream equivalence, and one behavioural check per fix |
| `../../tests/data/language_preaudit_stream.json` | the fixture the equivalence test compares against |

Reproduce:

```
.venv/bin/python research/lesson-audit/audit.py --train 600 --test 400 --workers 8 \
    --out research/lesson-audit/audit_after.json
.venv/bin/python research/lesson-audit/audit.py --train 600 --test 400 --workers 8 --null \
    --out research/lesson-audit/audit_null_after.json
.venv/bin/python research/lesson-audit/report.py --json research/lesson-audit/audit_after.json \
    --null research/lesson-audit/audit_null_after.json --md research/lesson-audit/table_after.md
.venv/bin/python research/lesson-audit/audit.py --hardening none --out /tmp/before.json   # the pre-audit draw
.venv/bin/python -m pytest tests/ -q                                                      # 202 passed
```
