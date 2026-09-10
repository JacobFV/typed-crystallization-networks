# Cross-domain semantic abstraction — does any of it reach the real curricula?

Branch `worktree-agent-a5dd347038d51263e`, working directory
`research/cross-domain/`. `tcn/` and `generators/` are **not modified** by this
track. `PREREGISTRATION.md` committed at `5d9ce5c`, **before any arm**;
amendments A1–A3 are recorded in it with what each replaces.

Every number below is recomputed from `out/*.json` by the scripts named beside
it, and the reproduction commands are in §11.

---

## 1. The headline

**Nothing non-trivial is shared.** Across 1,928 single-exit sub-DAG fragments
mined from the eight frozen artifact programs of the visual, language and
computer tracks, **every** semantic class that spans a domain boundary is a
**single core operator of `tcn/operators.py`** — `eq` on two bytes, `and` on two
Booleans, and under carrier abstraction `sub`, `add`, `identity`, `lt`. Zero
multi-node semantic classes cross a domain boundary at either configuration.

The same code, unchanged, finds **5 non-trivial shared semantic classes across
two different synthetic Boolean families** and **11–13 within one of them**. So
this is not a dead pipeline reporting zero. It is a live pipeline reporting that
the Boolean result does not reach the curricula.

Pre-registered falsification **F2 fires**: *fragments are shared but only trivial
ones*. **F4 fires in part**: 142–180 of the fragment classes have carriers that
cannot be exhausted, and their identity is reported as uncomputable rather than
replaced by a sampled one. Two exact bounds (§6) show what that cap can and
cannot be hiding — and it is not hiding a cross-domain match.

**The mechanism is the type system, and it is stated exactly.** The only carrier
types the three domains have *in common at all* are `BOOL` and
`u8[byte]` (§6.3). Every address, count, coordinate, buffer and action carrier is
domain-specific: `u16` in visual, `u32<=128` in language, `u32<=4096` and `f32`
in computer. §52's identity relation requires equal types before it compares
behaviour, so no exhaustion budget could ever match a visual address computation
to a language one.

---

## 2. The corpus — eight frozen artifacts, three domains

`artifacts.py`. Every program is the published artifact of a numbered section,
rebuilt from that section's recorded selections and checked against a recorded
digest or node count. Nothing is trained or searched here.

| id | domain | source | nodes | digest |
|---|---|---|---|---|
| `V_same` | visual | S0, `rung3.json:s0.chosen` (§32) | 15 | `b409b71ea0e41d77abb7cb9c` |
| `V_corner` | visual | S1′ masked, offsets [3, 96] (§33) | 22 | `6804983bda88826441cba10d` |
| `V_rect` | visual | S2′, steps [3, 96] (§33) | 586 | `009b0bc80630eb8fcf02493e` |
| `V_assembly` | visual | hold / pair / filter / map (§33) | 4 | `1626ea475ff615bada1a741a` |
| `L_stage_a` | language | `stage_a.json:module_program` (§19/§45) | 5 | `8063993bfeee7393683e47b3` |
| `L_stage_b16` | language | `stage_b.json`, positions=16, **pre-audit stream** (§19) | 84 | `3df671c3620c80fb4221ae9e` |
| `L_dyck22` | language | `stage_b_dyck` at §45's witness, positions=22, **post-audit stream** | 138 | `4d0ce927662af9fe54799de0` |
| `C_agent` | computer | `out/agent_program.json` (§23) | 23 | `4000a975cc7b84585d304fdf` |

**Streams, named as §39/§45 require.** The two language artifacts are different
programs on different streams and are never merged: `L_stage_b16` is §19's
pre-audit artifact (a bracket **counter**, 0.9986 on 724 held-out); `L_dyck22` is
§45's post-audit witness (`total == 0 and min_prefix >= 0`, 1.000 on 859 held-out
at unseen lengths 16–22). No generator is re-run by this track, so no
`hardening` setting is exercised; had step 2 opened, it would have been pinned to
`hardening='none'`.

`L_stage_a`'s `module_program` is taken as published rather than re-hardened:
re-hardening five already-selected single-candidate nodes rebuilds them under
digest `427732402e776f2c043b1be5`, and `stage_b.json` records the module name
`module:8063993bfeee7393683e47b3`, which the published form reproduces. The
`L_dyck22` rebuild reproduces §45's recorded `program_digest`
`4d0ce927662af9fe54799de0` exactly.

The visual assembly's execution cost is §33's: `same_module_cost` 15,
`corner_module_cost` 35, `rect_module_cost` 1453 over 1,024 positions, four
caller nodes.

---

## 3. The enumerator, and the gate it had to pass first

`research/earned-abstraction/mine.py`'s R1 takes every subset of a root's
ancestor set. On `V_rect` (586 nodes) that is order 10⁹ subsets per root and
cannot be run. `frag.py` enumerates the same family by connected growth, with the
equivalence argument written out in its docstring.

**Pre-registered correctness gate** (`run_check.py` → `out/equivalence.json`):
the two enumerators must return **bit-identical** fragment sets on every program
`mine.py` can run.

| | |
|---|---|
| programs checked | 5 artifact programs under 60 nodes + 426 Boolean corpus programs (§52 MAJ3, both bands) |
| checks | **862** (each program at `max_nodes` 3 and 5) |
| identical | **862 of 862** |
| any error rows | 0 |

The gate passed, so the inventory stands. `frag.canonicalize` reproduces
`mine.canonicalize` bit-for-bit when the registry argument is `None`.

---

## 4. Fragment inventory

`run_inventory.py` → `out/inventory_3.json`, `out/inventory_5.json`. Two
configurations, not one — §53 arm F, enforced in §55. `MAX_HOLES = 4` throughout
(`mine.py`'s default).

**"with registry"** canonicalises against the artifact's own registry, so
fragments containing a frozen `module:` call survive. **"mine.py-exact"** uses a
fresh `Registry()`, which is what `mine.py` does and which silently drops every
module-crossing fragment. Both are reported; the analysis uses the first, because
dropping module-crossing fragments would delete exactly the compositional
fragments a cross-domain claim would live in.

### `MAX_NODES = 3`

| artifact | domain | nodes | fragments | (mine.py-exact) | distinct D-classes |
|---|---|---|---|---|---|
| `V_same` | visual | 15 | 46 | 46 | 25 |
| `V_corner` | visual | 22 | 47 | 43 | 26 |
| `V_rect` | visual | 586 | 1,292 | 1,044 | 44 |
| `V_assembly` | visual | 4 | 9 | 3 | 9 |
| `L_stage_a` | language | 5 | 13 | 13 | 13 |
| `L_stage_b16` | language | 84 | 162 | 130 | 17 |
| `L_dyck22` | language | 138 | 251 | 207 | 23 |
| `C_agent` | computer | 23 | 54 | 54 | 50 |
| **total** | | **877** | **1,874** | **1,540** | **178 distinct** |

### `MAX_NODES = 5`

| artifact | domain | nodes | fragments | (mine.py-exact) | distinct D-classes |
|---|---|---|---|---|---|
| `V_same` | visual | 15 | 55 | 55 | 31 |
| `V_corner` | visual | 22 | 47 | 43 | 26 |
| `V_rect` | visual | 586 | 1,316 | 1,068 | 56 |
| `V_assembly` | visual | 4 | 10 | 3 | 10 |
| `L_stage_a` | language | 5 | 17 | 17 | 17 |
| `L_stage_b16` | language | 84 | 162 | 130 | 17 |
| `L_dyck22` | language | 138 | 254 | 210 | 26 |
| `C_agent` | computer | 23 | 67 | 67 | 63 |
| **total** | | **877** | **1,928** | **1,593** | **217 distinct** |

Wall clock: 92.9 s and 175.2 s respectively.

`V_rect` has 586 nodes and only 56 distinct fragment classes because it is a
31-fold unrolled prefix conjunction: the same handful of shapes repeated. That
is a fact about the artifact, not about the enumerator.

---

## 5. How far exactness reaches — F4, stated as a bound rather than dodged

`run_shared.py` → `out/shared_3.json`, `out/shared_5.json`.

| | `MAX_NODES=3` | `MAX_NODES=5` |
|---|---|---|
| D-classes | 178 | 217 |
| **S (type-exact semantic) computed exactly** | **36** | **37** |
| S not exhaustible | 142 | 180 |
| **S\* (carrier-abstracted) computed exactly** | **54** | **54** |
| S\* not computable | 124 | 163 |

Reasons a class has no exact S, at `MAX_NODES=5` (the full table is in the JSON):

| reason | classes |
|---|---|
| joint domain over the 2²⁰ cap (e.g. `(u16, u16)` is 2³²) | 72 |
| product carrier, 3,072 fields (the visual observation) | 28 |
| floating carrier (`f32`, the computer action encoder) | 21 |
| product carrier, 2 fields (`TEXT`, `record`) | 19 |
| product carrier, 4,096 fields (the terminal buffer) | 10 |
| set carrier (the assembly's positional set) | 10 |
| other product carriers | 20 |

**No sampled identity was computed anywhere.** §52's result depends on
exactness; a sampled match is not a match, and none is reported. Where a carrier
could not be exhausted the class simply has no S identity, and §6 establishes
what that cannot be concealing.

---

## 6. Cross-domain classes, and two exact bounds on what the cap could hide

### 6.1 The answer, at both configurations

`run_shared.py`. A class is **cross-domain** only if its occurrences span at
least two of {visual, language, computer}. Triviality is the pre-registered rule
as amended by A2: **a class that can be spelled with one primitive is that
primitive.**

| relation | `MAX_NODES=3` | | `MAX_NODES=5` | |
|---|---|---|---|---|
| | cross-domain | **non-trivial** | cross-domain | **non-trivial** |
| **D** — `Program.digest` (§44's identity) | 2 | **0** | 2 | **0** |
| **S** — type-exact semantic (§52's identity) | 2 | **0** | 2 | **0** |
| **S\*** — carrier-abstracted, certified at 4-bit *and* 8-bit | 6 | **0** | 6 | **0** |

The complete list — this is every semantic class in the project that spans two
of the real domains:

| relation | domains | class | type signature | occurrences |
|---|---|---|---|---|
| D, S, S\* | **all three** | `eq(x0, x1)` | `(u8[byte], u8[byte]) -> BOOL` | `V_same` 3, `L_stage_a` 1, `C_agent` 1 |
| D, S, S\* | language + visual | `and(x0, x1)` | `(BOOL, BOOL) -> BOOL` | `V_corner` 4, `V_rect` 122, `L_dyck22` 1 |
| S\* only | all three | `sub(x0, x1)` | 3 distinct carriers | `L_stage_b16` 1, `L_dyck22` 1, `V_corner` 2, `V_rect` 1, `C_agent` 1 |
| S\* only | all three | `identity(x0)` | 3 distinct carriers | `C_agent` 1, `V_corner` 2, `V_rect` 2, `L_stage_a` 1 |
| S\* only | all three | `add(x0, x1)` | 3 distinct carriers | `L_stage_a` 2, `C_agent` 1, `V_same` 4, `V_rect` 130 |
| S\* only | language + visual | `lt(x0, x1)` | 2 distinct carriers | `L_stage_b16` 16, `L_dyck22` 22, `V_rect` 62 |

**Every one of these six is a universal operator of `tcn/operators.py`.** There
is nothing to induce, nothing to register and nothing to transfer: the library
already "contains" them in the strongest possible sense — they are primitives of
the algebra. Reporting `eq` or `and` as a discovered reusable factorization of
intelligence would be exactly the overclaim this project has corrected four
times.

One member of the `add` S\*-class is the 2-node spelling `add(identity(x0), x1)`
(from `L_stage_a`). The first implementation of the triviality rule read `all`
over a class's representatives and therefore labelled that whole class
NON-TRIVIAL. **Amendment A2** corrected the reading to `any`, which is the
conservative direction; the `all` reading would have reported a cross-domain
"non-trivial" class that is `add`.

### 6.2 B1 — the type-signature bound (amendment A3)

Two fragments in different domains can be S-equal only if their **type
signatures** are equal, and the type signature is exact for every fragment,
exhaustible or not. So the count of type signatures occurring in more than one
domain is an exact **upper bound on cross-domain S classes at any cap**.

| | `MAX_NODES=3` | `MAX_NODES=5` |
|---|---|---|
| distinct type signatures over all fragments | 110 | 131 |
| **occurring in more than one domain** | **2** | **2** |

The two are `(u8[byte], u8[byte]) -> BOOL` (all three domains; largest member 2
nodes, `not(eq(x0,x1))`) and `(BOOL, BOOL) -> BOOL` (language + visual; largest
member 1 node). **Raising `EXHAUST_CAP` to infinity could not produce a third.**
The bound and the measurement agree: 2 cross-domain S classes, and 2 is the
ceiling.

### 6.3 B3 — the carrier inventory, which is the mechanism

Carriers touched by each domain's fragments at `MAX_NODES=5`:

| domain | carriers |
|---|---|
| visual | `BOOL`, `u8[byte]`, `u16`, `u24`, `(2xu16)`, `(31xu16)`, `(3xu8[byte])`, `(3072xu8[byte])`, 4 set types |
| language | `BOOL`, `u8[byte]`, `u32<=128`, `i16`, `(2xu32<=128)`, `(128xu8[byte])` |
| computer | `BOOL`, `u8[byte]`, `u8`, `u32<=4096`, `f32`, `(1xu8)`, `(3xf32)`, `(4xf32)`, `(2x(4xf32))`, `(1022xf32)`, `(2xu32<=4096)`, `(4096xu8[byte])` |
| **intersection of all three** | **`BOOL`, `u8[byte]`** |
| every pairwise intersection | **`BOOL`, `u8[byte]`** |

That is the whole story in one row. The three tracks each chose a different
address carrier — visual `u16`, language `u32` refined to `[0,128]`, computer
`u32` refined to `[0,4096]` — a different count carrier, and different buffer
widths (3,072 / 128 / 4,096 bytes). Under §52's identity relation those are
**different types**, so a fragment computing "the byte at base + offset" in
vision and the same thing in language are, by construction, not the same
abstraction. §53 and §55's width-polymorphism work is the machinery this would
need; §55 measured that a stored class saves 272× at unseen widths but declined
to justify a core change. This track is a second, independent reason to want it.

### 6.4 B2 — the motif that recurs and cannot be identified

The relation **T** erases types entirely and keeps only operator names and edges.
**T is not semantic and nothing here treats it as such.** It answers only:
*is there a recurring motif that the type system forbids identifying?*

At `MAX_NODES=5`: 188 operator shapes, **16 cross-domain**, of which **4 are
multi-node** and 7 span all three domains (the rest single-node).

| nodes | domains | motif | why it is not a semantic class |
|---|---|---|---|
| 3 | language + visual | `eq(index(x2, add(x0, x1)), x3)` | buffers `(128xu8)` vs `(3072xu8)`; addresses `u32<=128` vs `u16` |
| 2 | **all three** | `eq(index(x0, x1), x2)` | buffers 128 / 3,072 / 4,096 wide |
| 2 | language + visual | `index(x2, add(x0, x1))` | same |
| 2 | computer + language | `sub(project(x0), x1)` | `(2xu32<=128)` vs `(2xu32<=4096)` |

This is **exactly** the coincidence `research/algorithm-resynthesis/DESIGN.md` §8
predicted — "`locate`/`find-first` … recur across all four domains … sparse
detection in vision, span-finding in language, control-location in computer use".
The prediction is confirmed *structurally* and refuted *semantically*: the motif
"the byte at a computed address equals a literal" really does appear in all three
artifacts, and the abstraction machinery of §44/§46/§52/§54/§57 cannot see it,
because every one of its identity relations compares types before it compares
behaviour. DESIGN.md §8's own promotion test — "it must help a later held-out
task" — is unreachable for this motif: there is no class to promote.

---

## 7. The positive control — the same code finds plenty where plenty exists

`run_control.py` → `out/control_<n>_<band>.json`. A zero is worthless without
this.

**P1.** §52's MAJ3 corpus and §57's W4 corpus, first eight tasks of each, treated
as two pseudo-domains. Identical `frag.py`, identical `identity.py`, identical
triviality rule.

| band | `MAX_NODES` | within MAJ3, S-classes shared across tasks | of those **non-trivial** | MAJ3 vs W4, S-classes | of those **non-trivial** | MAJ3 vs W4, D-classes (non-trivial) |
|---|---|---|---|---|---|---|
| `C-minall` | 3 | 14 | **11** | 8 | **5** | 10 (7) |
| `C-minall` | 5 | 16 | **13** | 8 | **5** | 10 (7) |
| `C-trace` | 3 | 14 | **11** | 7 | **5** | 12 (10) |
| `C-trace` | 5 | 16 | **13** | 7 | **5** | 12 (10) |

All four cells agree: **5 non-trivial semantic classes shared between two
different Boolean families**, at both bands and both fragment budgets. Certified
at more than one configuration, as §53 arm F and §55 require.

The non-trivial cross-family classes are real composites:
`xor(and(x2, xor(x0,x1)), x3)`, `or(x3, and(x2, xor(x0,x1)))`,
`or(x2, and(x0,x1))`, `xor(x2, and(x0,x1))`, `and(x2, xor(x0,x1))`.

**So the contrast is exact and at matched configuration:**

| | non-trivial shared semantic classes |
|---|---|
| between two synthetic Boolean families | **5** |
| within one synthetic Boolean family | **11–13** |
| **between any two of the three real domains** | **0** |

**P2 — within-domain sharing in the real artifacts.** This separates "the
machinery finds nothing" from "the machinery finds things, but never across a
domain boundary".

| domain | programs | D-classes shared across ≥2 of that domain's programs (non-trivial) | S-classes shared (non-trivial) |
|---|---|---|---|
| visual | 4 | 14 (4) | 3 (**1**) |
| language | 3 | 12 (4) | 3 (**0**) |
| computer | 1 | — one program only | — |

The single non-trivial within-domain semantic class is worth naming, because it
is **§52's identity relation doing real work on a real artifact for the first
time**: the 3-input conjunction, spelled

* `truth_7` then `truth_2` in `V_same` — digest `d92069f3345c…`, 1 occurrence;
* `and` then `and` in `V_corner` / `V_rect` — digest `786564a1a651…`, 62 occurrences;

two **different** digests, one S-class `b0bd382015d5…`. Syntactic mining sees two
abstractions; semantic pooling sees one. That is §46/§52's mechanism, and it
fires on the visual parser. It does not cross a domain boundary.

---

## 8. Which §57 regime the artifacts are in

`run_regime.py` → `out/regime_<n>.json`. §57 found that pooling buys rank in
proportion to fragmentation and *inverts* where the competitor is the fragmented
class. **This track does no ranking at all** — it counts class membership, never
scores it, and `description_bits` is not used anywhere (§41, §52, §54, §57) — so
§57's inversion cannot arise here. The underlying fragmentation is still worth
stating, measured as the collapse from D-classes to S-classes among exhaustible
fragments:

| artifact | D-classes | exhaustible | S-classes | collapse D/S |
|---|---|---|---|---|
| `V_same` | 31 | 10 | 7 | **1.43** |
| `L_stage_a` | 17 | 4 | 3 | 1.33 |
| `V_corner` | 26 | 9 | 7 | 1.29 |
| `V_rect` | 56 | 6 | 6 | 1.00 |
| `C_agent` | 63 | 10 | 10 | 1.00 |
| `L_dyck22` | 26 | 4 | 4 | 1.00 |
| `L_stage_b16` | 17 | 3 | 3 | 1.00 |
| `V_assembly` | 10 | 0 | 0 | — |

Identical at `MAX_NODES=3` and 5. Fragmentation is **low**: five of eight
artifacts collapse not at all. That is `C-minall`'s regime, which is the band on
which §57 measured pooling to **lose** (0 against syntactic mining's 48). So the
real artifacts sit on the side of §57's boundary where §52's advantage does not
apply — a second, independent reason not to expect the Boolean library-induction
result to carry over.

---

## 9. Step 2 — the gate does not open

`PREREGISTRATION.md` gates step 2 on at least one **non-trivial cross-domain**
class under S or S\*. There are **zero**, at both configurations, under all three
relations. Step 2 was therefore **not run**, exactly as pre-registered.

This is not a budget decision. There is no candidate to test: registering `eq`,
`and`, `sub`, `add`, `identity` or `lt` as a library module is a no-op, because
all six are already primitives of `tcn/operators.py`. A transfer arm against
no-library, a wrong module and a hand-authored ceiling would compare a module to
itself.

---

## 10. Verdict

**Does the abstraction story reach the real domains? No, and the reason is
structural rather than empirical.**

1. **The mechanism works.** The fragment rule, the semantic identity relation and
   the triviality test all fire correctly: 5 non-trivial classes shared between
   two Boolean families, 11–13 within one, 1 within the visual domain (a genuine
   §52-style merge of two spellings of a 3-input conjunction that syntactic
   mining counts twice).
2. **It finds nothing across the curricula.** 1,928 fragments, 217 classes, 2
   cross-domain semantic classes, both single core operators. 0 non-trivial, at
   both `MAX_NODES = 3` and 5.
3. **The blocker is the type system, not the search.** The three domains share
   exactly two carrier types, `BOOL` and `u8[byte]`. The type-signature bound
   (B1) proves the exhaustion cap is not hiding a third match: only 2 of 131 type
   signatures cross a domain boundary at all.
4. **A real cross-domain motif exists and is invisible to every identity relation
   in this repo.** "The byte at a computed address equals a literal" occurs in
   all three artifacts. It is one structural family and three semantic classes,
   because the buffers are 128, 3,072 and 4,096 bytes wide and the address
   carriers are `u32<=128`, `u16` and `u32<=4096`. This is DESIGN.md §8's
   prediction confirmed structurally and refuted semantically.
5. **So §52/§54/§57 stand where they were measured and nowhere else.** The
   library-induction claim rests on 14 in-family tasks across two Boolean
   families (§57). This track adds the missing negative: on the visual, language
   and computer artifacts, the induction machinery has **nothing non-trivial to
   act on**, and the fragmentation regime of those artifacts (§8) is the one in
   which §57 measured semantic pooling to lose.

**What would change the answer**, stated so it is falsifiable rather than
hand-waved: a class identity that abstracts over *product width and integer
carrier together* — §53's width-polymorphism plus buffer-width abstraction —
would collapse B2's four multi-node cross-domain motifs into candidate classes.
Whether registering one of those then helps a held-out task in another domain is
untested and is the next experiment. This track does **not** claim it would; §57
is the reminder that identity without transfer is not a capability.

---

## 11. Verification

**Reproduce**, repo `.venv`, from the worktree root:

```bash
.venv/bin/python research/cross-domain/artifacts.py       # the 8 programs + digest checks
.venv/bin/python research/cross-domain/run_check.py       # the 862-check equivalence gate
.venv/bin/python research/cross-domain/run_inventory.py --max-nodes 3
.venv/bin/python research/cross-domain/run_inventory.py --max-nodes 5
.venv/bin/python research/cross-domain/run_shared.py --max-nodes 3
.venv/bin/python research/cross-domain/run_shared.py --max-nodes 5
.venv/bin/python research/cross-domain/run_bounds.py --max-nodes 5
.venv/bin/python research/cross-domain/run_regime.py --max-nodes 5
.venv/bin/python research/cross-domain/run_control.py --max-nodes 3 --band C-minall
.venv/bin/python research/cross-domain/run_control.py --max-nodes 3 --band C-trace
.venv/bin/python research/cross-domain/run_control.py --max-nodes 5 --band C-minall
.venv/bin/python research/cross-domain/run_control.py --max-nodes 5 --band C-trace
```

**Every headline re-derived independently.**
`.venv/bin/python research/cross-domain/verify_claims.py` re-reads the raw class
records and recomputes each claim without importing the analysis modules, so a
bug in `run_shared.py` cannot make its own claim true. **51 checks, all pass** —
fragment totals, class counts, cross-domain counts under all three relations at
both configurations, the operator list, the type-signature bound, the carrier
intersections, all four control cells, the fixture and the test tally.

**Repo constraints.**

* Tests: **324 passed, 13 failed** (`out/pytest.log`). All 13 are the known
  environmental class: `generators/computer/engine/bridge.ts` is launched with
  `node --import tsx` and the error is
  `ERR_MODULE_NOT_FOUND: Cannot find package 'tsx'` — the gitignored
  `node_modules` symlink is absent from this worktree *and from the main
  checkout*. Verified per the standing instruction by **moving
  `research/cross-domain/` out of the tree entirely** and re-running
  `tests/test_generators.py tests/test_panel_interface.py`: **the same 13 fail,
  17 pass**. 324 + 13 = 337.
* Shipped fixture (`out/fixture.log`): `initial_prediction_loss`
  **0.248835613951087** → `final_prediction_loss` **0.0022308224288281053**,
  `evaluation_mean_return` **4.0**, `frozen_evaluation_mean_return` **4.0** —
  0.248836 → 0.002231 at 4/4 frozen, as required.
* `tcn/` and `generators/` untouched; `research/emitter-guards/` untouched.
* Largest artifact written: `out/inventory_5.json` at ~150 kB. Full `Type`
  dictionaries are replaced in the stored records by a `label#sha256` tag — a
  3,072-field product type serialises to ~250 kB and would have put the inventory
  over 100 MB. The tag is exact (the sha is over the complete `to_dict()`), and
  every identity is computed in memory from the real `Type` objects, not from the
  tag.

**Disclosures this track makes about itself.**

* The first triviality implementation used `all` over a class's representatives
  and would have reported a cross-domain **non-trivial** class that is `add`.
  Corrected by amendment A2 to `any`, the conservative direction; recorded in
  `PREREGISTRATION.md` with what it replaced.
* `frag.py` is a re-implementation of another track's published rule. It is not
  trusted on argument: the 862-check bit-identity gate is what licenses it, and
  the inventory is declared void without it.
* The exhaustion cap is a real limit — 37 of 217 classes have an exact semantic
  identity at `MAX_NODES=5`. The B1 bound is what turns that from a hole into a
  bounded claim; without B1 the correct report would have been F4 alone
  ("identity cannot be computed exactly") rather than F2.
* `EXHAUST_CAP` is 2²⁰ rows per fragment. Raising it to 2³² would let
  `(u16, u16)` fragments be exhausted; B1 shows this cannot produce a
  cross-domain match, but it would add within-visual S classes that are not
  measured here.
* The visual artifacts are one screen configuration (FLAT: resolution 32,
  palette 32, nesting 5, widgets 20). The language artifacts are one lesson on
  two streams. The computer artifact is one task. This is **eight programs**, not
  a sample of the curricula, and the negative is about these eight.
