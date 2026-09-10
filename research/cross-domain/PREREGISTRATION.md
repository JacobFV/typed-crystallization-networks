# Pre-registration — cross-domain semantic abstraction

Written before any arm is run. House standard since §44. Branch
`worktree-agent-a5dd347038d51263e`, working directory `research/cross-domain/`.
`tcn/` and `generators/` are not modified by this track.

## The question

Every library-induction result in this project (§44, §46, §52, §54, §57) is
measured on synthetic 4- and 5-input Boolean families. §57 states the claim now
rests on "14 in-family tasks across two families", both Boolean. The stated
objective is *reusable factorizations of intelligence: compact typed
transformations usable by arbitrary later systems and domains, including the
end-to-end visual, language, computer-use and embodied curricula*
(`research/algorithm-resynthesis/DESIGN.md` §8).

**Question: does semantic abstraction find anything shared across the visual,
language and computer artifacts, or only within synthetic Boolean families?**

## What was known before this document was written

Only facts read out of committed code and committed result JSON — no arm was
run:

* the three artifacts and the recorded selections that freeze them exist and are
  loadable (`research/visual-ladder/out/rung3_root.json`,
  `research/visual-ladder/out/rung3.json`,
  `research/language-capability/stage_a.json`,
  `research/language-capability/stage_b.json`,
  `research/language-post-audit/dyck_witness.json`,
  `research/computer-capability/out/agent_program.json`);
* the recorded node counts: visual S0 15, S1' 22, S2' 586, assembly 4; language
  stage A 5, stage B p16 84, stage B Dyck p22 138 live; computer agent 23;
* the carrier types each track declares, and that `BYTE = integer(8,
  signed=False, role="byte")` is *the same `Type`* in all three tracks while the
  address carriers are not (`IDX = integer(16, unsigned)` in visual,
  `POS = integer(32, unsigned, bounds=(0,128))` in language,
  `LEN = integer(32, unsigned, bounds=(0,4096))` in computer).

Nothing about fragment counts, identity classes or sharing was known.

## The corpus — seven frozen programs, three domains

Every program is the *frozen, hardened, selected* artifact of a published
section, reconstructed from its recorded selections and checked against a
recorded digest or node count where one exists.

| id | domain | source | section |
|---|---|---|---|
| `V_same` | visual | S0, `rung3.json:s0.chosen` | §32 |
| `V_corner` | visual | S1' masked, `rung3_root.json:s1.chosen`, offsets [3, 96] | §33 |
| `V_rect` | visual | S2', `rung3_root.json:s2.chosen`, steps [3, 96] | §33 |
| `V_assembly` | visual | the four caller nodes (hold, pair, filter, map) | §33 |
| `L_stage_a` | language | `language-capability/stage_a.json:module_program` | §19 / §45 |
| `L_stage_b16` | language | `language-capability/stage_b.json`, positions=16 | §19, **pre-audit stream** |
| `L_dyck22` | language | `stage_b_dyck` at `dyck_witness.json:witness`, positions=22 | §45, **post-audit stream** |
| `C_agent` | computer | `computer-capability/out/agent_program.json` | §23 |

Both language streams are carried, because §39/§45 require every quotation of
§19 to name its stream and the two artifacts are different programs. The
language generator is not re-run in step 1 (static program analysis only); if
step 2 runs it, it is pinned to `hardening='none'` and the stream is stated.

## Step 1 — the fragment inventory and the identity test

### R1/R2 — fragments

Exactly `research/earned-abstraction/mine.py`'s rule, unchanged in meaning:
single-exit sub-DAGs rooted at each node, every member reaching the root inside
the set, `MAX_NODES` nodes, `1 <= holes <= MAX_HOLES` where a hole is one per
distinct external port (constants included), canonicalised to a standalone
`Program`.

`mine.py`'s enumerator is `O(|ancestors| choose k)` and cannot run on `V_rect`
(586 nodes). This track re-implements the same set by connected-subgraph growth.
**Pre-registered correctness condition:** the re-implementation must return a
**bit-identical** fragment set to `mine.py` on every program small enough for
`mine.py` to run — the two Boolean corpora of §52 and every artifact program
under 60 nodes. If it does not, the inventory is void.

Configurations run: `MAX_NODES ∈ {3, 5}`, `MAX_HOLES = 4` (mine.py defaults at
5/4). Two configurations, not one — §53 arm F, enforced in §55.

### Identity relations — all three exact, none sampled

§52's identity is `(arity, truth table)` and depends on exactness. A sampled
hash is not a match and none is computed here.

* **D — syntactic.** `Program.digest` of the canonical fragment. Exact,
  available for every fragment. This is §44's identity.
* **S — type-exact semantic.** `(input types, output type, exhaustive
  input/output map)`. Computed **only** when every hole type and the output type
  is finite and the product of their universe sizes is `<= 2**20`. Exhaustible
  carriers here are `BOOL` (2) and the 8-bit integers (256). Every other carrier
  — `IDX` (2**16), `POS`, `LEN`, `CNT` (2**16), `float32`, and every product
  type (`TEXT` 129×256**128, the 3072-byte observation, `ACTION`) — is **not**
  exhaustible and is reported as such, per fragment, with a count.
* **S\* — carrier-abstracted semantic.** Only reached for fragments D and S
  cannot relate. The canonical fragment is rewritten with every scalar integer
  carrier replaced by one common placeholder carrier and re-validated against
  the registry; where that type-checks, the exhaustive map is computed **at two
  distinct placeholder widths** (4-bit and 8-bit unsigned). A class is admitted
  only if the map agrees under the natural embedding at **both** widths — one
  width certifies nothing (§53 arm F, §55). Where re-validation fails, or a
  product carrier of differing width blocks it, that is reported as a structural
  match **only**, explicitly not a semantic one.

### Triviality — defined now, before the table is read

A shared class is **TRIVIAL** if either
* its canonical fragment has exactly **one** node, or
* its exhaustive map equals a projection `(x0..xk) -> xi`, a constant function,
  or the negation of a projection.

Otherwise it is **NON-TRIVIAL**. A shared `identity`, a shared `project`, a
shared single `eq`/`and`/`not`/`add` is trivial by this rule and will be
reported as trivial, not as a reusable factorization of intelligence.

A class is **CROSS-DOMAIN** only if its occurrences span at least two of
`{visual, language, computer}`. Two programs of the same domain sharing a
fragment is not cross-domain.

### Step 1 headline numbers, fixed now

1. fragments and distinct D-classes per artifact, per `MAX_NODES`;
2. how many fragments admit an exact S identity, and how many do not, with the
   reason (carrier width / product / float);
3. the number of D-classes, S-classes and S\*-classes spanning ≥2 domains;
4. of those, how many are NON-TRIVIAL by the rule above.

## Step 2 — gated

Step 2 runs **only if** at least one **NON-TRIVIAL CROSS-DOMAIN** class exists
under S or S\*. If it does, each such class is tested as §52 and §57 test:
register it and measure a task in *another* domain against

* no library,
* a wrong module of comparable node count drawn from the same inventory,
* a hand-authored equivalent (the ceiling),

with enumeration certificates (`evaluated == space_size`, `exhausted`,
`certificate`) beside every synthesis number and constant/random baselines
beside every accuracy. `description_bits` is **not** used to rank anything
(§41, §52, §54, §57).

If the gate does not open, step 1 is the deliverable.

## Falsification — stated now, honoured whatever the table says

* **F1.** No fragment is shared across domains → the cross-domain abstraction
  story is unsupported, and the Boolean library-induction results do not
  generalise to the curricula. *Pre-registered as the most likely outcome.*
* **F2.** Fragments are shared but every shared class is TRIVIAL by the rule
  above → report exactly that. A shared `identity` node is not a reusable
  factorization of intelligence.
* **F3.** Non-trivial shared classes exist but registering one helps nothing →
  same status as §57's ranking result: identity without transfer.
* **F4.** The carriers cannot be exhausted and exact identity is not computable
  → the method's reach is bounded by that, reported as a bound. **The identity
  test is not weakened to a sampled one to avoid this outcome.**

Any of F1–F4 firing is a result, not a failure of the track.

## Discipline

* Enumeration with certificates beside every synthesis number.
* "No solution exists in this family" is distinguished from "search failed".
* Achieved configuration reported, never the requested one.
* Surprising headlines verified against raw data.
* Negative results preserved.
* Any amendment to this document is recorded with the number it replaces.

## Amendments

*(none at the time of writing)*
