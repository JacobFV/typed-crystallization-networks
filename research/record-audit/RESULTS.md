# Record audit — `research/FINDINGS.md`, `STATUS.md`, `README.md`, `docs/`

**Track: record-audit. Not an experiment.** `tcn/` and `generators/` are
untouched; no other track's directory was modified. The deliverables are this
file and `research/record-audit/verify.py`, which re-runs every mechanical check
below so the audit is repeatable.

```bash
.venv/bin/python research/record-audit/verify.py            # passes 1 and 3, against artifacts/demo/summary.json
.venv/bin/python research/record-audit/verify.py --demo     # run the ten demonstrations first
.venv/bin/python research/record-audit/verify.py --tests    # also run pytest
```

Current state: **49 checks, 25 pass, 23 fail, 1 warn.**

`--tests` needs the gitignored `node_modules` symlinked into
`generators/computer/engine/` from the main checkout, per
`research/MERGE-QUEUE.md`; without it 13 tests fail for want of `node`/`tsx`,
with it exactly one does (the documented worktree-only panel replay). Both
states were reproduced here.

---

## 0. Achieved scope, stated before the findings

| | |
|---|---|
| FINDINGS.md headings read in full | **61** (60 distinct numbers — §14 appears twice, §42 does not exist) |
| Sections whose cited figures were checked against raw data | **58** exhaustively or near-exhaustively; **3** (§37, §59, §61) only via off-branch data, because their track directories are absent from this branch |
| **Distinct cited figures checked against raw JSON/logs** | **794** — 760 across five parallel per-section audits, plus 43 I verified directly, 9 of them deliberate re-verifications of the highest-severity findings before reporting them |
| Matched exactly | **≈637** |
| **Discrepancies** | **85** — 67 from the per-section audits, 18 from the cross-document and demo pass |
| Stylistic differences, reported as *not* errors | **29** |
| Uncheckable | **39 figures + 3 whole sections** |
| Demonstrations run | **10 of 10, twice** (quick 60 s, `--full` 192 s) |
| Python tests | 337 collected, **336 pass, 1 fails** — the documented worktree-only `test_panel_interface` replay failure. Confirmed environmental: 13 failures with `node_modules` removed, 1 with it symlinked |
| Node engine tests | **372 of 372 pass** |

**Sampled rather than exhaustive:** §1–§9 (the original eight-track summary) were
audited for *supersession* rather than figure-by-figure, because much of their
underlying data lives in `research/crystallization-ablation/`,
`research/search-scaling/`, `research/perturbation-selection/` and
`research/loss-gated-eligibility/`, which have `RESULTS.md` but no `out/`.

**Not audited at all:** `ARCHITECTURE.md`, `docs/BLUEPRINT.md`,
`docs/IMPLEMENTATION.md`, `docs/PROJECT-LOG.md`, `docs/handoffs/`,
`research/AGENDA.md` (except where a FINDINGS claim pointed into them).

**I did not find "nothing wrong."** The failure modes catalogued in
`docs/CORRECTIONS.md` recur, and two of them recur *inside `CORRECTIONS.md`
itself* — two rows in the catalogue of overturned claims are themselves wrong.

---

## 1. Discrepancy table

Severity: **high** = a load-bearing claim in a most-read document is wrong,
inverted, or does not reproduce. **medium** = a figure is quoted at a scope,
precision or denominator the raw data does not support, or two shipped documents
disagree. **low** = stale, mis-cited, or missing a baseline that exists, without
changing a verdict.

### 1.1 High

| # | where | claim as written | raw value | severity |
|---|---|---|---|---|
| H1 | `STATUS.md` §1 header + row 9; `docs/VALIDATION.md`:395, :526 | "**Ten** capabilities reproduce on the current tree"; "All of these reproduce through `scripts/demo.sh`"; row 9 reproduce command `scripts/demo.sh --only language` | **9 of 10.** `language` raises `ValueError: training examples required` at `tcn/search.py:162` via `_demo_language`. Reproduced twice (quick and `--full`). The cause is §39's own finding: the post-audit default stream never emits lengths 2/4/6, so stage A's training split is empty | **high** |
| H2 | `docs/VALIDATION.md`:407, :516 | "the grammaticality rule **1.000** at string lengths never trained on"; "a grammaticality rule reaches **1.000** on 724 held-out episodes" | **0.9986187845303868** (`research/language-capability/final_eval.json`). This is the *exact* claim `docs/CORRECTIONS.md` row 9 and §43 record as overturned. §43 says "§19, `STATUS.md` and `HANDOFF.md` **all three files are corrected**" — VALIDATION.md was never in that list and was never corrected, and it is the file README points readers to. No stream qualifier either | **high** |
| H3 | `docs/CORRECTIONS.md` row 10; FINDINGS §39; `research/inference-cost/RESULTS.md`:181 | "`common.balanced` agrees … **It is 9/12**. Verified from the raw file: **episode 2 disagrees**" | **7 of 12**, disagreeing at episodes **2, 3, 6, 8, 9** (`research/inference-cost/out/inproc.json`). Verified independently. A correction section, whose entire purpose was replacing a summary figure with a raw one, replaced it with a second wrong figure — and wrote that figure into the track's RESULTS.md as "CORRECTED" | **high** |
| H4 | `docs/CORRECTIONS.md` row 8; FINDINGS §20 | "A '3.00/4 constant baseline' → **Did not reproduce; measured 2.00/4**" | **3.00/4 is what reproduces.** On the recorded protocol (`split='test'`, indices 30000–30015, objectives cycled as `tcn/cli.py:_demo_joint` does) always-True scores **3.00**, always-False **1.00**. I searched three splits × three index ranges and two fixed-objective variants; **no protocol yields 2.00/2.00**. `docs/VALIDATION.md`:107 and the shipped demo both say 3.00. So the *correction* is the unreproduced claim, and CORRECTIONS row 8 is wrong | **high** |
| H5 | FINDINGS §14 (first, "Discrete perception"); `STATUS.md` B9 | "**Requiring exactness at every position of a validation split is what fixed it**"; STATUS: "requiring exactness on a validation split fixed both" | **Inverted.** `research/discrete-perception/out/tiebreak.json`: `lexicographic` and `validation_filtered` return the **identical selection vector**, identical `wrong_slots: 3` of 384, identical `accuracy: 0.9921875`. Only the `gradient` rule reaches `max_error 0.0`. The track's own `RESULTS.md`:62 states: "**Filtering the conforming set through a second supervision split does not fix it.**" FINDINGS reversed its own source | **high** |
| H6 | FINDINGS §19 | "Trained only on lengths {2,4,6} … **it is exact at lengths 8 through 16**" | `final_eval.json` `per_length = {"8":1.0,"10":1.0,"12":1.0,"14":1.0,"16":0.5}`. The single error in 724 **is** the length-16 case, so the two halves of §19's own paragraph contradict each other. §45 knows this ("the single pre-audit error sits at length 16"); §19 does not | **high** |
| H7 | FINDINGS §21 | "The **15-context by 12-target** ceiling table agrees — advantage **exactly 0.0000 at every context** for `object_ids`, `raster_rank` and `is_object_0`" | `research/object-identity/out/bounds.json` holds **210 rows = 15 contexts × 14 targets**, and advantage is non-zero in **8 / 8 / 2** contexts respectively, up to **+0.2018** (`eqbg_cross4` / `raster_rank`). The track's own report says "every **raw-byte** context is at advantage 0.0000" — FINDINGS dropped the qualifier that makes it true. §21's *structural* certificate (permutation) is independent and stands | **high** |

### 1.2 Medium — figures quoted at a scope the raw does not support

| # | where | claim | raw value |
|---|---|---|---|
| M1 | §36, §43, `README.md` | "**19–60 MB RSS, 51–410 ms cold start**, every artifact correct"; "typed side wins deployment footprint **across the board**: 24–60 MB" | The range covers mixed/language/computer and silently omits `visual.pyz`: **377.5 MB RSS, 5,179 ms cold start** (`inference-cost/out/pyz_visual.json`) — larger than the **270.8 MB** torch baseline it is compared against. §43's own `tcn_visual.json` records `peak_rss_mb 405.13` in-process |
| M2 | §43 | "against ~270 MB and **1.7–7.2 s** just to import torch" | `neural-baselines/out/latency.json`: `import_torch_ms` spans **828–9,587 ms** (0.83–9.59 s); per-arm medians 2.59 / 4.23 / 6.00 s. No grouping reconstructs 1.7–7.2 |
| M3 | §48, §41, `README.md`, `STATUS.md` §5 | "the parse's **27.6×** decomposes as **11.1×** more elementary operations × **2.25×** per-bytecode cost" | 11.052 × 2.249 = **24.86**, which is `attribute_visual.json`'s own `C_over_D_time`. The 27.6× is `bench_visual.json`, a different D timing (0.186 ms vs 0.209 ms). The decomposition does not multiply to the total it decomposes, in four documents |
| M4 | §41 | "certifies `none exists` at spans **4 through 29**" | Only spans **4, 8, 12, 16, 20, 24, 28, 29** were enumerated. **18 of 26 spans in the stated range were never run.** Plausible by monotonicity, but §41's framing is "certified rather than argued" |
| M5 | §47 Q2 | "both scaffolds select the **identical `c` at every seed**, unchanged by 5× the budget … **the seed-for-seed identity is what does** [the separating]" | Holds in **3 of 4** arm pairs. At `tau_lt=128, steps=3000`: dyck `[0,47,48,47,48,48,0,47]` vs counting `[62,62,62,61,62,62,62,61]`. Also `symbols` first-step gradient is exactly 0.0 only at `tau_lt=0`; at 128 it is 1.80e-08 / 8.29e-09. The Q2 *conclusion* survives (identical median 0.4738, 0 conforming in all 8 arms); the sentence carrying it does not |
| M6 | §43, `README.md` | "The typed program takes **20/20 rectangles, 20/20 parent links** and an exact tree **on every held-out screen**" | Per-screen widget counts are **15, 17, 17, 18, 20 × 8** across 12 screens; the totals are **227/227** and **12/12**. The verdict (perfect everywhere) is right; "20/20 on every screen" is true of 8 of 12 |
| M7 | §36 | "It gzips to **149 KB, consistent with the 206× compression measured in §35**" | 149 KB is the **minified-then-gzipped** figure = **830×**; 206× is the as-shipped `.pyz` gzip (0.573 MiB). `inference-cost/RESULTS.md` separates them explicitly and calls the 149 KB "another 4×" |
| M8 | §36 | "`visual.pyz` is **117.7 MB of which 99.68%** is repeated type declarations" | `type_declaration_share_of_minified = 0.9968` — the share of the **35.3 MB minified** JSON. 88 MB of the 117.7 MB is `indent_overhead_bytes` |
| M9 | §56 | R5→R6 log share "**−3.1%**" | Raw ratios give **−5.93%**. §56's own aggregates (93.7% representational, 60.7% strict) both *require* −5.9, so the section is internally inconsistent with itself |
| M10 | §53 | "pre-registered criterion 4 failed … **4 of 24 cells** sit above by 0.0625–0.1875" | **6 of 24** cells exceed the best constant. The section understates its own disclosed criterion failure |
| M11 | §57 | "C-minall: O1 **11 of 15**, O2 **11 of 15** — a tie" | Not reconstructible. `ranked_C-minall.json` + `sensitivity.json` give the window at rank 1 in 1 of 6 corpora and 0 of 4 grid cells; the denominator is also inconsistent with B1's "10 of 10" in the same row |
| M12 | §52 | table presents `arm2_syntactic` and `arm4s_runnerup` as two distinct controls | `class-identity/out/q2_partition.json`: both are class `tt/3/57`, truth table `[0,1,0,1,0,1,1,1]` — **one class**. §55 found and reported this; §52 is still uncorrected and carries no cross-reference |
| M13 | §57 | "**4 wrong-module arms**, including three distinct semantic classes" | C-trace has **five** distinct wrong-module modules; on C-minall `arm4s_runnerup` and `arm4s_matched` share digest `1e8b0e63e5f636a7e47b08cd` — the same duplicate-control defect §55 found in §52, undisclosed here |
| M14 | §25 | carrier scaling collapses the loss spread "by **two to three orders of magnitude**" | 10.009 → 0.2142 = **46.7×** ≈ 1.7 orders. The track's own text says "**47×**" |
| M15 | §17 | "**77%** of a step is contract — **0.394 ms** of MuJoCo inside a **1.68 ms** typed step" | `external-environments/out/replay_check.json`: **75.4%**, **0.423 ms**, **1.72 ms**. The quoted trio comes from the track's prose table, which disagrees with its own shipped JSON |
| M16 | §17 | "**all three** `artifacts/system/*/episode.json.gz` now fail `Host.restore`" | `STATUS.md` B5 says "all **eleven**"; the live tree has **11**. `artifacts/` is gitignored, so an external clone can check neither |
| M17 | §24 | old stream "**bit-identical** to the pre-audit generator over 17,184 episodes" | Content-identical 179/179; **`ids` digests differ on 179/179**. §24 elsewhere uses "identical **including instance ids**" for the hardened comparison, so the distinction is load-bearing |
| M18 | §11 | 12/12 seeds "at **2×2 and 3×3**" | `rung3_colour.json` keys are `R2` and **`R4`**; the track's report says "2×2 and **4×4**". The 2×2/3×3 pair belongs to the *mask* arm |
| M19 | §14 (first) | "**Every** rung-3 arm has **2,464–2,608** of 32,000 conforming"; "about **5%** disagree on fresh episodes" | Raw range across all arms is **1,664–2,752**. The 5.19% figure is R=4; **R=8 is 10.4%** |
| M20 | §14 (first) | tie-break "measurably wrong on fresh episodes at **both R=4 and R=8**" | `tiebreak.json` `rows` has **length 1**, resolution 4 only, despite `arguments.resolutions=[4,8]`. The R=8 row was never written |
| M21 | §14 (first) | RGB lookup "**fits training pixels perfectly**" for `object_ids` *and* `depth` | `depth`: train accuracy **0.9863** / **0.9883**, with 4–5 colliding keys |
| M22 | §14 (first) | "rung 3 at R=4 and R=8, **4/4 and 6/6** seeds exact" | Both are **4/4**; the only 6/6 is a different R=4 file |
| M23 | §16 | "M3, `index` kernel locality … **5 to 25** local minima, basins **1.3 to 2.5** wide" | `address-wall/out/landscape.json` per-run `local_minima` spans **1–81** and `basin_width` **0.09–14.02**. The quoted band is a selected sub-range presented as the kernel's property |
| M24 | §33 | "over all **8,385** held-out pairs it is **0.9957**" | True, but the trivial always-different baseline on that same set is **0.9741** and is recorded in the same object. The 0.500 baseline §33 quotes belongs only to the 80-pair balanced sample |
| M25 | §25 | "The language track is not unblocked: still **0 of 44**" | `core-gradient-fixes/out/language.json` records **36 runs, 22 of them post-fix**. There is no post-fix set of 44; the 44 is §19's pre-fix N |
| M26 | §17 | "`role="byte"` admits **only `eq` and `pack`**" | `out/operator_legality.json` lists **three**: `count`, `eq`, `pack` |
| M27 | §26 | "against **2.18e4** for the **best** constant" | 2.18e4 is `predict_zero`; the **best** constant is `predict_train_mean` at **2.17e4** |
| M28 | `STATUS.md` §1 row 3, `docs/VALIDATION.md`:401 | "**3.38–4.00** on gate families never trained on", reproduce with `scripts/demo.sh --only structure` | The quick pass (the command given) yields **3.25**. `--full` yields 4.00. Separately, 3.38–4.00 is the range of **8-seed arm means**; per-seed values in `run.log` reach as low as **2.75** |
| M29 | `STATUS.md` row 6, `docs/VALIDATION.md`:404 | positional reuse "at **0.93×** execution cost" | The demo measures **0.91×** at N=32 (322 / 353) in both quick and `--full`. 0.93× is the ratio at **N=4** in `positional-reuse/cost.json`, quoted beside an N=1,024 headline |

### 1.3 Low — stale, mis-cited, or missing an available baseline

| # | where | issue |
|---|---|---|
| L1 | `README.md`:112 | "the persistent module library and curriculum artifact flow (**section 25**)" — §25 is *The substrate fixes are merged* and contains neither. The work is `research/module-library/` on an unmerged branch with **no FINDINGS section at all**, and `STATUS.md` B1 states flatly "no registry outlives a single script … There is no curriculum stage that consumes a module another stage produced." README asserts as "Holds up under measurement" something STATUS says does not exist |
| L2 | `README.md`:114, `HANDOFF.md`:43, `research/AGENDA.md` | the crystallization refutation cited as "**sections 7, 12, 37**". §7 is *differentiable search is not earning its keep*; §12 is *recursive abstraction re-tested*. Neither refutes crystallization. The four refutations live in §1/§3, §10 and §37 — and two of them (`perturbation-selection`, `loss-gated-eligibility`) have **no FINDINGS section** |
| L3 | `README.md`:121, `HANDOFF.md`:39 & :71, FINDINGS §45:2261 | **§42 does not exist and never has.** `git log --all -S'## 42.'` over `research/FINDINGS.md` returns nothing, and `compiled-runtime`'s FINDINGS stops at §38. HANDOFF's explanation ("§42 is on branch `compiled-runtime` and arrives when it merges") is false; the content is on main as **§48**. `research/MERGE-QUEUE.md` shows §42 was a *planned* renumbering in a merge dry-run that was superseded |
| L4 | `research/FINDINGS.md` | **§14 is used twice** — "Discrete perception: rung 3.5" (line 536) and "Preference: the objective can now see program size" (line 782). §41 and `CORRECTIONS.md` row 14 both cite "§14" and mean the second |
| L5 | `docs/CORRECTIONS.md`:8 | preamble says "**Nine** wrong conclusions have been caught so far"; the table has **20 rows**. Rows are also ordered 1–14, then 20, 19, 18, 17, 16, 15 |
| L6 | `STATUS.md` | "**179 Python tests pass**" (337 collected today); "**370 of 372** computer-engine tests pass" (**372/372** today) |
| L7 | `STATUS.md` §3 and B2 | "`research/policy-learning/` has … **no `RESULTS.md`**", "Treat this track as raw data, not a report" — it has had one since §22. Also "`object-identity/RESULTS.md` still carries three unrendered `{{ }}` placeholders" — there are **zero** |
| L8 | `STATUS.md` B7 vs its own §"How to check any of it" | header: "is fixed (blocker **B7 closed**)"; B7: "**The documented entry point does not run**". The shebang is fixed (`typed-crystallization-networks/.venv/bin/python3`). STATUS.md contradicts itself |
| L9 | `HANDOFF.md` | "expect: **288 passed**" (337); "FINDINGS.md — **44** numbered sections" (61); "CORRECTIONS.md — **14** reversals" (20); "caught **nine** wrong conclusions" (20); open item 1 restates **§44's retracted mechanism** ("exact minimisation fuses the reusable fragment … MAJ3 survives in 1 of 6") that §46 and CORRECTIONS row 15 overturned, and prescribes work §46/§52/§54/§57 have since done; open item 4 ("the language capability on the post-audit stream — never measured") was closed by §45 |
| L10 | §12, `README.md`, `STATUS.md`, `docs/VALIDATION.md` | "module on the output path in **27/27** successes" is quoted with no chance rate. `recursive-abstraction-retest/baseline.json` records the null for the same metric over 20,000 trials: **0.6886** (wide) and **0.816** (tight); `tcn/cli.py:472` hard-codes `chance_rate_module_on_output_path: .816` with **no provenance anywhere under `research/`** |
| L11 | §21 | collinearity transfer "held-out **1.000000**" quoted without the majority baselines that sit in the same file (**0.9290** / **0.9489**) |
| L12 | §44 | "`MAJ3` survives in **1 of 6**" is order-dependent — `order_robustness.json` gives 2 of 6 under a different `and/or/xor` declaration order; not stated |
| L13 | §54 | O1 "flips — **two** digests" across "all six corpora" — there are **three** distinct rank-1 digests; and "B1/B2 … one" bundles B2, which flips |
| L14 | §51 | cold-start "**3.9×** faster" — median gives 3.833×, min 3.491×, deploy 3.597×; no denominator yields 3.9. "§49's 0.67× and 0.63× reproduce **exactly**" — the raw flag is `worst_ratio_reproduces_to_2dp`, and the underlying counts differ |
| L15 | §55 | "at **widths** 5, 7 and 12" contradicts §55's own opening "widths 15, 21 and 36 — depths 5, 7, 12" |
| L16 | §46 | "a richer corpus under syntactic identity still ranks it **11th to 18th**" — `C-plus1`, richer than `C-min`, ranks it **65th** in §46's own table two paragraphs above. "**79×** more" — the same sentence's detail gives 72× nodes / 82× CPU |
| L17 | §12 | "track 5: module on the output path in 0 of 20 arm-B runs, **0 of 10 successes**" — `recursive-abstraction/e1_results.json`+`e2_results.json` give **9** successes (2/12 and 7/8) |
| L18 | §14 (second) | pruning "preserves semantics over **3,875** row comparisons" — `prune_fixtures.json` sums to **5,475** |
| L19 | §15 | "exact … on 40 episodes at each of depths **1–8**" — `exactness.json` holds only depths 1, 2, 3, 4, 6, 8 (six, 240 episodes). "the soft model scores **3.20**–3.31" — the floor is **3.19** |
| L20 | §34 | "**7.2 hours** to 4.4 seconds" — `run.json`'s own field is **80,530.6 s = 22.4 h**; 7.2 h is the second of two documented projections, quoted without saying which |
| L21 | §26 | the legality table's "**symbol → intensity illegal**" row was never measured — `legality.json` has no `symbol` key |
| L22 | §49 | "20-of-961 = 2.08%" is not in `lazy-guard/out/`; it lives in `lazy-latency/out/crossover.json` |
| L23 | §13 | "enumeration … exhausts in 130 s" — `nondegenerate-generalization/out/results.json` records **no `exhausted` and no `unique` key**; exhaustion is inferable only from `evaluated == space_size` |
| L24 | §23 | "the default observation set … **across 27 configurations**" — no configuration count is recorded in any file |
| L25 | §19 | "0% string overlap" is certified for an **n=242** subset at lengths 8–14, not for the 724-episode set; and §19 omits that the **stage-B** search was `unique: false` with **10 conforming**, and that declaration-order ranking picked the member that fails at length 16 |
| L26 | §37 | the table's 2.938 / 2.531 are `post_crystallization.mean_return` while the prose's "3.625 against 2.594" are frozen returns — two metrics in one section, unlabelled. "**Step-matched** argmax … 9.3× fewer environment episodes" — 9.3× is against `argmax@shipped`; step-matched is **2.7×** |
| L27 | §59 | "`_m1` … **73.7%** of bytecodes" (raw: 76.0% new / 72.0% old); "language **−8.45%**, visual **−5.42%**" (raw totals: −7.76% / −5.30%) |
| L28 | §58 | "*(The track's summary said 5; I measured 7–10)*" — the track reports **both**: 5 is the `S` identity rule, 7–10 is the `D` rule. Not the same rule; the track was not wrong |
| L29 | §10 | "these read **1.007e-06 and 0.0 respectively**" — the ordering is inverted relative to the sentence that introduces them |
| L30 | §36 | "`Type.decode`, `Type.encode`, `validate_raw` **and their guards** account for 97.7%" — those four roles sum to **93.8%**; the 97.7% also includes `Type/Value dict round-trip` |

**Manufactured-discrepancy discipline.** These were checked and are **not**
errors: §17's "never exceeds −0.99" against a measured −1.0000 (a conservative
true bound); rounding such as §56's 2.834 or §24's "about 750,000" for 716,000;
§54 vs §57 (family-scoped, see §2.4); the demo's `structure` seen == unseen,
which trips the §4/§40 signature and is then **cleared** by §40's own
discriminator (the two pools' constants differ, 2.25 vs 2.00, so the override is
live; it is a single-seed coincidence — `verify.py` implements exactly this
two-stage test); and §45's 680,625-program run stopped at ~52%, which **no
section quotes as certified**.

---

## 2. Pass 2 — internal consistency

Verdict format: **ERROR** (one section is wrong) or **DISTINCTION** (both stand
under a distinction that must be stated).

### 2.1 §30 "selections transfer" vs §60 "that is a within-domain property"

**DISTINCTION, already stated by §60 — but unmarked at the source.**

§30's claim is scoped to *widths of one observation channel inside one domain*
(R=8/16/24/32 of the same raster; `object-identity/out/apply.json`, one wrong
slot in 5,520). §53 extended it to six widths of one channel, still one domain.
§60 tested it across a **domain boundary** and found the schema instantiates
bit-identically at all three widths while the certified vector scores **0
conforming on `T_C2` where no-library scores 1** — and re-selecting the vector
enumerates the identical 750 programs, buying nothing.

The distinction is: **a schema is width-polymorphic; a selection vector is
width-polymorphic only within the domain it was fitted in.** §60 says this. The
defect is that §30 and §53 carry no marker, and §53's phrasing is the stronger
one — "Selections transfer completely — the cheap outcome, and **it is the
answer**" — which §60 refutes across domains.

### 2.2 §56's 2.834× vs §59's 1.10× and §61's 1.08×

**DISTINCTION on the measurements, ERROR in §56's forward claim.**

The three numbers measure different things and §59/§61 both say so:

- §56's 2.834× is one rung of a **cumulative source-transform ladder** on the S2
  subroutine, from a **post-LICM** baseline, and R5 **bundles a copy-propagation**
  that §59 excludes.
- §59's 1.1073× (language, banked; visual not banked) is an **emitter pass in
  `tcn/compile.py`**, which already inlines its guards. The same elimination on
  the same S2 subroutine in isolation is **1.1355×**.
- §61's 1.084× requires a semantics-preserving rewrite plus a second
  off-by-default flag; declaring the bound alone is **29% slower**.

Not contradictory. But §56 also asserts, unconditionally, "**that engineering
closes the gap completely**" and "the next gain is in `tcn/compile.py`'s
emitter". Two subsequent implementations failed to realise it, and §61 names the
lesson: the attribution "has now **twice** failed to predict the gain from an
actual implementation." §56 carries no bounding marker while §54 does. Separately
**M9 above is a genuine arithmetic error inside §56** (−3.1% should be −5.93%).

### 2.3 §46/§52 semantic pooling vs §57's `C-minall` inversion

**DISTINCTION, and §57 states the mechanism — but §52's status line is
over-general.**

On family 1 (MAJ3, arity 3) both corpus bands agree: pooling reaches the
hand-authored ceiling (144 conforming, `complete`, 18/24, 8/8) and every
wrong-module control scores 0. On family 2 (`W4`, arity 4) the bands **split**:

| band | no library | syntactic | pooling | ceiling |
|---|---|---|---|---|
| C-trace (136 entries) | 0 | 0 | **48** | 48 |
| C-minall (10 entries) | 0 | **48** | **0** | 48 |

§57's mechanism: **pooling buys rank in proportion to fragmentation, and where
the competitor is itself the fragmented class it loses.** That is a real
boundary condition, not a contradiction.

The defect is that §52 closes with "Treat semantic pooling as an **established
improvement to abstraction identity**" and carries no `[BOUNDED BY §57]` banner.
The supportable statement after §57 is: *established on family 1 on both bands
and on family 2's high-fragmentation band; refuted on family 2's `C-minall`
band.* §52 also still presents one class as two controls (M12).

### 2.4 §54's ranking result vs §57's non-replication

**DISTINCTION, and it is the one the record already handles correctly.**

Both directions verified from raw. Family 1 (`reuse-ranking/out/heldout.json`):
O2 helps 5 of 5, B1 helps 0 of 5, O2's rank-1 digest stable across all six
corpora. Family 2 (`second-family/out/`): O1 and O2 pick the identical rank-1
digest on both bands (a true tie) and B1 helps 5 of 5 on both.

Different arity, different corpora, different eligible sets — **family-scoped,
not contradictory.** §54 carries `[BOUNDED BY §57 — read that before citing this
section.]`, which is the model the other four tensions should follow. The only
defect here is §57's own unreconstructible "11 of 15" cell (M11).

### 2.5 §55's "premature machinery" vs §58's "prerequisite for the objective"

**DISTINCTION — they name different objects — but §58's forward claim is
separately falsified by §60.**

§55's "a core change is premature" is about a `semantic_id` field in
`tcn/library.py`'s manifest: the entire 272× saving came from a **240-line
sidecar with `tcn/` untouched**, so nothing yet *needs* the core field. §58's
"prerequisite" is about the **schema-plus-instantiation mechanism** — which §55
itself built, as a sidecar. Both stand: *the mechanism is a prerequisite; the
core change is not.*

What does not stand is §58's framing that the mechanism "makes §53/§55's schema
mechanism … **the prerequisite for the owner's stated objective** of reusable
factorizations across visual, language and computer-use." §60 tested exactly
that and the certified class **scored zero where no-library succeeded**; the
hand-authored equivalent also scored 0, so the failure is the vector, not the
structure. §58 carries an inline correction about the 3-node motif but none about
this sentence.

### 2.6 Tensions found that the brief did not list

- **§20 / CORRECTIONS row 8 vs `tcn/cli.py` + `docs/VALIDATION.md`** — H4. This
  is an unresolved live contradiction between the record and the shipped code,
  and my measurement puts the record on the wrong side.
- **§12's 19/24 vs §44's 18/24 for what reads like the same arm.** *Not an
  error.* Verified from two independent raw sources:
  `recursive-abstraction-retest/results_tight24.json` gives arm B **19/24** at
  300 steps and `abstraction-preference/tables.json` independently reproduces
  **19/24** at `mdl_weight=0`; `earned-abstraction/out/gradient_tight.json` gives
  arm 3 **18/24** at 400 steps with `candidates_per_node [336,336,24]`. Different
  scaffolds and budgets. **§12's headline 27/27 = 8 wide + 19 tight is verified
  from raw** (`results.json` + `results_tight24.json`, every success has the
  module on the output path).
- **§1–§9 carry twelve verdicts that later sections overturned, with exactly one
  forward marker** (§3's "See section 13"). The worst is §5's header "**Core
  changes proposed, none applied**" sitting immediately above §10 "**Fixes
  applied**", and §5's recommendation 10 ("give `synthesis.fit` a learning-rate
  schedule"), which §10 measured as "**the wrong fix and … measured worse**".
  Others: §1 and §6 ("pure policy-gradient learning fails at chance for the typed
  program") overturned by §22; §3's "Recursive abstraction helps → No measurable
  benefit" overturned by §12; §6's "Depth generalization **requires** a recurrent
  or set-shaped gate encoding" overturned by §15; §4's F-conf/F-bench/F-soft
  listed as live faults, all fixed in §10; §3's "450×" superseded by §36's 146×.
  For an external evaluator §1–§9 read as the executive summary.
- **`docs/VALIDATION.md` §3 stops at correction 3.12**, covering roughly
  `CORRECTIONS.md` rows 1–8. Rows 9–20 never reached it, which is why H2 is
  still on the page.

---

## 3. Pass 3 — the shipped demonstrations

`scripts/demo.sh` (via `python -m tcn demo`) run twice. Exit status **1** in both
modes, correctly.

| # | demo | quick (60.2 s) | `--full` (192.0 s) | STATUS.md §1 claim | verdict |
|---|---|---|---|---|---|
| 1 | synthesis | PASS — held-out max err **5.93e-08** on 584 points | PASS | "≤ 1e-7 on 584 unfitted points" | **reproduces** |
| 2 | (enumeration cross-check) | unique of 96 in 2.67 ms, agrees | same | "unique program of 96, ~7 ms" | reproduces; the 7 ms is §10's figure, measured 2.7 ms here |
| 3 | structure | PASS — unseen **3.25** (seen 3.25), constant 2.00 | PASS — unseen **4.00**, constant 2.5625 | "**3.38–4.00**, both split directions" | **quick pass falls outside the quoted range** (M28) |
| 4 | depth | PASS — 4.00 at d1,2,3,4,6,8; constants **2.00–2.75** | PASS — constants **2.06–2.50** | "best constant **2.06–2.50** per depth" | reproduces under `--full` only; the reproduce column names the quick pass |
| 5 | abstraction | PASS — arm B conformant at step 20, module on output path | PASS | "27/27; arm B 8/8 wide, 19/24 tight" | reproduces (recorded rows verified from raw; chance rate 0.816 omitted from the record — L10) |
| 6 | positional | PASS — 1,024 positions / 3,072 values / 3 callers | PASS — **2,304 / 6,912** | "1,024 … `--full` does 2,304 over 6,912" | reproduces; **0.91× not 0.93×** (M29) |
| 7 | segmentation | PASS — max err 0.0 on 48, unique of 65,536, constant 0.854 | PASS | identical | **reproduces exactly** |
| 8 | edge | PASS — acc 1.000, unique of 48, 4.9e10 / 7.6 y, constant 0.844 | PASS | identical | **reproduces exactly** |
| 9 | control | PASS — replay/restore max \|Δ\| 0.0, upright 0.9994 | PASS | identical | reproduces (zero-torque measured **−1.0000**; "never exceeds −0.99" is a true weaker bound) |
| 10 | **language** | **FAIL** — `ValueError: training examples required` | **FAIL** — identical | "stage A … unique among 10,496; stage B **0.9986** … reproduce: `scripts/demo.sh --only language`" | **does not reproduce** (H1) |
| — | joint | PASS — 4.00/4, loss 0.24884 → 0.00223 | PASS | matches | reproduces, but prints a baseline the record retracted (H4) |

**On the language capability specifically.** §39/§43/§45 established that it is
**0.9986, not 1.000**, and reproducible **only with `hardening='none'`**. Where
that is quoted today:

| file | number | stream named? |
|---|---|---|
| FINDINGS §19 | 0.9986 ✓ | yes, and cites §43/§45 ✓ |
| FINDINGS §39, §43, §45 | 0.9986 ✓ | yes ✓ |
| `STATUS.md` row 9 | 0.9986 ✓ | "**pre-audit stream only**" ✓ |
| `docs/PROJECT-LOG.md` | 0.9986 ✓ | yes ✓ |
| **`README.md`:128** | 0.9986 ✓ | **no** |
| **`docs/VALIDATION.md`:407, :516** | **1.000** ✗ | **no** |
| `HANDOFF.md` | — | carries the rule ("never quote §19's number without saying which stream") ✓ |

Two of the seven places fail the rule §45 set, and one of them still carries the
retracted number.

**Framework health.** 337 Python tests collected: 336 pass, 1 fails
(`test_panel_interface::test_panel_episode_replays_and_restores`). Verified
environmental exactly as `research/MERGE-QUEUE.md` documents — removing the
`node_modules` symlink gives **13 failed, 324 passed**; symlinking it gives **1
failed, 336 passed**, which also reproduces §41's verification note. Node engine:
**372/372**.

---

## 4. Pass 4 — claims resting on one family, one width, one seed set, one configuration

§53's arm F and §60 both established that a *deliberately wrong* schema earns the
same `unique` certificate at a single width, and §57 showed a §54 result that
vanished on a second family. This is the inventory that tells a reader which
results are robust and which are anecdote-strength.

### 4.1 Robust — replicated across at least two widths, families, or domains

| claim | breadth |
|---|---|
| §53 width-polymorphic selection vector | `unique` at **six** widths (3–24), all six exhausted, selections identical |
| §55 stored-class instantiation | certified at 2 widths, applied at 3 unseen; refuses both unsound schemas by two independent rules |
| §14/§30 positional reuse | R=8, 16, 24, 32 and R=8→48; one wrong slot in 5,520 |
| §12 recursive abstraction | two scaffolds, 8 + 24 seeds, three arms, two independent tracks reproducing 19/24 |
| §58 cross-domain negative | bounded by a **type-signature argument independent of search** (2 of 131 signatures cross a boundary), plus a live positive control |
| §24 lesson audit | 179 lessons × 4 runs, null control, fully reproduced here from the track's own `report.py` |
| §11/§18 staged perception | replicated on a **second generator** (`gui`) with three ablated spaces returning the identical function |

### 4.2 Anecdote-strength — one family, width, seed set or configuration

| # | claim | rests on | disclosed? |
|---|---|---|---|
| A1 | §13 structural generalization 3.38–4.00 | **one** Boolean gate-table pool split, one generator, 8 seeds; the demo's single seed gives 3.25 | no |
| A2 | §34 code generation, "conforming = 1, certificate `unique` on both heads" | one target shape (Shannon skeleton) at **width 3**, one address layout. The track's own negative control earns 16 conforming per head from a different slice — the certificate is slice-dependent | partly |
| A3 | §32/§33 visual parse, "**S2: 25/25, 1 conforming, `unique`**" | a **25-program** space at **one** screen configuration (resolution 32, palette 32, nesting 5, min_size 4). No second resolution or palette; §33 explicitly declined to raise the palette | no |
| A4 | §19/§45 language | **one lesson of 179**, one scaffold, one stream. Stage A is `unique`; **stage B is `unique: false` with 10 conforming** | stream yes, stage-B non-uniqueness no |
| A5 | §23 computer use | one task, one document family, one generator config; "address unique in a 136-candidate space" is conditional — the `SearchResult` is `unique: false, conforming: 2` | partly |
| A6 | §29/§31 credit assignment | one `interface='panel'` configuration; §31's 0.9648 is **4 seeds** against a **2-seed** control; §29's `flat`/`probe_only`/`reward_percept` controls are 3/2/3 seeds quoted as bare point values | §31 yes, §29 no |
| A7 | §22 policy learning 8/8 | one 256-way choice on a generator whose state is constant; §22 itself says no result here has ever measured credit assignment | yes |
| A8 | §54 ranking (O2 5/5, α ∈ [0.08,3], O4 52/72) | **one family, seven tasks** | yes — `[BOUNDED BY §57]` |
| A9 | §52 semantic pooling | one family, one scaffold geometry; the 13-class sweep is one corpus | no |
| A10 | §57 replication | holds on **one band** (C-trace), inverts on the other | yes, it is the finding |
| A11 | §55 272× saving | **one class**, one task family, `certificate: none` at every instantiation; F-b fires (the ratio *is* the space size) | yes |
| A12 | §56 residual-gap ladder | one program pair, one subroutine; the "worst case" rung is **n = 1** | no |
| A13 | §49/§51 laziness | one subroutine + two miniatures; the 2.48× rests on **54 records**; worst case is a single record | partly |
| A14 | §59 emitter guards | 4 artifacts, **one banked** (language); visual explicitly not banked | yes |
| A15 | §61 refinement bounds | one artifact, one function (`_m1`), behind an off-by-default flag | yes |
| A16 | §50 scaffold diagnosis | 3 tasks | yes — "anecdote-strength and the track says so" |
| A17 | §21 collinearity | one renderer (the `world_2d`/`world_3d` transfer is the *same* renderer) | partly |
| A18 | §26 Sobel via `interpret` | 4 seeds, one kernel, one generator | no |
| A19 | §14 rung-3.5 edge detector, `unique` of 48 | one configuration — and the **sibling subsampled run on the same 48-program space returns `conforming: 0, unique: false`** | no |
| A20 | §18 "chosen over a distractor", `unique` | a **two-element** space at one configuration | no |
| A21 | §21 "caller nodes now discovered … exhausted, unique" | **eight** programs, one run, one resolution | no |
| A22 | §16 M1's "unique global optimum" | 3 seeds, n=4 addresses, one regime; the same field is 19/21 once other regimes are included | no |
| A23 | §27 "reward proves 1 of those 2", `unique` | a residual space of **2**, one setting | no |
| A24 | §12's exhaustion certificates | **tight scaffold only**; the wide arms' spaces (1.93e22, 3.46e24) were never enumerated, while the prose sits under a wide/tight table | no |
| A25 | §60's non-transfer negative | **one** target task `T_C2`, n_test = 20 | no |

**Nine single-configuration `unique` certificates (A2, A3, A19, A20, A21, A23,
plus §10's mixed 1/96, §11's 65,536 at R2/R4 and §41's visual S2 1/25) are
presented as evidence in prose.** §53's own standing caution — already in
`docs/CORRECTIONS.md` — says they are not. Only §53, §55 and §60 apply the
two-width rule.

---

## 5. Recommended corrections

Recommendations only. **I have not edited `FINDINGS.md`, `STATUS.md`,
`README.md` or `CORRECTIONS.md`**, so that each correction is reviewed by the
supervising session. Ordered by damage to an external evaluator.

### 5.1 Do first — a shipped claim is false today

1. **`STATUS.md` §1: change "Ten capabilities reproduce" to "Nine of ten".**
   Row 9's reproduce column must say the language demo **fails on the current
   tree** with `ValueError: training examples required`, and why (§39: the
   post-audit default stream emits no lengths 2/4/6, so stage A's training split
   is empty). Same edit in `docs/VALIDATION.md`:395 ("All of these reproduce")
   and :526. Evidence: `artifacts/demo/summary.json`, `verify.py`
   `pass3/language`, reproduced in quick and `--full`.
   *The right fix is arguably in `tcn/cli.py:_demo_language` (pin
   `hardening='none'`, as `research/program-length` did) — but that is a `tcn/`
   change and outside this track.*

2. **`docs/VALIDATION.md`:407 and :516 — replace 1.000 with 0.9986 and name the
   stream.** This is CORRECTIONS row 9, uncorrected in the document README
   directs readers to. Add a `CORRECTIONS.md` process row: *"§43 recorded that
   'all three files are corrected'; a fourth file carried the same claim and was
   missed. When correcting a figure, grep for it."*

### 5.2 Two rows of `CORRECTIONS.md` are themselves wrong

3. **Row 10 (and FINDINGS §39, and `research/inference-cost/RESULTS.md`:181):
   9/12 → 7/12, and "episode 2 disagrees" → "episodes 2, 3, 6, 8 and 9
   disagree".** Evidence: `research/inference-cost/out/inproc.json`
   `language.episodes[*].agree`. Verified independently; `verify.py`
   `pass1/§39-inproc-agreement-count`. §48's "its RESULTS.md claims 12/12" is
   also now stale in the other direction.

4. **Row 8 and FINDINGS §20: retract the retraction.** On the recorded protocol
   the best constant on the joint result's 16 episodes **is 3.00/4**; always-True
   3.00, always-False 1.00. No split, index range or objective assignment I
   tested yields 2.00/2.00. `docs/VALIDATION.md`:107 and the shipped demo already
   say 3.00. Evidence: `verify.py` `pass1/§20-joint-constant-baseline`, which
   reimplements the protocol from `tcn/generation.Host` directly.
   Recommended wording: *"Row 8 is withdrawn: the 3.00/4 figure reproduces on the
   recorded protocol and the 2.00/4 re-measurement does not. The demo's live
   computation and `docs/VALIDATION.md` agreed with the original throughout."*

### 5.3 Two sections state the opposite of their own raw data

5. **§14 (first): remove "Requiring exactness at every position of a validation
   split is what fixed it."** `tiebreak.json` shows `validation_filtered`
   returning the identical program with the identical 3 wrong slots of 384; the
   track's own `RESULTS.md`:62 says the filter "does not fix it". The rule that
   reached `max_error 0.0` in that file is the **gradient** arm. Propagate to
   `STATUS.md` B9 ("requiring exactness on a validation split fixed both").

6. **§19: "exact at lengths 8 through 16" → "exact at lengths 8, 10, 12 and 14,
   and 0.5 at length 16 — the single error in 724."** `final_eval.json`
   `per_length`. §45 already records this; §19 contradicts itself. While there,
   add that the **stage-B** search was `unique: false` with 10 conforming.

7. **§21: restore the qualifier.** "advantage exactly 0.0000 at every context" →
   "at every **raw-byte** context", and "15-context by 12-target" → "15 × 14".
   `bounds.json` has 210 rows and 18 non-zero advantages up to +0.2018. The
   structural permutation certificate is unaffected and should be the sentence
   that carries the section.

### 5.4 Bounding markers — copy §54's pattern

8. **Add `[BOUNDED BY …]` banners** in the style §54 already uses, at the top of:
   **§30 and §53** (→ §60: the selections transfer is a *within-domain* property);
   **§52** (→ §57: pooling wins where fragmentation is high and **loses** on
   family 2's `C-minall`); **§56** (→ §59 and §61: the 2.834× attribution is
   correct for its ladder and has twice failed to predict an implementation);
   **§58** (→ §60: the schema crosses, the certified vector does not, so
   "prerequisite for the objective" is not supported); **§47 Q3** (→ §50).
   Also add a "SUPERSEDED — see §N" line to each of the twelve §1–§9 verdicts
   listed in §2.6, and rename §5's header, which currently reads "Core changes
   proposed, **none applied**" directly above §10 "Fixes applied".

### 5.5 Numbers to fix in place

9. **§36 / §43 / `README.md`:** state the deployment footprint as **four**
   artifacts, not two. `visual.pyz` is **377.5 MB RSS / 5,179 ms cold start** —
   larger than the 270.8 MB torch baseline. "Across the board" is true of the
   *compiled* path (§48: 30.2 MB, 90 ms) and false of the shipped `.pyz`.
   Replace §43's "1.7–7.2 s just to import torch" with the measured **0.83–9.59 s**.

10. **§48 / §41 / `README.md` / `STATUS.md` §5:** the "11.1× × 2.25×"
    decomposition multiplies to **24.86×**, which is `attribute_visual.json`'s own
    total. Either quote 24.9× beside it or say which run each factor comes from.

11. **§41:** "certifies `none exists` at spans 4 through 29" → name the eight
    spans actually enumerated (4, 8, 12, 16, 20, 24, 28, 29), or run the missing
    18.

12. **§47 Q2:** the seed-for-seed identity holds in **3 of 4** arm pairs; at
    `tau_lt=128, steps=3000` the two scaffolds diverge, and `symbols`' first-step
    gradient there is 1.8e-08, not exactly 0.0. The conclusion survives on the
    identical medians; the sentence carrying it needs the qualifier.

13. **§56:** R5→R6's log share is **−5.93%**, not −3.1% — §56's own 93.7% and
    60.7% aggregates require it. **§53:** the criterion-4 failure is **6 of 24**
    cells, not 4. **§57:** the "11 of 15" cell does not reconstruct from the raw.
    **§52:** annotate that `arm2_syntactic` and `arm4s_runnerup` are one class
    (§55), and **§57:** that `arm4s_runnerup` and `arm4s_matched` share a digest.

### 5.6 Structural repairs to the record

14. **Renumber the duplicate §14** (the second, "Preference", is what §41 and
    CORRECTIONS row 14 cite) and **resolve §42**: it exists nowhere in git
    history. Either renumber §48 to §42, or fix the four references in
    `README.md`:121, `HANDOFF.md`:39 and :71, and `FINDINGS.md`:2261 to point at
    §48 — and delete HANDOFF's false claim that §42 is on branch
    `compiled-runtime`.

15. **`README.md`:** section 25 does not contain the module library or curriculum
    artifact flow (that is `research/module-library/`, unmerged and **absent from
    FINDINGS entirely**), and `STATUS.md` B1 says the capability does not exist.
    Either cite the branch and mark it unmerged, or drop the bullet. Same file:
    the crystallization refutation is §1/§3, §10 and §37 — not "sections 7, 12,
    37" (repeated in `HANDOFF.md` and `research/AGENDA.md`). Add the stream
    qualifier to the 0.9986 at line 128.

16. **`STATUS.md`:** 179 → 337 tests; 370/372 → 372/372; delete the "no
    `RESULTS.md`" and "`{{ }}` placeholders" claims about `policy-learning` and
    `object-identity`; and reconcile B7 with the header that says B7 is closed.
    **`HANDOFF.md`:** 288 → 337 tests, 44 → 61 sections, 14 → 20 reversals, nine →
    twenty caught; open item 1 restates §44's retracted mechanism and describes
    work §46/§52/§54/§57 completed; open item 4 was closed by §45.
    **`CORRECTIONS.md`:** preamble says nine, the table has twenty; rows run
    1–14 then 20–15.

17. **`docs/VALIDATION.md` §3 stops at correction 3.12** (≈ CORRECTIONS rows
    1–8). Rows 9–20 never reached it. Either bring it forward or state at the top
    that `docs/CORRECTIONS.md` supersedes it.

18. **Add the chance rate to every "27/27".** `recursive-abstraction-retest/baseline.json`
    records the null over 20,000 trials: 0.6886 wide, 0.816 tight. The shipped
    demo prints 0.816; the record does not. `tcn/cli.py:472` hard-codes it with
    no provenance under `research/` — that provenance should be added.

### 5.7 What an external evaluator cannot check

19. **State plainly that three sections cite paths absent from this branch:**
    `research/emitter-guards/` (§59), `research/refinement-bounds/` (§61) and
    `research/seasons/` (§37). §37 gives a reproduce command; §59 and §61 do not.
    A reader who clones `main` cannot verify any of the three. (Their data does
    exist on the named branches and was checked there.)

20. **Nine tracks with a `RESULTS.md` are never cited by path in `FINDINGS.md`,
    and seven are never named at all:** `crystallization-ablation`,
    `loss-gated-eligibility`, `search-scaling`, `module-library`,
    `structure-generalization`, `scaffold-autopsy` and `literature` appear
    nowhere; `perturbation-selection` and `positional-reuse` are named once each
    and cited by path never. Two of these are two of the four crystallization
    refutations §37 counts; `search-scaling` holds the 19–38% → 88–94%
    supervision result STATUS calls the most effective technique measured
    anywhere; `positional-reuse` is `STATUS.md` §1 row 6, a shipped demo; and
    `module-library` is what `README.md` mis-cites as §25. FINDINGS calls itself
    "the consolidated record"; these are outside it. (`§1–§9` refer to some of
    them as "track 1" / "track 3", which does not resolve to a path.)

21. **The following figures have no raw file anywhere** and should be marked as
    prose-only or re-measured: §10's `4.0x` / 113.8 µs / 16.4 µs module-call
    costs (which §12 explicitly argues against); §12's "1.4x–5.3x independent
    sweep" and "2-node flat scaffold reads 2.0"; §35's entire gzip-of-`.pyz`
    column (the `.pyz` files are gone and `sizes.json` stores only the
    minified-JSON gzip); §52's "C-trace costs 73× C-minall's DFS nodes"; §46's
    7.6× / 12.9× / 79× cost table; §23's "27 configurations"; §24's four
    sampled-lesson scores (0.580 / 0.205 / 0.230 / 0.300); and **every test-suite
    count in the record** (§25's 235, §26's 243, §27/§28's 264, §44's 287,
    §41's 336, §47's 275, §50's 324, §51's 336, §59's 343, §61's 348) — none is
    recorded in any artifact, and the current tree collects 337. The four `.pyz`
    files §35 measures are also absent: the only `.pyz` on disk are under
    `artifacts/demo*/` and `research/baselines/out/`.

---

## 6. What `verify.py` covers

Forty-nine mechanical checks over six groups, each printing the value it read and
the file it read it from:

- **record/** — duplicate and missing section numbers; dangling `§N` references
  (excluding references into other documents' own numbering); every backticked
  `research/…` path resolving on this branch.
- **retracted/** — claims `CORRECTIONS.md` records as overturned that are still
  asserted somewhere (the language 1.000 in VALIDATION, the unqualified 0.9986 in
  README, `(0, 3069)` as a type bound).
- **status/** — STATUS.md statements a later merge falsified, and its test count.
- **pass1/** — the language accuracy and its per-length table; both §39 stream
  rows; the `inproc.json` agreement count; a sweep of **573 enumeration records**
  asserting no `unique`/`complete` certificate is claimed over an unexhausted
  space (this one passes); the joint constant baseline recomputed from
  `Host` directly; the deployment-footprint band; the visual gap decomposition;
  the per-screen rectangle counts; span-sweep coverage; the §47 seed identity and
  surrogate gradient.
- **pass3/** — every STATUS.md §1 row against `artifacts/demo/summary.json`,
  the "Ten capabilities" sentence, the joint baseline conflict, and the
  seen-vs-unseen signature with §40's discriminator applied so it does not
  false-positive.
- **tests/** — suite result and collected count, tolerating only the documented
  worktree-only failure.
