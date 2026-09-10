# The integrated flagship — does the inherited library make an integrated task cheaper to acquire?

`PREREGISTRATION.md` committed at `22e17f8`, before any arm. `generators/` and
`tcn/` have no diff on this branch. Every figure-bearing table sits between
`<!-- BEGIN:… -->` / `<!-- END:… -->` markers and is written by `report.py` from
files under `out/`; `verify.py` re-renders every block, re-derives each headline
claim from raw JSON without importing the run scripts, and fails on any number in
the prose that does not trace to a block.

**Scope of the evidence, stated once and repeated where it matters:** one seed
set (seed 0), one palette (`palette_levels 2`), one relation vocabulary
(right_of, left_of, above), one colour vocabulary (red, green, blue, yellow), 12
training and 36 held-out episodes, resolution 16. Single-configuration evidence
throughout, with a second configuration (`gap 1`) where stated.

---

## VERDICT

(written after the arms; see §8)

---

## 1. The pre-registered criteria, resolved

Primary configuration (`gap 0`):

<!-- BEGIN:criteria_gap0 -->
- **C1 (material saving, ≥10× in programs and episodes):** N″/N = 10,925.41 → **FAIL** (episodes are programs × 12, so the same ratio).
- **N′ vs N:** N′/N = 1.88 → schema is organisational reuse (within 10×).
- **N″ vs N′ (does the prior earn its name, ≤ N′/10?):** N″/N′ = 5,806.89 → **no**.
<!-- END:criteria_gap0 -->

Second configuration (`gap 1`):

<!-- BEGIN:criteria_gap1 -->

<!-- END:criteria_gap1 -->

## 2. Every arm, exact

<!-- BEGIN:arms_gap0 -->
| arm | full space S | conformers K | certificate | first solvable tier | expected programs to first solution | expected episodes | saving vs N | first solution generalizes (of sampled) | held-out accuracy of first solution |
|---|---|---|---|---|---|---|---|---|---|
| N — flat, uniform | 5.03×10^29 | 5.29×10^21 | `complete` | 0 | 9.50×10^7 | 1.14×10^9 | 1× | 0/400 (95% 0.000–0.010) | 0.6444 |
| N′ — schema, uniform | 2.08×10^23 | 1.16×10^15 | `complete` | 0 | 1.79×10^8 | 2.14×10^9 | 0.53× | 0/400 (95% 0.000–0.010) | 0.6138 |
| N″ — schema + learned prior | 2.08×10^23 | 1.16×10^15 | `complete` | 2 | 1.04×10^12 | 1.25×10^13 | 0.00× | 400/400 (95% 0.990–1.000) | 1.0000 |
<!-- END:arms_gap0 -->

`gap 1`:

<!-- BEGIN:arms_gap1 -->
| arm | full space S | conformers K | certificate | first solvable tier | expected programs to first solution | expected episodes | saving vs N | first solution generalizes (of sampled) | held-out accuracy of first solution |
|---|---|---|---|---|---|---|---|---|---|
<!-- END:arms_gap1 -->

## 3. The saving against the space size (F3)

<!-- BEGIN:f3_gap0 -->
| arm | space ratio S_N / S_arm | solution retention K_arm / K_N | saving N / arm (expected programs) | saving vs space ratio |
|---|---|---|---|---|
| N′ — schema, uniform | 2.42×10^6 | 0.00 | 0.53 | 0.00 |
| N″ — schema + learned prior | 2.42×10^6 | 0.00 | 0.00 | 0.00 |
<!-- END:f3_gap0 -->

<!-- BEGIN:f3_gap1 -->
(N not run)
<!-- END:f3_gap1 -->

## 4. The learned prior's tiers

<!-- BEGIN:tiers_npp_gap0 -->
| tier | S_t | K_t | shell S | shell K | ADDR kept | MATCH kept | STEP kept | absent letters kept |
|---|---|---|---|---|---|---|---|---|
| 0 | 2.62×10^5 | 0 | 2.62×10^5 | 0 | 1 | 1 | 4 | no |
| 1 | 1.03×10^12 | 0 | 1.03×10^12 | 0 | 56 | 7 | 8 | no |
| 2 | 6.56×10^14 | 77,280 | 6.55×10^14 | 77,280 | 934 | 16 | 8 | no |
| 3 | 1.02×10^16 | 1.77×10^8 | 9.59×10^15 | 1.77×10^8 | 935 | 16 | 20 | no |
| 4 | 1.02×10^16 | 1.77×10^8 | 0 | 0 | 935 | 16 | 20 | no |
| 5 | 2.56×10^16 | 2.57×10^8 | 1.54×10^16 | 7.97×10^7 | 935 | 40 | 20 | no |
| 6 | 7.18×10^16 | 3.51×10^8 | 4.62×10^16 | 9.38×10^7 | 988 | 112 | 20 | no |
| 7 | 4.50×10^17 | 1.02×10^9 | 3.78×10^17 | 6.69×10^8 | 1244 | 112 | 20 | no |
| 8 | 4.57×10^17 | 1.02×10^9 | 7.00×10^15 | 0 | 1445 | 112 | 20 | no |
| 9 | 1.04×10^18 | 4.79×10^11 | 5.87×10^17 | 4.78×10^11 | 1445 | 256 | 20 | no |
| 10 | 1.04×10^18 | 4.79×10^11 | 0 | 0 | 1445 | 256 | 20 | no |
| 11 | 1.04×10^18 | 4.79×10^11 | 0 | 0 | 1445 | 256 | 20 | no |
| 12 | 2.08×10^23 | 1.16×10^15 | 2.08×10^23 | 1.16×10^15 | 1445 | 256 | 20 | yes |
<!-- END:tiers_npp_gap0 -->

Prior only, over the flat pools:

<!-- BEGIN:tiers_p_gap0 -->
(not run)
<!-- END:tiers_p_gap0 -->

What the prior learned from the earlier domains:

<!-- BEGIN:prior -->
| kind | feature tuple | survived / rows | score |
|---|---|---|---|
| ADDR | `('add', ('constant', 'constant'))` | 0/49 | 0.0020 |
| ADDR | `('add', ('constant', 'length'))` | 0/14 | 0.0066 |
| ADDR | `('add', ('constant', 'position'))` | 1/41 | 0.0262 |
| ADDR | `('add', ('length', 'length'))` | 0/1 | 0.0494 |
| ADDR | `('identity', ('constant',))` | 4/13 | 0.2928 |
| ADDR | `('identity', ('length',))` | 0/2 | 0.0329 |
| ADDR | `('sub', ('constant', 'constant'))` | 15/85 | 0.1756 |
| ADDR | `('sub', ('constant', 'length'))` | 2/26 | 0.0777 |
| ADDR | `('sub', ('length', 'length'))` | 1/2 | 0.3662 |
| ADDR | `('add', ('constant', 'constant'), False)` | 0/18 | 0.0052 |
| ADDR | `('add', ('constant', 'constant'), True)` | 0/31 | 0.0031 |
| ADDR | `('add', ('constant', 'length'), False)` | 0/14 | 0.0066 |
| ADDR | `('add', ('constant', 'position'), False)` | 0/11 | 0.0082 |
| ADDR | `('add', ('constant', 'position'), True)` | 1/30 | 0.0354 |
| ADDR | `('add', ('length', 'length'), False)` | 0/1 | 0.0494 |
| ADDR | `('identity', ('constant',), False)` | 0/1 | 0.0494 |
| ADDR | `('identity', ('constant',), True)` | 4/12 | 0.3153 |
| ADDR | `('identity', ('length',), False)` | 0/2 | 0.0329 |
| ADDR | `('sub', ('constant', 'constant'), False)` | 0/42 | 0.0023 |
| ADDR | `('sub', ('constant', 'constant'), True)` | 15/43 | 0.3432 |
| ADDR | `('sub', ('constant', 'length'), False)` | 0/15 | 0.0062 |
| ADDR | `('sub', ('constant', 'length'), True)` | 2/11 | 0.1749 |
| ADDR | `('sub', ('length', 'length'), True)` | 1/2 | 0.3662 |
| LIT | `(False,)` | 0/379 | 0.0001 |
| LIT | `(True,)` | 22/45 | 0.4794 |
| TRUTH | `(0,)` | 0/3 | 0.0260 |
| TRUTH | `(1,)` | 1/3 | 0.2760 |
| TRUTH | `(10,)` | 0/3 | 0.0260 |
| TRUTH | `(11,)` | 0/3 | 0.0260 |
| TRUTH | `(12,)` | 0/3 | 0.0260 |
| TRUTH | `(13,)` | 0/3 | 0.0260 |
| TRUTH | `(14,)` | 0/3 | 0.0260 |
| TRUTH | `(15,)` | 0/3 | 0.0260 |
| TRUTH | `(2,)` | 1/3 | 0.2760 |
| TRUTH | `(3,)` | 0/3 | 0.0260 |
| TRUTH | `(4,)` | 0/3 | 0.0260 |
| TRUTH | `(5,)` | 0/3 | 0.0260 |
| TRUTH | `(6,)` | 0/3 | 0.0260 |
| TRUTH | `(7,)` | 1/3 | 0.2760 |
| TRUTH | `(8,)` | 2/3 | 0.5260 |
| TRUTH | `(9,)` | 0/3 | 0.0260 |
| STEP | `('add',)` | 2/10 | 0.2091 |
| STEP | `('sub',)` | 4/10 | 0.3909 |
| STEP | `('add', '2pixel')` | 0/2 | 0.1000 |
| STEP | `('add', '3pixel')` | 0/2 | 0.1000 |
| STEP | `('add', 'pixel')` | 1/2 | 0.4333 |
| STEP | `('add', 'row')` | 1/2 | 0.4333 |
| STEP | `('add', 'row+pixel')` | 0/2 | 0.1000 |
| STEP | `('sub', '2pixel')` | 0/2 | 0.1000 |
| STEP | `('sub', '3pixel')` | 0/2 | 0.1000 |
| STEP | `('sub', 'pixel')` | 2/2 | 0.7667 |
| STEP | `('sub', 'row')` | 2/2 | 0.7667 |
| STEP | `('sub', 'row+pixel')` | 0/2 | 0.1000 |
<!-- END:prior -->

The source searches it was fitted on reproduce their recorded certificates:

<!-- BEGIN:sources -->
| source | space | conforming | certificate | recorded |
|---|---|---|---|---|
| s19_stage_a | 10496 | 1 | `unique` | {'base': 14, 'byte': 40, 'conforming': 1, 'space': 10496} |
| s23_transform | 7480 | 2 | `complete` | 2 |
| s23_policy | 448 | 21 | `complete` | 21 |
| §33 s0 (res 32, committed) | 256 | 2 | `complete` | same file |
| §33 s1 (res 32, committed) | 400 | 2 | `complete` | same file |
| §33 s2 (res 32, committed) | 25 | 1 | `unique` | same file |

Rows per kind (rows / survivors): ADDR 233/23, GROUND 0/0, LIT 424/22, STEP 20/6, TRUTH 48/5
<!-- END:sources -->

## 5. Hard-transferred vector, baselines, achieved difficulty

<!-- BEGIN:extras_gap0 -->
(not run)
<!-- END:extras_gap0 -->

<!-- BEGIN:extras_gap1 -->
(not run)
<!-- END:extras_gap1 -->

## 6. Why — instrumentation of the found programs

<!-- BEGIN:instrument_gap0 -->
(not run)
<!-- END:instrument_gap0 -->

## 7. Validation of the instruments

<!-- BEGIN:validation -->
| check | result |
|---|---|
| V2 counter = brute force, gap 0 | 35/35 sub-spaces agree on S and K; 10 contain conformers |
| V2 counter = `tcn.search.enumerate_prefix` | steps_and_relation: S 192, K 12, tcn `complete` 12; grounding_and_steps: S 64, K 1, tcn `unique` 1; matcher_and_literals: S 32, K 2, tcn `complete` 2 — 3/3 agree |
<!-- END:validation -->

## 8. Resources

<!-- BEGIN:resources -->
| capped job | peak RSS (GB) | wall clock | exit |
|---|---|---|---|
| `arm_gap0_N` | 4.23 | 1:38.80 | 0 |
| `arm_gap0_Np` | 0.31 | 0:30.33 | 0 |
| `arm_gap0_Npp` | 0.27 | 2:17.59 | 0 |
| `cache` | 0.22 | 0:24.52 | 0 |
| `diag_v1` | 0.25 | 2:29.91 | 0 |
| `sources` | 0.28 | 1:30.87 | 0 |
| `v2brute_gap0` | 0.49 | 0:41.46 | 0 |
| `v2tcn_gap0` | 0.30 | 7:04.85 | 0 |

Memory floor: 17 capped starts logged in `out/memory_floor.log`; MemAvailable at start ranged 36.8–42.2 GB; phases refused by the 25 GB floor: 0.
<!-- END:resources -->

---

## Pre-sweep prediction (written 2026-09-10T12:10:01-07:00, before any OCC_ or V_ arm has run)

**A conjecture, not a finding:** drawn from three R seeds and crude cell-mean statistics
(`R_0`: occurs ratio `2.4`, V positive, `0/400`; `R_1`: `3.9`, V positive, `400/400`;
`R_2`: `3.9`, V inverted, `0/400`). The coordinator proposed it; it is recorded here so that the
sensitivity sweeps test it rather than being fitted to it.

> Generalization requires BOTH (a) a positive V preference on the address slots AND (b) an occurs
> ratio on the literal slots above a threshold somewhere between `2.4` and `3.9`.

Predicted outcome of each sweep point, stated now:

| sweep point | what is held | predicted |
|---|---|---|
| `OCC_1` | V fitted (positive) | no generalization — (b) fails |
| `OCC_3` | V fitted (positive) | undetermined — inside the threshold band |
| `OCC_10`, `OCC_100`, `OCC_3500` | V fitted (positive) | generalization; `OCC_3500` should reproduce N″ |
| `V_1` | occurs at N″'s fitted ratio | no generalization — (a) fails (zero preference is not positive) |
| `V_3`, `V_10`, `V_100` | occurs at N″'s fitted ratio | generalization |

Any row whose outcome differs is reported as falsifying the corresponding clause.
