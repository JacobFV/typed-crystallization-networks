# The shipped `language` demonstration, repaired on the stream the repository ships

Branch `worktree-agent-af26d6fec4b924403`. `PREREGISTRATION.md` was written and
committed (`6da16ed`) **before any change**; the fix is `c80d939`. Nothing under
`generators/` is touched, and the shipped lesson distribution is unchanged.

---

## Verdict, up front

**Option B, with Option A retained inside the same demonstration as its control.**

`scripts/demo.sh` now reaches **10 of 10**, exit status **0**, in **73.9 s**
quick and **204.3 s** under `--full`. The `language` demonstration takes
**14.8 s** of that in quick mode and **14.7 s** under `--full`.

The demonstration prints **two rows and names the stream on each**:

| stream | what is demonstrated | accuracy | n | majority constant |
|---|---|---|---|---|
| **post-audit** — `hardening='context_free_language'`, what an unconfigured generator emits today | **balancedness**: a running `min` beside the running `add` | **1.000** | **859** | **0.5261932479627474** |
| **pre-audit** — `hardening='none'`, FINDINGS §19's stream, a lesson §24 found exploitable | **counting**: §19's own recorded program | **0.9986187845303868** | **724** | **0.5483425414364641** |

Both match their source of truth to every digit: `dyck_witness.json` and
`research/language-capability/final_eval.json` respectively.

---

## 1. The defect, reproduced here before anything was changed

```
$ scripts/demo.sh --only language
== language ...
   FAIL  raised ValueError: training examples required
0/1 demonstrations reproduced in 1 s
```

```
File ".../tcn/cli.py", line 576, in _demo_language
  found=enumerate_fit(lexical,position_examples(train),signals,registry,...)
File ".../tcn/search.py", line 162, in enumerate_fit
  if not examples: raise ValueError('training examples required')
ValueError: training examples required
```

Whole suite, on the unmodified tree (re-run here by checking `tcn/cli.py` and
`research/language-capability/common.py` back out to the pre-fix commit):

```
9/10 demonstrations reproduced in 58 s
exit=1
```

which reproduces FINDINGS §62 / `record-audit` pass 3 exactly — 9 of 10, the
`language` row the only failure, in both quick and `--full`.

The cause is §39's and is settled: `_demo_language` called
`research/language-capability/common.dataset`, which **inherited** the
generator's hardening default, then filtered `e['length'] in (2,4,6)`. §24's
re-draw moved `context_free_language` from lengths 2–16 to 10–22, so that filter
returns nothing and stage A's training split is empty.

## 2. Why Option B, on evidence

Both options were probed **read-only against unmodified repository code** before
the pre-registration was written. Both reproduce, and both are cheap:

| | rebuilt from committed files? | accuracy on held-out unseen lengths | majority | wall |
|---|---|---|---|---|
| **A** — pin `hardening='none'` | yes | **0.9986187845303868** on n=724 | 0.5483425414364641 | ~5 s |
| **B** — re-pose on the shipped stream | yes | **1.000** on n=859 | 0.5261932479627474 | ~8 s |

So cost did not decide it. What decided it:

1. **B runs on the stream the repository actually ships.** A's headline can only
   be obtained by switching the generator out of its shipped configuration. For
   the external evaluator §62 anticipates, that is the weaker artifact.
2. **B is the stronger claim.** A demonstrates *counting* — §45 proved it, and
   the demo now measures it live: §19's own program, run on the post-audit
   stream, scores **0.5261932479627474**, which is **exactly the majority
   constant**, because §24's hardened negative is a transposition and preserves
   the bracket multiset. B demonstrates balancedness.
3. **B needs no new operator and no core change.** `min` and `and` are already in
   `tcn/operators.py`; the scaffold adds one accumulator.
4. **§45's falsification did not fire.** The witness *is* rebuildable from what
   is committed: `research/language-post-audit/{common,splits,run_stage_a,
   run_stage_b,dyck_scaffold,dyck_witness}.py` are all tracked, and the probe
   rebuilt the program end to end from them.
5. **A is still needed, and is kept** — as the reproduction of §19's historical
   claim, and as the control showing the post-audit result is a fact about the
   stream and not about the harness (§45 §7).

## 3. What the demonstration now does

- **Stage A is searched live, on the post-audit stream.** 10,496 programs,
  10,496 evaluated, `exhausted: true`, certificate **`unique`**, 1.31 s;
  selection `base = 14`, `open_byte = 40` — ASCII `(`, searched over the whole
  0–255 alphabet and not supplied. Held-out **position max error 0.0** at
  lengths never trained on.
- **Stage B is applied from a recorded selection, not searched.** Enumerating
  those spaces costs 431 s (post-audit counting arm), 1,266 s (the Dyck window)
  and 363 s (pre-audit) — more than a first command may spend. Every certificate
  that produced them travels with the number, in `detail['recorded']`:

| space | evaluated | exhausted | conforming | certificate | source |
|---|---|---|---|---|---|
| post-audit, §19's counting scaffold | 45,375 | true | **0** | `complete` | §45 |
| post-audit, Dyck scaffold, window `c ∈ [95,110)` | 84,375 | true | 110 | `complete` | §45 |
| post-audit, Dyck scaffold, **full space** | 680,625 | true | 110 | `complete` | §47 |
| pre-audit, §19's scaffold | 45,375 | true | 10 | not `unique` | §19 |

- **The post-audit selection is verified live, not asserted**: `train_max_error`
  is **0.0** on the 24 training episodes at lengths 10/12/14 before any accuracy
  is reported.
- **Both arms draw the pools their recorded numbers were taken on** (900 train /
  1,500 test seeds) in quick and `--full` alike, because n=859 and n=724 are part
  of what is being reproduced. Verified: the two modes print identical figures.

## 4. The printed output, after

```
== language ...
   PASS  post-audit stream (hardening='context_free_language', the shipped default):
   stage A unique among 10,496 (base 14, open byte 40), held-out position error 0.0;
   stage B 1.000 on 859 held-out episodes at lengths 16-22 never trained on.
   Pre-audit stream (hardening='none', section 19's): 0.9986187845 on 724
   baseline: post-audit majority constant 0.5262, random 0.500, best fitted feature
   0.4738, training-string lookup 0.4738, and section 19's own counting program
   0.5262 on this stream; pre-audit majority constant 0.5483, best fitted feature
   0.6478; gradient descent conforms 0 of 64 runs on the post-audit space and 0 of
   44 pre-audit
```

```
10/10 demonstrations reproduced in 74 s; artifacts under artifacts/demo
exit=0
```

## 5. Verification

### 5.1 Every printed number against its source of truth

| figure | printed | recorded | source |
|---|---|---|---|
| post-audit accuracy | **1.0** | 1.0 | `language-post-audit/dyck_witness.json` |
| post-audit n | **859** | 859 | same |
| post-audit majority | **0.5261932479627474** | 0.5261932479627474 | same |
| post-audit per-length | 16:1.0, 18:1.0, 20:1.0, 22:1.0 | identical | same |
| post-audit best fitted feature | **0.47380675203725264** | 0.47380675203725264 | `language-post-audit/baselines.json` |
| post-audit training-string lookup | **0.47380675203725264** | 0.47380675203725264 | same |
| post-audit counting oracle | **0.5261932479627474** | 0.5261932479627474 | same |
| post-audit Dyck oracle | **1.0** | 1.0 | same |
| §19's program on this stream | **0.5261932479627474** | "exactly the majority" | §43, §45 §5(a) |
| pre-audit accuracy | **0.9986187845303868** | 0.9986187845303868 | `language-capability/final_eval.json`, §43, §45 §7 |
| pre-audit n | **724** | 724 | same |
| pre-audit majority | **0.5483425414364641** | 0.5483425414364641 | same |
| pre-audit per-length | 8:1.0, 10:1.0, 12:1.0, 14:1.0, **16:0.5** | identical | same |
| pre-audit best fitted feature | **0.6477900552486188** | "0.648" | §19 |
| stage A | 10,496 exhausted, `unique`, base 14 / byte 40, held-out error 0.0 | identical | `language-post-audit/stage_a.json` |

**The demo no longer prints 1.000 for the pre-audit arm, and this is a change of
number that was pre-registered in advance.** The shipped
`_LANGUAGE_RULE = {'symbols':101,'plus':0,'minus':4,'answer':6}` was **not**
§19's recorded selection. Measured here: it scores **1.000 on n=724**, per-length
`{8:1.0, 10:1.0, 12:1.0, 14:1.0, 16:1.0}`. `conforming.json` records that **ten**
programs conform on the training episodes and that they split **4 at 1.000 / 6 at
0.9986187845303868** on the held-out lengths — a tie training accuracy cannot
break. The shipped demo was therefore printing the maximum over a tie broken on
held-out data, and printing exactly the 1.000 that `docs/CORRECTIONS.md` row 9,
§43 and §62 H2 all record as overturned. The demo now applies §19's **recorded**
member (`symbols=99, plus=-1, minus=+1, eq(acc,2)`, from `stage_b.json` and
`final_eval.json`) and discloses the ten-way tie in its own caveats.

### 5.2 The suite

| run | mode | result | exit | wall | language row |
|---|---|---|---|---|---|
| before | quick | **9/10** | 1 | 58.5 s | FAIL, 0.0 s |
| after | quick | **10/10** | 0 | **73.9 s** | PASS, **14.81 s** |
| after | `--full` | **10/10** | 0 | **204.3 s** | PASS, **14.69 s** |

`--full` was 192.0 s in §62's audit with `language` failing at ~0 s, so the
language repair accounts for the whole of the increase in both modes. It stays
usable as a first command.

### 5.3 Tests — nothing moved

| | before the change | after the change |
|---|---|---|
| `.venv/bin/python -m pytest tests/ -q` | **336 passed, 1 failed**, 64.45 s | **336 passed, 1 failed**, 64.95 s |

337 collected in both. The single failure is
`tests/test_panel_interface.py::test_panel_episode_replays_and_restores`
(`AssertionError: replayed episode diverged`), **identical before and after the
change** on the identical environment, and is the worktree hazard the brief
names. It is not touched by anything here: no test imports this track, and the
only code changed is `tcn/cli.py:_demo_language` plus a keyword argument in
`research/language-capability/common.py`.

The `generators/computer` and `panel` tests need a gitignored `node_modules`; it
and `.venv` were symlinked into this worktree from the parent checkout before
anything was run, which is why 336 pass here rather than the 275/13 split
earlier tracks reported.

### 5.4 The shipped fixture — unmoved

```
$ .venv/bin/python -m tcn train --episodes 160
initial_prediction_loss   0.248835613951087
final_prediction_loss     0.0022308224288281053
fully_frozen              true
evaluation_mean_return    4.0
frozen_evaluation_mean_return 4.0
```

**0.248836 → 0.002231 at 4/4 frozen.**

## 6. What changed

| file | change |
|---|---|
| `tcn/cli.py` | `_demo_language` rewritten: post-audit headline, pre-audit control, stream named on every number, `ok` gate tightened |
| `research/language-capability/common.py` | `episode`/`dataset` take `hardening` by name (§39's own prescription) and record it on every episode |
| `STATUS.md` §1 row 9 | now states both streams and what the demo prints |
| `docs/VALIDATION.md` | the table row, and the §5 prose that still carried the retracted unqualified **1.000** with no stream, both corrected to 0.9986187845303868 with its stream, and the post-audit result added |

**Nothing under `generators/` was touched and the lesson distribution is
unchanged**, per the brief's third falsification: §24 re-drew
`context_free_language` deliberately to remove an exploit, and reverting that to
make a demo pass would be backwards.

Pinning `hardening='context_free_language'` is **bit-identical** to inheriting
the generator default for this lesson — verified over 240 episodes across both
splits, 0 differences — so no number `research/language-capability/` recorded
moves. It changes only `instance_id`, by design (`lesson.py:94`).

Every existing caller of `common.dataset` in that track passes its arguments by
keyword, so none of them is affected; and §39's own
`reproduce/stream_check.py` forces `hardening` at `Host.create` itself, so it
keeps both of its arms and still reproduces §39's table.

The `ok` gate was **tightened**, not relaxed. It now requires: stage A solved,
exhausted, `unique`, `base=14`, `open=40`, held-out position error exactly 0.0;
`train_max_error ≤ 1e-6` on the post-audit scaffold; `n=859`, accuracy exactly
1.0, majority within 1e-12 of the recorded 0.5261932479627474, and accuracy more
than 0.3 above the majority, the best fitted feature and the training-string
lookup; and on the pre-audit arm `n=724` with accuracy and majority both within
1e-12 of §19's recorded figures.

## 7. Disclosures

- **Stage B is applied, not searched, in the demo.** That is the same shape the
  demonstration always had, and the reason is cost (431 s / 1,266 s / 363 s), but
  it means the demo *exhibits* a program whose search certificate was earned
  elsewhere. The certificates are carried in the demo's own JSON rather than
  merely cited here.
- **The Dyck window enumeration returns 110 conforming programs, never
  `unique`.** On this stream the running minimum is never positive, so `eq(lo,0)`
  and `ge(lo,0)` are the same function here, and several `(plus, minus)` scalings
  compose with the same readouts.
- **§47's ordering caveat stands**: the selected member sits 84.0% through the
  680,625-program enumeration order, so enumeration order is doing real work and
  no sensible ordering was shown to find it early. The demo says so.
- **The pre-audit arm is a lesson the audit found exploitable** —
  nearest-neighbour scored 1.000 on it, 71% of prompts repeated — and the demo
  never prints its number without saying that.
- **Negative results preserved.** The demo's own caveats now carry §45's
  negative (§19's scaffold contains no conforming program post-audit, certificate
  `complete`) and §47's (the gradient path conforms 0 of 64 runs on the Dyck
  space, median held-out 0.4738, below both the 0.5262 majority and random).
- **This worktree needed two gitignored symlinks** (`.venv`,
  `generators/computer/engine/node_modules`) to run anything at all.

## 8. Files

| file | what it is |
|---|---|
| `PREREGISTRATION.md` | the option chosen, the reasons, and the falsifications — committed before any change |
| `out/before_language.txt` | the failing demo, reproduced here |
| `out/before_suite_quick.txt`, `out/demo-before-full-suite/` | the whole suite on the pre-fix tree: 9/10, exit 1 |
| `out/after_language_quick.txt`, `out/demo-after-quick/` | the repaired demo alone |
| `out/after_suite_quick.txt`, `out/demo-after-suite/` | the whole suite, quick: 10/10, exit 0 |
| `out/after_suite_full.txt`, `out/demo-after-full/` | the whole suite, `--full`: 10/10, exit 0 |
| `out/tests_before.txt`, `out/tests_after.txt` | 336 passed / 1 failed, identical either side |
| `out/fixture_after.txt` | `tcn train --episodes 160`, 0.248836 → 0.002231 at 4/4 |

Per-demo artifact directories other than `language/` were pruned from the saved
runs; the `summary.json` in each still carries every row in full.

## 9. Reproduction

```
scripts/demo.sh --only language          # ~15 s
scripts/demo.sh                          # 10/10, ~74 s
scripts/demo.sh --full                   # 10/10, ~204 s
.venv/bin/python -m pytest tests/ -q     # 336 passed, 1 failed (panel, environmental)
.venv/bin/python -m tcn train --episodes 160
```
