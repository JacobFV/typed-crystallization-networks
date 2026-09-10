# Pre-registration — repairing the shipped `language` demonstration

Written **before any change** to `tcn/cli.py`, `scripts/demo.sh` or anything
under `research/language-capability/`. The only work done before this file
existed was reading (FINDINGS §62, §39, §43, §45, §24, §47,
`research/record-audit/RESULTS.md`, `research/language-post-audit/RESULTS.md`,
the shipped demo code) and two **read-only** timing/feasibility probes run from
the scratchpad against unmodified repository code. Their outputs are quoted in
§3 below and are what this registration is decided on.

## 1. The defect

`scripts/demo.sh --only language` fails:

```
== language ...
   FAIL  raised ValueError: training examples required
0/1 demonstrations reproduced in 1 s
```

Reproduced here in this worktree, matching FINDINGS §62 / `record-audit`
pass 3 (which reproduced it in both quick and `--full`). `README.md` tells a
new reader to run `scripts/demo.sh` first and `STATUS.md` promises a non-zero
exit if anything fails to reproduce, so the repository currently fails its own
contract on the first command it recommends.

The cause is settled and is §39's: `tcn/cli.py:_demo_language` calls
`research/language-capability/common.dataset`, which **inherits** the generator's
hardening default, then filters `e['length'] in (2,4,6)`. §24 re-drew
`context_free_language` and moved string length from 2–16 to 10–22, so that
filter is empty and `enumerate_fit` raises.

## 2. The two options

**Option A — pin the stream to `hardening='none'`.** Minimal. Reproduces §19.
Demonstrates a capability on a lesson §24 established is exploitable
(nearest-neighbour 1.000, 71% of prompts repeated).

**Option B — re-pose the task on the post-audit (shipped default) stream.**
§45 exhibited a program scoring 1.000 on 859 held-out episodes at unseen
lengths 16–22 against a 0.5262 majority, adding a running `min` beside the
running `add` — `min` and `and`, both already in `tcn/operators.py`, no new
operator. §47 then exhausted the full 680,625-program space (110 conforming,
certificate `complete`) and showed the gradient path cannot find it.

## 3. What I will implement, and why

**Option B as the demonstration's headline, with Option A retained inside the
same demonstration as the pre-audit control.** The demo will run **both arms and
name the stream on each**.

The evidence for choosing B over A alone:

1. **B runs on the stream the repository actually ships.** A demonstrates a
   capability on a distribution no default draw produces any more. A demo whose
   headline can only be obtained by switching the generator out of its shipped
   configuration is a weaker artifact for the external evaluator §62 anticipates.
2. **B is stronger on the merits.** A's capability is *counting* (§45: post-audit
   the counting program scores exactly the majority), and §24 removed the
   exploit that made counting sufficient. B is balancedness.
3. **B is rebuildable from what is committed.** Checked before choosing:
   `research/language-post-audit/{common,splits,run_stage_a,run_stage_b,
   dyck_scaffold,dyck_witness}.py` are all tracked in git, and the read-only
   probe rebuilt the witness end to end from them — stage A `unique` at
   base 14 / byte 40, held-out position max error 0.0, train max error 0.0,
   **1.000 on n=859, majority 0.5261932479627474**, matching `dyck_witness.json`.
   The §45 falsification "the witness depends on files outside the repo" does
   **not** fire.
4. **Cost is not a reason to prefer A.** Probed: B's arm is ~8 s wall
   (2.7 s episodes + 1.3 s stage-A enumeration + ~3.5 s evaluation), A's ~5 s.
   Both arms together are ~13 s, so B does not cost the demo its usability as a
   first command.
5. **A is still needed and is kept**, exactly as the brief anticipates: it is the
   reproduction of §19's historical claim, and it is the control that proves the
   post-audit negative is about the stream and not about the harness (§45 §7).

## 4. What the demo will print

Every printed number will name its stream. Specifically:

- **post-audit arm** (`hardening='context_free_language'`, the shipped default
  draw): stage A certificate and selection; stage B accuracy on held-out
  **unseen** lengths 16/18/20/22, with `n`, the **majority constant**, random
  0.5, the best fitted feature, the training-string-lookup exploit, and the
  counting oracle beside it.
- **pre-audit arm** (`hardening='none'`, §19's stream): §19's accuracy with its
  `n` and its majority constant, explicitly labelled as the pre-audit stream and
  as a lesson §24 found exploitable.

**The pre-audit number the demo prints will be `0.9986187845303868` on
`n=724` against majority `0.5483425414364641`** — the selection
`enumerate_fit` returns in declaration order, recorded in
`research/language-capability/final_eval.json` and `stage_b.json`, and
reproduced digit-for-digit by §45's control.

This is a **change of number** and I register the reason in advance. The shipped
`_LANGUAGE_RULE = {'symbols':101,'plus':0,'minus':4,'answer':6}` is *not* §19's
recorded selection; the read-only probe scores it at **1.000 on n=724**
(per-length `{8:1.0,10:1.0,12:1.0,14:1.0,16:1.0}`). `conforming.json` records
that **10 programs conform on training and split 4 at 1.000 / 6 at
0.9986187845303868**, and training accuracy cannot separate them. So the shipped
demo prints the maximum over a tie broken on held-out data, and prints exactly
the 1.000 that `docs/CORRECTIONS.md` row 9, §43 and §62 H2 all record as
overturned. The demo will report the recorded member and disclose the tie.

## 5. Scope

- `tcn/cli.py:_demo_language` rewritten. No other core change.
- `research/language-capability/common.py`: `episode`/`dataset` gain an explicit
  `hardening` argument, defaulting to the named hardened draw — i.e. **exactly
  the stream those call sites already get today**, stated rather than inherited.
  This is §39's own prescription. No number that track recorded moves.
- Nothing under `generators/` is touched. **The shipped lesson distribution will
  not be changed**; §24 re-drew it deliberately and reverting that to make a demo
  pass would be backwards.
- `research/visual-width-reuse/` is another agent's and will not be touched.

## 6. Falsification — what would make me abandon this choice

- **The witness cannot be rebuilt from committed files** → fall back to Option A
  alone, and report the non-reproducibility as a finding. *(Checked in §3.3
  before choosing; it does rebuild.)*
- **The post-audit arm's accuracy does not equal `dyck_witness.json`'s 1.000 on
  n=859 against 0.5261932479627474** → the rebuild is not the recorded program;
  report and fall back to A.
- **The pre-audit arm does not reproduce `0.9986187845303868` on n=724 against
  `0.5483425414364641`** → the demo cannot honestly claim to reproduce §19;
  report the mismatch rather than printing whatever comes out.
- **Both arms together push the demo beyond roughly a minute of wall clock** →
  it stops being usable as a first command; drop the pre-audit arm to `--full`
  only, or fall back to A alone. *(Probed at ~13 s.)*
- **Any of the 337 tests, or the shipped fixture's `0.248836 → 0.002231 at 4/4
  frozen`, moves** → the change is not confined to the demo; revert it.
- **`scripts/demo.sh` does not reach 10 of 10** → report it as still failing
  rather than relaxing the demo's `ok` gate to make it pass.

The `ok` gate will be *tightened*, not relaxed: it will require the stage-A
`unique` certificate and its recorded selection, train max error 0.0 on the
post-audit scaffold, the post-audit accuracy at 1.000 and more than 0.3 above
both the majority constant and the best fitted feature, and the pre-audit arm to
equal §19's recorded figures to within 1e-12.

## 7. Environment disclosure

This worktree has no `.venv` and no `node_modules` of its own; both are
gitignored in the parent checkout. I symlinked
`/home/.../typed-crystallization-networks/.venv` and
`generators/computer/engine/node_modules` into the worktree before running
anything, per the brief's trap note, so the `computer`/`panel` tests are
runnable here rather than failing environmentally.
