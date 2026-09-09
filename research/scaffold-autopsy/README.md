# scaffold-autopsy

Files, in the order they were used.

| file | what it does |
|---|---|
| `arith_joint.py` | Rebuilds the failed experiment: `tcn.scaffold.arithmetic_scaffold` driving the `logic` generator with `{'depth':1,'table':6,'fixed_inputs':True}`, horizon 4, two objectives, `lr=.04` — the `examples/joint.py` setup with the generic scaffold substituted for the typed logic graph. |
| `repro.py` | Runs it for N episodes and records the headline numbers. |
| `instrument.py` | `InstrumentedTrainer`: re-implements `JointTrainer.episode`'s loss assembly keeping each weighted term separate, so per-term gradient norms and all pairwise gradient cosines can be measured. |
| `run_instrument.py` | Per-term gradient norms/conflicts for the arithmetic scaffold and for `examples/joint.py` side by side. |
| `diagnose.py` | Tracks internal state across training: value output vs. actual return, advantage, policy logit magnitude, weight norms per region. |
| `capacity.py` | Capacity control: fits the *same* scaffold's prediction head on the same targets by full-batch supervision (no RL, no sampling, no value head). |
| `snr.py` | Per-term gradient signal-to-noise ratio at batch size one: `||mean g|| / mean ||g||`. |
| `batched.py` | Gradient accumulation over episodes against the unmodified trainer, to test (and ultimately reject) the batching hypothesis. |
| `fixed.py` | Long runs of the same scaffold. Uses `TrainConfig.batch` when the trainer exposes it, otherwise `batched.run_batched`; at `batch=1` both are the stock code path. |
| `ablate.py` | Loss-weight/learning-rate ablation grid (not needed for the verdict; kept for reuse). |
| `condense.py` | Shrinks the raw instrumentation dumps and downsamples long histories so the record stays reviewable. |

`RESULTS.md` holds the findings. Logs and `*.json` are the raw output;
per-episode histories in `fixed_ep*.json` / `repro_seed*.json` have been averaged
into ~64 blocks by `condense.py`, and the headline `summary` object in each is
untouched.

Nothing under `tcn/` is modified by this branch. `RESULTS.md` records the core
patch that was written, tested and then rejected on the evidence.
