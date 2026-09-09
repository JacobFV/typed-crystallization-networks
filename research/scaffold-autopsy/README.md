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
| `batched.py` | Gradient accumulation over episodes, before the core patch, to check the hypothesis. |
| `fixed.py` | Decisive run of the same scaffold under the patched trainer. |

`RESULTS.md` holds the findings. Logs and `*.json` are raw output.
