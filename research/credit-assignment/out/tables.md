### baselines

| policy | mean return | sd | solved / n |
|---|---|---|---|
| `commit_now` | 0.0208 | 0.143 | 1/48 |
| `always_wait` | 0.0000 | 0.000 | 0/48 |
| `look_only` | 0.0000 | 0.000 | 0/48 |
| `uniform_random` | 0.0000 | 0.000 | 0/48 |
| `neutral_typed` | 0.0833 | 0.276 | 4/48 |
| `dial_then_commit` | 0.1042 | 0.305 | 5/48 |
| `fixed_slot_plan` | 0.2500 | 0.433 | 12/48 |
| `oracle` | 1.0000 | 0.000 | 48/48 |

### horizon sweep

| H | oracle | commit_now (myopic) | uniform_random | neutral_typed |
|---|---|---|---|---|
| 1 | 0.0000 | 0.0208 | 0.0000 | 0.0000 |
| 2 | 0.0000 | 0.0208 | 0.0000 | 0.0208 |
| 3 | 0.3750 | 0.0208 | 0.0000 | 0.0417 |
| 4 | 0.5208 | 0.0208 | 0.0000 | 0.0833 |
| 6 | 1.0000 | 0.0208 | 0.0000 | 0.0833 |
| 8 | 1.0000 | 0.0208 | 0.0000 | 0.1042 |

### learning arms

| arm | seeds | episodes | eval (deterministic) | sd | eval (stochastic) | seeds at 1.00 | env episodes |
|---|---|---|---|---|---|---|---|
| `flat` | 3 | 600 | **0.0260** | 0.037 | 0.1146 | 0/3 | 856 |
| `myopic` | 4 | 600 | **0.0664** | 0.023 | 0.0820 | 0/4 | 856 |
| `probe_only` | 2 | 300 | **0.0000** | 0.000 | 0.0781 | 0/2 | 524 |
| `reward` | 6 | 600 | **0.2396** | 0.058 | 0.5938 | 0/6 | 856 |
| `reward_g50` | 3 | 600 | **0.1979** | 0.064 | 0.3281 | 0/3 | 856 |
| `reward_percept` | 3 | 600 | **0.0625** | 0.044 | 0.0677 | 0/3 | 856 |
| `reward_sweep_given` | 3 | 600 | **1.0000** | 0.000 | 0.6927 | 3/3 | 856 |

### training reward rate, mean over seeds (fraction of training episodes rewarded)

| arm | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `flat` | 0.10 | 0.09 | 0.09 | 0.07 | 0.09 | 0.08 | 0.09 | 0.11 | 0.10 | 0.10 | 0.07 | 0.07 |
| `myopic` | 0.14 | 0.10 | 0.07 | 0.09 | 0.09 | 0.08 | 0.09 | 0.04 | 0.10 | 0.06 | 0.07 | 0.07 |
| `probe_only` | 0.08 | 0.05 | 0.06 | 0.12 | 0.11 | 0.05 | -- | -- | -- | -- | -- | -- |
| `reward` | 0.11 | 0.11 | 0.20 | 0.29 | 0.33 | 0.50 | 0.46 | 0.47 | 0.56 | 0.55 | 0.63 | 0.58 |
| `reward_g50` | 0.12 | 0.11 | 0.13 | 0.20 | 0.27 | 0.21 | 0.31 | 0.37 | 0.28 | 0.33 | 0.35 | 0.35 |
| `reward_percept` | 0.07 | 0.07 | 0.06 | 0.07 | 0.10 | 0.07 | 0.11 | 0.07 | 0.11 | 0.05 | 0.05 | 0.11 |
| `reward_sweep_given` | 0.08 | 0.12 | 0.19 | 0.29 | 0.47 | 0.64 | 0.62 | 0.61 | 0.61 | 0.63 | 0.59 | 0.74 |

### held-out evaluation curve, deterministic (mean over seeds)

| arm | 50 | 100 | 200 | 400 |
|---|---|---|---|---|
| `flat` | 0.000 | 0.000 | 0.000 | 0.000 |
| `myopic` | 0.031 | 0.070 | 0.070 | 0.070 |
| `probe_only` | 0.000 | 0.000 | 0.000 | -- |
| `reward` | 0.047 | 0.104 | 0.224 | 0.229 |
| `reward_g50` | 0.052 | 0.146 | 0.156 | 0.219 |
| `reward_percept` | 0.000 | 0.000 | 0.115 | 0.083 |
| `reward_sweep_given` | 0.073 | 0.094 | 1.000 | 1.000 |

### composite actions

| module library | seeds | eval | sd | chose `sweep` | env episodes |
|---|---|---|---|---|---|
| `stare` | 2 | **0.2969** | 0.016 | 0/2 | 528 |
| `sweep+stare` | 4 | **0.9648** | 0.061 | 4/4 | 528 |

### what each arm selected

* `flat`: next_slot ['identity(action.1.slot)', 'add(one_slot,one_slot)', 'add(action.1.slot,action.1.slot)'], log sigma(slot) [1.256, 2.234, 1.87], argmax verb per situation [{'idle': 'dial', 'found': 'wait', 'dialled': 'wait'}, {'idle': 'commit', 'found': 'wait', 'dialled': 'wait'}, {'idle': 'dial', 'found': 'wait', 'dialled': 'wait'}]
* `myopic`: next_slot ['add(action.1.slot,one_slot)', 'add(action.1.slot,one_slot)', 'identity(action.1.slot)', 'identity(one_slot)'], log sigma(slot) [2.402, 2.238, 1.267, 2.146], argmax verb per situation [{'idle': 'commit', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'commit', 'found': 'commit', 'dialled': 'commit'}, {'idle': 'commit', 'found': 'wait', 'dialled': 'commit'}, {'idle': 'commit', 'found': 'commit', 'dialled': 'commit'}]
* `probe_only`: next_slot ['identity(action.1.slot)', 'identity(action.1.slot)'], log sigma(slot) [0.0, 0.0], argmax verb per situation [{'idle': 'wait', 'found': 'wait', 'dialled': 'wait'}, {'idle': 'wait', 'found': 'wait', 'dialled': 'wait'}]
* `reward`: next_slot ['add(one_slot,one_slot)', 'add(one_slot,one_slot)', 'add(one_slot,one_slot)', 'add(one_slot,one_slot)', 'add(one_slot,one_slot)', 'identity(one_slot)'], log sigma(slot) [0.489, 0.259, 1.311, 2.245, 2.318, 0.567], argmax verb per situation [{'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}]
* `reward_g50`: next_slot ['add(one_slot,one_slot)', 'add(one_slot,one_slot)', 'identity(one_slot)'], log sigma(slot) [2.125, 2.138, -0.307], argmax verb per situation [{'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'dial', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}]
* `reward_percept`: next_slot ['identity(action.1.slot)', 'identity(one_slot)', 'add(action.1.slot,one_slot)'], log sigma(slot) [1.523, 2.246, 2.502], argmax verb per situation [{'idle': 'dial', 'found': 'wait', 'dialled': 'commit'}, {'idle': 'dial', 'found': 'wait', 'dialled': 'wait'}, {'idle': 'commit', 'found': 'wait', 'dialled': 'look'}]
* `reward_sweep_given`: next_slot [None, None, None], log sigma(slot) [-0.581, -0.39, 0.837], argmax verb per situation [{'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}, {'idle': 'look', 'found': 'dial', 'dialled': 'commit'}]