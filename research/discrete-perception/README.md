# discrete-perception

Search the per-position module discretely, apply it positionally.  The follow-on
where the enumerative-baseline track, the perception ladder and the
positional-reuse branch converge.

`RESULTS.md` is the report; it is generated from `RESULTS.template.md` plus
`out/*.json` by `tables.py`, so no number in it is transcribed by hand.  Nothing
under `tcn/` or `generators/` is modified, and neither unmerged branch is
merged: the three-node caller is copied into `common.py` and named as a copy.

Each script writes `out/<tag>.json` and `out/<tag>.log`:

| script | what it measures |
|---|---|
| `check_local_fit.py` | that the local copy of `synthesis.fit` is bit-identical at `init_noise=0` |
| `audit.py` | what the geometry probes are, as functions of a pixel, before any search |
| `rung3_mask.py` | rung 3 end to end: search, certificate, identification curve, gradient arm, positional application, `filter`/`count` aggregate |
| `tiebreak.py` | which selection rule to use when the conforming set is not a singleton |
| `resolution_transfer.py` | search once at R=8, apply at R=8..48 |
| `offsets_free.py` | rung 3 with the raster layout searched rather than supplied |
| `rung35_window.py` | the staged two-position window (`--per-image 0` is the working arm) |
| `rung35_flat.py` | the same window with nothing frozen -- the decomposition control |
| `rung4_objects.py` | `object_ids`: information ceilings and exhaustive certificates |
| `rung5_depth.py` | `depth`: the same, plus what position alone adds |
| `full_alphabet.py` | where brute force stops: brute force, scoping, hill climbing, gradient |
| `incremental.py` | what makes exhaustive search affordable: prefix reuse |

`common.py` holds the shared machinery; `tables.py` renders the report.

Every timing was taken on a host shared with other agents (load average 55-80 on
20 cores); programs evaluated is the load-independent cost measure and is
reported alongside every time.
