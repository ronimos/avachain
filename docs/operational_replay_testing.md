# Operational Run Testing Module (Replay Harness)

**Purpose:** run the avachain pipeline over a past season *as if it were
operational* — data gated to a simulated clock, UAS surveys arriving on their
real dates, SNOWPACK rerun from the last measured state, avalanches detected
and reinitialised as they happen — then compare the result against the
artifacts the season run actually produced.

**Status:** designed; prerequisites complete; harness not yet built.
**Last updated:** 2026-10-19
**Companion:** `operational_mode_design_concept.md` defines the three
operational modes this module replays. Read that first.

---

## 1. What this is for

Three distinct questions, deliberately kept separate:

1. **Does the chain work?** Do the three modes (daily forward, NWP forecast,
   survey correction) dispatch on the right days, hand off correctly, and
   produce well-formed artifacts without silent fallbacks? This is integration
   testing and it is the primary value.
2. **How wrong is the forcing between surveys?** Quantified, not fixed. See
   §7 — this is the central scientific output.
3. **Does the pipeline survive an update?** The module is the regression gate:
   after any change to the chain, replay a known window and diff.

It is **not** a physics validation and **not** a calibration exercise. See §9.

---

## 2. Agreed configuration

| Decision | Choice | Why |
|---|---|---|
| Window | `T_start = 2026-02-05` → `T_end = 2026-03-08` | 8 survey arrivals in 32 ticks, median gap 3 d — closest available to the daily-survey design target. Contains the largest detected event (02-19→02-21, −25 698 m³). **Avoids Jan 18.** |
| Tick scope | Full-domain SNOWPACK + Zarr append + `analyze` every tick; `scenarios` on survey dates + 03-07 | Scenarios are the slow step (7m20s for 75) and are near-redundant between surveys. |
| NWP | Exercise the path; label **plumbing-only**; record no skill numbers | No per-cycle WRF archive exists. See §8.1. |
| Cluster splitting | Frozen for the primary run; second pass with splits on | Keeps cluster IDs aligned so per-cluster comparison is interpretable; the splits-on pass is a pure orchestration check. |

Surveys inside the window: `02-10, 02-12, 02-14, 02-19, 02-21, 03-02, 03-05,
03-08`. The season run produced `reinit_stats` at `02-07, 02-11, 02-13, 02-20,
03-03, 03-06` — **six reinit events**, roughly one per survey pair. This window
is reinit-heavy, which is excellent coverage of the hardest path (§5.3 of the
design concept) but dominates runtime.

---

## 3. Isolation

A **replay slope**, not a repo copy. `ProjectConfig` already namespaces state
by slope, so a sibling TOML buys isolation almost free:

```bash
# slopes/little_prof_replay/slope_config.toml — copy of little_prof with:
#   [slope]  name       = "little_prof_replay"
#   [paths]  output_dir = "outputs/little_prof_replay"
#            slope_dir  = "snowpack/little_prof_replay"
python install.py slopes/little_prof_replay/slope_config.toml
```

`install.py` generates that slope's own `run_snowpack.sh`,
`config/master_config.ini` and `template.sno`. Read-only inputs (DEM,
`data/surveys`, boundaries, `data/weather`) stay shared — correct, since the
replay gates surveys by date rather than by staging files.

The harness must export `SLOPE_DIR` and `SMET_DIR` explicitly even though the
generated defaults are now correct, because the shell scripts honour the
environment and being explicit removes any chance of writing into the live
slope.

**The weather cache is shared on purpose.** `cfg.weather_cache_dir` is
`<project_dir>/outputs/weather`, not slope-namespaced. That is right: it holds
raw station observations, identical for every slope, and the gating happens via
`smet_append --until`, not via cache contents.

---

## 4. The clock

One lever: `$AVACHAIN_AS_OF`.

```python
config.replay_as_of(explicit=None)   # explicit arg → $AVACHAIN_AS_OF → None
```

`None` means live behaviour, unchanged. An environment variable rather than a
CLI flag because `run_daily.sh` and `run_little_prof.sh` spawn a dozen Python
subprocesses; exporting once reaches all of them, whereas threading `--as-of`
through every invocation would eventually miss one — and a replay that saw one
survey too many would still produce plausible output.

Two seams consume it. They are the only two places the pipeline can see the
future:

| Seam | Was | Now |
|---|---|---|
| `pipeline_io.discover_surveys(as_of=)` | globbed every survey, no date bound | hides surveys after the clock |
| `nwp_ingest.get_stable_ts(now=)` | read the wall clock | boundary tracks the clock |

The survey cutoff lives in `discover_surveys` rather than in staged TIFs
because `step_transport`, `step_avalanche` and `fetch_avalanche_obs` each
re-glob independently; one cutoff covers all three atomically.

Already date-parameterised, no change needed: `smet_append --until`,
`run_snowpack.sh <start> <end>`, `analyze|scenarios --snapshot-date`.

Measured: 23 surveys visible live → 14 at a 2026-02-14 clock → 11 at the
02-05 start; `T_stable` tracks the clock instead of running to the WRF file's
end. Pinned by `tests/test_replay_clock.py` (17 tests; 7 fail if the gating is
reverted).

---

## 5. Seeding

The replay needs a **temporally coherent** snowpack state at `T_start`.

**The existing `.sno` files cannot provide it.** Their `ProfileDate` is ragged:

```
1789 clusters @ 2026-01-31    608 @ 2026-01-10    300 @ 2026-01-18
 871 clusters @ 2026-02-07    505 @ 2026-03-31    176 @ 2025-12-27
 741 clusters @ 2026-02-13    476 @ 2026-01-03    ... 18 distinct dates
```

Those dates match the 18 `reinit_stats_*.json` files. Chained multi-event
reinit with partial pass-2 reruns (the `CLUSTERS_FILE` filter) left each
scoured group stranded at its own event date while only 505 clusters reached
season end. The `.pro` files are complete to 03-31, so the *history* is intact
and only the *restart state* is incoherent.

This is also a finding about the live tree, not merely a replay obstacle — see
§8.3.

### Seed procedure (run once, then freeze)

1. Create the replay slope (§3).
2. `export AVACHAIN_AS_OF=2026-02-05` — the seed itself must be gated, so the
   last complete transport window is 01-27→02-05, both known.
3. Generate SMETs for the replay slope and **truncate them to `T_start`**.
   Non-negotiable: see §8.2.
4. Run SNOWPACK from season start (SMETs begin 2025-11-26T18:00) to
   `2026-02-05T12:00`.
5. Build the replay Zarr from the resulting `.pro`.
6. **Snapshot the whole state immutably** — `.sno` + truncated SMETs + Zarr +
   cluster map — as `replay_seed_2026-02-05/`.

Every subsequent experiment (frozen-splits run, splits-on run, post-update
regression run) restores from that snapshot. The seed is the expensive part;
pay it once.

The replay's `.pro` can start fresh at `T_start`; the historical `.pro` are not
needed, which avoids duplicating ~300 GB.

---

## 6. The tick loop

```
restore replay_seed_<T_start>
for T_now in T_start .. T_end:
    export AVACHAIN_AS_OF=T_now
    mode = classify(T_now)          # survey arrived? event detected? plain day?
    case survey:  run_little_prof.sh --snapshot T_now   # resample→gap_fill→smet→SNOWPACK
    case event:   run_little_prof.sh --reinit --event-date ... --snapshot T_now
    case plain:   run_daily.sh --snapshot T_now
    archive(ticks/<T_now>/)         # features, scenarios, logs, anchor manifest
    assert(tier3_invariants)
```

Two rules:

- **Event detection must run, not be read from a list.** `detect_minkernel` on
  the newly-available survey pair decides whether reinit fires. Whether it
  fires on 02-20 — matching the season run — is a *test result*, not an input.
- **Archive every tick.** Operational value is "what did the system say on
  Feb 24?", and §11.4 forecast verification needs the T+0/T+1/T+2 archive.

---

## 7. Verification: three tiers

### Tier 1 — invariance (build and run this FIRST)

**SNOWPACK restart exactness.** `src/avachain/tier1_restart_check.py`. A shared
warmup produces one byte-identical `.sno` at `T_start`; arm A then runs
`T_start → T_end` in a single invocation while arms B_k run it in k-day steps,
restarting each time. Same binary, same SMETs, same `master_config.ini`.

```bash
python src/avachain/tier1_restart_check.py \
    --toml slopes/little_prof/slope_config.toml \
    --clusters 8 --start 2026-02-05 --end 2026-02-13 --steps 1,2,4,8
```

**Result: PASS** (8 clusters spread across the domain, 2026-02-05 → 02-13,
Snowpack 3.7.1).

| restarts | max ΔHS (m) | max Δthick (m) | max ΔT (K) |
|---|---|---|---|
| 0 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1 | 3.450e-04 | 1.550e-02 | 2.067e-02 |
| 3 | 4.760e-04 | 1.550e-02 | 2.704e-02 |
| 7 | 6.190e-04 | 1.551e-02 | 3.000e-02 |

Zero restarts gives exactly zero — the control, showing the harness itself
introduces nothing. Layer counts are identical everywhere; `mk`, `ne`, `Rho_S`,
`Conduc_S`, `HeatCapac_S`, `Vol_Frac_S` are bit-identical; total depth is
conserved to under 1 mm. **1 → 7 restarts (7×) grows ΔHS by 1.79× — sub-linear**,
consistent with the `.sno`'s 6-decimal storage quantum accumulating like a
random walk rather than a trajectory diverging. The harness asserts that
sub-linearity rather than merely printing it, because it is the property a
32-tick replay rests on.

**Three tiers of check, because the first version got this wrong.** A
per-element *relative* metric is useless — sphericity sits near 0, so a 1e-4
wobble read as a 26% error. And a blanket 1 mm per-layer bound also failed, for
a real reason: `COMBINE_ELEMENTS` merges thin elements and a restart can make
that decision land differently. In 1 of 8 clusters ~7 adjacent layers
redistributed by up to 1.55 cm while that cluster's total depth stayed within
0.19 mm; the other 7 moved < 0.3 mm. So internal boundary placement is a
legitimate discretisation freedom and **total depth is the invariant**.
Therefore: structural quantities exact, total depth conserved, per-layer
quantities bounded by values *measured* from this run at ~1.3× the worst
observed. Those bounds are characterisation, not a correctness claim — do not
widen them to make a run pass.

**Carry this caveat forward.** Per-layer density (`Vol_Frac_I`, which with
`Vol_Frac_V` sums to 1) differs by up to 1% in the worst layer. Since
E ∝ ρ^2.8 that is ~3% in E, hence in Λ and A_ca. Bounded and non-accumulating,
but it is the floor on how exactly any layer-resolved replay comparison can
agree.

### Tier 2 — the measurement (must *not* match)

The season run's `gap_fill` built each inter-survey transport field from **both**
bracketing surveys. At replay `T_now` mid-window the next survey does not
exist, so HS is station-extrapolated instead of interpolated.

Divergence therefore grows with days-since-survey and collapses at each
arrival — a sawtooth. **That sawtooth is the deliverable.** It is §11.1's
survey-correction magnitude: how wrong the HS forcing goes between surveys,
which is the dominant uncertainty in daily-forward mode.

Report per inter-survey period: median and P90 per-cluster HS RMSE, spatial
correlation. Targets in §11.1 of the design concept (good < 10 cm, acceptable
10–25 cm).

Nothing may be tuned to shrink this.

### Tier 3 — orchestration assertions

- Mode dispatched correctly for each tick.
- **Anchor `.sno` lifecycle.** Zero `_anchor.sno` exist anywhere today, so
  §3.2's restore mechanism has never run and will be created from scratch here.
- **Post-reinit restart coherence** — after pass 2, all clusters at one
  `ProfileDate`. This assertion is what the live tree fails (§5).
- Reinit scoured the expected cluster set.
- SMET append idempotence — no duplicate rows, no gaps.
- **Zarr append non-duplication.** ~32 sequential appends is exactly the path
  that produced the 9× duplicated store, so the replay doubles as a regression
  test for that fix.
- Cluster-ID stability (frozen-splits run).
- No silent `--toml` fallback to the default `ProjectConfig`.

---

## 8. Known constraints

### 8.1 NWP replay cannot measure forecast skill

The NWP source is `/ssd/snowpack/fcst/<season>/zone<NNN>/<grid>/<id>_res.smet`
— a CAIC **SNOWPACK result file** (`history = Run ... Snowpack-3.7.0`) driven
by WRF, updated every 6 h and extending ~72 h ahead. `nwp_ingest` reads its
forcing columns (TA, RH, VW, DW, ISWR, ILWR, MS_Snow).

For the 2025/26 season that file spans 2025-09-23T12:00 → 2026-06-04T18:00 at
6 h, 1018 rows, **no duplicate timestamps** — one continuous series, not an
archive of per-cycle issuances. A timestamp that is now in the past was last
written when it was near-term, so replaying a historical `T_stable` reads rows
at short lead and calls them a 48 h forecast. If those rows are effectively
analysis, forecast skill would look *spuriously perfect*.

The whole `fcst/2025` tree was last written **2026-06-01** (season end; no
`fcst/2026` yet), so the feed is dormant. Two consequences:

- **Good:** the input is frozen, so replay NWP output is perfectly
  reproducible.
- **Bad:** we cannot observe the update behaviour to determine the issuance
  lead of historical rows. A snapshot-and-diff experiment is impossible this
  season.

**Therefore:** exercise the path, write a `plumbing_only: true` marker into
every forecast artifact so nobody later mines them for skill numbers, and if
genuine verification is wanted, ask CAIC whether `res.smet` rows are written
once at first forecast or rewritten each cycle — and whether per-cycle
issuances are retained anywhere.

### 8.1b The season baseline was built by a SNOWPACK version that is gone

Every `.pro` in the tree carries
`#Run 2026-05-12T..., with Snowpack-3.7.0, compiled on Aug 26 2025`. The
installed binary is now **3.7.1 (compiled 2026-10-13)**, and
`/home/caic/caic/rtsys/snowpack/exe/` holds only `snowpack-git` (→ 3.7.1) and
`snowpack-3.6.0`. **3.7.0 cannot be reproduced.**

So a replay-vs-season-Zarr comparison confounds three variables: binary
version, restart pattern, and data gating. Only the third is of interest.

**Therefore the Tier 2 reference is not the season Zarr but a *full-knowledge
replay*** — same 3.7.1 binary, same incremental tick pattern, survey gating
**off**. Differencing that against the gated replay isolates the information
deficit exactly, which is what §11.1 of the design concept asks for. The season
Zarr drops to a sanity anchor: broad agreement expected, exact agreement not.

The same confound applies to the regenerated 03-07 full-chain baseline, which
is current analysis code reading 3.7.0 SNOWPACK output. Compare gated vs
full-knowledge replay there too, both 3.7.1.

### 8.2 SMETs must be truncated to `T_start`

Confirmed empirically: with SMETs still reaching 2026-03-31, `smet_append` at a
02-10 clock appended **0 rows across all 6565 files**, because
`last_ts >= until` short-circuits. Every append would be a silent no-op and the
replay would run entirely on pre-existing forcing.

### 8.3 Ragged restart state in the live tree

See §5. Worth fixing independently of the replay; the Tier 3 coherence
assertion is the test that would have caught it.

### 8.4 Runtime

| Step | Measured | Per 32 ticks |
|---|---|---|
| `smet_append` scan | 2m 37s | ~83 min |
| `analyze` (warm, deduped Zarr) | 18.9 s | ~10 min |
| `scenarios` (75) | 7m 20s | ~66 min (9 runs) |
| SNOWPACK daily increment | ~5 min (design doc estimate) | ~2.7 h |
| Seed run (season start → 02-05) | not measured | hours |

The `smet_append` cost is pure overhead: `_read_smet_last_timestamp` reads all
3002 rows × 64 fields of each file to find the last line. §6.1 of the design
concept already flags SMET I/O; reading backward would remove ~80 min per
replay. Not a correctness issue.

Plus six reinit three-pass runs, which dominate. Overnight-feasible.

---

## 9. Rules

- **Never tune to the Jan 18 event.** It is n = 1 and it is the validation
  target. The chosen window avoids it entirely. If a Jan pass is ever run, it is
  plumbing-only with no skill claim.
- **Tier 2 divergence is a measurement, not a defect.** Do not adjust gap-fill,
  thresholds or anything else to make replay match the season run.
- **Do not report a tier as passing without running it.** For a bug found by
  the replay, reproduce it first, then show the fix changes that specific
  behaviour, and check the test fails when the fix is reverted.

---

## 10. Comparison baselines

| Target | Baseline | Notes |
|---|---|---|
| Dense, per-cluster per-timestep | `slope_snowpack.zarr` (501 steps, 6-hourly, 6565 locations, 14 GB) | Raw SNOWPACK output, unaffected by downstream code changes — the trustworthy dense baseline. |
| Full chain | none (the 2026-03-07 scenarios were deleted 2026-10-25) | **There was no avalanche on 2026-03-07.** Its scenario set was a theoretical pre-storm envelope example, not an event; do not use it as one. `analysis/*_2026-03-07.*` remain as a snowpack date only. |

**Why regeneration was mandatory.** The pre-existing 03-07 artifacts predated
the θ-estimator port. Old vs new baseline shares **zero trigger clusters**
(52/1050/1898/2913/6365 → 1055/1721/2154/3646/4715), median release area 1.52×
and volume 1.36×. Comparing replay output against the stale set would not have
been noisy — it would have been meaningless. The stale set is archived at
`outputs/little_prof/stale_pre_theta_port/` with a README, including its
`T_0/T_1/T_2` NWP snapshots, which were deliberately **not** regenerated
(plumbing-only, §8.1).

Note this window has exactly **one** full-chain comparison date. The richer one
(01-18: features + 75 scenarios + metadata) is outside it by design.

---

## 11. Prerequisites — complete as of 2026-10-19

| # | Item | Result |
|---|---|---|
| 1 | Dedup Zarr swap | Live store 14 GB / 6565 locations; verified bit-identical before rename; 78 GB duplicate deleted after confirming `.pro` rebuild source intact |
| 2 | Regenerate 03-07 baseline | `analyze` 18.9 s (1608 clusters, not 14502); 75 scenarios in 7m20s |
| 3 | Path leaks | `SMET_DIR` default → `$SLOPE_DIR/input/smet` in template + generated script; `WEATHER_CACHE_DIR` config-derived; `--toml` added to `aws_ingest` and `smet_append` |
| 4 | Clock seams | `now=` / `as_of=` + `$AVACHAIN_AS_OF`; 17 tests |
| 5 | Weather cache | 5153 rows, 2025-09-01 → 2026-04-13, units match the SMET field-for-field |

| 6 | MeteoIO input plugins | **Fixed** — `METEO`/`SNOW = SMET` were missing from `[INPUT]` (and `SNOW` from `[OUTPUT]`), so SNOWPACK aborted with "No plugin defined for METEO" before reading data; `run_snowpack.sh` could not have run |

Four defects surfaced only because building the clock and Tier 1 made the
chain actually run for the first time:

- **`smet_append` wrote double-converted units.** It converted TA to Kelvin, RH
  to a fraction and treated HS as metres, but the SMET header already declares
  `units_offset = 273.15` and `units_multiplier = 0.01`, which SNOWPACK applies
  on read. A −10.7 °C row became 262.4 in the file and 535.6 K inside SNOWPACK;
  HS came out 100× small. Nothing downstream catches this. An existing test had
  pinned the divergence from `nwp_ingest` as intentional — but one file carries
  one units header, so two writers cannot hold two conventions.
- **`aws_ingest` and `smet_append` did not accept `--toml`**, which
  `run_daily.sh` passes to both, so it aborted at step 1 under `set -euo
  pipefail`. That script had never run.
- **`pyarrow` was undeclared** despite three modules using it.
- **The generated `master_config.ini` was missing the MeteoIO input plugins.**
  The complete config lived in the pre-rename tree at
  `snowpack/<slope>/config/`; three lines were dropped when `install.py` took
  over generating it at `slopes/<slope>/config/`, and since only one
  `master_config.ini` exists now, nothing shadowed the gap.

---

## 12. Build order

1. ~~**Tier 1 restart-exactness experiment** (§7)~~ — **done, PASS.**
2. Replay slope + seed procedure (§3, §5), ending in a frozen seed snapshot.
   Slope **done** (`slopes/little_prof_replay/`); seed commands verified on 3
   clusters, full-domain warmup not yet run. Runbook in §13.4-13.5.
3. Tick driver with mode dispatch and per-tick archiving (§6).
4. Tier 3 assertions (§7) — these are the module's real output.
5. Tier 2 comparison and reporting (§7, §10).
6. Second pass with cluster splitting enabled.

---

## 13. How to run

Status labels are literal. **[VERIFIED]** means it was run and the output in
this document came from it. **[NOT BUILT]** means the command is the intended
interface and does not exist yet — those blocks are the implementation spec.

### 13.1 Environment and the one lever

```bash
cd /home/ron/avachain
source .venv/bin/activate
```

Everything is gated by one environment variable:

```bash
export AVACHAIN_AS_OF=2026-02-14    # simulated UTC "now"
unset  AVACHAIN_AS_OF               # live behaviour (surveys unbounded, wall clock)
```

Confirm it is doing something before trusting a run:

```bash
python - <<'EOF'
import os, sys; sys.path.insert(0, 'src/avachain')
from pathlib import Path
from config import ProjectConfig
import pipeline_io
cfg = ProjectConfig.from_toml(Path('slopes/little_prof_replay/slope_config.toml'))
print(len(pipeline_io.discover_surveys(cfg)), "surveys visible")
EOF
```
23 unset, 14 at 2026-02-14, 11 at 2026-02-05. **[VERIFIED]**

### 13.2 One-time prerequisites

Already done on this checkout (§11). On a fresh clone, in this order:

```bash
# Zarr: deduplicate, verify, swap. Keeps the original as .duplicated.
python src/avachain/dedup_zarr.py --toml slopes/little_prof/slope_config.toml \
    --skip-write --swap --verify-clusters 32 --verify-times 8

# Weather cache from the archived CAIC CSV (the live fetch is still a stub).
python src/avachain/aws_ingest.py --toml slopes/little_prof/slope_config.toml \
    --from-csv            # add --dry-run first to see the span

# Historical (deleted 2026-10-25): 2026-03-07 was a theoretical pre-storm example,
# not an avalanche. Kept only as a record of how the set was built.
unset AVACHAIN_AS_OF
python src/avachain/analysis_pipeline.py analyze \
    --toml slopes/little_prof/slope_config.toml --snapshot-date 2026-03-07
python src/avachain/analysis_pipeline.py scenarios \
    --toml slopes/little_prof/slope_config.toml --snapshot-date 2026-03-07 \
    --n-triggers 5 --size-factors 0.7 0.85 1.0 1.15 1.3 --depth-pcts 10 50 90
```
**[VERIFIED]** — archive any pre-existing 03-07 artifacts first; see §10.

### 13.3 Tier 1 — restart exactness. Run this before anything else.

```bash
python src/avachain/tier1_restart_check.py \
    --toml slopes/little_prof/slope_config.toml \
    --clusters 8 --start 2026-02-05 --end 2026-02-13 --steps 1,2,4,8
```

Exit 0 = pass. Results in §7. Takes a few minutes; the warmup dominates, and
`--keep-warmup` reuses it across reruns. **[VERIFIED]**

Read the *drift vs restart count* table, not just the verdict. The verdict can
pass while the trend turns super-linear, and the trend is the thing a 32-tick
replay depends on.

### 13.4 Create the replay slope

Already exists as `slopes/little_prof_replay/`. To recreate:

```bash
mkdir -p slopes/little_prof_replay
sed -e 's/^name             = "little_prof"/name             = "little_prof_replay"/' \
    -e 's/^display_name     = "Little Professor"/display_name     = "Little Professor (replay)"/' \
    -e 's|^output_dir            = "outputs/little_prof"|output_dir            = "outputs/little_prof_replay"|' \
    -e 's|^slope_dir             = "snowpack/little_prof"|slope_dir             = "snowpack/little_prof_replay"|' \
    slopes/little_prof/slope_config.toml > slopes/little_prof_replay/slope_config.toml

python install.py slopes/little_prof_replay/slope_config.toml
```

Then check isolation — nothing may point at the live slope:

```bash
grep -nE "little_prof[^_]" slopes/little_prof_replay/run_snowpack.sh \
    run_little_prof_replay.sh | grep -v little_prof_replay    # expect no output
```
**[VERIFIED]**

### 13.5 Seed the replay

The seed produces one coherent restart state at `T_start`. Required because the
live `.sno` are ragged (§5).

```bash
export AVACHAIN_AS_OF=2026-02-05
export SLOPE_DIR=$PWD/snowpack/little_prof_replay
export SMET_DIR=$SLOPE_DIR/input/smet
T=2026-02-05T12:00
```

**Stage SMETs truncated to `T_start`.** Copying the live SMETs and cutting them
is correct, not a shortcut: rows up to 02-05 depend only on the 01-27 and 02-05
surveys, both of which are known at `T_start`. Cutting is also mandatory — an
untruncated SMET makes `smet_append` short-circuit on `last_ts >= until` and
every tick silently appends nothing (§8.2).

```bash
for src in snowpack/little_prof/input/smet/cluster_*.smet; do
  awk -v cut="$T" '
    /^\[DATA\]/ {print; indata=1; next}
    !indata      {print; next}
    { if (substr($1,1,16) <= cut) print }
  ' "$src" > "$SMET_DIR/$(basename "$src")"
done
```
**[VERIFIED on 3 clusters]** — 3001 rows → 1699, last row exactly
`2026-02-05T12:00:00`, header byte-identical. The full 6565 run has not been
done.

**Warm up SNOWPACK to `T_start`.** Begin date defaults to the first SMET row.

```bash
bash slopes/little_prof_replay/run_snowpack.sh "" "$T"
```
Expect `OK: <n>  Failed: 0` and `ProfileDate = 2026-02-05T12:00:00` in each
`output/cluster_*_cluster_*.sno`. **[VERIFIED on 3 clusters]** — the full-domain
warmup (~71 simulated days × 6565 clusters) has not been run and is the first
genuinely expensive step.

**Freeze the seed.** Every later experiment restores from this, so it is not
optional — and `run_snowpack.sh` is not idempotent (§13.8).

```bash
tar -C snowpack/little_prof_replay -czf \
    outputs/replay_seed_2026-02-05.tgz input output
```
**[NOT BUILT]** as a script; the command works.

### 13.6 Run the replay ticks — **[NOT BUILT]**

```bash
python src/avachain/replay_run.py \
    --toml slopes/little_prof_replay/slope_config.toml \
    --seed outputs/replay_seed_2026-02-05.tgz \
    --start 2026-02-06 --end 2026-03-08 \
    --scenarios-on surveys \
    --no-cluster-split \
    --archive outputs/little_prof_replay/ticks
```

Per tick it must: set `AVACHAIN_AS_OF`, classify the day (survey / detected
event / plain), dispatch to `run_little_prof.sh` or `run_daily.sh`, archive to
`ticks/<date>/`, and assert the Tier 3 invariants (§7). Event detection runs
`detect_minkernel` on the newly visible survey pair — whether it fires on
02-20 is a result, not an input.

A second run with `--cluster-split` and a separate archive covers the
splits-on pass.

### 13.7 Compare — **[NOT BUILT]**

Per §8.1b the reference is a **full-knowledge replay**, not the season Zarr:
same binary, same tick pattern, gating off. That isolates the information
deficit from the 3.7.0 → 3.7.1 version change and from restart drift.

```bash
# reference arm: identical, but with the survey cutoff disabled
python src/avachain/replay_run.py ... --no-gating --archive .../ticks_fullknowledge

python src/avachain/replay_compare.py \
    --gated outputs/little_prof_replay/ticks \
    --reference outputs/little_prof_replay/ticks_fullknowledge \
    --sanity-zarr snowpack/little_prof/output/slope_snowpack.zarr \
    --report outputs/little_prof_replay/replay_report.md
```

Expect the Tier 2 sawtooth (§7). Do not tune anything to flatten it.

### 13.8 Gotchas that cost time

- **`run_snowpack.sh` is not idempotent.** It restarts from the output `.sno`
  whenever one exists, so repeating a window asks SNOWPACK to simulate from a
  state already at the end date and it fails with `Cannot generate Xdata`. To
  redo a step, restore the seed. Each tick must advance.
- **Pass full timestamps or bare dates, consistently.** `EDATE` used to append
  `T18:00` unconditionally, producing `2026-02-05T12:00T18:00` and
  `Date object is undefined!`. Fixed, but the asymmetry is easy to reintroduce.
- **A `| tail` pipeline hides a non-zero exit.** Capture to a file and echo
  `$?`, or use `${PIPESTATUS[0]}`.
- **`export SMET_DIR` and `SLOPE_DIR` for the replay slope.** The generated
  defaults are correct now, but being explicit removes any chance of writing
  into the live slope.
- **Logs live in the slope output dir**, one per cluster:
  `snowpack/little_prof_replay/output/cluster_XXXX.log`. Filter the warnings:
  `grep -vE "^\[WARNING\]" <log> | tail`.
- **The weather cache is shared and that is intentional** — raw station
  observations, identical for every slope; gating happens via
  `smet_append --until`.

## 14. Open questions

- How are CAIC `res.smet` rows maintained — written once at first forecast, or
  rewritten each cycle? Are per-cycle issuances retained anywhere? Decides
  whether forecast skill is ever measurable (§8.1).
- Should the ragged restart state in the live tree be repaired now, or left as
  the thing the replay demonstrates (§8.3)?
- Is one full-chain comparison date enough, or should a Jan window be added as
  a plumbing-only second pass (§10)?
