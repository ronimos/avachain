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

**SNOWPACK restart exactness.** On a small cluster set, compare a day-by-day
incremental advance from `T_start` to `T_end` against one continuous run over
the same span. These must agree.

This has nothing to do with data gating — it is a property of SNOWPACK's
restart files. If it fails, every other replay number is uninterpretable. It is
cheap, and it gates the whole module. Do not skip it because the harness looks
more interesting.

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
| Full chain | `outputs/little_prof/scenarios/2026-03-07` + `analysis/*_2026-03-07.*` | **Regenerated 2026-10-19 with current code.** |

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

Three defects surfaced only because building the clock made the daily-forward
chain run for the first time:

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

---

## 12. Build order

1. **Tier 1 restart-exactness experiment** (§7). Gates everything.
2. Replay slope + seed procedure (§3, §5), ending in a frozen seed snapshot.
3. Tick driver with mode dispatch and per-tick archiving (§6).
4. Tier 3 assertions (§7) — these are the module's real output.
5. Tier 2 comparison and reporting (§7, §10).
6. Second pass with cluster splitting enabled.

---

## 13. Open questions

- How are CAIC `res.smet` rows maintained — written once at first forecast, or
  rewritten each cycle? Are per-cycle issuances retained anywhere? Decides
  whether forecast skill is ever measurable (§8.1).
- Should the ragged restart state in the live tree be repaired now, or left as
  the thing the replay demonstrates (§8.3)?
- Is one full-chain comparison date enough, or should a Jan window be added as
  a plumbing-only second pass (§10)?
