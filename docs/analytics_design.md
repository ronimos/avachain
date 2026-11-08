# Season analytics — design (draft for review)

Ron, 2026-11-08. Status: **phase 1 built** on `feat/season-analytics`
(`src/avachain/metrics.py`, tests in `tests/test_metrics.py`). Decisions are
in *Decisions (Ron, 2026-11-08)* at the end; where the build differs from the
first draft it is noted inline.

## Goal

Next summer we want to improve the chain from what happened this season. Some
of that can be recomputed from archived inputs. The rest exists only on the
day: what was issued, what the model believed before a survey corrected it,
how long things took, and what happened on the slope. This design captures
the second kind cheaply, every run, and leaves the first kind to summer
scripts.

So every item below is classified:

- **LIVE-ONLY**: lost unless recorded that day.
- **KEEP-INPUT**: derivable in summer *if* an input is kept; the design only
  makes sure it is kept.
- **EXISTS**: already written to a file; only needs to be collected.

## Rules

1. **Analytics never change a run's outcome.** Like `alerts.py`: every write
   is wrapped, and a failure prints a warning and carries on. The exit status
   and markers are the pipeline's, not the recorder's.
2. **Append-only, one line per record.** JSONL, one file per record type and
   season. No database. Every record carries `schema` (an int), `run_id`,
   `slope`, `commit`, `toml_sha256` and `written_utc`.
3. **Raw values, not judgements.** Store the numbers; compute skill scores
   in summer. A metric definition we change in July must not need a rerun.
4. **One small module.** `src/avachain/metrics.py` (`record(kind, **fields)`),
   called from the run scripts and the few Python places listed below. Run
   scripts call it through a tiny CLI, so bash steps can record too.

## Where it lives

`[paths] metrics_dir` and `issued_dir` (Professor_PTC:
`/home/avachain/work/metrics/Professor_PTC`, `/home/avachain/archive/Professor_PTC/issued`):

```
<metrics_dir>/<season>/            season = 2026-27, from [snowpack] season_start
    runs.jsonl          one line per daily or survey run
    surveys.jsonl       one line per survey processed
    cycles.jsonl        one line per release set published (model and shots, each T_n)
    observations.jsonl  ground truth: avalanches, shots, forecaster notes
<issued_dir>/<season>/<cycle_date>/
    forecast_store.zarr         the forecast store at the T_n snapshot timesteps
    superseded_<run_id>/        what one rerun replaced:
        release/<date>/T_n ...  the release sets (moved, not copied)
        www/{model,shots}/      the published reports (copied: the live page stays up)
        forecast_store.zarr     the previous snapshot, if the rerun replaced it
```

Every record carries `run_id`: the run scripts export `AVACHAIN_RUN_ID`
(UTC start stamp) once, so all subprocesses of one run share it. Every record
also carries `calibration_version` (`[calibration] version`, null until the
release model is calibrated).

Disk: `/home` has 6 TB free. At season end the main store is ~16 MB per
timestep, so three snapshot timesteps per cycle is ~50 MB, ~10 GB a season.

## 1. Run record (`runs.jsonl`) — LIVE-ONLY

One line at the end of every run, from the EXIT trap so a failed run also
writes one.

| Field | Source |
|---|---|
| `kind` (daily / survey), `as_of`, `started_utc`, `finished_utc`, `exit_code`, `failed_step` | run script; `failed_step` is what the alert already names |
| `steps{name: seconds}` | the `stepN_elapsed` vars already computed (install.py daily :896-975, survey :645-760); today printed only |
| `snowpack{pass: {seconds, ok, failed, clamped_slope}}` | `run_snowpack.sh` summary lines (observed and forecast passes) |
| `zarr_seconds`, `lock_wait_s`, `wrf_wait_s` | new timers around the build, `flock`, `nwp_ingest.wait_for_wrf` |
| `runout{status, per_horizon{n_sims, seconds}}` | `$RUNOUT_STATUS` + runout manifest |
| `report_published_local`, `deadline_margin_min` (vs 03:00) | mtime of the published `report.html` |
| `t_stable`, `nwp_edate`, `nwp_clamped`, `wrf_cycle` | already echoed by the daily script (install.py:927-928) |
| `stations{id: {rows, last_obs_utc, gap_h}}` | `aws_ingest` fetch (rows are printed today) |
| `alerts[]` | subjects sent during the run |
| `disk_free_gb` | `df` on /home |

Answers: what is slow, how close we run to 03:00, how often WRF or stations
are late, which step fails.

## 2. Survey record (`surveys.jsonl`) — mostly LIVE-ONLY

The model's HS *before* a survey corrects it is overwritten by the anchor.
That prior is the cleanest measure of forcing drift we have.

| Field | Source | Today |
|---|---|---|
| `survey`, `flown_date`, `tif_arrived_utc`, `run_started_utc`, `report_utc` | marker JSON (`survey_markers.py`), TIF mtime | partly |
| `frame_check{verdict, p10_m, p50_m, cover}` | `survey_check_<date>.json` | EXISTS |
| `anchor{top_down, thinned, below, held, no_survey}` | `anchor_to_survey` counts (reinitialize_snowpack.py:451-535) | returned, not saved |
| `prior_minus_survey{p1, p10, p50, p90, p99}` overall **and by group** (start zone / path / elevation band / aspect octant) | from `anchor_stats_<date>.json` (per cluster, EXISTS) + cluster terrain | per-cluster file exists, summary missing |
| `after_anchor{p1_m, p99_m, below}` | `check_anchor_held` (snowpack_cycle.py:447-484) | **returned and discarded** at :247 |
| `period_label`, `reinit`, `transport_corrected` | events file / `classify_period` | EXISTS |
| `reinit{n_clusters, scour_m p50/p90}` when a reinit ran | `reinit_stats_<date>.json` | EXISTS |

The per-cluster `anchor_stats_<date>.json` and `reinit_stats_*.json` already
survive; the record adds the summaries and the two numbers that are
currently thrown away. Summer use: drift by terrain group between surveys,
whether the forcing is biased high or low, and what daily-cadence thresholds
should be (see the survey-frequency constraint).

## 3. Cycle record (`cycles.jsonl`) — LIVE-ONLY for the funnel

One line per forecast day T_n published.

| Field | Source | Today |
|---|---|---|
| `cycle_date`, `horizon`, `snapshot_ts`, `store`, `clamped`, `status`, `trigger_source`, `n_scenarios` | handoff `manifest.json` | EXISTS |
| `funnel{start_zone, tau_g, slope, sk38, slab, s_t, elev}` counts | `_build_scenarios` (analysis_pipeline.py ~883-970) | **printed only** |
| `near_miss[]`: top ~10 start-zone clusters by Sk38 with each gate's margin (Sk38, tau_g, slab, S_t, slope) | features + meloche frames at that point | not kept |
| `triggers[]{cid, sk38, slab_m, r_c}` | trigger_locations.geojson | EXISTS |
| `areas{p10, p50, p90}`, `mean_depth_m`, `density` | summary.csv / metadata | EXISTS |
| `max_hs_m` | no_snow sets | EXISTS |

**Not recorded live (built):** day-to-day IoU and runout reach. Both are
derivable in summer from the kept sets and published products, so by rule 3
they are summer scripts.

The funnel and near-misses tell us in summer which gate decided each day.
That decides what is worth tuning. The day-to-day IoU is the forecasters'
"flicker" complaint, measured.

## 4. Keep the inputs (KEEP-INPUT)

| Item | Status | Action |
|---|---|---|
| WRF, every cycle | `wrf_archive.py` hourly, manifests with sha256 | none: WRF vs station skill is a summer script over this + the station cache |
| Station observations | `work/weather/<station>.parquet` | none (back it up with the archive) |
| **Forecast store** | rebuilt with `mode='w'` every cycle (snowpack_cycle.py:503-506); **nothing survives** | copy the T_n snapshot timesteps to `issued/<cycle>/` after Step 5 |
| Forecast feature CSVs | `analysis/forecast/<cycle>/T_n/`, never cleaned | none; include in the season archive |
| Handoff release sets | kept per date; a **rerun replaces** the date's set (handoff.py:120) | before replacing, move the old set to `issued/<cycle>/superseded_<utc>/` so the record holds what forecasters actually saw |
| Published reports | kept per date under `publish_root`; rerun overwrites | same: keep the replaced `report.html` |
| Survey TIFs, markers | kept | none |

With the forecast store and the WRF archive kept, "T+1 skill against T+0" and
"WRF against station" are summer scripts, not live code.

## 5. Ground truth (`observations.jsonl`) — LIVE-ONLY, and the one that matters most

Nothing else here can recalibrate the release model. One record per
observation:

| Field | Notes |
|---|---|
| `type` | `avalanche` / `control_shot` / `forecaster_note` |
| `time_local`, `observer`, `source` (CDOT log, CAIC obs, report form, Ron) | |
| shot: `location [lat, lon]` or shot-point name, `charge`, `result` (no result / sluff / slab), `size`, `crown_depth_m` | a shot with **no result is a measured negative** at a known point |
| avalanche: `size` (D / R), `trigger` (natural / explosive / skier), `crown_polygon` path or sketch, `crown_depth_m`, `width_m`, `reached_road` | polygon from survey when one is flown |
| note: `report_date`, `useful` (y/n), free text | |
| `linked_cycle` | filled automatically: the issued cycle covering that time |

**Input (Ron, 11-08): observations are emailed.** Claude has no inbox, so
an emailed observation reaches the record either by being forwarded or pasted
into a Claude session, or (later) through the Gmail connector reading a
dedicated label. Either way it is entered with:

```bash
python src/avachain/metrics.py --toml slopes/Professor_PTC/slope_config.toml observe \
    --type control_shot --time "2026-12-03 07:40" --source "CDOT log" --observer "<name>" \
    --field shot_point='"PW-3"' --field result='"no_result"' --note "..."
```

`--field` values are parsed as JSON when they parse (numbers, true/false),
else kept as text. Unlike the automatic records, `observe` fails loudly on a
bad type or time. CDOT log or CAIC imports can come later without a schema
change.

## Phasing

**Phase 1 (this week, before real snow):**
- `metrics.py` + CLI, `runs.jsonl` from both run scripts (timings, exit,
  stations, NWP, runout, deadline margin).
- `surveys.jsonl` (save the counts and the discarded `check_anchor_held`
  result; summarise `anchor_stats`).
- `cycles.jsonl` with the funnel and near-misses.
- Forecast-store snapshot and superseded-set keeping.
- `observations.jsonl` + `observe` CLI.
- Tests: a failing recorder never changes exit status; each record round-trips;
  the funnel counts match the printed lines on a fixture day.

**Phase 2 (summer):** analysis scripts over the records and kept inputs:
WRF-vs-station skill by lead, T+1/T+2 vs T+0, survey drift by terrain group,
step-time trends, gate attribution, flicker, observation-vs-forecast scoring
(reusing `event_scoring.py` and `runout/scripts/verify.py`).

A weekly digest (one email: runs, failures, deadline margin, survey drift,
observations logged) is optional, and cheap once the records exist.

## Decisions (Ron, 2026-11-08)

1. Observations are emailed (§5).
2. Superseded sets and reports are kept (§4).
3. Forecast store: snapshot timesteps only.
4. Terrain groups (default, not discussed): start zone vs outside, 100 m
   elevation bands, aspect octants from the SMET `slope_azi` (compass bearing,
   map frame).
5. Weekly digest: not now.

## Where each record is written (as built)

| Record | Written by |
|---|---|
| run | EXIT trap of `run_daily_<slope>.sh` / `run_<slope>.sh` (install.py templates) → `metrics.py run`; step times from `STEP_TIMES`, the rest parsed from the run log |
| survey | survey run's EXIT trap → `metrics.py survey`; `anchor_counts_<date>.json` (new, `anchor_to_survey`) and `anchor_check_<date>.json` (new, survey cycle) |
| cycle | `analysis_pipeline.step_scenarios` after each publish; funnel and near-misses from `_build_scenarios` |
| forecast store | `analysis_pipeline.step_forecast` → `metrics.keep_forecast_store` |
| superseded sets | `analysis_pipeline._publish_set` → `handoff.publish(keep_old=)` |
| superseded reports | `runout_step` (after the runout lock) → `metrics.keep_published_reports` |
