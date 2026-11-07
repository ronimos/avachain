# /home/avachain — how the chain fits together

This tree is the operational deployment of **avachain** for highway avalanche
operations on US-6 (Loveland Pass, CDOT). It takes UAS snow-depth surveys and
weather and produces probabilistic avalanche runout for each survey and each day:

```
 UAS survey ──► SNOWPACK ──► release areas ──► runout (com1DFA) ──► forecast
 (snow height)  (per cluster)  (scenario set)    (exceedance)         (products)
```

This file covers the **interfaces**: which part owns which directory, which
files cross from one part to the next, how a part knows its input is complete,
and what has to be in place for the chain to run. The physics and the code are
documented in the repo at `code/avachain/README.md` and `code/avachain/CLAUDE.md`.

The source of this file is `docs/deployment.md` in the repo; `/home/avachain/README.md`
is a copy of it. Edit the repo copy and copy it over.

One run serves one **path**, meaning one slope. Its name is the survey producer's
path token, which is also used for the folders, scripts and logs. Today the only
path is `Professor_PTC` (Little Professor; the dev tree calls it `little_prof`).
Wherever `<path>` appears below, read it as that token.

---

## 1. Layout and ownership

```
/home/avachain/
  code/avachain/                  the repo (git; dev = active, main = operational)
  data/                           EXTERNAL ARRIVALS — the pipeline never writes here,
    surveys/snow_height/<path>/     except to rename survey markers (§2.1)
    static/<path>/                DEM, boundary + start-zone KML, mapped crowns, validation/
    weather/                      archived station CSV
  work/                           PRIVATE working files, one subtree per part
    snowpack/<path>/              SMET, .sno, .pro, Zarr store, hourly grids, resampled surveys,
                                  avalanche_events.json
    weather/                      station cache (<station>.parquet)
    release/<path>/               cluster map, feature CSVs, analysis, event masks
    runout/<path>/                com1DFA run directories
  handoff/<path>/                 THE ONLY PLACE PARTS EXCHANGE FILES
    release/<date>/<horizon>/     scenario sets: release → runout
    runout/<date>/<horizon>/      runout sets: runout → forecast (planned)
  outputs/<path>/                 products: forecasts, plots, verification
  logs/<path>/                    one log per run, plus cron_<path>.log
  scratch/{ron,val}/              personal; nothing operational reads it
```

Rules:

1. **A part reads only `data/`, its own `work/` subtree, and `handoff/` sets
   addressed to it.** It never reads another part's `work/`. If a downstream
   part needs a file, the producer publishes it to `handoff/`.
2. **Only the producer writes a handoff set; only the consumer reads it.**
   For example, release never writes under `handoff/<path>/runout/`.
3. **`data/` is for things that come from outside the chain.** The one
   exception is the survey marker rename (§2.1). Files the pipeline edits, such
   as `avalanche_events.json`, which the avalanche step appends to, live in
   `work/`.

Every path the code uses comes from one file, the slope TOML at
`code/avachain/slopes/<path>/slope_config.toml`. All paths in it are absolute.
To see where a deployment actually reads and writes:

```bash
cd /home/avachain/code/avachain && source .venv/bin/activate
python src/avachain/config.py --print-paths --toml slopes/Professor_PTC/slope_config.toml
```

The shell scripts take their paths from the same output (`AVA_*` variables),
so nothing in a script hardcodes a directory.

**Permissions.** The tree is a setgid group tree: `val:caic`, `drwxrwsr-x`.
Anything that writes here, whether a person, a script or cron, needs
`umask 002`. Otherwise the other account cannot rerun or clean up after it.
The crontab line `install.py` prints already starts with `umask 002;`.

---

## 2. The interfaces

### 2.1 Survey → chain (survey processing → snowpack)

**The producer delivers two files in `data/surveys/snow_height/<path>/`:**

```
YYMMDD_<path>_snowHeight.tif      snow height raster, metres, EPSG:6342
YYMMDD_<path>_snowHeight.READY    marker, written AFTER the TIF is complete
```

**Snow height is snow DSM − ground DEM, and both must be in one frame.**
Since 2026-11-07 (Ron) the frame is NAD83(2011) / UTM 13N with NAD83(2011)
**ellipsoidal** heights, no geoid. A USGS 3DEP check showed every DEM in the
series was already ellipsoidal, whatever its NAVD88 tag said. At Little
Professor the geoid separation is −12.42 m (GEOID18), so a snow DSM with a
geoid applied, differenced against the ellipsoidal ground, reads about 12 m
of snow everywhere. For each survey the producer:

- processes the snow DSM with the same SiteScan output setting as the ground
  DEM: horizontal NAD83(2011) UTM 13N, vertical ellipsoidal, no geoid;
- uses the same control as the ground DEM: the 2026-10 GCPs and the local
  base station, not last season's distant base or the Kremmling CORS;
- differences against the path's current ground DEM
  (`static/<path>/terrain/…-ellipsoid.tif`), never against last season's
  surfaces or 3DEP;
- reports any change (receiver, base, SiteScan version or license, geoid
  setting) before delivering, so the change can be checked first.

The marker's state lives in its file extension. The producer only ever creates
`READY`; from then on the chain moves it along:

```
READY ──claim──► RUNNING ──► DONE
                        └──► FAILED ──(rename back to READY to retry)──► READY
```

| State | Meaning |
|---|---|
| `READY` | New survey, not started. Always means "not started". |
| `RUNNING` | Claimed by a run. The rename is the claim, so two overlapping runs cannot both take a survey. |
| `DONE` | The run for this survey finished. |
| `FAILED` | The run failed. Read the body, fix the cause, and rename it to `.READY` to retry. |

- **The marker body is a JSON record of the run.** It gives host, pid, commit,
  TOML, log path, start/end and exit code. On `FAILED` it also gives the failing
  step and the tail of the log. A retry keeps earlier attempts under `history`.
  Anything the producer wrote into `READY` is kept under `ready`.
- **Only `RUNNING` and `DONE` surveys are visible to the pipeline.** A bare TIF
  may still be copying, so it is ignored until its `READY` appears.
- **Surveys are processed oldest first.** A `READY` survey that is older than
  one already `DONE` is refused (exit 2), because snowpack state cannot go back
  in time. Re-running history is a manual rebuild (§3).
- **A run that died without cleaning up is caught by the next claim.** If a
  `RUNNING` marker's pid is gone (kill -9, reboot), the next claim marks it
  `FAILED`.

To see the state of every survey:

```bash
python src/avachain/survey_markers.py --toml slopes/Professor_PTC/slope_config.toml status
```

### 2.2 Weather → snowpack

| Input | Where | Notes |
|---|---|---|
| Station archive (CAIC A-Basin summit CAABT + base CAABM) | `data/weather/weather_data.csv` | Raw CAIC units: tenths of °F, mph and W/m². Converted only by `smet_writer.load_and_convert`. |
| Station cache | `work/weather/<station>.parquet` | Built from the CSV, then appended daily by `aws_ingest`. HS is stored in **cm**. |
| WRF forecast | `/ssd/snowpack/fcst/<season>/zone055/077963/0779632_res.smet` | Same host. It refills station gaps and drives the forecast (`[nwp] lead_hours`, 60 h, clamped to the outlook's last timestamp) past the last observation. The season is the year of `[snowpack] season_start`. A missing or short file stops the run (`RefillError`); the run does not continue on bad forcing. |

The forecast has its own forcing. `nwp_ingest` writes one SMET per cluster to
`work/snowpack/<path>/input/smet_forecast/`, with WRF precipitation as PSUM
and HS as nodata. The forecast pass reads them with
`slopes/<path>/config/master_config_forecast.ini`, which drops HS forcing.
Observed SMETs never carry forecast rows. See `docs/season_start.md` §5.

**WRF archive.** The CAIC run deletes `/ssd/snowpack/fcst/<season>/` and
rebuilds it every morning (about 07:40–07:55 local), so each cycle's forecast
is overwritten by the next. `src/avachain/wrf_archive.py` runs hourly from cron
and keeps one snapshot per cycle of the slope's grid cell (all files, about
1 MB) in `work/weather/wrf_archive/<season>/<zone>/<cell>/<cycle UTC>/`, with a
`manifest.json` of sha256 sums. It skips files younger than 10 min and cycles
identical to the last snapshot. This archive is what a 2026-27 forecast-skill
check needs; 2025-26 has none.

**The rebuild window.** During the rebuild the WRF file is missing, then present
without its forecast rows, then being written. Every reader waits it out
(`nwp_ingest.wait_for_wrf`). It needs the file present and unchanged for
10 min, and for `nwp_ingest` the data must reach at least 24 h past
`T_stable`. It polls every minute for up to 30 min, then fails as before.

**The forecast window** (Ron, 2026-11-04) is `T_stable` → `T_stable` +
`[nwp] lead_hours` (60 h), clamped to the WRF outlook's last timestamp
(`nwp_ingest.forecast_window`). A 00:00-local run uses the previous morning's
cycle and reaches exactly T+2 at 18 UTC. The run scripts take `T_stable` and
the forecast end from `nwp_ingest --print-window`, and only after `nwp_ingest`
has succeeded, so the forecast pass never runs past its forcing.

The SMET files the chain writes (`work/snowpack/<path>/input/smet/`) hold
**°C, % and cm**, because SNOWPACK applies the header's offsets and multipliers
when it reads them. Never write SI values into them.

### 2.3 Static inputs → chain

These go in `data/static/<path>/`: the snow-off DEM, the boundary KML, the
start-zone KML, mapped crowns (`avalanche_release_area_<YYYYMMDD>.geojson`), and
`validation/`, which holds the survey-derived release polygons used only for
scoring. They change only when the slope definition changes, and a change means
a full rebuild (§3).

### 2.4 Snowpack → release (within avachain, not a handoff yet)

Release reads the SNOWPACK Zarr store (`work/snowpack/<path>/output/slope_snowpack.zarr`)
and the cluster map (`work/release/<path>/analysis/`). Both parts run in the
same script and the same code, so this boundary is internal to avachain today.
If the two parts are ever split, the Zarr snapshot becomes a handoff set under
the same rules as §2.5.

### 2.5 Release → runout: the scenario set

The release part publishes one set per snapshot date and forecast horizon:

```
handoff/<path>/release/<YYYY-MM-DD>/<horizon>/
  READY                       written LAST; runout waits on this file
  manifest.json               what the set is and every file in it (below)
  scenarios/scenario_NNN/
    release.geojson           release polygon (EPSG:6342)
    depth.asc + depth.prj     release depth raster for AvaFrame (also depth.tif)
    density.json              slab density: median over the polygon's clusters, 280 kg/m³ if none
    params.json               trigger, size factor, depth percentile, physics settings
  scenario_weights.json       probability weight of each scenario
  scour_depth.tif             entrainable snow depth for com1DFA (m)
  trigger_locations.geojson
  summary.csv  metadata.json
```

The ensemble is `n_triggers` × `size_factors` × `depth_percentiles`. The values
are in the TOML's `[scenarios]` table; today that is 5 × 5 × 3.

On a day when no cluster passes the trigger gates (a thin early-season pack),
the triggers are the slope's explosive-control shot locations
(`[scenarios] explosive_trigger_points`). Any scenario, model or shot
location, whose release holds < 0.10 m of snow on average is dropped. If
nothing is left, the set is published **empty**: `manifest.json` and `READY`,
no `scenarios/`. `metadata.json` counts the dropped scenarios in
`n_below_depth_floor`.

`manifest.json` records:
- `producer`, `snapshot_date` and `horizon`;
- `status`: `"ok"`, or `"no_scenarios"` for an empty set; and `n_scenarios`;
- `trigger_source`: `"model"`, `"explosive_fallback"` (shot locations), or null
  when there were no triggers at all;
- `created_utc`;
- the code `commit` and `commit_dirty`, which is true if the code differed from
  that commit;
- the `toml` path and its `toml_sha256`;
- `files`: every file with its size and sha256.

**How the set is published.** The set is built in a hidden staging directory
next to its final place, gets its manifest, and is renamed into place.
`READY` is written after that. A consumer that waits for `READY` therefore
never sees a partial set.

**Republishing** happens on a manual rerun for the same date and horizon. It
removes the old `READY` first and then swaps the directories, so a consumer sees
the old set, no set, or the new set, never a mix. Subdirectories that the new
build did not produce, such as experiment runs, are carried across rather than
dropped.

**What a runout consumer must do:**
1. Act only on directories that contain `READY`.
2. Verify the files against `manifest.json`.
3. Treat `status: "no_scenarios"` as "nothing to run today", not as an error.
4. Write only under `work/runout/<path>/` and `handoff/<path>/runout/`.
5. Never modify the release set.

### 2.6 Release → runout → forecast

**Forecast days** (Ron, 2026-11-04). A slope with `[scenarios]
forecast_days` (Professor_PTC: 3) runs `analysis_pipeline forecast` in place
of `analyze` + `scenarios`. Day n of cycle date D is read at D+n,
`[scenarios] snapshot_hour_utc` (18 UTC = 11:00 MST / 12:00 MDT, on
SNOWPACK's 6 h profile grid). One snapshot time is used for every read:
features, release depth and scour. Each day reads:
- the observed store when it covers that time (an afternoon survey run's T+0);
- otherwise the NWP forecast store. The 00:00 run's observations end at
  `T_stable` ≈ 06 UTC, so all three days come from the forecast;
- past the forecast store's end, its last step (`clamped: true` in the manifest).

The forecast store is used only if it was built in this cycle, i.e. it starts
within one 6 h profile step after the observations end. When `nwp_ingest`
fails, the cycle keeps the previous forecast store, which would otherwise be
read silently as today's forecast.

A day nothing reaches is skipped. Sets go to
`handoff/<path>/release/<D>/T_0..T_2/`; their manifests add `cycle_date`,
`snapshot_ts`, `store` and `clamped`. Forecast days write their feature CSVs
to `analysis/forecast/<D>/T_n/`, never next to the observed analysis of that
date.

`src/avachain/runout_step.py` is the last step of both run scripts
(Ron, 2026-11-04). It runs inline, after the sets are published:

1. It reads every `handoff/<path>/release/<date>/T_n/` set (a single-day set at
   `<date>/` runs as T_0) only when `READY` exists and every file matches the
   manifest's size and sha256 (`handoff.verify`). A damaged day is left out
   with an error, and the others run; with none readable it exits 2.
2. It builds runout's tree under the TOML's `[runout] work_dir`
   (`work/runout/<path>/`). It writes `config/local.yaml`, links the repo's
   `runout/config/{paths,physics,profiles.yaml}`, and links each set as
   `scen/<profile>/<YYYYMMDD>/T_n`.
3. It takes `[runout] lock_file` (`work/runout/.runout.lock`, one for every
   slope), waiting if it must. com1DFA sims can reach >10 GB each, and the
   4-sim cap has to hold across slopes, not only per run.
4. It runs the runout venv (`runout/.venv`, built by `install.py` on the
   deployment's interpreter) with the `forecast` profile and
   `--snow-height set`: each day entrains its own set's `scour_depth.tif`
   (entrainment rule B).
   Runout writes its work tree, logs and products under `work_dir`. It
   publishes `report.html`, `details.html`, `viewer.html` and
   `scenarios.xlsx` to `[runout] publish_root`
   (`/home/www/html/ron/runout_forecast/forecasts/<path>/<date>/`, plus
   `latest/` and `index.html`).
5. **The report is published after each day finishes**, T+0 first (Ron: due
   at 03:00 local, partial if need be). Days not run yet show as "still
   running", and the viewer is built once at the end.

An empty release set (`status: "no_scenarios"`) needs no sims: the report says
why there is no forecast.

**A runout failure does not fail the survey.** Snowpack and release succeeded
and are published, and a FAILED survey would redo about 40 min of SNOWPACK to
retry runout. The run logs `RUNOUT_STATUS` and the command that reruns runout
alone. A run killed during runout still ends FAILED (signal traps).

---

## 3. How the chain runs

All three entry points live in `code/avachain/` and are generated by `install.py`
from the slope TOML.

| Script | When | What it does |
|---|---|---|
| `run_Professor_PTC.sh` | A survey has arrived | Claims the oldest `READY` survey, then runs resample → transport → avalanche → gap_fill → cluster_update → smet → smet_append → nwp_ingest → SNOWPACK survey cycle → analyze → scenarios (publishes §2.5) → runout (§2.6). Its EXIT trap marks the survey `DONE` or `FAILED`. A run killed by SIGTERM, SIGINT or SIGHUP (reboot, OOM, kill) exits 143 / 130 / 129 and is marked `FAILED`, so the survey is retried. Before 2026-11-04 it was marked `DONE`. If nothing is `READY`, it exits without doing anything. |
| `run_daily_Professor_PTC.sh` | Every night at 00:00 local, between surveys | aws_ingest → smet_append → nwp_ingest → SNOWPACK daily cycle (from yesterday's anchor to today, plus the forecast to T_stable + 60 h, clamped) → forecast days T_0..T_2 (analyze + scenarios each) → runout (§2.6), report published as each day finishes. It does not touch surveys, clusters or SMET rewrites. |
| `run_full_pipeline.sh` | Setup, or after a change to static inputs, physics or forcing | Full rebuild: all forcing steps, full-season SNOWPACK with the two-pass avalanche reinit (`--reinit`), a single Zarr build. Takes about 6–8 h. Run it detached (`setsid nohup …`) so it survives a logout. |

The SNOWPACK state carries over between runs as restart files in
`work/snowpack/<path>/output/`. There are three kinds:
- `<cid>_<cid>.sno`: the latest state;
- `<cid>_survey.sno`: the state at the last survey;
- `<cid>_anchor.sno`: the observed state for the daily runs.

**Cron** (Ron, 2026-10-29). `run_Professor_PTC.sh` is polled every
`[cron] poll_minutes` (10). A tick with nothing `READY` exits before its log
opens, so polling costs nothing, and a survey starts within minutes of its
marker. `run_daily_Professor_PTC.sh` fires once a day at `[cron] daily_hour`,
which is **00:00 America/Denver** (Ron, 2026-11-04: the report is due at
03:00). The run takes about 2 h 15 m with three days of runout. cron reads
`/etc/cron.d` in the host's local time, so the generated file declares
`CRON_TZ=` from `[cron] timezone` (default UTC), and DST is followed. Both run
under one `flock` on
`logs/<path>/.cron_<path>.lock`, because they write the same SMETs and restart
files. A poll that finds the lock taken skips that tick, and its survey stays
`READY` for the next one. The daily run waits for the lock instead.

**Where the cron jobs live** (Ron, 2026-11-04): in one system file per slope,
`/etc/cron.d/avachain-<path>`, running as the TOML's `[cron] user` (`ron`), not
in a personal crontab. `install.py` writes the source to `[cron] crond_dir`
(`/home/avachain/cron/avachain-<path>`): the survey poll, the daily run and the
hourly WRF archive. It then prints the command to install it:

```bash
sudo install -m 644 -o root -g root /home/avachain/cron/avachain-Professor_PTC /etc/cron.d/avachain-Professor_PTC
```

cron reads only root-owned files there, and skips symlinks and names with a
dot, so it must be a copy, and it must be reinstalled after every
`install.py`. Edit the TOML, never the installed file. Keep no avachain lines in
personal crontabs, or the jobs run twice.

A survey run restarts from the previous survey's state. A daily run restarts
from the daily anchor, never from a forecast state.

**Avalanches.** An avalanche re-initialises the snowpack of the scoured clusters
(the reinit). Only events confirmed in `work/snowpack/<path>/avalanche_events.json`
do this. Auto-detection appends candidate periods to that file but does not
relabel periods that are already listed. The 2025-26 season has one avalanche,
on 2026-01-18.

**Alerts** (2026-11-04). `src/avachain/alerts.py` e-mails the recipients in
`~/.config/avachain/alerts.toml` (`email = [...]`, of the user cron runs as;
mode 600, outside git, because the repo is public) plus any in the TOML's
`[alerts] email`, through the local sendmail, on:
- a survey marked `FAILED`. The mail names the step and carries the last
  40 log lines, and says to rename the marker to `.READY` once fixed;
- a runout failure, while the run stays successful;
- a daily run that exits non-zero, naming the last step it reached;
- no report by the deadline. `check-report` runs at `[cron] report_check`
  (03:15 local) from cron.d. During the season it alerts when today's
  `report.html` is not published, which also catches what no run can report
  itself: cron or the host down. Slopes without runout get `check-daily`
  instead (no daily run for 26 h, at 17:00).

With an empty list the alert is only printed to the run's log. An alert never
fails the run that sends it.

**Logs:** one file per run under `logs/<path>/`, and `logs/<path>/cron_<path>.log`
for cron's own output. The marker body of a failed survey points to its log.

---

## 4. What is needed for the chain to work

**On the host (in place today):**
- SNOWPACK binary `/home/caic/caic/rtsys/snowpack/exe/snowpack` and its library
  directory. Both paths are in the TOML `[snowpack]`.
- WRF SMET for the season under `/ssd/snowpack/fcst/`.
- Group `caic` with write access to the whole tree, and `umask 002` for every
  account that writes here, including in `~/.bashrc`.

**Setup, once per path:**
1. Clone the repo to `code/avachain`, then build the venv with a Python that
   lives inside the tree:
   `UV_PYTHON_INSTALL_DIR=/home/avachain/code/python uv sync --frozen --python 3.12.11`.
   uv's default Python sits under the installer's `~/.local` (mode 700), which
   other accounts and their cron jobs cannot reach. Then run
   `.venv/bin/python install.py --slope Professor_PTC`. This generates the scripts and
   `master_config.ini`/`template.sno`, creates the directories, and prints the
   crontab line.
2. Put the static inputs in `data/static/<path>/`, the station CSV in
   `data/weather/`, and the events file in `work/snowpack/<path>/`. Then seed
   the station cache from the CSV:
   `python src/avachain/aws_ingest.py --toml <TOML> --from-csv`.
   The full rebuild does not need the cache, but every daily run and replay
   does. Without it, `smet_append` stops with an error naming this command.
3. Put existing surveys in `data/surveys/snow_height/<path>/` with `.DONE`
   markers, so they count as history rather than new work.
4. Run `run_full_pipeline.sh --reinit …` to build the cluster map, SMET files,
   the full SNOWPACK season, the Zarr store and the first analysis.
5. Initialise the survey and daily anchors (`snowpack_cycle.py init-anchors`).
6. Pass acceptance: the Jan 18 feature CSVs and scores must be byte-identical
   to the dev tree's, followed by a February replay with `$AVACHAIN_AS_OF`.
7. Install cron last, using the lines `install.py` prints. Each starts with
   `umask 002;`.

**Per survey, from the producer:** the TIF, then its `READY` marker, named as
in §2.1. Nothing else is needed.

---

## 5. Status and open items (2026-11-01)

- **Installed 2026-11-01** from `main` into `code/avachain`. The rehearsal
  (`scratch/ron/rehearsal/`) and the Feb soak (`scratch/ron/rehearsal_soak/`)
  both passed. The install acceptance rebuild runs from
  `scratch/ron/install/acceptance.sh`; cron is added only after it compares
  byte-identical. The old `/home/avachain/avachain` clone and Val's empty
  `intermediate/` skeleton were removed on 2026-11-01.
- **Live station fetch** from the CAIC database has been deployed since
  2026-11-03 (`64a0d4b`). Stations come from the TOML, and credentials are in
  `~/.config/avachain/db.env`.
- **Season start (2026-11-04).** The season-start code, the NWP forecast pass,
  the thin-day shot-location triggers and the per-polygon density were merged
  after the season-start rehearsal passed. See `docs/season_start.md`, and the
  tick list in `GO_LIVE_CHECKLIST.md`.
- **No runout consumer yet.** Nothing reads `handoff/<path>/release/` yet. The
  com1DFA runner (Ron, since 2026-10-28) must follow §2.5–2.6.
- **Path name.** The dev tree keeps `little_prof` until the deployment passes
  acceptance, then gets one mechanical rename to `Professor_PTC`.
