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
YYMMDD_<path>_snowHeight.tif      snow height raster, metres, EPSG:6342 (+NAVD88)
YYMMDD_<path>_snowHeight.READY    marker, written AFTER the TIF is complete
```

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
| WRF forecast | `/ssd/snowpack/fcst/<season>/zone055/077963/0779632_res.smet` | Same host. It extends forcing 48 h past the last observation and refills station gaps. The season is the year of `[snowpack] season_start`. A missing or short file stops the run (`RefillError`); the run does not continue on bad forcing. |

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
    density.json              slab density
    params.json               trigger, size factor, depth percentile, physics settings
  scenario_weights.json       probability weight of each scenario
  scour_depth.tif             entrainable snow depth for com1DFA (m)
  trigger_locations.geojson
  summary.csv  metadata.json
```

The ensemble is `n_triggers` × `size_factors` × `depth_percentiles`. The values
are in the TOML's `[scenarios]` table; today that is 5 × 5 × 3.

`manifest.json` records:
- `producer`, `snapshot_date` and `horizon`;
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
3. Write only under `work/runout/<path>/` and `handoff/<path>/runout/`.
4. Never modify the release set.

### 2.6 Runout → forecast (planned)

This follows the same contract. Runout publishes
`handoff/<path>/runout/<date>/<horizon>/` with runout and exceedance rasters, a
`manifest.json` and `READY` last. The forecast step reads that set and writes
products to `outputs/<path>/forecasts/`. Use `code/avachain/src/avachain/handoff.py`
(`staging_dir`, `publish`, `is_ready`) instead of re-implementing it.

---

## 3. How the chain runs

All three entry points live in `code/avachain/` and are generated by `install.py`
from the slope TOML.

| Script | When | What it does |
|---|---|---|
| `run_Professor_PTC.sh` | A survey has arrived | Claims the oldest `READY` survey, then runs resample → transport → avalanche → gap_fill → cluster_update → smet → smet_append → nwp_ingest → SNOWPACK survey cycle → analyze → scenarios (publishes §2.5). Its EXIT trap marks the survey `DONE` or `FAILED`. If nothing is `READY`, it exits without doing anything. |
| `run_daily_Professor_PTC.sh` | Every morning, between surveys | aws_ingest → smet_append → nwp_ingest → SNOWPACK daily cycle (from yesterday's anchor to today, plus a 48 h forecast) → analyze → scenarios. It does not touch surveys, clusters or SMET rewrites. |
| `run_full_pipeline.sh` | Setup, or after a change to static inputs, physics or forcing | Full rebuild: all forcing steps, full-season SNOWPACK with the two-pass avalanche reinit (`--reinit`), a single Zarr build. Takes about 6–8 h. Run it detached (`setsid nohup …`) so it survives a logout. |

The SNOWPACK state carries over between runs as restart files in
`work/snowpack/<path>/output/`. There are three kinds:
- `<cid>_<cid>.sno`: the latest state;
- `<cid>_survey.sno`: the state at the last survey;
- `<cid>_anchor.sno`: the observed state for the daily runs.

A survey run restarts from the previous survey's state. A daily run restarts
from the daily anchor, never from a forecast state.

**Avalanches.** An avalanche re-initialises the snowpack of the scoured clusters
(the reinit). Only events confirmed in `work/snowpack/<path>/avalanche_events.json`
do this. Auto-detection appends candidate periods to that file but does not
relabel periods that are already listed. The 2025-26 season has one avalanche,
on 2026-01-18.

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
1. Clone the repo to `code/avachain`, `uv sync`, then
   `python install.py --slope Professor_PTC`. This generates the scripts and
   `master_config.ini`/`template.sno`, creates the directories, and prints the
   crontab line.
2. Put the static inputs in `data/static/<path>/`, the station CSV in
   `data/weather/`, and the events file in `work/snowpack/<path>/`.
3. Put existing surveys in `data/surveys/snow_height/<path>/` with `.DONE`
   markers, so they count as history rather than new work.
4. Run `run_full_pipeline.sh --reinit …` to build the cluster map, SMET files,
   the full SNOWPACK season, the Zarr store and the first analysis.
5. Initialise the survey and daily anchors (`snowpack_cycle.py init-anchors`).
6. Pass acceptance: the Jan 18 feature CSVs and scores must be byte-identical
   to the dev tree's, followed by a February replay with `$AVACHAIN_AS_OF`.
7. Install cron last, using the line `install.py` prints, which starts with
   `umask 002;`.

**Per survey, from the producer:** the TIF, then its `READY` marker, named as
in §2.1. Nothing else is needed.

---

## 5. Status and open items (2026-10-29)

- **Not installed yet.** `code/` is empty. The acceptance rehearsal is running
  in `scratch/ron/rehearsal/`, a full copy of this layout. The old clone at
  `/home/avachain/avachain` and Val's `intermediate/` skeleton are from before
  this layout and will be removed when the install lands.
- **Live station fetch is a stub.** `aws_ingest._fetch_station` returns nothing
  until the CAIC API or CDOT RWIS endpoint and token are confirmed. Until then,
  the daily run's station forcing comes only from the seeded CSV cache, so the
  daily path cannot run unattended on live data.
- **No runout consumer yet.** Nothing reads `handoff/<path>/release/` yet. The
  com1DFA runner (Ron, since 2026-10-28) must follow §2.5–2.6.
- **Cron cadence is undecided.** `install.py` currently schedules the survey
  script once a day at `[cron] hour_utc`. With markers, the proposal is to poll
  `run_<path>.sh` every 10–15 min (a tick with nothing `READY` exits silently)
  and run `run_daily_<path>.sh` once each morning.
- **Path name.** The dev tree keeps `little_prof` until the deployment passes
  acceptance, then gets one mechanical rename to `Professor_PTC`.
