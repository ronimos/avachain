# Adding a New Slope

This guide covers everything needed to configure and run the avachain pipeline
on a slope other than Little Professor.

---

## Overview

Each slope lives in `slopes/<name>/` and is described by a single configuration
file, `slope_config.toml`.  The `install.py` script reads that file to:

1. Create the expected directory structure
2. Find the closest WRF forecast grid cell (distance + aspect match)
3. Generate `slopes/<name>/config/master_config.ini` (SNOWPACK physics)
4. Generate `slopes/<name>/run_snowpack.sh`
5. Generate `run_<name>.sh` at the project root
6. Print the crontab line for daily automation

---

## Prerequisites

Before running `install.py` you need:

| Item | Where to get it |
|------|----------------|
| SNOWPACK binary | `/home/caic/caic/rtsys/snowpack/exe/snowpack` (already on the server) |
| Bare-ground DSM | UAS survey flight over bare ground (GeoTIFF, EPSG:6342 or UTM) |
| Initial-conditions template | Copy `snowpack/little_prof/input/snow/template.sno` and edit `SlopeAzi` |
| Season start date | First date snow is expected on the slope |
| Weather station IDs | CAIC `staname` values from the SQL database |
| Slope boundary KML | Drawn in Google Earth or exported from QGIS |
| Release zone KML | Drawn to match the avalanche start zone |

The WRF forecast data (`/ssd/snowpack/fcst/`) is scanned automatically by
`install.py` — you do not need to find the right zone manually.

---

## Step 1 — Create the slope configuration file

Copy the annotated template and fill in every field:

```bash
cp slopes/template/slope_config.toml slopes/<your_slope>/slope_config.toml
$EDITOR slopes/<your_slope>/slope_config.toml
```

### Key fields to fill in

**`[slope]`** — identifying metadata

| Field | Notes |
|-------|-------|
| `name` | Short identifier, no spaces (e.g. `"loveland_e2"`).  Used for all directory names and generated script names. |
| `display_name` | Human-readable label for plots and logs. |
| `aspect_deg` | Predominant aspect, degrees clockwise from north (N=0, E=90, S=180, W=270). Used to select the WRF aspect variant. |
| `elevation_m` | Representative elevation of the release zone. |
| `latitude`, `longitude` | Approximate centre of the slope (decimal degrees). Used to find the nearest WRF grid cell. |

**`[paths]`**

| Field | Notes |
|-------|-------|
| `project_dir` | Absolute path to this repository root. |
| `dem` | Path to the bare-ground DSM, relative to `project_dir`.  Use `data/<name>/dem/dem.tif`. |
| `survey_dir` | Directory for UAS snow-depth GeoTIFFs.  Use `data/<name>/surveys`. |
| `weather_csv` | Merged weather CSV written by `sql_util.py`.  Use `data/<name>/weather/weather_data.csv`. |
| `boundary_kml` | Full slope boundary KML.  Use `data/<name>/boundaries/boundary.kml`. |
| `start_zone_kml` | Avalanche start zone KML.  Same boundaries dir as above. |
| `output_dir` | Per-slope output root.  **Must be** `outputs/<name>` — using `outputs` (flat) will collide with other slopes. |
| `slope_dir` | Runtime SNOWPACK data (input/output).  Default: `snowpack/<name>`. |

**`[[stations]]`** — one entry per weather station

Roles `"summit"` and `"base"` are required by the pipeline.  `sql_id` must match
the `staname` column in the CAIC SQL weather database.

```toml
[[stations]]
role        = "summit"
sql_id      = "MYST1"            # staname from the SQL DB
name        = "My Summit Station"
lat         = 39.12
lon         = -106.34
elev_m      = 3600.0
sql_columns = ["swin", "temp", "dewp", "rh", "wspd", "wdir", "gust"]

[[stations]]
role        = "base"
sql_id      = "MYST2"
name        = "My Base Station"
lat         = 39.10
lon         = -106.34
elev_m      = 3200.0
sql_columns = ["pcpac", "depth", "snow24h"]
```

**`[snowpack]`**

| Field | Notes |
|-------|-------|
| `season_end` | YYYY-MM-DD date after which SNOWPACK stops (e.g. `"2026-04-30"`). |
| `binary` | Path to the SNOWPACK executable. |

**`[wrf]`**

Set `fcst_base_dir = "/ssd/snowpack/fcst"`.  Leave the `wrf_smet_*` fields blank —
`install.py` fills them in automatically.

**`[cron]`**

`hour_utc` is the UTC hour at which the daily pipeline fires.  03:00 UTC = 20:00 MST,
which is evening after the typical afternoon UAS flight.

---

## Step 2 — Run the installer

```bash
python install.py --slope <your_slope>
```

The installer will:

- Create all required directories under `paths.project_dir`
- Scan `/ssd/snowpack/fcst/<current_season>/zone*/` (all ~145 zones) to find the
  WRF grid cell nearest to your slope's lat/lon, then pick the aspect variant
  (N/E/S/W at 38° inclination) closest to `aspect_deg`
- Write the WRF selection back into your `slope_config.toml`
- Write `slopes/<name>/config/master_config.ini` (SNOWPACK physics config matching
  the CAIC operational setup; skip if file already exists)
- Write `slopes/<name>/run_snowpack.sh`
- Write `run_<name>.sh` at the project root
- Print the crontab line

If `/ssd` is not mounted at install time, pass `--skip-wrf` and re-run without it
later to fill in the WRF selection.

### Review the WRF selection

After running, verify the selected grid cell makes geographic sense:

```bash
grep "wrf_" slopes/<your_slope>/slope_config.toml
# wrf_zone  = "zone055"
# wrf_grid_id = "077963"
# wrf_dist_km = 0.714
# wrf_smet_file = "/ssd/snowpack/fcst/2025/zone055/077963/0779632_res.smet"
```

If the distance is unexpectedly large (> 5 km) or the zone looks wrong, check your
`latitude`/`longitude` values and re-run.  You can also set `season` explicitly:

```toml
[wrf]
fcst_base_dir = "/ssd/snowpack/fcst"
season        = "2026"           # use this year's directory
```

---

## Step 3 — Drop the required data files

Place these files before running the pipeline:

| What | Where | Notes |
|------|-------|-------|
| Bare-ground DSM | `data/dem/<filename>.tif` | Match `paths.dem` in the TOML |
| First UAS survey | `data/surveys/<YYMMDD>_*_snowHeight.tif` | See survey filename format below |
| Slope boundary | `data/boundaries/<slope>.kml` | Match `paths.boundary_kml` |
| Release zone | `data/boundaries/<start_zone>.kml` | Match `paths.start_zone_kml` |
| Weather CSV | `data/weather/weather_data.csv` | Generated by `sql_util.py` (Step 4) |
| `template.sno` | `snowpack/<name>/input/snow/template.sno` | Copy from little_prof and edit `SlopeAzi` |

**Survey filename format:** `YYMMDD_<anything>_snowHeight.tif`
where `YYMMDD` is the survey date (e.g. `251201_MySlope_snowHeight.tif`).

**`template.sno`:** The SNOWPACK initial conditions template.  Copy the little_prof
version and edit one line:

```
SlopeAzi = 67.5   ← change to your slope's aspect_deg
```

---

## Step 4 — Fetch initial weather data

Pull the full season's weather history from the CAIC SQL database:

```bash
cd src/avachain
python sql_util.py \
    --start-time "2025-09-01 00:00:00" \
    --target ../../data/weather \
    --project-dir ../.. \
    --slope-name <your_slope>
```

This reads the station IDs and SQL column lists from your `slope_config.toml`
and writes `data/weather/weather_data.csv`.

---

## Step 5 — Run the full pipeline

The first run builds the cluster map, WindNinja library, transport features, and
SNOWPACK restart files.  Subsequent runs (after new surveys) use the operational
script instead.

```bash
# Edit run_full_pipeline.sh to point at your project dir, or use:
cd /path/to/snowpack_model_feeder

# Run each forcing step in order
python src/avachain/forcing_pipeline.py --project-dir . resample
python src/avachain/forcing_pipeline.py --project-dir . transport
python src/avachain/forcing_pipeline.py --project-dir . features
python src/avachain/forcing_pipeline.py --project-dir . train
python src/avachain/forcing_pipeline.py --project-dir . cluster
python src/avachain/forcing_pipeline.py --project-dir . gap_fill
python src/avachain/forcing_pipeline.py --project-dir . smet

# Run SNOWPACK (full season)
bash slopes/<your_slope>/run_snowpack.sh

# Analyze and generate scenarios
python src/avachain/analysis_pipeline.py --project-dir . analyze
python src/avachain/analysis_pipeline.py --project-dir . scenarios
```

Full pipeline runtime is approximately 4–6 hours depending on slope size and
cluster count.

---

## Step 6 — Set up the daily cron job

The installer printed a crontab line at the end.  To activate it:

```bash
crontab -e
```

Paste the printed line, for example:

```
0 3 * * * /home/ron/snowpack_model_feeder/run_<name>.sh >> /home/ron/snowpack_model_feeder/outputs/logs/cron_<name>.log 2>&1
```

The daily script (`run_<name>.sh`) runs the incremental operational pipeline after
each afternoon survey flight:

```
resample → gap_fill → cluster_update → smet → SNOWPACK (incremental) → analyze → scenarios
```

Drop each new survey TIF into `data/surveys/` before the scheduled run time.
The pipeline automatically detects the most recent survey.

To trigger a manual run immediately:

```bash
./run_<name>.sh
# or with options:
./run_<name>.sh --snapshot 2026-02-15 --n-triggers 3
```

---

## Step 7 — Handling avalanche reinitialization

When an avalanche has released on the slope, the SNOWPACK restart files need to be
scoured before continuing the simulation.  Pass `--reinit` along with the event
and survey dates:

```bash
./run_<name>.sh \
    --snapshot 2026-01-17 \
    --reinit \
    --event-date  2026-01-18 \
    --date-before 2026-01-14 \
    --date-after  2026-01-20
```

The two-pass SNOWPACK run:
1. Simulates up to the event date
2. Analyzes snowpack for slab thickness at the event snapshot
3. Scours the release cluster restart files
4. Re-simulates from the event date to the end of season

If you have a hand-drawn release boundary (recommended when the automatic detection
misses part of the crown):

```bash
./run_<name>.sh --reinit ... \
    --reinit-geojson data/boundaries/avalanche_release_area_20260118.geojson
```

---

## Listing all configured slopes

```bash
python install.py --list
```

```
Slope name            Display name                    Config
--------------------------------------------------------------------------------
little_prof           Little Professor                slopes/little_prof/slope_config.toml
loveland_e2           Loveland E2                     slopes/loveland_e2/slope_config.toml
```

---

## Re-running the installer

Re-run `install.py` any time you change the TOML (new season, updated paths,
changed season end date).  It regenerates the shell scripts and updates the WRF
selection but does not overwrite `master_config.ini` if it already exists.

```bash
python install.py --slope <your_slope>
```

---

## Notes on new weather stations

If your slope uses station IDs that are not already CAABT/CAABM, the station info
flows through the pipeline automatically via `slope_config.toml` for:

- `sql_util.py` — which columns to pull from the SQL database
- `aws_ingest.py` — which stations to cache
- `smet_writer.py` — station lat/lon/elevation used for ILWR estimation
- `smet_append.py` — which cache file to read for daily SMET append

No code changes are needed for new station IDs.  The only manual verification
step is confirming that the `sql_id` values in your TOML match the `staname`
column in the database — you can check with:

```sql
SELECT DISTINCT staname FROM obsWX ORDER BY staname;
```

---

## Troubleshooting

**`install.py` reports WRF distance > 5 km**
Check `latitude`/`longitude` in the TOML — they must be the slope centre, not a
corner.  Distances over ~3 km are normal for high-elevation terrain with coarse WRF
grids; over 10 km suggests a coordinate error.

**SNOWPACK fails on first run with "file not found"**
Check that `template.sno` is in `snowpack/<name>/input/snow/` and that
`master_config.ini` is in `slopes/<name>/config/`.  The SNOWPACK run script prints
the exact paths it expects.

**`sql_util.py` returns empty data**
Verify the `sql_id` values with the SQL query above.  Also confirm the SSH tunnel
credentials in `.env` (`ssh_server`, `ssh_pw`) are current.

**Cluster map looks wrong (wrong terrain extent or too few clusters)**
The DEM must cover the full slope extent defined by `boundary_kml`.  If the cluster
count is much lower than expected, check `target_cells_per_cluster` and
`max_cells_per_cluster` in the TOML's `[scenarios]` section.  These are passed
through to `ProjectConfig`.
