# Starting a season

How a slope starts a new season from bare ground, before its first UAS survey,
and what changes on the first survey and on thin early-season days. Written
2026-11-04 for the 2026-27 start on Little Professor (`Professor_PTC`). The
steps were rehearsed end to end with real SNOWPACK on the 2025-26 data
(`/home/avachain/scratch/ron/season_start/rehearsal/`, results at the end).

The operational tick list is `/home/avachain/GO_LIVE_CHECKLIST.md` (§2).

---

## 1. Clusters for the new season

The cluster map must exist before anything else runs. There are two ways to get
it, and neither needs a survey.

**A slope with a previous season: carry its clusters** (Ron, 2026-11-03).

```bash
python src/avachain/cluster_carry.py --toml T \
       --from-map <last season archive>/cluster_map.tif [--max-fill-m 3]
```

The TOML's `[paths] dem` must already be the new ground surface.
`cluster_carry.py` resamples last season's cluster IDs onto the new DEM grid in
the map frame. It rebuilds the domain mask, fills cells within `--max-fill-m`
of an old cluster, and reports cells farther away. Those cells are left out,
because new terrain has no old cluster's snow history. **IDs are kept**, so
the outputs of the two seasons can be compared cluster by cluster. Per-cluster
slope and aspect are recomputed from the new DEM by the `smet` step.

The case for carrying: on the 2025-26 surveys, a map built on the late season
fitted the early-season surveys almost as well as one built on them (cell HS
R² 0.919 against 0.935). `cluster_update` splits clusters that turn out to be
heterogeneous. Dry run onto the 260605 DEM: 6 540 of 6 565 clusters kept.

**A slope's first season: terrain-only clusters.**

```bash
python src/avachain/cluster_terrain.py --toml T [--cells-per-cluster 10]
```

These are SLIC superpixels on slope, aspect (sin, cos), curvature, relative
height and elevation, at about one cluster per 10 cells. On the 2025-26 surveys
they fitted HS as well as the production clusters at the same count (median R²
0.928 / 0.894 for HS and HS change, against 0.918 / 0.886). On Jan 18 the
release model scored the same on both: expected IoU 0.228 against 0.200,
AUC 0.66 against 0.68, n = 1. Whether Little Professor should switch to them
is Ron's decision and still open.

## 2. TOML for the season

```toml
[snowpack]
season_start           = "2026-10-15"   # snow-free, and on or after the WRF file's start
season_end             = "2027-04-30"
start_from_bare_ground = true

[weather]
source = "cache"                       # station data from the database cache, no CSV
```

- `season_start` must be snow-free **and** on or after the first row of the
  season's WRF file. The `smet` step refuses a refill that starts after
  `season_start`. The 2026 file starts at 2026-10-14 12:00.
- Surveys outside `[season_start, season_end]` are ignored.
- `source = "cache"` reads `work/weather/<station>.parquet`, filled by
  `aws_ingest.py`. It is identical to the CSV path on all 5 153 hours of 2025-26.
- Re-run `install.py` after editing. It regenerates the scripts, `template.sno`
  and both SNOWPACK configs (§5), and the WRF file follows `season_start`.

## 3. Bootstrap: bare ground to now

```bash
python src/avachain/aws_ingest.py --toml T --since <season_start>
python src/avachain/season_start.py --toml T bootstrap
```

`bootstrap` needs empty SMET, `.pro` and Zarr locations. It refuses to run over
an existing season, so archive last season's tree first and never delete it.
It then:

1. writes station-only hourly HS grids from `season_start`, with HS 0 on every
   cell. Every cluster follows the station's HS until the first survey;
2. writes the cluster SMETs;
3. runs SNOWPACK from bare ground to the first profile step and saves that as
   the **survey anchor**;
4. runs on to the last observation, saves the **daily anchor** and builds the
   Zarr store.

From then on `run_daily_<slope>.sh` carries the state forward every morning.

## 4. The first survey

A survey cycle re-runs SNOWPACK from the survey anchor. For the season's first
survey that anchor is bare ground, so the whole early season is re-run on
forcing rebuilt from the survey. `gap_fill`'s bare-ground segment ramps from the
station's HS timing to the survey's HS at the flight hour. The timing stays the
station's, not WRF's (Ron, 2026-11-04), even though the A-Basin base station
melted out after every October storm while the WRF column kept 11–19 cm. The survey anchor then
moves to the survey, and later surveys behave as in any season. On a survey,
the `smet` step rewrites only the rows after the previous survey anchor:
3–4 min instead of about 1 h 15 min for 6 565 clusters.

## 5. The NWP forecast pass

Every survey and daily cycle ends with a forecast (`[nwp] lead_hours`, 60 h, clamped to
the WRF outlook's last timestamp; see `docs/deployment.md` §2.2) past the last
observation (`T_stable`), forced by the WRF column at the slope.

- WRF has no HS, and the observed config forces SNOWPACK by measured HS
  (`ENFORCE_MEASURED_SNOW_HEIGHTS = TRUE`). So the forecast pass runs on
  `master_config_forecast.ini`. That file imports `master_config.ini` and sets
  the flag to `FALSE` in both `[SNOWPACK]` and `[SNOWPACKADVANCED]`. Snow
  then comes from precipitation (PSUM), split into snow and rain by the
  existing PSUM_PH rule.
- `nwp_ingest` writes a **separate** SMET per cluster to
  `<slope_dir>/input/smet_forecast/`, with columns TA RH VW DW ISWR ILWR PSUM HS
  and HS as nodata. PSUM is WRF's MS_Snow + MS_Rain, held backward over each
  6 h step. Rows start 6 h before `T_stable`, because PSUM is accumulated over
  the 15 min before each step. **Observed SMETs are never touched.**
- The forecast state never becomes an anchor. Each cycle starts from the daily
  anchor.

Before 1cc40e2 the forecast pass stopped every cluster at `T_stable` with
`[E] missing { HS }`, and the survey was marked FAILED.

Every WRF cycle is archived (`wrf_archive.py`, hourly cron, see
`docs/deployment.md` §2.2), because the CAIC run overwrites the file each
morning. Without the archive a forecast cannot be checked afterwards.

## 5b. Forecast days and the 03:00 report

Ron, 2026-11-04. The runout report is due at 03:00 local, so the daily run
starts at 00:00 local (`CRON_TZ=America/Denver` in the cron.d file). Each
cycle produces three days, T_0..T_2, at 18 UTC (11:00 MST / 12:00 MDT). They
come from the observed store where it reaches and from the NWP forecast
store otherwise, and T_2 is clamped to the WRF outlook's end when it lies
beyond it. Runout runs each day with its own scour raster and publishes the
report after each one, so a partial report is out at 03:00 if the run is
late. A 03:15 alert fires when there is no report. Details: `docs/deployment.md` §2.6 and §3.

## 6. Thin early-season days: release scenarios

Early in the season no cluster passes the trigger gates (τ_g ≥ 40 Pa, slope,
Sk38 < 1, slab 0.5–2.0 m, tensile support). `scenarios` used to raise
`ValueError` and fail the survey. Since c38fc4c (Ron, 2026-11-04):

- the triggers are the slope's **explosive-control shot locations**,
  `[scenarios] explosive_trigger_points = [[lat, lon], ...]` (WGS84). Each
  trigger is the cluster that contains the point on the current map, so the
  points hold for any season's map. They get no gate, ranking or Π₁ test, and
  equal weight. The release model still decides what each one releases;
- any scenario, model or shot location, whose release holds less than
  **0.10 m** of snow on average is dropped (`MIN_RELEASE_DEPTH_M`). A model
  trigger with a real slab can still give a near-empty polygon on a patchy
  pack (run 5 below);
- if none is left, the set is published empty: manifest `status:
  "no_scenarios"`, `n_scenarios: 0`, and the survey ends DONE. The manifest's
  `trigger_source` is `"model"` or `"explosive_fallback"`, so runout and the
  report can tell why the set looks as it does.

Little Professor's points are the centres of 2025-26 clusters 3500, 2568, 2480
and 4137.

## 7. Rehearsal (2025-26 replayed as a live season)

`/home/avachain/scratch/ron/season_start/rehearsal/`, driver `run_rehearsal.sh`.

| Run | Date | Result |
|-----|------|--------|
| 1 | 11-03 | Bootstrap 2025-10-01 → 11-25 on 6 565 clusters OK. On the 251126 survey the bare-ground rebuild ramped 0 → 16 cm and the survey anchor held on 2 461/2 461 clusters (model − survey p1 −1.6 / p99 +1.9 cm). Then the NWP pass failed with `missing { HS }`. |
| 3 | 11-03 | With `--no-nwp`: `scenarios` raised "No candidate trigger clusters". |
| 4 | 11-04 | NWP on, with fallback. **251126 DONE**: forecast pass to 11-29 00:00 with no errors. The fallback's 60 releases held 0–3 mm (HS 0 at three of the four points) and were all dropped, so the set was empty. 251129 failed in a gap-fill plot (one period); fixed in 907db7d. |
| 5 | 11-04 | **251129 DONE, exit 0**, 43 min. This is the first normal survey cycle (11-26 → 11-29) of the season. The auto-detected period is `wind_scour` (transport correction, no reinit). The survey anchor held on 5 334/5 334 clusters (model − survey p1 −9.2 / p99 +1.8 cm). The forecast pass ran to 12-02 00:00 with no errors. The model found 2 triggers (Sk38 0.25 / 0.31, slab 0.68 / 1.11 m) and published 30 scenarios, `trigger_source: "model"`, density 116–117 kg/m³. |

In run 5 both triggers fell back to the oriented rectangle (790–1 630 m²), and
those rectangles averaged only 0.01–0.03 m of snow. Ron, 11-04: the 0.10 m
floor now applies to model triggers too. Rerun on that day, all 30 are
dropped and the set publishes as `no_scenarios` with `trigger_source:
"model"`.
