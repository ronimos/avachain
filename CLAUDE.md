# avachain — Claude Code context

UAS snow depth → distributed SNOWPACK → release geometry → com1DFA runout
probability, for highway avalanche operations (CDOT, US-6 / Loveland Pass).

`README.md` already covers repository layout, pipeline phases, the scenario
ensemble and the parameter table. **This file is only the things that are not
obvious from reading the code**, plus the places where the code and the docs
disagree. Read `README.md` for structure; read this for conventions and traps.

---

## Environment

```bash
source /home/ron/avachain/.venv/bin/activate
python -m pytest tests/ -q          # ~300 tests, runs in <10 s — no excuse to skip
```

Entry points are plain scripts, not installed console scripts:

```bash
# Pipeline A — forcing: UAS surveys → SMET files
python src/avachain/forcing_pipeline.py <step> [--project-dir .]
#   steps: resample transport features train avalanche cluster
#          gap_fill cluster_update smet reinit | all | validate

# Pipeline B — analysis: Zarr → release geometry → com1DFA scenarios
python src/avachain/analysis_pipeline.py {zarr_build|analyze|scenarios|all} \
       --toml slopes/little_prof/slope_config.toml

# Orchestration
./run_little_prof.sh --snapshot YYYY-MM-DD   # survey arrival
./run_daily_little_prof.sh                     # between-survey daily forward
./run_full_pipeline.sh                        # full rebuild
```

---

## Import convention — flat, and it matters

Modules import siblings **flat**, relying on `src/avachain` being on
`sys.path`:

```python
import arrest_indices as _ai          # correct
from config import RASTER_EPSG        # correct
from avachain import arrest_indices   # WRONG — loads a *second copy*
```

`from avachain import <sibling>` works but creates a separate module object, so
a constant set on one copy is invisible to the other. Entry points do
`sys.path.insert(0, Path(__file__).resolve().parent)`; tests do it themselves.
`config.py` imports nothing from the project, so importing it anywhere is
cycle-free.

## Physics constants live at module tops, not in `config.py`

`config.py` is **pipeline** config — paths, station lists, scenario defaults.
Physics constants sit at the top of the module that uses them. This is a
deliberate difference from `~/avalanche-release-areas`, which centralises them
in its own `config.py`. Do not "fix" it during a sync.

Where to look:

| Constants | File |
|---|---|
| `L_SS`, `THETA_*`, `TAU_G_FEATURE_FLOOR`, `E_RELATION`, `K_IC_RELATION`, `LIGAMENT_MODEL` | `snowpack_analysis.py` (~lines 40–130) |
| `STAUCHWALL_DEG`, `MAX_BFS_CLUSTERS`, `USE_*` flags, `LAMBDA_*_FACTOR` | `release_geometry.py` |
| Stauchwall angle — **three copies**, all 29° since 2026-10-27 (Ron; was 28): `release_geometry.STAUCHWALL_DEG` (BFS defaults, scoring scripts), `config.ProjectConfig.stauchwall_deg`, and the slope TOML's `[scenarios] stauchwall_deg`, which **wins** for `analysis_pipeline` (CLI `--stauchwall-deg` overrides it). Change all three. Each degree turns clusters whose *mean* slope falls below it into downslope walls, so polygons jump rather than shrink: Jan 18 median gate-on area 7382 / 4580 / 827 m² at 28 / 29 / 30. | `release_geometry.py`, `config.py`, TOML |
| BFS parameters: `TAU_G_ABS_FLOOR`, `MIN_PROPAGATION_*`, Λ/thickness factors, distance-cap floors `D_*` — one **PROPAGATION MODEL PARAMETERS** block (since 2026-11-10, Ron; `TAU_G_ABS_FLOOR` was function-local before). Keep new BFS parameters there, never as literals in the functions. | `release_geometry.py` top |
| `MODE2/3_SPEED_RATIO`, `SCH2004_EXP`, `C_FIT`, material relations | `arrest_indices.py` |
| `MIN_TAU_G`, `MIN_SLOPE`, `MAX_SK38`, slab-thickness bounds | `analysis_pipeline.py` (~line 505, inside the trigger block) |
| `RASTER_EPSG` | `config.py` |

---

## Feature flags — check which are active before reasoning about output

Several physics paths are implemented but **off**. Reading the code without
checking these leads to wrong conclusions about what produced a polygon.

| Flag | Default | Effect when off |
|---|---|---|
| `USE_MELOCHE_ARREST` | `True` (since 2026-10-23, both repos; standalone `config.USE_MELOCHE_ARREST`, 31e5192) | `_qualifies` stops at the directional Meloche A_ca test (Eq. 20 via `arrest_indices.arrest_length`, with C = 0.045); the Λ/thickness discontinuity heuristics are then **skipped**, not run alongside. `TAU_G_ABS_FLOOR`, thin slab, low Λ and the distance caps still apply. It fires rarely (6 of ~1200 clusters for trigger 1772) and areas grow ~5-20% against the heuristics. |
| `USE_SLAB_TENSILE_GATE` | `True` (since 2026-10-27, Ron, both repos; CLI `--no-slab-tensile-gate`) | No slab-tensile check. When on, a candidate is arrested when S_t = σ_t / (3ρg·r_c²/h) < 1 (Schweizer et al. 2014 ISSW Eq. 1, the closed form, not Reuter & Schweizer 2018's FE index). σ_t is `sigma_t_mean__jamieson1990` (J&J 1990 with the facet knockdown); a missing column means the gate never fires; `step_scenarios` and the standalone's `generate_scenarios` print a WARNING when that happens. It also gates the trigger. On the ten no-event days it rejects no trigger, so it does not stop false alarms. |
| `USE_TRIGGER_TOUGHNESS` | `True` (since 2026-11-10, Ron, both repos; standalone `config.`; CLI `--no-trigger-toughness`) | Plain S_t at the trigger. When on, **at the trigger only** (`cluster_tensile_support`: trigger gate, candidate filter, reporting) σ_t is capped by K_Ic / (1.122 √(π h/2)), K_Ic the `K_Ic_ligament` column. The BFS per-cluster tensile gate is unchanged. Jan 18 ski non-triggers 13/46 → 0/46 releasing, Jan 18 triggers and IoU unchanged; no-event false alarms unchanged. F = 1.122 is the semi-infinite edge-crack factor; the finite-thickness one (~2.8 at a/h = 0.5) would be stricter and was not evaluated. |
| `arrest_indices.K_IC_RANGE_MODE` | `'extrapolate'` (since 2026-11-10, Ron, both repos) | `'clamp'` evaluated out-of-range densities at the range edge (every CSV before 11-10; Jan 18 median ligament K_Ic 1318 vs 1650 Pa m^½). The density ranges are a guideline. `n_rho_clamped` now counts extrapolated layers. |
| `USE_DISTANCE_CAPS` | `True` (both repos; avachain per slope via `[calibration] distance_caps`; CLI `--[no-]distance-caps` overrides) | Off sets d_up = d_down = d_lat = ∞; the BFS is bounded by the per-cluster gates, the start zone and `MAX_BFS_CLUSTERS`. Off, the size factors scale nothing in the BFS. |
| `snowpack_analysis.FACET_SOURCE` | `'grain_class'` (since 2026-10-27, both repos; standalone `config.FACET_SOURCE`) | `'sphericity'` = f = 1 − sphericity, every CSV before 10-27. Grain class sets f = 1 for J&J's group II: primary FC/DH/FCxr, or PP/DF/RG with a faceted secondary (e.g. 341). `'grain_class_primary'` drops the secondary rule. It moves `sigma_t_mean/_wl/_ligament`, their `A_ca_*`, the energy mode and the tensile gate. It does **not** touch `A_ca_brittle` or the legacy Eq. 20, which read the plain `sigma_t` with no knockdown. Jan 18 start-zone median f goes from 0.45 to 0.97. |
| `USE_MODE3_SPEED_CAP` | `False` | `d_lat` is the Gaume θ-ratio width (15 m floor) alone. |
| `USE_MODE3_LAMBDA` | `False` | A single Λ is used in all directions. |
| `USE_ANISOTROPIC_RINGS` | `False` | BFS rings grow isotropically. |
| `USE_LIGAMENT_BOUND` | `True` | — |
| `THETA_ESTIMATOR` | `'plane_fit'` | `'knn'` reproduces every pre-2026-10-09 CSV. |
| `E_RELATION` | `'vanherwijnen2016'` | — |
| `release_geometry.ARREST_MODE` | `'legacy'` | `'energy'` (path-energy hypothesis, `energy_path.py`), `'slope_only'` / `'stability_connected'` (baselines, `baselines.py`). Non-legacy modes have no rectangle fallback. CLI `--arrest-mode`. |
| `release_geometry.MASK_MODE` | `'gate'` | `'diagnostic'` runs the BFS without the start-zone KML; needs features outside it (`analyze --feature-buffer-m`). |
| `energy_path.ENERGY_DELTA` | `0.0` | Energy-mode δ only; R was fitted at δ = 0. The Eq. 20 δ stays 1.0. |

**The Jan 18 target is the survey-derived polygon** (Ron, 2026-10-26; since
2026-11-08 it is a calibration target, not validation: see *Calibration*): `data/validation/avalanche_release_area_20260118.geojson`,
5 810 m², from `survey_release_polygon.py` (`avalanche.delineate_crown` on the
01-14 → 01-20 surveys). `[paths] validation_dir` makes
`cfg.release_geojsons_for_date` prefer it, which covers every scoring reader
(comparison plot, groups, `--use-observed-release`, `analyze_release_zone`,
energy experiments). It is off by default, so the blind Jan replay cannot see
it. IoU from before 2026-10-26 is against a mapped crown and is not comparable.

The mapped crowns stay in `data/boundaries/`: the **6 934 m²** one since
2026-10-25 (the standalone's) and the 4 550 m² one before it in
`data/boundaries/old/…_4550m2.geojson`. Each has IoU 0.65 against the survey
polygon. The 6 934 m² file is the reinit scour mask only on the
`reinit --all-events` path (`cfg.event_mask_for`, which never reads
`validation_dir`). `run_full_pipeline.sh --reinit` passes no GeoJSON, so the
reinit auto-detects the release from the 01-14 → 01-20 surveys (3 550 m², 420
clusters in the 2026-10-23 rebuild). The mapped crown gives 399 clusters. Conclusions drawn
against the mapped crown, such as "slab and τ_p rise outward at the crown", do
not hold against the survey polygon. Redo them before acting on them.

The README's parameter table was corrected on 2026-10-28. E is now
`vanherwijnen2016`, and σ_t has separate rows for the plain relation and the
J&J 1990 one. If they disagree again, trust the flags in this file.

---

## The release-area model must match `~/avalanche-release-areas`

That repo is the **standalone reference implementation**; avachain embeds the
same model in the operational pipeline. Any behavioural difference in the
physics is a porting defect, not a local choice. Module mapping:

| avachain | standalone |
|---|---|
| `arrest_indices.py` | `arrest_indices.py` |
| `release_geometry.py` | `release_geometry.py` |
| `snowpack_analysis.py` | `snowpack_features.py` |
| `analysis_pipeline.py` | `generate_scenarios.py` |
| `scenario_writer.py` | `scenario_writer.py` |

Four behaviours where **avachain is deliberately better** — fix these in the
standalone, never revert them here:

1. Release depth from slab thickness above the WL, not `hs`.
2. Depth raster clipped to the polygon before computing volume.
3. `depth_percentile` actually scales depth.
4. Release density per scenario: the median `slab_density` of the clusters
   inside that scenario's own release polygon, 280 kg/m³ when they have none
   (`analysis_pipeline._polygon_density`, Ron 2026-11-04). The standalone
   takes the trigger cluster's density (fallback 250). avachain's old source,
   the observed release group, was NaN on every day without an observed
   avalanche, and runout passes density straight to com1DFA.

When comparing constants, **grep case-insensitively and compare by value**. A
case-sensitive search for `tau_g.*50` silently misses
`TAU_G_FEATURE_FLOOR = 50.0`, which has already caused one false "the repos
agree" conclusion.

---

## Two tau_g thresholds that look like a bug and are not

```
snowpack_analysis.TAU_G_FEATURE_FLOOR = 50.0   # gates computing arrest indices
analysis_pipeline.MIN_TAU_G           = 40.0   # gates trigger selection
```

The feature floor sits **above** the trigger gate on purpose, matching the
standalone. The resulting 40–50 Pa band passes trigger selection with no
arrest indices — that is a known open item, not an oversight to tidy. Both
values are pinned by `tests/test_snowpack_analysis.py`, because **zero of the
1608 Jan 18 clusters fall in the band**, so the acceptance test cannot catch a
regression here.

Generalise that: **a parity item Jan 18 cannot see needs a unit test.**

---

## Direction must be derived in the map frame, never from array axes

Two bugs of exactly this kind were live in the production path until
2026-10-10. In a north-up raster **row increases southward**, so:

- `np.sign(cos/sin(aspect))` quantises to the four diagonals — it was a pure
  diagonal on 100% of Jan 18 start-zone pixels, median +13.9° error.
- Picking "abeam" neighbours by row/col offset assumes a north–south fall
  line; selected neighbours sat a mean 30° off the true cross axis.

Any new code deriving a direction must be checked against the map frame.
`tests/test_terrain_direction.py` covers both helpers with planar-DEM fixtures
at known bearings; neither had coverage before, which is how both survived.

---

## CRS — one constant, verified against the files

`config.RASTER_EPSG = 6342` (NAD83(2011) / UTM 13N). Last season's rasters are
**compound** CRS with a NAVD88 vertical, so `rasterio` returns `crs.to_epsg() is
None` — the horizontal component is 6342. Do not read that `None` as "no CRS".

**That NAVD88 label is false.** Every DEM in the series (251110, 260605, 261018)
holds NAD83(2011) **ellipsoidal** heights with no geoid applied. A 2026-11-07
check against USGS 3DEP on US-6 put 261018 at −12.24 m and 260605 at −12.32 m
from NAVD88, where GEOID18 N = −12.42 m. Ron (option 1): keep the series
ellipsoidal and fix the tags. From 261018 on, files are 6342 horizontal-only,
with a `VERTICAL_REFERENCE` tag and `-ellipsoid` in the name. `dem_check`
warns on a NAVD88 tag and on a > 5 m median offset against a compared DEM,
which is a mixed geoid/ellipsoid pair. Snow height is DSM − DEM, so a survey in
the other frame reads ~12 m of snow; the producer's rules are in
`docs/deployment.md` §2.1. Do not "convert to NAVD88" anything on its own: the
surveys would have to move with it. `survey_check.py` (Step 0 of the survey run)
FAILs a survey whose shallowest 10 % inside the boundary reads > 6 m from zero,
and WARNs above 1.5 m. Last season's 23 surveys read −0.57 to +0.69 m.

EPSG:32613 (WGS84/UTM 13N) must never appear as a reprojection target;
`tests/test_raster_crs.py` fails if it does. pyproj treats 6342→32613 as a
sub-millimetre null transform here, so a mismatch is latent rather than
visibly broken — which is why the old one survived for months.

The `depth.prj` WKT in `scenario_writer.py` is a deliberate hand-written
literal (the bytes go to AvaFrame); a test pins its AUTHORITY code instead.

---

## The Zarr store, and the dev fixture

`slope_snowpack.zarr` **was** 78 GB holding 59280 `location` entries for 6565
distinct clusters — the full cluster set appended ~9 times, each copy
byte-identical, at index strides of exactly 6565. It was deduplicated and
rechunked on 2026-10-13 (`dedup_zarr.py`) to **6565 locations, 14 GB,
chunks `(location=256, time=1, layer=338)`**. The **swap ran on 2026-10-19**
(`dedup_zarr.py --skip-write --swap`, after it verified all 30 variables over
32 clusters × 8 timesteps bit-identical), so the deduped store is now the live
one at `cfg.zarr_path`. The duplicated original is kept alongside as
`slope_snowpack.zarr.duplicated` — delete it once you are satisfied; it is
78 GB. Before 10-19 the live store was still the duplicated one and the
deduped candidate sat beside it as `.dedup`, so a note claiming the swap had
happened predated it actually happening.

Measured effect on the full domain, Jan 18: single-snapshot read 173 s → 1.3 s,
`analyze` 25m 08s → ~17 s warm. The feature CSVs are bit-identical before and
after. Two things that surprised us:

- `time=1` chunks **compress worse** (less temporal redundancy inside a chunk),
  so 78 GB / 9 predicted 8.7 GB and the result was 12.5 GB.
- The **first** read of a freshly written store is slow — 7m16s cold against
  ~17 s warm. Don't read a single cold run as a failed optimisation.

The duplication cause is fixed in `snowpack_io._resume_state`: append mode used
to come from `zarr_out.exists()` while the skip-list came from a separate
`xr.open_zarr` in a try/except, so an unreadable store printed "starting fresh"
and then appended a whole second copy. It now refuses and asks for `--rebuild`.
The skip is also per file rather than per batch, so an interrupted batch no
longer re-appends the locations that did land. Pinned by
`tests/test_zarr_cache_resume.py`.

Two things to keep true of any store built from here on:

- Selecting by `int(name.split('_')[-1]) in ids` (the idiom throughout
  `analyze_release_zone.py`) matches **every** copy of a duplicated cluster. On
  the old store the 1608-cluster Jan 18 start zone pulled 14502 locations and
  `analyze` logged "Extracted features for 14502 additional clusters". Output
  stayed correct — the CSV keys by cluster id — only the cost was wrong, so
  this failure mode is invisible except in the log line and the clock.
- `time=1` is right because **every** consumer reads one timestep
  (`isel(time=tidx)` / `sel(time=ts, method='nearest')` in
  `analysis_pipeline`, `analyze_release_zone`, `release_geometry`,
  `snowpack_analysis`); no 3D read spans a time slab, and the HS history behind
  dHS/dt comes from the SMET side, not from here. If that changes, a full
  time-series read of HS is the thing that got worse: 0.1 s → 3.2 s for 1608
  locations.

`make_dev_zarr.py` cuts a smaller fixture — dedup, **dilated** start zone, time
subset, `time=1` chunks:

```bash
python src/avachain/make_dev_zarr.py --toml slopes/little_prof/slope_config.toml \
       --dates 2026-01-18 --pad-days 2 --buffer-m 200
# → snowpack/little_prof/output/dev_snowpack.zarr, 427 MB
python src/avachain/analysis_pipeline.py analyze --toml ... \
       --zarr-path snowpack/little_prof/output/dev_snowpack.zarr --force
```

Be aware it is **mostly redundant now**: it was written when `analyze` took 25
minutes, and the deduped main store does the full domain in ~17 s against the
fixture's ~13 s. What it is still worth using for is a 427 MB store instead of
12.5 GB — a copyable fixture, a cold cache, or a disk-light checkout. Reach for
it for those reasons, not for speed.

Two things to keep in mind when you do:

- The buffer is not optional. The release BFS grows outward from its trigger
  and `_build_scour_depth_map` wants every cluster in the domain, with a
  full-depth fallback for path/deposition clusters. Cut to the start zone
  exactly and those reads miss — and a missing location fails as if it were
  physics. At 200 m the buffer selects 6229 of 6565 clusters, so the speedup is
  dedup + rechunk + time subset, not a smaller cluster count.
- **The full domain stays the acceptance gate.** A subset cannot see
  BFS-escape or anything else domain-scale — same reason the 40–50 Pa `tau_g`
  band needs a unit test rather than Jan 18.

Layer trimming is measured, not assumed: the median Jan 18 profile uses 102 of
338 layers, but across the dilated domain some profile uses all 338, so the
trim currently saves nothing. Don't hardcode 258.

**Units inside the store are metres** for `HS` and element `height` (xsnow
attrs say so), unlike the SMET files below. `_scour_depth_for_cluster` treated
them as cm until 52fe333. With a slab, every scour depth came out 0, and
`scour_depth.tif` (com1DFA's entrainable depth) carried that.

## SMET units — write values in the file's own units, never SI

The SMET header carries `units_offset = ... 273.15 ...` for `TA` and
`units_multiplier = ... 0.01 ...` for `RH` and `HS`, and **SNOWPACK applies
them on read**. So the numbers in the file are °C, percent and **centimetres**
— not Kelvin, fraction and metres. `smet_writer.write_smet` is the only
producer of these files and writes them that way; `nwp_ingest._format_nwp_row`
and `smet_append._format_smet_row` both append into the same files and must
match it.

`smet_append` converted as well until 2026-10-19, so a −10.7 °C row landed as
262.4 in the file and 535.6 K inside SNOWPACK, and HS came out 100× small.
Nothing downstream catches this: SNOWPACK runs happily on 535 K air and the
symptom is nonsense stratigraphy, not an error. A test had even pinned the
divergence (`test_nwp_ingest_convention_differs`) — one file has one units
header, so two writers cannot hold two conventions. Pinned now by
`tests/test_smet_append_units.py`.

The weather cache (`outputs/weather/<station>.parquet`) therefore stores
**HS in cm** too, so the ΔHS propagation in `smet_append` is dimensionally
consistent. It is built by `smet_writer.load_and_convert`; never hand-roll the
CAIC conversions (the raw CSV is tenths of °F, tenths of mph, tenths of W/m²).

## Replay clock — `$AVACHAIN_AS_OF`

`config.replay_as_of()` resolves a simulated UTC "now" from an explicit
argument, else `$AVACHAIN_AS_OF`, else `None` (live behaviour). It is an env
var rather than a flag because `run_daily_<slope>.sh` / `run_<slope>.sh` spawn a
dozen Python subprocesses; exporting once reaches all of them.

Two seams consume it, and they are the only two places the pipeline can see
the future:

- `pipeline_io.discover_surveys(as_of=)` — gate here, not by staging TIFs:
  `step_transport`, `step_avalanche` and `fetch_avalanche_obs` each re-glob
  independently, so one cutoff covers all three.
- `nwp_ingest.get_stable_ts(now=)`.

Two traps when replaying. SMETs must be **truncated to the replay start**, or
`smet_append` short-circuits on `last_ts >= until` and every append is a silent
no-op. And the existing `.sno` restart files are **temporally ragged** (1789
clusters at 2026-01-31, 871 at 02-07, only 505 at season end; the dates match
the 18 `reinit_stats_*.json`), because chained reinit with partial pass-2
reruns strands each scoured group at its own event date — so a coherent start
state needs a seed run, and post-reinit restart coherence is worth asserting.

There is no per-cycle WRF archive for this season, only one rolling file, so
rows after a historical `T_stable` were written later than that date. The NWP
path can be exercised for plumbing but **cannot support a forecast-skill
claim**. From 2026-27
on, `src/avachain/wrf_archive.py` (hourly cron) keeps every cycle of the
slope's grid cell under `work/weather/wrf_archive/<season>/<zone>/<cell>/
<cycle UTC>/`. Read a historical forecast from there, never from
`/ssd/snowpack/fcst/`: the CAIC run **deletes and rebuilds that tree every
morning (~07:40–07:55 local)**, so the live file is always today's cycle. In
that window it is missing, and then present **without its forecast rows**. A
reader there would build a short forecast without any error. Every reader
goes through `nwp_ingest.wait_for_wrf`; a new reader must too. In tests,
`tests/conftest.py` sets the wait to zero.

## Avalanche events and the reinit — Jan 18 is the only avalanche

Ron, 2026-10-27: the only avalanche of the 2025-26 season on Little Professor
is 2026-01-18. Until 6e8f744 the `avalanche` step made all 22 survey periods
avalanches:
- It gave every period listed in `data/boundaries/avalanche_events.json` the
  label `avalanche`, ignoring the label there.
- It appends auto-detected periods to that file, so after one run every period
  was listed.
- `gap_fill` applied `transport_corrected_*` / `avalanche_dhs_*` wherever the
  file existed.

So the forcing before the 10-27 rebuild carried 21 false avalanche corrections,
7 of them before Jan 18. Now:
- A listed period keeps its label: `forcing_pipeline.classify_period`.
- `gap_fill` follows the events file's `transport_corrected` flag:
  `avalanche_correction_path`.
- Every period except 2026-01-14__2026-01-20 is `wind_scour` or
  `no_avalanche`, with no reinit.

Do not relabel from auto-detection or CAIC hits. Ron's word is the ground
truth.

Three traps in the two-pass rebuild (`run_full_pipeline.sh --reinit`), all
fixed on 10-27:
1. **Pass 2 ran only the scoured clusters** (1b9a384), so 6 145 of 6 565
   `.pro` files ended at the event date.
2. **The Zarr store ended at the event date even after that fix** (ecc823d).
   `run_snowpack.sh`'s build resumes and skips locations already in the
   store, and pass 1 had written all of them. Both passes now run with
   `SKIP_ZARR=1`, and the pipeline then builds once with `--rebuild`.
3. **The reinit stamped `ProfileDate` 12:00 while pass 1 ran to 18:00**
   (3902380). Pass 2 then put scoured clusters back 6 h and rewrote their
   event-day 12:00 profile post-scour. `PASS1_END_TIME` now feeds both, and
   `run_reinit` refuses any `.sno` not at the event time
   (`profile_date_mismatches`).

`analyze` reads `Timestamp(date)`, which is 00:00, so an event-day analysis is
pre-scour. A 12:00 or 18:00 snapshot on the event day is post-scour on the
scoured clusters.

## Season start, the forecast pass and thin days

`docs/season_start.md` is the procedure. The traps:

- **The forecast pass has its own forcing and config.** WRF has no HS, and
  the observed config enforces measured HS, so every forecast stopped with
  `[E] missing { HS }` until 1cc40e2 (while SNOWPACK printed FINISHED).
  `nwp_ingest` writes `input/smet_forecast/` (PSUM, HS nodata), and the
  forecast pass runs `master_config_forecast.ini`, which turns off
  `ENFORCE_MEASURED_SNOW_HEIGHTS` in **both** `[SNOWPACK]` and
  `[SNOWPACKADVANCED]`. Overriding only one leaves HS forcing on. Observed
  SMETs never carry forecast rows. Do not reintroduce append-into-observed.
- **An empty scenario set is a valid result.** On a no-candidate day the
  triggers are the shot locations, `[scenarios] explosive_trigger_points`.
  These are **[lat, lon] points, never cluster IDs**: IDs change with every
  cluster map, the shot locations don't. Any release, model or shot,
  holding < 0.10 m on average is dropped (`MIN_RELEASE_DEPTH_M`). If none is
  left, the set publishes as `status: "no_scenarios"` and the run ends DONE. On such a day `meloche_df` has **no arrest-index columns at all**
  (nothing reached `TAU_G_FEATURE_FLOOR`), so guard column access, not just
  row access.
- **Forced shots are a separate set, not a fallback.** With
  `[scenarios] force_explosive_triggers` (or `--force-explosive-triggers`)
  every run also publishes `<date>/shots[/T_n]`: no initiation gate, and
  `make_release_polygon_2d(rectangle_fallback=False)`, so a shot that does not
  propagate gives nothing (Ron, 2026-11-06: "they will trigger, the question
  is will it propagate"). The no-candidate fallback above still puts shots
  into the model set, with rectangles. `shots/` is not a `T_n` folder, so
  runout's `discover_horizons` never takes it for a horizon, and
  `handoff.publish` of `<date>/` carries it across. `runout_step` runs it as a
  **second runout** (`shot_settings`): `<work_dir>/shots/`, `<publish_root>/
  shots/`, and a generated relabelled path profile. Keeping it a separate run
  avoids teaching the runout's manifest, report and viewer, which all
  parse `T_<int>`, a new label. This is avachain-only: it is not in
  `~/avalanche-release-areas`, and does not belong there.
- **Shot weights are absolute.** Each shot holds 1/N, and whatever does not
  release is `p_no_release` (Ron, 2026-11-06), so a shot set's weights sum
  to P(release) ≤ 1. Its `params.json` carries `ensemble_weight_total = 1.0`.
  The runout's weighted fractions (`stack_reach_masks`, road and reference
  point summaries, the report curves, impact P, `rethreshold_reach`) divide by
  that, never by `sum(w)`. A new weighted average in the runout must do the
  same, or it silently turns "per shot fired" back into "given release".
  Model sets carry no total and are unchanged. Professor_PTC has it on.
- **Release density is per polygon.** It comes from the clusters inside the
  release polygon (see the standalone section). Its old source, the observed
  release group, is empty on every live day.
- **A survey run killed by a signal must end FAILED.** The scripts trap
  HUP/INT/TERM to exit 128+n before the EXIT trap writes the marker. In tests,
  remember that bash cannot trap a signal that was ignored when it started:
  a pytest run as a background job (`… &`) ignores SIGINT, so start such
  children with default handlers.

## Runout in the chain

- `runout/` is its own uv project with its own venv (`runout/.venv`, built by
  `install.py` when the TOML has `[runout]`). AvaFrame 2.1 needs numpy < 2 and
  xsnow needs numpy >= 2.3, so the two can never share a venv. Don't import
  runout code from avachain or the other way round: `runout_step.py` calls it
  as a subprocess. The runout package is named `scripts`, and its `config.py`
  clashes with avachain's.
- `runout_step.py` reads a release set only after `handoff.verify`: `READY`
  present, and every file matching the manifest's sha256.
- Every runout on the host takes the one `[runout] lock_file`. The 4-sim
  cap is per run; the lock makes it hold across slopes. More than 4 sims at
  once ran the 256 GB host out of memory on 2026-11-02.
- The deployment's runout path profile is `runout/config/paths/Professor_PTC.yaml`,
  with **absolute** `/home/avachain/data/static` paths, because runout's
  `avachain_root` is its per-path work dir. Keep its `display:` in step with
  `little_professor.yaml`.
- Slopes without `[runout]` (all the dev slopes) skip the step.

## Forecast days, snapshot time, cron time zone

- **cron reads `/etc/cron.d` in the host's local time** (America/Denver), not
  UTC. The generated file declares `CRON_TZ=` from `[cron] timezone` (default
  UTC). A "13 UTC" job written without it fired at 13:00 MST.
- **One snapshot time per run.** `[scenarios] snapshot_hour_utc` (or
  `--snapshot-ts`) sets the time features, release depth and scour are all
  read at. Without it the historical per-read hours stay (features 00:00,
  depth 12:00, scour 18:00), which the Jan 18 validation path relies on. Do
  not "unify" those without the setting.
- **Forecast-day feature CSVs live in `analysis/forecast/<date>/T_n/`.** CSV
  names carry only the date. If a forecast day's CSVs sat in the analysis dir,
  the observed run of that date could reuse them (analyze skips when its CSVs
  are newer than the store).
- **HS in the store is metres everywhere.** `depth_from_snowpack`'s 0.8 × HS
  fallback divided by 100 until 2026-11-04 (the second time this unit
  mistake was made, after the scour depth). Every fallback release was 100×
  too thin.

## Alerts

`alerts.py` mails the recipients in `~/.config/avachain/alerts.toml` (never
in the TOML: the repo is public) on a FAILED survey, a runout failure, a
failed daily run, and no report by 03:15 local (cron, `check-report`). A new failure path in the run scripts should call `$ALERT`,
and never let an alert change the exit status. In tests, `$AVACHAIN_SENDMAIL`
points at a fake.

## Output layout

`output_dir` is slope-namespaced: `outputs/<slope_name>/`. For Little
Professor that means `outputs/little_prof/{analysis,plots,scenarios,...}/`.

Shell scripts that hardcode `outputs/analysis/...` are wrong and have bitten
us: the `reinit_stats` path bug silently made pass 2 of the reinit run rerun
every cluster. Derive paths from `ProjectConfig` properties
(`cfg.analysis_dir`, `cfg.scenarios_dir`, `cfg.pro_dir`, `cfg.zarr_path`) and
pass `--toml` through to subprocesses — a missing `--toml` makes a subprocess
silently fall back to the default `ProjectConfig` and look in the wrong place.

SMET files live under the slope runtime dir (`cfg.slope_dir/input/smet`), not
`outputs/`.

---

## Stale references in the docs

`docs/` predates the repo rename from `snowpack_model_feeder` to `avachain`,
and ~16 references still use the old name:

- **One stale module path** — `docs/TODO.md:20` gives
  `src/snowpack-model-feeder/analysis_pipeline.py`. The real location is
  `src/avachain/analysis_pipeline.py`. A grep against the documented path
  returns nothing; don't conclude the code is missing.
- The rest are old-repo **GitHub URLs** (`docs/project_description.md`) and
  absolute **cron paths** like `/home/ron/snowpack_model_feeder/run_daily.sh`
  (`docs/operational_mode_design_concept.md`, `docs/new_slope_setup.md`).

`docs/TODO.md` is the backlog of record. `CSV_FEATURE_PIPELINE_ISSUES.md` is a
point-in-time audit (2026-10-06) with a status table at the top — its file:line
references have all moved.

**Existing `meloche_features_*.csv` are stale** (θ changed with the estimator
port, so Π₁, the trigger ranking and the whole filter chain changed).
Regenerate before trusting any release polygon.

---

## Git

- `dev` is the active branch; `main` is stable operational.
- When something is deployment-ready, merge **and push both** `dev` and `main`.
- `science` is dormant — last touched 2026-09-02, 63 commits behind `dev` with
  8 of its own. Don't branch from it or assume it reflects current behaviour.
- Commit messages here explain the *why*, at length, including what was ruled
  out and what measurement justified the change. Match that; follow `git log`.
- Operational deployment lives at `/home/snowpath/avachain/` via `install.py`.

## Verification

Do not report something as working without having run it. For a bug fix,
reproduce the failure first — with synthetic data if needed — then show the
fix changes that specific behaviour. A test that passes before and after the
change proves nothing; check that it fails when the fix is reverted.

## Calibration — deliberate, versioned, and never called validation

Ron, 2026-11-08: this rule **replaced** "never fit or tune to Jan 18". The
chain is an operational tool for competent forecasters who know its limits,
and they are better served by a model anchored to the one event we have. So
the release-area model **is** calibrated to Jan 18, and recalibrated as new
avalanches occur, under these rules:

- **Jan 18 is a calibration target and a regression test, not evidence of
  skill.** Never report Jan 18 IoU as validation once a value was fitted to it.
  The report carries a caveat line for the release model, like the one for the
  runout's τ₀ = 125 Pa (itself already fitted to Jan 18).
- **A no-event day says nothing about release area** (Ron, 2026-11-09; this
  replaced "the no-event days are part of every fit", which scored the
  no-event E[area] against its baseline). A day passes without an avalanche
  for reasons the area model does not see: no trigger in the right place,
  a slope hard to trigger across, or an r_c too large. Its potential release
  can be larger or smaller than Jan 18's. So **do not constrain the area on
  no-event days**. Report it, never score it. Those days still bear on
  *initiation* (the trigger gates), and the model false-alarms on 10 of 10
  of them, but area knobs leave the trigger list unchanged.
- **Observed non-triggering constrains propagation.** These are points where a
  trigger was applied and nothing propagated: shots that did not release, and
  skiers. On Jan 18, people skied 10–20 m from the crown without triggering
  it. Score them by putting model triggers there with no rectangle fallback
  (the forced-shot path, `make_release_polygon_2d(rectangle_fallback=False)`).
  A fit that gets Jan 18 IoU by growing the polygon over skied terrain is
  wrong.
- **One free parameter, two at most.** One polygon constrains roughly size
  and maybe position; more knobs fit noise.
- **Pick a knob that changes area smoothly and has a physical reading** —
  e.g. arrest strictness (`C_FIT` in Eq. 20) or a size scaling on A_ca. The Λ
  factors are **dead** while `USE_MELOCHE_ARREST` is on (`_qualifies` returns
  after the Meloche test), and `C_FIT` is a default argument, so setting
  `arrest_indices.C_FIT` at runtime changes nothing: it must be passed as `C=`
  (analyze's `A_ca_brittle` and the BFS call at `release_geometry.py` ~1188). **Not** the Stauchwall angle or the slab-thickness cap: they move
  polygons in jumps (7 382 / 4 580 / 827 m² at 28 / 29 / 30°), so a fit lands
  on a knife edge. Choose a robust plateau, not the single best point.
- **Start with a sensitivity sweep** on Jan 18: IoU and area against each
  candidate knob, plus the non-triggering points above. Report the no-event
  areas, but don't score them. Pick the knob with the smoothest response.
- **Calibrated values live in the slope TOML `[calibration]`**, with
  provenance: events used, date, objective, commit. The physics stays
  identical to `~/avalanche-release-areas`; only per-slope parameters differ,
  so parity holds. Never edit module-top physics constants to calibrate.
- **Recalibration proposes, it does not apply.** A new event triggers a run
  that refits on all events plus every observed non-triggering point
  (shots, skiers), reports leave-one-out
  results (each event scored by a fit that did not see it), and changes
  nothing until Ron approves.
- **Every report and every metrics run record states the calibration
  version**, so the season's records say which version produced each forecast.
- **Shot outcomes are the main input** (`observations.jsonl`, metrics.py).
  Without them recalibration has one data point per avalanche. Record skier
  and ski-cut non-triggers the same way, with locations.

Fitting material parameters to the *start-zone snowpack distribution* (e.g.
G_wl so the start-zone median Π = 1) is a different thing and stays allowed:
its target is the snowpack, not the crown.
