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
| `TAU_G_ABS_FLOOR` — note it is **function-local**, not module top | `release_geometry.py:804` |
| `MODE2/3_SPEED_RATIO`, `SCH2004_EXP`, `C_FIT`, material relations | `arrest_indices.py` |
| `MIN_TAU_G`, `MIN_SLOPE`, `MAX_SK38`, slab-thickness bounds | `analysis_pipeline.py` (~line 505, inside the trigger block) |
| `RASTER_EPSG` | `config.py` |

---

## Feature flags — check which are active before reasoning about output

Several physics paths are implemented but **off**. Reading the code without
checking these leads to wrong conclusions about what produced a polygon.

| Flag | Default | Effect when off |
|---|---|---|
| `USE_MELOCHE_ARREST` | `True` (since 2026-10-23, **avachain only** — the standalone is still `False`, so this is a parity difference until Ron decides) | `_qualifies` stops at the directional Meloche A_ca test (Eq. 20 via `arrest_indices.arrest_length`, with C = 0.045); the Λ/thickness discontinuity heuristics are then **skipped**, not run alongside. `TAU_G_ABS_FLOOR`, thin slab, low Λ and the distance caps still apply. It fires rarely (6 of ~1200 clusters for trigger 1772) and areas grow ~5-20% against the heuristics. |
| `USE_MODE3_SPEED_CAP` | `False` | `d_lat` is the Gaume θ-ratio width (15 m floor) alone. |
| `USE_MODE3_LAMBDA` | `False` | A single Λ is used in all directions. |
| `USE_ANISOTROPIC_RINGS` | `False` | BFS rings grow isotropically. |
| `USE_LIGAMENT_BOUND` | `True` | — |
| `THETA_ESTIMATOR` | `'plane_fit'` | `'knn'` reproduces every pre-2026-10-09 CSV. |
| `E_RELATION` | `'vanherwijnen2016'` | — |

**The README's parameter table is stale on two rows.** It gives
`E = (ρ/300)^2.5 × 4 MPa` and `σ_t = (ρ/300)^1.4 × 5 kPa`, which describe
`E_RELATION = 'project_fit'`. The active default is `'vanherwijnen2016'`
(`E = 0.93·ρ^2.8` Pa, van Herwijnen et al. 2016 Eq. 8). Trust the flag.

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

Three behaviours where **avachain is deliberately better** — fix these in the
standalone, never revert them here:

1. Release depth from slab thickness above the WL, not `hs`.
2. Depth raster clipped to the polygon before computing volume.
3. `depth_percentile` actually scales depth.

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

`config.RASTER_EPSG = 6342` (NAD83(2011) / UTM 13N). All rasters on disk are
**compound** CRS with NAVD88 vertical, so `rasterio` returns `crs.to_epsg() is
None` — the horizontal component is 6342. Do not read that `None` as "no CRS".

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
claim**.

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

Never fit or tune parameters to the 18 January 2026 event. It is n=1 and it is
the validation target.
