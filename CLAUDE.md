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
./run_daily.sh                                # between-survey daily forward
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
| `USE_MELOCHE_ARREST` | `False` | The Meloche Π₁ arrest test is **not** what stops the BFS. `TAU_G_ABS_FLOOR = 350 Pa` plus the Λ/thickness discontinuity heuristics are. |
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
