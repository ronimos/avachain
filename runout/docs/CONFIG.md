# Config reference

Four files, three layers. Values resolve physics file, then profile
sections, then --set.

## config/local.yaml

Machine roots. Gitignored. Copy from local.example.yaml.

| key | required | meaning |
|---|---|---|
| `avachain_root` | yes | the avachain tree; `data/`, `intermediate/`, `outputs/` live under it |
| `scenario_root` | yes | where the snowpack chain writes `<slug>/<YYYYMMDD>/<T_n>/scenario_###/` |
| `publish_root` | no | web-served folder for the cycle products; omit to skip publishing |

## config/paths/<slug>.yaml

Static inputs for one path. Relative file paths resolve against
avachain_root.

| key | required | meaning |
|---|---|---|
| `slug` | yes | must equal the file name |
| `scenario_slug` | no | folder name under `scenario_root` when the snowpack chain names the path differently (default: `slug`) |
| `display.project_name` | yes | shown on the report and viewer |
| `display.subtitle` | no | shown after the name |
| `display.highway` | no | informational |
| `terrain.dem` | yes | DEM the simulations run on (bare ground). Its CRS is the reference every other input must match |
| `terrain.dem_priority` | no | high-resolution DEM mosaicked onto `dem` where it has data |
| `terrain.dem_feather_cells` | no | blend width in cells at the mosaic seam, default 5 |
| `terrain.dem_vertical_correction_m` | no | constant shift added to `dem_priority` before mosaicking |
| `canopy.forest` | yes | forest polygons used as the com1DFA resistance area |
| `infrastructure.road` | yes | road centerline |
| `infrastructure.reference_points` | no | point layer with a `ref_name` field; each scenario is tested against every point |
| `release.start_zone` | no | start zone polygon for `generate-scenarios` |
| `entrainment.polygon` | no | entrainment area for `entrainment.mode: polygon` |
| `road.buffer_m` | no | corridor half-width around the centerline for the reach test, default 1.0 |
| `road.chainage_origin` | no | `far_from_release`, `first_vertex` or `last_vertex` |
| `reference_points.radius_m` | no | a point is reached when a reach cell is within this distance, default 10 |

## config/physics/<name>.yaml

Model parameters. operational.yaml is the locked set. Every section
below may be overridden in a profile or with --set.

| section.key | meaning |
|---|---|
| `depth_convention` | `slope_normal`; recorded in the manifest, not acted on |
| `com1dfa.mesh_cell_size_m` | com1DFA mesh, match the DEM |
| `com1dfa.friction_model` | `samosAT`, `samosATSmall`, `samosATMedium`, `samosATAuto`, `Voellmy`, `VoellmyMinShear`, `Coulomb`, `CoulombMinShear`, `wetsnow` |
| `com1dfa.sim_type` | `'null'`, `'res'`, `'ent'`, `'entres'`; quote it, bare null becomes None |
| `com1dfa.stop_crit` | stop when kinetic energy falls below this fraction of peak |
| `com1dfa.t_end_s` | maximum simulated time |
| `com1dfa.result_types` | peak fields to keep: `ppr`, `pft`, `pfv`, `pta` |
| `friction.mu`, `friction.xi`, `friction.tau0` | values; null means the AvaFrame default. A model ignores keys it does not use |
| `friction.sources.<param>` | `fixed` (the value above) or `params` (each scenario's params.json) |
| `defaults.scour_depth_fallback_m` | scour cap when a scenario's `scour_depth_m` is 0 |
| `entrainment.mode` | `auto`, `raster`, `polygon`; normally set in the profile |
| `entrainment.min_slope_deg` | cells below this slope get no entrainment; 0 disables |
| `entrainment.ent_ero_energy_jkg` | com1DFA entrainment energy |
| `entrainment.auto_detect.alpha_beta.{min_alpha_deg,max_alpha_deg,buffer_m}` | corridor from the alpha-beta model; alpha outside the range falls back |
| `entrainment.auto_detect.flow_accumulation.{buffer_m,min_slope_deg}` | fallback corridor |
| `resistance.{cResH,detrainment,forestVMin,forestThMin,forestVMax,forestThMax}` | Feistl et al. (2014) forest resistance, used when `sim_type` includes `res` |
| `reach.flow_depth_threshold_m` | minimum peak flow depth for a cell to count as reached |
| `reach.min_velocity_ms` | also require this peak velocity; 0 disables |
| `reach.pressure_threshold_kpa` | also require this peak pressure; 0 disables |
| `probability.contour_levels` | levels written to contours.geojson |
| `probability.envelope_min_probability` | P that defines the envelope |
| `probability.weight_by_scenario_probability` | weight the stack by `scenario_probability` |

## config/profiles.yaml

Named jobs. Each profile has run keys and optional section overrides.

Run keys:

| key | meaning |
|---|---|
| `description` | free text, shown in the log and manifest |
| `extends` | inherit another profile, then override |
| `mode` | `forecast`, `sweep`, `calibrate`, `verify` |
| `physics` | physics file name without extension, default `operational` |
| `paths` | list of path slugs; `--path` on the command line narrows it |
| `horizons` | `all` or a list of labels; `--horizon` narrows it |
| `products` | any of `report`, `viewer`, `viewer_3d`. `report` also builds details.html and scenarios.xlsx |
| `sweep` | sweep settings, mode `sweep` only (table below) |
| `grid` | calibrate: `{section.key: [values], ...}`; the Cartesian product is run |
| `calibrate` | calibrate settings (table below) |
| `bracket` | verify: physics file names rerun beside the forecast's own physics |

sweep keys:

| key | meaning |
|---|---|
| `classes_file` | file under `config/physics/` holding the friction classes, default `friction_classes` |
| `classes` | `all` (every enabled row) or a list of class names |
| `depths` | uniform start zone depths to run (m, slope-normal) |
| `entrainment_depth_m` | constant entrainment depth inside the path's polygon |
| `entrainment_variants` | `[on, off]` runs each class both ways; `[on]` or `[off]` for one |
| `release` | path profile key under `release:` (default `start_zone`) or a polygon file path |

calibrate keys:

| key | meaning |
|---|---|
| `scenario` | `synthetic` (from the event's release polygon and slab) or an ensemble member id |
| `scenario_date` | cycle the member comes from; null means the event date |
| `horizon` | horizon folder the member sits in, default `T_0` |
| `thresholds` | reach thresholds (m) to score at |
| `judge_threshold` | the threshold the recommendation is made at |
| `tolerance.iou` | cells within this of the best IoU count as in range |
| `tolerance.toe_m` | cells within this of the best toe error count as in range |

Command-line for these modes: --event YYYYMMDD (or all for
calibrate), --scenario, --scenario-date, and --run-name, which both
modes require.

## Event records

data/static/<slug>/observed/<YYYYMMDD>/event.yaml with optional
debris_polygon, release_polygon, toe_point (file or [x, y]),
points (name: reached|not_reached), slab (hand_hardness,
density_kgm3, depth_m), source, notes. USER_GUIDE section 10 has
the full example.

## config/physics/friction_classes.yaml

A list under classes:. Each row: name, hand_hardness (SNOWPACK index,
-1 F to -5 K, or null), tau0 (Pa), density (kg/m3), optional
enabled: false. Sweep mode runs each enabled row as one friction
setting. The forecast side will read the same table once τ₀ is picked
per scenario by hardness.

Section overrides use the physics section names plus two run-side
sections:

| section.key | meaning |
|---|---|
| `entrainment.mode` | `auto`, `raster`, `polygon` |
| `execution.parallel` | run scenarios in a process pool |
| `execution.max_workers` | pool size; null uses every core |
| `execution.abort_on_crs_mismatch` | stop on any CRS mismatch |
| `output.cog` | write Cloud-Optimized GeoTIFFs |
| `output.probability_colormap` | matplotlib colormap for the heatmap |
| `output.report_plots_dpi` | report image resolution |
| `output.viewer_basemap` | `hillshade` |
| `output.viewer_scenario_layers` | include the per-scenario accordion in the viewer, default true |

Shipped profiles: forecast (raster entrainment, needs --snow-height),
forecast_nosurvey (auto), forecast_bethel (polygon, Bethel), sweep
(Bethel depth sweep), calibrate (τ₀ grid), calibrate_voellmy (μ, ξ
grid under a test physics file), and verify.

## Command-line overrides

--set section.key=value sets one value for one run. The section must be
one of the physics sections, execution, output, or run for run keys
(--set run.physics=name, --set run.horizons=T_0,T_1, --set run.sweep='{"classes":["soft"]}' as JSON). Keys are
checked against the physics file so a typo fails before anything runs.
Any --set requires --run-name.

--profile-file FILE adds or replaces named profiles from a yaml outside
the repo, or supplies a single profile with a name: key.

## What the manifest records

outputs/forecasts/<slug>/<YYYYMMDD>/manifest.json:

- pipeline_version, git_commit, avaframe_version
- run_name, mode, issued, cycle_date, path, display
- profile: name, description, source file, overrides, products,
  entrainment mode
- physics: name, file, hash, depth convention, and the resolved
  result-affecting sections in full
- path_profile: file, hash, resolved content
- local: the roots
- scenario_source: the dated scenario folder with size and modification time
- survey: file record and date
- horizons: per label, valid date, source, scenario count, outputs and
  work folders, status
- status: running, ok, partial, failed, dry-run; finished
