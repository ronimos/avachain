# Scenario contract

What the runout side reads from the snowpack side. If a file or key
below changes, both sides change.

## Folder layout

    <scenario_root>/<slug>/<YYYYMMDD>/<T_n>/scenario_###/inputs/

- scenario_root comes from config/local.yaml, normally
  /home/avachain/intermediate/scenarios.
- slug is the path folder name. By default it equals the runout side's
  slug (little_professor, bethel, widowmaker, seven_sisters); a
  different spelling is matched with scenario_slug in the path profile.
- YYYYMMDD is the forecast date (the T+0 valid date), no dashes.
- T_0, T_1, ... one folder per horizon. Any count. Only folders named
  T_<number> are read; anything else under the date is ignored.
- scenario_### three-digit, zero-padded. Any count per horizon; counts
  may differ between horizons.
- inputs/ holds the files. The runout side also accepts the files
  directly under scenario_###/ and the older <T_n>/scenarios/ nesting.

The runout side never writes into this tree. It copies each scenario's
inputs out before running. Once a date folder is complete it should not
change. Making it read-only (chmod -R a-w) after the last write prevents
accidental edits.

A depth sweep uses the same layout under sweep_<name>/<class>/ instead
of a date; those sets are written by the runout side itself.

## Files per scenario

| file | required | content |
|---|---|---|
| `release.geojson` | yes | one polygon feature, the release area, in the DEM's CRS (EPSG:6342). Holes are filled before com1DFA. |
| `depth.asc` + `depth.prj` | yes | release thickness in metres, slope-normal, on any grid in the DEM's CRS (resampled onto the DEM). 0 or nodata outside the release. `.tif` is accepted instead. |
| `density.json` | yes | `{"mean_kgm3": 315.0}`. Used as release and entrained snow density. |
| `params.json` | yes | keys below |
| `metadata.json` | no | anything; carried through, not read |

## params.json keys

Required (the run fails without them):

| key | meaning |
|---|---|
| `mu` | Voellmy μ. Read only under Voellmy-family friction models. |
| `xi` | Voellmy ξ. Same. |
| `scour_depth_m` | entrainable depth below the slab (m). 0 is allowed and means the pipeline falls back to `defaults.scour_depth_fallback_m`. |
| `scenario_probability` | the ensemble weight. Weights need not sum to 1. |

Used when present:

| key | meaning |
|---|---|
| `mean_depth_m` | mean release thickness; written as the shapefile `thickness` attribute AvaFrame requires |
| `rho_kgm3` | density fallback when `density.json` lacks `mean_kgm3` |
| `total_volume_m3` | reported in tables and sensitivity plots |
| `release_area_m2` | reported |
| `hand_hardness_value` | SNOWPACK hand hardness index at the hard layer (-1 F to -5 K); reported, and the key the friction class table will be joined on |
| `scour_depth_mid_m` | reported |
| `size_factor`, `depth_percentile` | the ensemble's varied inputs; reported and used as sensitivity axes |
| `trigger_cluster` | reported in the scenario table |

Any other key is carried into summary.json untouched.

## Conventions

- CRS: everything in the DEM's horizontal CRS. The runout side checks
  every input against the DEM and stops on a mismatch.
- Depths: slope-normal metres. The runout side applies no vertical to
  slope-normal conversion; if the upstream depth is vertical the
  conversion belongs before this handoff.
- Nodata in depth.asc: 0 or a negative sentinel; both are treated as no
  release.

## What comes back

Nothing is written to scenario_root. Outputs land under
<avachain_root>/outputs/forecasts/<slug>/<YYYYMMDD>/ and the web copy
under <publish_root>/forecasts/<slug>/. manifest.json there records
which scenario folder was read and its modification time.
