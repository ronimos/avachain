# Issues in the feature-CSV generation path

Audit date: 2026-10-06. Written for use with Claude Code in `~/avachain`.

## Why this file exists

`~/avalanche-release-areas` is a standalone extraction of the release-area
delineation code. Auditing it against its own methods doc turned up a set of
defects; this file records which of them are also present **here**, in the code
that actually generates the two CSVs from the zarr store:

- `all_start_zone_features_<DATE>.csv` — `profile_features()` output verbatim
- `meloche_features_all_<DATE>.csv` — `compute_meloche_features()` output

Everything below was verified by reading the avachain source, not inferred.
Nothing in this repo has been modified.

Three defects found in the standalone copy are **absent here** — avachain is the
better implementation and the extraction lost the behaviour. Those are listed
in the last section so they don't get "fixed" backwards.

---

## 1. HIGHEST RISK — CRS mismatch can mis-register the whole CSV

The rasters and outputs are **EPSG:6342** (NAD83(2011) / UTM 13N; the DEM is a
compound CRS with NAVD88 height). But `snowpack_io.py` reprojects to
**EPSG:32613** (WGS84 UTM 13N):

- `snowpack_io.py:180`, `:192`, `:273` — reprojection target 32613
- `snowpack_analysis.py:62` — 32613
- `make_release_raster.py:151`, `fetch_avalanche_obs.py:147-148` — 32613
- `snowpack_analysis.py:69`, `scenario_writer.py:203`/`:231`/`:246` — 6342

If the `cluster_map` or DEM that reaches `compute_meloche_features()` passes
through `snowpack_io.py`, then cluster centroids and per-cluster slope angles
are computed on a mis-registered grid, and **every row of
`meloche_features_all_*.csv` is affected** (`slope_angle`, `tau_g`, `theta`,
`Pi1_elastic`, and everything derived from them).

In practice pyproj currently treats NAD83(2011) → WGS84 as a null transform at
this location, so the numerical offset came out sub-millimetre when measured
directly. That makes this latent rather than actively broken — but it is one
pyproj/PROJ-data upgrade away from becoming a real ~1 m shift, and the two
declarations disagree about what the data *is*.

**Action:** pick one EPSG (6342, matching the rasters), put it in `config.py`,
and make every reprojection target reference it. Verify `cluster_map.tif` and
`dem_1m.tif` CRS with `rasterio` first — do not trust the literals.

## 2. `theta` is in Pa per *pixel*, not Pa/m

`snowpack_analysis.py:461` — `compute_meloche_features(…, transform, …)`
accepts `transform` and never uses it in the body (461–650).

`snowpack_analysis.py:533` has the comment "Convert pixel distance to meters
(1m resolution DEM)" but `:599` divides by the raw ball-tree distance:

```python
theta_vals.append(abs(tau_p - tau_j) / d)      # d is in PIXELS
```

Correct at 1 m, silently wrong at any other resolution — and `theta` feeds
`Pi1_elastic`, `Pi2_brittle`, `A_ca_brittle` and `A_ca_energy`, so the error
propagates into every arrest index.

**Fix:** multiply neighbour distances by `abs(transform.a)`.

## 3. `A_ca_elastic` is uncalibrated and should not be emitted

`snowpack_analysis.py:633` (emitted at `:644`, with `L_SS = 20.0` at `:485`):

```python
A_ca_elastic = L_SS * Pi1**(1.5)
```

JGR Eq. 19 is a *proportionality* — the paper publishes no constant. This line
silently assumes the constant is 1. In the Little Professor CSV that produces a
median of **7 800 m** and a maximum of **3.7 × 10⁶ m**; physically meaningless
numbers sitting in a shipped data file.

Nothing downstream reads it. **Fix:** drop the column and keep the
dimensionless `Pi1_elastic`, which carries the same information.

## 4. `profile_features(min_depth_cm=…)` is accepted and ignored

`snowpack_analysis.py:231` — the parameter is in the signature and documented,
but never referenced in the body (231–375). Callers passing 10.0 get no
filtering.

Implementing it is low-risk: `z` is negative downward from the surface, so the
test is `-interface_z < min_depth_cm / 100.0` → return without slab/WL
features. On the Jan 18 data this changes 2 of 3 542 clusters, both already
removed by the 0.5 m slab-thickness filter downstream.

## 5. 50 Pa feature floor sits above the 40 Pa downstream filter

`snowpack_analysis.py:580-581`:

```python
tau_g_min_valid = 50.0   # Pa — empirical minimum for D2+ events
```

The trigger-selection filter downstream gates on τg ≥ 40 Pa. Clusters between
40 and 50 Pa therefore pass the documented filter but have **no arrest indices
at all** — they were short-circuited here. Either raise the downstream gate to
50 Pa or lower this floor, but don't leave the two disagreeing.

## 6. Non-deterministic row selection for duplicate cluster IDs

`analyze_release_zone.py:898-906` concatenates the `snap_data` groups *plus* a
separate `start_zone` frame. `compute_meloche_features` then dedupes with
`keep='first'` (`snowpack_analysis.py:519`).

Which duplicate wins therefore depends on dict insertion order of `snap_data`.
Reordering that dict silently changes CSV contents. **Fix:** dedupe
explicitly on a documented priority, or assert uniqueness before concatenating.

## 7. Two code paths write the same CSV filename

`reinitialize_snowpack.py:346` writes a file using the same
`all_start_zone_features_` prefix as `analyze_release_zone.py:893`. Two
different producers, one filename — whichever ran last wins, with no way to
tell from the file. **Fix:** distinct names, or stamp the producer into the CSV.

## 8. Hardcoded 1 m pixel size

`np.gradient(fill_dem, 1.0)` at `snowpack_analysis.py:124`, `:493`;
`avalanche.py:64`, `:455`, `:663`. All slope/aspect grids assume 1 m. Derive
from `transform` instead.

## 9. Dead `mode3_scale`, with a docstring that still claims it works

`release_geometry.py:474` is honest — `mode3_scale: float = 2.0, # kept for API
compat, not used`. But two things still advertise it:

- `release_geometry.py:332` docstring: "Lateral: Π₁ < Π₁_trigger × mode3_scale
  (mode III)" — not what the code does
- `analysis_pipeline.py:1024` still exposes `--mode3-scale` on the CLI

**Fix:** remove the flag and correct the docstring, or implement mode III.
Pending Gaume's antiplane formula, removal is the honest option.

## 10. `max_clusters = 500` silently bounds the release region

Two independent implementations, each with its own hardcoded default:
`release_geometry.py:479` (default) / `:830` (loop condition), and
`probabilistic_release.py:44` (default) / `:123` (loop condition). No warning
when it binds, in either.

Measured in the standalone repo on Jan 18: the cap binds on 3 of 5 triggers,
and lifting it changes median release area by ~20% (5 366 → 6 405 m²) and best
IoU from 0.639 to 0.667. Region size saturates by ~1 000 clusters, so any value
≥ 1000 gives an identical answer — raising it is *not* tuning to the event.

**Fix:** single constant in `config.py`, and print a warning when the BFS exits
with clusters still propagating, so a capped polygon is never mistaken for a
physical result.

## 11. Cross-slope width never applies

`release_geometry.py:245-253` looks for `tau_p` / `wl_shear_strength` in
**`meloche_df`**, which does not carry either column (see schema below), then
returns `A_ca` unchanged on NaN. The Gaume (2015) / θ flank constraint is
therefore dead, and lateral extent is just `A_ca`.

**Fix:** either read τp from the `profile_features` frame, or add `tau_p` to the
`compute_meloche_features` output. The standalone repo now does both.

## 12. Meloche arrest test is disabled — document it

`release_geometry.py:49` `USE_MELOCHE_ARREST = False`. What actually arrests the
BFS is `TAU_G_ABS_FLOOR = 350.0` (`:542`) plus Λ/thickness discontinuity
heuristics (`:798-800`, 0.20/0.30). The 350 Pa floor — not the documented 40 Pa
gate — is the binding driving-stress criterion, and it is uncalibrated.

This may well be the right engineering choice. It just needs to be stated,
because the module names imply the Meloche criterion is doing the work.

---

## E and σt provenance — keep this, it was nearly lost

`snowpack_analysis.py:358-363` carries the provenance that the standalone
extraction dropped:

```python
# Slab elastic modulus — power law fit to van Herwijnen et al. (2016) range
# ~2 MPa at ρ=200, ~4 MPa at ρ=300, ~6 MPa at ρ=350 kg/m³
E_slab  = (rho / 300.0)**2.5 * 4.0e6             # Pa
# Slab tensile strength — ~5 kPa at ρ=300 kg/m³ (Meloche 2-10 kPa range)
sigma_t = (rho / 300.0)**1.4 * 5.0e3             # Pa
```

These are hand-fits through anchor points of a *reported range*, not published
regressions, so they carry no fitted uncertainty and are unvalidated outside
200–350 kg m⁻³. Λ ∝ √E and G_slab ∝ σt²/E, so both arrest frameworks inherit
the choice. Replacing them with a cited regression is the highest-value
improvement to the input chain.

Note there is a **duplicate** of these formulas in MPa/kPa units at
`snowpack_analysis.py:758-759`. Two copies will drift — consolidate.

---

## CSV schema the standalone package consumes

`meloche_features_all_*.csv` (written `snowpack_analysis.py:629-645` via
`analyze_release_zone.py:910`), indexed by `cluster_id`:

```
slope_angle, tau_g, theta, Pi1_elastic, Pi2_brittle, Lambda, L_t,
A_ca_elastic, A_ca_brittle, rc_wl, G_slab, tau_p_star, R0, A_ca_energy
```

All columns the standalone package requires are present. Caveats:

- **No `tau_p` / `wl_shear_strength`** → this is exactly why issue 11 fails.
  Adding `tau_p` here is the cleanest fix.
- Short-circuit rows (`:574`, `:583`, `:602`) emit only `cluster_id` plus some
  of `tau_g` / `theta` / `slope_angle` / `note`. So a `note` column may appear
  and most columns will be NaN for those clusters. Consumers must tolerate it.
- `A_ca_elastic` should be removed (issue 3).

`all_start_zone_features_*.csv` is `profile_features()` output verbatim
(`analyze_release_zone.py:893`) and contains everything needed: `hs`,
`min_sk38`, `slab_density`, `slab_thickness`, `wl_shear_strength`,
`wl_thickness`, `E_slab`, `sigma_t`, `Lambda`.

**Use `slab_thickness`, never `hs`, as release depth.** `hs` is total snowpack
height (ground → surface); using it implies a full-depth release to ground. The
standalone repo had this bug; avachain does not (see below).

---

## Where avachain is already correct — do not "fix" these backwards

The standalone extraction regressed three behaviours that are right here:

| Behaviour | avachain | standalone (now fixed) |
|---|---|---|
| Release depth source | `release_geometry.py:905` `depth_from_snowpack` uses per-cluster slab thickness above the WL, falling back to `wl_fraction × HS` | had used `hs` directly |
| Depth raster clipped to polygon | `analysis_pipeline.py:822` calls `rasterize_release_polygon(polygon, scaled_depth, …)`, passed at `:842` | never clipped — volume was domain-wide and identical across scenarios |
| `depth_percentile` actually does something | `config.py:57` `depth_scales = {10:0.75, 50:1.00, 90:1.30}`, applied at `analysis_pipeline.py:820-821` | was a pure no-op |

---

## Suggested order of work

1. **Issue 1 (CRS)** — resolve before regenerating any CSV; everything else is
   downstream of the grid being right.
2. **Issue 2 (θ units)** — one-line fix, affects every arrest index.
3. **Issues 5, 6, 7** — cheap determinism/consistency fixes to the CSV path.
4. **Issue 3** — drop `A_ca_elastic`; add `tau_p` while touching the schema,
   which also fixes issue 11.
5. **Issues 9, 10, 12** — honesty about what the BFS does; 10 changes results.
6. **Issue 4, 8** — low risk, do whenever.

After regenerating, cross-check against the standalone package:

```bash
cd ~/avalanche-release-areas
python -m release_areas.generate_scenarios \
    --features-csv <new all_start_zone_features_*.csv> \
    --meloche-csv  <new meloche_features_all_*.csv> \
    --cluster-map  data/little_prof/spatial/cluster_map.tif \
    --dem          data/little_prof/dem_1m.tif \
    --start-zone   data/little_prof/boundaries/start_zone.kml \
    --release-poly data/little_prof/boundaries/avalanche_release_area_20260118.geojson \
    --max-clusters 2000 \
    --out-dir      /tmp/check
```

Current reference values with the shipped CSVs (observed crown 6 934 m²):
filter chain 1608 → 493 → 408 → 204 → 102, triggers
`[2859, 5656, 2858, 348, 1068]`, median area 6 405 m², best IoU 0.667.
A θ-unit fix should leave these unchanged at 1 m resolution; anything else
moving means something upstream changed too.
