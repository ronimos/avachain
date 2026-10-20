# Little Professor Pipeline — TODO

**Updated:** 2026-10-13  
**Context:** UAS → distributed SNOWPACK → release geometry → com1DFA runout probability

See [`CSV_FEATURE_PIPELINE_ISSUES.md`](../CSV_FEATURE_PIPELINE_ISSUES.md) for the
full 2026-10-06 audit of the feature-CSV generation path (verified file:line
evidence for every item in the "Feature-CSV pipeline audit" section below).

---

## High Priority (before next operational run)

- [x] **Post-avalanche SNOWPACK reinitialization** — `reinitialize_snowpack.py` implemented with min-kernel release detection and `.sno` layer scouring. Integrated as `step_reinit` in `forcing_pipeline.py`. Tested on Jan 18 event. Remaining: validate post-reinit SNOWPACK output against post-event UAS HS observation.

- [ ] **Spatially distributed scour depth raster** — currently `scour_depth_m` is extracted only at the trigger cluster and applied uniformly across the path. com1DFA supports an entrainable-depth raster where each cell holds its own scour cap. **Raster generation done:** `step_scenarios` now calls `_build_scour_depth_map()` which runs the failure-plane-to-hard-layer scan for every cluster in the domain (using `slab_thickness` from features for start-zone clusters; full-depth scan for path clusters without a defined failure plane) and writes `scour_depth.tif` alongside `depth.tif` in the scenario output directory. **Remaining:** wire `scour_depth.tif` into the com1DFA ini file as the entrainable-depth raster input. **Blocked by:** com1DFA entrainable-depth raster input bug (may be config issue). Revisit once the raster input path is working.

- [ ] **Run observed-release validation for Val (AvaFrame)**
  ```bash
  python src/snowpack-model-feeder/analysis_pipeline.py scenarios \
      --snapshot-date 2026-01-17 --n-triggers 1 \
      --size-factors 0.85 1.00 1.15 --depth-pcts 10 50 90 \
      --use-observed-release
  ```
  Validates flow model independently of release geometry.

- [ ] **Calibrate com1DFA Voellmy μ/ξ** against Jan 18 runout observation.  
  Current values are **uncalibrated defaults**: μ=0.155 (dry friction), ξ=1500 m/s² (turbulent friction). These are flow dynamics parameters independent of SNOWPACK — SNOWPACK provides release conditions (geometry, depth, density), but μ and ξ govern the flowing avalanche's interaction with the terrain surface (path roughness, confinement, entrainment). They cannot be derived from the snowpack state and must be calibrated empirically against observed runout.  
  **Calibration approach:** run com1DFA with the Jan 18 Meloche-derived release polygon (size_factor=1.00, P50 depth) across a μ/ξ grid, e.g. μ ∈ [0.10, 0.15, 0.20, 0.25, 0.30] × ξ ∈ [1000, 1500, 2000, 2500]. Score each combination against observed deposit extent (debris toe position, lateral spread). Select the (μ, ξ) pair that minimizes runout error. Typical D2 dry slab values on open terrain: μ ≈ 0.15–0.30, ξ ≈ 1000–2500 m/s².  
  **Dependency:** requires the observed-release validation run and the US-6 road polygon to also evaluate P(exceedance) sensitivity to μ/ξ.  
  **Note for multi-path extension:** μ/ξ are path-specific. Calibration on Little Professor does not transfer directly to Seven Sisters or Muleshoe — each path needs its own calibration against observed events or must use literature defaults with wider uncertainty bounds.

- [x] **Resolve EPSG:32613 vs EPSG:6342 split** — DONE 2026-10-13. Added
  `RASTER_EPSG = 6342` to `config.py`. Updated all seven 32613 occurrences
  in `snowpack_io.py`, `analysis_pipeline.py`, `fetch_avalanche_obs.py`,
  `avalanche.py`, `probabilistic_release.py`, `snowpack_analysis.py`,
  `make_release_raster.py`. The one remaining 32613 in
  `snowpack_analysis.py:159` is a comment showing example input formats.

- [ ] **US-6 road polygon** for P(exceedance) calculation at infrastructure.

- [ ] **Confirm AvaFrame version** supports raster depth input format.

---

## Pipeline Completeness

- [ ] **Couple `scenarios` step to run both physical and probabilistic models** — add `--probabilistic` flag to `analysis_pipeline.py scenarios` that calls `probabilistic_release.py` for each trigger cluster after physical scenarios are written. Produces `release_comparison_YYYY-MM-DD_probability_model.png` alongside `_physical_model.png`.

- [ ] **Probabilistic model → AvaFrame scenario set** — integrate probabilistic release polygons as additional scenarios in the com1DFA ensemble. Weights derived from P(arrest) distributions rather than Sk38-based weights. This propagates release zone uncertainty through to runout probability envelopes, enabling the full probabilistic decision-support chain:
  ```
  P(release boundary) × P(runout | release) = P(exceedance at road)
  ```
  Currently the probabilistic model produces comparison plots only.

- [ ] **`--restrict-to-release-area` validation run** — confirm trigger selection after all filter fixes (Sk38 < 0.5, elevation P75).

- [ ] **Tune size_factor ensemble** against Jan 18 to confirm correct Mel/Obs ratio spread across 0.70–1.30.

- [ ] **Cross-date Meloche analysis** — run scenarios for Jan 10, Jan 14, Jan 17 to confirm Π₁ separation growing toward event. Should show increasing propagation probability as buried WL weakens.

- [ ] **Step 6 cleanup** — remove remaining duplicated logic between diagnostic scripts (`analyze_release_zone.py`, `visualize_snowpack.py`) and modules.

- [ ] **Multi-event SNOWPACK reinitialization** — current shell scripts support a single `--reinit` event. For seasons with multiple avalanches (same or different paths), extend to accept an events JSON file listing all events chronologically. The pipeline would loop: run SNOWPACK to event A → reinit A → run to event B → reinit B → ... → run to end of season. Each reinit scours a different (possibly overlapping) set of clusters. Events JSON format:
  ```json
  [
      {"event_date": "2026-01-18", "date_before": "2026-01-14",
       "date_after": "2026-01-20", "label": "Little Professor D2"},
      {"event_date": "2026-02-05", "date_before": "2026-01-27",
       "date_after": "2026-02-10", "label": "Seven Sisters D1.5"}
  ]
  ```
  Each event needs its own survey pair for detection (or a pre-drawn GeoJSON) and its own snapshot date for slab thickness. Clusters affected by earlier events will already have reduced HS when subsequent reinits run.

---

## Physical Model Improvements

### Release-areas sync, 2026-10-10 — what landed here

The standalone repo (`~/avalanche-release-areas`) had accumulated physics this
repo did not have, including two direction bugs that were live in the production
path. Ported in one pass; 280 tests pass here (`tests/` was 232).

- [x] **Two terrain-direction bugs — both fixed here.** They confused map
  direction with raster indexing, and **row increases southward**.
  - `find_stauchwall` stepped `dr = sign(-cos(asp))`, `dc = sign(sin(asp))`.
    `np.sign` is ±1 for any nonzero component, so only the four bearings
    45/135/225/315 were reachable — a pure diagonal on **100.00%** of Jan 18
    start-zone pixels, median **+13.9°** clockwise error, and the walk was a
    straight ray that never followed terrain. Now carries sub-pixel position.
    `transform` is still accepted (call sites pass it positionally) and unused.
  - `estimate_cross_slope_width` picked "abeam" neighbours by raster row/col
    (`abs(c_row - t_row) < 50 and abs(c_col - t_col) > 5`), which assumes a
    north–south fall line. Selected neighbours sat a mean **30°** off the true
    cross axis — one within 1.2° of the fall line — so `theta_cross` absorbed
    the along-slope gradient. Now samples in the map frame within
    ±`THETA_CROSS_SECTOR_DEG` of the cross axis, and needs `aspect_deg`: without
    it the θ widening is skipped rather than taken along the raster axes.
  - **The lesson generalises.** Any new code here that derives a direction must
    be checked against the map frame, never assumed from the array axes.
    `tests/test_terrain_direction.py` (34 tests, ported) covers both helpers
    with planar-DEM fixtures at known bearings. Neither had coverage before.

- [x] **The Gaume width ratio no longer depends on the CSV θ.**
  `estimate_cross_slope_width` measures *both* halves itself via
  `_sector_theta` — θ_along about the fall line, θ_cross about the cross axis —
  from the same τp field in the same separation band. Pairing the CSV's
  isotropic k-NN mean at a ~3.3 m lag with a directional mean at 5–50 m
  inflates the ratio several-fold on its own, given θ(d) ≈ 3.87 + 72/d Pa/m.
  The CSV θ survives only as a fallback numerator when no cluster lies in the
  along-slope sector. Note the Gaume path is effectively **dormant** on Jan 18:
  `d_lat = min(gaume_width, A_ca·0.712)` and the mode III speed cap binds.

- [x] **θ estimator is selectable, default `plane_fit`** (`snowpack_analysis.py`).
  θ is a property of the **lag it is measured at**, not of the snowpack. Our
  clusters are ~3 m across and k=6 averages 3.3 m, inside the 0.5–10 m band
  Meloche et al. treat as local noise (Appendix E). The τp variogram gives
  θ(d) = 3.87 + 72/d Pa/m: at 3.3 m θ is **15% slope-scale trend, 85% local
  noise**. `plane_fit` fits a local least-squares plane within
  `THETA_FIT_RADIUS_M = L_SS` and takes |∇τp|; `knn` reproduces every
  pre-2026-10-09 CSV. Median θ 25.19 (`knn`) vs 7.28 (`plane_fit`) Pa/m on
  Jan 18, so A_ca × ~3.3. **The argument is the calibration scale, not the fit**
  — Meloche's θ is the gradient of a ramp over L_ss, so the neighbourhood should
  be L_ss. Do not pick it by IoU: θ moves Π₁, hence the whole filter chain.
  Also emits `theta_grad_east` / `theta_grad_north` (NaN under `knn`), so a θ
  split is available from the estimator itself.

- [x] **`arrest_indices.py` is now identical to the standalone's** (modulo doc
  paths). That brings `mode3_length_ratio`, `cp_over_cs`, `k_x_cross`,
  `L_dyn_cross`, the named `C_FIT`, and **`SCH2004_EXP = 1.9`** — the exponent
  as printed in Schweizer, Michot & Kirchner (2004) Eq. 8, verified against the
  PDF. The old 2.0 was this project's rounding of the abstract's "about 2" and
  runs 11% low at ρ = 300, 17% at ρ = 150; `SCH2004_EXP_ROUND` reproduces it.
  This **cannot** move a release polygon — `k_ic()` reaches only the ligament
  columns, which the BFS never reads — so it had to be decided from the source.

**Operationally: existing meloche CSVs are stale.** θ changes, so Π₁, the
trigger ranking and the whole filter chain change with it. Nothing moves until
`compute_meloche_features` is re-run. Validated end-to-end in the standalone
repo on Jan 18 (best IoU 0.677 at `--max-clusters 2000`, mean area/observed
0.82 against the 0.83 mask-limited ceiling); **not yet re-run through this
pipeline.** Do that before trusting any avachain release polygon.

- [ ] **Re-run the Jan 18 analysis here and reconcile against the standalone.**
  The two repos should now agree on geometry given the same CSVs. Any
  divergence is a porting defect, so this is the acceptance test for the sync.

- [ ] **Anisotropic BFS rings — to investigate (Ron's idea, 2026-10-09).** As
  `propagate_release` adds rings to the flood-fill, grow roughly **1.6× further
  along-slope than cross-slope** per ring, instead of growing isotropically and
  only clipping afterwards with `d_up` / `d_down` / `d_lat`. The ratio is
  already in the model as `MODE2_SPEED_RATIO`: mode II runs supershear at
  ~1.6 c_s while mode III caps at c_s (Broberg 1989). If a ring represents one
  increment of crack-front advance in *time*, the front should already be ~1.6×
  further along-slope when it arrives — the anisotropy belongs in the growth,
  not only in a terminal cap; at present the front shape is an artifact of
  cluster geometry. Settle first whether this *replaces* the speed cap or
  composes with it (both would double-count the same 1.6). Judge it against the
  observed crown's aspect (major axis 119°, L/W 3.51), a sharper test than IoU,
  and do not tune the ratio to this event.

- [x] **Sk38 ties are now broken by r_c** (2026-10-20, `_rank_triggers`). Places
  3–5 used to turn on CSV row order inside two-decimal Sk38 ties (4-way at 0.79
  on Jan 18, 10-way on Mar 7). They now go to the lowest critical cut length at
  the same WL. Needed the `rc_flat` fix (782575c); rc_wl had been NaN everywhere.

- [ ] **Trigger ranking is decided by the slab-thickness cap, not by stability.**
  Within the eligible pool, Sk38 at the WL is almost a function of slab
  thickness (Spearman −0.93 on Jan 18), so the ranking picks the thickest slabs
  under `--max-slab-thickness`. Moving the cap 2.0 → 2.5 m replaced all five
  triggers on both Jan 18 and Mar 7. 1.5 m left two candidates on Mar 7. Tried
  2026-10-20, none adopted:
  - SSI: it is exactly Sk38 + 2 here.
  - r_c as the primary key, or Sk38 × r_c: r_c also tracks thickness (−0.70).
  - Sk38 residuals after a quadratic fit on slab thickness: whole-zone fit
    (cap-stable, but drifts to thin slabs with Sk38 0.92–0.99); pool fit
    (thickness-neutral, but the fit moves with the cap); pre-cap fit (3–4 of
    5 triggers kept across caps on Jan 18, only 2/5 on Mar 7).
  None beat the current Jan 18 IoU (0.46 mean, 0.70 best); alternatives scored
  0.34–0.41. The likely real fix is a metric that is not depth-driven. Next
  candidate: skier stability evaluated at each cluster's own slope angle rather
  than the fixed 38° of Sk38. Keep the weak-layer definition unchanged. Judge on
  cap stability and Mar 7, not on Jan 18 IoU alone.

- [ ] **θ validity is the strongest open objection to the whole arrest route.**
  Under `plane_fit`, **97.2%** of Jan 18 start-zone clusters fall below
  Meloche's `THETA_VALID_MIN` (20 Pa/m), against 36.2% under `knn`. Read
  literally, the Jan 18 weak layer has no slope-scale ramp steep enough to
  arrest within L_ss. This is **not** an argument for reverting to `knn`, which
  only looked compliant because it was measuring noise. Recorded as
  `THETA_VALID_MIN/MAX`, not enforced; enforcing it is an open modelling
  decision that one event cannot settle.

- [ ] **Adding a τp Gaussian random field is NOT the indicated next step** —
  a reading of JGR Appendix E that earlier notes here got wrong. Appendix E is a
  sensitivity analysis whose stated result is that local noise "mainly affects
  the crack speed"; the paper's arrest-relevant heterogeneity *is* the linear
  ramp. Our measured σ_local is 69 Pa against Appendix E's 500 Pa, and
  superimposing that field would inject 55–169 Pa/m of spurious θ (2–7× the real
  median), shrinking A_ca further in the wrong direction.

- [ ] **Asymmetric gradient thresholds — premise is now out of date.** The
  constants were renamed and *already made asymmetric*:
  `release_geometry.py:798-801` has `LAMBDA_DROP_FACTOR = 0.20` /
  `LAMBDA_RISE_FACTOR = 0.30` and `THICKNESS_DROP_FACTOR = 0.20` /
  `THICKNESS_RISE_FACTOR = 0.30` ("tight: softening arrests readily / loose:
  stiffening rarely arrests"). `LAMBDA_JUMP_FACTOR` and `THICKNESS_JUMP_FACTOR`
  no longer exist. **What is still open:** the 0.20/0.30 split is an
  uncalibrated guess. Measure actual |ΔΛ/Λ| and |Δh/h| at observed crown and
  flank boundaries to check whether 0.30 is the right loosening, and whether Λ
  and thickness warrant different values from each other.

- [ ] **Cross-slope (mode III) elastic length Λ_III — awaiting Gaume's formula.**
  The cross-slope Λ is a *different and smaller* quantity than the up/downslope
  (mode II) Λ, and the antiplane formula is not published yet. Ron expects it
  from Johan Gaume soon.
  **Current state:** `arrest_indices.elastic_length_cross()` uses the slab shear
  modulus, `Λ_III = √(G h D_wl / G_wl)`, giving `Λ_III/Λ_II = √((1−ν)/2) ≈ 0.59`
  at ν = 0.3 — Ron's own derivation, not published. Nothing in the propagation
  path uses it: the BFS applies a single Λ in all directions.
  **Prep done in both repos** (ported here 2026-10-10): arrest evaluation splits
  into along-slope vs cross-slope branches and `release_geometry.directional_lambda(props, is_cross)`
  is the single place the two diverge, with its own continuity thresholds
  (`LAMBDA_CROSS_DROP_FACTOR` / `..._RISE_FACTOR`, currently equal to the
  along-slope pair). `compute_meloche_features` emits `Lambda_cross`. Both
  branches are still fed by the mode II formula, so results are unchanged. To
  land the real formula: (1) update `arrest_indices.elastic_length_cross()`,
  (2) set `release_geometry.USE_MODE3_LAMBDA = True`, (3) regenerate the meloche
  CSV. Nothing else should need to change — if it does, the split has drifted.
  **Measured caveat — a constant Λ_III/Λ_II ratio changes nothing.** The Λ
  continuity gate tests the *relative* change (Λ_nbr − Λ_cur)/Λ_cur, which is
  invariant under uniform rescaling of Λ, and the only absolute threshold
  (`MIN_PROPAGATION_LAMBDA` = 0.1 m) is never approached (min Λ_II = 0.54 m on
  Jan 18 → min Λ_III = 0.32 m). The current derivation gives
  Λ_III/Λ_II = √((1−ν)/2), a constant, so switching to it provably changes
  nothing — verified by running both flag states in the standalone repo:
  identical areas and IoU. **A real Λ_III will only move the flanks if it
  carries different spatial structure**, i.e. depends on properties that vary
  across clusters differently from E′. (A spatially varying Λ_cross does change
  every release area, confirming the plumbing is live.) Worth checking against
  Gaume's formula as soon as it arrives: if it is just a constant factor, the
  flank behaviour will need a different mechanism — most likely the mode III
  crack-speed cap rather than Λ.
  **Expected physics once implemented** (methods §3, not yet simulated): mode III
  cracks cannot exceed `c_s` (Broberg 1989) versus ~1.6 `c_s` upslope, capping
  lateral energy flux; and because G < E′ an equal-strength slab has a *larger*
  energy cap cross-slope. If both hold, lateral arrest is driven mainly by the
  speed cap and weak-layer heterogeneity rather than slab fracture — consistent
  with the JGR paper attributing lateral arrest to WL heterogeneity.
  (The old `--mode3-scale` flag that once stood in for this was a no-op and was
  removed 2026-10-13; the real mechanism is `directional_lambda` plus the
  speed cap, so nothing is waiting on a CLI surface.)

- [x] **Mode III crack-speed cap — PORTED HERE 2026-10-10.**
  Implemented 2026-10-06 in `~/avalanche-release-areas`, mirrored here with the
  rest of the release-areas physics (see the port entry at the top of this
  section). `arrest_indices.mode3_length_ratio()` gives
  `L_dyn,III/L_dyn,II = k_x,II/k_x,III = 0.712` at ν=0.3 (E and ρ cancel; the
  ratio depends only on ν and the two speed ratios). `evaluate()` now also
  returns `k_x_cross`, `L_dyn_cross`, `mode3_length_ratio`, and
  `compute_meloche_features` emits `L_dyn`, `L_dyn_cross`,
  `mode3_length_ratio`. `propagate_release` composes it with the Gaume width as a
  minimum: `d_lat = min(gaume_width, A_ca·0.712)·size_factor`, via
  `release_geometry.mode3_speed_cap_ratio()`, which prefers the CSV column and
  falls back to computing the ratio from ν so pre-column CSVs still get the cap.
  Behind `release_geometry.USE_MODE3_SPEED_CAP` (default on); off is
  bit-for-bit identical to pre-cap.
  **Measured on Jan 18:** unlike Λ_III this is *not* scale-invariant — it is an
  absolute distance cap, so it does change results. Binds on 2 of 5 triggers
  (the other 3 were already Gaume-limited below it): median area
  6 405 → 5 389 m², ratio 0.92 → 0.78, best IoU unchanged 0.667, mean IoU
  0.567 → 0.561.
  **So it makes the aggregate fit slightly worse, and that is expected.** Only
  the *restrictive* half of mode III is coded: because G < E′ an equal-strength
  slab has a *larger* energy cap cross-slope (G_slab,III = τ_flank²h/2G vs
  σt²h/2E′), which pushes the other way, but it needs a flank strength τ_flank
  with no parameterisation. Left on because Broberg's limit holds independently
  of this event — do not tune it against Jan 18.
  **Next, in priority order:** (1) find or assume a τ_flank parameterisation so
  `slab_energy_cap_cross` can supply the permissive half; (2) revisit the
  uncalibrated `TAU_G_ABS_FLOOR` (350 Pa) and the Λ/thickness discontinuity
  factors, the likely reason the model under-predicts area before the cap is
  even applied; (3) if the JGR attribution of lateral arrest to weak-layer
  heterogeneity is right, neither Λ_III nor the speed cap is the dominant
  control and the missing ingredient is spatial τp variability on the flanks.

- [ ] **Downslope arrest: direct τ_g < τ_p criterion** — replace 28° slope proxy with direct comparison of distributed τ_g and τ_p from SNOWPACK. Makes stauchwall arrest parameter-free and physically explicit. τ_g and τ_p are both already computed.

- [x] **Critical cut length** — ~~expose SNOWPACK variable 0606 (critical crack length) via xsnow~~. Side-loader `add_critical_cut_length.py` reads 0606 directly from `.pro` files and writes `critical_cut_length` into the Zarr. Functional for downstream use. Upstream xsnow API integration (flag to Florian/SLF) still open — see new item below.

- [ ] **Integrate 0606 parsing into `build_zarr_chunked.py`** — currently the side-loader has to run after every Zarr rebuild. Add a `parse_var` pass inside each batch in the builder; concatenate `critical_cut_length` (and likely `0607 rta`) into the xsnow Dataset before write. Removes the need for the side-loader on full rebuilds.

- [ ] **Fix `build_zarr_chunked.py` resume logic** — existing filename-stem skip check (`f.stem.replace('_cluster_', '_')`) doesn't match xsnow's `StationName=` location coord, so every rerun rebuilds from scratch. Read `StationName=` from each `.pro` header and compare against the Zarr's `location` coordinate.

- [x] **Mayer et al. (2022) P_unstable evaluation on Colorado snowpack** — DONE. Pipeline implemented across two envs (main + `.venv-punstable` with sklearn 0.22.1) and documented in `punstable_evaluation_method.md`. Scripts: `extract_punstable_features.py` → `predict_punstable.py` → `ingest_punstable_to_zarr.py`. Spatial validation against Jan 18 release-area vs adjacent skied-but-not-triggered terrain via `validate_punstable_spatial.py` and `plot_punstable_distributions.py`.

  **Summary of findings:**
  - Internal validation: model picks basal FC/DH as the worst layer at 99.9% of clusters (95.6% FC, 4.3% DH) — Colorado continental signature, without being told to look there. Implicitly cross-validates `split_wl_slab()`.
  - Spatial discrimination: median P_unstable_max essentially identical (release 0.349 vs skied 0.340, Δ +0.009); marginal upper-tail separation (p95 Δ +0.083); 2.5× hit-rate lift at P≥0.77 but on tiny absolute counts (6 vs 5 clusters).
  - Mann-Whitney p=3.8e-9 reflects sample size, not operational signal.
  - Conclusion: Mayer P_unstable correctly identifies *what* the weak layer is but cannot reliably tell us *where* on the slope it will fail. Complements the gradient-arrest physics model (which captures spatial discrimination); does not replace it. ISSW story is the dual-model design empirically justified.

- [ ] **`compare_wl_methods.py` column rename** — the column currently labeled "P_unstable" in this script is `stab_deformation_rate` (S′), which saturates at 6.0. Now that actual P_unstable exists in the Zarr (`p_unstable_max`), rename the misleading column and optionally add the real P_unstable as a second comparison panel.

- [ ] **Mayer Pk crust algorithm refinement** — current implementation in `extract_punstable_features.py` uses single-layer MFcr detection rather than Mayer's contiguous-block accumulation. Effect on Colorado snow likely minimal (strong crusts rare); revisit only if a reviewer asks.

- [ ] **Per-cluster slope angle in Pk** — `extract_punstable_features.py` uses a global default (38°) used only for MFcr thickness projection. Pull `SlopeAngle=` from each `.pro` header into a sidecar or add as a Zarr coord. Bounded effect.

- [ ] **Formal smoke test of P_unstable predictions against SLF's `input_example`** — we use sklearn 0.22.1 in an isolated env but on a newer OS / joblib version. Run the published example profile through our pipeline and compare predictions against SLF's reference outputs to close the model-validity loop.

- [ ] **P_unstable follow-ups (low priority, only if ISSW reviewer asks):**
  - Time-evolution of P_unstable_max at trigger cluster vs skied clusters through the season — could reveal temporal discrimination the snapshot test misses.
  - Timestep sensitivity: re-run validation at 20:00 UTC and 22:00 UTC. If signal is stronger at a different snapshot, reconsider temporal-saturation interpretation.
  - The "0.952 out-of-crown" cluster case study — high P_unstable but Ron's SNOWPACK review suggests insufficient slab for crack propagation. Document as a case study of why high P_unstable alone doesn't imply triggering.

- [ ] **BFS propagation at 1 m grid resolution** — test the BFS crack propagation model on the native 1 m DEM grid instead of the cluster neighbor graph, restricted to the area around the observed Jan 18 release area. Currently the BFS operates on ~6,636 clusters (~3 m median diameter); running at 1 m resolution within a cropped domain (~200×200 m around the release) would evaluate arrest criteria at the pixel scale (~64K cells but only ~4K in the cropped area). This would reveal whether cluster-scale smoothing of slab properties masks sharp gradients that control arrest at finer scales, and whether the BFS boundary converges to the same location as the cluster-based result. If boundaries differ meaningfully, the cluster resolution may need tightening (smaller `max_cells_per_cluster`) or the gradient thresholds may need recalibration for the finer grid. This is a diagnostic test — 1 m resolution across the full start zone is not operationally viable (would require ~65K SNOWPACK simulations).

---

## Feature-CSV pipeline audit *(2026-10-06)*

Found by auditing `~/avalanche-release-areas` (the standalone extraction of this
code) against its own methods doc, then verifying each item against the avachain
source. Every file:line below was checked directly. Full writeup with code
quotes: [`CSV_FEATURE_PIPELINE_ISSUES.md`](../CSV_FEATURE_PIPELINE_ISSUES.md).

CRS split is the highest-risk item and lives in **High Priority** above.

- [x] **`theta` is Pa per *pixel*, not Pa/m** — FIXED 2026-10-10 with the θ
  estimator port. `compute_meloche_features()` now takes
  `px_m = abs(transform.a)` and both estimators (`theta_knn`, `theta_plane_fit`)
  work in metres; `np.gradient(fill_dem, px_m)` too. A no-op on the current 1 m
  data, as predicted. Covered by
  `tests/test_theta_estimators.py::test_px_m_scales_theta_inversely`.

- [x] **Drop `A_ca_elastic` — it is uncalibrated** — DONE 2026-10-13.
  Removed the computation and output column from `snowpack_analysis.py`.
  `Pi1_elastic` carries the same information without the bogus constant.

- [x] **Add `tau_p` to the meloche CSV — unblocks cross-slope width** — DONE
  2026-10-10. `compute_meloche_features()` now emits `tau_p`, and
  `estimate_cross_slope_width()` also accepts `snap_features` and falls back to
  it via `_tau_p_lookup()`, so the Gaume (2015)/θ flank constraint is live
  either way. Expect release polygons to move: `d_lat` was effectively a
  constant `A_ca` and becomes per-trigger. Note it is now composed with the
  mode III speed cap as a `min()`, and on Jan 18 the speed cap is what binds.

- [x] **`max_clusters = 500` silently bounds the release region** — DONE
  2026-10-12. `MAX_BFS_CLUSTERS = 2000` in `release_geometry.py`, wired through
  `PipelineConfig.max_bfs_clusters` and `--max-clusters` CLI flag. A `[CAP-BOUND]`
  warning is emitted when the BFS exits with clusters still propagating.

- [x] **50 Pa feature floor sits above the 40 Pa downstream filter** — DONE
  2026-10-13. `tau_g_min_valid` lowered to 40 Pa in `snowpack_analysis.py` to
  match `MIN_TAU_G` in `analysis_pipeline.py`. Both now use the scaling-law
  applicability threshold; the D2+ character of the event is enforced by the
  slab-thickness and slope filters downstream.

- [ ] **Duplicate cluster IDs resolve by dict insertion order** —
  `analyze_release_zone.py:898-906` concatenates the `snap_data` groups *plus* a
  separate `start_zone` frame; `compute_meloche_features` then dedupes with
  `keep='first'` (`snowpack_analysis.py:519`). Which duplicate wins depends on
  dict ordering, so reordering `snap_data` silently changes CSV contents.
  **Fix:** dedupe on a documented priority, or assert uniqueness before concat.

- [ ] **Two code paths write the same CSV filename** —
  `reinitialize_snowpack.py:346` writes a file using the same
  `all_start_zone_features_` prefix as `analyze_release_zone.py:893`. Whichever
  ran last wins, with no way to tell from the file. **Fix:** distinct names, or
  stamp the producer into the CSV.

- [ ] **`profile_features(min_depth_cm=…)` is accepted and ignored** —
  `snowpack_analysis.py:231`; parameter is in the signature and documented but
  never referenced in the body (231–375). **Fix:** `z` is negative downward from
  the surface, so the test is `-interface_z < min_depth_cm / 100.0` → return
  without slab/WL features. Low risk: changes 2 of 3 542 clusters on Jan 18, both
  already removed by the 0.5 m slab-thickness filter downstream.

- [ ] **Consolidate the duplicated E / σt parameterisation** — the provenance
  comment at `snowpack_analysis.py:358-363` is the *only* record that these are
  power-law hand-fits through anchor points of the **van Herwijnen et al. (2016)
  range** (~2 MPa at ρ=200, ~4 at 300, ~6 at 350) rather than published
  regressions. A second copy in MPa/kPa units exists at `:758-759`; two copies
  will drift. **Keep the comment** — the standalone extraction lost it and it had
  to be recovered from here. Since Λ ∝ √E and G_slab ∝ σt²/E, both arrest
  frameworks inherit this choice; replacing it with a cited regression is the
  highest-value improvement to the input chain. Unvalidated outside
  200–350 kg m⁻³ (Jan 18 densities 264–332 are inside).

- [x] **Remove the dead `--mode3-scale` flag** — DONE 2026-10-13. Removed from
  `propagate_release()` and `make_release_polygon_2d()` signatures, from both
  call sites in `analysis_pipeline.py`, and from argparse. The docstring for
  `make_release_polygon_2d` now correctly states that lateral extent is governed
  by the mode III speed cap via `d_lat` in `propagate_release`.

- [ ] **Document that the Meloche arrest test is disabled** —
  `release_geometry.py:49` `USE_MELOCHE_ARREST = False`. What actually arrests the
  BFS is `TAU_G_ABS_FLOOR = 350.0` (`:542`, checked `:711`) plus the Λ/thickness
  discontinuity heuristics (`:798-801`). So the **350 Pa floor, not the
  documented 40 Pa gate, is the binding driving-stress criterion**, and it is
  uncalibrated. This may be the right engineering choice — it just needs saying,
  because the module names imply the Meloche criterion is doing the work.

- [ ] **Derive pixel size from `transform` instead of hardcoding 1 m** —
  partially done 2026-10-10: `compute_meloche_features` now uses
  `px_m = abs(transform.a)` for both the DEM gradient and the θ estimators.
  Still hardcoded: `snowpack_analysis.py:203`; `avalanche.py:64`, `:75-76`,
  `:455`, `:663`. Those slope/aspect/HS-difference grids all assume 1 m.

### Already correct here — do not "fix" backwards

The standalone extraction regressed three behaviours that avachain gets right.
Noted so a future sync does not propagate the worse version:

| Behaviour | avachain | standalone (since fixed) |
|---|---|---|
| Release depth source | `release_geometry.py:905` `depth_from_snowpack` uses per-cluster slab thickness above the WL, falling back to `wl_fraction × HS` | used `hs` (ground→surface) — implied a full-depth release |
| Depth raster clipped to polygon | `analysis_pipeline.py:822` calls `rasterize_release_polygon(polygon, scaled_depth, …)` | never clipped — volume was domain-wide and identical across scenarios |
| `depth_percentile` is real | `config.py:57` `depth_scales = {10:0.75, 50:1.00, 90:1.30}`, applied `analysis_pipeline.py:820-821` | pure no-op |

---

## Probabilistic Model *(low priority — need more avalanche events)*

- [ ] **Resolve Λ coefficient sign ambiguity** — `delta_lambda_rel` has unexpected negative coefficient (larger Λ gradient → interior, not boundary). Hypothesized cause: internal crown-to-stauchwall Λ gradient dominates within the release area. Collect 3–5 more events to test whether coefficient stabilizes to positive sign.

- [ ] **Add events to training dataset** — rerun `fit_boundary_model.py` with `--append-pairs` after each new observed event. Target ROC-AUC > 0.70 with 3–5 events.

- [ ] **Calibrate propagate_threshold** — current CLI default is 0.6 (Jan 18 validation used 0.5, producing 4,862 m² vs 3,902 m² observed). As training dataset grows, calibrate threshold to maximize F1 on held-out events. Recommended operational range: 0.5–0.75.

- [x] **~~Platt scaling / isotonic regression~~** — DONE. Isotonic calibration implemented via `CalibratedClassifierCV(method='isotonic', cv=5)` in `fit_boundary_model.py`. The remaining P(arrest) compression (P10=0.058, P90=0.583) is a sample-size limitation (294 boundary pairs), not a calibration defect. Distribution will widen as events accumulate.

- [ ] **Test Bayesian updating framework** — as events accumulate, switch from refitting logistic regression to Bayesian logistic regression (PyMC) to properly track posterior uncertainty on coefficients. The δτ_p coefficient (+8.30) has high uncertainty with n=1 event; credible intervals will tighten with additional data.

---

## Data / Infrastructure

- [ ] **Environment setup and data collection guide** — write a step-by-step guide covering: (1) Python environment setup (which conda/venv environments are needed and why, e.g. `.venv-punstable` for sklearn 0.22.1); (2) required input files by step (DEM, cluster map, survey HS grids, start-zone KML, weather station SMET); (3) what data to collect before running the chain (UAS survey dates and file naming conventions, station IDs, SNOWPACK installation path, AvaFrame/com1DFA version requirements); (4) directory layout expected by `run_full_pipeline.sh`; (5) a minimal worked example going from raw survey to a hazard assessment output. Target audience: a new field team member who has the data in hand but has never run the pipeline.

- [ ] **Min-kernel auto-detection of release area** — automated start zone delineation from terrain and snowpack, reducing dependence on manually drawn KML.

- [ ] **InSAR snow depth integration** — pipeline currently accepts UAS SfM and lidar. Add InSAR-derived snow depth as third input option (noted in ISSW abstract as future capability).

- [ ] **Multi-path extension** — Seven Sisters, Star Mountain, Muleshoe paths beyond Little Professor. Requires per-path start zone KMLs and terrain parameters.

---

## Post-ISSW Cleanup *(October 2026)*

- [ ] **Delete one-off test scripts** — these were used for ISSW prep and won't be needed after the conference: `expand_march7_envelopes.sh`, `run_reinit_comparison.sh`, `scripts/compare_reinit_runs.py`. Currently left with unstaged changes; delete and commit after ISSW.

---

## Visualization / Presentation

- [ ] **High-resolution avalanche photo with Google Earth overlay** — acquire a higher-quality version of the Jan 18 avalanche photo (current image is a phone photo from Loveland Ski Area). Georeference the release area, track, and deposit boundaries and export as a KMZ overlay for Google Earth. This would allow interactive comparison of the observed avalanche extent against the BFS/probabilistic model polygons in 3D terrain context, and produce presentation-quality figures showing the model chain output draped on satellite imagery.

---

## Operational Daily Mode

Design and implement a daily-run operational pipeline that provides continuous hazard assessment between UAS surveys and auto-corrects when new survey data arrives.

### Daily forward mode

- [ ] **Daily SMET extension** — append new hourly weather station records to existing SMET files instead of regenerating all ~6,636 files. Only new rows (since last run) get added. Requires tracking the last-written timestamp per file.

- [ ] **Wind loading/scouring in operational HS fill** — `smet_append.py` currently propagates cluster HS using station ΔHS only (uniform add for snowfall, proportional scale for settlement). This ignores wind redistribution between surveys, so wind-loaded lee clusters will be under-estimated and wind-scoured ridge clusters over-estimated as the gap grows. **Proposed approach: WindNinja + Winstral Sx.** For each hourly timestep, run WindNinja to downscale the station wind speed and direction to the 1 m DEM grid, producing a spatially distributed wind field. Apply the Winstral Sx shelter index (`compute_sx_vectorized()`, already implemented in `spatial_model.py`) using the WindNinja-derived local wind direction per cell to weight deposition vs. erosion. The resulting spatial transport increment is added on top of the station ΔHS each hour, with total magnitude scaled to the station wind energy (`ws²`). This replaces the RF-predicted transport field (removed — poor field performance) and the last-survey-pair proxy. Station-only path remains the fallback when WindNinja output is unavailable. **Implementation path:** (1) add a WindNinja runner wrapper that accepts station VW/DW and outputs a gridded wind field per timestep; (2) compute per-hour directional Sx from the WindNinja wind direction field; (3) extend `_propagate_hs_station_only()` to accept optional `windninja_wind_df` (gridded hourly VW/DW) and dispatch to the spatially distributed logic when provided.

- [ ] **Daily SNOWPACK incremental run** — run SNOWPACK forward one day from restart files. Extract stability snapshot (Sk38, SSI, Λ, τ_g) for the current day and generate a daily hazard assessment. Between surveys, the HS spatial distribution stays frozen at the last survey — only the stratigraphy evolves (sintering, TG metamorphism, new snow loading).

- [ ] **Daily scenario refresh** — rerun `step_scenarios` on the current snapshot date to update trigger locations and release polygons as the snowpack evolves. The daily spatial variability is station-driven only (no UAS data), so the release geometry changes are from stratigraphy evolution, not from new spatial HS information.

### Survey correction mode

- [x] **Cluster splitting after every survey** — `cluster_update` (adaptive threshold bisection) now runs automatically in `run_operational.sh` after `gap_fill` and before `smet`, so every new survey triggers a quality check and splits any clusters whose internal HS std exceeds the adaptive threshold. Order: resample → gap_fill → cluster_update → smet → SNOWPACK. `smet` then regenerates SMETs for all clusters including new children.

- [ ] **Back-correction on new survey** — when a new UAS survey arrives, the gap-filled hourly HS between the last two surveys was station-only extrapolation. The new survey reveals the actual spatial transport. Pipeline must: (1) replace gap-filled HS with observed transport for the inter-survey period, (2) regenerate SMETs only from the previous survey date forward, (3) rerun SNOWPACK from that date via restart files, (4) rebuild the Zarr for affected timesteps, (5) rerun analysis/scenarios.

- [ ] **Partial DEM resampling** — only resample the new survey to the 1 m grid. Don't reprocess the existing surveys. Currently `step_resample` regenerates everything.

### Efficiency improvements

- [ ] **SMET append-only writes** — current `step_smet` rewrites all files from scratch. An append mode would read the existing file, find the last timestamp, and write only new rows. Saves I/O and ~10 min per run on 6,636 files.

- [ ] **Zarr append** — add new timesteps to the existing Zarr store instead of rebuilding from all `.pro` files (~2 hours saved). Requires tracking which timesteps are already in the store and appending new ones from the latest `.pro` output. **Important:** on full rebuilds (new DEM, new cluster map), the Zarr must be deleted before rebuilding — stale locations from previous cluster maps cause duplicate entries and silently corrupt feature extraction (doubled layer arrays, wrong slab_thickness). Add `rm -rf slope_snowpack.zarr` to the `--clean` block in `run_full_pipeline.sh`.

- [ ] **Persist Meloche features in Zarr** — currently `compute_meloche_features` runs from scratch every `analyze` call, re-reading the full Zarr snapshot and rerunning KNN + Π₁/Π₂/A_ca for all clusters. Instead, compute the features once during the Zarr build/update step and write them as time-varying Zarr variables (e.g., `theta`, `Pi1_elastic`, `Pi2_brittle`, `A_ca_brittle`, `Lambda`, `E_slab`, `sigma_t` keyed by `[location, time]`). The `analyze` step then reads them directly from the store; `step_scenarios` gets them from the meloche CSV as now, which is populated cheaply from the Zarr slice. Benefits: (1) eliminates repeated KNN computation on every run, (2) makes Meloche features available for time-evolution analysis without re-running `analyze` for each date, (3) enables the Zarr to serve as the single source of truth for all derived snowpack variables. **Dependency:** Zarr append (#above) should land first so the feature write is incremental, not a full rebuild.

- [ ] **Feature caching with change detection** — only recompute Meloche features for clusters where HS changed by more than a threshold (e.g., 5 cm) since the last extraction. Cache the previous feature values and diff against the new Zarr snapshot. For daily forward mode, only a fraction of clusters will change meaningfully. Superseded by "Persist Meloche features in Zarr" if that item lands — persisting to Zarr is the cleaner solution.

- [ ] **Partial SNOWPACK rerun** — when correcting after a new survey, only rerun clusters whose SMET files actually changed (i.e., clusters where the gap-filled HS differs from the survey-corrected HS by more than a threshold). Use the existing `CLUSTERS_FILE` filter in `run_snowpack.sh`.

### NWP forecast mode

- [ ] **3-day forecast with NWP forcing** — extend the daily pipeline to run SNOWPACK 3 days into the future using NWP (numerical weather prediction) forecast data as forcing. Generate scenario ensembles at three snapshot dates: today (T+0), tomorrow (T+1), and day-after-tomorrow (T+2). Each snapshot produces trigger locations, release polygons, and runout probability envelopes, enabling hazard teams to plan mitigation timing and resource allocation before conditions deteriorate.

  Workflow:
  1. Ingest NWP forecast (HRRR, NAM, or GFS) for the next 72 hours — extract T, precip, wind speed/direction, RH at the study site
  2. Convert NWP fields to SMET format and append to existing cluster SMET files (after the last observed timestamp)
  3. Run SNOWPACK incrementally from now to T+72h
  4. Extract stability snapshots at T+0, T+1, T+2
  5. Run `step_scenarios` at each snapshot → 3 sets of release polygons + runout envelopes
  6. Generate a forecast summary: which paths are approaching critical instability, when, and with what confidence

  Key considerations:
  - NWP forcing has spatial resolution of 1-3 km vs the 1 m pipeline grid — needs downscaling, at minimum lapse-rate correction for temperature and orographic enhancement for precipitation
  - Wind fields from NWP are coarse — WindNinja could re-downscale NWP winds to 1 m, but this adds compute time. For forecast mode, station wind + NWP trend may be sufficient
  - **Rolling forecast convergence:** each day the forecast refreshes with the latest AWS observations and newest NWP run. Today's T+0 replaces yesterday's T+1 with 24h of real data, so forecasts converge toward reality as the target date approaches. Yesterday's T+2 (48h NWP) → today's T+1 (24h observed + 24h NWP) → tomorrow's T+0 (mostly observed). Archive the daily forecast snapshots to track convergence and calibrate forecast skill over the season
  - **Uncertainty scaling with lead time:** widen size_factor range at longer lead times to reflect decreasing forecast confidence. E.g.:
    - T+0: size_factors [0.85, 1.00, 1.15] (observed forcing, tight bounds)
    - T+1: size_factors [0.70, 0.85, 1.00, 1.15, 1.30] (mixed forcing)
    - T+2: size_factors [0.55, 0.70, 0.85, 1.00, 1.15, 1.30, 1.45] (NWP-dominated, wide bounds)
    This naturally produces wider runout envelopes at longer lead times — the hazard team sees "could reach the road at T+2" narrowing to "won't reach the road at T+0" as observations replace NWP
  - When observed data arrives (next station update or next survey), the forecast branch gets pruned and replaced with observed forcing — same back-correction workflow as survey correction mode
  - NWP data source needs to be automated (cron job pulling from NOMADS/AWS)

### Cluster management

- [ ] **Initial clustering strategy: native survey points vs. 1 m resampled grid** — decide whether to cluster on the original (irregular) UAS survey point cloud or to first resample to the 1 m DEM grid and then cluster. Both paths produce the same downstream cluster-map raster, but the spatial fidelity and compute cost differ meaningfully.

  **Option A — cluster from native survey points:**
  - **Pros:**
    - Preserves native spatial variability nuances: sharp wind-scoured ridges, lee-deposit pockets, and cornice edges are represented at survey resolution without interpolation blur
    - No interpolation artifacts — avoids smearing abrupt HS gradients across under-sampled gaps
    - Fewer input points (survey spacing typically 5–10 m equivalent density) means faster PCA + MiniBatchKMeans
    - PCA captures the true measured variance structure, not the variance of an interpolated surface
  - **Cons:**
    - Irregular point density biases cluster centroids toward densely sampled flight strips; windward ridges with sparse returns can be under-represented
    - Survey-to-survey comparisons are harder if flight-line density or point spacing varies between flights — the sample the clusterer sees changes even if the snowpack did not
    - Cluster boundary rasterization back to 1 m grid still requires nearest-neighbour assignment, so the final raster is not free of interpolation
    - Cannot directly use pixel-count weighting for SMET gap-fill without an intermediate density correction

  **Option B — resample to 1 m grid first, then cluster:**
  - **Pros:**
    - Regular grid aligns directly with DEM, all downstream 1 m raster products (depth.tif, entrainable depth, BFS map), and com1DFA input layers — no re-registration step
    - Pixel-count weighting for SMET gap-fill is unambiguous (each cell = 1 m²)
    - Consistent input geometry across surveys regardless of flight-line density or SfM point density — clustering sees the same spatial structure every time
    - Cluster map is immediately usable as a labeled raster mask; no secondary rasterization needed
    - Enables BFS crack propagation at 1 m resolution on the same grid without remapping
  - **Cons:**
    - Interpolation (IDW, kriging, or nearest-neighbour) smears sharp HS gradients — wind-slab / soft-slab transitions may be blurred, weakening the cluster-feature signal that drives PCA separation
    - Resampling a 5–10 m-equivalent survey to 1 m oversamples by 25–100×, inflating spatial autocorrelation and artificially increasing the apparent cluster count needed to explain the variance; can lead to over-clustering of essentially uniform areas
    - Substantially higher compute: clustering ~65 K cells (1 m grid, 250×260 m domain) vs. clustering ~3–6 K survey-representative points
    - Interpolated values in survey gaps carry fabricated uncertainty — the cluster model treats them as real observations

  **Current status:** pipeline uses the 1 m resampled grid (Option B) via `step_resample` → `step_cluster`. The primary open question is whether interpolation blur is masking gradient features (particularly at wind-scoured ridge crests) that would drive better cluster separation under Option A. Worth a side-by-side test on the Jan 17 survey: run both options with the same `n_clusters`, compare cluster boundary locations against the observed Jan 18 release perimeter.

- [ ] **Mid-season cluster splitting with inheritance** — when a new survey reveals that a cluster's internal HS variability exceeds the adaptive threshold (see below), split via MiniBatchKMeans bisection. One child inherits the parent's identity: same cluster ID, `.sno` restart file, `.smet`, `.pro` history, and Zarr entries — continues seamlessly. The other child is new and needs:
  1. A copy of the parent's `.sno` at the split timestamp (same stratigraphy — they were one cluster until the survey revealed heterogeneity)
  2. A new SMET generated from gap-filled hourly HS for the new child's pixel membership
  3. SNOWPACK run from the split date forward (incremental from the copied `.sno`)

  **Adaptive split threshold** — fixed `max_cluster_std_m` is too aggressive in early season (when HS is shallow, small absolute std matters more) and too permissive in deep snow. Use a relative threshold with floor and ceiling:
  ```
  threshold = clip(rel_frac × median_HS, min=min_abs, max=max_abs)
  ```
  Suggested starting values: `rel_frac=0.10`, `min_abs=0.03 m` (noise floor — UAV SfM has ~2–3 cm registration error), `max_abs=0.12 m`. At HS=30 cm → 3 cm threshold; HS=80 cm → 8 cm (same as current fixed); HS=200 cm → 12 cm (saturates). These are uncalibrated — tune against actual survey variability.

  Implementation considerations:
  - Child identity rule: larger pixel-count child inherits parent ID; smaller child gets `max_existing_id + 1`. If equal size, closest-centroid inherits.
  - Cluster map raster updated in-place; neighbour graph rebuilt only for affected clusters
  - Downstream consumers (release geometry, boundary model) pick up new cluster map automatically
  - Track split lineage in `cluster_splits.json`: `{new_id: {parent_id, split_date, split_survey_path, parent_pixels, child_pixels}}`
  - Guard against infinite splitting: if a child would fall below `min_cluster_size` (currently 4 pixels), abort the split for that cluster

- [ ] **Cluster quality monitoring** — at each new survey, compute per-cluster HS std using the adaptive threshold above. Report: how many clusters exceed threshold (split candidates), current total cluster count, mean/median cluster size, size distribution histogram. Log to `outputs/analysis/cluster_quality_YYYY-MM-DD.json`. Flag if total cluster count has grown by >50% since season start (signals aggressive splitting or poor initial clustering — tighten initial parameters next season).

- [ ] **Mid-season cluster merging** — when two neighboring clusters have converged to similar HS trajectories across all surveys to date, merge them into one. This is the counterpart to splitting and can reduce cluster count bloat from aggressive early-season splitting or from surveys that revealed heterogeneity that later homogenized (e.g., large uniform snowfall event).

  **Physical justification for mid-season merging:** HS is the *only* cluster-varying SNOWPACK input (temperature, wind, radiation, new snow all come from the shared weather station). Therefore HS trajectory similarity across the full seasonal vector genuinely implies SNOWPACK stratigraphy similarity — not just current-snapshot similarity. Two clusters that are merge candidates have been receiving nearly identical forcing throughout the season, so their `.sno` stratigraphies should be close.

  **Merge trigger — what to compare:**
  - Primary: similarity of full seasonal HS vectors in PCA space (same feature space used for initial clustering). Require distance below a merge threshold in the *original* high-dimensional space (not just the reduced PCA space — PCA compression can mask early-season differences that still matter).
  - Secondary / validation: compare `.sno` files directly. Key variables to diff: layer count, `grain_type` sequence, `shear_strength` profile, `density` profile, `temperature` profile at the WL depth. If `.sno` diff exceeds a tolerance on any of these, don't merge even if HS vectors look similar. This guards against cases where PCA compressed away a meaningful early-season divergence. Exact tolerances TBD — need to examine actual `.sno` output format and decide which variables are load-bearing for stability metrics.
  - Both criteria must pass: HS trajectory AND `.sno` profile similarity.

  **Merge procedure:**
  1. Pick the larger cluster as the "survivor" (inherits its ID)
  2. Reassign all pixels of the smaller cluster to the survivor's cluster map entry
  3. Generate merged SMET: pixel-membership-weighted average of both clusters' HS histories for all surveys to date, then rerun gap-fill for the merged pixel set
  4. Rerun SNOWPACK from season start to current date using the merged SMET (in background — see below). Since HS trajectories were similar, the merged stratigraphy should closely match either original. The survivor's `.sno` can be used as a reference to validate the rerun output before swapping.
  5. Atomic swap when background rerun completes: replace survivor's `.sno` with merged output, update cluster map, rebuild neighbour graph, update `cluster_splits.json` lineage.
  6. The smaller cluster's ID is retired (mark in lineage log, never reuse).

  **Conservative merge criteria:** merge only if the merge would reduce cluster count (don't merge if it would trigger an immediate re-split). Require similarity across the full seasonal vector, not just recent surveys — early-season divergences that later converged are still reflected in the stratigraphy. A cluster pair that diverged in December and converged in February should not be merged: their WL stratigraphy from December is different, and that's exactly what stability metrics depend on.

- [ ] **Background reclustering** — periodic full recluster + SNOWPACK rerun from scratch, run as a background process so it doesn't block the operational forecast pipeline. When complete, swap the cluster map and all associated files atomically. Forecasts continue running on the old cluster map until the swap.

  **Physical justification:** SNOWPACK is deterministic given the same SMET input — a from-scratch run from season start to the current date produces the same output as incremental updates to the same date, at higher compute cost. So a background full recluster sacrifices no accuracy vs. the existing incremental approach, and can fix problems that splits and merges can't: (1) initial clustering that was poorly positioned relative to the season's actual HS gradients, (2) accumulated cluster count bloat from many splits.

  **Trigger conditions (any one suffices):**
  - Total cluster count has grown by >50% since season start
  - Mean cluster size has fallen below `min_cluster_size × 2` (clusters too small for reliable SNOWPACK)
  - Cluster quality monitoring shows >20% of clusters exceeding the adaptive split threshold for two consecutive surveys (systematic degradation, not isolated splits)
  - Complete melt-out event (HS ≈ 0 across the domain) — stratigraphy is reset, nothing to lose
  - Explicit operator trigger (e.g., `run_full_pipeline.sh --recluster`)

  **What the background process does:**
  1. Re-run initial clustering on the full HS matrix (all surveys to date) with the same PCA + MiniBatchKMeans pipeline
  2. Generate new SMETs for all new clusters (pixel-membership-weighted HS histories)
  3. Run SNOWPACK from season start to current date for all new clusters (this is the expensive step — same wall time as initial pipeline, ~2 hrs)
  4. Validate output: check that new cluster map stability metrics (median Sk38, Pi1) are consistent with old map within some tolerance. If validation fails, abort swap and alert.
  5. Atomic swap: replace `cluster_map.npy`, all `.sno`/`.smet`/`.pro` files, Zarr store. Archive old files with a timestamp suffix.
  6. Update `cluster_splits.json` to mark the recluster event (all lineage history from before the recluster is archived but no longer active).

  **Timing:** prefer low-hazard periods or overnight. If the server is idle (no active SNOWPACK runs, no forecast in progress), the background recluster can use all available cores. Implement a lock file so the operational pipeline and background recluster don't run SNOWPACK simultaneously on the same slope directory.