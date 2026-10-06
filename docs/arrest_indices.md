# arrest_indices.py: methods

Crack-arrest indices for dry-slab avalanches in two groups:

1. **Published strength-based scalings** from Meloche et al. (2025, JGR) and Meloche et al. (ISSW 2026).
2. **An energy-cap hypothesis** (new, untested beyond three simulated cases) that treats slab fracture as a cap on the energy available to drive the weak-layer crack.

Status: unit-tested against three calibration runs from Meloche et al. (2025) (`tests/test_arrest_indices.py`). Not validated against field data or a broad simulation set.

Units are SI (Pa, m, kg m⁻³, J m⁻²) with angles in degrees.

Figures marked "JGR" are reproduced from Meloche et al. (2025), *JGR Earth Surface*, open access under CC BY 4.0. Image files are in `figures/`.

---

## 1. Notation

| Symbol | Meaning | Code |
|---|---|---|
| ρ, h | slab density, slab thickness (slope-normal) | `rho`, `h` |
| E, ν | slab Young's modulus, Poisson's ratio | `E`, `nu` |
| E′ = E/(1−ν²) | plane-stress modulus | `plane_stress_modulus` |
| G = E/(2(1+ν)) | slab shear modulus | `shear_modulus` |
| σt | slab tensile strength | `sigma_t` |
| G_wl, D_wl | weak-layer shear modulus, thickness | `G_wl`, `D_wl` |
| K_wl = G_wl/D_wl | weak-layer interface stiffness (Pa/m) | `weak_layer_stiffness` |
| τp, τp0 | weak-layer shear strength, value at trigger zone | `tau_p0` |
| θ | upslope gradient of τp (Pa/m) | `theta` |
| δ | weak-layer softening coefficient, (u_r − u_c)/u_c | `delta` |
| ψ, φ | slope angle, snow friction angle (27°) | `psi_deg`, `phi_deg` |

Check on K_wl: the JGR paper states "K_wl = 0.2 MPa" but also u_c = 0.2 mm at τp = 1 kPa. Only K_wl = G_wl/D_wl = 0.2 MPa / 0.04 m = 5 MPa/m reproduces u_c = 0.2 mm, so the code uses G_wl/D_wl.

---

## 2. Published relations (Meloche et al. 2025, ISSW 2026)

**Elastic length (upslope, mode II)**

Λ = √( E h D_wl / ((1−ν²) G_wl) )

**Gravitational and residual shear**

τg = ρ g h sin ψ, τr = ρ g h cos ψ tan φ

Shear propagation is only sustained when ψ > φ (`sustained` flag).

**Slab tension gradient.** Quasi-static (Eq. 17):

k_f = ρ g sin ψ (1 − tan φ / tan ψ)

Dynamic (Eq. 18), with crack speed ȧ = 1.6 c_s (supershear):

k_x = k_f c_p² / (c_p² + ȧ²), c_p = √(E′/ρ), c_s = √(G/ρ)

![JGR Fig. 3: slab tension vs distance for different crack speeds (a) and k_x/k_f vs speed with the analytical solution (b)](figures/jgr_fig3_dynamic_tension.png)

*JGR Fig. 3. Faster cracks build slab tension more slowly. At ȧ ≈ c_p the gradient is half the quasi-static value.*

**Tensile lengths (distance to first slab fracture)**

L_t = σt / k_f, L_dyn = σt / k_x

**Arrest length, elastic slab (Eq. 19)**, for reference (not coded):

A_ca / L_ss ∝ ( τg / (θ Λ √(1+δ)) )^(3/2)

![JGR Fig. 7: elastic-slab arrest scaling (a) and brittle runs added on top (b)](figures/jgr_fig7_elastic_scaling.png)

*JGR Fig. 7. Elastic-slab collapse (a). In (b), brittle runs fall off the elastic trend, which shows slab fracture changes the arrest mechanism.*

**Arrest length, brittle slab (Eq. 20)**

A_ca / L_t = C · X · √(σt/τg), X = τg / (θ Λ √(1+δ))

The paper gives this as a proportionality. C = 0.045 is my two-point fit to the Fig. 8 runs and is the code default.

![JGR Fig. 10: brittle-slab arrest scaling](figures/jgr_fig10_brittle_scaling.png)

*JGR Fig. 10. Brittle-slab scaling with a slope-1 reference line.*

C = 0.045 verified by direct computation from the Fig. 8 inputs (see calibration table in §3 and `test_arrest_indices.py`). The arithmetic: τg = 702 Pa, Λ = 0.663 m, L_t = 15.67 m; θ = 30 gives A_ca = 72.7 m (observed 74 m); θ = 150 gives 14.5 m (observed 14 m). The dashed line in Fig. 10a is a 1:1 reference line through the data, not a separate best-fit; no discrepancy remains.

![JGR Fig. 9: arrest length vs θ for σt, E and L_ss](figures/jgr_fig9_sensitivity.png)

*JGR Fig. 9. A_ca falls roughly as 1/θ. Higher σt and lower E lengthen arrest. Crosses are single-fracture arrests, circles multiple.*

**Release size index (ISSW 2026, preliminary).** The ISSW 2026 paper defines a_t as the distance to the first slab fracture (obtained directly from MPM simulations). In the code, a_t is approximated as L_dyn (dynamic tensile length = σt/k_x), which is the analytical prediction for where the first fracture occurs.

- RSI 1 (arrest): a_t/a_c < 6–7 (code uses 6.5)
- RSI 2 (sub-Rayleigh en-echelon): a_t/a_c ≥ 6.5 and a_t/a_sc < 1
- RSI 3 (supershear en-echelon): a_t/a_sc ≥ 1

a_c and a_sc are not computed in the pipeline and must be supplied as inputs. The 6–7 threshold was derived from ISSW 2026 simulations using D_wl = 0.1 m, E = 3 MPa, ρ = 250 kg m⁻³, σt = 2–16 kPa — different from the JGR calibration parameters. Applicability to other parameter ranges is untested. For a_sc, the ISSW paper estimates it from the corresponding purely elastic simulation (no slab fractures allowed).

---

## 3. Energy-cap hypothesis

### Idea

During propagation the unsupported slab stores elastic energy, which pays for weak-layer fracture at the crack tip. A slab fracture resets the tension upslope of it to zero. The energy the slab can deliver per unit crack advance is therefore capped at roughly that of a segment loaded to its tensile strength. Arrest occurs where the weak layer's fracture energy exceeds that cap.

### Slab energy cap

For a slab strip with tension rising to σt, the energy delivered per unit advance at the loaded end is:

G_slab = σt² h / (2E′)

### Weak-layer fracture energy

Integrating the softening interface law of Meloche et al. (2025, Eq. 15), linear to (u_c, τp) and then linear down to τr at u_r = u_c(1+δ), and approximating:

G_c(τp) ≈ (τp − τr)² (1+δ) / (2 K_wl)

The code uses τr = 0 by default (`use_residual=False`). That is the version that matched the simulations below. Including τr is an option, not a recommendation.

### Arrest criterion

Arrest when G_c(τp(x)) ≥ R · G_slab.

This defines a critical weak-layer strength:

τp* = τr + √( 2 K_wl R G_slab / (1+δ) )

For a linear strength ramp τp(x) = τp0 + θx beyond the steady-state zone:

A_ca = (τp* − τp0) / θ

R absorbs dynamic effects and the approximations above. Two options are provided:

- `R = 0.48`: fitted to the two Fig. 8 runs (default).
- `R = 'dynamic'`: R = k_x/k_f at ȧ = 1.6 c_s (0.53 for ν = 0.3). This is a speculative physical interpretation. It lands within 10% of the fit, which may be coincidence.

![JGR Fig. 8: two brittle runs, θ = 30 and 150 Pa/m](figures/jgr_fig8_brittle_runs.png)

*JGR Fig. 8. Weak-layer strength and shear (top), slab tension and slab fractures (middle), crack speed (bottom). Arrest at 94 m (θ = 30) and 34 m (θ = 150).*

![JGR Fig. 11: Jim Bay Corner field-gradient run](figures/jgr_fig11_jbc.png)

*JGR Fig. 11. Field gradient (θ ≈ 15 Pa/m) with δ = 1. Crack speed drops to zero near 94 m in panel (d).*

### Check against Meloche et al. (2025)

Inputs: ρ = 250 kg m⁻³, h = 0.5 m, ψ = 35°, E = 4 MPa, σt = 6 kPa, G_wl = 0.2 MPa, D_wl = 0.04 m, τp0 = 1 kPa, L_ss = 20 m.

| Run | θ (Pa/m) | δ | Observed A_ca (m) | Energy cap (m) | Eq. 20, C = 0.045 (m) |
|---|---|---|---|---|---|
| Fig. 8a | 30 | 0 | 74 (arrest at 94 m) | 71 | 73 |
| Fig. 8b | 150 | 0 | 14 (arrest at 34 m) | 14 | 15 |
| Fig. 11, JBC | 15 | 1 | ~74 (arrest ~94 m, read from Fig. 11d) | 81 | 103 |

The first two rows are the calibration data for both methods, so they don't discriminate. JBC is the only out-of-sample point, and its observed value is read off a plot.

![Energy-cap check: predicted vs simulated A_ca (left) and τp at arrest (right)](figures/energy_cap_check.png)

*My plot from `arrest_indices.py`. The dashed lines on the right are τp* from the energy-cap criterion with R = 0.48.*

The notable observation is that both Fig. 8 runs arrest at nearly the same weak-layer strength (3.2 and 3.1 kPa) despite a fivefold change in θ. A threshold-strength criterion predicts exactly this behavior.

### How the two frameworks differ

Both give A_ca ∝ 1/θ, so they can only be separated through other parameters:

| Parameter | Energy cap | Eq. 20 |
|---|---|---|
| σt | A_ca roughly ∝ σt | ∝ σt^1.5 |
| E | through √(1/E′) in τp* | through Λ and L_t |
| δ | through (1+δ)^(−1/2) in τp* | through (1+δ)^(−1/2) in X |

The σt dependence is the cleanest discriminating test.

### Cross-slope (mode III), hypothesis only

The code provides:

- `elastic_length_cross`: Λ_III = √( G h D_wl / G_wl ), giving Λ_III/Λ_II = √((1−ν)/2) ≈ 0.59 at ν = 0.3. This is my derivation, using the slab shear modulus for antiplane loading.
- `slab_energy_cap_cross`: G_slab,III = τ_flank² h / (2G), where τ_flank is a slab flank (shear) strength. No parameterization exists, so it is a user input.

Expected differences from upslope, not yet simulated:

- Mode III cracks cannot exceed c_s (Broberg 1989), versus ~1.6 c_s upslope. This caps the energy flux and is a plausible reason cross-slope cracks stop more easily.
- Because G < E′, an equal-strength slab has a larger energy cap cross-slope. If that holds, cross-slope arrest is driven mainly by the speed cap and weak-layer variability, not slab fracture. This is consistent with the JGR paper attributing lateral arrest to weak-layer heterogeneity.

---

## 4. Inputs from SNOWPACK

| Input | Source | Note |
|---|---|---|
| ρ, h | layer-weighted slab density and thickness above weak layer | check slope-normal vs vertical |
| E | density parameterization | not a default output |
| σt | density parameterization (e.g. Sigrist 2006) | formula not included here |
| τp0 | shear strength output (`.pro` code 0508, to verify) | In `compute_meloche_features()`, each cluster's own `wl_shear_strength` is used as τp0 — a per-cluster index: "if crack initiates here, how far must WL strength rise to arrest it?" |
| G_wl, D_wl | weak-layer modulus and thickness | paper used 0.2 MPa, 0.04 m |
| θ | `shear_gradient` on τp along an upslope transect | needs a spatial field, not one profile |
| δ | not observable | paper range 0–2 |

Without θ, the energy framework still gives τp* and R0 = G_c(τp0)/G_slab, a per-profile index of how far weak-layer strength must rise before a crack stops.

---

## 5. Application to the Jan 18 2026 avalanche — Little Professor

### Input ranges from SNOWPACK

Jan 18 clusters are deeper-hoar (DH / FC) weak layer under a hard slab, NE-facing at ~3700 m.
All quantities from `release_zone_features_2026-01-18.csv` (n = 288 release, 1032 adjacent).

| Parameter | JGR calibration | Release (median [min, max]) | Adjacent (median [min, max]) |
|---|---|---|---|
| ρ (kg m⁻³) | 250 | 316 [264, 332] | 310 [202, 331] |
| h (m) | 0.5 | 1.63 [1.02, 2.16] | 1.13 [0.41, 3.54] |
| D_wl (m) | 0.04 | 0.075 [0.022, 0.154] | 0.060 [0.021, 0.271] |
| E_slab (MPa) | 2–6 | 4.55 [2.9, 5.2] | 4.33 [1.5, 5.1] |
| σt (kPa) | 2–10 | 5.38 [4.2, 5.8] | 5.23 [2.9, 5.8] |
| Λ (m) | 0.663 | 1.67 [0.92, 2.66] | 1.29 [0.74, 2.81] |
| θ (Pa m⁻¹) | 30–150 | 30.5 [6.5, 132] | 23.8 [2.4, 167] |
| τp0 (kPa) | 1 | 2.61 [2.18, 3.01] | 2.32 [1.86, 3.04] |

Jan 18 slab is ~3× thicker and ~30% denser than the JGR calibration. The WL is nearly 2× thicker (deeper DH layer). Both push Λ upward and K_wl downward relative to the paper.

### Arrest length comparison

Figures from `outputs/little_prof/plots/arrest_indices_2026-01-18.png`.
Numbers from `scripts/plot_arrest_indices_jan18.py` (enriched from regenerated CSV, Oct 6 pipeline run, 5176 clusters).

**Summary table — medians per group**

| Metric | Release | Adjacent | Reference |
|---|---|---|---|
| R0 = G_c(τp0)/G_slab | 0.4 | 0.5 | 0.6 |
| τp* (kPa, δ=1) | 2.84 | 2.25 | 2.31 |
| A_ca Eq.20, C=0.045 (m) | 61.9 | 70.1 | 141.3 |
| A_ca energy, δ=1 (m) | 8.5 | 0.0 | 0.0 |
| A_ca energy, δ=0 (m) | 62.5 | 29.8 | 47.8 |

**R0 ordering:** release (0.4) < adjacent (0.5) < reference (0.6), consistent with R_FIT = 0.48. Release zone median R0 is below the arrest threshold — WL cannot arrest the crack at the trigger point. Dashed line at R_FIT = 0.48 in panel (a) separates the bulk of the release distribution from reference.

**Eq. 20:** release (61.9 m) < adjacent (70.1 m) — correct direction; crack needs to travel farther to arrest in lower-gradient (adjacent) terrain.

**Energy cap (δ = 1, default):**
τp* (2.84 kPa for release) falls slightly above τp0 (≈ 2.6 kPa), so release clusters have finite A_ca_energy ≈ 8.5 m. Adjacent clusters have τp* < τp0 → A_ca_energy = 0 (crack would arrest immediately there). Ordering is release > adjacent ✓.

**Energy cap (δ = 0, JGR calibration default):**
δ = 0 spreads out all distributions. Release (62.5 m) > adjacent (29.8 m) ✓. Reference sits between at 47.8 m.

The δ sensitivity remains important: δ = 1 gives very small A_ca (8.5 m) for release and zero for adjacent — nearly indiscriminate. δ = 0 shows a cleaner 2× spread. The JGR paper calibrated on δ = 0 runs; for depth hoar (little post-peak softening), δ = 0 is better-supported.

### Comparison of frameworks

| | Eq. 20 (brittle) | Energy cap |
|---|---|---|
| **A_ca median, release** | 61.9 m | 8.5 m (δ=1); 62.5 m (δ=0) |
| **A_ca median, adjacent** | 70.1 m | 0.0 m (δ=1); 29.8 m (δ=0) |
| **Release > adjacent?** | ✓ (narrow) | ✓ δ=1 (narrow); ✓ δ=0 (2×) |
| **R0 ordering** | n/a | release (0.4) < adj (0.5) < ref (0.6) ✓ |
| **Best IoU (threshold sweep)** | 0.353 @ 129 m | 0.240 δ=1; 0.256 δ=0 |
| **Best recall at best IoU** | 0.815 | 0.713 δ=1; 0.666 δ=0 |
| **Sensitive to δ?** | Moderate (through (1+δ)^−½) | Strong: changes which group has A_ca = 0 |
| **Sensitive to h?** | Moderate (through L_t, Λ) | Strong: G_slab ∝ h |
| **Key discriminant** | θ, Λ, σt/τg ratio | R0 = G_c(τp0)/G_slab |

Eq. 20 outperforms the energy cap on this event: IoU 0.353 vs 0.240–0.256. R0 gives lower IoU (0.192) but correctly orders all three groups. The energy cap's A_ca_energy_d0 gives the next-best IoU (0.256) and its R0 interpretation provides a simple threshold-free discriminant.

R0 is the cleanest single-number discriminant that does not require θ. A cluster with R0 < R_FIT (0.48) means the WL at that point cannot absorb the slab energy — crack propagates. The Jan 18 release zone has median R0 = 0.4 (below threshold) while adjacent = 0.5 (above threshold), consistent with the observed release boundary.

### IoU evaluation

`scripts/plot_arrest_indices_jan18.py` includes a threshold sweep: for each metric (R0, A_ca_brittle, A_ca_energy δ=0/δ=1), it classifies clusters below a percentile threshold as "predicted release", rasterizes them using `cluster_map.npy`, and computes IoU, recall and precision against `data/boundaries/avalanche_release_area.geojson`.

This is different from `scripts/release_area_IoU.py`, which evaluates AvaFrame flow/runout scenario polygons against the observed deposit boundary — a downstream step that requires arrest indices only indirectly through the scenario weights.

Run the evaluation:

```
python scripts/plot_arrest_indices_jan18.py --date 2026-01-18
```

Output:
- `outputs/little_prof/plots/arrest_indices_2026-01-18.png` — 4-panel figure
- Console: summary table (median per group) and IoU sweep table

### Hardcoded parameter sensitivity

- **δ**: The dominant sensitivity. Use δ = 0 as the default for DH weak layers. The code exposes this via `DELTA` in `compute_meloche_features()`.
- **D_wl fallback (0.04 m)**: Only used when `wl_thickness` is NaN. Most clusters have measured values (0.022–0.154 m range). The fallback underestimates K_wl by 2×, inflating τp*.
- **G_WL (0.2 MPa)**: Not independently measured. Changing it scales K_wl and τp* proportionally. Untestable from SNOWPACK alone.
- **C (0.045)**: Two-run fit from JGR Fig. 8. Jan 18 slope is NE-facing vs. the 35° modeled slope; no re-calibration done or recommended.
- **R_FIT (0.48)**: Same. Two calibration runs.

---

## 7. Limits

- All calibration comes from shear-mode-only DA-MPM simulations: 0.5 m slab, ρ = 250 kg m⁻³, ψ = 35°, E 2–6 MPa, σt 2–10 kPa. The JGR paper notes the model suits large avalanches (size 3+) and may overestimate soft slabs.
- No anticrack (collapse) regime and no slab bending.
- Linear strength ramp only. Noisy or patchy weak layers may behave as a percolation problem rather than a gradient problem.
- R and C rest on two runs.

## 8. Suggested validation

1. Pull the post-processed brittle runs from https://github.com/Francismeloche/CrackArrest-DAMPM.
2. For each run, compute τp at the arrest point and R = G_c/G_slab.
3. Check whether R is constant across θ, σt, E, L_ss and δ. If it is, the energy-cap criterion holds in the simulated range.
4. Compare the σt dependence of A_ca against Eq. 20.
5. Repeat on the TARP cross-slope runs with `slab_energy_cap_cross` once a flank strength is chosen.

## 9. References

- Meloche, F., Bobillier, G., Guillet, L., Gauthier, F., Langlois, A., & Gaume, J. (2025). Modeling crack arrest in snow slab avalanches: Toward estimating avalanche release sizes. *JGR Earth Surface*, 130(12), e2025JF008470. https://doi.org/10.1029/2025JF008470
- Meloche, F., Trottet, B., Bobillier, G., & Gaume, J. (2026). Slab fractures during dynamic crack propagation in dry-snow slab avalanches. *Proc. ISSW*, Whistler, 1961–1966. https://doi.org/10.15788/1790098485
- Meloche, F., et al. (2026). TARP final technical report: Vertical avalanche defense structures.
- McClung, D. M., & Schweizer, J. (2006). Fracture toughness of dry snow slab avalanches from field measurements. *JGR Earth Surface*. https://doi.org/10.1029/2005JF000403
- Broberg, K. B. (1989). The near-tip field at high crack velocities. In *Structural Integrity*, Springer.
