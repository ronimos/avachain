"""
model_params.py — every parameter, switch and calibration knob of the
release-area model, in one place (Ron, 2026-11-11).

Import flat and read at CALL time:

    import model_params as mp
    if mp.USE_SLAB_TENSILE_GATE: ...

Never `from model_params import X` and never `def f(x=mp.X)`: both freeze the
value when the importing module loads, so a slope's [calibration]
(calibration.apply), a CLI flag or a test's monkeypatch.setattr(mp, 'X', ...)
would no longer reach it. Functions take `x=None` and resolve `mp.X` inside.

Names follow ~/avalanche-release-areas src/release_areas/config.py, which is
the standalone's central parameter set; the two must hold the same values
(CLAUDE.md, "The release-area model must match ~/avalanche-release-areas").
A per-slope calibration changes values only through the slope TOML
[calibration]; never edit a value here to calibrate.

What is NOT here, on purpose:
  * coefficients of published relations (G_ACC, RHO_REF, E_RHO_*, ST_RHO_*,
    VH2016_*, JJ1990_*, SCH2004_*, RHO_ICE) and the relation registries
    (E_RELATIONS, ST_RELATIONS, K_IC_RELATIONS, FACET_SOURCES): they sit next
    to the formula they belong to in arrest_indices.py;
  * data-format and plotting constants (WL_TYPES, SLAB_TYPES, RC_VAR,
    RC_UNDEFINED, EVENT_DATE, GROUP_*) in snowpack_analysis.py;
  * pipeline / operational settings (paths, stations, MIN_RELEASE_DEPTH_M,
    scenario sizes, snapshot time) in config.py, the slope TOML and
    analysis_pipeline.py.

This module imports nothing from the project, so importing it is cycle-free
(config.py may import it).
"""

# ============================================================================
# MATERIAL CONSTANTS — Meloche et al. (2025) JGR Table 1
# ============================================================================
G_WL          = 0.2e6   # Pa; weak-layer shear modulus (Reiweger et al. 2010; Gaume et al. 2018)
NU            = 0.3     # slab Poisson's ratio
PHI_DEG       = 27.0    # degrees; snow friction angle
DELTA         = 1.0     # Meloche softening coefficient δ in Eq. 20 (their Table 1)
D_WL_FALLBACK = 0.04    # m; weak-layer thickness used only when wl_thickness is NaN
L_SS          = 20.0    # m; steady-state propagation length, Meloche et al. (2025)

# ============================================================================
# MATERIAL RELATIONS (registries in arrest_indices.py)
# ============================================================================
E_RELATION         = 'vanherwijnen2016'   # slab E(rho) behind E_eff; or 'project_fit'
E_RELATIONS_EXTRA  = ('project_fit',)     # also emitted as E_eff__<name>
ST_RELATIONS_EXTRA = ('jamieson1990',)    # extra sigma_t(rho), emitted as sigma_t_*__<name>

# Slab fracture toughness K_Ic(rho). SLAB_K_IC overrides the relation with a
# constant (Pa m^0.5); None = use K_IC_RELATION. 'borstad2013' is NOT
# implemented -- its published equation has not been sourced and quoted back
# for checking.
#
# The density ranges in arrest_indices.K_IC_RELATIONS are a GUIDELINE, not a
# limit (Ron, 2026-11-09). They are what each relation was fitted over;
# Schweizer et al. (2004) is 80-300 kg/m^3 from Swiss beam tests. Little
# Professor is a Colorado high-alpine, wind-affected snowpack. On Jan 18 2026,
# 82% of the start-zone clusters (avachain) have a mean slab density above
# 300, and every one has at least one intact slab layer outside 80-300.
# Clamping evaluated those layers at the range edge and understated K_Ic of
# dense slabs: the median ligament K_Ic is 1318 clamped against 1650 Pa m^0.5
# extrapolated (+25%). Kirchner et al. (2000), fitted to 540, has 3% of
# clusters out of range. So since 2026-11-10 (Ron) the power law is
# EXTRAPOLATED past its range by default (K_IC_RANGE_MODE); n_rho_clamped now
# counts the extrapolated layers. The trigger toughness gate
# (USE_TRIGGER_TOUGHNESS) reads the ligament K_Ic, so this moves which
# triggers can start a release.
SLAB_K_IC       = None
K_IC_RELATION   = 'schweizer2004'
K_IC_RANGE_MODE = 'extrapolate'   # 'extrapolate' | 'clamp' | 'nan'

# Edge-crack (ligament) bound on slab strength: a HYPOTHESIS (see
# arrest_indices.py, "Ligament bound").
USE_LIGAMENT_BOUND = True
LIGAMENT_MODEL     = 'elhaddad'   # or 'lefm' for the uncorrected form
SELF_CONSISTENT_A0 = False        # True evaluates a0 at F(a0/h) iteratively
DMAX_FACTOR        = 1.0          # d_max = DMAX_FACTOR * grain size (both m)
LIGAMENT_VARIANTS = (
    ('kirchner2000',     'kirchner2000',  1.0),
    ('schweizer2004_d1', 'schweizer2004', 1.0),
    ('schweizer2004_d2', 'schweizer2004', 2.0),
)

# ============================================================================
# FACET STRENGTH KNOCKDOWN (layered-slab sigma_t, arrest_indices.aggregate_slab)
# ============================================================================
# Faceted snow is ~half as strong in tension as rounded snow at equal density
# (Jamieson & Johnston 1990): sigma_t,i = sigma_t,RG(rho_i) (1 - a f_i), with
# a = FACET_STRENGTH_FACTOR (per slope: [calibration] facet_a).
# FACET_SOURCE = 'grain_class' (Ron, 2026-10-27) sets f = 1 for J&J's group
# II: primary FC/DH/FCxr, or a PP/DF/RG primary with a faceted secondary
# (e.g. 341, "both faceting and rounding"), and 0 otherwise. Code 341 is 24%
# of Jan 18 slab thickness inside the release, 12% outside.
# 'grain_class_primary' ignores the secondary form (163ea0f). 'sphericity' is
# the earlier continuous f = 1 - sphericity, which under-counts FCxr: over the
# Jan 18 start-zone slabs it gave f 0.35 inside the release / 0.52 outside,
# against 0.50 / 1.00 by primary class. FACET_SP_REF applies to 'sphericity'
# only (None = 1 - sphericity outright; rounded layers at ~0.86 keep f ~0.14).
# The legacy Eq. 20 arrest reads the plain sigma_t column, which has no
# knockdown, so this moves sigma_t_mean/_wl/_ligament (and A_ca_* built on
# them), the energy mode and the slab tensile gate.
# 2026-11-13 (Ron): 'grain_class_primary' is the default. The mixed RG/FC
# forms keep RG strength: they are 51% of the top 0.45 m of the Jan 18 release
# slab (pure RG 6%), so the secondary rule knocked the shallow slab down as if
# it were facets and its S_t came out 0.44x a uniformly thinned slab; with the
# primary rule it equals it (start-zone median f 0.97 -> 0.79).
FACET_STRENGTH_FACTOR = 0.5
FACET_SOURCE          = 'grain_class_primary'
FACET_SP_REF          = None
# Grain classes counted as faceted (SNOWPACK codes: 4 FC, 5 DH, 9 FCxr).
# Jamieson & Johnston (1990), Microstructure: group II is "layers of faceted
# grains and of layers of grains which showed both faceting and rounding"
# (8 of their 23 group II layers), group I "new snow, partly settled grains or
# rounded grains", both with no evidence of melting. So FCxr is faceted, and
# so is a group I primary (PP/DF/RG) with a faceted secondary form, e.g. 341.
FACETED_GRAIN_CLASSES = (4, 5, 9)
MIXED_PRIMARY_CLASSES = (1, 2, 3)

# ============================================================================
# WEAK-LAYER DETECTION (snowpack_analysis.split_wl_slab)
# ============================================================================
# Basal FCxr (9xx, rounding facets) under the FC/DH sequence counts as part of
# the basal weak layer (Ron, 2026-10-25). Without it, 251 of 732 Jan 18 crown
# clusters had no weak layer: 1-4 basal 990 elements (median 4.7 cm) sat under
# FC and the first-element test failed. Only a run of FCxr AT THE GROUND is
# admitted; FCxr above the first FC/DH element still ends the block, so every
# profile with an FC/DH base splits exactly as before.
BASAL_FCXR_IS_WL = True

# Which weak layer the release model uses (snowpack_analysis.select_wl_slab).
# 'basal' (default, operational): the basal FC/DH block above. It stays the
# operational choice because the forecast cares about the larger avalanches
# (Ron, 2026-11-12). 'threshold_sum' is an ADDITION, the layer of concern by
# the threshold-sum approach (Schweizer & Jamieson 2007, Cold Reg. Sci. Technol.
# 47, 50-59; Monti et al. 2014): the interface with the most "lemons". On the
# Jan 18 re-referenced replay it put the bed at a median 0.53 m against the
# observed 0.44 m (basal: 0.82 m). CLI: analyze --wl-selection.
WL_SELECTION = 'basal'
WL_SELECTIONS = ('basal', 'threshold_sum')

# Threshold-sum lemons (Schweizer & Jamieson 2007, Table 3). One lemon each:
TSA_GS_DIFF_MM   = 0.75    # grain size difference across the interface >= (mm)
TSA_HH_DIFF      = 1.7     # hand-hardness index difference >= (1 F, 2 4F, 3 1F, 4 P, 5 K)
TSA_DEPTH_M      = (0.18, 0.94)  # weak-layer depth within (m)
TSA_PERSISTENT   = (4, 5, 6, 9)  # weak-layer primary grain class FC, DH, SH, FCxr
TSA_WL_GS_MM     = 1.25    # weak-layer grain size >= (mm)
TSA_WL_HH        = 1.3     # weak-layer hand-hardness index <=

# ============================================================================
# SHEAR-STRENGTH GRADIENT θ AND THE ARREST-INDEX FLOOR (snowpack_analysis)
# ============================================================================
# How θ is estimated from the per-cluster τp field. θ is a property of the LAG
# it is measured at, not of the snowpack.
#   'knn'        mean |Δτp|/d over the k nearest cluster centroids
#                (theta_knn, k = 6). Reproduces every pre-2026-10-09 CSV.
#   'plane_fit'  local least-squares plane through τp over all centroids within
#                THETA_FIT_RADIUS_M, θ = |∇τp|. Estimates the slope-scale ramp
#                that Meloche's θ represents, instead of the local noise.
#
# Measured on Jan 18: start-zone clusters are 3.0 m across and the k = 6
# neighbourhood averages 3.3 m, so 'knn' samples θ inside the 0.5-10 m band the
# paper treats as local noise. The τp variogram gives θ(d) = 3.87 + 72/d Pa/m,
# i.e. at 3.3 m θ is 15% slope-scale trend and 85% local noise. A_ca ∝ 1/θ, so
# the two estimators differ by ~3.3x in A_ca (median θ 25.19 vs 7.28 Pa/m).
#
# Default 'plane_fit' since 2026-10-09, argued from the calibration scale, not
# the fit: Meloche's θ is the gradient of a linear ramp over L_ss, so the
# estimator's neighbourhood should be L_ss, not whatever the cluster spacing
# happens to be. Do NOT pick the estimator by Jan 18 IoU — changing θ moves Π₁,
# hence the trigger ranking and the whole filter chain.
#
# Open caveat: under 'plane_fit' 97.2% of Jan 18 start-zone clusters fall below
# Meloche's 20 Pa/m validity floor, against 36.2% under 'knn'. That is the
# strongest objection to the θ-based arrest route as a whole, and it is NOT an
# argument for reverting — 'knn' only looked compliant because it was measuring
# noise. See ~/avalanche-release-areas/docs/release_area_methods.md §5.
THETA_ESTIMATOR          = 'plane_fit'   # 'knn' | 'plane_fit'
# The radius IS L_ss, and that identity is the argument for the default: θ is
# the gradient of a ramp over L_ss, so the neighbourhood must be L_ss.
THETA_FIT_RADIUS_M       = L_SS
THETA_FIT_MIN_NEIGHBOURS = 8      # minimum centroids in radius, else θ = NaN
THETA_MIN                = 1e-6   # Pa/m; below this there is no usable gradient

# Pa; below this the arrest scaling laws are not evaluated and the cluster gets
# no arrest indices. Deliberately ABOVE the 40 Pa trigger gate (MIN_TAU_G),
# which leaves a 40-50 Pa band that passes trigger selection with no indices —
# kept to match the standalone. UNSUPPORTED there and here: Meloche et al.
# never varied tau_g (h=0.5 m, rho=250, psi=35 fixed in all four campaigns, so
# tau_g = 703 Pa is the only simulated value). The paper's size caveat is
# "size 3 or more", not D2+. See THETA_VALID_* for the thresholds the paper
# does give. Pinned by tests/test_snowpack_analysis.py (no Jan 18 cluster
# falls in the band).
TAU_G_FEATURE_FLOOR = 50.0

# Validity range of the arrest scaling laws in θ, from Meloche et al. (2025).
# The only quantitative propagation thresholds the paper gives:
#   p.12 "Shear strength gradients of more than 20 Pa m-1 were needed to obtain
#         crack arrest within the PST simulated length."
#   p.13 runs with θ < 20 Pa/m were REMOVED from their analysis (no arrest).
#   p.10 "When the shear strength gradient is larger than around 5,000 Pa m-1,
#         the gradient is too large, causing crack arrest after L_ss."
# Recorded, not enforced.
THETA_VALID_MIN = 20.0      # Pa/m
THETA_VALID_MAX = 5000.0    # Pa/m

# ============================================================================
# ARREST-INDEX FITS AND SLAB-FRACTURE CHOICES (arrest_indices)
# ============================================================================
C_FIT = 0.045  # Eq. 20 prefactor, fitted from Meloche et al. 2025 Fig. 8 (two runs); see docs/arrest_indices.md. Per slope: [calibration] c_fit
R_FIT = 0.48   # energy ratio, fitted from Meloche et al. 2025 Fig. 8 (two runs); see docs/arrest_indices.md

# Crack-speed regimes. Upslope (mode II) cracks run supershear; mode III
# (antiplane) cracks cannot exceed the shear wave speed.
#
# On the 1.6: Meloche et al. (2025) write the supershear speed as 1.6 c_s but
# identify it with c_p -- "A horizontal dashed line is set at 1.6 c_s ~= sqrt(E'/rho),
# which is the supershear speed (Trottet et al., 2022)" -- and plot "until
# c_p ~= 1.6 c_s". So 1.6 is their rounding of c_p/c_s = sqrt(2/(1-nu)) = 1.690
# at nu = 0.3. Taking adot = c_p exactly gives k_x/k_f = 1/2, which is what the
# paper's own Fig. 3 caption states; the literal 1.6 gives 0.527.
# We keep the literal 1.6 so published figures reproduce; cp_over_cs()
# gives the exact alternative. The choice moves mode3_length_ratio by ~5%
# (0.712 -> 0.675).
MODE2_SPEED_RATIO = 1.6   # adot / c_s upslope, Meloche et al. (2025) as written
MODE3_SPEED_RATIO = 1.0   # adot / c_s cross-slope cap, Broberg (1989)

# Floor on r_c in slab_tensile_support (m). SNOWPACK reports r_c = 0 where the
# slab's own weight overloads the weak layer (no cut needed): every such cluster
# on the 2025-26 Little Professor CSVs has tau_g > tau_p, and they are the deep
# slabs (97-100% of clusters with h > 3 m; 14 of 1608 on Jan 18, ~300 from late
# Feb). r_c -> 0 drives the bending load to 0 and S_t to infinity, which is the
# right limit (the slab cannot break before a zero-length crack), but it was a
# divide by zero (891 RuntimeWarnings per 11-date run). 1 mm keeps S_t finite
# (~1e5) without changing any gate outcome: the smallest positive r_c seen is
# ~1 mm and S_t there is far above 1. Ron, 2026-11-10; same in both repos.
RC_MIN_M = 1.0e-3

# Trigger fracture toughness (Ron, 2026-11-09/10). A slab fracture at the
# trigger starts at one point and propagates, so the slab's tensile capacity
# there is also limited by its toughness: an edge crack of depth a in a slab of
# toughness K_Ic fails at K_Ic / (F sqrt(pi a)). F = 1.122 is the
# semi-infinite edge-crack factor and a = h/2 the crack depth; both are the
# values the 2026-11-09 sweep evaluated (ski non-triggers 13/46 -> 0/46, Jan 18
# triggers unchanged, robust to the K source). The finite-thickness factor
# (edge_crack_strength's polynomial) would be ~2.8 at a/h = 0.5, i.e. stricter;
# that was not evaluated.
TOUGHNESS_EDGE_F = 1.122
TOUGHNESS_A_FRAC = 0.5

# ============================================================================
# TRIGGER SELECTION (analysis_pipeline.step_scenarios)
# ============================================================================
# Distinct from TAU_G_FEATURE_FLOOR (50 Pa), which gates whether arrest
# indices are computed at all; see the note there.
MIN_TAU_G          = 40.0   # Pa; minimum gravitational driving shear stress
MAX_SK38           = 1.0    # Sk38 >= 1.0 = stable for skier triggering (Schweizer & Jamieson 2007);
                            # also the 'stability_connected' baseline threshold
MIN_SLAB_THICKNESS = 0.3    # m; Ron 2026-11-13: was 0.5 ("D1 at most"), but Jan 18 failed at ~0.45 m
# m; the deepest slab we expect explosive control to trigger at the basal WL.
# This is an operational choice, not a physical result, and it binds hard. Sk38
# at the WL falls almost monotonically with slab thickness (Spearman -0.93
# within the 2026-01-18 pool), so ranking by Sk38 picks the thickest slabs
# under this cap. 2.0 -> 2.5 m replaced all five triggers on both 2026-01-18
# and 2026-03-07. 1.5 m left only two candidates on 2026-03-07 (median slab
# 1.8 m). Change it only for an operational reason. It is the default of
# analysis_pipeline's --max-slab-thickness, which the run reads.
MAX_SLAB_THICKNESS = 2.0
SLOPE_MARGIN_DEG   = 2.0    # degrees added to the Stauchwall angle for the trigger gate
# Triggers must be in the upper portion of the candidate zone: this elevation
# percentile in operational mode (full start zone); the restricted one with
# --restrict-to-release-area (release area = known crown terrain).
ELEVATION_PCTILE            = 50
ELEVATION_PCTILE_RESTRICTED = 75
# Trigger A_ca used to size the release: A_ca_brittle, capped (the start zone
# is ~250 m wide; beyond this the scaling law is unreliable at very low tau_g),
# with a fallback when the trigger has none.
A_CA_MAX_M      = 300.0
A_CA_FALLBACK_M = 50.0

# ============================================================================
# PROPAGATION MODEL (release_geometry: propagate_release / _qualifies)
# Every parameter of the release BFS is here (Ron, 2026-11-10: no literals
# inside the functions). They are read at call time, so a slope's
# [calibration] (calibration.py) or a CLI flag can set them.
# ============================================================================

# ---- Terrain and polygon -----------------------------------------------------
# Stauchwall angle: also config.ProjectConfig.stauchwall_deg and the slope
# TOML's [scenarios] stauchwall_deg, which wins for analysis_pipeline (CLI
# --stauchwall-deg overrides it). This value is the default for the BFS and the
# scoring scripts.
STAUCHWALL_DEG   = 29.0   # slope threshold for downslope arrest (degrees); Ron 2026-10-27, was 28.0
GAUME_ASPECT_CAP = 2.5    # max width/A_ca ratio (prevents runaway cross-slope)
MIN_POLYGON_AREA = 200.0  # m² — discard degenerate polygons smaller than this

# ---- BFS graph and region-size cap -------------------------------------------
BFS_K_NEIGHBOURS = 8      # cluster adjacency degree for the flood-fill
# Safety cap on the BFS flood-fill.  When it binds, the polygon is labelled
# [CAP-BOUND] so it is never mistaken for a physics-limited result.
# Measured on Jan 18: 500 → 2000 moves median area ~20 % (5 366 → 6 405 m²)
# and best IoU 0.639 → 0.667; region size saturates by ~1 000 clusters so
# any value ≥ 1 000 gives the same answer — raising it is not tuning.
MAX_BFS_CLUSTERS = 2000

# ---- Per-cluster arrest floors (_qualifies) ---------------------------------
# Absolute gravitational-driving-stress floor for crack propagation (Pa).
# tau_g is the driving LOAD on the weak layer; below this value the WL
# cannot sustain dynamic fracture, regardless of slope or trigger. Unlike a
# relative floor, an absolute one can represent a stable slope: if no
# connected terrain exceeds it, the crack does not propagate.
# The 540/230 Pa (inside/adjacent) reference this was set from is from older
# features. On the 11-10 calibrated Jan 18 features the start-zone minimum is
# 409 Pa (release p5 1554, ski p5 605 Pa), so 350 never binds and Jan 18 does
# not move below ~1200 Pa (sweep 200-2500, 11-10). Was a function-local literal
# in propagate_release until 2026-11-10; per slope: [calibration] tau_g_floor.
TAU_G_ABS_FLOOR        = 350.0  # Pa
# Load-to-strength floor (Ron, 2026-11-10; both repos; default 0 = off): arrest
# a cluster whose tau_g / tau_p (wl_shear_strength) is below it. Dimensionless,
# so unlike TAU_G_ABS_FLOOR it does not encode one snowpack's slab density. On
# Jan 18 (calibrated features) it separates release from the skied slope like
# tau_g does (AUC 0.97; medians 0.92 release, 0.54 ski), because SNOWPACK's
# tau_p is a function of WL density alone (r = 0.99 with rho_wl^2) and varies
# little. Per slope: [calibration] tau_ratio_floor (Little Professor 0.65 since
# calibration 2026-11-10.2; Jan 18 IoU plateau 0.60-0.70).
TAU_RATIO_FLOOR        = 0.0
MIN_PROPAGATION_SLAB   = 0.30   # m — slab too thin to sustain crack propagation (0.50 until 2026-11-13; Jan 18 failed at ~0.45 m)
MIN_PROPAGATION_LAMBDA = 0.1    # m — slab too weak/compliant to transmit fracture
                                # Λ = √(E×h³ / (12×G_wl×D_wl)): integrates
                                # slab thickness, stiffness, and WL properties

# ---- Meloche (2025) directional crack arrest ----------------------------------
# When True, _qualifies replaces the heuristic Lambda/thickness gradient gates
# with the field-validated A_ca scaling (Eq. 20): arrest a C->N step where the
# directed WL shear-strength gradient makes the arrest length shorter than the
# step. ON since 2026-10-23 (Ron). The Lambda/thickness gates are then
# skipped, not run alongside. Eq. 20 comes from arrest_indices.arrest_length
# (with C_FIT and DELTA), so the C = 0.045 prefactor cannot be dropped; the
# inline copy that did drop it was 22x too long.
USE_MELOCHE_ARREST = True

# ---- Λ / thickness gradient arrest (dead while USE_MELOCHE_ARREST) ----------
# DIRECTIONAL relative change. Softening/thinning (a DROP in Lambda or slab
# thickness) is the physical arrest mechanism and uses a tight threshold;
# stiffening/thickening (a RISE) favors continued propagation and uses a
# looser threshold, so the crack is not stopped prematurely by running into
# stronger/thicker snow.
#   arrest if  (Λ_nbr - Λ_cur)/Λ_cur  < -LAMBDA_DROP_FACTOR    (softening)
#   arrest if  (Λ_nbr - Λ_cur)/Λ_cur  > +LAMBDA_RISE_FACTOR    (stiffening)
#   (same pattern for slab thickness h)
# size_factor multiplies thresholds → larger size_factor = more tolerant = larger release.
LAMBDA_DROP_FACTOR     = 0.20   # tight: softening arrests readily
LAMBDA_RISE_FACTOR     = 0.30   # loose: stiffening rarely arrests
THICKNESS_DROP_FACTOR  = 0.20   # tight: thinning arrests readily
THICKNESS_RISE_FACTOR  = 0.30   # loose: thickening rarely arrests

# ---- Slab tensile-failure gate ----------------------------------------------
# Ron, 2026-10-26; ON by default since 2026-10-27, Ron, both repos; CLI
# --no-slab-tensile-gate turns it off. Arrest a candidate whose slab would
# break in tension before its weak-layer crack reaches critical length:
# S_t = sigma_t / (3 rho g r_c^2 / h) < 1, Schweizer et al. (2014), via
# arrest_indices.slab_tensile_support. Per-cluster, like gates 4-6, and placed
# after them. sigma_t is Jamieson & Johnston (1990) with the facet knockdown
# (FACET_STRENGTH_FACTOR; f by grain class since 2026-10-27, FACET_SOURCE),
# the strength relation the 2014 paper uses; faceted slabs are ~half as strong
# at equal density, which is the RG-inside / FC-outside contrast at Jan 18.
# The threshold is 1, not a fit. Why on: S_t beat every random-arrest control
# of the same count, and with grain-class sigma_t it costs no trigger it lets
# through. Against: where S_t = 1 falls depends on how r_c is scaled. A
# features CSV without SLAB_TENSILE_SIGMA_COL makes it a no-op; step_scenarios
# warns.
USE_SLAB_TENSILE_GATE  = True
SLAB_TENSILE_SIGMA_COL = 'sigma_t_mean__jamieson1990'

# ---- Trigger fracture toughness ---------------------------------------------
# Ron, 2026-11-10; both repos; CLI --no-trigger-toughness. At the TRIGGER only,
# sigma_t in S_t is capped by the slab's edge-crack strength
# K_Ic / (TOUGHNESS_EDGE_F sqrt(pi TOUGHNESS_A_FRAC h))
# (arrest_indices.toughness_limited_strength), K_Ic the ligament column
# (Schweizer 2004, extrapolated past its density range). The BFS per-cluster
# tensile gate is unchanged. On 2026-11-09 it took the Jan 18 ski-track
# non-triggers from 13/46 releasing to 0/46 and left the five Jan 18 triggers
# and IoU unchanged; it does not cut the no-event false alarms. A features CSV
# without TOUGHNESS_K_COL leaves the plain S_t.
USE_TRIGGER_TOUGHNESS = True
TOUGHNESS_K_COL       = 'K_Ic_ligament'

# ---- Distance caps -------------------------------------------------------------
# BFS distance caps (up: A_ca, down: Stauchwall distance, cross: Gaume width).
# Default on, both repos. Off (a slope's [calibration] distance_caps = false,
# or CLI --no-distance-caps) leaves the release bounded by the per-cluster
# gates, the start zone and MAX_BFS_CLUSTERS. Off, the scenario size factors
# scale nothing in the BFS (they scale the caps and the rectangle fallback;
# the Λ/thickness factors are dead while USE_MELOCHE_ARREST is on).
USE_DISTANCE_CAPS = True
# d_up = max(A_ca, D_UP_MIN_M) (D_CAP_DEFAULT_M without an A_ca),
# d_down = max(Stauchwall distance, D_DOWN_MIN_M), d_lat = max(Gaume width,
# D_LAT_MIN_M); all but the floor on d_lat scale with size_factor.
D_UP_MIN_M      = 20.0   # m
D_DOWN_MIN_M    = 20.0   # m
D_LAT_MIN_M     = 15.0   # m
D_CAP_DEFAULT_M = 50.0   # m; A_ca stand-in when the trigger has none

# ---- Running slab cracks -----------------------------------------------------
# Ron, 2026-11-10; both repos; CLI --no-slab-crack-lines. A slab that the
# per-cluster tensile gate breaks (slab_tensile_failure) starts a crack IN THE
# SLAB: a crown if the BFS reached it by an upslope step (runs cross-slope), a
# flank if by a cross-slope step (runs along the fall line); downslope steps
# start none (Stauchwall, compression). After the BFS each crack runs as a
# line, one best-aligned neighbour per step (within SLAB_CRACK_ALIGN_COS of the
# line), while the slab there is loaded: inside the admitted region, with
# detached slab below a crown / beside a flank within SLAB_CRACK_BAND_M, and
# the band-mean slope above PHI_DEG. The area the lines sever from the trigger
# (4-connected pixels) is removed; the line clusters stay (they are the
# boundary). Order-independent by construction. No fracture toughness in the
# run test: with any static load of a detached slab (full weight, or
# Stauchwall- and flank-supported, q*min(L,W)/2), K was 10-75x the slab K_Ic
# on Jan 18 (0 of 2220 steps stopped), because a crack a few metres long needs
# < ~0.5 kPa to arrest. Jan 18 (calibration 2026-11-10.2): IoU 0.601 -> 0.610,
# area 7429 -> 6831 m2, AUC 0.801 -> 0.848, ski 0/46, same five triggers;
# 419 m2 overshoot removed for 179 m2 of release; 9/10 no-event days unchanged
# (the tensile gate never fires there).
USE_SLAB_CRACK_LINES = True
SLAB_CRACK_BAND_M    = 5.0     # m; half-width of the strip that sets L and W
SLAB_CRACK_ALIGN_COS = 0.707   # a line step must lie within 45 deg of the line

# Severed slab slides (Ron, 2026-11-13; on, both repos). The
# crack lines above drop every piece they cut off from the trigger. But the
# weak layer under that piece has already fractured (the BFS reached it), so
# once the slab breaks around it nothing holds it but crack-face friction. With
# this on, a severed piece (connected in the BFS graph) stays in the release
# when its mean cluster slope exceeds atan(SEVERED_SLIDE_MU); only a piece on
# gentler ground is dropped. 0.62 is the crack-face friction where slabs came
# to rest, van Herwijnen & Heierli 2009 GRL (31.8 deg). Motivation: on a 0.45 m
# slab (Jan 18, h x 0.324) one line across the neck at the trigger dropped
# ~410 of ~650 reached clusters at facet_a 0.5 and none at 0.6.
USE_SEVERED_SLIDE = True     # adopted 2026-11-13 (Ron), both repos
SEVERED_SLIDE_MU  = 0.62

# ---- Cross-slope θ sampling (estimate_cross_slope_width) ----------------------
# The lateral sample has to sit near the true cross-slope axis (aspect ± 90°).
# Selecting it by raster row/col instead — "same row ± 50 px, column offset
# > 5 px" — silently assumes the fall line runs north-south. On the Jan 18 ESE
# start zone (aspect 113-120°, cross axis 23-30°) that put the chosen
# neighbours a mean 30° off the cross axis, with some within 1.2° of the fall
# line, so theta_cross was contaminated by the along-slope gradient. Fixed
# 2026-10-09 in release_areas, ported here.
THETA_CROSS_SECTOR_DEG = 30.0  # ± half-angle accepted about the sampling axis
THETA_CROSS_MIN_SEP_M  = 5.0   # m; closer pairs make |Δτp|/d noise-dominated
THETA_CROSS_MAX_SEP_M  = 50.0  # m; beyond this the pair is not abeam

# ---- Mode III (cross-slope) elastic length -----------------------------
# Λ_III is a different and SMALLER quantity than the along-slope (mode II) Λ.
# elastic_length_cross() now matches Gaume's own antiplane-shear derivation
# exactly (personal communication, 2026-10-18; not yet published) — see
# ~/avalanche-release-areas/docs/release_area_methods.md §3. Whether antiplane
# shear (vs. bending) is the right cross-slope regime is still open; Gaume's
# open item. The flag stays off because a single global NU makes Λ_III/Λ_II
# a constant, which the continuity gate is provably invariant to. Only a
# Λ_III with different spatial structure can move the flanks.
USE_MODE3_LAMBDA = False

# Λ continuity thresholds, cross-slope (mode III / flanks). Separate knobs so
# the flanks can be tuned independently once Λ_III is known; identical to the
# along-slope pair for now, so splitting the code changed no results.
LAMBDA_CROSS_DROP_FACTOR = 0.20
LAMBDA_CROSS_RISE_FACTOR = 0.30

# ---- Mode III (cross-slope) crack-speed cap ----------------------------
# Upslope (mode II) cracks run supershear at ~1.6 c_s; mode III cracks cannot
# exceed c_s (Broberg 1989). Since k_x = k_f c_p²/(c_p² + ȧ²), a slower crack
# builds slab tension FASTER per unit advance, so the distance to first slab
# fracture is shorter cross-slope: L_dyn,III / L_dyn,II = 0.712 at ν = 0.3
# (MODE2_SPEED_RATIO / MODE3_SPEED_RATIO above). Unlike USE_MODE3_LAMBDA this
# is an absolute distance cap, so it does change results. Only the restrictive
# half of mode III is coded — because G < E′ an equal-strength slab has a
# *larger* energy cap cross-slope, but that needs a flank strength τ_flank with
# no parameterisation. Do not tune the ratio to Jan 18.
#
# DEFAULT FLIPPED True -> False on 2026-10-10 (Ron's call), mirroring
# release_areas/config.py. The argument is COMPLETENESS, not fit: only the
# restrictive half of mode III is coded. Because G < E′ an equal-strength slab
# also has a LARGER energy cap cross-slope (G_slab,III = τ_flank²h/2G against
# σt²h/2E′), which pushes the other way, but that needs a flank strength
# τ_flank with no parameterisation (blocked on a Cam-Clay β ambiguity). The
# honest default for one side of a two-sided effect is off.
#
# Jan 18 corroborates without deciding it: cap off gives mean IoU 0.622 vs
# 0.562 and area/observed 0.78 vs 0.63, i.e. the cap is what makes the model
# under-predict area. On n = 1 that cannot carry the decision.
#
# TO FLIP BACK: once τ_flank lands, enable BOTH halves together — the cap and
# slab_energy_cap_cross. This one alone restores the same one-sided bias.
#
# Consequence worth knowing: with the cap off, d_lat IS the Gaume θ-ratio
# width, so that path is live again rather than dormant.
USE_MODE3_SPEED_CAP = False  # apply the ratio to the lateral distance cap

# ---- Anisotropic BFS rings (cross-slope growth rate) -------------------
# The same speed ratio applied to the *growth* of the flood-fill rather than
# only to a terminal distance cap: if one ring stands for one increment of
# crack-front advance in TIME, the front should already be ~1.6x further
# along-slope when it arrives. A cross-slope candidate is admitted at ring k
# only once dist <= k * RING_STEP_M * (MODE3/MODE2), and is otherwise DEFERRED,
# so arrest is still evaluated on arrival. The clock counts only rings that
# admitted an along-slope cluster — the flanks are fed by the advancing front,
# so driving the budget from the global ring counter lets flank growth buy
# itself time and the constraint evaporates.
#
# RING_STEP_M = None derives the step from the median adjacent-cluster centroid
# separation, i.e. one hop, so the fast axis keeps the unmodified BFS rate.
#
# Measured on Jan 18 (release-areas, 2026-10-10): with the default caps this is
# INERT — d_lat = A_ca x 0.712 = 29-38 m binds before the ring budget (50-62 m),
# so areas and IoU are unchanged. With the lateral cap removed it recovers most
# of the cap's effect from the speed ratio alone (mean area 1.69x -> 0.85x
# observed, best IoU 0.392 -> 0.642) and aligns the crown axis to 119 deg
# against an observed 120.5 deg, but it under-elongates: L/W 1.9-2.1 against an
# observed 3.2. Off by default. Do not tune the ratio — it is Broberg's.
USE_ANISOTROPIC_RINGS = False
RING_STEP_M           = None

# ---- Arrest rule and start-zone mask use (path-energy brief) ---------------
# 'legacy' is propagate_release as before. 'energy' is the path-energy
# hypothesis (energy_path.py, propagate_release_energy). 'slope_only' and
# 'stability_connected' are mechanics-free baselines (baselines.py). Every
# mode but 'legacy' has no rectangle fallback and no MIN_POLYGON_AREA floor.
ARREST_MODE = 'legacy'
# 'gate': the BFS rejects clusters outside the start-zone KML and the polygon is
# clipped to it. 'diagnostic': the KML is used only in reporting
# (frac_outside_mask); 17% of the observed Jan 18 crown lies outside it.
MASK_MODE   = 'gate'

# ============================================================================
# PATH-ENERGY ARREST MODE (energy_path; hypothesis, not validated)
# ============================================================================
# ENERGY_DELTA is separate from the Eq. 20 DELTA (1.0): R = 0.48 was fitted to
# the JGR runs at delta = 0, so the R-based test uses 0.
ENERGY_DELTA        = 0.0
SIGMA_T_COLUMN      = 'sigma_t_mean'  # thickness-weighted slab σt; 'sigma_t_wl' = weakest-link bound
ENERGY_USE_RESIDUAL = False           # τr = 0 in G_c, as arrest_indices.evaluate(use_residual=False)
ENERGY_R            = None            # energy ratio R; None = R_FIT. Sensitivity runs only
# Weak-layer thickness D_wl behind K_wl = G_wl / D_wl, and so G_c ∝ D_wl:
# 'sequence' = wl_thickness, the whole basal FC/DH sequence (as Eq. 20 uses);
# 'weakest_element' = wl_weakest_thickness, the lowest-tau_p WL element;
# a float = that thickness in m (the JGR calibration used 0.04).
KWL_THICKNESS       = 'sequence'
# Slab tension at the trigger. False: N = 0, built along the fall line, so the
# first steps need Pi ~ (N_max / (tau_net ds))^2 >> 1. True: N = N_max, i.e. the
# crack has already reached shear-driven propagation, which the brief assumes
# (no a_c / anticrack stage is modelled).
ENERGY_TRIGGER_SATURATED = False
ENERGY_E_RELATION   = None            # E(rho) for the energy mode; None = E_RELATION (plain E_eff)
