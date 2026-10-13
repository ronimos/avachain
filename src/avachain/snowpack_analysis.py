"""
snowpack_analysis.py — Core SNOWPACK analysis functions.

Provides:
    geojson_to_mask()            Rasterize a GeoJSON polygon to a boolean mask
    assign_cluster_groups()      Assign clusters to release / adjacent / reference groups
    split_wl_slab()              Identify WL / slab boundary via grain type
    element_thickness()          Per-element thickness dz from the z coordinate
    profile_features()           Per-cluster WL/slab/stability features at one timestep
    extract_group_timeseries()   Time series of features for a cluster group
    compute_meloche_features()   Meloche et al. (2025) crack arrest parameters
    reduce_scalars_at_time()     Per-cluster scalar reduction for visualization

No CLI. Import from analysis_pipeline.py, visualize_snowpack.py,
and analyze_release_zone.py.
"""

from __future__ import annotations

import json
import warnings
import rasterio
import rasterio.features
import rasterio.transform as rt
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import arrest_indices as _ai
from arrest_indices import evaluate as _ai_evaluate

# Slab/weak-layer material constants shared across profile_features and
# compute_meloche_features (Reiweger et al. 2010; Gaume et al. 2018)
G_WL = 0.2e6    # weak-layer shear modulus (Pa)
NU   = 0.3      # slab Poisson's ratio

# Layered-slab aggregation of sigma_t / E (arrest_indices.aggregate_slab).
# Faceted snow is ~half as strong in tension as rounded snow at equal density
# (Jamieson & Johnston 1990), applied continuously via f = 1 - sphericity so
# neighbouring cells cannot step across a grain-type boundary.
# FACET_SP_REF = None means f is 1 - sphericity outright; our rounded layers
# sit at sphericity ~0.86, so they still carry f ~0.14.
FACET_STRENGTH_FACTOR = 0.5
FACET_SP_REF          = None
# Edge-crack bound: HYPOTHESIS, off because snow has no sourced slab-tension
# K_Ic(rho). Supply SLAB_K_IC (Pa m^0.5) to enable.
USE_LIGAMENT_BOUND = True
# SLAB_K_IC overrides the relation with a constant (Pa m^0.5); None = use
# K_IC_RELATION. 'borstad2013'/'schweizer2004' are NOT implemented -- their
# published equations have not been sourced and quoted back for checking.
SLAB_K_IC          = None
K_IC_RELATION      = 'schweizer2004'
LIGAMENT_MODEL     = 'elhaddad'   # or 'lefm' for the uncorrected form
SELF_CONSISTENT_A0 = False        # True evaluates a0 at F(a0/h) iteratively
DMAX_FACTOR        = 1.0          # d_max = DMAX_FACTOR * grain_size (both m)
E_RELATION         = 'vanherwijnen2016'   # or 'project_fit'
E_RELATIONS_EXTRA  = ('project_fit',)
LIGAMENT_VARIANTS = (
    ('kirchner2000',     'kirchner2000',  1.0),
    ('schweizer2004_d1', 'schweizer2004', 1.0),
    ('schweizer2004_d2', 'schweizer2004', 2.0),
)

# How θ is estimated from the per-cluster τp field. θ is a property of the LAG
# it is measured at, not of the snowpack.
#   'knn'        mean |Δτp|/d over the K_NEIGHBORS nearest cluster centroids.
#                Reproduces every pre-2026-10-09 CSV and published number.
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
L_SS                     = 20.0   # m; steady-state propagation length, Meloche et al. (2025)
THETA_ESTIMATOR          = 'plane_fit'   # 'knn' | 'plane_fit'
# The radius IS L_ss, and that identity is the argument for the default: θ is
# the gradient of a ramp over L_ss, so the neighbourhood must be L_ss.
THETA_FIT_RADIUS_M       = L_SS
THETA_FIT_MIN_NEIGHBOURS = 8      # minimum centroids in radius, else θ = NaN
THETA_MIN                = 1e-6   # Pa/m; below this there is no usable gradient

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

# Crack-speed ratios for the mode II / mode III dynamic tension gradient; the
# source of truth is arrest_indices. Mode III cannot exceed c_s (Broberg 1989)
# where mode II runs supershear at ~1.6 c_s.
MODE2_SPEED_RATIO = _ai.MODE2_SPEED_RATIO
MODE3_SPEED_RATIO = _ai.MODE3_SPEED_RATIO

# SNOWPACK grain type codes (first digit = grain class)
WL_TYPES   = set(range(400, 500)) | set(range(500, 600))  # FC (4xx), DH (5xx)
SLAB_TYPES = set(range(200, 300)) | set(range(300, 400))  # DF (2xx), RG (3xx)

# Zarr variable holding SNOWPACK code 0606 (critical cut length r_c, m).
RC_VAR = 'critical_cut_length'

EVENT_DATE = pd.Timestamp("2026-01-18")

GROUP_COLORS = {
    'release':   '#d32f2f',
    'adjacent':  '#1976d2',
    'reference': '#666666',
}
GROUP_LABELS = {
    'release':   'Release zone',
    'adjacent':  'Adjacent slope (start zone)',
    'reference': 'Reference (terrain-matched)',
}


# -----------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------

def _reproject_lonlat_to_utm(pts_lonlat, epsg: int = 32613):
    """Reproject list of (lon, lat) points to UTM."""
    from pyproj import Transformer
    t = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True)
    return [t.transform(x, y) for x, y in pts_lonlat]


def geojson_to_mask(path, dem_shape, transform, dst_epsg: int = 6342) -> np.ndarray:
    from shapely.geometry import shape
    from shapely.ops import unary_union
    from pyproj import Transformer as _T, CRS as _CRS

    with open(str(path)) as f:
        gj = json.load(f)
    polys = [shape(feat['geometry']) for feat in gj['features']]
    merged = unary_union(polys)

    # Detect source CRS from the file's crs node; fall back to WGS84.
    # Handles both 'EPSG:32613' and 'urn:ogc:def:crs:EPSG::32613' formats.
    import re as _re
    src_epsg = 4326
    crs_node = gj.get('crs', {}).get('properties', {}).get('name', '')
    _m = _re.search(r'EPSG:+(\d+)', crs_node, _re.IGNORECASE)
    if _m:
        src_epsg = int(_m.group(1))

    if src_epsg != dst_epsg:
        tr = _T.from_crs(f'EPSG:{src_epsg}', f'EPSG:{dst_epsg}', always_xy=True)
        from shapely.geometry import Polygon, MultiPolygon

        def _reproj_ring(r):
            xs, ys = zip(*r.coords)
            xs2, ys2 = tr.transform(xs, ys)
            return list(zip(xs2, ys2))

        def reproj(poly):
            if poly.geom_type == 'Polygon':
                return Polygon(_reproj_ring(poly.exterior),
                               [_reproj_ring(i) for i in poly.interiors])
            return MultiPolygon([reproj(p) for p in poly.geoms])

        merged = reproj(merged)

    aff = transform if hasattr(transform, 'c') else rt.Affine(*transform[:6])
    mask = rasterio.features.geometry_mask(
        [merged.__geo_interface__], out_shape=dem_shape,
        transform=aff, invert=True)
    print(f"  Release zone mask:  {mask.sum()} cells")
    return mask


# -----------------------------------------------------------------------
# Cluster group assignment
# -----------------------------------------------------------------------

def assign_cluster_groups(cluster_map, release_mask, start_zone_mask,
                           dem, domain_mask) -> dict:
    """
    Assign clusters to groups. Reference group is terrain-matched to
    release zone (similar elevation and slope angle).
    """
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx   = np.gradient(fill_dem, 1.0)
    slope    = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    groups = {'release': set(), 'adjacent': set(), 'reference': set()}

    # When no observed release is available, only adjacent can be assigned;
    # reference terrain-matching requires a known release zone for statistics.
    no_release = not release_mask.any()
    if no_release:
        rel_elev_mean = rel_elev_std = rel_slope_mean = None
    else:
        rel_elev_mean  = float(np.nanmean(dem[release_mask & domain_mask]))
        rel_elev_std   = float(np.nanstd(dem[release_mask & domain_mask]))
        rel_slope_mean = float(np.nanmean(slope[release_mask & domain_mask]))

    for cid in np.unique(cluster_map[domain_mask]):
        if cid <= 0:
            continue
        cells = cluster_map == cid
        n     = cells.sum()
        if n == 0:
            continue

        frac_rel   = 0.0 if no_release else (cells & release_mask).sum() / n
        frac_start = (cells & start_zone_mask).sum() / n

        if frac_rel >= 0.3:
            groups['release'].add(cid)
        elif frac_start >= 0.3:
            groups['adjacent'].add(cid)
        elif not no_release:
            # Terrain-match: within 1 std of release zone elevation
            c_elev  = float(np.nanmean(dem[cells]))
            c_slope = float(np.nanmean(slope[cells]))
            if (abs(c_elev - rel_elev_mean) < rel_elev_std * 1.5 and
                    abs(c_slope - rel_slope_mean) < 10.0):
                groups['reference'].add(cid)

    for g, ids in groups.items():
        print(f"  {GROUP_LABELS[g]}: {len(ids)} clusters")
    return groups


# -----------------------------------------------------------------------
# WL / slab boundary detection
# -----------------------------------------------------------------------

def split_wl_slab(grain_type, z):
    """
    Find the BASAL weak layer by scanning upward from the bottom.

    The persistent weak layer (early-season facets/depth hoar) sits at
    the base. Near-surface facets are NOT the target WL.

    Strategy: scan from the bottom upward through FC/DH layers (4xx/5xx),
    stop at the FIRST transition to slab types (code < 400).
    That initial FC/DH block = basal WL. Everything above = slab.

    Returns (slab_mask, wl_mask, interface_z) or (None, None, None).
    """
    ok = ~np.isnan(grain_type) & ~np.isnan(z) & (z != 0) & (grain_type > 0)
    if ok.sum() < 3:
        return None, None, None

    gt_ok = np.round(grain_type[ok]).astype(int)
    z_ok  = z[ok]

    # Sort bottom-to-top: z is negative (depth below surface), most-negative = deepest
    order        = np.argsort(z_ok)
    gt_sorted    = gt_ok[order]
    z_sorted     = z_ok[order]
    # Only FC (4xx) and DH (5xx) are weak layer grain types
    is_wl_sorted = (gt_sorted // 100 == 4) | (gt_sorted // 100 == 5)

    # Must start with at least one WL layer at the base
    if not is_wl_sorted[0]:
        return None, None, None

    # Walk upward: find top of the first contiguous basal WL block
    wl_top_sorted = 0
    for i in range(len(is_wl_sorted)):
        if is_wl_sorted[i]:
            wl_top_sorted = i
        else:
            break   # first non-WL layer = interface

    interface_z = float(z_sorted[wl_top_sorted])

    # Map back to original array indices
    ok_indices = np.where(ok)[0]
    orig_order = ok_indices[order]

    slab_mask = np.zeros(len(grain_type), dtype=bool)
    wl_mask   = np.zeros(len(grain_type), dtype=bool)
    for sorted_i, orig_i in enumerate(orig_order):
        if sorted_i <= wl_top_sorted:
            wl_mask[orig_i] = True
        else:
            slab_mask[orig_i] = True

    return slab_mask, wl_mask, interface_z


def element_thickness(z, hs, valid):
    """
    Per-element thickness dz (m), in the original element index order.

    SNOWPACK's `z` is depth below the snow surface, negative downward, and is
    exactly `height_above_ground - HS` (verified to machine precision against
    the zarr `height` variable). So sorting by z and differencing gives each
    element's own thickness, with the basal element measured from the ground at
    z = -HS. `np.sum(dz) == HS` exactly when no element is excluded.

    Bulk slab properties must be weighted by dz: SNOWPACK element thicknesses
    span a factor of ~10 within a single profile (3-38 mm measured on the
    Little Professor domain), so an unweighted element mean over-weights thin
    layers. Likewise a layer's thickness is the SUM of its elements' dz, not
    the span of their z, which omits the lowest element entirely.

    valid : boolean mask of elements to include; dz is NaN elsewhere.
    """
    z = np.asarray(z, dtype=float)
    dz = np.full(z.shape, np.nan)
    idx = np.where(valid)[0]
    if len(idx) == 0 or not np.isfinite(hs):
        return dz
    order = idx[np.argsort(z[idx])]
    dz[order] = np.diff(z[order], prepend=-float(hs))
    return dz


def thickness_weighted_mean(values, mask, dz):
    """Thickness-weighted mean of `values` over `mask`; NaN if no usable element."""
    values = np.asarray(values, dtype=float)
    m = mask & np.isfinite(values) & np.isfinite(dz) & (dz > 0)
    if not m.any():
        return np.nan
    return float(np.sum(values[m] * dz[m]) / np.sum(dz[m]))


# -----------------------------------------------------------------------
# Per-cluster feature extraction
# -----------------------------------------------------------------------

def profile_features(ds_t_loc, min_depth_cm: float) -> dict:
    """
    Extract slab and WL features for one cluster at one timestep.

    Stability indices: min value within ±5cm of WL/slab interface.
    Slab properties:   mean over slab layers above interface.
    WL properties:     mean over WL layers.
    """
    z  = ds_t_loc['z'].values.ravel()
    gt = ds_t_loc['grain_type'].values.ravel()
    hs_arr = ds_t_loc['HS'].values.ravel()
    hs = float(np.nanmean(hs_arr))

    ok = ~np.isnan(z) & ~np.isnan(gt) & (z != 0) & (gt > 0)
    if ok.sum() < 2:
        return {}

    result = {'hs': hs}
    nu = NU

    slab_m, wl_m, interface_z = split_wl_slab(gt, z)

    if slab_m is None:
        return result

    # Element thicknesses, used to weight every bulk mean and to sum layer
    # thicknesses. See element_thickness() for why a plain element mean and a
    # z-span are both wrong.
    dz = element_thickness(z, hs, ok)

    # --- Stability at interface (±5cm window) ---
    near_interface = ok & (np.abs(z - interface_z) <= 0.05)
    for var in ('sk38', 'ssi', 'sn38', 'stab_deformation_rate'):
        try:
            v = ds_t_loc[var].values.ravel()
            vals = v[near_interface & ~np.isnan(v)]
            result[f'min_{var}'] = float(np.nanmin(vals))                 if len(vals) else np.nan
        except Exception:
            result[f'min_{var}'] = np.nan

    # --- Slab properties ---
    if slab_m.any():
        # Thickness-weighted, so that rho * h is the true slab load per unit
        # area and E/sigma_t see a mass-weighted bulk density.
        for var in ('density', 'hand_hardness', 'grain_size'):
            try:
                v = ds_t_loc[var].values.ravel()
                result[f'slab_{var}'] = thickness_weighted_mean(v, slab_m & ok, dz)
            except Exception:
                result[f'slab_{var}'] = np.nan

        # Slab thickness: distance from interface to snow surface
        slab_z = z[slab_m & ok]
        # interface_z is in m (negative from surface)
        result['slab_thickness'] = float(-interface_z)                   if len(slab_z) else np.nan

        # Most common grain type in slab (first digit = grain class)
        gt_slab = np.round(gt[slab_m & ok]).astype(int)
        gt_slab = gt_slab[gt_slab > 0]
        if len(gt_slab):
            classes, counts = np.unique(gt_slab // 100, return_counts=True)
            result['slab_dominant_grain_class'] = int(classes[np.argmax(counts)])
        else:
            result['slab_dominant_grain_class'] = np.nan

        # --- Crust/ice layer detection (MF=7xx, IF=8xx) ---
        gt_slab_all = np.round(gt[slab_m & ok]).astype(int)
        z_slab_all  = z[slab_m & ok]
        dz_slab_all = dz[slab_m & ok]
        # MF (7xx) and IF (8xx) only. The previous `>= 7` also caught FCxr
        # (9xx, rounding facets), which is 29% of slab thickness here and
        # present in every profile — so every crust flag in the domain was a
        # false positive (zero MF/IF elements exist in it). Faceted layers are
        # handled by the layered sigma_t aggregation instead.
        is_crust    = np.isin(gt_slab_all // 100, (7, 8))
        result['has_crust']      = bool(is_crust.any())
        result['n_crust_layers'] = int(is_crust.sum())
        # Sum of crust element thicknesses. The z span this replaced both
        # omitted the lowest crust element and counted any non-crust elements
        # sandwiched between two crusts.
        result['crust_thickness'] = float(
            np.nansum(dz_slab_all[is_crust])
        ) if is_crust.any() else 0.0
        # Crust immediately above WL (<10cm of interface) — bridging layer
        near_iface = np.abs(z_slab_all - interface_z) <= 0.10
        result['crust_at_interface'] = bool((is_crust & near_iface).any())
        # Depth of shallowest crust (m below surface)
        result['crust_top_depth'] = float(
            -z_slab_all[is_crust].max()
        ) if is_crust.any() else np.nan
    else:
        for k in ('slab_density', 'slab_hand_hardness', 'slab_grain_size',
                  'slab_thickness', 'slab_dominant_grain_class',
                  'has_crust', 'n_crust_layers', 'crust_thickness',
                  'crust_at_interface', 'crust_top_depth'):
            result[k] = np.nan

    # --- WL properties ---
    if wl_m.any():
        # wl_shear_strength (tau_p) is deliberately left as a plain element
        # mean. Thickness-weighting it would be a modelling change, not a bug
        # fix: a crack runs in the weakest sublayer, so the governing value is
        # arguably the minimum rather than any mean, and tau_p cascades into
        # theta, tau_p_star and the trigger ranking. Decide that separately.
        for var in ('grain_size', 'density', 'hand_hardness'):
            try:
                v = ds_t_loc[var].values.ravel()
                result[f'wl_{var}'] = thickness_weighted_mean(v, wl_m & ok, dz)
            except Exception:
                result[f'wl_{var}'] = np.nan

        try:
            v    = ds_t_loc['shear_strength'].values.ravel()
            vals = v[wl_m & ~np.isnan(v)]
            result['wl_shear_strength'] = float(np.nanmean(vals)) if len(vals) else np.nan
        except Exception:
            result['wl_shear_strength'] = np.nan

        wl_z = z[wl_m & ok]
        result['wl_burial_depth'] = float(-wl_z.max())          if len(wl_z) else np.nan
        # D_wl is the SUM of the WL elements' thicknesses. The z span this
        # replaced omitted the basal element, undercounting D_wl by a median
        # 23% on the Jan 18 2026 Little Professor profiles, which propagates as
        # Lambda ~ sqrt(D_wl) and K_wl = G_wl / D_wl.
        result['wl_thickness']    = float(np.nansum(dz[wl_m & ok])) if len(wl_z) else np.nan
        result['interface_z']     = float(interface_z)

        # Critical cut length r_c (SNOWPACK code 0606) AT the basal WL, read over
        # the same wl_m elements as wl_shear_strength — not min over the full
        # profile, which could land on a different (e.g. near-surface) layer.
        # Onset / propagation-propensity metric (Gaume 2017; Richter 2019);
        # used on the initiation side, complementary to the Meloche A_ca arrest.
        try:
            rc_v    = ds_t_loc[RC_VAR].values.ravel()       # per-element, m
            rc_vals = rc_v[wl_m & ~np.isnan(rc_v)]
            result['rc_wl'] = float(np.nanmean(rc_vals)) if len(rc_vals) else np.nan
            # (swap nanmean -> nanmin if you want the most propagation-prone element)
        except Exception:
            result['rc_wl'] = np.nan
    else:
        for k in ('wl_shear_strength', 'wl_grain_size', 'wl_density',
                  'wl_hand_hardness', 'wl_burial_depth', 'wl_thickness',
                  'interface_z', 'rc_wl'):
            result[k] = np.nan

    # --- Meloche et al. (2025) parameters (per-cluster, no neighbors needed) ---
    rho   = result.get('slab_density',      np.nan)
    h_m   = result.get('slab_thickness',    np.nan)   # m
    D_wl  = result.get('wl_thickness',      np.nan)   # m (from z)
    tau_p = result.get('wl_shear_strength', np.nan)   # Pa

    if all(not np.isnan(v) for v in [rho, h_m, D_wl, tau_p]) and D_wl > 0:
        D_wl_m  = D_wl                                    # already in m
        # Slab elastic modulus — power law fit to van Herwijnen et al. (2016) range
        # ~2 MPa at ρ=200, ~4 MPa at ρ=300, ~6 MPa at ρ=350 kg/m³
        # Same expressions, lifted into arrest_indices so the per-layer path
        # can reuse them; see their docstrings for the (absent) provenance.
        E_slab  = float(_ai.slab_modulus(rho))            # Pa
        sigma_t = float(_ai.slab_tensile_strength(rho))   # Pa
        E_prime = E_slab / (1.0 - nu**2)
        K_wl    = G_WL / D_wl_m
        Lambda  = float(np.sqrt(E_prime * h_m / K_wl))  # characteristic elastic length (m)
        result['E_slab']  = E_slab
        result['sigma_t'] = sigma_t
        result['Lambda']  = Lambda
        result['K_wl']    = K_wl
    else:
        for k in ('E_slab', 'sigma_t', 'Lambda', 'K_wl'):
            result[k] = np.nan

    # --- Layered-slab aggregation (added alongside; nothing above changes) ---
    result.update(_slab_layer_aggregate(ds_t_loc, slab_m & ok, dz, z))
    return result


def _slab_layer_aggregate(ds_t_loc, slab_sel, dz, z) -> dict:
    """Per-layer sigma_t/E aggregation over the slab. All-NaN if unavailable."""
    keys = ('E_eff', 'sigma_t_mean', 'sigma_t_wl', 'f_mean_weighted',
            'wl_ctrl_index', 'wl_ctrl_depth', 'wl_ctrl_thickness',
            'sigma_t_ligament', 'K_Ic_ligament', 'sigma_t_lig_intact',
            'a0_ligament', 'l_ch_ligament', 'n_slab_layers', 'n_rho_clamped')
    try:
        rho = ds_t_loc['density'].values.ravel()[slab_sel]
        sp  = ds_t_loc['sphericity'].values.ravel()[slab_sel]
        dd  = ds_t_loc['dendricity'].values.ravel()[slab_sel]
    except Exception:
        return {k: np.nan for k in keys}
    try:
        gsz = ds_t_loc['grain_size'].values.ravel()[slab_sel]
    except Exception:
        gsz = None   # schweizer2004 needs d_max; it degrades to NaN without it
    return _ai.aggregate_slab(
        rho, dz[slab_sel], sp, dd,
        a=FACET_STRENGTH_FACTOR, sp_ref=FACET_SP_REF,
        depth_i=-z[slab_sel],
        K_Ic=SLAB_K_IC if USE_LIGAMENT_BOUND else None,
        k_ic_relation=(K_IC_RELATION if USE_LIGAMENT_BOUND
                       and SLAB_K_IC is None else None),
        ligament_model=LIGAMENT_MODEL,
        grain_size_i=gsz, dmax_factor=DMAX_FACTOR,
        e_relation=E_RELATION, e_relations_extra=E_RELATIONS_EXTRA,
        k_ic_variants=LIGAMENT_VARIANTS if USE_LIGAMENT_BOUND else (),
        self_consistent_a0=SELF_CONSISTENT_A0)


# -----------------------------------------------------------------------
# Group time series extraction
# -----------------------------------------------------------------------

def extract_group_timeseries(ds, group_ids: set,
                              location_names,
                              min_depth_cm: float,
                              end_ts: pd.Timestamp,
                              per_cluster: bool = False) -> pd.DataFrame:
    """
    Extract feature time series for a group of clusters.

    per_cluster=False (default): returns group-median DataFrame indexed by time.
      Used for plotting.

    per_cluster=True: returns one row per (cluster_id, time), with columns
      for all features. Used for the classifier — each cluster is a sample.
    """
    if not group_ids:
        return pd.DataFrame()

    loc_mask = np.array([
        int(str(loc).split('_')[-1]) in group_ids
        for loc in location_names])
    if not loc_mask.any():
        return pd.DataFrame()

    ds_group   = ds.isel(location=loc_mask)
    loc_ids    = [int(str(loc).split('_')[-1])
                  for loc in location_names[loc_mask]]
    times      = pd.DatetimeIndex(ds_group.coords['time'].values)
    noon_ts    = [t for t in times if t.hour == 12 and t <= end_ts]

    all_rows = []
    for ts in noon_ts:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            ds_t = ds_group.sel(time=ts, method='nearest').compute()

        n_locs = ds_t.sizes.get('location', 1)
        ts_rows = []
        for i in range(n_locs):
            try:
                ds_loc = ds_t.isel(location=i) \
                    if 'location' in ds_t.dims else ds_t
                feat = profile_features(ds_loc, min_depth_cm)
                if feat:
                    feat['time']       = ts
                    feat['cluster_id'] = loc_ids[i] if i < len(loc_ids) else -1
                    ts_rows.append(feat)
            except Exception:
                continue

        if not ts_rows:
            continue

        if per_cluster:
            all_rows.extend(ts_rows)
        else:
            # Group median for plotting
            keys = [k for k in ts_rows[0]
                    if k not in ('time', 'cluster_id')]
            row = {'time': ts}
            for k in keys:
                vals = [f[k] for f in ts_rows
                        if k in f and f[k] is not None
                        and not np.isnan(f[k])]
                row[k] = float(np.nanmedian(vals)) if vals else np.nan
            all_rows.append(row)

    if not all_rows:
        return pd.DataFrame()

    if per_cluster:
        df = pd.DataFrame(all_rows)
        df['time'] = pd.to_datetime(df['time'])
        return df.set_index(['time', 'cluster_id']).sort_index()
    return pd.DataFrame(all_rows).set_index('time').sort_index()


# -----------------------------------------------------------------------
# Meloche et al. (2025) crack arrest parameters
# -----------------------------------------------------------------------

def theta_knn(tau_p_arr, indices, distances, px_m) -> np.ndarray:
    """θ (Pa/m) as the mean |Δτp|/d over each cluster's k nearest centroids.

    `indices`/`distances` come from a k-NN query on *pixel* centroids with the
    cluster itself in column 0, so column 0 is dropped and `px_m` converts the
    separations to metres. Without that conversion θ is Pa per *pixel*, which
    only coincides with Pa/m on a 1 m grid.

    This samples θ at whatever lag the cluster spacing happens to be — 3.3 m on
    Jan 18, where θ is 85% local noise. See `theta_plane_fit`.
    """
    tau_p_arr = np.asarray(tau_p_arr, float)
    out = np.full(len(tau_p_arr), np.nan)
    for i in range(len(tau_p_arr)):
        tau_i = tau_p_arr[i]
        if np.isnan(tau_i):
            continue
        vals = []
        for j, d in zip(indices[i][1:], distances[i][1:] * px_m):
            if d < 1e-6:
                continue
            tau_j = tau_p_arr[j]
            if not np.isnan(tau_j):
                vals.append(abs(tau_i - tau_j) / d)
        if vals:
            out[i] = float(np.mean(vals))
    return out


def theta_plane_fit(east_m, north_m, tau_p_arr,
                    radius_m: float = None,
                    min_neighbours: int = None) -> tuple:
    """θ (Pa/m) as |∇τp| from a local least-squares plane through τp.

    Fits τp ≈ a·E + b·N + c over every cluster centroid within `radius_m` and
    returns (theta, grad_east, grad_north) with theta = hypot(a, b).

    Averaging over a neighbourhood of radius ≈ L_ss suppresses the local noise
    that dominates a short-lag difference estimator and leaves the slope-scale
    ramp, which is the quantity Meloche et al.'s θ represents (their
    heterogeneity is "a linear increase of the weak layer strength"). The
    gradient vector is returned as well, so an along-slope/cross-slope split
    becomes possible without a second estimator.

    θ is NaN where fewer than `min_neighbours` centroids carry τp inside the
    radius, or where they are too collinear to define a plane. On Jan 18 that
    is 15.8% of start-zone clusters, which costs almost nothing end to end
    because those clusters were already NaN for other reasons.

    `radius_m` / `min_neighbours` default to the module constants, read at call
    time so a caller can override them by setting the module attribute.
    """
    if radius_m is None:
        radius_m = THETA_FIT_RADIUS_M
    if min_neighbours is None:
        min_neighbours = THETA_FIT_MIN_NEIGHBOURS

    east_m = np.asarray(east_m, float)
    north_m = np.asarray(north_m, float)
    tau_p_arr = np.asarray(tau_p_arr, float)
    n = len(tau_p_arr)
    theta = np.full(n, np.nan)
    g_e = np.full(n, np.nan)
    g_n = np.full(n, np.nan)

    ok = ~np.isnan(tau_p_arr)
    if ok.sum() < min_neighbours:
        return theta, g_e, g_n

    from sklearn.neighbors import NearestNeighbors
    pts = np.column_stack([east_m[ok], north_m[ok]])
    tau_ok = tau_p_arr[ok]
    tree = NearestNeighbors(radius=radius_m, algorithm='ball_tree').fit(pts)
    neigh = tree.radius_neighbors(np.column_stack([east_m, north_m]),
                                  return_distance=False)

    for i in range(n):
        if np.isnan(tau_p_arr[i]):
            continue
        sel = neigh[i]
        if len(sel) < min_neighbours:
            continue
        e = pts[sel, 0]
        nn = pts[sel, 1]
        A = np.column_stack([e - e.mean(), nn - nn.mean(), np.ones(len(sel))])
        # rank < 3 means the neighbourhood is a point or a line: no plane.
        coef, _, rank, _ = np.linalg.lstsq(A, tau_ok[sel], rcond=None)
        if rank < 3:
            continue
        g_e[i], g_n[i] = float(coef[0]), float(coef[1])
        theta[i] = float(np.hypot(coef[0], coef[1]))

    return theta, g_e, g_n


def compute_meloche_features(snap_data: dict, cluster_map: np.ndarray,
                              dem: np.ndarray, transform,
                              snap_ts=None) -> pd.DataFrame:
    """
    Compute Meloche et al. (2025) spatial features requiring neighbor information.

    For each cluster computes:
      τ_g  — gravitational shear stress (Pa)
      θ    — WL shear strength gradient to k nearest neighbors (Pa/m)
      Π₁   — elastic dimensionless number: τ_g / (θ·Λ·√(1+δ))
      Π₂   — brittle dimensionless number: Π₁ · √(σ_t/τ_g)
      A_ca — crack arrest length estimate from brittle scaling (m)
      L_t  — quasi-static tensile length (m)

    Parameters match Meloche et al. (2025) Table 1:
      ψ  = slope angle per cluster (from DEM)
      ϕ  = 27° (snow friction angle, fixed)
      δ  = 1   (softening coefficient, fixed — slope scale default)
      L_ss = 20 m (steady-state length, fixed — paper default)

    θ is estimated by THETA_ESTIMATOR ('plane_fit' by default); `transform`
    supplies the pixel size, so θ comes out in Pa/m on any grid resolution.

    Returns DataFrame indexed by cluster_id with Meloche features.
    """
    PHI_DEG = 27.0          # snow friction angle (degrees)
    DELTA   = 1.0           # softening coefficient
    K_NEIGHBORS = 6         # nearest neighbors for θ computation
    G_GRAV  = 9.81          # m/s²
    px_m    = abs(transform.a) if transform is not None else 1.0

    phi_rad = np.radians(PHI_DEG)

    # Compute slope angle per cluster from DEM
    fill_dem = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx   = np.gradient(fill_dem, px_m)
    slope    = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))

    # Compute cluster centroids in pixel space
    cluster_ids = np.unique(cluster_map[~np.isnan(dem)])
    cluster_ids = cluster_ids[cluster_ids > 0]

    centroids = {}  # cluster_id -> (row_mean, col_mean)
    slopes_cl = {}  # cluster_id -> mean slope angle (deg)
    for cid in cluster_ids:
        mask = cluster_map == cid
        rows, cols = np.where(mask)
        if len(rows):
            centroids[cid]  = (float(rows.mean()), float(cols.mean()))
            slopes_cl[cid]  = float(slope[mask].mean())

    # Collect per-cluster τ_p, Λ, E, σ_t from snap_data
    all_rows = []
    for grp, df in snap_data.items():
        if df.empty:
            continue
        df2 = df.copy()
        df2['group'] = grp
        all_rows.append(df2)

    if not all_rows:
        return pd.DataFrame()

    features = pd.concat(all_rows)
    features = features[~features.index.duplicated(keep='first')]
    if 'wl_shear_strength' not in features.columns:
        return pd.DataFrame()

    # Build array of centroid positions for distance computation
    cids_arr   = np.array([cid for cid in features.index
                           if cid in centroids])
    if len(cids_arr) < 2:
        return pd.DataFrame()

    cents = np.array([centroids[c] for c in cids_arr])  # (N, 2) pixel coords
    # px_m converts centroid separations to metres, so θ is Pa/m at any grid
    # resolution rather than Pa per pixel.
    def _scalar(df, cid, col):
        """Safely extract a scalar from a possibly multi-row index."""
        if cid not in df.index:
            return np.nan
        val = df.loc[cid, col]
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        return float(val)

    # wl_shear_strength is stored in Pa (zarr units: Pa)
    tau_p_arr = np.array([_scalar(features, c, 'wl_shear_strength')
                          for c in cids_arr])

    # KNN for θ computation
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(K_NEIGHBORS+1, len(cids_arr)),
                             algorithm='ball_tree').fit(cents)
    distances, indices = nbrs.kneighbors(cents)

    # θ for every cluster up front, by the configured estimator. `cents` are
    # pixel (row, col); row increases southward, so north = -row.
    if THETA_ESTIMATOR == 'plane_fit':
        theta_arr, theta_ge, theta_gn = theta_plane_fit(
            east_m=cents[:, 1] * px_m, north_m=-cents[:, 0] * px_m,
            tau_p_arr=tau_p_arr)
    elif THETA_ESTIMATOR == 'knn':
        theta_arr = theta_knn(tau_p_arr, indices, distances, px_m)
        theta_ge = theta_gn = np.full(len(cids_arr), np.nan)
    else:
        raise ValueError(
            f"THETA_ESTIMATOR must be 'knn' or 'plane_fit', "
            f"got {THETA_ESTIMATOR!r}")
    theta_by_cid = {int(c): theta_arr[i] for i, c in enumerate(cids_arr)}
    theta_vec_by_cid = {int(c): (theta_ge[i], theta_gn[i])
                        for i, c in enumerate(cids_arr)}

    rows_out = []
    for i, cid in enumerate(cids_arr):
        if cid not in features.index:
            continue
        loc_data = features.loc[cid]
        row = (loc_data.iloc[0].copy()
               if isinstance(loc_data, pd.DataFrame)
               else loc_data.copy())
        tau_p = row.get('wl_shear_strength', np.nan)  # Pa (zarr units: Pa)
        Lambda = row.get('Lambda', np.nan)
        E_slab = row.get('E_slab', np.nan)
        sigma_t = row.get('sigma_t', np.nan)
        rho   = row.get('slab_density', np.nan)
        h_m   = row.get('slab_thickness', np.nan)
        psi_deg = slopes_cl.get(cid, np.nan)

        if any(np.isnan(v) for v in [tau_p, Lambda, rho, h_m, psi_deg]):
            rows_out.append({'cluster_id': cid})
            continue

        psi_rad = np.radians(psi_deg)
        sin_psi = np.sin(psi_rad)
        if sin_psi < 0.01:
            rows_out.append({'cluster_id': cid})
            continue

        # Validity guard: use full gravitational shear (not friction-adjusted) to match
        # the tau_g definition in arrest_indices.evaluate() and the paper.
        tau_g_guard = rho * G_GRAV * h_m * sin_psi
        tau_g_min_valid = 40.0   # Pa — below this the scaling law is not applicable
        if tau_g_guard < tau_g_min_valid:
            rows_out.append({'cluster_id': cid, 'tau_g': tau_g_guard,
                             'slope_angle': psi_deg,
                             'note': 'tau_g below valid range for scaling law'})
            continue

        # θ — WL shear strength gradient (Pa/m), by THETA_ESTIMATOR
        theta = float(theta_by_cid.get(int(cid), np.nan))

        if np.isnan(theta) or theta < THETA_MIN:
            rows_out.append({'cluster_id': cid, 'tau_g': tau_g_guard, 'theta': theta,
                             'slope_angle': psi_deg})
            continue

        D_wl_val = row.get('wl_thickness', np.nan)
        if np.isnan(D_wl_val) or D_wl_val <= 0:
            D_wl_val = 0.04   # fallback to Meloche et al. paper default
        tau_p0_val = row.get('wl_shear_strength', np.nan)

        ai = _ai_evaluate(
            rho=rho, h=h_m, psi_deg=psi_deg, E=E_slab,
            sigma_t=sigma_t, D_wl=D_wl_val, G_wl=G_WL,
            theta=theta, nu=NU, phi_deg=PHI_DEG, delta=DELTA,
            tau_p0=tau_p0_val if not np.isnan(tau_p0_val) else None,
            speed_ratio_along=MODE2_SPEED_RATIO,
            speed_ratio_cross=MODE3_SPEED_RATIO,
        )

        tau_g_ai   = ai['tau_g']
        Lambda_ai  = ai['Lambda']
        L_t_ai     = ai.get('L_t', np.nan)
        A_ca_ai    = ai.get('A_ca', np.nan)
        G_slab     = ai.get('G_slab', np.nan)
        tau_p_star = ai.get('tau_p_star', np.nan)
        R0         = ai.get('R0', np.nan)
        A_ca_energy = ai.get('A_ca_energy', np.nan)

        # Dimensionless numbers (Meloche et al. 2025 Eqs. 19, 20)
        denom        = theta * Lambda_ai * np.sqrt(1.0 + DELTA)
        Pi1          = tau_g_ai / denom
        Pi2          = Pi1 * np.sqrt(sigma_t / tau_g_ai)

        # Eq. 20 again under the two layered sigma_t bounds, alongside (never
        # replacing) A_ca_brittle. Only sigma_t and E change: k_f and tau_g
        # come from the bulk slab density, so they are shared.
        layered = _layered_arrest(row, tau_g_ai, theta, psi_deg, h_m, rho,
                                  PHI_DEG, DELTA)

        rows_out.append({
            **layered,
            'cluster_id':    cid,
            'slope_angle':   psi_deg,
            'tau_g':         tau_g_ai,
            'theta':         theta,
            # NaN under the 'knn' estimator, which yields no gradient vector.
            'theta_grad_east':  theta_vec_by_cid.get(int(cid), (np.nan,))[0],
            'theta_grad_north': theta_vec_by_cid.get(int(cid), (np.nan, np.nan))[1],
            # τp carried through so estimate_cross_slope_width can measure both
            # halves of the Gaume ratio from the meloche frame alone.
            'tau_p':         tau_p,
            'Pi1_elastic':   Pi1,
            'Pi2_brittle':   Pi2,
            'Lambda':        Lambda_ai,
            # Mode III elastic length from the current (unpublished) derivation.
            # Written so the data side is ready; propagation only reads it when
            # release_geometry.USE_MODE3_LAMBDA is set.
            'Lambda_cross':  ai.get('Lambda_cross', np.nan),
            'L_t':           L_t_ai,
            # Distance to first slab fracture, along-slope (supershear) and
            # cross-slope (mode III capped at c_s). The ratio is the
            # crack-speed cap applied to lateral propagation.
            'L_dyn':         ai.get('L_dyn',       np.nan),
            'L_dyn_cross':   ai.get('L_dyn_cross', np.nan),
            'mode3_length_ratio': ai.get('mode3_length_ratio', np.nan),
            'A_ca_brittle':  A_ca_ai,
            'rc_wl':         row.get('rc_wl', np.nan),
            'G_slab':        G_slab,
            'tau_p_star':    tau_p_star,
            'R0':            R0,
            'A_ca_energy':   A_ca_energy,
        })

    if not rows_out:
        return pd.DataFrame()
    return pd.DataFrame(rows_out).set_index('cluster_id')


_LAYERED_KEYS = ('E_eff', 'sigma_t_mean', 'sigma_t_wl', 'sigma_t_ligament',
                 'f_mean_weighted', 'n_slab_layers', 'wl_ctrl_index',
                 'wl_ctrl_depth', 'wl_ctrl_thickness', 'K_Ic_ligament',
                 'sigma_t_lig_intact', 'a0_ligament', 'l_ch_ligament')


def _layered_arrest(row, tau_g, theta, psi_deg, h_m, rho_bulk,
                    phi_deg, delta) -> dict:
    """Eq. 20 under the layered sigma_t bounds. NaN when the columns are absent."""
    out = {k: _nan_get(row, k) for k in _LAYERED_KEYS}
    out.update({k: np.nan for k in
                ('Lambda_eff', 'A_ca_mean', 'A_ca_wl', 'A_ca_ratio',
                 'A_ca_ligament')})

    E_eff = out['E_eff']
    D_wl  = _nan_get(row, 'wl_thickness')
    if not (np.isfinite(E_eff) and np.isfinite(D_wl) and D_wl > 0
            and np.isfinite(h_m) and np.isfinite(theta) and theta > 0):
        return out

    E_prime_eff = E_eff / (1.0 - NU ** 2)
    out['Lambda_eff'] = float(np.sqrt(E_prime_eff * h_m / (G_WL / D_wl)))
    k_f = float(_ai.tension_gradient(rho_bulk, psi_deg, phi_deg))

    for key, st in (('A_ca_mean', out['sigma_t_mean']),
                    ('A_ca_wl', out['sigma_t_wl']),
                    ('A_ca_ligament', out['sigma_t_ligament'])):
        if not np.isfinite(st) or st <= 0:
            continue
        L_t = float(_ai.tensile_length(st, k_f))
        if not np.isfinite(L_t):
            continue
        out[key] = float(_ai.arrest_length(
            tau_g, theta, out['Lambda_eff'], st, delta, L_t)[0])

    if (np.isfinite(out['A_ca_mean']) and np.isfinite(out['A_ca_wl'])
            and out['A_ca_wl'] > 0):
        out['A_ca_ratio'] = out['A_ca_mean'] / out['A_ca_wl']
    return out


def _nan_get(row, key):
    v = row.get(key, np.nan)
    if isinstance(v, pd.Series):
        v = v.iloc[0] if len(v) else np.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan



# -----------------------------------------------------------------------
# Per-frame scalar reduction (for visualization)
# -----------------------------------------------------------------------

def reduce_scalars_at_time(ds: xr.Dataset,
                   timestamp: pd.Timestamp,
                   prev_hs: dict,
                   min_depth_cm: float,
                   max_depth_cm: float,
                   wl_method: str = 'simple') -> tuple[dict, dict]:
    """
    Reduce all variables to per-cluster scalars at one timestep.

    prev_hs: dict mapping location -> HS value at previous timestep,
             used to compute dHS/dt.
    """
    ds_t = ds.sel(time=timestamp, method='nearest')

    z   = ds_t['z']
    in_depth = (z <= -min_depth_cm) & (z >= -max_depth_cm) & (~np.isnan(z))

    def layer_min(var):
        return (ds_t[var].where(in_depth)
                         .min(dim='layer')
                         .squeeze([d for d in ['slope','realization'] if d in ds_t.dims])
                         .compute().values)

    def layer_mean(var):
        return (ds_t[var].where(in_depth)
                         .mean(dim='layer')
                         .squeeze([d for d in ['slope','realization'] if d in ds_t.dims])
                         .compute().values)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        hs       = ds_t['HS'].squeeze([d for d in ['slope','realization'] if d in ds_t.dims]).compute().values
        sk38_min = layer_min('sk38')
        ssi_min  = layer_min('ssi')
        sn38_min = layer_min('sn38')
        tg_min   = layer_min('temperature_gradient')
        atg_mean = layer_mean('accumulated_temperature_gradient')
        sdr_min  = layer_min('stab_deformation_rate')

    # dHS/dt in cm/day from previous timestep
    locs = ds.coords['location'].values
    if prev_hs:
        dt_days = (timestamp - prev_hs['time']).total_seconds() / 86400
        if dt_days > 0:
            dhs_dt = (hs - prev_hs['hs']) / dt_days
        else:
            dhs_dt = np.zeros_like(hs)
    else:
        dhs_dt = np.zeros_like(hs)

    # --- Structure variables (WL/slab properties) ---
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        sq_dims   = [d for d in ['slope','realization'] if d in ds_t.dims]
        hs_vals   = ds_t['HS'].squeeze(sq_dims).compute().values
        z_da      = ds_t['z'].squeeze(sq_dims).compute()   # (location, layer)

        if wl_method == 'simple':
            # Fast: bottom 20% of HS = WL proxy, top 80% = slab
            hs_bc     = xr.DataArray(hs_vals, dims=['location'])
            wl_zone   = (z_da < -0.8 * hs_bc) & ~np.isnan(z_da)
            slab_zone = (z_da > -0.8 * hs_bc) & (z_da < 0) & ~np.isnan(z_da)
            wl_burial_raw  = hs_vals * 0.8
            slab_thick_raw = hs_vals * 0.8
        else:
            # Proper: FC/DH grain-type detection per cluster
            import sys as _sys
            from pathlib import Path as _Path
            _sys.path.insert(0, str(_Path(__file__).resolve().parent))
            from analyze_release_zone import split_wl_slab
            z_np  = z_da.values          # (location, layer)
            gt_np = ds_t['grain_type'].squeeze(sq_dims).compute().values
            n_loc = z_np.shape[0]
            wl_mask_np   = np.zeros(z_np.shape, dtype=bool)
            slab_mask_np = np.zeros(z_np.shape, dtype=bool)
            wl_burial_raw  = np.full(n_loc, np.nan)
            slab_thick_raw = np.full(n_loc, np.nan)
            for li in range(n_loc):
                sm, wm, iz = split_wl_slab(gt_np[li], z_np[li])
                if sm is not None:
                    wl_mask_np[li]   = wm
                    slab_mask_np[li] = sm
                    wl_burial_raw[li]  = -iz               # already in m
                    slab_thick_raw[li] = -iz
            wl_zone   = xr.DataArray(wl_mask_np,   dims=z_da.dims)
            slab_zone = xr.DataArray(slab_mask_np, dims=z_da.dims)

        wl_str_raw = (ds_t['shear_strength'].squeeze(sq_dims).where(wl_zone)
                      .mean(dim='layer').compute().values)
        wl_grain_raw = (ds_t['grain_size'].squeeze(sq_dims).where(wl_zone)
                        .mean(dim='layer').compute().values)
        slab_dens_raw = (ds_t['density'].squeeze(sq_dims).where(slab_zone)
                         .mean(dim='layer').compute().values)

        # Meloche et al. (2025) per-cluster parameters
        # E and σ_t from slab density (power laws, van Herwijnen 2016)
        E_slab_raw   = (slab_dens_raw / 300.0)**2.5 * 4.0    # MPa
        sigma_t_raw  = (slab_dens_raw / 300.0)**1.4 * 5.0    # kPa
        # Characteristic elastic length Λ = sqrt(E'·h·D_wl / G_wl)
        G_wl_pa      = 0.2e6                                   # Pa
        nu           = 0.3
        E_prime_raw  = E_slab_raw * 1e6 / (1 - nu**2)         # Pa
        D_wl_m       = np.where(slab_thick_raw > 0,
                                slab_thick_raw * 0.2, np.nan)  # 20% of slab as WL
        h_m          = slab_thick_raw
        Lambda_raw   = np.where(
            (h_m > 0) & (D_wl_m > 0),
            np.sqrt(E_prime_raw * h_m * D_wl_m / G_wl_pa), np.nan)
        # Gravitational driving stress τ_g (Pa)
        # Use fixed slope ψ=32° (mean release zone slope)
        psi, phi = np.radians(32), np.radians(27)
        tau_g_raw = (slab_dens_raw * 9.81 * h_m *
                     np.sin(psi) * (1 - np.tan(phi)/np.tan(psi)))

    return {
        'HS':          hs,
        'dhs_dt':      dhs_dt,
        'sk38_min':    sk38_min,
        'ssi_min':     ssi_min,
        'sn38_min':    sn38_min,
        'tg_min':      np.abs(tg_min),
        'atg_mean':    atg_mean,
        'sdr_min':     sdr_min,
        'wl_strength':  wl_str_raw,
        'wl_grain':     wl_grain_raw,
        'wl_burial':    wl_burial_raw,
        'slab_thick':   slab_thick_raw,
        'slab_density': slab_dens_raw,
        'E_slab':       E_slab_raw,
        'sigma_t':      sigma_t_raw,
        'Lambda':       Lambda_raw,
        'tau_g':        tau_g_raw,
    }, {'hs': hs, 'time': timestamp}