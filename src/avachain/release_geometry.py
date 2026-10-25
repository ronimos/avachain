"""
release_geometry.py — Physically-motivated avalanche release polygon generation.

Provides:
    find_stauchwall()          Walk downslope from trigger until slope < threshold
    estimate_cross_slope_width() Cross-slope arrest width from Gaume (2015) / θ
    project_along_aspect()     Project a UTM point distance along slope aspect
    make_release_polygon_2d()  Full 2D release polygon from four constraints:
                                 1. Meloche A_ca (upslope)
                                 2. ~28° slope threshold (downslope / stauchwall)
                                 3. θ cross-slope gradient (flanks)
                                 4. Start zone KML (hard lateral boundary)
    rasterize_release_polygon() Burn polygon to a depth raster on the DEM grid
    depth_from_snowpack()       Per-pixel slab depth (m) from cluster HS

No CLI. Import from analysis_pipeline.py and scenario_writer.py.

References
----------
Upslope:   Meloche et al. (2025) JGR Earth Surface, doi:10.1029/2025JF008470
Downslope: Perzl (2007) JRC; Swiss ALIP/PRA; Maggioni & Gruber;
           Bühler et al.; Veitinger et al. (2016)
Flanks:    Gaume et al. (2015) The Cryosphere, doi:10.5194/tc-9-795-2015
           + Meloche (2025) cross-slope θ gradient
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

import arrest_indices as _ai

# Default physical parameters
STAUCHWALL_DEG   = 28.0   # slope threshold for downslope arrest (degrees)
FRICTION_DEG     = 27.0   # snow friction angle (degrees), Meloche Table 1
GAUME_ASPECT_CAP = 2.5    # max width/A_ca ratio (prevents runaway cross-slope)
MIN_POLYGON_AREA = 200.0  # m² — discard degenerate polygons smaller than this

# Meloche (2025) directional crack-arrest criterion. When True, _qualifies
# replaces the heuristic Lambda/thickness gradient gates with the field-
# validated A_ca scaling (Eq. 20): arrest a C->N step where the directed WL
# shear-strength gradient makes the arrest length shorter than the step.
# ON since 2026-10-23 (Ron). The Lambda/thickness gates are then skipped, not
# run alongside. Eq. 20 comes from arrest_indices.arrest_length so the C = 0.045
# prefactor cannot be dropped; the inline copy that did drop it was 22x too long.
USE_MELOCHE_ARREST = True
MELOCHE_DELTA      = 1.0   # softening coefficient δ (Meloche default; their Table 1)
NU                 = 0.3   # slab Poisson's ratio

THETA_MIN = 1e-6   # Pa/m; θ below this is treated as no usable gradient

# Cross-slope θ sampling in estimate_cross_slope_width(). The lateral sample has
# to sit near the true cross-slope axis (aspect ± 90°). Selecting it by raster
# row/col instead — "same row ± 50 px, column offset > 5 px" — silently assumes
# the fall line runs north-south. On the Jan 18 ESE start zone (aspect 113-120°,
# cross axis 23-30°) that put the chosen neighbours a mean 30° off the cross
# axis, with some within 1.2° of the fall line, so theta_cross was contaminated
# by the along-slope gradient. Fixed 2026-10-09 in release_areas, ported here.
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
# fracture is shorter cross-slope: L_dyn,III / L_dyn,II = 0.712 at ν = 0.3.
# Unlike USE_MODE3_LAMBDA this is an absolute distance cap, so it does change
# results. Only the restrictive half of mode III is coded — because G < E′ an
# equal-strength slab has a *larger* energy cap cross-slope, but that needs a
# flank strength τ_flank with no parameterisation. Left on because Broberg's
# limit holds independently of any one event; do not tune it to Jan 18.
MODE2_SPEED_RATIO   = _ai.MODE2_SPEED_RATIO   # ȧ / c_s upslope, Meloche et al. (2025)
MODE3_SPEED_RATIO   = _ai.MODE3_SPEED_RATIO   # ȧ / c_s cross-slope, Broberg (1989)
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

# ---- BFS region-size cap ---------------------------------------------------
# Safety cap on the BFS flood-fill.  When it binds, the polygon is labelled
# [CAP-BOUND] so it is never mistaken for a physics-limited result.
# Measured on Jan 18: 500 → 2000 moves median area ~20 % (5 366 → 6 405 m²)
# and best IoU 0.639 → 0.667; region size saturates by ~1 000 clusters so
# any value ≥ 1 000 gives the same answer — raising it is not tuning.
MAX_BFS_CLUSTERS = 2000

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


def mode3_speed_cap_ratio(trigger_cluster_id: int,
                          meloche_df: pd.DataFrame) -> Optional[float]:
    """L_dyn,III / L_dyn,II for the trigger, or None if the cap is disabled.

    Prefers the `mode3_length_ratio` column written by
    compute_meloche_features; falls back to computing it from nu so CSVs
    predating the column still get the cap. Returns None when
    USE_MODE3_SPEED_CAP is off, which restores pre-cap behaviour.
    """
    if not USE_MODE3_SPEED_CAP:
        return None

    if (not meloche_df.empty and 'mode3_length_ratio' in meloche_df.columns
            and trigger_cluster_id in meloche_df.index):
        ratio = float(first_scalar(
            meloche_df.loc[trigger_cluster_id, 'mode3_length_ratio']))
        if np.isfinite(ratio) and 0.0 < ratio <= 1.0:
            return ratio

    # E and rho cancel out of the ratio, so any positive pair works here.
    return float(_ai.mode3_length_ratio(
        4.0e6, 250.0, NU, MODE2_SPEED_RATIO, MODE3_SPEED_RATIO))


def _hop_length_m(neighbours: dict, centroids: dict) -> float:
    """Median centroid separation of adjacent clusters in the BFS graph (m).

    One ring of the flood-fill advances one graph hop, so this is the natural
    per-ring advance and the length scale the anisotropic ring budget is
    expressed in. Taken over every edge rather than per cluster so it is a
    property of the graph, not of where the trigger happens to sit.
    """
    seps = [float(np.hypot(centroids[a][0] - centroids[b][0],
                           centroids[a][1] - centroids[b][1]))
            for a, nbrs in neighbours.items() if a in centroids
            for b in nbrs if b in centroids]
    return float(np.median(seps)) if seps else 0.0


def directional_lambda(props: dict, is_cross: bool) -> float:
    """Elastic length Λ to use for propagation in a given direction.

    Along-slope (mode II) always uses `Lambda`. Cross-slope (mode III) uses
    `Lambda_cross` only when USE_MODE3_LAMBDA is set; otherwise it falls back to
    the mode II value. `elastic_length_cross()` now matches Gaume's own
    antiplane-shear derivation exactly (personal communication, 2026-10-18;
    not yet published), but the flag stays off because a single global NU makes
    Λ_III/Λ_II a constant, which the continuity gate is provably invariant to
    — see ~/avalanche-release-areas/docs/release_area_methods.md §3.

    This is the ONLY place the two directions diverge. If Λ ever carries
    spatially-varying material properties (a per-cluster ν, or a
    bending-stiffness-based Λ_III per Gaume's still-open regime question),
    update `arrest_indices.elastic_length_cross()` and set USE_MODE3_LAMBDA —
    no other change should be needed.
    """
    if is_cross and USE_MODE3_LAMBDA:
        lam = props.get('Lambda_cross', np.nan)
        if not np.isnan(lam):
            return lam
    return props.get('Lambda', np.nan)


def first_scalar(value):
    """Unwrap a pandas lookup that returned a Series/DataFrame to one scalar.

    Cluster ids are not guaranteed unique across concatenated feature frames,
    so .loc[cid, col] can yield a Series rather than a scalar.
    """
    if isinstance(value, pd.DataFrame):
        return value.iloc[0, 0]
    if isinstance(value, pd.Series):
        return value.iloc[0]
    return value


# -----------------------------------------------------------------------
# Polygon cleanup
# -----------------------------------------------------------------------

def fill_polygon_holes(geom, max_hole_frac: float | None = None):
    """Remove interior rings (holes) from a (Multi)Polygon.

    Holes arise when an interior cluster arrests inside the released set, or
    from raster pixelation; com1DFA expects a simply-connected release area,
    and the observed (hand-digitised) polygon used for IoU has no holes, so
    filling also makes that comparison fair.

    max_hole_frac : None  -> fill ALL holes (default; what com1DFA wants).
                    float -> keep interior rings whose area exceeds
                             max_hole_frac * exterior area. A large hole may be
                             a genuine non-released island (rock outcrop, stable
                             interior) that shouldn't be claimed as release.
    """
    from shapely.geometry import Polygon, MultiPolygon

    def _fill(poly):
        if not poly.interiors:
            return poly
        if max_hole_frac is None:
            return Polygon(poly.exterior)
        ext_area = Polygon(poly.exterior).area
        keep = [r for r in poly.interiors
                if Polygon(r).area > max_hole_frac * ext_area]
        return Polygon(poly.exterior, keep)

    if geom is None or geom.is_empty:
        return geom
    if geom.geom_type == 'Polygon':
        return _fill(geom)
    if geom.geom_type == 'MultiPolygon':
        return MultiPolygon([_fill(p) for p in geom.geoms])
    return geom


# -----------------------------------------------------------------------
# Terrain helpers
# -----------------------------------------------------------------------

def compute_slope_aspect(dem: np.ndarray,
                         pixel_size: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
    """
    Return slope (degrees) and aspect (degrees from N, clockwise) grids.

    Parameters
    ----------
    dem        : 2D elevation array (m), NaN for nodata
    pixel_size : pixel size in metres (assumes square pixels)
    """
    fill = np.where(np.isnan(dem), np.nanmean(dem), dem)
    dy, dx = np.gradient(fill, pixel_size)
    slope  = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    aspect = np.degrees(np.arctan2(-dx, dy)) % 360.0
    return slope, aspect


def pixel_to_utm(row: int, col: int, transform) -> tuple[float, float]:
    """Convert raster pixel (row, col) to UTM (x, y) using Affine transform."""
    x = transform.c + col * transform.a + row * transform.b
    y = transform.f + col * transform.d + row * transform.e
    return x, y


def utm_to_pixel(x: float, y: float, transform) -> tuple[int, int]:
    """Convert UTM (x, y) to nearest raster pixel (row, col)."""
    col = int(round((x - transform.c) / transform.a))
    row = int(round((y - transform.f) / transform.e))
    return row, col


def cluster_pixel_index(cluster_map: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Map every positive cluster id to its (rows, cols) pixel arrays.

    Built in a single sort instead of one full-raster scan per cluster, which
    is what made the per-cluster `np.argwhere(cluster_map == cid)` calls the
    dominant cost of polygon generation.
    """
    flat    = cluster_map.ravel()
    valid   = np.flatnonzero(flat > 0)
    if len(valid) == 0:
        return {}
    ids     = flat[valid].astype(np.int64)
    order   = np.argsort(ids, kind='stable')
    ids     = ids[order]
    linear  = valid[order]
    splits  = np.flatnonzero(np.diff(ids)) + 1
    ncols   = cluster_map.shape[1]
    return {int(ids[seg[0]]): (linear[seg] // ncols, linear[seg] % ncols)
            for seg in np.split(np.arange(len(ids)), splits) if len(seg)}


def cluster_centroids_px(pixel_index: dict) -> dict[int, tuple[float, float]]:
    """Per-cluster centroid in pixel (row, col) coordinates."""
    return {cid: (float(rows.mean()), float(cols.mean()))
            for cid, (rows, cols) in pixel_index.items()}


# -----------------------------------------------------------------------
# Downslope: stauchwall location
# -----------------------------------------------------------------------

def find_stauchwall(trigger_row: int,
                    trigger_col: int,
                    slope_grid: np.ndarray,
                    aspect_grid: np.ndarray,
                    transform=None,
                    threshold_deg: float = STAUCHWALL_DEG,
                    max_steps: int = 500
                    ) -> tuple[int, int]:
    """
    Walk downslope from trigger pixel until slope drops below threshold.

    Each step advances one pixel width along the *local* aspect, carrying
    sub-pixel position across iterations. Returns (row, col) of the stauchwall
    pixel; if slope never drops below threshold within max_steps, returns the
    pixel reached at max_steps — a conservative (large) release estimate.

    Taking `int(np.sign(...))` of the direction components instead — which this
    did before 2026-10-09 — quantises every step to a pure diagonal: sign() is
    ±1 for any nonzero component, so only the four bearings 45/135/225/315 are
    reachable and the walk degenerates into a straight ray that never follows
    terrain. On the Little Professor ESE start zone (aspect 110-120°) that sent
    every trigger off at 135°, a systematic +14° median clockwise error that
    rotated the release polygons toward south, on 100% of start-zone pixels.

    `transform` is accepted for call-site compatibility and unused: the walk is
    in pixel units and one step is one pixel width.

    References: Perzl (2007) JRC; Swiss ALIP/PRA; Maggioni & Gruber;
                Bühler et al.; Veitinger et al. (2016).
    """
    nrows, ncols = slope_grid.shape
    row, col     = trigger_row, trigger_col
    row_f, col_f = float(trigger_row), float(trigger_col)

    for _ in range(max_steps):
        if slope_grid[row, col] < threshold_deg:
            return row, col
        asp_rad = np.radians(aspect_grid[row, col])
        # Downslope unit vector, map frame -> pixel frame:
        # east = +sin(aspect) -> +col, north = +cos(aspect) -> -row.
        row_f += -np.cos(asp_rad)
        col_f +=  np.sin(asp_rad)
        new_row = int(round(row_f))
        new_col = int(round(col_f))
        if not (0 <= new_row < nrows and 0 <= new_col < ncols):
            break
        if np.isnan(slope_grid[new_row, new_col]):
            break
        row, col = new_row, new_col

    return row, col


# -----------------------------------------------------------------------
# Cross-slope: Gaume (2015) + θ
# -----------------------------------------------------------------------

def _tau_p_lookup(meloche_df: pd.DataFrame,
                  snap_features: Optional[pd.DataFrame]) -> tuple:
    """Pick the frame/column holding weak-layer shear strength τp.

    compute_meloche_features() does not carry τp through to its output on older
    CSVs, so the value normally lives in the profile_features frame. Checking
    meloche_df first keeps hand-assembled frames that do carry it working.
    """
    for frame in (meloche_df, snap_features):
        if frame is None or frame.empty:
            continue
        for col in ('tau_p', 'wl_shear_strength'):
            if col in frame.columns:
                return frame, col
    return None, None


def _sector_theta(offsets, axis_x: float, axis_y: float,
                  tau_trigger: float, tau_frame, tau_col: str,
                  n_neighbors: int) -> float:
    """Mean |Δτp|/d over the nearest clusters lying along a given axis.

    `offsets` is a prebuilt list of (cid, dx, dy, dist_m) already filtered to
    the [MIN_SEP, MAX_SEP] separation band. A cluster qualifies when its
    separation points within THETA_CROSS_SECTOR_DEG of ±(axis_x, axis_y) —
    |d·axis|, so both halves of the axis count.

    Used for both the along-slope and cross-slope θ so the ratio that sets the
    Gaume width is formed from two estimates made the same way, at the same
    lag, from the same τp field.
    """
    cos_min = np.cos(np.radians(THETA_CROSS_SECTOR_DEG))
    sel = sorted(((cid, d) for cid, dx, dy, d in offsets
                  if abs(dx * axis_x + dy * axis_y) / d >= cos_min),
                 key=lambda t: t[1])[:n_neighbors]

    vals = []
    for cid, dist_m in sel:
        if cid not in tau_frame.index:
            continue
        tau_j = float(pd.to_numeric(
            first_scalar(tau_frame.loc[cid, tau_col]), errors='coerce'))
        if not np.isnan(tau_j) and dist_m > 0:
            vals.append(abs(tau_trigger - tau_j) / dist_m)
    return float(np.mean(vals)) if vals else np.nan


def estimate_cross_slope_width(
        trigger_cluster_id: int,
        A_ca: float,
        meloche_df: pd.DataFrame,
        cluster_map: np.ndarray,
        transform,
        snap_features: Optional[pd.DataFrame] = None,
        pixel_index: Optional[dict] = None,
        n_lateral_neighbors: int = 6,
        gaume_aspect_cap: float = GAUME_ASPECT_CAP,
        aspect_deg: Optional[float] = None) -> float:
    """
    Estimate cross-slope release width (m) using Gaume et al. (2015) / θ.

    Implements the Gaume et al. (2015) / Meloche (2025) framework: cross-slope
    crack arrest is controlled by the WL shear-strength gradient θ in the
    lateral direction. A higher cross-slope θ means stronger heterogeneity and
    earlier arrest → narrower release. Width is A_ca × (θ_along/θ_cross),
    capped at gaume_aspect_cap × A_ca.

    Falls back to A_ca whenever θ, τp or `aspect_deg` is unavailable.

    `aspect_deg` is the downslope bearing at the trigger and is what defines
    "cross-slope": the lateral sample is restricted to a ±THETA_CROSS_SECTOR_DEG
    sector about the cross axis (aspect ± 90°), on either flank. Without it
    there is no cross-slope direction to sample along, so the θ widening is
    skipped rather than taken along the raster axes — which is what this did
    before 2026-10-09 ("same row ± 50 px"), silently assuming a north-south
    fall line and letting the along-slope gradient leak into theta_cross.

    Both halves of the ratio are measured here, by `_sector_theta`, from the
    same τp field in the same separation band: θ_along about the fall line,
    θ_cross about the cross axis. Taking the numerator from the meloche CSV
    instead pairs an *isotropic* k-NN mean at a ~3.3 m lag with a directional
    mean at 5-50 m, and since θ(d) ≈ 3.87 + 72/d Pa/m on this slope the lag
    mismatch alone inflates the ratio several-fold and drives it into
    gaume_aspect_cap. The CSV θ survives only as a fallback numerator when no
    cluster lies in the along-slope sector.

    References
    ----------
    Gaume et al. (2015) The Cryosphere 9(2):795–804
    Meloche et al. (2025) JGR Earth Surface, Fig.13 and Sec.4
    """
    if meloche_df.empty or 'theta' not in meloche_df.columns:
        return A_ca

    if aspect_deg is None or np.isnan(aspect_deg):
        return A_ca

    try:
        theta_down = float(meloche_df.loc[trigger_cluster_id, 'theta'])
    except (KeyError, TypeError):
        return A_ca

    if np.isnan(theta_down) or theta_down < THETA_MIN:
        return A_ca

    tau_frame, tau_col = _tau_p_lookup(meloche_df, snap_features)
    if tau_frame is None or trigger_cluster_id not in tau_frame.index:
        return A_ca
    tau_trigger = float(pd.to_numeric(
        first_scalar(tau_frame.loc[trigger_cluster_id, tau_col]), errors='coerce'))
    if np.isnan(tau_trigger):
        return A_ca

    if pixel_index is None:
        pixel_index = cluster_pixel_index(cluster_map)
    centroids = cluster_centroids_px(pixel_index)
    if trigger_cluster_id not in centroids:
        return A_ca
    t_row, t_col = centroids[trigger_cluster_id]
    t_x, t_y = pixel_to_utm(int(round(t_row)), int(round(t_col)), transform)

    # Separation of every other cluster from the trigger, in the map frame,
    # restricted once to the usable band and reused for both axes.
    asp_rad  = np.radians(float(aspect_deg))
    fall_x, fall_y   = np.sin(asp_rad), np.cos(asp_rad)
    cross_x, cross_y = -fall_y, fall_x

    offsets = []
    for cid, (c_row, c_col) in centroids.items():
        if cid == trigger_cluster_id:
            continue
        c_x, c_y = pixel_to_utm(int(round(c_row)), int(round(c_col)), transform)
        dx, dy   = c_x - t_x, c_y - t_y
        dist_m   = float(np.hypot(dx, dy))
        if THETA_CROSS_MIN_SEP_M <= dist_m <= THETA_CROSS_MAX_SEP_M:
            offsets.append((cid, dx, dy, dist_m))
    if not offsets:
        return A_ca

    theta_cross = _sector_theta(offsets, cross_x, cross_y, tau_trigger,
                                tau_frame, tau_col, n_lateral_neighbors)
    if np.isnan(theta_cross):
        return A_ca

    # Numerator measured the same way, about the fall line. Falling back to the
    # CSV θ reintroduces the estimator/lag mismatch, so it is a last resort.
    theta_along = _sector_theta(offsets, fall_x, fall_y, tau_trigger,
                                tau_frame, tau_col, n_lateral_neighbors)
    if np.isnan(theta_along) or theta_along < THETA_MIN:
        theta_along = theta_down

    if theta_cross > THETA_MIN:
        width_factor = min(theta_along / theta_cross, gaume_aspect_cap)
    else:
        width_factor = gaume_aspect_cap

    return float(A_ca * width_factor)


# -----------------------------------------------------------------------
# Upslope projection
# -----------------------------------------------------------------------

def project_along_aspect(x: float, y: float,
                          distance: float,
                          aspect_deg: float,
                          upslope: bool = True) -> tuple[float, float]:
    """
    Project a UTM point by `distance` metres along (or against) aspect.

    Parameters
    ----------
    x, y        : UTM coordinates of start point
    distance    : metres to project
    aspect_deg  : slope aspect in degrees clockwise from north
    upslope     : if True, project upslope (opposite of aspect)
    """
    asp_rad = np.radians(aspect_deg)
    dx = np.sin(asp_rad) * distance
    dy = np.cos(asp_rad) * distance
    sign = -1 if upslope else 1
    return x + sign * dx, y + sign * dy


# -----------------------------------------------------------------------
# Full 2D release polygon
# -----------------------------------------------------------------------

def make_release_polygon_2d(
        trigger_cluster_id: int,
        A_ca: float,
        meloche_df: pd.DataFrame,
        cluster_map: np.ndarray,
        dem: np.ndarray,
        transform,
        start_zone_mask: Optional[np.ndarray] = None,
        snap_features: 'Optional[pd.DataFrame]' = None,
        size_factor: float = 1.0,
        stauchwall_deg: float = STAUCHWALL_DEG,
        max_clusters: int = MAX_BFS_CLUSTERS,
        use_propagation: bool = True,
        pixel_index: Optional[dict] = None,
        state_out: Optional[dict] = None):
    """
    Build a 2D release polygon.

    MASK_MODE == 'diagnostic' ignores start_zone_mask (no BFS gate, no clip).
    In any ARREST_MODE but 'legacy' there is no rectangle fallback and no
    MIN_POLYGON_AREA floor: a trigger-only arrest is the result, not a failure.

    Primary method (use_propagation=True):
        BFS cluster flood-fill with per-direction arrest criteria:
          - Upslope:   Meloche brittle Π₁ at candidate < Π₁ at trigger
          - Downslope: slope < stauchwall_deg (~28°)
          - Lateral:   distance > d_lat — the Gaume θ-ratio cross-slope width
                       (floor 15 m), additionally capped by the mode III crack
                       speed when USE_MODE3_SPEED_CAP is on (currently off)
        size_factor relaxes (>1) or tightens (<1) arrest thresholds.

    Fallback (use_propagation=False, or if propagation returns None):
        Oriented rectangle from A_ca + stauchwall + Gaume cross-slope width.

    Returns shapely Polygon (UTM) or None.
    """
    if MASK_MODE not in ('gate', 'diagnostic'):
        raise ValueError(f"unknown MASK_MODE {MASK_MODE!r}")
    if MASK_MODE == 'diagnostic':
        start_zone_mask = None      # mask is used only in reporting

    slope_grid, _ = compute_slope_aspect(dem)

    # --- Build start zone polygon for post-clipping ---
    # Both the BFS and fallback paths must clip to start zone + stauchwall
    from shapely.geometry import Polygon, MultiPolygon
    from shapely.ops import unary_union
    import rasterio.features

    sz_polygon = None
    if start_zone_mask is not None:
        shapes = list(rasterio.features.shapes(
            start_zone_mask.astype(np.uint8),
            mask=start_zone_mask, transform=transform))
        if shapes:
            sz_polygon = unary_union(
                [Polygon(s['coordinates'][0]) for s, v in shapes if v == 1])

    # Build stauchwall mask: only terrain ≥ stauchwall_deg can be in release
    stauchwall_mask = slope_grid >= stauchwall_deg
    sw_shapes = list(rasterio.features.shapes(
        stauchwall_mask.astype(np.uint8),
        mask=stauchwall_mask, transform=transform))
    sw_polygon = None
    if sw_shapes:
        sw_polygon = unary_union(
            [Polygon(s['coordinates'][0]) for s, v in sw_shapes if v == 1])
        # Simplify to keep intersection fast — 1m tolerance at 1m grid
        sw_polygon = sw_polygon.simplify(1.0, preserve_topology=True)

    def _clip_polygon(poly, min_area=MIN_POLYGON_AREA):
        """Clip polygon to start zone AND stauchwall, return largest piece."""
        if poly is None or poly.is_empty:
            return None
        if sz_polygon is not None:
            poly = poly.intersection(sz_polygon)
        if sw_polygon is not None:
            poly = poly.intersection(sw_polygon)
        if poly.is_empty:
            return None
        # Intersection can produce GeometryCollection (mix of polygons,
        # lines, points at boundaries). Extract only polygon geometries.
        if poly.geom_type == 'GeometryCollection':
            polys = [g for g in poly.geoms
                     if g.geom_type in ('Polygon', 'MultiPolygon')]
            if not polys:
                return None
            poly = unary_union(polys)
        if poly.geom_type == 'MultiPolygon':
            poly = max(poly.geoms, key=lambda p: p.area)
        if poly.is_empty or poly.area < min_area:
            return None
        return poly

    energy_mode = ARREST_MODE != 'legacy'    # energy and the baselines

    # --- Primary: BFS propagation ---
    if use_propagation and (energy_mode or not meloche_df.empty):
        polygon, failed = propagate_release(
            trigger_cluster_id = trigger_cluster_id,
            meloche_df         = meloche_df,
            cluster_map        = cluster_map,
            dem                = dem,
            slope_grid         = slope_grid,
            transform          = transform,
            start_zone_mask    = start_zone_mask,
            stauchwall_deg     = stauchwall_deg,
            size_factor        = size_factor,
            A_ca               = A_ca,
            snap_features      = snap_features,
            max_clusters       = max_clusters,
            pixel_index        = pixel_index,
            state_out          = state_out,
        )
        if polygon is not None:
            polygon = _clip_polygon(polygon, 0.0 if energy_mode else MIN_POLYGON_AREA)
            if polygon is not None:
                return polygon
        if energy_mode:
            return None
        print(f"  propagate_release returned None for cluster {trigger_cluster_id}"
              f" — falling back to rectangle")

    # --- Fallback: oriented rectangle ---
    _, aspect_grid = compute_slope_aspect(dem)
    trig_px = np.argwhere(cluster_map == trigger_cluster_id)
    if len(trig_px) == 0:
        return None
    t_row, t_col = trig_px.mean(axis=0).astype(int)
    t_x, t_y    = pixel_to_utm(t_row, t_col, transform)
    t_aspect    = float(aspect_grid[t_row, t_col])

    up_x, up_y  = project_along_aspect(t_x, t_y, A_ca * size_factor,
                                        t_aspect, upslope=True)
    sw_row, sw_col = find_stauchwall(t_row, t_col, slope_grid, aspect_grid,
                                     transform, threshold_deg=stauchwall_deg)
    sw_x, sw_y  = pixel_to_utm(sw_row, sw_col, transform)
    half_width  = estimate_cross_slope_width(
        trigger_cluster_id, A_ca, meloche_df, cluster_map, transform,
        snap_features=snap_features, aspect_deg=t_aspect
    ) * size_factor / 2.0

    asp_rad = np.radians(t_aspect)
    fall_x, fall_y   = np.sin(asp_rad), np.cos(asp_rad)
    cross_x, cross_y = -fall_y, fall_x
    corners = [
        (up_x - cross_x * half_width, up_y - cross_y * half_width),
        (up_x + cross_x * half_width, up_y + cross_y * half_width),
        (sw_x + cross_x * half_width, sw_y + cross_y * half_width),
        (sw_x - cross_x * half_width, sw_y - cross_y * half_width),
    ]
    polygon = Polygon(corners)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)

    return _clip_polygon(polygon)



# -----------------------------------------------------------------------
# Depth raster
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Cluster-level crack propagation (BFS flood-fill)
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Connected-region instability approach (replaces BFS crack propagation)
# -----------------------------------------------------------------------

def propagate_release(
        trigger_cluster_id: int,
        meloche_df: pd.DataFrame,
        cluster_map: np.ndarray,
        dem: np.ndarray,
        slope_grid: np.ndarray,
        transform,
        start_zone_mask: Optional[np.ndarray] = None,
        stauchwall_deg: float = STAUCHWALL_DEG,
        size_factor: float = 1.0,
        A_ca: Optional[float] = None,
        snap_features: 'Optional[pd.DataFrame]' = None,
        k_neighbours: int = 8,
        max_clusters: int = MAX_BFS_CLUSTERS,
        wave_callback=None,
        pixel_index: Optional[dict] = None,
        aspect_grid: Optional[np.ndarray] = None,
        state_out: Optional[dict] = None):
    """
    Identify release zone as the connected region of clusters with
    Pi1 >= Pi1_trigger that is reachable from the trigger cluster.

    ARREST_MODE == 'energy' hands off to propagate_release_energy(), and
    'slope_only' / 'stability_connected' to baselines.py; pixel_index,
    aspect_grid and state_out are used only by those modes.

    Parameters
    ----------
    wave_callback : callable, optional
        If provided, called once per BFS ring with signature:
            wave_callback(ring_idx, propagated_cids, arrested_list)
        where arrested_list is a list of {'cid': int, 'reason': str}.
        Used by animate_propagation.py for ring-level visualisation.

    Physical basis (Gaume et al. 2015, Meloche et al. 2025):
      - Pi1 = tau_g / (theta * Lambda * sqrt(1+delta))
      - Where Pi1 >= Pi1_trigger the WL crack propagates without arrest
      - Where Pi1 < Pi1_trigger the crack arrests (stronger / more
        heterogeneous terrain)
      - The release zone is the CONNECTED region of Pi1 >= threshold
        that contains the trigger cluster, further constrained by:
          * slope >= stauchwall_deg (terrain must support release)
          * within start_zone_mask (hard operational boundary)
          * distance <= A_ca upslope (Meloche upslope arrest distance)

    size_factor scales the Pi1 threshold:
      > 1.0 → stricter threshold → smaller release (less terrain qualifies)
      < 1.0 → looser threshold  → larger release  (more terrain qualifies)
      = 1.0 → baseline: all terrain as unstable as trigger
    """
    if ARREST_MODE == 'energy':
        return propagate_release_energy(
            trigger_cluster_id, cluster_map, dem, slope_grid, transform,
            start_zone_mask=start_zone_mask, stauchwall_deg=stauchwall_deg,
            snap_features=snap_features, k_neighbours=k_neighbours,
            max_clusters=max_clusters, pixel_index=pixel_index,
            aspect_grid=aspect_grid, wave_callback=wave_callback,
            state_out=state_out)
    if ARREST_MODE in ('slope_only', 'stability_connected'):
        from baselines import propagate_baseline
        return propagate_baseline(
            ARREST_MODE, trigger_cluster_id, cluster_map, dem, slope_grid,
            transform, start_zone_mask=start_zone_mask,
            stauchwall_deg=stauchwall_deg, snap_features=snap_features,
            k_neighbours=k_neighbours, max_clusters=max_clusters,
            pixel_index=pixel_index, state_out=state_out)
    if ARREST_MODE != 'legacy':
        raise ValueError(f"unknown ARREST_MODE {ARREST_MODE!r}")

    from collections import deque
    from shapely.geometry import MultiPolygon
    from shapely.ops import unary_union
    import rasterio.features

    if meloche_df.empty or trigger_cluster_id not in meloche_df.index:
        return None, set()

    # --- Pi1 at trigger cluster ---
    loc = meloche_df.loc[trigger_cluster_id]
    row = loc.iloc[0] if isinstance(loc, pd.DataFrame) else loc

    col = 'Pi1_elastic' if 'Pi1_elastic' in meloche_df.columns else None
    if col is None:
        return None, set()

    pi1_raw = row[col]
    pi1_trigger = float(pi1_raw.iloc[0] if isinstance(pi1_raw, pd.Series)
                        else pi1_raw)
    if np.isnan(pi1_trigger) or pi1_trigger <= 0:
        return None, set()

    # Pi1 retained for reporting only
    threshold = pi1_trigger  # informational

    # Absolute gravitational-driving-stress floor for crack propagation (Pa).
    # tau_g is the driving LOAD on the weak layer; below this value the WL
    # cannot sustain dynamic fracture, regardless of slope or trigger. Unlike a
    # relative floor, an absolute one can represent a stable slope: if no
    # connected terrain exceeds it, the crack does not propagate.
    # Jan 18 reference: ~540 Pa inside release, ~230 Pa adjacent (non-released).
    # A floor in 230-540 reproduces that boundary on tau_g alone; ~350 is a
    # lower physical bound. Sweep on the re-run.
    TAU_G_ABS_FLOOR = 350.0   # Pa

    print(f"    Pi1_trigger={pi1_trigger:.3f}  "
          f"tau_g_floor={TAU_G_ABS_FLOOR:.0f}Pa  "
          f"size_factor={size_factor:.2f}")

    # --- Build per-cluster lookup: Pi1, mean slope ---
    def _scalar(cid, c):
        if cid not in meloche_df.index:
            return np.nan
        v = meloche_df.loc[cid, c]
        if isinstance(v, pd.DataFrame): v = v.iloc[0][c]
        elif isinstance(v, pd.Series):  v = v.iloc[0]
        return float(v)

    def _mean_slope(cid):
        pxs = np.argwhere(cluster_map == cid)
        if len(pxs) == 0:
            return 0.0
        return float(slope_grid[pxs[:, 0], pxs[:, 1]].mean())

    # --- Precompute trigger centroid for upslope distance cap ---
    t_pxs = np.argwhere(cluster_map == trigger_cluster_id)
    if len(t_pxs) == 0:
        return None, set()
    t_row  = int(t_pxs.mean(axis=0)[0])
    t_col  = int(t_pxs.mean(axis=0)[1])
    t_x, t_y = pixel_to_utm(t_row, t_col, transform)
    _, aspect_grid = compute_slope_aspect(dem)
    t_aspect = float(aspect_grid[t_row, t_col])

    # Upslope distance cap: A_ca (Meloche) — no upslope limit for downslope/lateral
    d_up = (max(A_ca, 20.0) if A_ca else 50.0) * size_factor

    # Stauchwall distance (downslope hard cap)
    sw_row, sw_col = find_stauchwall(
        t_row, t_col, slope_grid, aspect_grid, transform,
        threshold_deg=stauchwall_deg)
    sw_x, sw_y = pixel_to_utm(sw_row, sw_col, transform)
    d_down = max(float(np.sqrt((sw_x - t_x)**2 + (sw_y - t_y)**2)),
                 20.0) * size_factor

    print(f"    distance caps: up={d_up:.0f}m  down={d_down:.0f}m")

    # --- Build neighbour graph (k nearest centroids) ---
    from sklearn.neighbors import NearestNeighbors
    cids = np.array([c for c in np.unique(cluster_map) if c > 0])
    pxs  = np.array([np.argwhere(cluster_map == c).mean(axis=0) for c in cids])
    k_actual = min(k_neighbours + 1, len(cids))
    nbrs     = NearestNeighbors(n_neighbors=k_actual).fit(pxs)
    _, idxs  = nbrs.kneighbors(pxs)
    neighbours = {int(cids[i]): [int(cids[j]) for j in idxs[i][1:]]
                  for i in range(len(cids))}

    # Centroids in UTM
    # round, not truncate: matches the standalone (int() put a centroid up to
    # one pixel upslope/west of the standalone's, changing <=9 clusters).
    centroids = {int(c): pixel_to_utm(int(round(pxs[i][0])), int(round(pxs[i][1])), transform)
                 for i, c in enumerate(cids)}

    # WL shear strength at trigger (Pa) — crack arrests where tau_p increases
    # tau_p lives in snap_features (release_zone_features CSV), not meloche_df
    tau_p_trigger = None
    if snap_features is not None and trigger_cluster_id in snap_features.index:
        raw = snap_features.loc[trigger_cluster_id, 'wl_shear_strength'] \
              if 'wl_shear_strength' in snap_features.columns else np.nan
        if isinstance(raw, pd.Series): raw = raw.iloc[0]
        v = float(raw)  # Pa (zarr units: Pa)
        tau_p_trigger = v if not np.isnan(v) and v > 0 else None
    if tau_p_trigger is None:
        print(f"    Warning: no tau_p for trigger {trigger_cluster_id}, "
              f"WL strength criterion disabled")

    # Lateral distance cap from Gaume cross-slope width estimate
    A_ca_base   = A_ca if A_ca else 50.0
    d_lat_gaume = estimate_cross_slope_width(
        trigger_cluster_id, A_ca_base,
        meloche_df, cluster_map, transform,
        snap_features=snap_features, aspect_deg=t_aspect) * size_factor

    # Mode III crack-speed cap. Mode III cannot exceed c_s where upslope runs
    # supershear at ~1.6 c_s; the slower crack builds slab tension faster per
    # unit advance, so first slab fracture — and hence arrest — comes sooner
    # cross-slope. Composed as a min() with the Gaume width because arrest
    # happens at whichever constraint binds first.
    d_lat       = d_lat_gaume
    mode3_ratio = mode3_speed_cap_ratio(trigger_cluster_id, meloche_df)
    if mode3_ratio is not None:
        d_lat_speed = A_ca_base * mode3_ratio * size_factor
        d_lat = min(d_lat, d_lat_speed)
        print(f"    d_lat={max(d_lat, 15.0):.0f}m  "
              f"(gaume={d_lat_gaume:.0f}m, mode3_speed_cap={d_lat_speed:.0f}m "
              f"@ ratio {mode3_ratio:.3f})")
    else:
        print(f"    d_lat={max(d_lat, 15.0):.0f}m  (gaume only, "
              f"mode III speed cap off)")
    d_lat = max(d_lat, 15.0)

    tau_p_str = f"{tau_p_trigger:.0f} Pa" if tau_p_trigger else "N/A"

    # Elastic length Λ threshold for lateral arrest (Gaume et al. 2015)
    # Use median Λ of all clusters within the connected region rather than
    # trigger's own Λ — the trigger may be an outlier (lowest Λ in the zone)
    # and using it would block ALL lateral propagation immediately.
    # Median reflects the typical slab stiffness of the release terrain.
    lambda_trigger = None
    if not meloche_df.empty and 'Lambda' in meloche_df.columns:
        # Use median Λ of all start-zone clusters as the baseline
        lam_vals = meloche_df['Lambda'].dropna()
        if len(lam_vals):
            lam_raw = float(lam_vals.median())
            if lam_raw > 0:
                lambda_trigger = lam_raw
                print(f"    Lambda_median={lambda_trigger:.2f}m  "
                      f"(trigger own={_scalar(trigger_cluster_id, 'Lambda'):.2f}m)")

    print(f"    tau_p_trigger={tau_p_str}  d_lat={d_lat:.0f}m")

    # --- Per-cluster property lookup helpers ---
    def _get_props(cid):
        """Return dict of crack-relevant properties for a cluster."""
        props = {'cid': cid}
        if not meloche_df.empty and cid in meloche_df.index:
            for col in ['Lambda', 'Lambda_cross', 'tau_g', 'rc_wl', 'L_t']:
                if col in meloche_df.columns:
                    v = meloche_df.loc[cid, col]
                    if isinstance(v, pd.DataFrame): v = v.iloc[0][col]
                    elif isinstance(v, pd.Series):  v = v.iloc[0]
                    props[col] = float(v)
        if snap_features is not None and cid in snap_features.index:
            for col in ['slab_thickness', 'slab_density',
                        'wl_shear_strength', 'sigma_t']:
                if col in snap_features.columns:
                    v = snap_features.loc[cid, col]
                    if isinstance(v, pd.Series): v = v.iloc[0]
                    try:
                        props[col] = float(v)
                    except (ValueError, TypeError):
                        pass
        return props

    # Pre-fetch trigger properties as baseline
    trigger_props = _get_props(trigger_cluster_id)

    def _direction(cid):
        """(dist, is_upslope, is_downslope, is_cross) of a cluster from the trigger.

        Within ±45° of the downslope bearing is downslope, within ±45° of the
        opposite is upslope, everything else is cross-slope (mode III / flank).
        None when the cluster has no centroid.
        """
        xy = centroids.get(cid)
        if xy is None:
            return None
        asp_rad        = np.radians(t_aspect)
        fall_x, fall_y = np.sin(asp_rad), np.cos(asp_rad)
        dx, dy         = xy[0] - t_x, xy[1] - t_y
        dist           = float(np.hypot(dx, dy))
        dot            = (dx * fall_x + dy * fall_y) / max(dist, 1e-6)

        is_downslope = dot >  0.707
        is_upslope   = dot < -0.707
        return dist, is_upslope, is_downslope, not is_upslope and not is_downslope

    def _qualifies(cid, current_props):
        """
        Does this cluster qualify for crack propagation?
        Returns (True, 'propagated') or (False, reason_string).
        """
        # Hard boundary: start zone KML
        if start_zone_mask is not None:
            px = np.argwhere(cluster_map == cid)
            if len(px) == 0:
                return False, 'no_data'
            r, c = int(px.mean(axis=0)[0]), int(px.mean(axis=0)[1])
            if not start_zone_mask[r, c]:
                return False, 'outside_start_zone'

        # Direction and distance from trigger
        direction = _direction(cid)
        if direction is None:
            return False, 'no_data'
        dist, is_upslope, is_downslope, is_cross = direction

        # Distance caps: upslope A_ca, downslope stauchwall, lateral Gaume width
        # composed with the mode III crack-speed cap.
        if is_upslope   and dist > d_up:   return False, 'upslope_distance_cap'
        if is_downslope and dist > d_down: return False, 'downslope_distance_cap'
        if is_cross and dist > d_lat:
            return False, 'lateral_distance_cap'

        # Terrain slope threshold — applies to all directions (downslope and lateral).
        # Upslope clusters are exempt: the crack nucleates from above and the
        # upslope boundary is already governed by the A_ca distance cap.
        if not is_upslope and _mean_slope(cid) < stauchwall_deg:
            return False, 'stauchwall_slope'

        # Absolute tau_g arrest — driving stress below the physical floor
        # cannot sustain crack propagation, regardless of slope or trigger.
        # Applies in all directions (tau_g is a local load, not relative to
        # where nucleation happened). This replaces the earlier trigger/region-
        # relative floor, which could never represent a genuinely stable slope.
        nbr_props = _get_props(cid)
        tau_g_nbr = nbr_props.get('tau_g', np.nan)
        if not np.isnan(tau_g_nbr) and tau_g_nbr < TAU_G_ABS_FLOOR:
            return False, 'tau_g_below_floor'

        # Minimum slab thickness
        h_nbr = nbr_props.get('slab_thickness', np.nan)
        if not np.isnan(h_nbr) and h_nbr < MIN_PROPAGATION_SLAB:
            return False, 'thin_slab'

        # Minimum Lambda — slab too weak/compliant
        lam_nbr_abs = directional_lambda(nbr_props, is_cross)
        if not np.isnan(lam_nbr_abs) and lam_nbr_abs < MIN_PROPAGATION_LAMBDA:
            return False, 'low_lambda'

        # --- Crack-arrest gradient test -------------------------------------
        if USE_MELOCHE_ARREST:
            # Directional Meloche (2025) A_ca arrest. The crack arrests when it
            # propagates INTO stronger weak layer (positive directed θ) and the
            # resulting arrest length is shorter than the C->N step. Propagation
            # into weaker/equal WL (θ_dir <= 0) never arrests here — the same
            # asymmetry as the heuristic gates, but with the physically correct
            # driver (WL shear-strength gradient) instead of the Λ/h proxies.
            cur_cid = current_props.get('cid')
            xy_c    = centroids.get(cur_cid)
            xy_n    = centroids.get(cid)
            tau_p_c = current_props.get('wl_shear_strength', np.nan)  # Pa
            tau_p_n = nbr_props.get('wl_shear_strength', np.nan)      # Pa
            if (xy_c is not None and xy_n is not None
                    and not np.isnan(tau_p_c) and not np.isnan(tau_p_n)):
                dist_cn = float(np.hypot(xy_n[0] - xy_c[0], xy_n[1] - xy_c[1]))
                if dist_cn > 1e-6:
                    # directed gradient toward stronger WL (Pa/m)
                    theta_dir = max(0.0, (tau_p_n - tau_p_c) / dist_cn)
                    if theta_dir > 0.0:
                        Lam   = directional_lambda(nbr_props, is_cross)
                        tg    = tau_g_nbr
                        sig_t = nbr_props.get('sigma_t', np.nan)
                        L_t   = nbr_props.get('L_t',     np.nan)
                        if (not any(np.isnan(v) for v in (Lam, tg, sig_t, L_t))
                                and tg > 0.0):
                            A_ca_d, _ = _ai.arrest_length(
                                tg, theta_dir, Lam, sig_t,
                                delta=MELOCHE_DELTA, L_t=L_t)  # Eq. 20 (m)
                            if A_ca_d < dist_cn:
                                return False, 'meloche_arrest'
            return True, 'propagated'

        # Lambda jump (local gradient) — directional.
        # A DROP in Lambda (softening / more compliant slab over the WL) is the
        # physical arrest direction; a RISE (stiffening) should rarely arrest.
        # Thresholds are therefore asymmetric: tight on drops, loose on rises.
        # Cross-slope carries its own Λ and its own thresholds, so the flanks
        # can diverge from the along-slope behaviour once Λ_III is known. The
        # two pairs are equal for now, so this split changed no results.
        lam_current = directional_lambda(current_props, is_cross)
        lam_nbr     = directional_lambda(nbr_props,     is_cross)
        lam_drop, lam_rise = (
            (LAMBDA_CROSS_DROP_FACTOR, LAMBDA_CROSS_RISE_FACTOR)
            if is_cross else
            (LAMBDA_DROP_FACTOR, LAMBDA_RISE_FACTOR))
        if (not np.isnan(lam_current) and not np.isnan(lam_nbr)
                and lam_current > 0):
            signed_change = (lam_nbr - lam_current) / lam_current
            if signed_change < 0:   # softening — arrest-prone
                if -signed_change > lam_drop * size_factor:
                    return False, ('lambda_cross_discontinuity_drop' if is_cross
                                   else 'lambda_discontinuity_drop')
            else:                   # stiffening — propagation-favorable
                if signed_change > lam_rise * size_factor:
                    return False, ('lambda_cross_discontinuity_rise' if is_cross
                                   else 'lambda_discontinuity_rise')

        # Slab thickness jump (local gradient) — directional.
        # A DROP in thickness (toward a thinner slab that cannot sustain the
        # fracture) arrests readily; a RISE (thicker slab) sustains propagation.
        h_current = current_props.get('slab_thickness', np.nan)
        if (not np.isnan(h_current) and not np.isnan(h_nbr)
                and h_current > 0):
            signed_change = (h_nbr - h_current) / h_current
            if signed_change < 0:   # thinning — arrest-prone
                if -signed_change > THICKNESS_DROP_FACTOR * size_factor:
                    return False, 'thickness_discontinuity_drop'
            else:                   # thickening — propagation-favorable
                if signed_change > THICKNESS_RISE_FACTOR * size_factor:
                    return False, 'thickness_discontinuity_rise'

        return True, 'propagated'

    # Gradient arrest thresholds — DIRECTIONAL relative change.
    # Softening/thinning (a DROP in Lambda or slab thickness) is the physical
    # arrest mechanism and uses a tight threshold; stiffening/thickening (a RISE)
    # favors continued propagation and uses a looser threshold, so the crack is
    # not stopped prematurely by running into stronger/thicker snow.
    #   arrest if  (Λ_nbr - Λ_cur)/Λ_cur  < -LAMBDA_DROP_FACTOR    (softening)
    #   arrest if  (Λ_nbr - Λ_cur)/Λ_cur  > +LAMBDA_RISE_FACTOR    (stiffening)
    #   (same pattern for slab thickness h)
    # size_factor multiplies thresholds → larger size_factor = more tolerant = larger release.
    LAMBDA_DROP_FACTOR     = 0.20   # tight: softening arrests readily
    LAMBDA_RISE_FACTOR     = 0.30   # loose: stiffening rarely arrests
    THICKNESS_DROP_FACTOR  = 0.20   # tight: thinning arrests readily
    THICKNESS_RISE_FACTOR  = 0.30   # loose: thickening rarely arrests
    # TAU_G_ABS_FLOOR (absolute driving-stress floor, Pa) is defined earlier,
    # before the diagnostic print, and applied directly in _qualifies.
    MIN_PROPAGATION_SLAB  = 0.50   # m — slab too thin to sustain crack propagation
    MIN_PROPAGATION_LAMBDA = 0.1   # m — slab too weak/compliant to transmit fracture
                                   # Λ = √(E×h³ / (12×G_wl×D_wl)): integrates
                                   # slab thickness, stiffness, and WL properties

    # (tau_g arrest is now an absolute floor, TAU_G_ABS_FLOOR, applied directly
    # in _qualifies — no per-trigger or per-region reference value to compute.)

    # --- Flood-fill from trigger through qualifying clusters ---
    qual_ok, qual_reason = _qualifies(trigger_cluster_id, trigger_props)
    if not qual_ok:
        pxs_t = np.argwhere(cluster_map == trigger_cluster_id)
        if len(pxs_t) == 0:
            return None, set()

    failed         = {trigger_cluster_id}
    cluster_props  = {trigger_cluster_id: trigger_props}
    visited        = {trigger_cluster_id}

    # Ring-by-ring BFS for wave_callback support
    current_ring = [trigger_cluster_id]
    ring_idx = 0

    # Anisotropic rings: the cross-slope front advances
    # MODE3_SPEED_RATIO / MODE2_SPEED_RATIO as far per ring as the along-slope
    # front, so a flank candidate waits ("deferred") until the budget reaches
    # it. The clock counts only rings that admitted an along-slope cluster —
    # see the USE_ANISOTROPIC_RINGS note at the top of this module.
    rings_on   = USE_ANISOTROPIC_RINGS
    ring_ratio = (MODE3_SPEED_RATIO / MODE2_SPEED_RATIO
                  if MODE2_SPEED_RATIO > 0 else 1.0)
    ring_step  = RING_STEP_M if RING_STEP_M else _hop_length_m(neighbours,
                                                               centroids)
    deferred: list = []     # (cid, parent_props, dist) still outside the budget
    along_clock = 0
    if rings_on:
        print(f"    anisotropic rings ON: cross advance "
              f"{ring_ratio:.3f} x {ring_step:.2f} m per ring")
        if not (ring_step > 0 and ring_ratio > 0):
            # A zero budget would never admit a flank cluster and the deferral
            # queue would never drain, so refuse rather than hang.
            raise ValueError("anisotropic rings need a positive ring_step and "
                             f"speed ratio, got {ring_step} and {ring_ratio}")

    def _cross_budget():
        return along_clock * ring_step * ring_ratio

    if wave_callback is not None:
        wave_callback(0, [trigger_cluster_id], [])

    # Deferred candidates are drained inside this loop, so they are only ever
    # admitted while the front is still advancing — the front stopping IS the
    # end of the event.
    while current_ring and len(failed) < max_clusters:
        ring_idx += 1
        next_ring = []
        ring_arrested = []
        budget = _cross_budget()
        admitted_along = False

        def _admit(cid, parent_props):
            """Evaluate arrest for one candidate and record the outcome."""
            nonlocal admitted_along
            qualifies, reason = _qualifies(cid, parent_props)
            if qualifies:
                failed.add(cid)
                cluster_props[cid] = _get_props(cid)
                next_ring.append(cid)
                direction = _direction(cid)
                if direction is not None and not direction[3]:
                    admitted_along = True
            else:
                ring_arrested.append({'cid': cid, 'reason': reason})

        # Deferred flank candidates the budget has now caught up with. Arrest
        # is evaluated here, on arrival, not when the cluster was deferred.
        if deferred:
            still_deferred = []
            for cid, parent_props, dist in deferred:
                if dist <= budget:
                    _admit(cid, parent_props)
                else:
                    still_deferred.append((cid, parent_props, dist))
            deferred = still_deferred

        for current in current_ring:
            current_props = cluster_props.get(current, trigger_props)
            for nbr in neighbours.get(current, []):
                if nbr in visited:
                    continue
                visited.add(nbr)
                if rings_on:
                    direction = _direction(nbr)
                    if direction is not None:
                        dist, _, _, is_cross = direction
                        if is_cross and dist > budget:
                            deferred.append((nbr, current_props, dist))
                            continue
                _admit(nbr, current_props)

        if admitted_along:
            along_clock += 1

        if not next_ring and not ring_arrested:
            break

        if wave_callback is not None:
            wave_callback(ring_idx, next_ring, ring_arrested)

        current_ring = next_ring

    # The event has a finite duration: once nothing more can be admitted, any
    # flank candidate still outside the cross-slope budget is never reached.
    if deferred:
        print(f"    {len(deferred)} flank clusters dropped — still outside the "
              f"cross-slope budget ({_cross_budget():.0f} m) when propagation "
              f"stopped at ring {ring_idx} (along-slope clock {along_clock})")

    # Distinguish "arrest criteria stopped the crack" from "the safety cap did".
    # max_clusters is a safety cap, not physics: when it binds, the polygon is
    # not a physical result. Measured on Jan 18 in release-areas, raising it
    # from 500 to 2000 moved median area ~20% and best IoU 0.669 -> 0.677,
    # with region size saturating by ~1000 clusters.
    cap_bound = len(failed) >= max_clusters and bool(current_ring)
    if cap_bound:
        print(f"  WARNING: BFS hit the max_clusters cap ({max_clusters}) with "
              f"{len(current_ring)} clusters still propagating — this polygon "
              f"is bounded by the cap, not by arrest criteria. Raise it to let "
              f"the physics terminate it.")

    # Debug direction breakdown
    dir_counts = {'upslope': 0, 'downslope': 0, 'lateral': 0, 'other': 0}
    for cid in failed:
        if cid == trigger_cluster_id:
            continue
        xy = centroids.get(cid)
        if xy is None:
            continue
        asp_rad = np.radians(t_aspect)
        fall_x, fall_y = np.sin(asp_rad), np.cos(asp_rad)
        dx, dy = xy[0] - t_x, xy[1] - t_y
        dist   = np.sqrt(dx**2 + dy**2)
        dot    = (dx * fall_x + dy * fall_y) / max(dist, 1e-6)
        if   dot < -0.707: dir_counts['upslope']   += 1
        elif dot >  0.707: dir_counts['downslope']  += 1
        else:              dir_counts['lateral']    += 1
    print(f"  Connected region: {len(failed)} clusters "
          f"(up={dir_counts['upslope']} "
          f"down={dir_counts['downslope']} "
          f"lat={dir_counts['lateral']})"
          f"{'  [CAP-BOUND]' if cap_bound else ''}")

    if len(failed) < 2:
        return None, failed

    # Rasterize to polygon
    mask   = np.isin(cluster_map, list(failed)).astype(np.uint8)
    shapes = list(rasterio.features.shapes(
        mask, mask=mask.astype(bool), transform=transform))
    if not shapes:
        return None, failed

    polys   = [__import__('shapely.geometry', fromlist=['shape']).shape(s)
               for s, v in shapes if v == 1]
    polygon = unary_union(polys)
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda p: p.area)
    if polygon.is_empty or polygon.area < MIN_POLYGON_AREA:
        return None, failed

    # Fill interior holes (arrested interior clusters / pixelation) so com1DFA
    # gets a simply-connected release area. Pass max_hole_frac to preserve large
    # genuine non-released islands.
    polygon = fill_polygon_holes(polygon)

    return polygon, failed


# -----------------------------------------------------------------------
# Path-energy arrest (ARREST_MODE == 'energy') — hypothesis, not validated
# -----------------------------------------------------------------------

ENERGY_REASONS = ('no_weak_layer', 'energy_deficit', 'no_drive', 'stauchwall_slope',
                  'outside_start_zone', 'domain_edge', 'no_data')


def _domain_edge_pixels(cluster_map: np.ndarray, dem: np.ndarray) -> np.ndarray:
    """Valid pixels with a 4-neighbour off the array, outside the cluster map or on NaN DEM."""
    valid = (cluster_map > 0) & np.isfinite(dem)
    pad = np.pad(valid, 1, constant_values=False)
    interior = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    return valid & ~interior


def _cluster_graph(pixel_index: dict, transform, k_neighbours: int):
    """k-NN cluster adjacency and UTM centroids (rounded to the pixel) for the BFS."""
    from sklearn.neighbors import NearestNeighbors
    cents_px   = cluster_centroids_px(pixel_index)
    cids       = np.array(sorted(cents_px))
    pxs        = np.array([cents_px[c] for c in cids])
    k_actual   = min(k_neighbours + 1, len(cids))
    nbrs       = NearestNeighbors(n_neighbors=k_actual).fit(pxs)
    _, idxs    = nbrs.kneighbors(pxs)
    neighbours = {int(cids[i]): [int(cids[j]) for j in idxs[i][1:]]
                  for i in range(len(cids))}
    centroids = {int(c): pixel_to_utm(int(round(pxs[i][0])),
                                      int(round(pxs[i][1])), transform)
                 for i, c in enumerate(cids)}
    return neighbours, centroids


def _reached_polygon(reached, cluster_map, transform, min_clusters: int = 2,
                     min_area: float = MIN_POLYGON_AREA):
    """Largest hole-filled polygon of the reached clusters, or None if too small."""
    from shapely.ops import unary_union
    from shapely.geometry import shape as shapely_shape
    import rasterio.features

    if len(reached) < min_clusters:
        return None
    mask   = np.isin(cluster_map, list(reached)).astype(np.uint8)
    shapes = list(rasterio.features.shapes(
        mask, mask=mask.astype(bool), transform=transform))
    if not shapes:
        return None
    polygon = unary_union([shapely_shape(s) for s, v in shapes if v == 1])
    if polygon.geom_type == 'MultiPolygon':
        polygon = max(polygon.geoms, key=lambda p: p.area)
    if polygon.is_empty or polygon.area < min_area:
        return None
    return fill_polygon_holes(polygon)


def propagate_release_energy(
        trigger_cluster_id: int,
        cluster_map: np.ndarray,
        dem: np.ndarray,
        slope_grid: np.ndarray,
        transform,
        start_zone_mask: Optional[np.ndarray] = None,
        stauchwall_deg: float = STAUCHWALL_DEG,
        snap_features: 'Optional[pd.DataFrame]' = None,
        k_neighbours: int = 8,
        max_clusters: int = MAX_BFS_CLUSTERS,
        pixel_index: Optional[dict] = None,
        aspect_grid: Optional[np.ndarray] = None,
        wave_callback=None,
        state_out: Optional[dict] = None):
    """Path-energy flood-fill (hypothesis): admit a cluster while the slab tension
    carried to it releases G_avail >= G_c.

    Steps only move away from the trigger, so the graph is acyclic and the best
    (highest) tension per cluster is exact when clusters are taken in order of
    distance. Each step c -> n is classed up/down/cross by the ±45° sector rule
    of propagate_release, applied to the step vector against the trigger fall line.
    Along-slope: N_n = clip(N_c + tau_net(c) ds, 0, N_max(n)), test G_avail(N_n),
    with ds the growth of |y - y0|: y is the fall-line coordinate and y0 where this
    path's tension started, so N is the detached fall-line extent, not path length.
    Cross-slope: N_n = 0, admitted on tau_net(n) > 0 and R G_slab(n) >= G_c(n).

    Returns (polygon, reached_cluster_ids); a trigger-only region still gives a polygon.
    """
    import heapq
    import energy_path as ep

    if snap_features is None:
        return None, set()
    if pixel_index is None:
        pixel_index = cluster_pixel_index(cluster_map)
    if trigger_cluster_id not in pixel_index:
        return None, set()
    px_m = abs(transform.a) if transform is not None else 1.0
    if aspect_grid is None:
        _, aspect_grid = compute_slope_aspect(dem, px_m)
    R = ep.energy_r()

    neighbours, centroids = _cluster_graph(pixel_index, transform, k_neighbours)
    edge_px = _domain_edge_pixels(cluster_map, dem)

    t_rows, t_cols = pixel_index[trigger_cluster_id]
    t_aspect = float(aspect_grid[int(t_rows.mean()), int(t_cols.mean())])
    fall_x, fall_y = np.sin(np.radians(t_aspect)), np.cos(np.radians(t_aspect))
    t_xy = centroids[trigger_cluster_id]

    cells: dict[int, dict] = {}

    def _cell(cid):
        if cid not in cells:
            px = pixel_index[cid]
            psi = float(slope_grid[px[0], px[1]].mean())
            row = (snap_features.loc[cid] if cid in snap_features.index
                   else pd.Series(dtype=float))
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            e = ep.cell_energy(row, psi)
            tau_p = pd.to_numeric(row.get('wl_shear_strength', np.nan), errors='coerce')
            h = pd.to_numeric(row.get('slab_thickness', np.nan), errors='coerce')
            e.update(psi=psi, wl=bool(np.isfinite(tau_p) and np.isfinite(h)),
                     edge=bool(edge_px[px[0], px[1]].any()))
            if start_zone_mask is not None:
                e['in_mask'] = bool(start_zone_mask[int(px[0].mean()), int(px[1].mean())])
            cells[cid] = e
        return cells[cid]

    def _radius(cid):
        xy = centroids[cid]
        return float(np.hypot(xy[0] - t_xy[0], xy[1] - t_xy[1]))

    def _fall(cid):
        xy = centroids[cid]
        return (xy[0] - t_xy[0]) * fall_x + (xy[1] - t_xy[1]) * fall_y

    def _step(c, n):
        (xc, yc), (xn, yn) = centroids[c], centroids[n]
        ds = float(np.hypot(xn - xc, yn - yc))
        dot = ((xn - xc) * fall_x + (yn - yc) * fall_y) / max(ds, 1e-6)
        return 'down' if dot > 0.707 else 'up' if dot < -0.707 else 'cross'

    def _try(c, n):
        """(reason, kind, N_n, G_avail) for the step c -> n; reason None means admitted."""
        kind = _step(c, n)
        e = _cell(n)
        if start_zone_mask is not None and not e.get('in_mask', True):
            return 'outside_start_zone', kind, -1.0, np.nan
        if kind != 'up' and e['psi'] < stauchwall_deg:
            return 'stauchwall_slope', kind, -1.0, np.nan
        if not e['wl']:
            return 'no_weak_layer', kind, -1.0, np.nan
        if not all(np.isfinite(e[k]) for k in ('tau_net', 'G_c', 'G_slab_path', 'N_max')) \
                or e['N_max'] <= 0:
            return 'no_data', kind, -1.0, np.nan
        if kind == 'cross':
            g_cap = R * e['G_slab_path']
            if e['tau_net'] <= 0:
                return 'no_drive', kind, 0.0, g_cap
            if g_cap < e['G_c']:
                return 'energy_deficit', kind, 0.0, g_cap
            return None, kind, 0.0, g_cap
        drive = _cell(c)['tau_net']
        drive = drive if np.isfinite(drive) else 0.0
        y0 = best[c]['y0']
        ds = abs(_fall(n) - y0) - abs(_fall(c) - y0)
        N = float(np.clip(best[c]['N'] + drive * ds, 0.0, e['N_max']))
        if N <= 0.0:
            return 'no_drive', kind, N, 0.0
        G_av = R * e['G_slab_path'] * (N / e['N_max'])**2   # = R N^2 / (2 E' h)
        if G_av < e['G_c']:
            return 'energy_deficit', kind, N, G_av
        return None, kind, N, G_av

    best = {trigger_cluster_id: {'N': 0.0, 'G_avail': 0.0, 'parent': None,
                                 'step': 'trigger', 'y0': 0.0}}
    offers: dict[int, tuple] = {}        # never-reached cid -> (N, reason) of its best offer
    first_rej: dict[int, dict] = {}
    reached: set = set()
    heap = [(0.0, trigger_cluster_id)]
    n_pop = 0
    if wave_callback is not None:
        wave_callback(0, [trigger_cluster_id], [])

    while heap and len(reached) < max_clusters:
        r_c, c = heapq.heappop(heap)
        if c in reached:
            continue
        reached.add(c)
        n_pop += 1
        admitted, arrested = [], []
        if _cell(c)['edge']:
            arrested.append({'cid': c, 'reason': 'domain_edge'})
            first_rej.setdefault(c, {}).setdefault('edge', 'domain_edge')
        for n in neighbours.get(c, []):
            if n in reached or n not in centroids or _radius(n) <= r_c:
                continue
            reason, kind, N, G_av = _try(c, n)
            if reason is None:
                if n not in best or N > best[n]['N']:
                    if n not in best:
                        heapq.heappush(heap, (_radius(n), n))
                    y0 = _fall(n) if kind == 'cross' else best[c]['y0']
                    best[n] = {'N': N, 'G_avail': G_av, 'parent': c, 'step': kind, 'y0': y0}
                admitted.append(n)
            else:
                arrested.append({'cid': n, 'reason': reason})
                first_rej.setdefault(c, {}).setdefault(kind, reason)
                if n not in offers or N > offers[n][0]:
                    offers[n] = (N, reason)
        if wave_callback is not None:
            wave_callback(n_pop, admitted, arrested)

    cap_bound = bool(heap) and len(reached) >= max_clusters
    rejected = {n: r for n, (_, r) in offers.items() if n not in best}
    counts = pd.Series(list(rejected.values()), dtype=object).value_counts().to_dict()
    print(f"  Energy region: {len(reached)} clusters (R={R}, delta={ep.ENERGY_DELTA}, "
          f"sigma_t={ep.SIGMA_T_COLUMN}); boundary rejections {counts}"
          f"{'  [CAP-BOUND]' if cap_bound else ''}")
    if cap_bound:
        print(f"  WARNING: BFS hit the max_clusters cap ({max_clusters}); this "
              f"polygon is bounded by the cap, not by arrest criteria.")

    if state_out is not None:
        state_out['cells'] = {
            cid: {**{k: cells[cid][k] for k in
                     ('tau_net', 'G_c', 'G_slab_path', 'N_max', 'psi')},
                  'N': best[cid]['N'], 'G_avail': best[cid]['G_avail'],
                  'parent': best[cid]['parent'], 'step': best[cid]['step'],
                  **{f'first_rej_{k}': v for k, v in first_rej.get(cid, {}).items()}}
            for cid in reached}
        state_out['rejected'] = rejected
        state_out['cap_bound'] = cap_bound

    polygon = _reached_polygon(reached, cluster_map, transform,
                               min_clusters=1, min_area=0.0)
    return polygon, reached



def depth_from_snowpack(cluster_map: np.ndarray,
                        ds: xr.Dataset,
                        snapshot: str,
                        snap_features: pd.DataFrame = None,
                        wl_fraction: float = 0.80) -> np.ndarray:
    """
    Per-pixel slab depth (m) from SNOWPACK at snapshot date.

    If snap_features is provided and contains 'slab_thickness', uses the
    per-cluster slab thickness above the WL interface (the mass that
    actually releases). Falls back to wl_fraction × HS when slab_thickness
    is unavailable for a cluster.

    Parameters
    ----------
    cluster_map   : 2D array of cluster IDs
    ds            : xsnow Dataset (Zarr-backed)
    snapshot      : date string 'YYYY-MM-DD'
    snap_features : DataFrame indexed by cluster_id with 'slab_thickness' (m)
                    from profile_features(). If None, uses wl_fraction × HS.
    wl_fraction   : fallback fraction of HS assumed to be slab (default 0.80)

    Returns
    -------
    float32 array, shape = cluster_map.shape, values in metres
    """
    snap_ts = np.datetime64(f"{snapshot}T12:00")
    times   = ds.coords['time'].values
    tidx    = int(np.argmin(np.abs(times - snap_ts)))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ds_t = ds.isel(time=tidx).compute()
        sq   = [d for d in ['slope', 'realization'] if d in ds_t.dims]
        hs   = ds_t['HS'].squeeze(sq).values    # cm, (n_locations,)

    locs    = ds.coords['location'].values
    loc_ids = np.array([int(str(l).split('_')[-1]) for l in locs])
    id_to_hs = dict(zip(loc_ids, hs))

    # Build per-cluster slab thickness lookup
    use_slab_thickness = (snap_features is not None
                          and 'slab_thickness' in snap_features.columns)
    n_from_features = 0
    n_from_fallback = 0

    depth_m = np.full(cluster_map.shape, np.nan, dtype=np.float32)
    for cid, h in id_to_hs.items():
        if np.isnan(h):
            continue

        slab_depth_m = None
        if use_slab_thickness and cid in snap_features.index:
            st = snap_features.loc[cid, 'slab_thickness']
            if isinstance(st, pd.Series):
                st = st.iloc[0]
            try:
                st = float(st)
                if not np.isnan(st) and st > 0:
                    slab_depth_m = st  # already in metres
                    n_from_features += 1
            except (ValueError, TypeError):
                pass

        if slab_depth_m is None:
            slab_depth_m = h * wl_fraction / 100.0  # cm → m
            n_from_fallback += 1

        depth_m[cluster_map == cid] = slab_depth_m

    if use_slab_thickness:
        print(f"  Depth raster: {n_from_features} clusters from slab_thickness, "
              f"{n_from_fallback} from HS×{wl_fraction} fallback")

    return depth_m


def rasterize_release_polygon(polygon,
                               depth_grid: np.ndarray,
                               dem_shape: tuple,
                               transform) -> np.ndarray:
    """
    Burn release polygon onto the depth grid.

    Returns float32 raster with depth values inside the polygon,
    NaN outside.

    Parameters
    ----------
    polygon    : shapely Polygon in UTM coordinates
    depth_grid : per-pixel depth (m), same shape as DEM
    dem_shape  : (nrows, ncols)
    transform  : rasterio Affine transform
    """
    import rasterio.features

    if polygon is None or polygon.is_empty:
        return np.full(dem_shape, np.nan, dtype=np.float32)

    mask = rasterio.features.geometry_mask(
        [polygon.__geo_interface__],
        out_shape=dem_shape,
        transform=transform,
        invert=True)

    release = np.where(mask & ~np.isnan(depth_grid),
                       depth_grid.astype(np.float32), np.nan)
    return release


# -----------------------------------------------------------------------
# Comparison plot: Meloche polygons vs observed release area
# -----------------------------------------------------------------------


# -----------------------------------------------------------------------
# Comparison plot: Meloche polygons vs observed release area
# -----------------------------------------------------------------------

def plot_release_comparison(
        meloche_polygons: list,
        observed_polygon,
        dem: np.ndarray,
        transform,
        start_zone_mask=None,
        trigger_labels=None,
        trigger_centroids=None,
        most_likely_polygon=None,
        most_likely_label=None,
        out_path=None,
        title: str = "Release polygon comparison") -> None:
    """
    Plot Meloche-derived polygons vs observed release area.
    One star marker per trigger nucleation point, labels in legend.

    trigger_centroids : list of (x, y) UTM tuples, one per trigger.
                        If None, falls back to polygon centroid.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    from matplotlib.colors import LightSource
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(10, 10))

    fill_dem  = np.where(np.isnan(dem), np.nanmean(dem), dem)
    hillshade = LightSource(azdeg=315, altdeg=45).hillshade(fill_dem, dx=1.0, dy=1.0)
    nrows, ncols = dem.shape
    extent = [transform.c,
              transform.c + ncols * transform.a,
              transform.f + nrows * transform.e,
              transform.f]
    ax.imshow(hillshade, cmap='gray', extent=extent,
              alpha=0.6, aspect='auto', origin='upper')

    # Start zone
    if start_zone_mask is not None:
        ax.contour(start_zone_mask.astype(float), levels=[0.5],
                   colors=['limegreen'], linewidths=2.0, alpha=0.9,
                   extent=extent, origin='upper')

    # Observed release area
    obs_area = 0.0
    if observed_polygon is not None and not observed_polygon.is_empty:
        obs_area = observed_polygon.area
        xs, ys = observed_polygon.exterior.xy
        ax.fill(xs, ys, alpha=0.30, color='red', zorder=3)
        ax.plot(xs, ys, color='red', linewidth=2.5, zorder=4)

    # Meloche polygons — one colour per trigger, star at centroid
    colors    = plt.cm.tab10.colors
    mel_areas = []

    for i, (poly, size_f) in enumerate(meloche_polygons):
        if poly is None or poly.is_empty:
            continue
        color = colors[i % len(colors)]
        xs, ys = poly.exterior.xy
        ax.fill(xs, ys, alpha=0.20, color=color, zorder=2)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.85, zorder=5)
        mel_areas.append(poly.area)

        # Nucleation star — use cluster centroid if provided, else polygon centroid
        if trigger_centroids is not None and i < len(trigger_centroids):
            cx, cy = trigger_centroids[i]
        else:
            cx, cy = poly.centroid.x, poly.centroid.y
        ax.plot(cx, cy, marker='*', markersize=14, color=color,
                markeredgecolor='white', markeredgewidth=0.8, zorder=10)

        # Short map label with outline effect — no box
        short = f"T{i+1}"
        txt   = ax.text(cx + 8, cy + 8, short, fontsize=9,
                        color=color, fontweight='bold', zorder=11)
        txt.set_path_effects([pe.Stroke(linewidth=2.5, foreground='white'),
                               pe.Normal()])

    # Most-likely scenario polygon — bold black dashed outline on top, so it
    # reads against the coloured per-trigger polygons. exterior.xy ignores holes.
    ml_area = 0.0
    if most_likely_polygon is not None and not most_likely_polygon.is_empty:
        ml_area = most_likely_polygon.area
        mxs, mys = most_likely_polygon.exterior.xy
        ax.plot(mxs, mys, color='white', linewidth=4.2, zorder=7)   # halo
        ax.plot(mxs, mys, color='black', linewidth=2.4, linestyle=(0, (6, 3)),
                zorder=8)
    # Stats box
    if mel_areas and obs_area:
        ratio = float(np.median(mel_areas)) / obs_area
        line1 = "Observed:       %6.0f m2" % obs_area
        line2 = "Meloche P50:    %6.0f m2" % float(np.median(mel_areas))
        line3 = "Ratio Mel/Obs:    %.2f" % ratio
        stats = line1 + "\n" + line2 + "\n" + line3
    elif mel_areas:
        stats = "Meloche P50: %.0f m2" % float(np.median(mel_areas))
    else:
        stats = "No polygons generated"
    if ml_area:
        stats += "\nMost likely:    %6.0f m2" % ml_area
        if obs_area:
            stats += "\n  ML/Obs ratio:   %.2f" % (ml_area / obs_area)
    ax.text(0.02, 0.02, stats, transform=ax.transAxes, fontsize=8.5,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    # Legend
    handles = []
    handles.append(Line2D([0],[0], color='red',       linewidth=2.5,
           label="Observed Jan 18  (%.0f m2)" % obs_area))
    handles.append(Line2D([0],[0], color='limegreen', linewidth=2.0, label='Start zone'))
    if ml_area:
        handles.append(Line2D([0],[0], color='black', linewidth=2.4,
                              linestyle=(0, (6, 3)),
                              label=(most_likely_label
                                     or "Most likely  (%.0f m2)" % ml_area)))
    for i, (poly, size_f) in enumerate(meloche_polygons):
        if poly is None or poly.is_empty:
            continue
        color = colors[i % len(colors)]
        lbl   = trigger_labels[i] if trigger_labels else ("T%d" % (i+1))
        handles.append(Patch(facecolor=color, alpha=0.5, edgecolor=color,
                             label="%s  (%.0f m2)" % (lbl, poly.area)))

    ax.legend(handles=handles, loc='upper right', fontsize=7.5, framealpha=0.90)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel('Easting (m UTM)')
    ax.set_ylabel('Northing (m UTM)')
    ax.ticklabel_format(style='plain', axis='both')
    plt.tight_layout()

    if out_path:
        fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
        print(f"Release comparison plot: {out_path}")
        plt.close(fig)
    else:
        plt.show()