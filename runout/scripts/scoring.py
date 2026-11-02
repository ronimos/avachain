"""
Event Scoring

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Scores a modeled runout against an observed event. Each score needs a
    kind of observation and is skipped when it is absent: IoU, recovery
    and over-prediction against a debris polygon; signed runout error
    against a toe point; per reference point agreement; and, for an
    ensemble, the share of the debris inside the envelope.
"""

# --------- Load Libraries --------
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import avalanche_helpers as h


def reach_mask_at(pft_path, depth_threshold_m, ppr_path=None,
                  pressure_threshold_kpa=0.0):
    # Threshold peak flow depth (and optionally pressure) into a reach mask
    # @param pft_path: Path to peak flow depth raster
    # @param depth_threshold_m: float
    # @param ppr_path: optional Path to peak pressure
    # @param pressure_threshold_kpa: float, 0 disables
    # @returns: tuple (bool array, meta dict)
    depth, meta = h.read_raster(pft_path)
    reach = depth >= depth_threshold_m
    if pressure_threshold_kpa > 0 and ppr_path is not None:
        pres, _ = h.read_raster(ppr_path)
        reach &= pres >= pressure_threshold_kpa
    return reach, meta


def runout_from(mask, meta, centroid):
    # Farthest reached cell from a point, straight line
    # @param mask: bool array
    # @param meta: dict from read_raster
    # @param centroid: shapely Point
    # @returns: float metres (0 when nothing reached)
    if not mask.any():
        return 0.0
    ys, xs = np.where(mask)
    wx, wy = h.pixels_to_world(ys, xs, meta['transform'])
    return float(np.hypot(wx - centroid.x, wy - centroid.y).max())


def score_polygon(mask, meta, debris):
    # IoU, recovery and over-prediction of a reach mask against debris
    # @param mask: bool array
    # @param meta: dict from read_raster
    # @param debris: shapely polygon in the raster CRS
    # @returns: dict with iou, recovery, overprediction, model_area_m2,
    #          debris_area_m2, overlap_area_m2
    obs = h.polygon_to_raster_mask(debris, mask.shape, meta['transform'], 1.0) > 0
    cell = abs(meta['transform'].a * meta['transform'].e)
    inter = int((mask & obs).sum())
    union = int((mask | obs).sum())
    n_mod = int(mask.sum())
    n_obs = int(obs.sum())
    return {
        'iou': inter / union if union else 0.0,
        'recovery': inter / n_obs if n_obs else 0.0,
        'overprediction': (n_mod - inter) / n_mod if n_mod else 0.0,
        'model_area_m2': n_mod * cell,
        'debris_area_m2': n_obs * cell,
        'overlap_area_m2': inter * cell,
    }


def score_toe(mask, meta, centroid, toe):
    # Signed runout error against an observed toe point
    # @param mask: bool array
    # @param meta: dict from read_raster
    # @param centroid: shapely Point, release centroid
    # @param toe: shapely Point
    # @returns: dict with runout_m, observed_m, toe_error_m
    modeled = runout_from(mask, meta, centroid)
    observed = float(centroid.distance(toe))
    return {'runout_m': modeled, 'observed_m': observed,
            'toe_error_m': modeled - observed}


def score_points(mask, meta, refpoints_path, outcomes, centroid, radius_m,
                 road_path=None, road_buffer_m=1.0):
    # Modeled against observed reach at each named reference point
    # @param mask: bool array
    # @param meta: dict from read_raster
    # @param refpoints_path: Path to the point layer with ref_name
    # @param outcomes: dict of ref_name -> bool observed
    # @param centroid: shapely Point, release centroid
    # @param radius_m: reach radius for points
    # @param road_path: optional road line; a point named road uses the
    #                   road corridor test when given
    # @param road_buffer_m: corridor half-width
    # @returns: dict with agree (share), n, per_point (list of dicts)
    pts = gpd.read_file(refpoints_path)
    if pts.crs is not None and meta['crs'] is not None and pts.crs != meta['crs']:
        pts = pts.to_crs(meta['crs'])
    if mask.any():
        ys, xs = np.where(mask)
        wx, wy = h.pixels_to_world(ys, xs, meta['transform'])
    else:
        wx = wy = np.array([])

    road_hit = None
    if road_path is not None:
        road = gpd.read_file(road_path).geometry.unary_union
        corridor = h.polygon_to_raster_mask(road.buffer(road_buffer_m, cap_style=2),
                                            mask.shape, meta['transform'], 1.0) > 0
        road_hit = bool((mask & corridor).any())

    per = []
    for _, row in pts.iterrows():
        name = str(row['ref_name'])
        if name not in outcomes:
            continue
        pt = row.geometry if row.geometry.geom_type == 'Point' else row.geometry.centroid
        if name.lower() == 'road' and road_hit is not None:
            modeled = road_hit
        elif len(wx):
            modeled = bool(np.hypot(wx - pt.x, wy - pt.y).min() <= radius_m)
        else:
            modeled = False
        per.append({'ref_name': name, 'observed': outcomes[name],
                    'modeled': modeled, 'agree': modeled == outcomes[name]})
    n = len(per)
    return {'agree': sum(p['agree'] for p in per) / n if n else None,
            'n': n, 'per_point': per}


def score_envelope(prob_path, debris, level=0.05):
    # How much of the debris an ensemble's probability covered
    # @param prob_path: Path to probability.tif
    # @param debris: shapely polygon in the raster CRS
    # @param level: envelope probability level
    # @returns: dict with debris_in_envelope (share), mean_p_over_debris,
    #          max_p_over_debris
    prob, meta = h.read_raster(prob_path)
    obs = h.polygon_to_raster_mask(debris, prob.shape, meta['transform'], 1.0) > 0
    if not obs.any():
        return {'debris_in_envelope': None, 'mean_p_over_debris': None,
                'max_p_over_debris': None}
    p = np.where(prob < 0, 0.0, prob)
    return {
        'debris_in_envelope': float((p[obs] >= level).mean()),
        'mean_p_over_debris': float(p[obs].mean()),
        'max_p_over_debris': float(p[obs].max()),
    }


def score_scenario(pft_path, ppr_path, event, centroid, thresholds,
                   refpoints_path=None, radius_m=10.0, road_path=None,
                   road_buffer_m=1.0, pressure_threshold_kpa=0.0):
    # Every applicable score for one scenario at each reach threshold
    # @param pft_path: Path to peak flow depth
    # @param ppr_path: Path to peak pressure or None
    # @param event: dict from events.load_event
    # @param centroid: shapely Point, release centroid used for runout
    # @param thresholds: list of flow depth thresholds (m)
    # @param refpoints_path: Path to reference points, for point scores
    # @param radius_m: point reach radius
    # @param road_path: road line for the road point
    # @param road_buffer_m: corridor half-width
    # @param pressure_threshold_kpa: pressure threshold, 0 disables
    # @returns: list of dicts, one per threshold, with the scores present
    #          for the event's observations (others absent)
    out = []
    for t in thresholds:
        mask, meta = reach_mask_at(pft_path, t, ppr_path, pressure_threshold_kpa)
        row = {'threshold_m': t, 'runout_m': runout_from(mask, meta, centroid),
               'reach_area_m2': float(mask.sum() * abs(meta['transform'].a * meta['transform'].e))}
        if event['debris'] is not None:
            row.update(score_polygon(mask, meta, event['debris']))
        if event['toe'] is not None:
            row.update(score_toe(mask, meta, centroid, event['toe']))
        if event['points'] and refpoints_path is not None:
            sp = score_points(mask, meta, refpoints_path, event['points'], centroid,
                              radius_m, road_path, road_buffer_m)
            row['points_agree'] = sp['agree']
            row['points_n'] = sp['n']
            row['points_detail'] = sp['per_point']
        out.append(row)
    return out
