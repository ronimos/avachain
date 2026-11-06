"""
avalanche_stack.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    After all per-scenario sims complete, stack their reach masks into
    ensemble-level outputs:
        - probability.tif: P(reach) at each cell across the ensemble
        - envelope.geojson: outer boundary at P >= envelope_min_probability
        - contours.geojson: polygons at each contour level
        - road_impact_summary.json: N-of-M reached + impact points
        - scenarios_summary.csv: one row per scenario
        - most_likely_path.geojson: medial-axis line through the high-P core

    No command line. Imported by avalanche_runout.py.

Requirements:
    - rasterio, geopandas, shapely, numpy, pandas, scikit-image (optional)
"""

# --------- Load Libraries --------
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from shapely.geometry import LineString

import avalanche_helpers as h




def ensemble_weight_total(params_list):
    # The fixed denominator for an ensemble with absolute weights, or None
    # @param params_list: iterable of scenario params dicts
    # @returns: float or None
    # Forced explosive shots (avachain, Ron 2026-11-06): each shot holds
    #       1/N and what does not release is P(no release), so the weights
    #       sum to P(release) < 1. Dividing by their own sum would spread
    #       the no-release share over the shots that did release. The
    #       release side writes ensemble_weight_total (1.0) into params.json
    #       for such sets; model sets carry none and keep sum(weights).
    totals = []
    for prm in params_list:
        t = (prm or {}).get('ensemble_weight_total')
        if t is not None and float(t) > 0:
            totals.append(float(t))
    return max(totals) if totals else None


def stack_reach_masks(mask_paths, out_path, weights=None, total=None):
    # Stack N binary reach masks into a probability raster
    # @param mask_paths: list of Paths to reach mask GeoTIFFs
    # @param out_path: Path to write probability raster
    # @param weights: optional list of float weights matching mask_paths.
    #                 None = equal weighting (simple mean). Otherwise:
    #                     P(reach) = sum(w_i * mask_i) / sum(w_i)
    # @param total: optional fixed denominator in place of sum(w_i)
    #               (absolute weights; see ensemble_weight_total)
    # @returns: Path to written raster
    # @raises ValueError: if no masks are given, weights do not match the
    #                     mask count, weights sum to zero, or grids differ
    # output is float32 with nodata = -1.0
    
    if not mask_paths:
        # throw error if no masks supplied
        raise ValueError("No reach masks provided to stack")
    
    # Validate weights if provided
    if weights is not None:
        if len(weights) != len(mask_paths):
            # throw error if weight count doesn't match
            raise ValueError(
                f"weights length ({len(weights)}) doesn't match "
                f"mask count ({len(mask_paths)})"
            )
        weights = [float(w) for w in weights]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            # throw error if weights are degenerate
            raise ValueError(
                f"sum of weights is {weight_sum}; need positive total"
            )
    
    # Establish reference grid from first mask
    _, ref_meta = h.read_raster(mask_paths[0])
    ref_shape = ref_meta['shape']
    accum = np.zeros(ref_shape, dtype='float32')
    
    # Accumulate (weighted or unweighted)
    for i, p in enumerate(mask_paths):
        data, meta = h.read_raster(p)
        if meta['shape'] != ref_shape:
            # throw error if grids don't align
            raise ValueError(
                f"Mask {p} shape {meta['shape']} != reference {ref_shape}. "
                "All scenario reach masks must share the same grid."
            )
        if weights is not None:
            accum += data.astype('float32') * weights[i]
        else:
            accum += data.astype('float32')
    
    # Normalize -> probability
    if weights is not None and total:
        probability = accum / total
        method = (f"weighted by absolute scenario_probability "
                  f"(sum={sum(weights):.4f} of {total:.4f})")
    elif weights is not None:
        probability = accum / sum(weights)
        method = f"weighted by scenario_probability (sum={sum(weights):.4f})"
    else:
        probability = accum / len(mask_paths)
        method = "equal-weight mean"
    
    # Write with -1 nodata (so true zeros stay distinct from "no data")
    profile = ref_meta['profile']
    profile['nodata'] = -1.0
    h.write_raster(out_path, probability, profile, driver='GTiff', cog=True)
    
    logging.info(f"Stacked {len(mask_paths)} masks ({method}) -> {out_path}")
    return out_path


def polygonize_probability(probability_path, contour_levels,
                           envelope_path, contours_path,
                           envelope_threshold):
    # Polygonize the probability raster at each contour level
    # @param probability_path: Path to probability raster
    # @param contour_levels: list of P thresholds (e.g. [0.05, 0.25, 0.5, 0.75, 1.0])
    # @param envelope_path: Path to write envelope GeoJSON (single threshold)
    # @param contours_path: Path to write contours GeoJSON (multiple thresholds)
    # @param envelope_threshold: P-threshold for the outer envelope (e.g. 0.05)
    # @returns: None (writes both GeoJSON files)
    
    prob, meta = h.read_raster(probability_path)
    transform = meta['transform']
    crs = meta['crs']
    
    # Envelope: outer boundary above threshold
    envelope_mask = (prob >= envelope_threshold)
    env_polys = h.mask_to_polygons(envelope_mask, transform)
    
    env_gdf = gpd.GeoDataFrame(
        {'min_prob': [envelope_threshold] * len(env_polys)},
        geometry=env_polys,
        crs=crs
    )
    env_gdf.to_file(envelope_path, driver='GeoJSON')
    
    # Contours: one feature per (level, polygon)
    rows = []
    for level in contour_levels:
        mask = (prob >= level)
        polys = h.mask_to_polygons(mask, transform)
        for poly in polys:
            rows.append({'level': level, 'geometry': poly})
    
    if rows:
        contour_gdf = gpd.GeoDataFrame(rows, crs=crs)
    else:
        # Empty case - still write valid GeoJSON with empty features
        contour_gdf = gpd.GeoDataFrame({'level': []}, geometry=[], crs=crs)
    
    contour_gdf.to_file(contours_path, driver='GeoJSON')
    
    logging.info(
        f"Wrote envelope ({len(env_polys)} polys) and "
        f"contours ({len(rows)} features)"
    )


def extract_most_likely_path(probability_path, threshold, out_path):
    # Extract the medial axis of the P >= threshold region as a polyline
    # @param probability_path: Path to probability raster
    # @param threshold: P value defining the "high probability core"
    # @param out_path: Path to write GeoJSON polyline
    # @returns: Path or None (if scikit-image missing or region too small)
    # this is the skeleton of the high-P region, not a statistical
    #       mode of trajectories; branching means multi-path terrain
    
    # Soft dependency on scikit-image
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        logging.warning("scikit-image not installed; skipping most_likely_path")
        return None
    
    prob, meta = h.read_raster(probability_path)
    transform = meta['transform']
    crs = meta['crs']
    
    mask = (prob >= threshold)
    if mask.sum() < 10:
        logging.warning(
            f"High-P region (P >= {threshold:.2f}) has < 10 cells; "
            "skipping most_likely_path"
        )
        return None
    
    # Skeletonize and convert to world coords
    skeleton = skeletonize(mask)
    ys, xs = np.where(skeleton)
    
    if len(xs) < 2:
        return None
    
    wx, wy = h.pixels_to_world(ys, xs, transform)
    
    # Order by scanline (top -> bottom, left -> right). Good enough for
    # visualization; not a true graph traversal.
    coords = sorted(zip(wx, wy), key=lambda p: (-p[1], p[0]))
    line = LineString(coords)
    
    gdf = gpd.GeoDataFrame(
        {'p_threshold': [threshold]}, geometry=[line], crs=crs
    )
    gdf.to_file(out_path, driver='GeoJSON')
    return out_path


def build_road_impact_summary(scenario_summaries, out_path, scenarios=None,
                                use_weights=False, weight_total=None):
    # Aggregate road-reach results across all scenarios
    # @param weight_total: optional fixed denominator (absolute weights)
    # @param scenario_summaries: list of summary dicts (one per scenario)
    # @param out_path: Path to write JSON
    # @param scenarios: optional list of scenario dicts for weight lookup
    # @param use_weights: bool - if True, compute weighted P(reached road)
    #                            using scenario_probability from params.json
    # @returns: dict (also written to disk)
    # Wilson 95% CI handles small ensembles properly; nearest miss =
    #       scenario that stopped closest to the road
    
    n_total = len(scenario_summaries)
    reached = [s for s in scenario_summaries if s['road']['reached']]
    n_reached = len(reached)
    frac = n_reached / n_total if n_total > 0 else 0.0
    ci_low, ci_high = h.wilson_ci(n_reached, n_total)
    
    # Optional: weighted fraction (more meaningful when scenario_probability varies)
    weighted_frac = None
    if use_weights and scenarios is not None:
        params_by_id = {sc['scenario_id']: sc['params'] for sc in scenarios}
        fixed_total = weight_total
        weight_total = 0.0
        weight_reached = 0.0
        for s in scenario_summaries:
            w = params_by_id.get(s['scenario_id'], {}).get(
                'scenario_probability', 1.0
            )
            weight_total += w
            if s['road']['reached']:
                weight_reached += w
        if fixed_total:
            weight_total = fixed_total
        if weight_total > 0:
            weighted_frac = weight_reached / weight_total
    
    # Impact points list
    impact_points = []
    for s in reached:
        impact_points.append({
            'scenario': s['scenario_id'],
            'xy': s['road']['intersection_xy'],
            'depth_m': s['road']['flow_depth_at_road_m'],
            'velocity_ms': s['road']['velocity_at_road_ms'],
            'chainage_m': s['road']['chainage_m']
        })
    
    # Nearest miss
    misses = [s for s in scenario_summaries if not s['road']['reached']]
    if misses:
        nearest = min(misses, key=lambda s: s['road']['distance_to_road_m'])
        nearest_miss = {
            'scenario': nearest['scenario_id'],
            'distance_short_m': nearest['road']['distance_to_road_m']
        }
    else:
        nearest_miss = None
    
    summary = {
        'n_scenarios': n_total,
        'n_reached': n_reached,
        'fraction_reached': round(frac, 4),
        'wilson_95_ci': [round(ci_low, 4), round(ci_high, 4)],
        'fraction_reached_weighted': (
            round(weighted_frac, 4) if weighted_frac is not None else None
        ),
        'impact_points': impact_points,
        'nearest_miss': nearest_miss
    }
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def write_scenarios_csv(scenario_summaries, out_path):
    # Flatten scenario summaries into a CSV - one row per scenario
    # @param scenario_summaries: list of summary dicts
    # @param out_path: Path to write CSV
    # @returns: Path to written CSV
    # sorted by max runout, most extreme first
    
    rows = []
    for s in scenario_summaries:
        # Extract optional road sub-fields safely
        depth_at_road = s['road']['flow_depth_at_road_m']
        vel_at_road = s['road']['velocity_at_road_ms']
        chainage = s['road']['chainage_m']
        
        rows.append({
            'scenario_id': s['scenario_id'],
            'scenario_probability': s['params'].get('scenario_probability'),
            'ensemble_weight_total': s['params'].get('ensemble_weight_total'),
            'mu': s['params'].get('mu'),
            'xi': s['params'].get('xi'),
            'rho_kgm3': s['params'].get('rho_kgm3'),
            'release_volume_m3': s['params'].get('total_volume_m3'),
            'scour_depth_params_m': s['params'].get('scour_depth_m'),
            'hand_hardness_value': s['params'].get('hand_hardness_value'),
            'mean_depth_m': s['params'].get('mean_depth_m'),
            'release_area_m2': s['params'].get('release_area_m2'),
            'size_factor': s['params'].get('size_factor'),
            'depth_percentile': s['params'].get('depth_percentile'),
            'trigger_cluster': s['params'].get('trigger_cluster'),
            'runout_max_m': round(s['runout_max_distance_m'], 1),
            'peak_velocity_ms': round(s['peak_velocity_max_ms'], 2),
            'peak_flow_depth_m': round(s['peak_flow_depth_max_m'], 2),
            'peak_pressure_kpa': round(s['peak_pressure_max_kpa'], 2),
            'reach_area_m2': round(s['reach_area_m2'], 1),
            'reached_road': s['road']['reached'],
            'distance_to_road_m': round(s['road']['distance_to_road_m'], 1),
            'depth_at_road_m': round(depth_at_road, 2) if depth_at_road else None,
            'velocity_at_road_ms': round(vel_at_road, 2) if vel_at_road else None,
            'chainage_m': round(chainage, 1) if chainage else None,
            'entrainment_method': s['entrainment'].get('method_used'),
            'entrainment_area_m2': s['entrainment'].get('area_m2'),
            'entrainment_depth_used_m': s['entrainment'].get('depth_m'),
            'entrainment_depth_source': s['entrainment'].get('depth_source'),
        })
        # One reached_<ref_name> column per reference point
        for r in s.get('reference_points') or []:
            rows[-1][f"reached_{r['ref_name']}"] = r['reached']
    
    df = pd.DataFrame(rows)
    df = df.sort_values('runout_max_m', ascending=False)
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    
    return out_path


def run_ensemble_stage(scenario_summaries, scenarios, shared_paths, config):
    # Run the full ensemble post-processing (call after all sims complete)
    # @param scenario_summaries: list of dicts (one per completed scenario)
    # @param scenarios: list of scenario dicts (for paths back to inputs)
    # @param shared_paths: dict with 'dem', 'forest', 'road'
    # @param config: top-level config dict
    # @returns: dict with paths to all generated outputs
    # @raises RuntimeError: if no reach masks are found to stack
    
    outputs_dir = Path(config['paths']['outputs_dir'])
    outputs_dir.mkdir(exist_ok=True)
    
    # Build a lookup of scenario_id -> params from scenarios list
    # so we can pull scenario_probability for stacking weights
    params_by_id = {sc['scenario_id']: sc['params'] for sc in scenarios}
    
    # Lookup of scenario_id -> directory that holds per-scenario outputs.
    # work_dir when the run used a separate work tree, scenario_dir otherwise.
    dir_by_id = {
        sc['scenario_id']: Path(sc.get('work_dir') or sc['scenario_dir'])
        for sc in scenarios
    }
    
    # 1. Collect reach mask paths AND parallel weights array
    # Lists must stay in the same order so weights[i] corresponds to mask_paths[i]
    mask_paths = []
    weights = []
    for s in scenario_summaries:
        # Find this scenario's reach mask via its scenario_id
        sid = s['scenario_id']
        base = dir_by_id.get(
            sid, Path(config['paths']['scenarios_dir']) / sid
        )
        mask_path = base / 'outputs' / 'reach_mask.tif'
        if mask_path.exists():
            mask_paths.append(mask_path)
            # Collect scenario_probability for weighting (default 1.0 if missing)
            weight = params_by_id.get(sid, {}).get('scenario_probability', 1.0)
            weights.append(float(weight))
        else:
            logging.warning(f"Missing reach mask for {sid}: {mask_path}")
    
    if not mask_paths:
        # throw error if nothing to stack
        raise RuntimeError("No reach masks found; cannot build probability raster")
    
    # Decide whether to use weighting
    # Honor config flag, but only if weights actually vary (not all 1.0)
    use_weights = config.get('probability', {}).get(
        'weight_by_scenario_probability', False
    )
    weights_vary = len(set(round(w, 6) for w in weights)) > 1
    abs_total = ensemble_weight_total(params_by_id.values())
    
    if use_weights and abs_total:
        # absolute weights: weight even when they are all equal, because
        # the denominator is fixed, not their sum
        stack_weights = weights
        weights_vary = True
        logging.info(
            f"Stacking with absolute scenario_probability weights "
            f"(sum={sum(weights):.4f} of {abs_total:.4f}; the rest is no release)"
        )
    elif use_weights and weights_vary:
        stack_weights = weights
        logging.info(
            f"Stacking with scenario_probability weights "
            f"(min={min(weights):.4f}, max={max(weights):.4f}, "
            f"sum={sum(weights):.4f})"
        )
    elif use_weights and not weights_vary:
        stack_weights = None
        logging.warning(
            "weight_by_scenario_probability=True but all weights are equal; "
            "falling back to equal-weight stacking"
        )
    else:
        stack_weights = None
        logging.info("Stacking with equal weights (probability weighting disabled)")
    
    # 2. Probability raster
    prob_path = outputs_dir / 'probability.tif'
    stack_reach_masks(mask_paths, prob_path, weights=stack_weights,
                      total=abs_total if stack_weights is not None else None)
    
    # 3. Envelope + contours
    polygonize_probability(
        probability_path=prob_path,
        contour_levels=config['probability']['contour_levels'],
        envelope_path=outputs_dir / 'envelope.geojson',
        contours_path=outputs_dir / 'contours.geojson',
        envelope_threshold=config['probability']['envelope_min_probability']
    )
    
    # 4. Most-likely-path (medial axis at P >= 0.5)
    extract_most_likely_path(
        probability_path=prob_path,
        threshold=0.5,
        out_path=outputs_dir / 'most_likely_path.geojson'
    )
    
    # 5. Road impact summary
    road_summary = build_road_impact_summary(
        scenario_summaries,
        outputs_dir / 'road_impact_summary.json',
        scenarios=scenarios,
        use_weights=use_weights and weights_vary,
        weight_total=abs_total if use_weights else None
    )
    
    # 6. Scenarios CSV
    write_scenarios_csv(
        scenario_summaries,
        outputs_dir / 'scenarios_summary.csv'
    )

    # 7. Reference points (tower, bench, road, ...) when the path has them
    refpoint_summary = build_reference_point_summary(
        scenario_summaries,
        outputs_dir / 'reference_points_summary.json',
        scenarios=scenarios,
        use_weights=use_weights and weights_vary,
        weight_total=abs_total if use_weights else None
    )

    logging.info(f"Ensemble outputs written to {outputs_dir}")
    
    return {
        'probability': prob_path,
        'envelope': outputs_dir / 'envelope.geojson',
        'contours': outputs_dir / 'contours.geojson',
        'most_likely_path': outputs_dir / 'most_likely_path.geojson',
        'road_summary': outputs_dir / 'road_impact_summary.json',
        'scenarios_csv': outputs_dir / 'scenarios_summary.csv',
        'reference_points': outputs_dir / 'reference_points_summary.json',
        # In-memory copies for the report builder
        'road_summary_dict': road_summary,
        'reference_points_dict': refpoint_summary,
    }


def build_reference_point_summary(scenario_summaries, out_path,
                                  scenarios=None, use_weights=False,
                                  weight_total=None):
    # Aggregate reach results at each reference point across the ensemble
    # @param weight_total: optional fixed denominator (absolute weights)
    # @param scenario_summaries: list of summary dicts (one per scenario)
    # @param out_path: Path to write JSON (a CSV is written beside it)
    # @param scenarios: optional list of scenario dicts for weight lookup
    # @param use_weights: bool - also compute the weighted fraction
    # @returns: dict with one entry per point (also written to disk); an
    #          empty dict when no scenario carries reference point results
    # nearest miss per point is the scenario that stopped closest

    points = {}
    params_by_id = {}
    if scenarios is not None:
        params_by_id = {sc['scenario_id']: sc['params'] for sc in scenarios}

    for s in scenario_summaries:
        w = float(params_by_id.get(s['scenario_id'], {}).get(
            'scenario_probability', 1.0) or 1.0)
        for r in s.get('reference_points') or []:
            p = points.setdefault(r['ref_name'], {
                'ref_name': r['ref_name'],
                'distance_from_release_m': r.get('distance_from_release_m'),
                'n_scenarios': 0, 'n_reached': 0,
                'weight_total': 0.0, 'weight_reached': 0.0,
                'nearest_miss': None, 'reached_by': [],
            })
            p['n_scenarios'] += 1
            p['weight_total'] += w
            if r['reached']:
                p['n_reached'] += 1
                p['weight_reached'] += w
                p['reached_by'].append(s['scenario_id'])
            else:
                nm = p['nearest_miss']
                if nm is None or r['distance_short_m'] < nm['distance_short_m']:
                    p['nearest_miss'] = {
                        'scenario': s['scenario_id'],
                        'distance_short_m': float(r['distance_short_m']),
                        'scenario_probability': w,
                    }

    if not points:
        return {}

    summary = {}
    rows = []
    for name, p in points.items():
        n = p['n_scenarios']
        frac = p['n_reached'] / n if n else 0.0
        ci_low, ci_high = h.wilson_ci(p['n_reached'], n)
        denom = weight_total or p['weight_total']
        wfrac = (p['weight_reached'] / denom
                 if use_weights and denom > 0 else None)
        summary[name] = {
            'ref_name': name,
            'distance_from_release_m': p['distance_from_release_m'],
            'n_scenarios': n,
            'n_reached': p['n_reached'],
            'fraction_reached': round(frac, 4),
            'wilson_95_ci': [round(ci_low, 4), round(ci_high, 4)],
            'fraction_reached_weighted': round(wfrac, 4) if wfrac is not None else None,
            'nearest_miss': p['nearest_miss'],
            'reached_by': p['reached_by'],
        }
        rows.append({k: v for k, v in summary[name].items()
                     if k not in ('reached_by', 'nearest_miss', 'wilson_95_ci')})

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    pd.DataFrame(rows).sort_values('distance_from_release_m').to_csv(
        out_path.with_suffix('.csv'), index=False
    )
    logging.info(f"Reference points: {len(summary)} point(s) summarized")
    return summary