"""
rethreshold_reach.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Re-threshold reach masks and rebuild the ensemble outputs from existing
    per-scenario peak rasters. Does not run com1DFA. Use to test different
    reach thresholds (depth, pressure, velocity) without re-simulating.

Usage:
    python rethreshold_reach.py [--config CONFIG] [--outputs-dir DIR]
        [--depth-threshold M] [--pressure-threshold KPA]
        [--velocity-threshold MS]
"""
import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import yaml
import numpy as np
import rasterio
import geopandas as gpd

import avalanche_helpers as h
import avalanche_simulation as sim
import avalanche_stack as stack
import build_viewer
try:
    import build_3d_viewer
    HAS_3D = True
except Exception:
    HAS_3D = False


def find_scenarios(scenarios_dir, work_dir=None):
    # Discover scenarios that have peak rasters to re-threshold
    # @param scenarios_dir: Path to scenarios/ directory (inputs)
    # @param work_dir: optional Path to a separate work tree holding the
    #                  per-scenario outputs (paths.scenario_work_dir). When
    #                  given, peak rasters are found there and inputs
    #                  (release.geojson, params.json) fall back to
    #                  scenarios_dir. scenario_dir in the returned dicts is
    #                  the directory that holds outputs, so downstream
    #                  writes land in the work tree.
    # @returns: list of dicts with scenario_id, scenario_dir,
    #          release_geojson, peak_files, params
    # only scenarios with a peak_flow_depth.tif are included; peak
    #       files may sit in the scenario root or under outputs/
    scenarios_dir = Path(scenarios_dir)
    scan_dir = Path(work_dir) if work_dir else scenarios_dir
    out = []
    for sd in sorted(scan_dir.glob('scenario_*')):
        if not sd.is_dir():
            continue
        # Matching input folder in the data tree (same folder when no
        # separate work tree is in use)
        data_sd = scenarios_dir / sd.name
        # Locate peak rasters
        pf = {}
        for short, names in [
            ('pft', ['peak_flow_depth.tif', 'depth.tif']),
            ('pfv', ['peak_velocity.tif']),
            ('ppr', ['peak_pressure.tif']),
            ('pta', ['peak_travel_angle.tif']),
        ]:
            found = None
            for nm in names:
                for candidate in (sd / nm, sd / 'outputs' / nm):
                    if candidate.exists():
                        found = candidate
                        break
                if found:
                    break
            pf[short] = found
        if pf.get('pft') is None:
            logging.debug(f"Skip {sd.name}: no peak_flow_depth.tif")
            continue

        release_path = None
        for candidate in (sd / 'release' / 'release.geojson',
                          sd / 'release.geojson',
                          sd / 'inputs' / 'release.geojson',
                          data_sd / 'release' / 'release.geojson',
                          data_sd / 'release.geojson',
                          data_sd / 'inputs' / 'release.geojson'):
            if candidate.exists():
                release_path = candidate
                break
        if release_path is None:
            logging.debug(f"Skip {sd.name}: no release.geojson")
            continue

        # scenario_probability comes from params.json (used for weighted stack)
        params = {}
        params_path = None
        for candidate in (sd / 'release' / 'params.json',
                          sd / 'params.json',
                          sd / 'inputs' / 'params.json',
                          data_sd / 'release' / 'params.json',
                          data_sd / 'params.json',
                          data_sd / 'inputs' / 'params.json'):
            if candidate.exists():
                params_path = candidate
                break
        if params_path is not None:
            try:
                with open(params_path) as f:
                    params = json.load(f)
            except Exception:
                pass

        out.append({
            'scenario_id': sd.name,
            'scenario_dir': str(sd),
            'release_geojson': str(release_path),
            'peak_files': pf,
            'params': params,
        })

    return out


def rebuild_scenario_reach(scenario, depth_threshold_m,
                           pressure_threshold_kpa, velocity_threshold_ms):
    # Write a new reach_mask.tif and reach_polygon.geojson for one scenario
    # @param scenario: scenario dict from find_scenarios
    # @param depth_threshold_m: flow depth threshold (m)
    # @param pressure_threshold_kpa: pressure threshold (kPa)
    # @param velocity_threshold_ms: velocity threshold (m/s)
    # @returns: Path to the written reach mask
    sd = Path(scenario['scenario_dir'])
    out_dir = sd / 'outputs'
    out_dir.mkdir(exist_ok=True)

    pf = scenario['peak_files']
    reach_path = out_dir / 'reach_mask.tif'

    sim.build_reach_mask(
        pft_path=pf['pft'],
        out_path=reach_path,
        depth_threshold_m=depth_threshold_m,
        pfv_path=pf.get('pfv'),
        velocity_threshold_ms=velocity_threshold_ms,
        ppr_path=pf.get('ppr'),
        pressure_threshold_kpa=pressure_threshold_kpa,
    )

    # Vectorize mask to polygon
    reach, meta = h.read_raster(reach_path)
    polys = h.mask_to_polygons(reach.astype(bool), meta['transform'])
    if polys:
        gdf = gpd.GeoDataFrame(
            {'scenario': [scenario['scenario_id']] * len(polys)},
            geometry=polys, crs=meta['crs']
        ).to_crs('EPSG:4326')
        gdf.to_file(out_dir / 'reach_polygon.geojson', driver='GeoJSON')
    else:
        # Empty - write an empty file so downstream can read it
        gdf = gpd.GeoDataFrame({'scenario': []}, geometry=[], crs='EPSG:4326')
        gdf.to_file(out_dir / 'reach_polygon.geojson', driver='GeoJSON')

    return reach_path


def update_scenario_summary(scenario, depth_threshold_m, pressure_threshold_kpa):
    # Update summary.json reach_area_m2 and threshold record so the report
    # and viewers see consistent numbers
    # @param scenario: scenario dict from find_scenarios
    # @param depth_threshold_m: flow depth threshold (m)
    # @param pressure_threshold_kpa: pressure threshold (kPa)
    # @returns: None
    sd = Path(scenario['scenario_dir'])
    summary_path = sd / 'outputs' / 'summary.json'
    if not summary_path.exists():
        return

    try:
        with open(summary_path) as f:
            s = json.load(f)
    except Exception:
        return

    reach_path = sd / 'outputs' / 'reach_mask.tif'
    if reach_path.exists():
        reach, meta = h.read_raster(reach_path)
        cell_area_m2 = abs(meta['transform'].a * meta['transform'].e)
        s['reach_area_m2'] = float(reach.sum() * cell_area_m2)

    # Record the threshold actually used so we know what produced these numbers
    s.setdefault('reach', {})
    s['reach']['flow_depth_threshold_m'] = depth_threshold_m
    s['reach']['pressure_threshold_kpa'] = pressure_threshold_kpa

    with open(summary_path, 'w') as f:
        json.dump(s, f, indent=2)


def main():
    # Re-threshold all scenarios and rebuild ensemble outputs + viewers
    # @param: None (args from command line)
    # @returns: None (exits with status code)

    parser = argparse.ArgumentParser(
        description='Re-threshold reach masks and rebuild ensemble outputs'
    )
    parser.add_argument('--config', default='config.yaml')
    parser.add_argument('--outputs-dir', default=None,
                        help='Override outputs_dir from config')
    parser.add_argument('--depth-threshold', type=float, default=None,
                        help='Flow depth threshold (m); default: from config')
    parser.add_argument('--pressure-threshold', type=float, default=None,
                        help='Pressure threshold (kPa); default: from config')
    parser.add_argument('--velocity-threshold', type=float, default=None,
                        help='Velocity threshold (m/s); default: from config')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Resolve thresholds
    rc = config.get('reach', {})
    depth_threshold = (
        args.depth_threshold if args.depth_threshold is not None
        else float(rc.get('flow_depth_threshold_m', 0.1))
    )
    pressure_threshold = (
        args.pressure_threshold if args.pressure_threshold is not None
        else float(rc.get('pressure_threshold_kpa', 0.0))
    )
    velocity_threshold = (
        args.velocity_threshold if args.velocity_threshold is not None
        else float(rc.get('min_velocity_ms', 0.0))
    )

    logging.info(f"Reach thresholds: depth>={depth_threshold} m, "
                 f"pressure>={pressure_threshold} kPa, "
                 f"velocity>={velocity_threshold} m/s")

    # Resolve outputs_dir
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        outputs_dir = Path(config['paths']['outputs_dir'])
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Outputs dir: {outputs_dir}")
    # Carry the thresholds into the config the viewer/stack will read so the
    # numbers in the viewer info panel match what we actually used.
    config['paths']['outputs_dir'] = str(outputs_dir)
    config.setdefault('reach', {})
    config['reach']['flow_depth_threshold_m'] = depth_threshold
    config['reach']['pressure_threshold_kpa'] = pressure_threshold
    config['reach']['min_velocity_ms'] = velocity_threshold

    # Resolve shared paths
    shared_paths = {
        'dem': Path(config['paths']['dem']),
        'forest': Path(config['paths']['forest']),
        'road': Path(config['paths']['road']),
    }
    # Prefer merged DEM if it exists from a previous run
    merged = outputs_dir / 'merged_dem.tif'
    if merged.exists():
        shared_paths['dem'] = merged
    for label, p in shared_paths.items():
        if not p.exists():
            logging.error(f"Missing shared {label}: {p}")
            sys.exit(1)

    # Discover scenarios. If the run used a separate work tree
    # (paths.scenario_work_dir), the peak rasters live there.
    scenarios_dir = Path(config['paths']['scenarios_dir'])
    work_root = config['paths'].get('scenario_work_dir')
    scenarios = find_scenarios(scenarios_dir, work_dir=work_root)
    logging.info(f"Found {len(scenarios)} scenarios with peak_flow_depth.tif")
    if not scenarios:
        logging.error("No scenarios found; aborting")
        sys.exit(1)

    # Stage 1: rebuild per-scenario reach masks + polygons
    logging.info("Stage 1: rebuilding reach masks and polygons")
    for sc in scenarios:
        rebuild_scenario_reach(
            sc,
            depth_threshold_m=depth_threshold,
            pressure_threshold_kpa=pressure_threshold,
            velocity_threshold_ms=velocity_threshold,
        )
        update_scenario_summary(sc, depth_threshold, pressure_threshold)
    logging.info(f"  Rebuilt reach masks for {len(scenarios)} scenarios")

    # Stage 2: stack into probability raster
    logging.info("Stage 2: stacking probability raster")
    mask_paths = [
        Path(sc['scenario_dir']) / 'outputs' / 'reach_mask.tif'
        for sc in scenarios
    ]
    # Drop any that don't exist (shouldn't happen, but safe)
    mask_paths = [p for p in mask_paths if p.exists()]

    weights = None
    if config['probability'].get('weight_by_scenario_probability'):
        weights = []
        for sc in scenarios:
            mp = Path(sc['scenario_dir']) / 'outputs' / 'reach_mask.tif'
            if not mp.exists():
                continue
            w = float(sc['params'].get('scenario_probability', 0) or 0)
            weights.append(w if w > 0 else 1.0)
        if not any(w != weights[0] for w in weights):
            weights = None  # all equal -> unweighted

    prob_path = outputs_dir / 'probability.tif'
    stack.stack_reach_masks(mask_paths, prob_path, weights=weights)

    # Stage 3: polygonize envelope + contours
    logging.info("Stage 3: building envelope and contours")
    stack.polygonize_probability(
        probability_path=prob_path,
        contour_levels=config['probability']['contour_levels'],
        envelope_path=outputs_dir / 'envelope.geojson',
        contours_path=outputs_dir / 'contours.geojson',
        envelope_threshold=config['probability']['envelope_min_probability'],
    )

    # Stage 4: rebuild road impact summary
    logging.info("Stage 4: rebuilding road impact summary")
    scenario_summaries = []
    for sc in scenarios:
        sp = Path(sc['scenario_dir']) / 'outputs' / 'summary.json'
        if sp.exists():
            try:
                with open(sp) as f:
                    scenario_summaries.append(json.load(f))
            except Exception:
                pass

    road_summary = stack.build_road_impact_summary(
        scenario_summaries=scenario_summaries,
        out_path=outputs_dir / 'road_impact_summary.json',
        scenarios=scenarios,
        use_weights=bool(
            config['probability'].get('weight_by_scenario_probability', False)
        ),
    )

    # Stage 5: scenarios CSV
    logging.info("Stage 5: writing scenarios summary CSV")
    stack.write_scenarios_csv(
        scenario_summaries=scenario_summaries,
        out_path=outputs_dir / 'scenarios_summary.csv',
    )

    # Stage 6: viewers
    logging.info("Stage 6: building viewers")
    impact_points = road_summary.get('impact_points', []) or []

    build_viewer.build_viewer(
        outputs_dir=outputs_dir,
        scenarios=scenarios,
        shared_paths=shared_paths,
        impact_points=impact_points,
        config=config,
    )

    if HAS_3D:
        build_3d_viewer.build_3d_viewer(
            outputs_dir=outputs_dir,
            scenarios=scenarios,
            shared_paths=shared_paths,
            impact_points=impact_points,
            config=config,
        )
    else:
        logging.info("  (build_3d_viewer not available; skipping 3D viewer)")

    logging.info(f"Done. Outputs in: {outputs_dir}")


if __name__ == '__main__':
    main()
