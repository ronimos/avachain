"""
avalanche_runout.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Top-level workflow for the avalanche runout envelope. Reads N release
    scenarios (each with its own snowpack model output), runs each through
    AvaFrame com1DFA, stacks the per-scenario reach masks into a probability
    raster, and answers: does the slide reach the road?

    Per-scenario inputs live in scenarios/scenario_NNN/inputs/.
    Shared inputs (DEM, forest, road) are set in config.yaml.
    Per-scenario outputs go to scenarios/scenario_NNN/outputs/.
    Ensemble outputs go to the configured outputs directory.

Three Stages:
    Stage 1: Validation
        - Discover scenarios, validate CRS consistency across all inputs
    Stage 2: Per-scenario sims (parallel)
        - For each scenario: build AvaFrame dir, run com1DFA, post-process
    Stage 3: Ensemble stacking
        - Stack reach masks -> probability + envelope + contours + report

Outputs:
    - Per scenario: peak rasters (COG), reach mask, flowpath, summary.json
    - Ensemble:
        * probability.tif
        * envelope.geojson
        * contours.geojson
        * most_likely_path.geojson
        * road_impact_summary.json
        * scenarios_summary.csv
        * report.html
        * viewer.html

Usage:
    - Validate inputs only:    python avalanche_runout.py --dry-run
    - Run all scenarios:       python avalanche_runout.py
    - Re-run post only:        python avalanche_runout.py --post-only
    - Run one scenario:        python avalanche_runout.py --scenario scenario_001
    - Sequential (debug):      python avalanche_runout.py --no-parallel
    - Skip 3D viewer:          python avalanche_runout.py --no-3d
    - Verbose logging:         python avalanche_runout.py --verbose

Requirements:
    - avaframe >= 2.1, rasterio, fiona, geopandas, shapely, pyproj, numpy,
      scipy, pandas, matplotlib, jinja2, folium, branca, scikit-image,
      pillow, pyyaml
"""

# --------- Load Libraries --------
import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import rasterio

import avalanche_helpers as h
import avalanche_simulation as sim
import avalanche_stack as stack
import build_report
import build_viewer
try:
    import build_3d_viewer
    HAS_3D = True
except Exception:
    build_3d_viewer = None
    HAS_3D = False


__version__ = '1.0.0'




def setup_logging(verbose=False):
    # Configure logging for the workflow
    # @param verbose: If True, set logging to DEBUG level, otherwise INFO
    # @returns: None
    
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def load_config(config_path):
    # Load and validate the YAML config
    # @param config_path: Path to config.yaml
    # @returns: dict of configuration parameters
    # @raises FileNotFoundError: if the config file does not exist
    # @raises ValueError: if a required config section is missing
    
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        # throw error if no config file found
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load YAML
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required sections
    required = ['paths', 'com1dfa', 'defaults', 'entrainment', 'reach',
                'probability', 'road', 'execution', 'output']
    for key in required:
        if key not in config:
            # throw error if required config section is missing
            raise ValueError(f"Missing required config section: {key}")
    
    logging.info(f"Loaded config: {config_path}")
    return config


def parse_args():
    # Parse command-line arguments
    # @param: None
    # @returns: argparse.Namespace with parsed arguments
    #     Arguments:
    #       - config: Path to config YAML
    #       - dry_run: Validate only, don't run sims
    #       - post_only: Skip sims, redo post-processing
    #       - scenario: Run only specified scenarios (can give multiple)
    #       - no_parallel: Force sequential execution
    #       - verbose: DEBUG-level logging
    
    parser = argparse.ArgumentParser(
        description='Avalanche Runout Envelope Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--config', default='config.yaml', help='Path to config YAML file')
    parser.add_argument('--dry-run', action='store_true', help='Validate inputs only; do not run simulations')
    parser.add_argument('--post-only', action='store_true', help='Skip sims; re-run post-processing on existing outputs')
    parser.add_argument('--scenario', action='append', default=None, help='Run only this scenario (can give multiple times)')
    parser.add_argument('--no-parallel', action='store_true', help='Force sequential execution (easier debugging)')
    parser.add_argument('--no-3d', action='store_true', help='Skip building the 3D viewer')
    parser.add_argument('--skip-viewers', action='store_true', help='Skip the Leaflet and 3D viewers (report.html is still built)')
    parser.add_argument('--skip-report', action='store_true', help='Skip the per-horizon report.html (the cycle report replaces it)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable DEBUG-level logging')
    
    return parser.parse_args()


def scenario_input_dir(scenario_dir):
    # Pick the folder that holds a scenario's input files
    # @param scenario_dir: Path to scenario_NNN/
    # @returns: Path - release/ or inputs/ when present, else scenario_dir
    for sub in ('release', 'inputs'):
        if (scenario_dir / sub).is_dir():
            return scenario_dir / sub
    return scenario_dir


def load_scenario(scenario_dir):
    # Load a single scenario's input files into a dict
    # @param scenario_dir: Path to scenarios/scenario_NNN/
    # @returns: dict with all per-scenario data + paths
    # @raises FileNotFoundError: if a required input file is missing
    # @raises ValueError: if params.json lacks a required key
    # required files are release.geojson, depth.asc/.tif, density.json,
    #       params.json; files can live in inputs/ or the scenario folder
    
    sid = scenario_dir.name
    
    # Input directory: release/ (avachain work tree), inputs/ (the chain's
    # layout), or the scenario folder itself
    inp = scenario_input_dir(scenario_dir)
    
    # Required: release polygon
    release_gj = inp / 'release.geojson'
    if not release_gj.exists():
        # throw error if no release polygon
        raise FileNotFoundError(f"{sid}: missing release.geojson")
    
    # Required: depth raster (.asc preferred, .tif accepted)
    depth_asc = inp / 'depth.asc'
    depth_tif = inp / 'depth.tif'
    depth = depth_asc if depth_asc.exists() else depth_tif
    if not depth.exists():
        # throw error if no depth raster
        raise FileNotFoundError(f"{sid}: missing depth.asc or depth.tif")
    
    # Required: density.json
    density_path = inp / 'density.json'
    if not density_path.exists():
        # throw error if no density file
        raise FileNotFoundError(f"{sid}: missing density.json")
    density = json.loads(density_path.read_text())
    
    # Required: params.json
    params_path = inp / 'params.json'
    if not params_path.exists():
        # throw error if no params file
        raise FileNotFoundError(f"{sid}: missing params.json")
    params = json.loads(params_path.read_text())
    
    # Validate required param keys.
    # mu/xi: per-scenario friction (used by Voellmy-type models).
    # scour_depth_m: entrainable depth; 0 is legitimate on hard slabs
    #                (the config fallback covers that case).
    # scenario_probability: ensemble weight. Required so runs without
    #                       weights fail loudly.
    for required_key in ('mu', 'xi', 'scour_depth_m', 'scenario_probability'):
        if required_key not in params:
            raise ValueError(
                f"{sid}: params.json missing required key '{required_key}'"
            )
    
    # Soft-validate optional keys. Missing is OK (logged at debug level so it
    # doesn't spam the run log).
    for optional_key in ('scour_depth_mid_m', 'hand_hardness_value'):
        if optional_key not in params:
            logging.debug(
                f"{sid}: params.json missing optional key '{optional_key}'"
            )
    
    # Optional: metadata
    meta_path = inp / 'metadata.json'
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    
    return {
        'scenario_id': sid,
        'scenario_dir': scenario_dir,
        'release_geojson': release_gj,
        'depth_raster': depth,
        'density': density,
        'params': params,
        'metadata': metadata,
    }


def discover_scenarios(scenarios_dir):
    # Find all scenario folders and load their inputs
    # @param scenarios_dir: Path to scenarios/ directory
    # @returns: list of scenario dicts (one per scenario)
    # @raises FileNotFoundError: if scenarios_dir does not exist
    # @raises RuntimeError: if no valid scenarios are found
    
    scenarios_dir = Path(scenarios_dir)
    if not scenarios_dir.exists():
        # throw error if scenarios dir is missing
        raise FileNotFoundError(f"Scenarios directory not found: {scenarios_dir}")
    
    scenarios = []
    for sub in sorted(scenarios_dir.iterdir()):
        # Skip non-scenario folders
        if not sub.is_dir() or not sub.name.startswith('scenario_'):
            continue
        
        # Skip folders with no release.geojson in any input layout
        inp = scenario_input_dir(sub)
        if not (inp / 'release.geojson').exists():
            logging.warning(f"Skipping {sub.name}: no release.geojson in {inp}")
            continue
        
        scenarios.append(load_scenario(sub))
    
    if not scenarios:
        # throw error if no scenarios found at all
        raise RuntimeError(f"No valid scenarios found in {scenarios_dir}")
    
    logging.info(f"Discovered {len(scenarios)} scenario(s)")
    return scenarios


def validate_all_crs(scenarios, shared_paths):
    # Ensure all inputs share the same CRS (anchored on the DEM)
    # @param scenarios: list of scenario dicts
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @returns: rasterio CRS object (the reference CRS)
    # @raises ValueError: if the DEM has no CRS or any input mismatches
    # the DEM is authoritative
    
    # Reference CRS from DEM
    dem_crs = h.get_crs_of(shared_paths['dem'])
    if dem_crs is None:
        # throw error if DEM has no CRS
        raise ValueError(
            f"DEM {shared_paths['dem']} has no CRS. "
            f"Assign one (e.g., gdal_translate -a_srs EPSG:XXXX) and retry."
        )
    
    logging.info(f"Reference CRS (from DEM): {dem_crs}")
    
    # Build (path, label) list for everything that needs to match
    paths_and_labels = [
        (shared_paths['forest'], 'forest'),
        (shared_paths['road'], 'road'),
    ]
    for key in ('reference_points', 'entrainment_polygon'):
        if shared_paths.get(key):
            paths_and_labels.append((shared_paths[key], key))
    for sc in scenarios:
        paths_and_labels.append(
            (sc['release_geojson'], f"{sc['scenario_id']}/release")
        )
        paths_and_labels.append(
            (sc['depth_raster'], f"{sc['scenario_id']}/depth")
        )
    
    # Validate
    success, mismatches = h.validate_crs(paths_and_labels, dem_crs)
    
    if not success:
        # throw error on any CRS mismatch
        msg = (
            "CRS validation failed. All inputs must share the DEM's CRS.\n"
            + "\n".join(f"  - {m}" for m in mismatches)
            + "\nFix with gdalwarp/ogr2ogr before re-running."
        )
        raise ValueError(msg)
    
    logging.info(f"CRS validation passed (all inputs in {dem_crs})")
    return dem_crs


def run_scenarios_parallel(scenarios, shared_paths, config, skip_sim, max_workers):
    # Run scenarios in parallel via ProcessPoolExecutor
    # @param scenarios: list of scenario dicts
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param skip_sim: bool - skip com1DFA call (post-only mode)
    # @param max_workers: int or None - process pool size
    # @returns: list of summary dicts (one per successful scenario)
    
    summaries = []
    
    logging.info(
        f"Running {len(scenarios)} scenarios in parallel "
        f"(max_workers={max_workers or 'auto'})"
    )
    
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        # Submit all scenarios
        futures = {
            ex.submit(sim.run_one_scenario, sc, shared_paths, config, skip_sim):
                sc['scenario_id']
            for sc in scenarios
        }
        
        # Collect as they finish
        for fut in as_completed(futures):
            sid = futures[fut]
            try:
                summaries.append(fut.result())
            except Exception as e:
                logging.error(f"[{sid}] pipeline failed: {e}", exc_info=True)
    
    return summaries


def run_scenarios_sequential(scenarios, shared_paths, config, skip_sim):
    # Run scenarios one at a time (easier to debug)
    # @param scenarios: list of scenario dicts
    # @param shared_paths: dict with 'dem', 'forest', 'road' Paths
    # @param config: top-level config dict
    # @param skip_sim: bool - skip com1DFA call (post-only mode)
    # @returns: list of summary dicts (one per successful scenario)
    
    summaries = []
    for sc in scenarios:
        try:
            summaries.append(
                sim.run_one_scenario(sc, shared_paths, config, skip_sim)
            )
        except Exception as e:
            logging.error(f"[{sc['scenario_id']}] pipeline failed: {e}", exc_info=True)
    
    return summaries


def main():
    # Main workflow function
    # @param: None (args from CLI, params from config.yaml)
    # @returns: None (exits with status code)
    # @raises: nothing propagates; all errors are caught, logged, and
    #          converted to exit code 1
    
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Discover scenarios
        scenarios = discover_scenarios(config['paths']['scenarios_dir'])
        
        # Optional separate work tree. When paths.scenario_work_dir is set,
        # every per-scenario write (avaframe/, outputs/, scenario.ini) goes
        # to scenario_work_dir/scenario_NNN/ and the scenario input folders
        # stay untouched. When unset, writes go into the scenario folders
        # (original layout).
        work_root = config['paths'].get('scenario_work_dir')
        if work_root:
            work_root = Path(work_root)
            work_root.mkdir(parents=True, exist_ok=True)
            for sc in scenarios:
                sc['work_dir'] = work_root / sc['scenario_id']
            logging.info(f"Per-scenario work dirs under: {work_root}")
        
        # Optional filter to specific scenarios
        if args.scenario:
            requested = set(args.scenario)
            scenarios = [s for s in scenarios if s['scenario_id'] in requested]
            if not scenarios:
                # throw error if filter excluded everything
                raise ValueError(f"No scenarios match: {args.scenario}")
        
        # Validate shared inputs exist
        shared_paths = {
            'dem': Path(config['paths']['dem']),
            'forest': Path(config['paths']['forest']),
            'road': Path(config['paths']['road'])
        }
        for label, _p in shared_paths.items():
            if not _p.exists():
                # throw error if shared input is missing
                raise FileNotFoundError(f"Missing shared {label}: {_p}")

        # Optional per-path layers. Reference points are skipped with a
        # warning when absent; an entrainment polygon is required when the
        # profile asked for polygon mode.
        rp = config['paths'].get('reference_points')
        if rp and Path(rp).exists():
            shared_paths['reference_points'] = Path(rp)
        elif rp:
            logging.warning(f"reference_points not found, skipped: {rp}")
        ent_cfg0 = config.get('entrainment', {}) or {}
        if ent_cfg0.get('mode') == 'polygon' and 'ent' in config['com1dfa']['sim_type']:
            poly = ent_cfg0.get('polygon')
            if not poly or not Path(poly).exists():
                raise FileNotFoundError(
                    f"entrainment.mode is polygon but no polygon found: {poly}"
                )
            shared_paths['entrainment_polygon'] = Path(poly)
        
        # Optional: mosaic a high-res priority DEM (bare-ground DSM) onto
        # the fallback DEM. The DSM wins where it has data; the fallback
        # covers the rest. Result is cached under the outputs dir and
        # used as the working DEM for all downstream steps.
        dem_priority = config['paths'].get('dem_priority')
        if dem_priority:
            dem_priority = Path(dem_priority)
            if not dem_priority.exists():
                raise FileNotFoundError(
                    f"dem_priority not found: {dem_priority}"
                )
            outputs_dir = Path(config['paths']['outputs_dir'])
            outputs_dir.mkdir(parents=True, exist_ok=True)
            merged_dem = outputs_dir / 'merged_dem.tif'
            overlap_stats = outputs_dir / 'dem_overlap_stats.json'
            feather_cells = int(
                config['paths'].get('dem_feather_cells', 5)
            )
            vcorr = float(
                config['paths'].get('dem_vertical_correction_m', 0.0)
            )
            h.mosaic_dem_with_priority(
                primary_path=dem_priority,
                fallback_path=shared_paths['dem'],
                out_path=merged_dem,
                feather_cells=feather_cells,
                overlap_check=True,
                overlap_stats_out=overlap_stats,
                vertical_correction_m=vcorr
            )
            shared_paths['dem'] = merged_dem
            logging.info(
                f"Using merged DEM for simulations: {merged_dem}"
            )
        
        # Validate CRS consistency across all inputs
        if config['execution']['abort_on_crs_mismatch']:
            validate_all_crs(scenarios, shared_paths)
        
        # Validate snow-height DSM upfront when enabled. .npy files have no
        # CRS or transform, so we check shape against the DEM here rather
        # than letting the first scenario crash mid-run.
        ent_cfg = config.get('entrainment', {})
        if ent_cfg.get('use_snow_height_dsm', False):
            sh_path = ent_cfg.get('snow_height_raster')
            if not sh_path or not Path(sh_path).exists():
                raise FileNotFoundError(
                    f"use_snow_height_dsm=True but snow_height_raster not "
                    f"found: {sh_path}"
                )
            sh_path = Path(sh_path)
            if sh_path.suffix.lower() == '.npy':
                # Cheap shape check via memmap (avoids loading the whole array)
                arr = np.load(sh_path, mmap_mode='r')
                with rasterio.open(shared_paths['dem']) as dem_src:
                    dem_shape = (dem_src.height, dem_src.width)
                if arr.shape != dem_shape:
                    raise ValueError(
                        f"Snow height .npy shape {arr.shape} does not match "
                        f"DEM shape {dem_shape}. .npy files must be on the "
                        f"DEM grid (no spatial metadata to resample with)."
                    )
                logging.info(
                    f"Snow height DSM validated: {sh_path.name} "
                    f"shape={arr.shape}"
                )
            else:
                # GeoTIFF / .asc - confirm it opens and matches the DEM CRS
                # (horizontal sub-CRS only, since DEMs commonly carry a
                # vertical datum the vector layers don't have).
                with rasterio.open(sh_path) as src:
                    sh_crs = src.crs
                dem_crs = h.get_crs_of(shared_paths['dem'])
                sh_horiz = h.horizontal_crs(sh_crs)
                dem_horiz = h.horizontal_crs(dem_crs)
                if (
                    config['execution']['abort_on_crs_mismatch']
                    and sh_horiz is not None and sh_horiz != dem_horiz
                ):
                    sh_str = sh_horiz.to_authority() or sh_horiz.to_string()
                    dem_str = dem_horiz.to_authority() or dem_horiz.to_string()
                    raise ValueError(
                        f"Snow height DSM horizontal CRS {sh_str} does not "
                        f"match DEM horizontal CRS {dem_str}. Reproject "
                        f"with gdalwarp before re-running."
                    )
                logging.info(f"Snow height DSM validated: {sh_path.name}")
        
        # Stop here if --dry-run
        if args.dry_run:
            logging.info(f"Dry run complete. {len(scenarios)} scenarios validated.")
            return
        
        # Stage 2: per-scenario sims (parallel or sequential)
        use_parallel = (
            config['execution']['parallel']
            and not args.no_parallel
            and len(scenarios) > 1
        )
        
        if use_parallel:
            summaries = run_scenarios_parallel(
                scenarios=scenarios,
                shared_paths=shared_paths,
                config=config,
                skip_sim=args.post_only,
                max_workers=config['execution'].get('max_workers')
            )
        else:
            logging.info(f"Running {len(scenarios)} scenarios sequentially")
            summaries = run_scenarios_sequential(
                scenarios=scenarios,
                shared_paths=shared_paths,
                config=config,
                skip_sim=args.post_only
            )
        
        if not summaries:
            # throw error if everything failed
            raise RuntimeError(
                "No scenarios completed successfully; aborting ensemble stage"
            )
        
        # Stage 3: ensemble stacking
        logging.info("Building ensemble outputs")
        ensemble_paths = stack.run_ensemble_stage(
            scenario_summaries=summaries,
            scenarios=scenarios,
            shared_paths=shared_paths,
            config=config
        )
        
        # Stage 4: reports
        if args.skip_report:
            logging.info("Skipping per-horizon report (--skip-report)")
        else:
            logging.info("Building HTML report")
            build_report.build_report(
                outputs_dir=Path(config['paths']['outputs_dir']),
                scenario_summaries=summaries,
                road_impact=ensemble_paths['road_summary_dict'],
                shared_paths=shared_paths,
                config=config,
                wrapper_version=__version__
            )
        
        if args.skip_viewers:
            logging.info("Skipping viewers (--skip-viewers)")
        else:
            logging.info("Building Leaflet viewer")
            build_viewer.build_viewer(
                outputs_dir=Path(config['paths']['outputs_dir']),
                scenarios=scenarios,
                shared_paths=shared_paths,
                impact_points=ensemble_paths['road_summary_dict']['impact_points'],
                config=config
            )
        
        if args.skip_viewers:
            pass
        elif args.no_3d:
            logging.info("Skipping 3D viewer (--no-3d)")
        elif not HAS_3D:
            logging.info("build_3d_viewer not available; skipping 3D viewer")
        else:
            logging.info("Building 3D viewer")
            build_3d_viewer.build_3d_viewer(
                outputs_dir=Path(config['paths']['outputs_dir']),
                scenarios=scenarios,
                shared_paths=shared_paths,
                impact_points=ensemble_paths['road_summary_dict']['impact_points'],
                config=config
            )
        
        # woohoo success
        logging.info("\n" + "="*70)
        logging.info(f" Workflow Complete - {len(summaries)}/{len(scenarios)} scenarios")
        logging.info("="*70)
    
    except Exception as e:
        # Error handling
        logging.error(f"\nWorkflow Failed: {e}")
        
        # Print full traceback if verbose
        if args.verbose:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)


if __name__ == "__main__":
    main()