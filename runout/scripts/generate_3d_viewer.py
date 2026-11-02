"""
generate_3d_viewer.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    Standalone runner for build_3d_viewer. Generates viewer_3d.html for an
    existing completed run without re-running simulations. Reads the
    ensemble outputs (probability.tif, envelope.geojson, contours.geojson)
    and per-scenario summaries directly.

Usage:
    python3 generate_3d_viewer.py [--config CONFIG] [--outputs-dir DIR]

    --config       config.yaml in the current directory (default)
    --outputs-dir  value of paths.outputs_dir from the config (default)
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

import avalanche_helpers as h
import build_3d_viewer


def discover_scenarios(scenarios_dir, outputs_dir):
    # Find every scenario folder with a completed summary.json
    # @param scenarios_dir: Path to scenarios/ directory
    # @param outputs_dir: ensemble outputs dir (unused; kept for symmetry)
    # @returns: list of scenario dicts shaped like avalanche_runout's
    # @raises FileNotFoundError: if scenarios_dir does not exist

    scenarios_dir = Path(scenarios_dir)
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"scenarios_dir not found: {scenarios_dir}")

    out = []
    for sd in sorted(scenarios_dir.glob('scenario_*')):
        if not sd.is_dir():
            continue
        summary_path = sd / 'outputs' / 'summary.json'
        if not summary_path.exists():
            logging.debug(f"  Skipping {sd.name}: no summary.json")
            continue

        release_path = sd / 'release' / 'release.geojson'
        if not release_path.exists():
            release_path = sd / 'release.geojson'
        if not release_path.exists():
            release_path = sd / 'inputs' / 'release.geojson'
        if not release_path.exists():
            logging.debug(f"  Skipping {sd.name}: no release.geojson")
            continue

        out.append({
            'scenario_id': sd.name,
            'scenario_dir': str(sd),
            'release_geojson': str(release_path),
        })

    return out


def load_impact_points(outputs_dir):
    # Pull the impact_points list out of road_impact_summary.json
    # @param outputs_dir: ensemble outputs dir
    # @returns: list of impact point dicts ([] if missing/unreadable)
    p = Path(outputs_dir) / 'road_impact_summary.json'
    if not p.exists():
        logging.warning(f"No road_impact_summary.json at {p}; viewer will have no impact points")
        return []
    try:
        with open(p) as f:
            data = json.load(f)
        return data.get('impact_points', []) or []
    except Exception as e:
        logging.warning(f"Could not read road_impact_summary.json: {e}")
        return []


def main():
    # Entry point: load config, find scenarios, build the 3D viewer
    # @param: None (args from command line)
    # @returns: None (exits with status code)

    parser = argparse.ArgumentParser(
        description='Generate viewer_3d.html from existing ensemble outputs'
    )
    parser.add_argument('--config', default='config.yaml',
                        help='Path to config YAML (default: config.yaml)')
    parser.add_argument('--outputs-dir', default=None,
                        help='Override outputs_dir from config')
    parser.add_argument('--verbose', '-v', action='store_true')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
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

    # Resolve outputs_dir (CLI override > config)
    if args.outputs_dir:
        outputs_dir = Path(args.outputs_dir)
    else:
        outputs_dir = Path(config['paths']['outputs_dir'])
    if not outputs_dir.exists():
        logging.error(f"outputs_dir not found: {outputs_dir}")
        sys.exit(1)
    logging.info(f"Using outputs_dir: {outputs_dir}")

    # Sanity-check that the ensemble outputs are present
    required = ['probability.tif', 'envelope.geojson', 'contours.geojson']
    missing = [f for f in required if not (outputs_dir / f).exists()]
    if missing:
        logging.error(
            f"outputs_dir is missing ensemble outputs: {missing}. "
            "This script only regenerates the 3D viewer; "
            "the underlying ensemble outputs must already exist."
        )
        sys.exit(1)
    logging.info(f"Found ensemble outputs: {required}")

    # Resolve shared_paths
    shared_paths = {
        'dem': Path(config['paths']['dem']),
        'forest': Path(config['paths']['forest']),
        'road': Path(config['paths']['road']),
    }
    # If a merged DEM exists in the outputs dir (from a previous mosaic run),
    # prefer it - that's what was used for the simulation
    merged = outputs_dir / 'merged_dem.tif'
    if merged.exists():
        shared_paths['dem'] = merged
        logging.info(f"Using merged DEM from run: {merged}")

    for label, p in shared_paths.items():
        if not p.exists():
            logging.error(f"Missing shared {label}: {p}")
            sys.exit(1)

    # Discover scenarios
    scenarios_dir = Path(config['paths']['scenarios_dir'])
    logging.info(f"Discovering scenarios under: {scenarios_dir}")
    scenarios = discover_scenarios(scenarios_dir, outputs_dir)
    logging.info(f"Found {len(scenarios)} completed scenario(s)")
    if not scenarios:
        logging.error("No completed scenarios found; cannot build viewer")
        sys.exit(1)

    # Impact points
    impact_points = load_impact_points(outputs_dir)
    logging.info(f"Loaded {len(impact_points)} impact point(s)")

    # Build the 3D viewer
    logging.info("Building 3D viewer...")
    out_path = build_3d_viewer.build_3d_viewer(
        outputs_dir=outputs_dir,
        scenarios=scenarios,
        shared_paths=shared_paths,
        impact_points=impact_points,
        config=config,
    )

    size_kb = out_path.stat().st_size / 1024
    logging.info(f"Done. Wrote {out_path} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()
