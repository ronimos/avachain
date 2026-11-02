"""
smoke_test.py

Author: Valerie Foley
Last Updated: 10/2026

Description:
    End-to-end check of the runout side without AvaFrame. Builds a
    synthetic path (planar DEM, release box, forest patch, road, three
    reference points), a fake scenario drop in the scenario layout, an observed
    event record, and a config folder pointing at them. Then runs:

        forecast   two horizons, two scenarios each, polygon entrainment
        sweep      two friction classes, three depths, with and without
                   entrainment
        calibrate  a tau0 grid against the event
        verify     the forecast against the event

    com1DFA is replaced by a stand-in (AVACHAIN_STUB_COM1DFA) that draws a
    band of flow downslope whose length grows with depth and shrinks with
    tau0, so rankings and reach tests behave. It is not physics.

    Run from the repo root. Takes a few minutes. Exit code 0 when every
    stage succeeds and the expected products exist.

Usage:
    python tests/smoke_test.py                 # work dir under /tmp
    python tests/smoke_test.py --keep DIR      # keep the tree at DIR
"""

# --------- Load Libraries --------
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
from rasterio.transform import from_origin
import geopandas as gpd
import yaml
from pyproj import CRS
from shapely.geometry import box, LineString, Point

REPO = Path(__file__).resolve().parent.parent
CRS_CODE = 'EPSG:6342'
X0, Y0 = 440000.0, 4390000.0
NROW, NCOL = 400, 200


def _gj(geom, path, **attrs):
    # Write one feature as GeoJSON (or shapefile by extension)
    cols = {k: [v] for k, v in attrs.items()} or {'id': [1]}
    gdf = gpd.GeoDataFrame(cols, geometry=[geom], crs=CRS_CODE)
    driver = 'ESRI Shapefile' if path.suffix == '.shp' else 'GeoJSON'
    gdf.to_file(path, driver=driver)


def build_tree(root):
    # Synthetic avachain tree, scenario drop, event record and config
    # @param root: Path to build under
    # @returns: Path to the config folder
    sp = root / "avachain"
    transform = from_origin(X0, Y0, 1.0, 1.0)
    rows = np.arange(NROW)[:, None] * np.ones((1, NCOL))
    z = 3500.0 - np.where(rows < 250, rows * 0.70, 250 * 0.70 + (rows - 250) * 0.14)
    prof = dict(driver='GTiff', height=NROW, width=NCOL, count=1, dtype='float32',
                crs=CRS_CODE, transform=transform, nodata=-9999.0)

    release = box(X0 + 80, Y0 - 60, X0 + 120, Y0 - 20)
    for slug, dem_name in (('little_professor', 'dem.tif'),
                           ('bethel', 'bethel_bare_ground_dsm_1m.tif')):
        st = sp / 'data' / 'static' / slug
        for sub in ('terrain', 'canopy', 'infrastructure', 'release'):
            (st / sub).mkdir(parents=True, exist_ok=True)
        with rasterio.open(st / 'terrain' / dem_name, 'w', **prof) as d:
            d.write(z.astype('float32'), 1)
        _gj(release, st / 'release' / 'start_zone.geojson', name='start_zone')
        _gj(box(X0 + 70, Y0 - 300, X0 + 130, Y0 - 60),
            st / 'release' / 'entrainment_polygon.geojson', name='ent')
        _gj(box(X0 + 150, Y0 - 300, X0 + 190, Y0 - 200), st / 'canopy' / 'forest.shp')
        _gj(LineString([(X0 + 5, Y0 - 360), (X0 + 195, Y0 - 360)]),
            st / 'infrastructure' / 'road.shp', name='US-6')
        gpd.GeoDataFrame({'ref_name': ['tower', 'bench', 'road']},
                         geometry=[Point(X0 + 100, Y0 - 150), Point(X0 + 100, Y0 - 280),
                                   Point(X0 + 100, Y0 - 360)], crs=CRS_CODE
                         ).to_file(st / 'infrastructure' / 'reference_points.geojson',
                                   driver='GeoJSON')

    # Scenario drop: two horizons, two scenarios each
    scen = sp / 'intermediate' / 'scenarios' / 'little_professor' / '20261201'
    mask = rasterio.features.geometry_mask([release.__geo_interface__], (NROW, NCOL),
                                           transform, invert=True)
    wkt = CRS.from_user_input(CRS_CODE).to_wkt(version='WKT1_ESRI')
    for hz, depths in (('T_0', [1.2, 0.6]), ('T_1', [1.5, 0.8])):
        for i, dep in enumerate(depths, 1):
            inp = scen / hz / f'scenario_{i:03d}' / 'inputs'
            inp.mkdir(parents=True)
            _gj(release, inp / 'release.geojson', name='rel')
            p2 = dict(prof)
            p2.update(driver='AAIGrid', nodata=0.0)
            with rasterio.open(inp / 'depth.asc', 'w', **p2) as d:
                d.write((mask * dep).astype('float32'), 1)
            (inp / 'depth.prj').write_text(wkt)
            (inp / 'density.json').write_text(json.dumps({'mean_kgm3': 320.0}))
            (inp / 'params.json').write_text(json.dumps({
                'mu': 0.155, 'xi': 1500, 'rho_kgm3': 320.0, 'scour_depth_m': 0.2,
                'scenario_probability': 0.7 if i == 1 else 0.3, 'mean_depth_m': dep,
                'release_area_m2': 1600.0, 'total_volume_m3': 1600.0 * dep,
                'hand_hardness_value': -3.0}))

    # Observed event on the forecast date: stopped short of the road
    ev = sp / 'data' / 'static' / 'little_professor' / 'observed' / '20261201'
    ev.mkdir(parents=True)
    _gj(box(X0 + 78, Y0 - 335, X0 + 122, Y0 - 62), ev / 'debris.geojson')
    _gj(Point(X0 + 100, Y0 - 335), ev / 'toe.geojson')
    _gj(release, ev / 'release.geojson')
    (ev / 'event.yaml').write_text(
        "date: 2026-12-01\npath: little_professor\ndebris_polygon: debris.geojson\n"
        "release_polygon: release.geojson\ntoe_point: toe.geojson\n"
        "points:\n  tower: reached\n  bench: reached\n  road: not_reached\n"
        "slab:\n  hand_hardness: -3\n  density_kgm3: 300\n  depth_m: 0.9\n"
        "source: smoke test\n")

    # Config: the repo's, with local.yaml and the synthetic file names
    cfg = root / 'config'
    shutil.copytree(REPO / 'config', cfg)
    (cfg / 'local.yaml').write_text(
        f"avachain_root: {sp}\nscenario_root: {sp / 'intermediate' / 'scenarios'}\n"
        f"publish_root: {root / 'www'}\n")
    pp = cfg / 'paths' / 'little_professor.yaml'
    pp.write_text(pp.read_text()
                  .replace('260605_Professor_Bare_Ground_Full_Path-DSM_1m.tiff', 'dem.tif')
                  .replace('cdot_US6_LP_LittleProfessor.shp', 'road.shp'))
    profiles = yaml.safe_load((cfg / 'profiles.yaml').read_text())
    profiles['sweep']['sweep']['depths'] = [0.5, 1.0, 1.5]
    profiles['sweep']['sweep']['classes'] = ['soft', 'hard']
    profiles['calibrate']['grid'] = {'friction.tau0': [100, 150, 200]}
    (cfg / 'profiles.yaml').write_text(yaml.safe_dump(profiles, sort_keys=False))
    return cfg


def run(cmd, cfg):
    # Run one avachain command with the stub on; return the exit code
    env = dict(os.environ, AVACHAIN_STUB_COM1DFA='1')
    full = [sys.executable, '-m', 'scripts'] + cmd + ['--config-dir', str(cfg), '--no-log']
    print('\n$ ' + ' '.join(cmd))
    return subprocess.run(full, cwd=str(REPO), env=env).returncode


def main():
    # Build the tree, run every mode, check the products
    parser = argparse.ArgumentParser()
    parser.add_argument('--keep', default=None, help='Build under this folder and keep it')
    args = parser.parse_args()
    root = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix='avachain_smoke_'))
    if root.exists() and args.keep:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    cfg = build_tree(root)
    sp = root / "avachain"

    stages = [
        (['run', '--profile', 'forecast_nosurvey', '--date', '20261201',
          '--set', 'entrainment.mode=polygon', '--run-name', 'smoke'],
         [sp / 'outputs/forecasts/little_professor/20261201/report.html',
          sp / 'outputs/forecasts/little_professor/20261201/viewer.html',
          sp / 'outputs/forecasts/little_professor/20261201/scenarios.xlsx',
          root / 'www/forecasts/little_professor/latest/report.html']),
        (['run', '--profile', 'sweep', '--path', 'bethel', '--run-name', 'smoke'],
         [sp / 'outputs/calibration/bethel/sweep_smoke/sweep.html',
          sp / 'outputs/calibration/bethel/sweep_smoke/lookup.csv']),
        (['run', '--profile', 'calibrate', '--path', 'little_professor',
          '--event', '20261201', '--run-name', 'smoke', '--set', 'entrainment.mode=polygon'],
         [sp / 'outputs/calibration/little_professor/20261201/smoke/calibrate.html',
          sp / 'outputs/calibration/catalog.csv']),
        (['run', '--profile', 'verify', '--path', 'little_professor',
          '--date', '20261201', '--run-name', 'smoke'],
         [sp / 'outputs/verify/little_professor/20261201/smoke/verify.html',
          sp / 'outputs/verify/verification.csv']),
    ]
    failures = []
    for cmd, expect in stages:
        rc = run(cmd, cfg)
        missing = [str(p) for p in expect if not p.exists()]
        if rc != 0 or missing:
            failures.append((cmd[2], rc, missing))

    # A couple of value checks on the stub's known behaviour
    checks = []
    cat = sp / 'outputs/calibration/catalog.csv'
    if cat.exists():
        import pandas as pd
        row = pd.read_csv(cat).iloc[-1]
        checks.append(('calibrate picks tau0 150', float(row['best']) == 150.0))
    rs = sp / 'outputs/forecasts/little_professor/20261201/T_0/road_impact_summary.json'
    if rs.exists():
        checks.append(('T_0 reaches road with 1 of 2',
                       json.loads(rs.read_text())['n_reached'] == 1))

    print('\n' + '=' * 70)
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        if not ok:
            failures.append((name, 1, []))
    if failures:
        print(' SMOKE TEST FAILED')
        for f in failures:
            print(f"   {f}")
    else:
        print(f' SMOKE TEST PASSED  (tree at {root})')
    print('=' * 70)
    if not args.keep:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(1 if failures else 0)


if __name__ == '__main__':
    main()
